import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import *
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile
import numpy as np
import os
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import empty_cache
from torch.optim import SGD
from torch.cuda.amp import GradScaler
from contextlib import contextmanager



class nnUNetTrainerCosAnneal(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """
        知识蒸馏训练器 - 通过环境变量获取配置
        """
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # 从环境变量读取教师模型配置
        self.teacher_plans_path = os.environ.get('TEACHER_PLANS_PATH', None)
        self.teacher_configuration = os.environ.get('TEACHER_CONFIGURATION', '3d_fullres')
        self.teacher_checkpoint = os.environ.get('TEACHER_CHECKPOINT', None)
        
        # 蒸馏参数
        self.distill_loss_type = os.environ.get('DISTILL_LOSS_TYPE', 'kl')
        self.distill_temperature = float(os.environ.get('DISTILL_TEMPERATURE', '3.0'))
        self.distill_alpha = float(os.environ.get('DISTILL_ALPHA', '0.7'))
        self.distill_beta = float(os.environ.get('DISTILL_BETA', '0.3'))
        
        # 检查必要参数是否设置
        if not self.teacher_plans_path or not self.teacher_checkpoint:
            raise ValueError("教师模型路径和检查点必须通过环境变量设置")
        
        # 教师模型将在initialize方法中加载
        self.teacher_model = None
        self.teacher_plans_manager = None
        self.teacher_configuration_manager = None

    def initialize(self):
        # 首先初始化学生模型
        super().initialize()
        
        # 然后加载教师模型
        self._load_teacher_model()
        
        # 初始化蒸馏损失函数
        self.distill_loss = KnowledgeDistillationLoss(
            loss_type=self.distill_loss_type,
            temperature=self.distill_temperature,
            alpha=self.distill_alpha,
            beta=self.distill_beta
        )

    def _load_teacher_model(self):
        """加载预训练的教师模型"""
        # 加载教师模型的计划文件
        teacher_plans = load_json(self.teacher_plans_path)
        self.teacher_plans_manager = PlansManager(teacher_plans)
        self.teacher_configuration_manager = self.teacher_plans_manager.get_configuration(self.teacher_configuration)
        
        # 获取教师模型的网络架构
        teacher_network_class_name = self.teacher_configuration_manager.architecture['network_class_name']
        teacher_arch_kwargs = self.teacher_configuration_manager.architecture['arch_kwargs']
        
        # 根据数据集调整输入输出通道
        num_input_channels = self.dataset_json['num_channels']
        label_manager = LabelManager(self.dataset_json['labels'], 
                                     self.dataset_json.get("regions_class_order"))
        num_output_channels = label_manager.num_segmentation_classes
        
        # 创建教师模型
        self.teacher_model = get_network_from_plans(
            teacher_network_class_name,
            teacher_arch_kwargs,
            num_input_channels,
            num_output_channels,
            deep_supervision=self.enable_deep_supervision
        ).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(self.teacher_checkpoint, map_location='cpu')
        self.teacher_model.load_state_dict(checkpoint['network_weights'])
        
        # 移动设备并设置评估模式
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        
        # 如果是DDP环境，包装教师模型
        if self.is_ddp:
            self.teacher_model = DDP(self.teacher_model, device_ids=[self.local_rank])
        
        self.print_to_log_file(f"教师模型加载完成: {self.teacher_configuration}")

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        # 学生模型前向传播
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            student_output = self.network(data)
            
            # 教师模型前向传播 (无梯度)
            with torch.no_grad():
                teacher_output = self.teacher_model(data)
            
            # 计算蒸馏损失
            l = self.distill_loss(student_output, teacher_output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            
        return {'loss': l.detach().cpu().numpy()}

    def configure_optimizers(self):
        """为学生模型配置优化器"""
        optimizer = SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                        momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def on_train_end(self):
        """训练结束时保存最终模型"""
        super().on_train_end()
        self.save_checkpoint(join(self.output_folder, "distilled_model_final.pth"))

    def save_checkpoint(self, filename: str) -> None:
        """保存蒸馏模型检查点"""
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod
                
                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                    'distill_params': {
                        'loss_type': self.distill_loss_type,
                        'temperature': self.distill_temperature,
                        'alpha': self.distill_alpha,
                        'beta': self.distill_beta
                    }
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')
    # 修复 1: 定义 dummy_context
@contextmanager
def dummy_context():
    yield None

# 修复 2: 定义 OptimizedModule
class OptimizedModule(torch.nn.Module):
    def __init__(self, original_module):
        super().__init__()
        self._orig_mod = original_module

    def forward(self, *args, **kwargs):
        return self._orig_mod(*args, **kwargs)

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, loss_type='kl', temperature=3.0, alpha=0.7, beta=0.3):
        """
        FLARE25数据集优化的知识蒸馏损失函数
        """
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        # 基础分割损失函数 (Dice+CE)
        self.task_loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, 
                                       {}, weight_ce=1, weight_dice=1,
                                       dice_class=MemoryEfficientSoftDiceLoss)
    
    def kl_divergence_loss(self, student_logits, teacher_logits):
        """适用于医学图像分割的KL散度损失"""
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        log_soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        kl_loss = F.kl_div(log_soft_student, soft_teacher, reduction='batchmean')
        return kl_loss * (self.temperature ** 2)
    
    def mse_loss(self, student_logits, teacher_logits):
        """适用于体素级预测的MSE损失"""
        return F.mse_loss(student_logits, teacher_logits)
    
    def attention_transfer_loss(self, student_features, teacher_features):
        """针对分割任务的注意力转移损失"""
        loss = 0
        for s_feat, t_feat in zip(student_features, teacher_features):
            s_att = self._spatial_attention_map(s_feat)
            t_att = self._spatial_attention_map(t_feat)
            
            if s_att.shape != t_att.shape:
                t_att = F.interpolate(t_att, size=s_att.shape[2:], mode='trilinear', align_corners=False)
            
            loss += F.mse_loss(s_att, t_att)
        
        return loss / len(student_features)
    
    def _spatial_attention_map(self, feature: torch.Tensor) -> torch.Tensor:
        """计算3D分割任务的空间注意力图"""
        return torch.norm(feature, p=2, dim=1, keepdim=True)
    
    def forward(self, student_output: torch.Tensor, 
                teacher_output: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        计算总损失 - 针对FLARE25数据集优化
        """
        task_loss = self.task_loss(student_output, target)
        
        if self.loss_type == 'attention':
            if isinstance(student_output, tuple) and len(student_output) > 1:
                distill_loss = self.attention_transfer_loss(student_output[1], teacher_output[1])
            else:
                distill_loss = self.kl_divergence_loss(student_output[0], teacher_output[0])
        else:
            distill_loss = self.kl_divergence_loss(student_output[0], teacher_output[0])
        
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distill_loss
        return total_loss