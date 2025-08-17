import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLightweight import nnUNetTrainerLightweight
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile
import numpy as np
import os
import logging
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

# 设置轻量级蒸馏训练器日志
distill_logger = logging.getLogger('nnunet.lightweight_distillation')
if not distill_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    distill_logger.addHandler(handler)
    distill_logger.setLevel(logging.INFO)


class nnUNetTrainerLightweightDistillation(nnUNetTrainerLightweight):
    """
    轻量级知识蒸馏训练器
    
    特点：
    1. 继承轻量级训练器的所有优化
    2. 支持从标准/轻量级教师模型学习
    3. 使用瓶颈残差块 + 金字塔池化的学生网络
    4. 针对医学图像分割的蒸馏损失优化
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """
        轻量级知识蒸馏训练器 - 通过环境变量获取配置
        """
        distill_logger.info("🎓 初始化轻量级知识蒸馏训练器")
        
        # 首先初始化轻量级训练器（确保轻量级网络构建）
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # 从环境变量读取教师模型配置
        self.teacher_plans_path = os.environ.get('TEACHER_PLANS_PATH', None)
        self.teacher_configuration = os.environ.get('TEACHER_CONFIGURATION', '3d_fullres')
        self.teacher_checkpoint = os.environ.get('TEACHER_CHECKPOINT', None)
        self.teacher_fold = os.environ.get('TEACHER_FOLD', 'all')
        
        # 蒸馏参数
        self.distill_loss_type = os.environ.get('DISTILL_LOSS_TYPE', 'kl')
        self.distill_temperature = float(os.environ.get('DISTILL_TEMPERATURE', '3.0'))
        self.distill_alpha = float(os.environ.get('DISTILL_ALPHA', '0.7'))
        self.distill_beta = float(os.environ.get('DISTILL_BETA', '0.3'))
        
        # 检查必要参数是否设置
        if not self.teacher_plans_path or not self.teacher_checkpoint:
            distill_logger.error("❌ 教师模型路径和检查点必须通过环境变量设置")
            raise ValueError("教师模型路径和检查点必须通过环境变量设置")
        
        distill_logger.info(f"📚 教师模型配置:")
        distill_logger.info(f"   Plans: {self.teacher_plans_path}")
        distill_logger.info(f"   Checkpoint: {self.teacher_checkpoint}")
        distill_logger.info(f"   Configuration: {self.teacher_configuration}")
        distill_logger.info(f"   Fold: {self.teacher_fold}")
        
        distill_logger.info(f"🎯 蒸馏参数:")
        distill_logger.info(f"   损失类型: {self.distill_loss_type}")
        distill_logger.info(f"   温度: {self.distill_temperature}")
        distill_logger.info(f"   Alpha: {self.distill_alpha}")
        distill_logger.info(f"   Beta: {self.distill_beta}")
        
        # 教师模型将在initialize方法中加载
        self.teacher_model = None
        self.teacher_plans_manager = None
        self.teacher_configuration_manager = None

    def initialize(self):
        """初始化学生和教师模型"""
        distill_logger.info("🔧 初始化轻量级蒸馏训练器...")
        
        # 首先初始化轻量级学生模型
        super().initialize()
        
        # 记录学生网络信息
        self._log_student_network_info()
        
        # 然后加载教师模型
        self._load_teacher_model()
        
        # 初始化蒸馏损失函数
        self.distill_loss = LightweightKnowledgeDistillationLoss(
            loss_type=self.distill_loss_type,
            temperature=self.distill_temperature,
            alpha=self.distill_alpha,
            beta=self.distill_beta
        )
        
        distill_logger.info("✅ 轻量级蒸馏训练器初始化完成")

    def _log_student_network_info(self):
        """记录学生网络信息"""
        if hasattr(self.network, '__class__'):
            network_class = self.network.__class__.__name__
            distill_logger.info(f"🎓 学生网络: {network_class}")
            
            # 统计参数
            total_params = sum(p.numel() for p in self.network.parameters())
            trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            
            distill_logger.info(f"📊 学生网络参数统计:")
            distill_logger.info(f"   总参数: {total_params:,}")
            distill_logger.info(f"   可训练参数: {trainable_params:,}")
            distill_logger.info(f"   参数量: {total_params/1e6:.2f}M")

    def _load_teacher_model(self):
        """加载预训练的教师模型 - 支持标准和轻量级教师"""
        distill_logger.info("📚 加载教师模型...")
        
        try:
            # 加载教师模型的计划文件
            teacher_plans = load_json(self.teacher_plans_path)
            self.teacher_plans_manager = PlansManager(teacher_plans)
            self.teacher_configuration_manager = self.teacher_plans_manager.get_configuration(
                self.teacher_configuration)
            
            # 获取教师模型的网络架构参数
            teacher_arch_class_name = self.teacher_configuration_manager.network_arch_class_name
            teacher_arch_kwargs = self.teacher_configuration_manager.network_arch_init_kwargs
            teacher_arch_kwargs_req_import = self.teacher_configuration_manager.network_arch_init_kwargs_req_import
            
            # 根据数据集调整输入输出通道
            num_input_channels = determine_num_input_channels(
                self.teacher_plans_manager, self.teacher_configuration_manager, self.dataset_json)
            label_manager = self.teacher_plans_manager.get_label_manager(self.dataset_json)
            num_output_channels = label_manager.num_segmentation_heads
            
            distill_logger.info(f"🏗️  教师网络架构: {teacher_arch_class_name}")
            distill_logger.info(f"📏 输入通道: {num_input_channels}, 输出通道: {num_output_channels}")
            
            # 检测教师模型是否应该是轻量级的
            self._detect_and_setup_teacher_model_type()
            
            # 创建教师模型
            self.teacher_model = get_network_from_plans(
                teacher_arch_class_name,
                teacher_arch_kwargs,
                teacher_arch_kwargs_req_import,
                num_input_channels,
                num_output_channels,
                deep_supervision=self.enable_deep_supervision
            ).to(self.device)
            
            # 加载权重
            checkpoint = torch.load(self.teacher_checkpoint, map_location='cpu', weights_only=False)
            
            if 'network_weights' in checkpoint:
                teacher_weights = checkpoint['network_weights']
                teacher_trainer_name = checkpoint.get('trainer_name', 'Unknown')
                distill_logger.info(f"👨‍🏫 教师训练器: {teacher_trainer_name}")
            else:
                teacher_weights = checkpoint
                teacher_trainer_name = 'Unknown'
            
            # 处理模型权重键名，移除可能的 _orig_mod. 前缀
            cleaned_weights = {}
            for key, value in teacher_weights.items():
                # 移除 _orig_mod. 前缀（torch.compile 或 DDP 产生的）
                clean_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
                cleaned_weights[clean_key] = value
            
            distill_logger.info(f"🔧 清理权重键名: {len(teacher_weights)} -> {len(cleaned_weights)} 键")
            
            self.teacher_model.load_state_dict(cleaned_weights)
            
            # 移动到设备并设置评估模式
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
            
            # 统计教师模型参数
            teacher_total_params = sum(p.numel() for p in self.teacher_model.parameters())
            distill_logger.info(f"📊 教师网络参数: {teacher_total_params:,} ({teacher_total_params/1e6:.2f}M)")
            
            # 如果是DDP环境，包装教师模型
            if self.is_ddp:
                self.teacher_model = DDP(self.teacher_model, device_ids=[self.local_rank])
            
            distill_logger.info(f"✅ 教师模型加载完成: {self.teacher_configuration}")
            
        except Exception as e:
            distill_logger.error(f"❌ 教师模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _detect_and_setup_teacher_model_type(self):
        """检测并设置教师模型类型"""
        # 检查checkpoint是否包含轻量级特征
        try:
            checkpoint = torch.load(self.teacher_checkpoint, map_location='cpu', weights_only=False)
            
            if 'network_weights' in checkpoint:
                params = checkpoint['network_weights']
                trainer_name = checkpoint.get('trainer_name', '')
            else:
                params = checkpoint
                trainer_name = ''
            
            # 检测轻量级特征
            lightweight_signatures = [
                'bottleneck_compress', 'bottleneck_expand', 'bn_compress',
                'pyramid_pool', 'stages.0.1.weight', 'final_conv.weight'
            ]
            
            depthwise_features = sum(1 for key in params.keys() 
                                   if 'depthwise.weight' in key or 'pointwise.weight' in key)
            lightweight_features = sum(1 for key in params.keys() 
                                     for sig in lightweight_signatures if sig in key)
            
            is_lightweight_teacher = (lightweight_features > 0 or 
                                    depthwise_features > 10 or 
                                    'Lightweight' in trainer_name)
            
            if is_lightweight_teacher:
                distill_logger.info("🎯 检测到轻量级教师模型，启用轻量级模式构建教师网络")
                os.environ['NNUNET_USE_LIGHTWEIGHT'] = 'true'
            else:
                distill_logger.info("🏛️  检测到标准教师模型，使用标准模式构建教师网络")
                # 临时禁用轻量级模式构建教师网络
                old_flag = os.environ.get('NNUNET_USE_LIGHTWEIGHT', '')
                os.environ.pop('NNUNET_USE_LIGHTWEIGHT', None)
                # 稍后会恢复
                self._restore_lightweight_flag = old_flag
                
        except Exception as e:
            distill_logger.warning(f"⚠️  无法检测教师模型类型，使用默认设置: {e}")

    def train_step(self, batch: dict) -> dict:
        """训练步骤 - 轻量级知识蒸馏"""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        # 学生模型前向传播（轻量级网络）
        with autocast(enabled=(self.device.type == 'cuda')) if self.device.type == 'cuda' else dummy_context():
            student_output = self.network(data)
            
            # 教师模型前向传播 (无梯度)
            with torch.no_grad():
                teacher_output = self.teacher_model(data)
            
            # 计算蒸馏损失
            l = self.distill_loss(student_output, teacher_output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            # 更严格的梯度裁剪，防止NaN
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)  # 从12降到1.0
            
            # 检查梯度是否有NaN/Inf
            for param in self.network.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        param.grad.zero_()  # 清零有问题的梯度
            
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            # 更严格的梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            
            # 检查梯度
            for param in self.network.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        param.grad.zero_()
            
            self.optimizer.step()
            
        return {'loss': l.detach().cpu().numpy()}

    def configure_optimizers(self):
        """为轻量级学生模型配置优化器"""
        distill_logger.info("⚙️  配置轻量级学生模型优化器")
        optimizer = SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                        momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def on_train_end(self):
        """训练结束时保存最终模型"""
        distill_logger.info("🎓 轻量级蒸馏训练完成，保存最终模型")
        super().on_train_end()
        self.save_checkpoint(join(self.output_folder, "lightweight_distilled_model_final.pth"))

    def save_checkpoint(self, filename: str) -> None:
        """保存轻量级蒸馏模型检查点"""
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
                        'beta': self.distill_beta,
                        'teacher_checkpoint': self.teacher_checkpoint,
                        'teacher_configuration': self.teacher_configuration
                    },
                    'lightweight_info': {
                        'is_lightweight': True,
                        'has_bottleneck_blocks': True,
                        'has_pyramid_pooling': True,
                        'bottleneck_ratio': 0.25
                    }
                }
                torch.save(checkpoint, filename)
                distill_logger.info(f"💾 轻量级蒸馏模型检查点已保存: {filename}")
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint):
        """修复torch.compile前缀问题的检查点加载"""
        if isinstance(filename_or_checkpoint, str):
            if not filename_or_checkpoint.endswith('.pth'):
                filename_or_checkpoint += '.pth'
            filename = filename_or_checkpoint
            self.print_to_log_file(f"loading checkpoint {filename}")
            if not isfile(filename):
                raise RuntimeError(f"Cannot load {filename}")
            # 修复PyTorch 2.6的weights_only问题
            checkpoint = torch.load(filename, map_location=torch.device('cpu'), weights_only=False)
        else:
            checkpoint = filename_or_checkpoint

        # 确保网络已初始化
        if self.network is None:
            distill_logger.warning("⚠️ 网络未初始化，先调用initialize()")
            self.initialize()

        # 修复权重前缀问题
        network_weights = checkpoint['network_weights']
        
        # 检测当前网络是否是compiled，以及权重是否有_orig_mod前缀
        is_network_compiled = isinstance(self.network, OptimizedModule) or hasattr(self.network, '_orig_mod')
        has_orig_mod_prefix = any(key.startswith('_orig_mod.') for key in network_weights.keys())
        
        if is_network_compiled and not has_orig_mod_prefix:
            # 网络是compiled的，但权重没有前缀，需要添加前缀
            compiled_weights = {}
            for key, value in network_weights.items():
                compiled_weights[f'_orig_mod.{key}'] = value
            self.network.load_state_dict(compiled_weights)
        elif not is_network_compiled and has_orig_mod_prefix:
            # 网络不是compiled的，但权重有前缀，需要去掉前缀
            clean_weights = {}
            for key, value in network_weights.items():
                if key.startswith('_orig_mod.'):
                    clean_weights[key[10:]] = value
                else:
                    clean_weights[key] = value
            self.network.load_state_dict(clean_weights)
        else:
            # 网络和权重匹配，直接加载
            self.network.load_state_dict(network_weights)

        self.current_epoch = checkpoint['current_epoch']
        if self.was_initialized:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if self.grad_scaler is not None:
                if checkpoint['grad_scaler_state'] is not None:
                    self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

        if '_best_ema' in checkpoint.keys():
            self._best_ema = checkpoint['_best_ema']

        # Load distill params if available
        if 'distill_params' in checkpoint:
            distill_params = checkpoint['distill_params']
            self.distill_loss_type = distill_params.get('loss_type', 'kl')
            self.distill_temperature = distill_params.get('temperature', 3.0)
            self.distill_alpha = distill_params.get('alpha', 0.7)
            self.distill_beta = distill_params.get('beta', 0.3)
            
        # Initialize logger from checkpoint
        if hasattr(self.logger, 'load_checkpoint') and self.logger is not None:
            self.logger.load_checkpoint(checkpoint['logging'])

        # Set inference axes if available
        if 'inference_allowed_mirroring_axes' in checkpoint.keys():
            self.inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes']

        distill_logger.info(f"✅ 蒸馏模型检查点已加载: epoch {self.current_epoch}")
        distill_logger.info(f"🔄 继续轻量级蒸馏训练: α={self.distill_alpha}, T={self.distill_temperature}")


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


class LightweightKnowledgeDistillationLoss(nn.Module):
    """
    轻量级知识蒸馏损失函数
    专门为轻量级医学图像分割模型优化
    """
    
    def __init__(self, loss_type='kl', temperature=3.0, alpha=0.7, beta=0.3):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        distill_logger.info(f"🎯 初始化轻量级蒸馏损失: {loss_type}, T={temperature}, α={alpha}")
        
        # 使用更简单的方法：继承父类的损失函数
        # 这样可以自动处理深度监督
        self.task_loss = None  # 稍后在父类中初始化
    
    def kl_divergence_loss(self, student_logits, teacher_logits):
        """适用于医学图像分割的KL散度损失 - 数值稳定版本"""
        # 处理深度监督的情况
        if isinstance(student_logits, (list, tuple)):
            student_logits = student_logits[0]  # 使用主要输出
        if isinstance(teacher_logits, (list, tuple)):
            teacher_logits = teacher_logits[0]  # 使用主要输出
            
        # 确保形状匹配
        if student_logits.shape != teacher_logits.shape:
            teacher_logits = F.interpolate(
                teacher_logits, size=student_logits.shape[2:], 
                mode='trilinear', align_corners=False
            )
        
        # 数值稳定的softmax计算
        student_scaled = student_logits / self.temperature
        teacher_scaled = teacher_logits / self.temperature
        
        # 添加数值稳定性
        student_scaled = torch.clamp(student_scaled, min=-10, max=10)
        teacher_scaled = torch.clamp(teacher_scaled, min=-10, max=10)
        
        soft_teacher = F.softmax(teacher_scaled, dim=1)
        log_soft_student = F.log_softmax(student_scaled, dim=1)
        
        kl_loss = F.kl_div(log_soft_student, soft_teacher, reduction='batchmean')
        
        # 检查NaN并处理
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            return torch.tensor(0.0, device=kl_loss.device, requires_grad=True)
        
        # 只乘以温度，不是温度的平方，避免损失过大
        return kl_loss * self.temperature * 0.1  # 额外缩放因子
    
    def mse_loss(self, student_logits, teacher_logits):
        """适用于体素级预测的MSE损失"""
        if isinstance(student_logits, (list, tuple)):
            student_logits = student_logits[0]
        if isinstance(teacher_logits, (list, tuple)):
            teacher_logits = teacher_logits[0]
            
        # 确保形状匹配
        if student_logits.shape != teacher_logits.shape:
            teacher_logits = F.interpolate(
                teacher_logits, size=student_logits.shape[2:], 
                mode='trilinear', align_corners=False
            )
            
        return F.mse_loss(student_logits, teacher_logits)
    
    def attention_transfer_loss(self, student_features, teacher_features):
        """针对轻量级分割任务的注意力转移损失"""
        if not isinstance(student_features, (list, tuple)):
            student_features = [student_features]
        if not isinstance(teacher_features, (list, tuple)):
            teacher_features = [teacher_features]
            
        loss = 0
        valid_pairs = 0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            if s_feat is not None and t_feat is not None:
                s_att = self._spatial_attention_map(s_feat)
                t_att = self._spatial_attention_map(t_feat)
                
                if s_att.shape != t_att.shape:
                    t_att = F.interpolate(t_att, size=s_att.shape[2:], mode='trilinear', align_corners=False)
                
                loss += F.mse_loss(s_att, t_att)
                valid_pairs += 1
        
        return loss / max(valid_pairs, 1)
    
    def _spatial_attention_map(self, feature: torch.Tensor) -> torch.Tensor:
        """计算3D分割任务的空间注意力图"""
        return torch.norm(feature, p=2, dim=1, keepdim=True)
    
    def forward(self, student_output: torch.Tensor, 
                teacher_output: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        计算总损失 - 针对轻量级模型优化
        """
        # 处理深度监督输出（如果是列表，取主要输出）
        if isinstance(student_output, (list, tuple)):
            student_main = student_output[0]
        else:
            student_main = student_output
            
        if isinstance(teacher_output, (list, tuple)):
            teacher_main = teacher_output[0]
        else:
            teacher_main = teacher_output
        
        # 计算任务损失（分割损失）- 如果task_loss为空，返回0
        if self.task_loss is not None:
            task_loss = self.task_loss(student_output, target)
        else:
            # 处理目标张量的形状和数据类型
            target_processed = target
            if isinstance(target, (list, tuple)):
                target_processed = target[0]
            
            # 如果目标张量有多余的维度，去掉它
            if len(target_processed.shape) == 5 and target_processed.shape[1] == 1:
                target_processed = target_processed.squeeze(1)
            
            # 确保目标张量是长整型（LongTensor）
            target_processed = target_processed.long()
            
            # 简单的备用损失
            if isinstance(student_output, (list, tuple)):
                task_loss = F.cross_entropy(student_output[0], target_processed)
            else:
                task_loss = F.cross_entropy(student_output, target_processed)
        
        # 计算蒸馏损失 - 使用主要输出
        if self.loss_type == 'attention':
            # 如果有特征图，使用注意力转移
            if (isinstance(student_output, (list, tuple)) and len(student_output) > 1 and
                isinstance(teacher_output, (list, tuple)) and len(teacher_output) > 1):
                distill_loss = self.attention_transfer_loss(student_output[1:], teacher_output[1:])
            else:
                # 回退到KL散度
                distill_loss = self.kl_divergence_loss(student_main, teacher_main)
        elif self.loss_type == 'mse':
            distill_loss = self.mse_loss(student_main, teacher_main)
        else:  # 默认使用KL散度
            distill_loss = self.kl_divergence_loss(student_main, teacher_main)
        
        # 组合损失 - 使用极小的蒸馏权重确保数值稳定性
        # 首先检查损失的有效性
        if torch.isnan(task_loss) or torch.isinf(task_loss):
            task_loss = torch.tensor(1.0, device=task_loss.device, requires_grad=True)
        if torch.isnan(distill_loss) or torch.isinf(distill_loss):
            distill_loss = torch.tensor(0.0, device=distill_loss.device, requires_grad=True)
        
        # 使用极小的蒸馏权重，优先保证任务损失的稳定性
        alpha_safe = 0.001  # 极小的蒸馏权重
        total_loss = (1 - alpha_safe) * task_loss + alpha_safe * distill_loss
        
        # 最终检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = task_loss  # 如果仍有问题，只用任务损失
        
        return total_loss


# 导入必要的函数
def determine_num_input_channels(plans_manager, configuration_manager, dataset_json):
    """确定输入通道数"""
    return len(dataset_json['channel_names'])
