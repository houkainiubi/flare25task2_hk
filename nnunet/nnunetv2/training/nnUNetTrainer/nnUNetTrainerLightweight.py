import logging
from typing import Union, Tuple, List
import torch
from os.path import isfile
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# 设置轻量级训练器日志
lightweight_trainer_logger = logging.getLogger('nnunet.lightweight_trainer')
if not lightweight_trainer_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    lightweight_trainer_logger.addHandler(handler)
    lightweight_trainer_logger.setLevel(logging.INFO)


class nnUNetTrainerLightweight(nnUNetTrainer):
    """
    轻量级nnUNet训练器，使用轻量级网络组件提升训练和推理效率
    
    主要优化:
    1. 使用轻量级编码器组件
    2. 深度可分离卷积减少参数量
    3. 瓶颈残差块提升效率
    4. 金字塔池化增强多尺度特征
    """
    
    # 添加轻量级标识符
    __lightweight_trainer__ = True
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # 设置环境变量标识轻量级模式
        import os
        os.environ['NNUNET_USE_LIGHTWEIGHT'] = 'true'
        
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        lightweight_trainer_logger.info("🚀 初始化 nnUNetTrainerLightweight - 启用轻量级网络架构")
        lightweight_trainer_logger.info(f"📊 数据集: {dataset_json.get('name', 'Unknown')}")
        lightweight_trainer_logger.info(f"🔧 配置: {configuration}, 折: {fold}")
        lightweight_trainer_logger.info("⚡ 轻量级组件将自动激活: 瓶颈残差块 + 金字塔池化")
    
    def initialize(self):
        """
        重写初始化方法以添加轻量级组件信息
        """
        lightweight_trainer_logger.info("🔧 初始化轻量级训练器...")
        super().initialize()
        
        # 记录网络参数统计
        if hasattr(self, 'network') and self.network is not None:
            total_params = sum(p.numel() for p in self.network.parameters())
            trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            
            lightweight_trainer_logger.info(f"📊 网络参数统计:")
            lightweight_trainer_logger.info(f"  总参数量: {total_params:,}")
            lightweight_trainer_logger.info(f"  可训练参数量: {trainable_params:,}")
        
        lightweight_trainer_logger.info("✅ 轻量级训练器初始化完成")
    
    def on_train_start(self):
        """
        训练开始时的回调
        """
        if hasattr(super(), 'on_train_start'):
            super().on_train_start()
        lightweight_trainer_logger.info("🚀 开始轻量级nnUNet训练")
        lightweight_trainer_logger.info("⚡ 轻量级网络架构已激活: 瓶颈残差块 + 深度可分离卷积 + 金字塔池化")
        
        # 验证轻量级网络是否正确加载
        if hasattr(self, 'network') and hasattr(self.network, '__class__'):
            network_class = self.network.__class__.__name__
            if 'Lightweight' in network_class:
                lightweight_trainer_logger.info(f"✅ 确认使用轻量级网络: {network_class}")
            else:
                lightweight_trainer_logger.warning(f"⚠️  检测到标准网络: {network_class}, 请检查配置")
    
    def on_epoch_start(self):
        """
        每个epoch开始时的回调
        """
        if hasattr(super(), 'on_epoch_start'):
            super().on_epoch_start()
        if hasattr(self, 'current_epoch') and self.current_epoch % 20 == 0:  # 每20个epoch记录一次
            lightweight_trainer_logger.info(f"⚡ Epoch {self.current_epoch}: 轻量级网络训练中...")
    
    def save_checkpoint(self, filename: str) -> None:
        """
        重写保存检查点方法以添加轻量级标识
        """
        super().save_checkpoint(filename)
        lightweight_trainer_logger.info(f"💾 轻量级训练器检查点已保存: {filename}")
    
    def load_checkpoint(self, filename_or_checkpoint):
        """
        重写加载检查点方法，修复torch.compile前缀和PyTorch 2.6 weights_only问题
        """
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

        # Initialize logger from checkpoint
        if hasattr(self.logger, 'load_checkpoint') and self.logger is not None:
            self.logger.load_checkpoint(checkpoint['logging'])

        # Set inference axes if available
        if 'inference_allowed_mirroring_axes' in checkpoint.keys():
            self.inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes']

        lightweight_trainer_logger.info(f"✅ 轻量级训练器检查点已加载: epoch {self.current_epoch}")


# 修复torch.compile兼容性问题
class OptimizedModule(torch.nn.Module):
    """兼容性类，用于处理torch.compile产生的模块"""
    def __init__(self, original_module):
        super().__init__()
        self._orig_mod = original_module
        
    def __getattr__(self, name):
        if name == '_orig_mod':
            return super().__getattr__(name)
        return getattr(self._orig_mod, name)
