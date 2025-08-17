import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list

# 设置轻量级池化组件日志
pooling_logger = logging.getLogger('nnunet.lightweight_pooling')
if not pooling_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    pooling_logger.addHandler(handler)
    pooling_logger.setLevel(logging.INFO)


class PyramidPooling(nn.Module):
    """
    金字塔池化模块，用于多尺度特征提取
    """
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 pool_sizes: List[int] = [1, 2, 3, 6],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None):
        super().__init__()
        
        # 添加日志信息
        pooling_logger.info(f"🔥 初始化 PyramidPooling: 输入通道={input_channels}, 池化尺寸={pool_sizes}")
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}
        
        self.conv_op = conv_op
        self.input_channels = input_channels
        self.pool_sizes = pool_sizes
        
        # 减少通道数以降低计算复杂度
        reduced_channels = max(input_channels // 4, 32)
        
        self.stages = nn.ModuleList()
        for pool_size in pool_sizes:
            stage = nn.Sequential(
                self._get_adaptive_pool(pool_size),
                conv_op(input_channels, reduced_channels, 1, bias=conv_bias),
                norm_op(reduced_channels, **norm_op_kwargs) if norm_op else nn.Identity(),
                nonlin(**nonlin_kwargs) if nonlin else nn.Identity()
            )
            self.stages.append(stage)
        
        # 最终卷积层融合所有特征
        total_channels = input_channels + len(pool_sizes) * reduced_channels
        self.final_conv = nn.Sequential(
            conv_op(total_channels, input_channels, 1, bias=conv_bias),
            norm_op(input_channels, **norm_op_kwargs) if norm_op else nn.Identity(),
            nonlin(**nonlin_kwargs) if nonlin else nn.Identity()
        )
    
    def _get_adaptive_pool(self, pool_size):
        if self.conv_op == nn.Conv3d:
            return nn.AdaptiveAvgPool3d(pool_size)
        elif self.conv_op == nn.Conv2d:
            return nn.AdaptiveAvgPool2d(pool_size)
        else:
            return nn.AdaptiveAvgPool1d(pool_size)
    
    def forward(self, x):
        # 原始特征
        features = [x]
        
        # 多尺度池化特征
        for stage in self.stages:
            pooled = stage(x)
            # 上采样到原始尺寸
            upsampled = F.interpolate(pooled, size=x.shape[2:], 
                                    mode='trilinear' if len(x.shape) == 5 else 'bilinear',
                                    align_corners=False)
            features.append(upsampled)
        
        # 拼接所有特征
        concat_features = torch.cat(features, dim=1)
        
        # 最终融合
        return self.final_conv(concat_features)


class LightweightPyramidPooling(nn.Module):
    """
    轻量级金字塔池化，使用更少的参数和计算
    """
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 pool_sizes: List[int] = [2, 4],  # 减少池化尺度
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None):
        super().__init__()
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}
        
        self.conv_op = conv_op
        self.input_channels = input_channels
        self.pool_sizes = pool_sizes
        
        # 使用更少的通道
        reduced_channels = max(input_channels // 8, 16)
        
        self.stages = nn.ModuleList()
        for pool_size in pool_sizes:
            # 使用组卷积进一步减少参数
            groups = min(input_channels, 8)
            stage = nn.Sequential(
                self._get_adaptive_pool(pool_size),
                conv_op(input_channels, reduced_channels, 1, groups=groups, bias=conv_bias),
                norm_op(reduced_channels, **norm_op_kwargs) if norm_op else nn.Identity(),
                nonlin(**nonlin_kwargs) if nonlin else nn.Identity()
            )
            self.stages.append(stage)
        
        # 轻量级融合
        total_channels = input_channels + len(pool_sizes) * reduced_channels
        self.final_conv = conv_op(total_channels, input_channels, 1, bias=conv_bias)
    
    def _get_adaptive_pool(self, pool_size):
        if self.conv_op == nn.Conv3d:
            return nn.AdaptiveAvgPool3d(pool_size)
        elif self.conv_op == nn.Conv2d:
            return nn.AdaptiveAvgPool2d(pool_size)
        else:
            return nn.AdaptiveAvgPool1d(pool_size)
    
    def forward(self, x):
        features = [x]
        
        for stage in self.stages:
            pooled = stage(x)
            upsampled = F.interpolate(pooled, size=x.shape[2:], 
                                    mode='nearest')  # 使用最近邻插值，更快
            features.append(upsampled)
        
        concat_features = torch.cat(features, dim=1)
        return self.final_conv(concat_features)


class LightweightPooling(nn.Module):
    """
    轻量级池化模块，可替代标准池化操作
    """
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 pool_type: str = 'avg',  # 'avg', 'max', 'adaptive_avg', 'stride_conv'
                 conv_bias: bool = False):
        super().__init__()
        
        self.pool_type = pool_type
        self.stride = maybe_convert_scalar_to_list(conv_op, stride)
        
        if pool_type == 'avg':
            if conv_op == nn.Conv3d:
                self.pool = nn.AvgPool3d(kernel_size, stride)
            elif conv_op == nn.Conv2d:
                self.pool = nn.AvgPool2d(kernel_size, stride)
            else:
                self.pool = nn.AvgPool1d(kernel_size, stride)
        elif pool_type == 'max':
            if conv_op == nn.Conv3d:
                self.pool = nn.MaxPool3d(kernel_size, stride)
            elif conv_op == nn.Conv2d:
                self.pool = nn.MaxPool2d(kernel_size, stride)
            else:
                self.pool = nn.MaxPool1d(kernel_size, stride)
        elif pool_type == 'adaptive_avg':
            if conv_op == nn.Conv3d:
                self.pool = nn.AdaptiveAvgPool3d(1)
            elif conv_op == nn.Conv2d:
                self.pool = nn.AdaptiveAvgPool2d(1)
            else:
                self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_type == 'stride_conv':
            # 使用卷积代替池化，可学习的下采样
            padding = [k//2 for k in maybe_convert_scalar_to_list(conv_op, kernel_size)]
            self.pool = conv_op(channels, channels, kernel_size, stride, 
                              padding=padding, bias=conv_bias, groups=channels)
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}")
    
    def forward(self, x):
        return self.pool(x)


def get_lightweight_pool_op(conv_op, pool_type='avg'):
    """
    获取轻量级池化操作
    """
    if pool_type == 'avg':
        if conv_op == nn.Conv3d:
            return nn.AvgPool3d
        elif conv_op == nn.Conv2d:
            return nn.AvgPool2d
        else:
            return nn.AvgPool1d
    elif pool_type == 'max':
        if conv_op == nn.Conv3d:
            return nn.MaxPool3d
        elif conv_op == nn.Conv2d:
            return nn.MaxPool2d
        else:
            return nn.MaxPool1d
    elif pool_type == 'adaptive_avg':
        if conv_op == nn.Conv3d:
            return nn.AdaptiveAvgPool3d
        elif conv_op == nn.Conv2d:
            return nn.AdaptiveAvgPool2d
        else:
            return nn.AdaptiveAvgPool1d
    else:
        return LightweightPooling


def get_efficient_downsampling(conv_op, channels, kernel_size=2, stride=2):
    """
    获取高效的下采样操作
    """
    # 使用深度可分离卷积进行下采样
    return nn.Sequential(
        # 深度卷积
        conv_op(channels, channels, kernel_size, stride, 
               padding=kernel_size//2, groups=channels, bias=False),
        # 点卷积
        conv_op(channels, channels, 1, bias=False)
    )
