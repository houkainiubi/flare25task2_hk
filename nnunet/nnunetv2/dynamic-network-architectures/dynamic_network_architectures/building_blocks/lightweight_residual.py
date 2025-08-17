import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.depthwise_separable_conv import DepthwiseSeparableConv

# 设置轻量级组件日志
lightweight_logger = logging.getLogger('nnunet.lightweight_components')
if not lightweight_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    lightweight_logger.addHandler(handler)
    lightweight_logger.setLevel(logging.INFO)


class LightweightBottleneckBlock(nn.Module):
    """
    轻量级瓶颈残差块，使用深度可分离卷积
    """
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int = None,
                 kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 expansion_ratio: float = 1.0,  # 减小扩展比例
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16):
        super().__init__()
        
        # 添加日志信息
        lightweight_logger.info(f"🚀 初始化 LightweightBottleneckBlock: 输入通道={input_channels}, 输出通道={output_channels or input_channels}, 扩展比例={expansion_ratio}")
        
        if output_channels is None:
            output_channels = input_channels
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = maybe_convert_scalar_to_list(conv_op, stride)
        
        # 计算中间通道数
        intermediate_channels = max(int(input_channels * expansion_ratio), 16)
        
        # 1x1 扩展卷积（如果需要）
        self.expand_conv = None
        if expansion_ratio > 1.0:
            self.expand_conv = nn.Sequential(
                conv_op(input_channels, intermediate_channels, 1, bias=conv_bias),
                norm_op(intermediate_channels, **norm_op_kwargs) if norm_op else nn.Identity(),
                nonlin(**nonlin_kwargs) if nonlin else nn.Identity()
            )
        else:
            intermediate_channels = input_channels
        
        # 深度可分离卷积
        self.dw_conv = DepthwiseSeparableConv(
            intermediate_channels,
            intermediate_channels,
            kernel_size,
            stride,
            padding=[(k - 1) // 2 for k in maybe_convert_scalar_to_list(conv_op, kernel_size)],
            conv_op=conv_op,
            conv_bias=conv_bias
        )
        
        # 批归一化和激活
        self.bn1 = norm_op(intermediate_channels, **norm_op_kwargs) if norm_op else nn.Identity()
        self.nonlin1 = nonlin(**nonlin_kwargs) if nonlin else nn.Identity()
        
        # SE模块（可选）
        self.se = None
        if squeeze_excitation:
            self.se = SEBlock(conv_op, intermediate_channels, squeeze_excitation_reduction_ratio)
        
        # 1x1 压缩卷积
        self.project_conv = nn.Sequential(
            conv_op(intermediate_channels, output_channels, 1, bias=conv_bias),
            norm_op(output_channels, **norm_op_kwargs) if norm_op else nn.Identity()
        )
        
        # Dropout
        self.dropout = dropout_op(**dropout_op_kwargs) if dropout_op else nn.Identity()
        
        # 残差连接
        self.use_residual = (input_channels == output_channels and 
                           all(s == 1 for s in self.stride))
        
        # 如果输入输出维度不匹配，添加投影层
        self.shortcut = nn.Identity()
        if not self.use_residual and input_channels != output_channels:
            self.shortcut = conv_op(input_channels, output_channels, 1, stride, bias=conv_bias)
    
    def forward(self, x):
        lightweight_logger.debug(f"⚡ LightweightBottleneckBlock 前向传播: 输入形状={x.shape}")
        
        identity = x
        
        # 扩展
        if self.expand_conv is not None:
            out = self.expand_conv(x)
        else:
            out = x
        
        # 深度可分离卷积
        out = self.dw_conv(out)
        out = self.bn1(out)
        out = self.nonlin1(out)
        
        # SE模块
        if self.se is not None:
            out = self.se(out)
        
        # 压缩
        out = self.project_conv(out)
        out = self.dropout(out)
        
        # 残差连接
        if self.use_residual:
            out = out + identity
        else:
            out = out + self.shortcut(identity)
        
        return out


class EfficientBlock(nn.Module):
    """
    高效残差块，针对推理速度优化
    """
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int = None,
                 kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 groups: int = 1,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None):
        super().__init__()
        
        # 添加日志信息
        lightweight_logger.info(f"🔧 初始化 EfficientBlock: 输入通道={input_channels}, 输出通道={output_channels or input_channels}, 组数={groups}")
        
        if output_channels is None:
            output_channels = input_channels
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = maybe_convert_scalar_to_list(conv_op, stride)
        
        # 使用组卷积减少计算量
        groups = min(groups, input_channels)
        
        # 第一个卷积层
        self.conv1 = conv_op(
            input_channels, output_channels, kernel_size, stride,
            padding=[(k - 1) // 2 for k in maybe_convert_scalar_to_list(conv_op, kernel_size)],
            groups=groups, bias=conv_bias
        )
        self.bn1 = norm_op(output_channels, **norm_op_kwargs) if norm_op else nn.Identity()
        self.nonlin1 = nonlin(**nonlin_kwargs) if nonlin else nn.Identity()
        
        # 第二个卷积层
        self.conv2 = conv_op(
            output_channels, output_channels, kernel_size, 1,
            padding=[(k - 1) // 2 for k in maybe_convert_scalar_to_list(conv_op, kernel_size)],
            groups=groups, bias=conv_bias
        )
        self.bn2 = norm_op(output_channels, **norm_op_kwargs) if norm_op else nn.Identity()
        
        # 残差连接
        self.use_residual = (input_channels == output_channels and 
                           all(s == 1 for s in self.stride))
        
        if not self.use_residual:
            self.shortcut = conv_op(input_channels, output_channels, 1, stride, bias=conv_bias)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlin1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        out = out + self.shortcut(identity)
        
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 块
    """
    def __init__(self, conv_op: Type[_ConvNd], channels: int, reduction_ratio: float = 1./16):
        super().__init__()
        
        reduced_channels = max(int(channels * reduction_ratio), 4)
        
        if conv_op == nn.Conv3d:
            self.global_pool = nn.AdaptiveAvgPool3d(1)
        elif conv_op == nn.Conv2d:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            conv_op(channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            conv_op(reduced_channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 全局平均池化
        y = self.global_pool(x)
        # 激励
        y = self.fc(y)
        # 重新加权
        return x * y


class LightweightBasicBlock(nn.Module):
    """
    轻量级基础残差块
    """
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int = None,
                 kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None):
        super().__init__()
        
        if output_channels is None:
            output_channels = input_channels
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = maybe_convert_scalar_to_list(conv_op, stride)
        
        # 使用深度可分离卷积
        self.conv1 = DepthwiseSeparableConv(
            input_channels, output_channels, kernel_size, stride,
            padding=[(k - 1) // 2 for k in maybe_convert_scalar_to_list(conv_op, kernel_size)],
            conv_op=conv_op, conv_bias=conv_bias
        )
        self.bn1 = norm_op(output_channels, **norm_op_kwargs) if norm_op else nn.Identity()
        self.nonlin1 = nonlin(**nonlin_kwargs) if nonlin else nn.Identity()
        
        self.conv2 = DepthwiseSeparableConv(
            output_channels, output_channels, kernel_size, 1,
            padding=[(k - 1) // 2 for k in maybe_convert_scalar_to_list(conv_op, kernel_size)],
            conv_op=conv_op, conv_bias=conv_bias
        )
        self.bn2 = norm_op(output_channels, **norm_op_kwargs) if norm_op else nn.Identity()
        
        # 残差连接
        self.use_residual = (input_channels == output_channels and 
                           all(s == 1 for s in self.stride))
        
        if not self.use_residual:
            self.shortcut = conv_op(input_channels, output_channels, 1, stride, bias=conv_bias)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlin1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        out = out + self.shortcut(identity)
        
        return out


class MobileInvertedResidualBlock(nn.Module):
    """
    移动端倒残差块（MobileNetV2风格）
    """
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 expansion_ratio: int = 6,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None):
        super().__init__()
        
        # 添加日志信息
        lightweight_logger.info(f"📱 初始化 MobileInvertedResidualBlock: 输入通道={input_channels}, 输出通道={output_channels}, 扩展比例={expansion_ratio}")
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.use_residual = (input_channels == output_channels and 
                           all(s == 1 for s in self.stride))
        
        hidden_dim = input_channels * expansion_ratio
        
        layers = []
        
        # 扩展层
        if expansion_ratio != 1:
            layers.extend([
                conv_op(input_channels, hidden_dim, 1, bias=conv_bias),
                norm_op(hidden_dim, **norm_op_kwargs) if norm_op else nn.Identity(),
                nonlin(**nonlin_kwargs) if nonlin else nn.Identity()
            ])
        
        # 深度卷积
        layers.extend([
            conv_op(hidden_dim, hidden_dim, kernel_size, stride,
                   padding=[(k - 1) // 2 for k in maybe_convert_scalar_to_list(conv_op, kernel_size)],
                   groups=hidden_dim, bias=conv_bias),
            norm_op(hidden_dim, **norm_op_kwargs) if norm_op else nn.Identity(),
            nonlin(**nonlin_kwargs) if nonlin else nn.Identity()
        ])
        
        # 投影层
        layers.extend([
            conv_op(hidden_dim, output_channels, 1, bias=conv_bias),
            norm_op(output_channels, **norm_op_kwargs) if norm_op else nn.Identity()
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out
