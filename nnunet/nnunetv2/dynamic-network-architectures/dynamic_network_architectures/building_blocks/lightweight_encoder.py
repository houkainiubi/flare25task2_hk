import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.lightweight_residual import LightweightBottleneckBlock, EfficientBlock
from dynamic_network_architectures.building_blocks.lightweight_pooling import PyramidPooling, LightweightPyramidPooling, LightweightPooling
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks


class FusedMBConvBlock(nn.Module):
    """
    融合的MobileNet块，减少内存访问，提升推理速度
    """
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 expansion_ratio: float = 4.0,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 squeeze_excitation: bool = True,
                 squeeze_excitation_reduction_ratio: float = 1. / 16):
        super().__init__()
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}
            
        expanded_channels = int(input_channels * expansion_ratio)
        
        # 融合的扩展+深度卷积，减少内存访问
        if isinstance(kernel_size, int):
            if conv_op == nn.Conv3d:
                kernel_size = [kernel_size, kernel_size, kernel_size]
            elif conv_op == nn.Conv2d:
                kernel_size = [kernel_size, kernel_size]
            else:
                kernel_size = [kernel_size]
                
        if isinstance(stride, int):
            if conv_op == nn.Conv3d:
                stride = [stride, stride, stride]
            elif conv_op == nn.Conv2d:
                stride = [stride, stride]
            else:
                stride = [stride]
                
        padding = [(k - 1) // 2 for k in kernel_size]
        
        # 融合的扩展+深度卷积
        self.fused_conv = conv_op(input_channels, expanded_channels, kernel_size, stride, padding, bias=conv_bias)
        self.fused_norm = norm_op(expanded_channels, **norm_op_kwargs) if norm_op is not None else nn.Identity()
        self.nonlin1 = nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()
        
        # SE模块
        if squeeze_excitation and expanded_channels > 1:
            se_channels = max(1, int(expanded_channels * squeeze_excitation_reduction_ratio))
            if conv_op == nn.Conv3d:
                self.se = nn.Sequential(
                    nn.AdaptiveAvgPool3d(1),
                    nn.Conv3d(expanded_channels, se_channels, 1, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Conv3d(se_channels, expanded_channels, 1, bias=True),
                    nn.Sigmoid()
                )
            elif conv_op == nn.Conv2d:
                self.se = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(expanded_channels, se_channels, 1, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(se_channels, expanded_channels, 1, bias=True),
                    nn.Sigmoid()
                )
            else:
                self.se = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Conv1d(expanded_channels, se_channels, 1, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Conv1d(se_channels, expanded_channels, 1, bias=True),
                    nn.Sigmoid()
                )
        else:
            self.se = None
        
        # 输出投影
        self.output_conv = conv_op(expanded_channels, output_channels, 1, 1, 0, bias=conv_bias)
        self.output_norm = norm_op(output_channels, **norm_op_kwargs) if norm_op is not None else nn.Identity()
        
        # 跳跃连接
        self.use_skip_connection = (input_channels == output_channels and all(s == 1 for s in stride))
        
    def forward(self, x):
        identity = x
        
        # 融合的扩展+深度卷积
        x = self.fused_conv(x)
        x = self.fused_norm(x)
        x = self.nonlin1(x)
        
        # SE注意力
        if self.se is not None:
            se_out = self.se(x)
            x = x * se_out
        
        # 输出投影
        x = self.output_conv(x)
        x = self.output_norm(x)
        
        # 跳跃连接
        if self.use_skip_connection:
            x = x + identity
            
        return x


class LightweightBlock(nn.Module):
    """基础轻量级块，使用深度可分离卷积"""
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None):
        super().__init__()
        
        # 处理参数
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        padding = [(k - 1) // 2 for k in kernel_size]
        
        # 深度卷积
        self.depthwise = conv_op(
            input_channels, input_channels, kernel_size,
            stride=stride, padding=padding, groups=input_channels, bias=conv_bias
        )
        self.bn1 = norm_op(input_channels, **norm_op_kwargs) if norm_op is not None else nn.Identity()
        self.act1 = nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()
        
        # 点卷积
        self.pointwise = conv_op(input_channels, output_channels, 1, bias=conv_bias)
        self.bn2 = norm_op(output_channels, **norm_op_kwargs) if norm_op is not None else nn.Identity()
        self.act2 = nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()
        
        # 跳跃连接
        self.use_residual = (input_channels == output_channels and 
                           all(s == 1 for s in stride))
        
        # Dropout
        self.dropout = dropout_op(**dropout_op_kwargs) if dropout_op is not None else None

    def forward(self, x):
        residual = x if self.use_residual else None
        
        # 深度可分离卷积
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.pointwise(out)
        out = self.bn2(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        # 残差连接
        if self.use_residual:
            out = out + residual
        
        out = self.act2(out)
        return out


class FastPyramidPooling(nn.Module):
    """
    快速金字塔池化，减少计算复杂度
    """
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 in_channels: int,
                 pool_sizes: List[int] = [1, 2, 4],  # 减少池化尺度
                 reduction_ratio: int = 4,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None):
        super().__init__()
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}
            
        reduced_channels = max(1, in_channels // reduction_ratio)
        
        # 选择对应的池化操作
        if conv_op == nn.Conv3d:
            self.adaptive_pool = nn.AdaptiveAvgPool3d
        elif conv_op == nn.Conv2d:
            self.adaptive_pool = nn.AdaptiveAvgPool2d
        else:
            self.adaptive_pool = nn.AdaptiveAvgPool1d
            
        self.pool_sizes = pool_sizes
        
        # 每个池化分支
        self.branches = nn.ModuleList()
        for pool_size in pool_sizes:
            branch = nn.Sequential(
                self.adaptive_pool(pool_size),
                conv_op(in_channels, reduced_channels, 1, 1, 0, bias=False),
                norm_op(reduced_channels, **norm_op_kwargs) if norm_op is not None else nn.Identity(),
                nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()
            )
            self.branches.append(branch)
            
        # 特征融合
        total_channels = in_channels + reduced_channels * len(pool_sizes)
        self.fusion = nn.Sequential(
            conv_op(total_channels, in_channels, 1, 1, 0, bias=False),
            norm_op(in_channels, **norm_op_kwargs) if norm_op is not None else nn.Identity(),
            nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()
        )
        
    def forward(self, x):
        input_shape = x.shape[2:]
        features = [x]
        
        for branch in self.branches:
            pooled = branch(x)
            # 快速上采样
            upsampled = F.interpolate(pooled, size=input_shape, mode='nearest')
            features.append(upsampled)
            
        concatenated = torch.cat(features, dim=1)
        output = self.fusion(concatenated)
        return output


class LightweightResidualEncoder(nn.Module):
    """轻量级残差编码器，使用瓶颈残差块和金字塔池化"""
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block_type: str = 'bottleneck',
                 expansion_ratio: float = 4.0,
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 use_pyramid_pooling: bool = True,
                 pyramid_pool_sizes: List[int] = [1, 2, 3, 6],
                 squeeze_excitation: bool = True,
                 squeeze_excitation_reduction_ratio: float = 1. / 16
                 ):
        """
        轻量级残差编码器，专门针对nnUNet优化
        
        :param block_type: 'bottleneck' for bottleneck blocks, 'lightweight' for depth-wise separable
        :param expansion_ratio: 瓶颈块的扩展比例
        :param use_pyramid_pooling: 是否使用金字塔池化
        :param pyramid_pool_sizes: 金字塔池化的尺度
        """
        super().__init__()
        
        # 参数处理
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(kernel_sizes) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(strides) == n_stages

        # 轻量级stem
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            
            # 使用depthwise separable convolution作为stem
            kernel_size = kernel_sizes[0]
            kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
            
            # 计算padding
            padding = [(k - 1) // 2 for k in kernel_size]
            
            # 点卷积 + 深度卷积
            self.stem = nn.Sequential(
                # 点卷积降维
                conv_op(input_channels, stem_channels // 2, 1, bias=conv_bias),
                norm_op(stem_channels // 2, **norm_op_kwargs) if norm_op is not None else nn.Identity(),
                nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity(),
                
                # 深度卷积
                conv_op(stem_channels // 2, stem_channels // 2, kernel_size, padding=padding, 
                       groups=stem_channels // 2, bias=conv_bias),
                norm_op(stem_channels // 2, **norm_op_kwargs) if norm_op is not None else nn.Identity(),
                nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity(),
                
                # 点卷积升维
                conv_op(stem_channels // 2, stem_channels, 1, bias=conv_bias),
                norm_op(stem_channels, **norm_op_kwargs) if norm_op is not None else nn.Identity(),
                nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()
            )
            input_channels = stem_channels
        else:
            self.stem = None

        # 构建各个阶段
        self.stages = nn.ModuleList()
        current_channels = input_channels
        
        for s in range(n_stages):
            stage_blocks = []
            
            for b in range(n_blocks_per_stage[s]):
                # 第一个块可能有stride
                block_stride = strides[s] if b == 0 else 1
                
                if block_type == 'bottleneck':
                    block = LightweightBottleneckBlock(
                        conv_op=conv_op,
                        input_channels=current_channels,
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        stride=block_stride,
                        expansion_ratio=expansion_ratio,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        dropout_op=dropout_op,
                        dropout_op_kwargs=dropout_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                        squeeze_excitation=squeeze_excitation,
                        squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
                    )
                elif block_type == 'lightweight':
                    block = LightweightBlock(
                        conv_op=conv_op,
                        input_channels=current_channels,
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        stride=block_stride,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        dropout_op=dropout_op,
                        dropout_op_kwargs=dropout_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs
                    )
                else:
                    raise ValueError(f"Unknown block_type: {block_type}")
                
                stage_blocks.append(block)
                current_channels = features_per_stage[s]
            
            stage = nn.Sequential(*stage_blocks)
            self.stages.append(stage)
        
        # 金字塔池化（仅在最后阶段）
        self.use_pyramid_pooling = use_pyramid_pooling
        if use_pyramid_pooling:
            self.pyramid_pooling = PyramidPooling(
                conv_op=conv_op,
                in_channels=features_per_stage[-1],
                pool_sizes=pyramid_pool_sizes,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs
            )
        else:
            self.pyramid_pooling = None

        # 存储必要的属性
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # 存储解码器可能需要的信息
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        
        ret = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            
            # 在最后一个阶段应用金字塔池化
            if i == len(self.stages) - 1 and self.use_pyramid_pooling and self.pyramid_pooling is not None:
                x = self.pyramid_pooling(x)
            
            ret.append(x)

        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        """计算卷积特征图大小"""
        if self.stem is not None:
            output = np.prod([self.stem[0].out_channels, *input_size], dtype=np.int64)
        else:
            output = np.int64(0)

        current_size = input_size
        for s in range(len(self.stages)):
            # 计算当前阶段的特征图大小  
            stage_output = np.prod([self.output_channels[s], *current_size], dtype=np.int64)
            output += stage_output
            
            # 更新空间尺寸
            current_size = [max(1, i // j) for i, j in zip(current_size, self.strides[s])]

        # 金字塔池化的额外计算量
        if self.use_pyramid_pooling:
            pyramid_output = np.prod([self.output_channels[-1] * 2, *current_size], dtype=np.int64)
            output += pyramid_output

        return output


class FastLightweightResidualEncoder(nn.Module):
    """高速轻量级残差编码器，专门优化推理速度"""
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block_type: str = 'fused_mbconv',  # 'fused_mbconv', 'lightweight_bottleneck', 'efficient'
                 expansion_ratio: float = 4.0,
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 use_fast_pyramid_pooling: bool = True,
                 pyramid_pool_sizes: List[int] = [1, 2, 4],  # 更少的池化尺度
                 squeeze_excitation: bool = True,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 enable_memory_efficient: bool = True  # 启用内存效率优化
                 ):
        """
        专门为推理速度优化的轻量级编码器
        
        :param block_type: 'fused_mbconv' 使用融合的MobileNet块以获得最佳速度
        :param enable_memory_efficient: 启用内存效率优化
        """
        super().__init__()
        
        # 参数处理
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(kernel_sizes) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(strides) == n_stages

        # 轻量级stem - 使用1x1卷积减少计算
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0] // 2  # 进一步减少stem通道
            
            # 使用更简单的stem
            kernel_1x1 = 1 if conv_op == nn.Conv3d else (1, 1) if conv_op == nn.Conv2d else 1
            self.stem = nn.Sequential(
                conv_op(input_channels, stem_channels, kernel_1x1, bias=conv_bias),
                norm_op(stem_channels, **norm_op_kwargs) if norm_op is not None else nn.Identity(),
                nonlin(**nonlin_kwargs) if nonlin is not None else nn.Identity()
            )
            input_channels = stem_channels
        else:
            self.stem = None

        # 构建各个阶段
        self.stages = nn.ModuleList()
        current_channels = input_channels
        
        for s in range(n_stages):
            stage_blocks = []
            
            for b in range(n_blocks_per_stage[s]):
                # 第一个块可能有stride
                block_stride = strides[s] if b == 0 else 1
                
                if block_type == 'fused_mbconv':
                    block = FusedMBConvBlock(
                        conv_op=conv_op,
                        input_channels=current_channels,
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        stride=block_stride,
                        expansion_ratio=expansion_ratio,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                        squeeze_excitation=squeeze_excitation,
                        squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
                    )
                elif block_type == 'lightweight_bottleneck':
                    block = LightweightBottleneckBlock(
                        conv_op=conv_op,
                        input_channels=current_channels,
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        stride=block_stride,
                        expansion_ratio=expansion_ratio,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        dropout_op=dropout_op,
                        dropout_op_kwargs=dropout_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                        squeeze_excitation=squeeze_excitation,
                        squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
                    )
                elif block_type == 'efficient':
                    block = EfficientBlock(
                        conv_op=conv_op,
                        input_channels=current_channels,
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        stride=block_stride,
                        expansion_ratio=expansion_ratio,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        dropout_op=dropout_op,
                        dropout_op_kwargs=dropout_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                        squeeze_excitation=squeeze_excitation,
                        squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
                    )
                else:
                    raise ValueError(f"Unknown block_type: {block_type}")
                
                stage_blocks.append(block)
                current_channels = features_per_stage[s]
            
            stage = nn.Sequential(*stage_blocks)
            self.stages.append(stage)
        
        # 快速金字塔池化（仅在最后阶段）
        self.use_fast_pyramid_pooling = use_fast_pyramid_pooling
        if use_fast_pyramid_pooling:
            self.fast_pyramid_pooling = FastPyramidPooling(
                conv_op=conv_op,
                in_channels=features_per_stage[-1],
                pool_sizes=pyramid_pool_sizes,
                reduction_ratio=4,  # 更激进的通道压缩
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs
            )
        else:
            self.fast_pyramid_pooling = None

        # 存储必要的属性
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips
        self.enable_memory_efficient = enable_memory_efficient

        # 存储解码器可能需要的信息
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        
        ret = []
        
        # 内存效率模式
        if self.enable_memory_efficient and not self.return_skips:
            # 如果不需要跳跃连接，不保存中间特征
            for stage in self.stages:
                x = stage(x)
            
            # 仅在最后应用金字塔池化
            if self.use_fast_pyramid_pooling and self.fast_pyramid_pooling is not None:
                x = self.fast_pyramid_pooling(x)
            
            return x
        else:
            # 常规模式，保存所有中间特征
            for i, stage in enumerate(self.stages):
                x = stage(x)
                
                # 在最后一个阶段应用金字塔池化
                if i == len(self.stages) - 1 and self.use_fast_pyramid_pooling and self.fast_pyramid_pooling is not None:
                    x = self.fast_pyramid_pooling(x)
                
                ret.append(x)

            if self.return_skips:
                return ret
            else:
                return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        """快速计算特征图大小"""
        if self.stem is not None:
            output = np.prod([self.stem[0].out_channels, *input_size], dtype=np.int64)
        else:
            output = np.int64(0)

        current_size = input_size
        for s in range(len(self.stages)):
            # 计算当前阶段的特征图大小
            stage_output = np.prod([self.output_channels[s], *current_size], dtype=np.int64)
            output += stage_output
            
            # 更新空间尺寸
            current_size = [max(1, i // j) for i, j in zip(current_size, self.strides[s])]

        # 金字塔池化的额外计算量
        if self.use_fast_pyramid_pooling:
            final_size = np.prod([self.output_channels[-1], *current_size], dtype=np.int64)
            output += final_size  # 金字塔池化不增加太多计算量

        return output
    """轻量级残差编码器，使用瓶颈块和金字塔池化"""
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block_type: str = 'lightweight_bottleneck',  # 'lightweight_bottleneck' or 'efficient'
                 expansion_ratio: float = 6.0,
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'lightweight',  # 'conv', 'avg', 'max', 'lightweight'
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = True,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 use_pyramid_pooling: bool = True,
                 pyramid_pooling_type: str = 'lightweight',  # 'standard', 'lightweight', 'attention'
                 pyramid_reduction_ratio: int = 8
                 ):
        """
        轻量级残差编码器
        
        :param block_type: 'lightweight_bottleneck' 或 'efficient'
        :param expansion_ratio: 瓶颈块的扩展比例
        :param use_pyramid_pooling: 是否使用金字塔池化
        :param pyramid_pooling_type: 金字塔池化类型
        :param pyramid_reduction_ratio: 金字塔池化的通道压缩比例
        """
        super().__init__()
        
        # 参数处理
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(kernel_sizes) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(strides) == n_stages

        # 选择池化操作
        if pool_type == 'lightweight':
            pool_op = None  # 使用自定义轻量级池化
        else:
            pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        # 构建stem
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            self.stem = StackedConvBlocks(1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, conv_bias,
                                          norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
            input_channels = stem_channels
        else:
            self.stem = None

        # 构建各个阶段
        stages = []
        for s in range(n_stages):
            stride_for_conv = strides[s] if pool_op is None and pool_type != 'lightweight' else 1

            # 构建残差块序列
            stage_blocks = []
            stage_input_channels = input_channels
            
            for b in range(n_blocks_per_stage[s]):
                # 第一个块可能有stride
                block_stride = stride_for_conv if b == 0 else 1
                
                if block_type == 'lightweight_bottleneck':
                    block = LightweightBottleneckBlock(
                        conv_op=conv_op,
                        input_channels=stage_input_channels,
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        stride=block_stride,
                        expansion_ratio=expansion_ratio,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        dropout_op=dropout_op,
                        dropout_op_kwargs=dropout_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                        stochastic_depth_p=stochastic_depth_p,
                        squeeze_excitation=squeeze_excitation,
                        squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
                    )
                elif block_type == 'efficient':
                    block = EfficientBlock(
                        conv_op=conv_op,
                        input_channels=stage_input_channels,
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        stride=block_stride,
                        expansion_ratio=expansion_ratio,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        dropout_op=dropout_op,
                        dropout_op_kwargs=dropout_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                        stochastic_depth_p=stochastic_depth_p,
                        squeeze_excitation=squeeze_excitation,
                        squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
                    )
                else:
                    raise ValueError(f"Unknown block_type: {block_type}")
                
                stage_blocks.append(block)
                stage_input_channels = features_per_stage[s]
            
            # 组合残差块
            stage = nn.Sequential(*stage_blocks)

            # 添加池化（如果需要）
            if pool_type == 'lightweight' and strides[s] > 1:
                # 使用轻量级池化
                pool = LightweightPooling(conv_op, strides[s], strides[s], 'adaptive_avg')
                stage = nn.Sequential(pool, stage)
            elif pool_op is not None and strides[s] > 1:
                stage = nn.Sequential(pool_op(strides[s]), stage)

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        
        # 添加金字塔池化（在最后一个阶段后）
        self.use_pyramid_pooling = use_pyramid_pooling
        if use_pyramid_pooling:
            final_channels = features_per_stage[-1]
            if pyramid_pooling_type == 'standard':
                self.pyramid_pooling = PyramidPooling(
                    conv_op=conv_op,
                    in_channels=final_channels,
                    reduction_ratio=pyramid_reduction_ratio,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                )
            elif pyramid_pooling_type == 'lightweight':
                self.pyramid_pooling = LightweightPyramidPooling(
                    conv_op=conv_op,
                    in_channels=final_channels,
                    reduction_ratio=pyramid_reduction_ratio,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                )
            else:
                raise ValueError(f"Unknown pyramid_pooling_type: {pyramid_pooling_type}")
        else:
            self.pyramid_pooling = None

        # 存储必要的属性
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # 存储解码器可能需要的信息
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        
        # 应用金字塔池化到最后的特征
        if self.use_pyramid_pooling and self.pyramid_pooling is not None:
            x = self.pyramid_pooling(x)
            ret[-1] = x  # 更新最后一个特征

        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            # 这里简化计算，实际应该调用每个块的计算方法
            stage_output = np.prod([self.output_channels[s], *input_size], dtype=np.int64)
            output += stage_output
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        # 金字塔池化的额外计算量
        if self.use_pyramid_pooling:
            final_size = np.prod([self.output_channels[-1], *input_size], dtype=np.int64)
            output += final_size * 2  # 估算

        return output


class UltraLightweightEncoder(nn.Module):
    """超轻量级编码器，适用于资源极度受限的环境"""
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 expansion_ratio: float = 2.0,  # 更小的扩展比例
                 return_skips: bool = False,
                 squeeze_excitation: bool = False,  # 默认关闭SE
                 ):
        """
        超轻量级编码器，参数量和计算量都极度压缩
        """
        super().__init__()
        
        # 参数处理
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        # 极简stem
        kernel_size_1x1 = 1 if conv_op == nn.Conv3d else (1, 1) if conv_op == nn.Conv2d else 1
        self.stem = conv_op(input_channels, features_per_stage[0], kernel_size_1x1, bias=conv_bias)
        
        # 构建超轻量级阶段
        stages = []
        input_ch = features_per_stage[0]
        
        for s in range(n_stages):
            stage_blocks = []
            
            # 每个阶段只用一个轻量级块
            for b in range(n_blocks_per_stage[s]):
                stride_for_block = strides[s] if b == 0 else 1
                
                block = EfficientBlock(
                    conv_op=conv_op,
                    input_channels=input_ch,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    stride=stride_for_block,
                    expansion_ratio=expansion_ratio,  # 小扩展比例
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    squeeze_excitation=squeeze_excitation
                )
                stage_blocks.append(block)
                input_ch = features_per_stage[s]
            
            stages.append(nn.Sequential(*stage_blocks))
        
        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.return_skips = return_skips

    def forward(self, x):
        x = self.stem(x)
        
        ret = []
        for stage in self.stages:
            x = stage(x)
            ret.append(x)

        if self.return_skips:
            return ret
        else:
            return ret[-1]


if __name__ == '__main__':
    # 测试代码
    data = torch.rand((1, 1, 64, 64, 64))  # 3D数据
    
    # 轻量级编码器
    model = LightweightResidualEncoder(
        input_channels=1,
        n_stages=4,
        features_per_stage=[32, 64, 128, 256],
        conv_op=nn.Conv3d,
        kernel_sizes=3,
        strides=[1, 2, 2, 2],
        n_blocks_per_stage=2,
        norm_op=nn.InstanceNorm3d,
        nonlin=nn.ReLU,
        block_type='lightweight_bottleneck',
        expansion_ratio=4.0,
        use_pyramid_pooling=True,
        pyramid_pooling_type='lightweight'
    )
    
    output = model(data)
    print(f"输出形状: {output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    # 计算feature map大小
    fmap_size = model.compute_conv_feature_map_size((64, 64, 64))
    print(f"Feature map大小: {fmap_size:,}")
