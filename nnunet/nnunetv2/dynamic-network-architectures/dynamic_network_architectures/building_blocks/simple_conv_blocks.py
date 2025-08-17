from typing import Tuple, List, Union, Type

import numpy as np
import torch.nn
import logging
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.depthwise_separable_conv import DepthwiseSeparableConv

# 设置轻量级卷积块日志
conv_logger = logging.getLogger('nnunet.lightweight_conv')
if not conv_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    conv_logger.addHandler(handler)
    conv_logger.setLevel(logging.INFO)


class LightweightConvDropoutNormReLU(nn.Module):
    """
    轻量级卷积块，可选择使用深度可分离卷积或瓶颈结构
    """
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
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False,
                 use_bottleneck: bool = True,
                 bottleneck_ratio: float = 0.25,
                 use_depthwise_separable: bool = True
                 ):
        super(LightweightConvDropoutNormReLU, self).__init__()
        
        # 添加日志信息
        conv_logger.info(f"✨ 初始化 LightweightConvDropoutNormReLU: 输入通道={input_channels}, 输出通道={output_channels}, 瓶颈={use_bottleneck}, 深度可分离={use_depthwise_separable}")
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        if use_bottleneck and input_channels != output_channels:
            # 瓶颈结构：1x1压缩 -> 3x3卷积 -> 1x1扩展
            bottleneck_channels = max(int(input_channels * bottleneck_ratio), 16)
            
            # 1x1压缩
            self.bottleneck_compress = conv_op(
                input_channels, bottleneck_channels, 1, 1, 0, bias=conv_bias
            )
            ops.append(self.bottleneck_compress)
            
            if norm_op is not None:
                self.bn_compress = norm_op(bottleneck_channels, **norm_op_kwargs)
                ops.append(self.bn_compress)
            
            if nonlin is not None:
                self.nonlin_compress = nonlin(**nonlin_kwargs)
                ops.append(self.nonlin_compress)
            
            # 主卷积
            if use_depthwise_separable:
                self.conv = DepthwiseSeparableConv(
                    bottleneck_channels,
                    bottleneck_channels,
                    kernel_size,
                    stride,
                    padding=[(i - 1) // 2 for i in kernel_size],
                    conv_op=conv_op,
                    conv_bias=conv_bias
                )
            else:
                self.conv = conv_op(
                    bottleneck_channels,
                    bottleneck_channels,
                    kernel_size,
                    stride,
                    padding=[(i - 1) // 2 for i in kernel_size],
                    dilation=1,
                    bias=conv_bias,
                )
            ops.append(self.conv)
            
            # 1x1扩展
            self.bottleneck_expand = conv_op(
                bottleneck_channels, output_channels, 1, 1, 0, bias=conv_bias
            )
            ops.append(self.bottleneck_expand)
            
        else:
            # 标准结构或深度可分离卷积
            if use_depthwise_separable:
                self.conv = DepthwiseSeparableConv(
                    input_channels,
                    output_channels,
                    kernel_size,
                    stride,
                    padding=[(i - 1) // 2 for i in kernel_size],
                    conv_op=conv_op,
                    conv_bias=conv_bias
                )
            else:
                self.conv = conv_op(
                    input_channels,
                    output_channels,
                    kernel_size,
                    stride,
                    padding=[(i - 1) // 2 for i in kernel_size],
                    dilation=1,
                    bias=conv_bias,
                )
            ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)


class ConvDropoutNormReLU(nn.Module):
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
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        super(ConvDropoutNormReLU, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        self.conv = DepthwiseSeparableConv(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding=[(i - 1) // 2 for i in kernel_size],
                conv_op=conv_op,
                conv_bias=conv_bias
            )
        ops.append(self.conv)


        # self.conv = conv_op(
        #     input_channels,
        #     output_channels,
        #     kernel_size,
        #     stride,
        #     padding=[(i - 1) // 2 for i in kernel_size],
        #     dilation=1,
        #     bias=conv_bias,
        # )
        # ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)


class StackedConvBlocks(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        """

        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        """
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
                for i in range(1, num_convs)
            ]
        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


class LightweightStackedConvBlocks(nn.Module):
    """
    轻量级堆叠卷积块，使用瓶颈结构和深度可分离卷积
    """
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False,
                 use_bottleneck: bool = True,
                 bottleneck_ratio: float = 0.25
                 ):
        """
        轻量级版本的StackedConvBlocks，使用瓶颈结构减少参数量
        """
        super().__init__()
        
        # 添加日志信息
        conv_logger.info(f"🏗️ 初始化 LightweightStackedConvBlocks: 层数={num_convs}, 输入通道={input_channels}, 输出通道={output_channels}, 瓶颈={use_bottleneck}")
        
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        # 第一层使用轻量级卷积
        self.convs = nn.Sequential(
            LightweightConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, 
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, 
                nonlin, nonlin_kwargs, nonlin_first, use_bottleneck, bottleneck_ratio
            ),
            *[
                LightweightConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, 
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, 
                    nonlin, nonlin_kwargs, nonlin_first, use_bottleneck, bottleneck_ratio
                )
                for i in range(1, num_convs)
            ]
        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


if __name__ == '__main__':
    data = torch.rand((1, 3, 40, 32))

    stx = StackedConvBlocks(2, nn.Conv2d, 24, 16, (3, 3), 2,
                            norm_op=nn.BatchNorm2d, nonlin=nn.ReLU, nonlin_kwargs={'inplace': True},
                            )
    model = nn.Sequential(ConvDropoutNormReLU(nn.Conv2d,
                                              3, 24, 3, 1, True, nn.BatchNorm2d, {}, None, None, nn.LeakyReLU,
                                              {'inplace': True}),
                          stx)
    import hiddenlayer as hl

    g = hl.build_graph(model, data,
                       transforms=None)
    g.save("network_architecture.pdf")
    del g

    stx.compute_conv_feature_map_size((40, 32))