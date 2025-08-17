import torch.nn as nn
import torch.nn.init as init

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, conv_op, conv_bias):
        super(DepthwiseSeparableConv, self).__init__()
        
        # 深度卷积 (depthwise convolution)
        self.depthwise = conv_op(
            in_channels,
            in_channels,  # 输出通道与输入相同
            kernel_size,
            stride,
            padding=padding,
            groups=in_channels,  # 关键：分组数量等于输入通道数
            bias=conv_bias
        )
        
        # 点卷积 (pointwise convolution)
        self.pointwise = conv_op(
            in_channels,
            out_channels,
            1,  # 1x1卷积核
            1,  # 步长1
            padding=0,
            bias=conv_bias
        )
        
        # 在定义时直接初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化深度可分离卷积的权重和偏置"""
        # 初始化深度卷积
        init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')
        if self.depthwise.bias is not None:
            init.constant_(self.depthwise.bias, 0)
            
        # 初始化点卷积
        init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')
        if self.pointwise.bias is not None:
            init.constant_(self.pointwise.bias, 0)
    
    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)