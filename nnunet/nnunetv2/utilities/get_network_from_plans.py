import pydoc
import warnings
import logging
from typing import Union

from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join

# 设置网络构建日志
network_builder_logger = logging.getLogger('nnunet.network_builder')
if not network_builder_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    network_builder_logger.addHandler(handler)
    network_builder_logger.setLevel(logging.INFO)


def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)
    
    # 检查是否应该使用轻量级版本
    use_lightweight = False
    
    # 方法1: 检查环境变量
    import os
    if os.environ.get('NNUNET_USE_LIGHTWEIGHT', '').lower() == 'true':
        use_lightweight = True
        network_builder_logger.info("🚀 检测到轻量级模式环境变量")
    
    # 方法2: 检查调用栈中的轻量级训练器
    if not use_lightweight:
        import inspect
        frame = inspect.currentframe()
        try:
            caller_frame = frame
            for _ in range(10):  # 最多查找10层调用栈
                caller_frame = caller_frame.f_back
                if caller_frame is None:
                    break
                
                # 检查调用者的类名或变量
                if 'self' in caller_frame.f_locals:
                    caller_obj = caller_frame.f_locals['self']
                    if hasattr(caller_obj, '__class__'):
                        class_name = caller_obj.__class__.__name__
                        if 'Lightweight' in class_name:
                            use_lightweight = True
                            network_builder_logger.info(f"🚀 检测到轻量级训练器: {class_name}")
                            break
        finally:
            del frame
    
    # 如果是PlainConvUNet且检测到轻量级模式，自动使用轻量级版本
    if (use_lightweight and 
        'PlainConvUNet' in network_class and 
        'Lightweight' not in network_class):
        
        original_class = network_class
        network_class = network_class.replace('PlainConvUNet', 'LightweightPlainConvUNet')
        
        network_builder_logger.info(f"⚡ 自动切换到轻量级网络:")
        network_builder_logger.info(f"   原始: {original_class}")
        network_builder_logger.info(f"   轻量级: {network_class}")
        
        # LightweightPlainConvUNet有自己的轻量级参数，只添加它接受的参数
        architecture_kwargs.update({
            'enable_pyramid_pooling': True,
            'bottleneck_ratio': 0.25,
            'pyramid_pool_stages': [1, 2]  # 在第1,2阶段添加金字塔池化
        })
        
        network_builder_logger.info("🔧 轻量级参数: 瓶颈比例=0.25, 金字塔池化阶段=[1,2]")
    
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = pydoc.locate(network_class)
    # sometimes things move around, this makes it so that we can at least recover some of that
    if nw_class is None:
        warnings.warn(f'Network class {network_class} not found. Attempting to locate it within '
                      f'dynamic_network_architectures.architectures...')
        import dynamic_network_architectures
        nw_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                               network_class.split(".")[-1],
                                               'dynamic_network_architectures.architectures')
        if nw_class is not None:
            print(f'FOUND IT: {nw_class}')
        else:
            raise ImportError('Network class could not be found, please check/correct your plans file')

    if deep_supervision is not None:
        architecture_kwargs['deep_supervision'] = deep_supervision

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network

if __name__ == "__main__":
    import torch

    model = get_network_from_plans(
        arch_class_name="dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
        arch_kwargs={
            "n_stages": 7,
            "features_per_stage": [32, 64, 128, 256, 512, 512, 512],
            "conv_op": "torch.nn.modules.conv.Conv2d",
            "kernel_sizes": [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
            "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
            "n_blocks_per_stage": [1, 3, 4, 6, 6, 6, 6],
            "n_conv_per_stage_decoder": [1, 1, 1, 1, 1, 1],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
        input_channels=1,
        output_channels=4,
        allow_init=True,
        deep_supervision=True,
    )
    data = torch.rand((8, 1, 256, 256))
    target = torch.rand(size=(8, 1, 256, 256))
    outputs = model(data) # this should be a list of torch.Tensor