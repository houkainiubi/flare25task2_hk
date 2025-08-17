import torch
import os
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# 1. 设置模型文件夹和checkpoint路径
model_folder = '/home/fanggang_1/hk/nnunet/nnUNet/nnunetv2/DATASET/nnUNet_results/Dataset015_flare25/nnUNetTrainerCosAnneal__nnUNetPlans__3d_lowres'  # 模型目录
checkpoint_name = 'checkpoint_final.pth'  # 只需文件名
onnx_path = 'exported_model.onnx'

# 2. 创建predictor并加载模型
predictor = nnUNetPredictor()
predictor.initialize_from_trained_model_folder(model_folder, use_folds=None, checkpoint_name=checkpoint_name)

# 3. 构造dummy输入（根据你的模型输入shape调整）
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
in_channels = determine_num_input_channels(
    predictor.plans_manager,
    predictor.configuration_manager,
    predictor.dataset_json
)
input_shape = (1, in_channels, *predictor.configuration_manager.patch_size)
# 强制模型和dummy_input都在CPU上，避免无GPU环境报错
predictor.network.to('cpu')
dummy_input = torch.randn(*input_shape, device='cpu')

# 4. 导出ONNX
predictor.network.eval()
torch.onnx.export(
    predictor.network,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f"模型已导出为ONNX: {onnx_path}")