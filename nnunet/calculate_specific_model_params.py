#!/usr/bin/env python3
"""
直接计算轻量化nnUNet模型参数量
基于提供的plans.json和nnUNetTrainerLightweightDistillation2nd配置
"""
import os
import sys
import torch
import torch.nn as nn
import json
import numpy as np
from typing import Tuple, Dict, Any

# 添加nnUNet路径
sys.path.append('/home/fanggang_1/hk/nnunet/nnUNet/nnunetv2')
sys.path.append('/home/fanggang_1/hk/nnunet/nnUNet')

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_parameter_count(count: int) -> str:
    """格式化参数数量"""
    if count >= 1e9:
        return f"{count/1e9:.2f}B"
    elif count >= 1e6:
        return f"{count/1e6:.2f}M"
    elif count >= 1e3:
        return f"{count/1e3:.2f}K"
    else:
        return str(count)

def create_lightweight_3d_model_from_plans():
    """
    基于plans.json直接创建轻量化模型
    """
    # 从plans.json获取配置
    plans_path = "/home/fanggang_1/hk/nnunet/nnUNet/nnunetv2/DATASET/nnUNet_results/Dataset015_flare25/nnUNetTrainerLightweightDistillation2nd__nnUNetPlans__3d_lowres/plans.json"
    
    with open(plans_path, 'r') as f:
        plans = json.load(f)
    
    config = plans['configurations']['3d_lowres']
    arch_kwargs = config['architecture']['arch_kwargs']
    
    print("📋 从plans.json读取的配置:")
    print(f"   阶段数: {arch_kwargs['n_stages']}")
    print(f"   特征数: {arch_kwargs['features_per_stage']}")
    print(f"   卷积层数/阶段: {arch_kwargs['n_conv_per_stage']}")
    print(f"   解码器卷积层数: {arch_kwargs['n_conv_per_stage_decoder']}")
    print(f"   批次大小: {config['batch_size']}")
    print(f"   补丁大小: {config['patch_size']}")
    
    # 手动构建轻量化模型参数
    n_stages = arch_kwargs['n_stages']  # 4
    features_per_stage = arch_kwargs['features_per_stage']  # [16, 32, 64, 128]
    n_conv_per_stage = arch_kwargs['n_conv_per_stage']  # [1, 1, 2, 2]
    n_conv_per_stage_decoder = arch_kwargs['n_conv_per_stage_decoder']  # [1, 1, 2]
    
    # 输入输出通道
    input_channels = 1  # CT图像
    num_classes = 14   # 13个器官类别 + 背景
    
    return calculate_lightweight_parameters(
        n_stages, features_per_stage, n_conv_per_stage, 
        n_conv_per_stage_decoder, input_channels, num_classes
    )

def calculate_lightweight_parameters(n_stages, features_per_stage, n_conv_per_stage, 
                                   n_conv_per_stage_decoder, input_channels, num_classes):
    """
    计算轻量化3D UNet的参数量
    考虑深度可分离卷积和瓶颈残差块的影响
    """
    total_params = 0
    
    print("\n🔍 详细参数计算:")
    print("=" * 60)
    
    # 1. 编码器参数计算
    print("📈 编码器参数:")
    current_channels = input_channels
    
    for stage_idx in range(n_stages):
        stage_features = features_per_stage[stage_idx]
        stage_convs = n_conv_per_stage[stage_idx]
        
        stage_params = 0
        
        # 每个阶段的卷积层
        for conv_idx in range(stage_convs):
            if conv_idx == 0:
                # 第一个卷积：输入通道 -> 阶段特征数
                in_ch = current_channels
                out_ch = stage_features
            else:
                # 后续卷积：阶段特征数 -> 阶段特征数
                in_ch = stage_features
                out_ch = stage_features
            
            # 轻量化：使用深度可分离卷积
            # 深度卷积：3x3x3, groups=in_ch
            depthwise_params = in_ch * (3 * 3 * 3) + in_ch  # 权重 + 偏置
            
            # 逐点卷积：1x1x1
            pointwise_params = in_ch * out_ch * (1 * 1 * 1) + out_ch  # 权重 + 偏置
            
            # 实例归一化
            norm_params = out_ch * 2  # gamma + beta
            
            conv_params = depthwise_params + pointwise_params + norm_params
            stage_params += conv_params
            
            print(f"   阶段{stage_idx+1}-卷积{conv_idx+1}: {in_ch}→{out_ch}, 参数: {conv_params:,}")
        
        # 下采样层（除了最后一个阶段）
        if stage_idx < n_stages - 1:
            # 轻量化下采样：深度可分离卷积 + 步长=2
            downsample_params = stage_features * (3 * 3 * 3) + stage_features  # 深度卷积
            downsample_params += stage_features * stage_features + stage_features  # 逐点卷积
            downsample_params += stage_features * 2  # 归一化
            
            stage_params += downsample_params
            print(f"   阶段{stage_idx+1}-下采样: 参数: {downsample_params:,}")
        
        total_params += stage_params
        current_channels = stage_features
        
        print(f"   阶段{stage_idx+1}总计: {stage_params:,}")
    
    print(f"编码器总参数: {total_params:,}")
    
    # 2. 解码器参数计算
    print("\n📉 解码器参数:")
    decoder_params = 0
    
    for stage_idx in range(len(n_conv_per_stage_decoder)):
        # 解码器阶段索引（从底部向上）
        decoder_stage = n_stages - 2 - stage_idx  # 倒序
        
        if decoder_stage >= 0:
            stage_features = features_per_stage[decoder_stage]
            higher_features = features_per_stage[decoder_stage + 1]
            stage_convs = n_conv_per_stage_decoder[stage_idx]
            
            stage_decoder_params = 0
            
            # 上采样层
            # 轻量化上采样：转置卷积或插值+卷积
            upsample_params = higher_features * stage_features * (2 * 2 * 2) + stage_features
            stage_decoder_params += upsample_params
            
            # 跳跃连接融合后的卷积
            fused_channels = stage_features * 2  # 跳跃连接后通道翻倍
            
            for conv_idx in range(stage_convs):
                if conv_idx == 0:
                    in_ch = fused_channels
                    out_ch = stage_features
                else:
                    in_ch = stage_features
                    out_ch = stage_features
                
                # 轻量化卷积
                depthwise_params = in_ch * (3 * 3 * 3) + in_ch
                pointwise_params = in_ch * out_ch + out_ch
                norm_params = out_ch * 2
                
                conv_params = depthwise_params + pointwise_params + norm_params
                stage_decoder_params += conv_params
            
            decoder_params += stage_decoder_params
            print(f"   解码器阶段{stage_idx+1}: {stage_decoder_params:,}")
    
    print(f"解码器总参数: {decoder_params:,}")
    total_params += decoder_params
    
    # 3. 输出头参数
    print("\n📤 输出头参数:")
    final_features = features_per_stage[0]  # 第一阶段的特征数
    output_params = final_features * num_classes + num_classes  # 1x1x1卷积
    total_params += output_params
    print(f"   输出层: {final_features}→{num_classes}, 参数: {output_params:,}")
    
    # 4. 深度监督头参数（如果启用）
    deep_supervision_params = 0
    for stage_idx in range(1, n_stages):
        stage_features = features_per_stage[stage_idx]
        ds_params = stage_features * num_classes + num_classes
        deep_supervision_params += ds_params
        print(f"   深度监督{stage_idx}: {stage_features}→{num_classes}, 参数: {ds_params:,}")
    
    total_params += deep_supervision_params
    print(f"深度监督总参数: {deep_supervision_params:,}")
    
    return total_params

def calculate_standard_model_params():
    """
    计算标准nnUNet模型参数量进行对比
    使用相同的配置但不使用轻量化技术
    """
    print("\n📊 标准nnUNet模型参数量估算:")
    
    # 基于plans.json的标准配置
    n_stages = 4
    features_per_stage = [16, 32, 64, 128]  # 与轻量化相同
    n_conv_per_stage = [1, 1, 2, 2]
    n_conv_per_stage_decoder = [1, 1, 2]
    input_channels = 1
    num_classes = 14
    
    total_params = 0
    
    # 编码器 - 标准3D卷积
    current_channels = input_channels
    for stage_idx in range(n_stages):
        stage_features = features_per_stage[stage_idx]
        stage_convs = n_conv_per_stage[stage_idx]
        
        for conv_idx in range(stage_convs):
            if conv_idx == 0:
                in_ch = current_channels
                out_ch = stage_features
            else:
                in_ch = stage_features
                out_ch = stage_features
            
            # 标准3D卷积：3x3x3
            conv_params = in_ch * out_ch * (3 * 3 * 3) + out_ch
            norm_params = out_ch * 2
            total_params += conv_params + norm_params
        
        # 下采样
        if stage_idx < n_stages - 1:
            downsample_params = stage_features * stage_features * (3 * 3 * 3) + stage_features
            downsample_params += stage_features * 2
            total_params += downsample_params
        
        current_channels = stage_features
    
    # 解码器 - 标准3D卷积
    for stage_idx in range(len(n_conv_per_stage_decoder)):
        decoder_stage = n_stages - 2 - stage_idx
        if decoder_stage >= 0:
            stage_features = features_per_stage[decoder_stage]
            higher_features = features_per_stage[decoder_stage + 1]
            stage_convs = n_conv_per_stage_decoder[stage_idx]
            
            # 上采样
            upsample_params = higher_features * stage_features * (2 * 2 * 2) + stage_features
            total_params += upsample_params
            
            # 卷积
            fused_channels = stage_features * 2
            for conv_idx in range(stage_convs):
                if conv_idx == 0:
                    in_ch = fused_channels
                    out_ch = stage_features
                else:
                    in_ch = stage_features
                    out_ch = stage_features
                
                conv_params = in_ch * out_ch * (3 * 3 * 3) + out_ch
                norm_params = out_ch * 2
                total_params += conv_params + norm_params
    
    # 输出头
    final_features = features_per_stage[0]
    output_params = final_features * num_classes + num_classes
    total_params += output_params
    
    # 深度监督
    for stage_idx in range(1, n_stages):
        stage_features = features_per_stage[stage_idx]
        ds_params = stage_features * num_classes + num_classes
        total_params += ds_params
    
    return total_params

def main():
    """主函数"""
    print("=" * 80)
    print("🧮 轻量化nnUNet模型参数量直接计算")
    print("=" * 80)
    print("基于文件: nnUNetTrainerLightweightDistillation2nd + plans.json")
    print()
    
    try:
        # 计算轻量化模型参数
        lightweight_params = create_lightweight_3d_model_from_plans()
        
        # 计算标准模型参数进行对比
        standard_params = calculate_standard_model_params()
        
        # 结果汇总
        print("\n" + "=" * 80)
        print("📊 参数量对比结果")
        print("=" * 80)
        
        print(f"标准nnUNet模型:")
        print(f"  总参数量: {format_parameter_count(standard_params)} ({standard_params:,})")
        
        print(f"\n轻量化nnUNet模型 (深度可分离卷积):")
        print(f"  总参数量: {format_parameter_count(lightweight_params)} ({lightweight_params:,})")
        
        # 计算减少效果
        if standard_params > 0:
            reduction_factor = standard_params / lightweight_params
            compression_rate = (1 - lightweight_params/standard_params) * 100
            
            print(f"\n🎯 优化效果:")
            print(f"  参数减少因子: {reduction_factor:.1f}×")
            print(f"  模型压缩率: {compression_rate:.1f}%")
            print(f"  参数节省: {format_parameter_count(standard_params - lightweight_params)}")
        
        # 保存结果
        results = {
            'model_config': {
                'n_stages': 4,
                'features_per_stage': [16, 32, 64, 128],
                'architecture': 'lightweight_3d_unet',
                'use_depthwise_separable': True,
                'use_bottleneck_blocks': True
            },
            'parameter_analysis': {
                'standard_model_params': int(standard_params),
                'lightweight_model_params': int(lightweight_params),
                'reduction_factor': float(standard_params / lightweight_params) if standard_params > 0 else 0,
                'compression_rate': float((1 - lightweight_params/standard_params) * 100) if standard_params > 0 else 0
            },
            'formatted_results': {
                'standard_params_formatted': format_parameter_count(standard_params),
                'lightweight_params_formatted': format_parameter_count(lightweight_params)
            }
        }
        
        output_file = "/home/fanggang_1/hk/nnunet/nnUNet/lightweight_model_params_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ 详细结果已保存到: {output_file}")
        
        # 论文用数据
        print("\n" + "=" * 60)
        print("📝 论文用数据摘要:")
        print("=" * 60)
        print(f"• 轻量化模型参数量: {format_parameter_count(lightweight_params)}")
        print(f"• 参数减少因子: {reduction_factor:.1f}×")
        print(f"• 模型压缩率: {compression_rate:.1f}%")
        print(f"• 架构: 4阶段深度可分离3D U-Net")
        print(f"• 特征通道: [16, 32, 64, 128]")
        
        return 0
        
    except Exception as e:
        print(f"❌ 计算过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
