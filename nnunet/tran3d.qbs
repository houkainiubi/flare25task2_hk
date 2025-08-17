#!/bin/bash
#PBS -N tran3d
#PBS -l nodes=node5:ppn=8
export STUDENT_PLANS_NAME="nnUNetPlans"  # 快速版本（更大spacing）

# 🔧 训练优化参数
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"  # 优化GPU内存分配
export OMP_NUM_THREADS="4"                               # 限制CPU线程数

# 📊 spacing优化说明:
# 快速版本spacing变化:
#   3d_lowres:  [1.97→2.5, 1.54→2.0, 1.54→2.0] mm  (~27%速度提升)
#   3d_fullres: [1.0→1.5, 0.78→1.2, 0.78→1.2] mm   (~35%速度提升)
# 预计总体推理加速: 30-40%，精度损失: <3%ode5:ppn=8
#PBS -j oe
#PBS -o trani-output.log   # 合并输出和错误到一个文件
#PBS -q gentai
cd $PBS_O_WORKDIR
NPROCS=`wc -l < $PBS_NODEFILE`

#dos2unix test.qbs命令行运行，消除win换行符
# 加载 Conda 环境
source ~/.bashrc
conda activate hk_nnunet_test1
export nnUNet_preprocessed="/home/fanggang_1/hk/nnunet/nnUNet/nnunetv2/DATASET/nnUNet_preprocessed"
export nnUNet_raw="/home/fanggang_1/hk/nnunet/nnUNet/nnunetv2/DATASET/nnUNet_raw"
export nnUNet_results="/home/fanggang_1/hk/nnunet/nnUNet/nnunetv2/DATASET/nnUNet_results"

# =====================================================
# 二次轻量级知识蒸馏训练环境变量设置 (5层→4层)
# =====================================================

# 🎓🎓 二次蒸馏：教师模型配置 (使用第一阶段的5层轻量级模型)
export TEACHER_PLANS_PATH="/home/fanggang_1/hk/nnunet/nnUNet/nnunetv2/DATASET/nnUNet_results/midstudent/nnUNetPlans.json"
export TEACHER_CONFIGURATION="3d_lowres"
export TEACHER_CHECKPOINT="/home/fanggang_1/hk/nnunet/nnUNet/nnunetv2/DATASET/nnUNet_results/Dataset015_flare25/nnUNetTrainerLightweightDistillation__nnUNetPlans__3d_lowres/fold_0/checkpoint_final.pth"
export TEACHER_FOLD="0"

# 🎯 二次蒸馏参数 (针对4层超轻量级模型优化)
export DISTILL_LOSS_TYPE="progressive_kl"  # 渐进式蒸馏损失
export DISTILL_TEMPERATURE_2ND="2.0"       # 二次蒸馏温度 (降低以保持性能)
export DISTILL_ALPHA_2ND="0.8"             # 二次蒸馏权重 (提高)
export DISTILL_BETA_2ND="0.2"              # 任务损失权重 (降低)

# ⚡ 轻量级网络标识
export NNUNET_USE_LIGHTWEIGHT="true"       # 启用轻量级架构

# 🏗️  4层超轻量级学生模型Plans文件选项
export STUDENT_PLANS_NAME="nnUNetPlans"  # 快速版本（更大spacing）
# export STUDENT_PLANS_NAME="（student-4stage）nnUNetPlans"     # 标准版本（原spacing）
# export STUDENT_PLANS_NAME="nnUNetPlans"                      # 使用标准Plans（已是4层）

# 🔧 训练优化参数
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"  # 优化GPU内存分配
export OMP_NUM_THREADS="4"                               # 限制CPU线程数

# 关键修改：使用特殊重定向方式实时输出日志
{
    # 显示开始时间
    echo "==================== JOB STARTED ===================="
    echo "Start time: $(date)"
    echo "Working directory: $(pwd)"
    echo "Job ID: $PBS_JOBID"
    
    # 📊 系统信息
    nvidia-smi
    echo "Python version: $(python --version)"
    echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
    
    # 🔍 验证二次蒸馏环境
    echo "==================== 二次蒸馏环境验证 ===================="
    echo "教师模型路径: $TEACHER_CHECKPOINT"
    echo "4层学生Plans: $STUDENT_PLANS_NAME"
    echo "蒸馏温度: $DISTILL_TEMPERATURE_2ND"
    echo "蒸馏权重 α: $DISTILL_ALPHA_2ND, β: $DISTILL_BETA_2ND"
    
    # 检查关键文件
    if [ -f "$TEACHER_CHECKPOINT" ]; then
        echo "✅ 教师模型文件存在"
    else
        echo "❌ 教师模型文件不存在: $TEACHER_CHECKPOINT"
    fi
    
    STUDENT_PLANS_PATH="$nnUNet_preprocessed/Dataset015_flare25/$STUDENT_PLANS_NAME.json"
    if [ -f "$STUDENT_PLANS_PATH" ]; then
        echo "✅ 4层学生Plans文件存在: $STUDENT_PLANS_NAME"
        echo "   路径: $STUDENT_PLANS_PATH"
    else
        echo "❌ 4层学生Plans文件不存在: $STUDENT_PLANS_PATH"
        echo "   请确保已创建对应的Plans配置文件"
    fi
    echo "==========================================================="
    
    # 添加标准错误输出到日志
    echo "Running nnUNet 二次蒸馏训练 with real-time logging..."
    
    # 使用 exec 重定向所有后续命令的输出
    exec > >(tee -a trani-output.log) 2>&1
    
    # 输入需要执行的命令
    #nnUNetv2_plan_and_preprocess -d 15 --verify_dataset_integrity
    export CUDA_VISIBLE_DEVICES=0,1
    
    # ===============================
    # 训练选项 (请根据需要取消注释)
    # ===============================
    
    # 🎓🎓 选项1: 二次轻量级知识蒸馏训练 (5层→4层超轻量级) - 快速版本
    # 使用第一阶段训练好的5层轻量级模型作为教师，训练4层超轻量级学生模型
    # 快速版本：更大spacing，预计加速30-40%，batch_size提升至8
    nnUNetv2_train 15 3d_lowres 0 -tr nnUNetTrainerLightweightDistillation2nd -p "$STUDENT_PLANS_NAME" 
    
    # 🎓 选项2: 第一阶段轻量级知识蒸馏训练 (6层→5层) [已完成]
    #nnUNetv2_train 15 3d_lowres 0 -tr nnUNetTrainerLightweightDistillation --c
    
    # ⚡ 选项3: 纯轻量级训练 (无蒸馏)
    #nnUNetv2_train 15 3d_lowres 0 -tr nnUNetTrainerLightweight --c
    
    # 🔄 选项4: 原始余弦退火训练
    #nnUNetv2_train 15 3d_lowres 0 -num_gpus 2 -tr nnUNetTrainerCosAnneal --c

    
    # 显示结束时间和退出状态
    echo "Exit status: $?"
    echo "End time: $(date)"
    echo "==================== JOB FINISHED ===================="
} | tee -a trani-output.log
