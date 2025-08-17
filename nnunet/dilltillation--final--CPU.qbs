#!/bin/bash
#PBS -N tran3d
#PBS -l nodes=node4:ppn=8
#PBS -j oe
#PBS -o dilltillation--final--CPU.log   # 合并输出和错误到一个文件
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



# 关键修改：使用特殊重定向方式实时输出日志
{
    # 显示开始时间
    echo "==================== JOB STARTED ===================="
    echo "Start time: $(date)"
    echo "Working directory: $(pwd)"
    echo "Job ID: $PBS_JOBID"
    nvidia-smi
    # 添加标准错误输出到日志
    echo "Running nnUNet command with real-time logging..."
    
    # 使用 exec 重定向所有后续命令的输出
    exec > >(tee -a dilltillation--final--CPU.log) 2>&1
    
    # 输入需要执行的命令 - 添加后处理优化参数
    #nnUNetv2_predict -d 15 -device cpu -i /home/fanggang_1/hk/nnunet/nnUNet/200in -o /home/fanggang_1/hk/nnunet/nnUNet/200CPUoutstudent -f 0 -tr nnUNetTrainerLightweightDistillation -c 3d_lowres -p nnUNetPlans --disable_tta -npp 0 -nps 0
    nnUNetv2_predict -d 15 -device cpu -i /home/fanggang_1/hk/nnunet/nnUNet/200in -o /home/fanggang_1/hk/nnunet/nnUNet/200CPUoutstudent -f 0 -tr nnUNetTrainerLightweightDistillation2nd -c 3d_lowres -p nnUNetPlans --disable_tta -npp 0 -nps 0

    # 显示结束时间和退出状态
    echo "Exit status: $?"
    echo "End time: $(date)"
    echo "==================== JOB FINISHED ===================="
} | tee -a dilltillation--final--CPU.log
