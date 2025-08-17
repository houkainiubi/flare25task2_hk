#!/bin/bash
#PBS -N tran3d
#PBS -l nodes=node1:ppn=7
#PBS -j oe
#PBS -o trani-outputasdasd.log   # 合并输出和错误到一个文件
#PBS -q gentai
cd $PBS_O_WORKDIR
NPROCS=`wc -l < $PBS_NODEFILE`

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
    exec > >(tee -a trani-outputasdasd.log) 2>&1
    
    # 输入需要执行的命令
    nvidia-smi
    lscpu
    python /home/fanggang_1/hk/nnunet/nnUNet/fix_model_weights.py
    # 显示结束时间和退出状态
    echo "Exit status: $?"
    echo "End time: $(date)"
    echo "==================== JOB FINISHED ===================="
} | tee -a trani-outputasdasd.log
