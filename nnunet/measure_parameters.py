from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer

# 初始化nnUNet训练器
trainer = nnUNetTrainer(...)
network = trainer.network

# 计算总参数量
total_params = sum(p.numel() for p in network.parameters())
print(f"总参数量: {total_params}")

# 计算可训练参数量
trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
print(f"可训练参数量: {trainable_params}")