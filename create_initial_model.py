#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建初始模型
"""

import torch
import torch.nn as nn
import os
from Utils.Acousticmod import SpeechRecognition

# 创建目录
os.makedirs('models', exist_ok=True)

# 定义模型参数
hidden_size = 1024
num_classes = 30
n_feats = 32
num_layers = 2
dropout = 0.1

# 创建设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建模型
model = SpeechRecognition(hidden_size, num_classes, n_feats, num_layers, dropout).to(device)

# 创建优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 创建检查点
checkpoint = {
    "epoch": 0,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}

# 保存模型
torch.save(checkpoint, "models/initial_model.pth")
print("初始模型已保存到 models/initial_model.pth") 