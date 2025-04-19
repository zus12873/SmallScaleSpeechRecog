#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将训练好的PyTorch模型转换为TorchScript格式
"""

import torch
import argparse
from Utils.Acousticmod import SpeechRecognition

def convert_model(input_path, output_path):
    """将模型转换为TorchScript格式"""
    # 加载模型权重
    print(f"加载模型: {input_path}")
    checkpoint = torch.load(input_path, map_location=torch.device('cpu'))
    
    # 获取模型超参数
    h_params = SpeechRecognition.hyper_parameters
    
    # 创建模型
    model = SpeechRecognition(**h_params)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 1, 32, 100)  # 批次大小, 通道, 特征数, 时间步
    
    # 转换为TorchScript
    print("转换为TorchScript格式...")
    traced_model = torch.jit.trace(model, dummy_input)
    
    # 保存模型
    print(f"保存模型到: {output_path}")
    traced_model.save(output_path)
    
    print("转换完成!")

def main():
    parser = argparse.ArgumentParser(description="将PyTorch模型转换为TorchScript格式")
    parser.add_argument("--input", type=str, required=True, help="输入模型路径 (.pth)")
    parser.add_argument("--output", type=str, required=True, help="输出模型路径 (.zip)")
    
    args = parser.parse_args()
    
    convert_model(args.input, args.output)

if __name__ == "__main__":
    main() 