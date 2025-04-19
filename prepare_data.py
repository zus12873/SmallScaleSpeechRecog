#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备LibriSpeech数据集，生成JSON文件
"""

import os
import json
import pandas as pd
from tqdm import tqdm
import argparse

def find_wav_files(root_dir):
    """找到所有的.flac文件及其对应的文本文件"""
    wav_files = []
    text_dict = {}
    
    # 首先找到所有的文本文件并解析
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                txt_path = os.path.join(dirpath, filename)
                with open(txt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            file_id, text = parts
                            text_dict[file_id] = text
    
    # 然后找到所有的音频文件
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.flac'):
                file_path = os.path.join(dirpath, filename)
                file_id = filename.split('.')[0]
                if file_id in text_dict:
                    wav_files.append({
                        'key': file_path,
                        'text': text_dict[file_id]
                    })
    
    return wav_files

def main():
    parser = argparse.ArgumentParser(description='准备LibriSpeech数据集')
    parser.add_argument('--librispeech_path', type=str, required=True, help='LibriSpeech数据集路径')
    parser.add_argument('--output_dir', type=str, default='data', help='输出目录')
    parser.add_argument('--train_dirs', nargs='+', default=['train-clean-100'], help='训练数据子目录')
    parser.add_argument('--valid_dirs', nargs='+', default=['dev-clean'], help='验证数据子目录')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'valid'), exist_ok=True)
    
    # 处理训练数据
    train_files = []
    for train_dir in args.train_dirs:
        path = os.path.join(args.librispeech_path, train_dir)
        if os.path.exists(path):
            print(f"处理训练数据: {path}")
            train_files.extend(find_wav_files(path))
    
    # 处理验证数据
    valid_files = []
    for valid_dir in args.valid_dirs:
        path = os.path.join(args.librispeech_path, valid_dir)
        if os.path.exists(path):
            print(f"处理验证数据: {path}")
            valid_files.extend(find_wav_files(path))
    
    # 保存为JSON文件
    train_df = pd.DataFrame(train_files)
    valid_df = pd.DataFrame(valid_files)
    
    print(f"训练数据: {len(train_df)}条")
    print(f"验证数据: {len(valid_df)}条")
    
    train_df.to_json(os.path.join(args.output_dir, 'train.json'), orient='records', lines=True)
    valid_df.to_json(os.path.join(args.output_dir, 'valid.json'), orient='records', lines=True)
    
    print("数据处理完成！")

if __name__ == "__main__":
    main() 