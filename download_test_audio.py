#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载LibriSpeech测试音频文件
"""

import os
import urllib.request
import tarfile
import argparse

def download_and_extract(output_dir="test_audio"):
    """下载并解压LibriSpeech测试数据集"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载URL，使用小数据集以快速下载
    url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    tar_path = os.path.join(output_dir, "test-clean.tar.gz")
    
    # 下载文件
    if not os.path.exists(tar_path):
        print(f"正在下载LibriSpeech测试数据集到 {tar_path}...")
        urllib.request.urlretrieve(url, tar_path)
        print("下载完成!")
    else:
        print(f"文件已存在: {tar_path}")
    
    # 解压文件
    extract_path = os.path.join(output_dir, "LibriSpeech")
    if not os.path.exists(extract_path):
        print(f"正在解压数据集到 {extract_path}...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=output_dir)
        print("解压完成!")
    else:
        print(f"数据已解压: {extract_path}")
    
    # 查找示例音频文件
    audio_files = []
    test_dir = os.path.join(output_dir, "LibriSpeech", "test-clean")
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith(".flac"):
                audio_files.append(os.path.join(root, file))
                if len(audio_files) >= 5:  # 只获取5个示例
                    break
        if len(audio_files) >= 5:
            break
    
    print("\n示例音频文件:")
    for i, file in enumerate(audio_files):
        print(f"{i+1}. {file}")
    
    return audio_files

def main():
    parser = argparse.ArgumentParser(description="下载LibriSpeech测试音频文件")
    parser.add_argument("--output_dir", type=str, default="test_audio", help="输出目录")
    
    args = parser.parse_args()
    
    download_and_extract(args.output_dir)

if __name__ == "__main__":
    main() 