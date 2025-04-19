#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的模型进行语音识别
"""

import torch
import torchaudio
import argparse
from Utils.Preprocessing import MFCCfull
from Utils.decoder_simple import DecodeGreedy, CTCBeamDecoder
from Utils.Tokenizer import TextProcess

def recognize_audio(model_path, audio_path, use_beam_search=False, lm_path=None):
    """使用模型识别音频文件内容"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = torch.jit.load(model_path)
    model.eval().to(device)
    
    # 创建特征提取器
    sample_rate = 16000
    n_mfcc = 16
    n_feats = 32
    featurizer = MFCCfull(sample_rate=sample_rate, n_mfcc=n_mfcc, n_feats=n_feats)
    
    # 加载解码器
    if use_beam_search and lm_path:
        print(f"使用波束搜索解码，语言模型路径: {lm_path}")
        decoder = CTCBeamDecoder(beam_size=100, kenlm_path=lm_path)
    else:
        print("使用贪婪解码")
        decoder = DecodeGreedy
    
    # 加载并预处理音频
    print(f"加载音频: {audio_path}")
    waveform, sample_rate_orig = torchaudio.load(audio_path)
    
    # 重采样到16kHz
    if sample_rate_orig != sample_rate:
        print(f"重采样音频从 {sample_rate_orig}Hz 到 {sample_rate}Hz")
        waveform = torchaudio.functional.resample(
            waveform=waveform, 
            orig_freq=sample_rate_orig, 
            new_freq=sample_rate
        )
    
    # 提取特征
    print("提取特征...")
    features = featurizer(waveform).to(device)
    
    # 模型推理
    print("执行语音识别...")
    with torch.no_grad():
        output = model(features.unsqueeze(1))
        output = torch.nn.functional.softmax(output, dim=2)
        output = output.transpose(0, 1)
        
        # 解码
        if use_beam_search and lm_path:
            text = decoder(output)
        else:
            text = decoder(output)
    
    return text

def main():
    parser = argparse.ArgumentParser(description="使用训练好的模型进行语音识别")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径 (.pth 或 .zip)")
    parser.add_argument("--audio", type=str, required=True, help="待识别的音频文件路径")
    parser.add_argument("--beam_search", action="store_true", help="是否使用波束搜索解码")
    parser.add_argument("--lm", type=str, help="语言模型路径（使用波束搜索时可选）")
    
    args = parser.parse_args()
    
    # 识别音频
    text = recognize_audio(args.model, args.audio, args.beam_search, args.lm)
    
    # 打印结果
    print("\n识别结果:")
    print("-" * 40)
    print(text)
    print("-" * 40)

if __name__ == "__main__":
    main() 