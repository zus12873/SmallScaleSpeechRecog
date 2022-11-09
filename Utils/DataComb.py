#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:12:30 2022

@author: alexloubser
"""
import torch
import torchaudio
from Utils.Tokenizer import TextProcess

import pandas as pd
# import string as stringfunc

from Utils.Preprocessing import MFCCfull

class Data(torch.utils.data.Dataset):

    parameters = {
        "sample_rate": 16000, "n_mfcc": 16
    }
    
    def __init__(self, json_path, sample_rate, n_mfcc, shuffle=True, text_to_int=True, log_ex=True):
        self.text_process = TextProcess()

        print("Loading data json file from", json_path)
        self.data = pd.read_json(json_path, lines=True)
 
        self.audio_transforms = torch.nn.Sequential(
            MFCCfull(sample_rate=sample_rate, n_mfcc=n_mfcc)
        )
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        try:
            workidx = 0
            file_path = self.data.key.iloc[idx]
            waveform, sample_rate_data = torchaudio.load(file_path)
            # if idx == 0:
            #     metadata = torchaudio.info(file_path)
            #     print(metadata)
            waveform = torchaudio.functional.resample(waveform = waveform, orig_freq = sample_rate_data, new_freq = 16000)
            
            sent = self.data['text'].iloc[idx]
            # sent = sent.translate(str.maketrans('', '', stringfunc.punctuation))
            label = self.text_process.unicode_to_ascii(sent)
            label = self.text_process.text_to_int_sequence(label)
            # print(label)
            spectrogram = self.audio_transforms(waveform) # (channel, feature, time)
            spec_len = spectrogram.shape[-1] // 2
            label_len = len(label)
            if spec_len < label_len:
                raise Exception('spectrogram len is bigger then label len')
            if spectrogram.shape[0] > 1:
                raise Exception('dual channel, skipping audio file %s'%file_path)
            if spectrogram.shape[2] > 2048:
                raise Exception(str(idx) + '-index spectrogram to big. size %s'%spectrogram.shape[2])
            if label_len == 0:
                raise Exception('label len is zero... skipping %s'%file_path)
            workidx = idx    
        except Exception as e:
            # if self.log_ex:
            print(str(e)) #, file_path
            # return self.__getitem__(idx - 1 if idx != 0 else idx + 1)  
            return self.__getitem__(idx + 1 if idx != self.__len__() else workidx) 
        return spectrogram, label, spec_len, label_len
    
    