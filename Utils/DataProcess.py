#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 23:38:44 2022

@author: alexloubser
"""

import torch
import pandas as pd
# import string as stringfunc
from Utils.Tokenizer import TextProcess
import torch.nn as nn
import torchaudio
from Utils.Preprocessing import MFCCfull, MFCC,  SpecStretch, Spec  #SpecAugment,

class Data(torch.utils.data.Dataset):

    parameters = {
        "sample_rate": 16000, "n_feats": 81,
        "specaug_rate": 0.5, "specaug_policy": 3,
        "time_mask": 30, "freq_mask": 8, "n_mfcc": 16, "n_fft" : 400
    }
    
    def __init__(self, json_path, data_path, sample_rate, n_feats, specaug_rate, specaug_policy,
                time_mask, freq_mask, n_mfcc, n_fft, valid=False, stretch_max=0.9, stretch_min=1.1, shuffle=True, text_to_int=True, log_ex=True):
        
        self.text_process = TextProcess()
        self.sample_rate = sample_rate
        # print("Loading data json file from", json_path)
        self.data = pd.read_json(json_path, lines=True)
        # self.features = data_path
        
        if valid:
            self.audio_transforms = torch.nn.Sequential(
                MFCCfull(sample_rate, n_mfcc, n_feats)
                )
        else:
            self.audio_transforms = torch.nn.Sequential(
                Spec(n_fft = n_fft),
                # SpecAugment(specaug_rate, specaug_policy, freq_mask, time_mask),
                SpecStretch(specaug_rate, specaug_policy, stretch_max=stretch_max, stretch_min=stretch_min, n_freq= (n_fft//2+1)),
                MFCC(n_feats,sample_rate, n_mfcc, n_fft) 
                )
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        try:
            workidx = 0

            sent = self.data['text'].iloc[idx]
            label = self.text_process.unicode_to_ascii(sent)
            label = self.text_process.text_to_int_sequence(label)
            
# =============================================================================
            file_path = self.data.key.iloc[idx]
            waveform, sample_rate_data = torchaudio.load(file_path)
            waveform = torchaudio.functional.resample(waveform = waveform, orig_freq = sample_rate_data, new_freq = self.sample_rate)
            MFCCvar = self.audio_transforms(waveform)
# =============================================================================
            
            
            # print('1')
            # melspec = torch.load(self.features + str(idx)+".pt")
            # print('2')
            # MFCCvar = self.audio_transforms(melspec)
            # print('3')
            
            spec_len = MFCCvar.shape[-1] // 2
            label_len = len(label)
            
            # print(spectrogram.size())
            # print(label_len)
            
            # if spec_len < label_len:
                # raise Exception('spectrogram len is bigger then label len')
            if MFCCvar.shape[0] > 1:
                raise Exception('dual channel, skipping audio file %s'%file_path)
            if MFCCvar.shape[2] > 1650:
                raise Exception(str(idx) + '-index spectrogram to big. size %s'%MFCCvar.shape[2])
            if label_len == 0:
                raise Exception('label len is zero... skipping %s')
            workidx = idx    
        except Exception as e:
            # if self.log_ex:
            print(str(e)) #, file_path
            # return self.__getitem__(idx - 1 if idx != 0 else idx + 1)  
            return self.__getitem__(idx + 1 if idx != self.__len__() else workidx) 
        return MFCCvar, label, spec_len, label_len
    
    
    
def collate_fn_padd(data):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    # print(data)
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (spectrogram, label, input_length, label_length) in data:
        if spectrogram is None:
            print("help")
            continue
        # print(spectrogram.shape)
        spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
        labels.append(torch.Tensor(label))
        input_lengths.append(input_length)
        label_lengths.append(label_length)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    input_lengths = input_lengths
    # print(spectrograms.shape)
    label_lengths = label_lengths
    # ## compute mask
    # mask = (batch != 0).cuda(gpu)
    # return batch, lengths, mask
    return spectrograms, labels, input_lengths, label_lengths   