#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:57:31 2022

@author: alexloubser
"""
import torch
import torchaudio
import torch.nn as nn

import math
# import warnings
from typing import  Optional  #Callable,
from torch import Tensor
from torchaudio import functional as F

class AmplitudeToDB(torch.nn.Module):
    __constants__ = ['multiplier', 'amin', 'ref_value', 'db_multiplier']

    def __init__(self, stype: str = 'power', top_db: Optional[float] = None) -> None:
        super(AmplitudeToDB, self).__init__()
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError('top_db must be positive value')
        self.top_db = top_db
        self.multiplier = 10.0 if stype == 'power' else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def forward(self, x: Tensor) -> Tensor:

        return F.amplitude_to_DB(x, self.multiplier, self.amin, self.db_multiplier, self.top_db)

class MFCC(nn.Module):

    def __init__(self, n_feats=128, sample_rate=16000, n_mfcc=16, n_fft=400): # sample_rate=8000, n_mels=81, n_mfcc=16, win_length=160, hop_length=80
        super(MFCC, self).__init__()

        self.deltas = torchaudio.transforms.ComputeDeltas(win_length=4)
        self.dct_mat = F.create_dct(n_mfcc, n_feats, norm='ortho')
        self.top_db = 80
        self.amplitude_to_DB_var = AmplitudeToDB('power', self.top_db)
        self.mel_scale = torchaudio.transforms.MelScale(n_feats, sample_rate, n_stft=(n_fft // 2 + 1))
        

    def forward(self, x):
        # x = self.transform(x)  # mel spectrogram
        x=x.abs().pow(2)
        x = self.mel_scale(x)
        x = self.amplitude_to_DB_var(x)
        x = torch.matmul(x.transpose(-2, -1), self.dct_mat).transpose(-2, -1)
        # x = self.MFCC(x)
        y = self.deltas(x)
        z = torch.cat((x,y),1)
        
        return z
    
class MFCCfull(nn.Module):

    def __init__(self, sample_rate=16000, n_mfcc=16, n_feats=81):
        super(MFCCfull, self).__init__()

        self.MFCC = torchaudio.transforms.MFCC(
                            sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs = {"n_mels": n_feats})
        self.deltas = torchaudio.transforms.ComputeDeltas(win_length=4)

    def forward(self, x):
        # x = self.transform(x)  # mel spectrogram
        
        x = self.MFCC(x)
        y = self.deltas(x)
        z = torch.cat((x,y),1)
        
        # z = self.transform(x)  # mel spectrogram
        # z = np.log(z + 1e-14)  # logrithmic, add small value to avoid inf
        return z    

# class MFCCfull(nn.Module):

#     def __init__(self, sample_rate=16000,n_feats=128, win_length=400, hop_length=200, n_mfcc=16):
#         super(MFCCfull, self).__init__()

#         self.transform = torchaudio.transforms.MelSpectrogram(
#                             sample_rate=sample_rate, n_mels=n_feats,
#                             win_length=win_length, hop_length=hop_length)
#         self.deltas = torchaudio.transforms.ComputeDeltas(win_length=4)
#         self.dct_mat = F.create_dct(n_mfcc, n_feats, norm='ortho')
#         self.top_db = 80
#         self.amplitude_to_DB_var = AmplitudeToDB('power', self.top_db)


#     def forward(self, x):
#         # x = self.transform(x)  # mel spectrogram
#         x = self.transform(x) 
#         x = self.amplitude_to_DB_var(x)
#         x = torch.matmul(x.transpose(-2, -1), self.dct_mat).transpose(-2, -1)
#         # x = self.MFCC(x)
#         y = self.deltas(x)
#         z = torch.cat((x,y),1)
        
#         # z = self.transform(x)  # mel spectrogram
#         # z = np.log(z + 1e-14)  # logrithmic, add small value to avoid inf
#         return z    
    
class Spec(nn.Module):

    def __init__(self, sample_rate=16000,n_feats=128,n_fft=400):
        super(Spec, self).__init__()

        self.transform = torchaudio.transforms.Spectrogram(power=None, n_fft=n_fft)
       
    def forward(self, x):
        # x = self.transform(x)  # mel spectrogram
        z = self.transform(x) 

        return z    
    
# class SpecAugment(nn.Module):

#     def __init__(self, rate, policy=3, freq_mask=15, time_mask=35):
#         super(SpecAugment, self).__init__()

#         self.rate = rate
        
#         self.specaug = nn.Sequential(
#             torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
#             torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
#         )

#         self.specaug2 = nn.Sequential(
#             torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
#             torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
#             torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
#             torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
#         )
        

#         policies = { 1: self.policy1, 2: self.policy2, 3: self.policy3 }
#         self._forward = policies[policy]

#     def forward(self, x):
#         return self._forward(x)

#     def policy1(self, x):
#         probability = torch.rand(1, 1).item()
#         if self.rate > probability:
#             return  self.specaug(x)
#         return x

#     def policy2(self, x):
#         probability = torch.rand(1, 1).item()
#         if self.rate > probability:
#             return  self.specaug2(x)
#         return x

#     def policy3(self, x):
#         probability = torch.rand(1, 1).item()
#         if probability > 0.5:
#             return self.policy1(x)
#         return self.policy2(x)   
    

class SpecStretch(nn.Module):

    def __init__(self, rate, policy=3, stretch_max=0.9, stretch_min=1.1, n_freq= 201):
        super(SpecStretch, self).__init__()

        self.rate = rate
        
        self.stretch = nn.Sequential(
            torchaudio.transforms.TimeStretch(fixed_rate=stretch_max, n_freq = n_freq)
        )

        self.stretch2 = nn.Sequential(
            torchaudio.transforms.TimeStretch(fixed_rate=stretch_min, n_freq = n_freq)
        )
        
        policies = { 1: self.policy1, 2: self.policy2, 3: self.policy3 }
        self._forward = policies[policy]

    def forward(self, x):
        # print(x.size())
        y = self._forward(x)
        # print(y.size())
        
        return y

    def policy1(self, x):
        probability = torch.rand(1, 1).item()
        # print(probability)
        if self.rate > probability:
            return  self.stretch(x)
        return x

    def policy2(self, x):
        probability = torch.rand(1, 1).item()
        # print(probability)
        if self.rate > probability:
            return  self.stretch2(x)
        return x

    def policy3(self, x):
        probability = torch.rand(1, 1).item()
        # print(probability)
        # print(x.size())
        if probability > self.rate:
            return self.policy1(x)
        return self.policy2(x)      