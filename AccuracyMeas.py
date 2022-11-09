#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 21:58:21 2021

@author: alexloubser
"""
import torch
import torchaudio
# import torch.nn as nn
# from torch.nn import functional as F

import pandas as pd
# import string as stringfunc

# import pyaudio
# import threading
# import time
# import wave
import numpy as np

from Utils.Preprocessing import MFCCfull
# from Tokenizer import TextProcess
from Utils.decoder import DecodeGreedy, CTCBeamDecoder
from Utils.scorer import cer, wer

#device config
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

test_path = "/media/alexloubser/DataSSD/Masters/ComboV8/16KHzAugmentResults/Clip4"
model_file= "/media/alexloubser/DataSSD/Masters/ComboV8/16KHzAugmentResults/models/Bigmod.zip"
# model_file= "/media/alexloubser/AlexExternal/MastersCode/NewAcoustV1models/speechrecognition.zip" 
langmod = "/media/alexloubser/AlexExternal/MastersCode/kenLM/mixed-lower.binary"
# testwav = "/media/alexloubser/AlexExternal/MastersCode/testaudv1.wav" 

# general
batch_size = 1
lr = 1e-3


#hparamsnew
# num_classes = 29
n_mfcc = 16



def get_featurizer(sample_rate, n_mfcc=16):
    return MFCCfull(sample_rate=sample_rate, n_mfcc=n_mfcc)


# model = SpeechRecognition(hidden_size, num_classes, n_feats, num_layers, dropout).to(device)
model = torch.jit.load(model_file)

model.eval().to('cpu')  #run on cpu
featurizer = get_featurizer(16000, n_mfcc)
# audio_q = list()
# hidden = (torch.zeros(1, 1, 1024), torch.zeros(1, 1, 1024))
beam_results = ""
out_args = None
beam_search = CTCBeamDecoder(beam_size=100, kenlm_path=langmod)
context_length=10
context_length = context_length * 50 # multiply by 50 because each 50 from output frame is 1 second

with torch.no_grad():
  
    
    # text_process = TextProcess()
    idx = 0
    # data = pd.read_json(test_path, lines=True)
    # file_path = data.key.iloc[idx]
    waveform, samp_rate_val = torchaudio.load(test_path)
    waveform = torchaudio.functional.resample(waveform = waveform, orig_freq = samp_rate_val, new_freq = 16000)
    # sent = data['text'].iloc[idx]
    # sent = sent.translate(str.maketrans('', '', stringfunc.punctuation))
    # label = text_process.unicode_to_ascii(sent)
    # label = text_process.text_to_int_sequence(label)
    
    # waveform, _ = torchaudio.load(fname)  # don't normalize on train
    log_mel = featurizer(waveform).unsqueeze(1)
    out = model(log_mel)
    out = torch.nn.functional.softmax(out, dim=2)
    out = out.transpose(0, 1)
    out1 = out.numpy()[0]
    out2 = np.argmax(out1,1)
    out_args = out if out_args is None else torch.cat((out_args, out), dim=1)
    # results = DecodeGreedy(out_args)
    results = beam_search(out_args)
    current_context_length = out_args.shape[1] / 100  # in seconds
    print(results)
    #print(out2)
    # print(sent)

# lossval = 0
# n_samples = len(sent)
# n_pred = len(results)
# lengthmax = max(n_samples, n_pred)
# lengthmin = min(n_samples, n_pred)

# for c in range(lengthmin):
#     if sent[c] == results[c]:
#         lossval += 1
    
# acc = round(lossval / lengthmax,2)
# print(f'loss = {acc}')
    
# charerr = cer(results, sent, ignore_case=True, remove_space=False)
# print(f' cer = {charerr}')

# worderr = wer(results, sent, ignore_case=True)
# print(f' wer = {worderr}')
         
    




        
        
        
        