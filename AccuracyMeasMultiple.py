#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:58:53 2021

@author: alexloubser
"""
import torch
# import torchaudio
# import torch.nn as nn
# from torch.nn import functional as F

# import pandas as pd
# import string as stringfunc

# import pyaudio
# import threading
# import time
# import wave
import numpy as np

from Utils.Tokenizer import TextProcess
# from Utils.Preprocessing import LogMelSpec
from Utils.DataComb import Data
from Utils.DataProcess import collate_fn_padd

from Utils.decoder_simple import DecodeGreedy, CTCBeamDecoder
from Utils.scorer import cer, wer

# =============================================================================
# BERT
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# import numpy as np

# =============================================================================

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# test_path = "/media/alexloubser/DataSSD/Masters/ComboV8/GenerateData/Test1.json"
test_path = "/media/alexloubser/DataSSD/LibriSpeech/TestClean.json"

model_file= "/media/alexloubser/DataSSD/Masters/NewModLibri/models/Bigmod90.zip" #16KHzResults/

# general
batch_size = 1
lr = 1e-3

#hparamsnew
# num_classes = 30
# n_feats = 32
# dropout = 0.1
# hidden_size = 1024
# num_layers = 2


# def get_featurizer(sample_rate, n_feats=32):
    # return LogMelSpec(sample_rate=sample_rate, n_mels=n_feats,  win_length=160, hop_length=80)

# model = SpeechRecognition(hidden_size, num_classes, n_feats, num_layers, dropout).to(device)
model = torch.jit.load(model_file)

model.eval().to('cpu')  #run on cpu
# featurizer = get_featurizer(16000)
# hidden = (torch.zeros(1, 1, 1024), torch.zeros(1, 1, 1024))

# kenLMpath = None
kenLMpath = "/media/alexloubser/DataSSD/LibriLangMod/4-gramlow.bin"

# kenLMpath = "/home/alexloubser/Desktop/Masters1/ken-lm/kenlm/python/../lm/europarl.binlm.3"
# kenLMpath = "/media/alexloubser/AlexExternal/MastersCode/kenLM/kenLMold/5gramen.bin" 
# kenLMpath = "/media/alexloubser/AlexExternal/MastersCode/kenLM/mixed-lower.binary"
# kenLMpath = "/media/alexloubser/AlexExternal/MastersCode/kenLM/mixed_lm-lower.bin"
print("Getting LM")
# kenLMpath = "/media/alexloubser/DataSSD/LibriLangMod/3-gramlow.bin"
# kenLMpath = "/media/alexloubser/DataSSD/LibriLangMod/3-gramlow.pruned.1e-7.bin"
# kenLMpath = "/media/alexloubser/DataSSD/LibriLangMod/4-gramlow.bin"
# kenLMpath = "/media/alexloubser/DataSSD/LibriLangMod/g2p-model-5"




# from textblob import TextBlob
# import jamspell

# corrector = jamspell.TSpellCorrector()
# corrector.LoadLangModel('en.bin')

# =============================================================================
# BERT
# tokenizerlang = AutoTokenizer.from_pretrained("bert-base-uncased")

# modellang = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")


def convert(lst):
    return (lst.split())
# =============================================================================

with torch.no_grad():
    totlossval = 0
    totn_samples = 0
    totcer = 0
    totwer = 0
    out_args = None
    beam_search = CTCBeamDecoder(beam_size=500, kenlm_path= kenLMpath )
    
    # context_length=20
    # hidden = (torch.zeros(1, 1, 1024), torch.zeros(1, 1, 1024))
    # text_process = TextProcess()
    # context_length = context_length * 50 # multiply by 50 because each 50 from output frame is 1 second
    d_params = Data.parameters
     
    test_dataset = Data(json_path=test_path, **d_params)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                        batch_size=batch_size,
                        # num_workers=data_work,
                        pin_memory=True,
                        collate_fn=collate_fn_padd)  
    

    for idx,(spectrograms, labels, spec_len, label_len) in enumerate(test_loader):
        labelout = ''
        beam_results = ""
        lossval = 0
        n_samples = 0
        
        
    
    
    
        # data = pd.read_json(test_path, lines=True)
        # file_path = data.key.iloc[idx]
        # waveform, _ = torchaudio.load(file_path)
        # waveform = torchaudio.functional.resample(waveform = waveform, orig_freq = 48000, new_freq = 8000)
        # sent = data['text'].iloc[idx]
        # sent = sent.translate(str.maketrans('', '', stringfunc.punctuation))
        # label = text_process.unicode_to_ascii(sent)
        # label = text_process.text_to_int_sequence(label)
    
        # waveform, _ = torchaudio.load(fname)  # don't normalize on train
        # log_mel = featurizer(spectrograms)#.unsqueeze(1)
        
        out = model(spectrograms)
        
        out = torch.nn.functional.softmax(out, dim=2)
        out = out.transpose(0, 1)
        out1 = out.numpy()[0]
        out2 = np.argmax(out1,1)
        out_args = out #if out_args is None else torch.cat((out_args, out), dim=1)
        # print("ok")
        results = beam_search(out_args)
        # results = DecodeGreedy(out_args)
        current_context_length = out_args.shape[1] / 100  # in seconds
        # if out_args.shape[1] > context_length:
        #     out_args = None
        # return results, current_context_length
        if results[0] == " ":
            results = results[1:]
        # print("")
        # print(results)
        
         
        # results1 = TextBlob(results)
        # results1 = results1.correct()
        # resultstext = str(results1)
        # print(results1)
        # results2 = corrector.FixFragment(resultstext)
        # print(results2)
        
        # results = results1
        
        
# =============================================================================
#        BERT

        # listlet = convert(results)
        
        # finalsent = []
        # for k in range(len(listlet)):   #len(listlet)
        #     sentences = listlet.copy()
        #     sentences[k] = "[MASK]" 
        
        #     sentence = " ".join(sentences)    
            
            # print(sentence)
        
        
        #     inputs = tokenizerlang(sentence, return_tensors="pt")
        #     token_logits = modellang(**inputs).logits
        #     # Find the location of [MASK] and extract its logits
        #     mask_token_index = torch.where(inputs["input_ids"] == tokenizerlang.mask_token_id)[1]
        #     # print(mask_token_index)
        #     mask_token_logits = token_logits[0, mask_token_index, :]
        #     # print(mask_token_index)
        #     # Pick the [MASK] candidates with the highest logits
        #     top_5_tokens = torch.topk(mask_token_logits, 50, dim=1).indices[0].tolist()
            
        #     selected_words = []
        #     probs = []
            
        #     oldword = listlet[k]
            
        #     for token in top_5_tokens:
                
        #         word = tokenizerlang.decode([token])
                
        #         if any(not c.isalnum() for c in word) == False:
                    
                
        #             charerr = cer(word, oldword, ignore_case=True, remove_space=True)
                
        #             selected_words.append(word)
        #             probs.append(charerr)
        #         else :
        #             selected_words.append(word)
        #             probs.append(20)    
                
            
        #     probs = np.array(probs)
        #     minval =np.argmin(probs)  
        #     if probs[minval] > 0.9:
        #         selected_word =  oldword   
        #     else:      
        #         selected_word =  selected_words[minval]
            
        #     # listlet[k] = selected_word
        #     finalsent.append(selected_word)
            
        #     resultsf = " ".join(finalsent) 
            
        # print(f"'>>> {resultsf}")
        # # print("")
        # results = resultsf
# =============================================================================
                
        # print("")
        # print(out_args)
        # print(current_context_length)
        
        labels = labels.numpy()[0]
        # print(labels)
        text_process = TextProcess()
        labelout = text_process.int_to_text_sequence(labels)
        # print(labelout)

        # lossval = 0
        # n_samples = len(labelout)
        # n_pred = len(results)
        # lengthmax = max(n_samples, n_pred)
        # lengthmin = min(n_samples, n_pred)
        
        # for c in range(lengthmin):
            # if labelout[c] == results[c]:
                # lossval += 1
            
        # acc = round(lossval / n_pred,2)
        # print(f'Accuracy = {acc}')
        
            
        charerr = cer(labelout, results, ignore_case=True, remove_space=False)
        # print(f' cer = {charerr}')
        
        worderr = wer(labelout, results, ignore_case=True)
        # print(f' wer = {worderr}')

        # totlossval += acc     
        totn_samples +=1
        totcer +=charerr
        totwer +=worderr
       
# print(f'Total Ave Accuracy = {totlossval/totn_samples}')
print(f'Total Ave cer = {totcer/totn_samples}')
print(f'Total Ave wer = {totwer/totn_samples}')
        
        
        
        