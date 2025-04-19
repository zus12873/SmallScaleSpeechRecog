#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:00:19 2021

@author: alexloubser
"""
# =============================================================================
# Imports of modules
# =============================================================================
import torch
import torch.nn as nn

from Utils.Acousticmod import SpeechRecognition
from Utils.DataProcess import Data, collate_fn_padd

from torch.utils.tensorboard import SummaryWriter
# import logging
# import sys

# =============================================================================
# Paths for needed data files
# =============================================================================

writer = SummaryWriter("runs/runenv2")
# logger = logging.getLogger(__name__)

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# train and valid
# train_path = "/media/alexloubser/DataSSD/Masters/ComboV8/GenerateData/Train2022.json"
# valid_path = "/media/alexloubser/DataSSD/Masters/ComboV8/GenerateData/Val2022.json"

# train_path = "/media/alexloubser/DataSSD/LibriSpeech/TrainComb.json"
# valid_path = "/media/alexloubser/DataSSD/LibriSpeech/DevOther.json"

# data_train_path = "/media/alexloubser/DataSSD/Masters/v8data/datatrain/"
# data_val_path = "/media/alexloubser/DataSSD/Masters/v8data/dataval/"

# 修改为本地路径
train_path = "data/train.json"
valid_path = "data/valid.json"

data_train_path = "data/train/"
data_val_path = "data/valid/"

# =============================================================================
# Previous Model To Continue From
# =============================================================================
#model
# read_mod = "/media/alexloubser/DataSSD/Masters/NewModLibri/models/modelMelN100.pth"
read_mod = "models/initial_model.pth"


# =============================================================================
# General Parameters and HyperParameters
# =============================================================================
# general
epochfrom = 0
epochs = 101
batch_size = 32
lr = 1e-3

#hparamsnew
n_mfcc = 16

num_classes = 30
n_feats = 32
dropout = 0.1
hidden_size = 1024
num_layers = 2
    
#dparamsnew
# sample_rate = 16000
# specaug_rate = 0.5
# specaug_policy = 3
# time_mask = 70 
# freq_mask = 15
data_work = 8




# =============================================================================
# Logging
# =============================================================================
# # Setup logging
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )
# logger.setLevel(logging.INFO)

# # Log on each process the small summary:
# logger.warning(
#     f"device: {device}"
# )
# # Set the verbosity to info of the Transformers logger (on main process only):

# logger.setLevel(logging.INFO)

# =============================================================================
# Define Model
# =============================================================================
          
model = SpeechRecognition(hidden_size, num_classes, n_feats, num_layers, dropout).to(device)

# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(pytorch_total_params)

print("loading model from", read_mod)
checkpoint = torch.load(read_mod, map_location=torch.device('cpu'))
h_params = SpeechRecognition.hyper_parameters
model = SpeechRecognition(**h_params).to(device)

model_state_dict = checkpoint['model_state']
# new_state_dict = OrderedDict()
# for k, v in model_state_dict.items():
#     name = k.replace("model.", "") # remove `model.`
#     new_state_dict[name] = v

model.load_state_dict(model_state_dict)

epochfrom = checkpoint['epoch']+1
epochs = epochs + epochfrom

optim_state = checkpoint['optim_state']

#loss and optimizer
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                        optimizer, mode='min',
                                        factor=0.50, patience=6)

optimizer.load_state_dict(optim_state)


# =============================================================================
# Training Data
# =============================================================================

#training loop
d_params = Data.parameters
train_data = Data(json_path=train_path, data_path = data_train_path, **d_params)
val_data = Data(json_path=valid_path, data_path = data_val_path, **d_params, valid=True)


for epoch in range(epochfrom, epochs):
    # d_params = Data.parameters
    # dataset = Data(json_path=train_path, data_path = data_train_path, **d_params)
    # spectrograms, labels, spec_len, label_len = train_dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                        batch_size=batch_size,
                        num_workers=data_work,
                        shuffle=True,
                        pin_memory=True,
                        collate_fn=collate_fn_padd)
    
    totlen = len(train_loader)
    totloss = 0
    for i,(spectrograms, labels, spec_len, label_len) in enumerate(train_loader):
              
        # print(i)
        hid_size = spectrograms.size(0)
                
        # print(spectrograms.size())
        # print(labels.size())       
        
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        # forward
        # hidden = model._init_hidden(hid_size)
        # outor = model._init_out(hid_size)
        
        if epoch == 0 and i == 0:
            writer.add_graph(model, spectrograms)
        # print(n)
        # print(hidden.size())
        # hidden = hidden.reshape(-1,1,batch_size,1024).to(device)
        # outputs, hidden = model(spectrograms, hidden)
        
        outputs = model(spectrograms)
        
        loss = criterion(outputs, labels, spec_len, label_len)
        writer.add_scalar("Step Train Loss", abs(loss.item()), epoch)
        totloss += abs(loss.item())
        #backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(valloss.item())
        
        if (i)%500 ==0:
            print(f'epoch {epoch+1} / {epochs}, step {i} / {totlen}, loss = {loss.item():.4f}')
               
        
    totloss = totloss/(i+1)        
    writer.add_scalar("Average Train Epoch Loss", totloss, epoch)
    if epoch % 5 ==0:
        checkpoint = {"epoch":epoch, "model_state":model.state_dict(), "optim_state":optimizer.state_dict()}
        torch.save(checkpoint,"models/modelMelN"+str(epoch)+".pth")

    # dataset = None
    # del dataset
    train_loader = None
    del train_loader
    
    del spectrograms
    del labels
    del loss
    torch.cuda.empty_cache()
       
    # dataset = Data(json_path=valid_path, data_path = data_test_path, **d_params, valid=True)
    # spectrograms, labels, spec_len, label_len = train_dataset
    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                        batch_size=batch_size,
                        num_workers=data_work,
                        pin_memory=True,
                        collate_fn=collate_fn_padd)

# =============================================================================
# Validating Data        
# =============================================================================

    with torch.no_grad():
        lossval = 0
        n_samples = 0
        for i,(spectrograms, labels, spec_len, label_len) in enumerate(val_loader):
            # print(i)
            # hid_size = spectrograms.size(0)
            
            # tester = spectrograms.numpy()[0][0]
            # plt.plot(tester)

            spectrograms = spectrograms.to(device)
            
            labels = labels.to(device)
            
            # hidden = model._init_hidden(hid_size)

            outputs = model(spectrograms)
            loss = criterion(outputs, labels, spec_len, label_len)
            writer.add_scalar("Step Val Loss", abs(loss.item()), epoch)
            # print(avg_loss)
            
            
            #value, index
            # _, predictions = torch.max(outputs,2)
            
            # scheduler.step(avg_loss)
            # tensorboard_logs = {'val_loss': avg_loss}
            lossval += abs(loss.item())

            n_samples += 1            
          
    acc = lossval / (i+1)
    scheduler.step(acc)  
    print(f'loss = {acc}') 
    writer.add_scalar("Average Valid Epoch Loss", acc, epoch)
    
    fsave = open("DataMFCCbigdata2l2h.txt","a")
    # \n is placed to indicate EOL (End of Line)
    fsave.write(str(epoch) + " Loss " + str(totloss) + " ValLoss " + str(acc) + "\n")
    fsave.close() 
    
    # dataset = None
    # del dataset
    val_loader = None
    del val_loader
    del spectrograms
    del labels
    del loss
    torch.cuda.empty_cache()
    
writer.flush()
writer.close()




