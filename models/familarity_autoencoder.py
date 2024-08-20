import copy
import argparse
import numpy as np
import os, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent))

import torch.nn as nn
import torch

from models.st_gcn import st_gcn
from utils.utils import get_device

class GraphAutoEncoder(nn.Module):
    def __init__(self, in_channels=2, latent_dim=64, kernel_size=3,seq_len=20):
        super(GraphAutoEncoder, self).__init__()
        middle_dim = int(latent_dim/2)
        self.encoder_1 = st_gcn(in_channels,middle_dim,(kernel_size,seq_len))
        self.encoder_2 = st_gcn(middle_dim,latent_dim,(kernel_size,seq_len))
        self.decoder_1 = st_gcn(latent_dim,middle_dim,(kernel_size,seq_len))
        self.decoder_2 = st_gcn(middle_dim,in_channels,(kernel_size,seq_len))
        
    def forward(self, v,a):
        v,a = self.encoder_1(v,a)
        ev,ea = self.encoder_2(v,a)
        v,a = self.decoder_1(ev,ea)
        dv,da = self.decoder_2(v,a)
        return ev,dv
    
    def save_model(self, save_path):
        state_dict = copy.deepcopy(self.state_dict())
        bn_mean = []
        bn_var = []
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_mean.append(module.running_mean)
                bn_var.append(module.running_var)
        saved_para = dict(state_dict=state_dict, bn_mean=bn_mean, bn_var=bn_var)
        torch.save(saved_para, save_path)
    
    def load_model(self, load_path):
        saved_para = torch.load(load_path, map_location=get_device())
        state_dict = saved_para['state_dict']
        bn_mean = saved_para['bn_mean']
        bn_var = saved_para['bn_var']
        self.load_state_dict(state_dict)
        n = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean = bn_mean[n].to(get_device())
                module.running_var = bn_var[n].to(get_device())
                n += 1
        
def train(model, optimizer, train_loader, batch_size=64):
    
    # initialization
    epoch_loss = 0
    batch_num = 0
    is_1st_loss = True
    loss_func = nn.MSELoss()
    model.train()

    # start training
    for step, case in enumerate(train_loader):        
        optimizer.zero_grad()
        case = [tensor.to(get_device()) for tensor in case]
        _, _, _, _, _, _,V_obs,A_obs,V_tr,A_tr = case
        V_obs_tmp = V_obs.permute(0,3,1,2)
        ev,dv = model(V_obs_tmp,A_obs.squeeze())
        loss = loss_func(dv,V_obs_tmp)

        if (step+1) % batch_size != 0:
            if is_1st_loss:
                 batch_loss = loss
                 is_1st_loss = False
            else:
                batch_loss += loss
        else:
            batch_loss += loss
            batch_loss = batch_loss / batch_size
            is_1st_loss = True
            batch_loss.backward()
            batch_num += 1
            
            optimizer.step()
            epoch_loss += batch_loss.item()
            
    return model, epoch_loss/ (step+1)

def valid(model, val_loader, batch_size=64):
    # initialization
    epoch_loss = 0
    batch_num = 0
    is_1st_loss = True
    loss_func = nn.MSELoss()
    model.eval()

    with torch.no_grad():
        for step, case in enumerate(val_loader):        
            case = [tensor.to(get_device()) for tensor in case]
            _, _, _, _, _, _,V_obs,A_obs,V_tr,A_tr = case
            V_obs_tmp = V_obs.permute(0,3,1,2)
            ev,dv = model(V_obs_tmp,A_obs.squeeze())

            loss = loss_func(dv,V_obs_tmp)

            if (step+1) % batch_size != 0:
                if is_1st_loss:
                    batch_loss = loss
                    is_1st_loss = False
                else:
                    batch_loss += loss
            else:
                batch_loss += loss
                batch_loss = batch_loss / batch_size
                is_1st_loss = True
                batch_num += 1

                epoch_loss += batch_loss.item()
            
    return epoch_loss/ (step+1)