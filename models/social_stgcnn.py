import copy

import torch
import torch.nn as nn

from models.st_gcn import st_gcn
from models.block import LateralBlock

class social_stgcnn(nn.Module):
    '''
    A implementation of Social-STGCNN that can be extended to multiple columns
    '''
    def __init__(self,n_stgcnn =1,n_txpcnn=5,input_feat=2,output_feat=5,
                 seq_len=20,pred_seq_len=40,kernel_size=3):
        super(social_stgcnn,self).__init__()
        # basic network parameters
        self.n_stgcnn= n_stgcnn
        self.n_txpcnn = n_txpcnn
        self.input_feat = input_feat
        self.output_feat = output_feat
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.pred_seq_len = pred_seq_len
        
        # muti-columns structure
        self.columns = nn.ModuleList()
        self.block_number = n_stgcnn + n_txpcnn + 1
        
    def add_column(self):
        column_id = len(self.columns)
        
        # define the blocks contained in the baseline model
        defined_blocks = []
        for k in range(self.block_number):
            # stgcnns
            if k < self.n_stgcnn:
                defined_blocks.append(st_gcn(self.input_feat,self.output_feat,
                                             (self.kernel_size,self.seq_len)))
            # tpcnns
            elif k == self.n_stgcnn:
                defined_blocks.append(nn.Sequential(
                    nn.Conv2d(self.seq_len,self.pred_seq_len,3,padding=1), 
                    nn.PReLU()))
            elif k < self.block_number - 1:
                defined_blocks.append(nn.Sequential(
                    nn.Conv2d(self.pred_seq_len,self.pred_seq_len,3,padding=1),
                    nn.PReLU()))
            else:
                defined_blocks.append(nn.Conv2d(
                    self.pred_seq_len,self.pred_seq_len,3,padding=1))
        
        # for a new task, add a new list contains lateral blocks of all layers
        new_blocks = [] 

        for k in range(self.block_number):
            new_blocks.append(LateralBlock(col_id=column_id,
                             depth=k,
                             block=defined_blocks[k]))
        
        new_column = nn.ModuleList(new_blocks)
        self.columns.append(new_column)
        
    def forward(self, v, a, column_id=-1):
        # check whether columns exist
        assert self.columns

        # calculate the first layer output as inputs
        inputs = []
        for column in self.columns:
            v_,a_ = column[0]([v,a],is_1st_layer=True) # column[0] -> st_gcns
            
            # reshape 
            v_ = v_.view(v_.shape[0],v_.shape[2],v_.shape[1],v_.shape[3])
            inputs.append(v_)

        # from the 2nd block to the last block
        for l in range(1,self.block_number): 
            out = []
            for col_id, col in enumerate(self.columns): 
                if l == 1: # 1st layer of tpcnn
                    layer_out = col[l](inputs=inputs[:col_id+1],
                                        prev_lateral_connections=self.get_prev_lateral_connections(col_id,l))
                    out.append(layer_out)
                elif l < self.block_number - 1: # tpcnn out before the output layer
                    layer_out = col[l](inputs=inputs[:col_id+1],
                                        prev_lateral_connections=self.get_prev_lateral_connections(col_id,l)) + inputs[col_id]
                    out.append(layer_out)
                elif l == self.block_number - 1:
                    layer_out = col[l](inputs=inputs[:col_id+1],
                                        prev_lateral_connections=self.get_prev_lateral_connections(col_id,l))
                    layer_out = layer_out.view(layer_out.shape[0],layer_out.shape[2],
                                                layer_out.shape[1],layer_out.shape[3])
                    out.append(layer_out)
            inputs = out
        return out[column_id]

    def get_prev_lateral_connections(self,column_id, layer_id):
        return None