import copy

import torch
import torch.nn as nn

from models.social_stgcnn import social_stgcnn

class social_stgcnn_pnn(social_stgcnn):
    def __init__(self, n_stgcnn=1, n_txpcnn=1, input_feat=2, output_feat=5,
                 seq_len=8, pred_seq_len=12, kernel_size=3):
        super(social_stgcnn_pnn,self).__init__(n_stgcnn, n_txpcnn, input_feat, output_feat,
                         seq_len, pred_seq_len, kernel_size)
        
        self.batch_norm_para = dict()
        
    def freeze_columns(self, skip=None):
        if skip == None:
            skip = []
        for i, c in enumerate(self.columns):
            if i not in skip:
                for name, params in c.named_parameters():
                    params.requires_grad = False
                    
    def get_prev_lateral_connections(self,column_id, layer_id):
        prev_lateral_connections = dict()
        if column_id <= 1:
            return None
        else:
            for i in range(1, column_id):
                prev_lateral_connections[i] = self.columns[i][layer_id].lateral_connection
        return prev_lateral_connections
    
    def load_batch_norm_para(self, task_id):
        expert_id = self.select_expert(task_id)
        n = 0 # number of loaded params
        for i in range(expert_id+1):
            for k, module in enumerate(self.columns[i].modules()):
                if isinstance(module, nn.BatchNorm2d):      
                    module.running_mean = copy.deepcopy(self.batch_norm_para[task_id][0][n])
                    module.running_var = copy.deepcopy(self.batch_norm_para[task_id][1][n])
                    n+=1
                    
    def save_batch_norm_para(self, task_id):
        # save the batch norm para of all columns respect to task_id
        bn_mean = []
        bn_var = []
        for i in range(len(self.columns)):    # for all column 
            for module in self.columns[i].modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn_mean.append(module.running_mean)
                    bn_var.append(module.running_var)
        self.batch_norm_para[task_id] = copy.deepcopy([bn_mean,bn_var])
    
    def select_expert(self, task_id):
        return task_id
    
    def init_new_expert(self, trained_expert=None):
        # initialize the new expert with previous expert
        assert len(self.columns) > 0
        if len(self.columns) == 1:
            if trained_expert is None:
                pass
            else:            
                # load trained model for the first expert
                pass
        else:
            # load -2 expert or trained expert for -1 expert
            if trained_expert is None:
                trained_expert = self.columns[-2]
            
            target_expert_id = len(self.columns) - 1 
            target_state_dict = copy.deepcopy(self.columns[-1].state_dict())
            
            for key, value in trained_expert.state_dict().items():
                if "lateral" in key or "running" in key:
                    continue
                else:
                    assert key in target_state_dict.keys()
                    target_state_dict[key] = value
            
            self.columns[-1].load_state_dict(target_state_dict)
    
    