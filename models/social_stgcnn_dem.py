import numpy as np
import torch
import torch.nn as nn

from models.social_stgcnn_pnn import social_stgcnn_pnn
from models.familarity_autoencoder import GraphAutoEncoder
from utils.utils import get_device

class social_stgcnn_dem(social_stgcnn_pnn):
    def __init__(self, n_stgcnn=1, n_txpcnn=1, input_feat=2, output_feat=5, seq_len=8,
                 pred_seq_len=12, kernel_size=3):
        super().__init__(n_stgcnn, n_txpcnn, input_feat, output_feat, seq_len,
                         pred_seq_len, kernel_size)
        
        self.expert_selector = dict()
        self.task_faes = []
        self.use_task_detector = False
    
    # basic
    def forward(self, v, a, column_id=-1):
        if self.use_task_detector and not self.training: # if use detector in training
            # task_id = self.detect_task(v,a)
            # column_id = self.select_expert(task_id)
            column_id = self.detect_expert(v,a)
            
        return super().forward(v, a, column_id)
    
    # structure expansion related
    def add_column(self):
        super().add_column()
        expert_id = len(self.columns) - 1
        self.expert_selector[expert_id] = []
        if self.use_task_detector:
            self.task_faes.append(GraphAutoEncoder(self.input_feat,self.task_fae_latent_dim).to(get_device()))
    
    # task detecting related    
    def detect_task(self,v,a):
        assert not len(self.task_faes) == 0
        recon_loss_list = []
        loss_func = nn.MSELoss()
        # record the reconstruction loss of every fae
        for fae in self.task_faes:
            fae.eval()
            _, recon = fae(v,a)
            recon_loss = loss_func(recon,v)
            recon_loss_list.append(recon_loss.item())        
        task_id = recon_loss_list.index(min(recon_loss_list))
        return task_id
    
    def detect_expert(self, v, a):
        assert not len(self.task_faes) == 0
        all_tasks = []
        for eid, tids in self.expert_selector.items():
            all_tasks += tids
        assert len(all_tasks) == len(self.task_faes)
        case_recon_res = np.zeros(len(self.task_faes))
        loss_func = nn.MSELoss()
        # record the reconstruction loss of every fae
        for fid, fae in enumerate(self.task_faes):
            fae.eval()
            _, recon = fae(v,a)
            recon_loss = loss_func(recon,v)
            case_recon_res[fid] = recon_loss.item()
        # select the expert with the minimum reconstruction loss
        # print("Recon_res: {}".format(case_recon_res))
        task_id = np.argmin(case_recon_res)
        expert_id = self.select_expert(task_id)
        # print("[Task detector] Selected expert id: {} for task id: {}.".format(expert_id, task_id))
        return expert_id

    def set_task_detector(self, use_task_detector=True, latent_dim=32):
        # initialize the task_faes
        self.use_task_detector = use_task_detector
        print("[Task detector] Use task detector set as {}.".format(use_task_detector))
        self.task_fae_latent_dim = latent_dim
    
    # expert selection related
    def select_expert(self, task_id):
        expert_id = len(self.columns) - 1
        # task id is recorded
        seen_tasks = []
        for key, value in self.expert_selector.items():
            for v in value:
                    seen_tasks.append(v)
            if task_id in value:
                expert_id = key
                return expert_id
        # task id is not recorded
        if task_id == len(seen_tasks):
            return len(self.columns) - 1
        return None