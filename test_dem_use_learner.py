import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from torch.utils.data import DataLoader

from utils.utils import get_device,get_dataloader, set_seeds
from models.familarity_autoencoder import GraphAutoEncoder
from utils.train_eval import task_test_with_given_expert, eval_case
from utils.scenarios import get_continual_scenario_benchmark
from models.social_stgcnn_dem import social_stgcnn_dem
from learners.dem_learner import DEMLearner
from utils.utils import get_argument_parser
from models.familarity_autoencoder import GraphAutoEncoder

def main(args): 
    # set seeds
    scenarios = get_continual_scenario_benchmark(args)
    model = social_stgcnn_dem(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                        output_feat=args.output_size, seq_len=args.obs_seq_len,
                        kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len)
    for i in range(4-len(model.columns)):
        model.add_column()
    if args.task_free:
        model.set_task_detector(True)
        model.task_faes = []
        task_num = len(scenarios.test_stream)
        for i in range(task_num):
            fae = GraphAutoEncoder(latent_dim=32)
            state_dict = torch.load("./logs/task_detector/experiment/checkpoint/FAE_task_{}.pth".format(i), map_location=get_device())
            fae.load_state_dict(state_dict)
            model.task_faes.append(fae)

    learner = DEMLearner(model,scenarios,args)
    learner.learn_tasks()
        
if __name__ == '__main__':
    args = get_argument_parser()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.deterministic: set_seeds(args.seed)
    main(args)
    