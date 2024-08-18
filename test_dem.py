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

def main(args): 
    # set seeds
    scenarios = get_continual_scenario_benchmark(args)
    task_ids = [int(tid) for tid in args.task_seq.split('-')]
    task_ids = task_ids + [tid + 5*(args.datasets_split_num-1) for tid in task_ids] 
    task_num = len(task_ids)
    print("Test on task sequence: ", task_ids)
    expert_ids = [int(eid) for eid in args.expert_ids.split(',')] if args.expert_ids is not None else  task_ids
    print("Test using expert sequence: ", expert_ids)

    dem = social_stgcnn_dem(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                        output_feat=args.output_size, seq_len=args.obs_seq_len,
                        kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len)

    dem.set_task_detector(use_task_detector=False)
    ckpt_name = "./logs/DEM/experiment/checkpoint/checkpoint_task_1.pth"
    print("column num:",len(dem.columns))
    for i in range(2):
        dem.add_column()
    ckpt = torch.load(ckpt_name, map_location=get_device())
    dem.load_state_dict(ckpt['model_state_dict'])
    dem.batch_norm_para = ckpt['batch_norm_para']
    dem.expert_selector = ckpt['expert_selector']
    for tid, test_task in enumerate(scenarios.test_stream):
        if tid > 0: break
        eid = dem.select_expert(tid)
        print("Test on task {} using expert {}".format(tid, eid))
        ade, fde = task_test_with_given_expert(dem, tid, eid,
                    test_task=test_task, args=args)
        print("[Test] ADE: {:.2f}, FDE: {:.2f}".format(np.mean(ade), np.mean(fde)))
        
if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description="test autoencoder")
    parser.add_argument('--experiment_name', type=str, default='experiment')
    parser.add_argument('--is_demo', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', type=int, default=1)
    parser.add_argument('--vis_detect_res', type=int, default=0)
    parser.add_argument('--train_start_task', type=int, default=0)
    # scenario parameters
    parser.add_argument('--task_seq', type=str, default="1-2-3-4-5")
    parser.add_argument('--datasets_split_num', type=int, default=2)
    parser.add_argument('--test_start_task', type=int, default=0)
    # Model parameters
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
    parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--obs_seq_len', type=int, default=20)
    parser.add_argument('--pred_seq_len', type=int, default=40)
    parser.add_argument('--latent_dim', type=int, default=32)
    # test parameters
    parser.add_argument('--expert_ids', type=str, default=None)
    parser.add_argument('--test_times', type=int, default=1)
    parser.add_argument('--test_case_num', type=int, default=1000)
    parser.add_argument('--task_free', type=int, default=1)
    parser.add_argument('--task_detect', type=int, default=0)
    parser.add_argument('--task_predict', type=int, default=1)
    args = parser.parse_args()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.deterministic: set_seeds(args.seed)
    main(args)
    