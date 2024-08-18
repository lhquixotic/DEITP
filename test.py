import pickle
import copy
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.utils import get_device,get_dataloader, set_seeds
from models.familarity_autoencoder import GraphAutoEncoder
from utils.train_eval import task_test_with_given_expert, eval_case
from utils.scenarios import get_continual_scenario_benchmark
from models.social_stgcnn_dem import social_stgcnn_dem

def main(args):
    # Load data
    scenarios = get_continual_scenario_benchmark(args)
    task_ids = [int(tid) for tid in args.task_seq.split('-')]
    task_ids = task_ids + [tid + 5*(args.datasets_split_num-1) for tid in task_ids] 
    task_num = len(task_ids)
    print("Test on task sequence: ", task_ids)
    expert_ids = [int(eid) for eid in args.expert_ids.split(',')] if args.expert_ids is not None else  task_ids
    print("Test using expert sequence: ", expert_ids)

    # Detect tasks
    if args.task_free or args.task_detect:
        faes = load_faes(args, task_ids)
    else:
        faes = None
        print("No task detector is used.")

    if args.task_detect:
        detect_result = task_detect(args, faes, scenarios, device=get_device())

    # Predict tasks
    if args.task_predict:
        ckpt_name = "./logs/DEM/{}/checkpoint/checkpoint_task_{}.pth".format(args.experiment_name, task_ids[-1]-1)
        print("Loading trained DEM model from {}...".format(ckpt_name))
        dem = load_trained_dem(args, ckpt_name=ckpt_name)

        predict_result = task_predict(args, dem, scenarios, device=get_device(), task_free=args.task_free, faes=faes, detect_result=detect_result if args.task_detect else None)

def load_faes(args, task_ids):
    print("Loading Familiarity Autoencoder models for task ids: {}...".format(task_ids))
    faes = []
    for tid in task_ids:
        fae = GraphAutoEncoder(seq_len=args.obs_seq_len, latent_dim=args.latent_dim)
        loaded = torch.load("./logs/task_detector/{}/checkpoint/FAE_task_{}.pth".format(args.experiment_name, tid - 1),map_location=torch.device('cpu'))
        fae.load_state_dict(loaded)
        faes.append(fae)
        print('Loaded FAE for task {}.'.format(tid))
    return faes

def load_previous_knowledge(model, task_id, ckpt_name=None):
    root_path = "./logs/DEM/result"
    ckpt_fname = root_path+'/checkpoint/checkpoint_task_{}.pth'.format(task_id) if ckpt_name is None else ckpt_name

    print(f"Load model in {ckpt_fname}")
    assert os.path.exists(ckpt_fname)
    checkpoint = torch.load(ckpt_fname, map_location=torch.device('cpu'))
    
    # load expert selector
    assert "expert_selector" in checkpoint.keys()
    model.expert_selector = checkpoint['expert_selector']
    prev_expert_num = len(checkpoint['expert_selector'])
    
    # load model state dict only when expert number equals to column num        
    if prev_expert_num == len(model.columns):
        model.load_state_dict(checkpoint['model_state_dict'])
        assert "batch_norm_para" in checkpoint.keys()
        model.batch_norm_para = copy.deepcopy(checkpoint["batch_norm_para"])
        model.freeze_columns()
        model.load_batch_norm_para(task_id-1)
    return model

def load_best_model(model, task_id, ckpt_name=None):
    root_path = "./logs/DEM/result"
    ckpt_fname = root_path+'/checkpoint/checkpoint_task_{}.pth'.format(task_id) if ckpt_name is None else ckpt_name
    assert os.path.exists(ckpt_fname), "No checkpoint named {}".format(ckpt_fname)
    checkpoint = torch.load(ckpt_fname, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    # load batch_norm parameters
    assert "batch_norm_para" in checkpoint.keys()
    model.batch_norm_para = checkpoint["batch_norm_para"]
    # load expert_select dict
    assert "expert_selector" in checkpoint.keys()
    model.expert_selector = checkpoint["expert_selector"]
    print('expert_selector:', model.expert_selector)
    return model

def load_trained_dem(args, ckpt_name=None):
    dem = social_stgcnn_dem(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                        output_feat=args.output_size, seq_len=args.obs_seq_len,
                        kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len)
    if args.task_free:
        dem.set_task_detector(use_task_detector=True, latent_dim=args.latent_dim)
    else:
        dem.set_task_detector(use_task_detector=False)
    ckpt_name = "./logs/checkpoint_task_9.pth" if ckpt_name is None else ckpt_name
    dem = load_previous_knowledge(dem,9,ckpt_name=ckpt_name) 
    expand_times = len(dem.expert_selector) - len(dem.columns)
    for t in range(expand_times):
        dem.add_column()
    dem = load_best_model(dem, 9, ckpt_name=ckpt_name)
    return dem

def task_detect(args, models, scenarios, device):
    expert_dict = {0:[0,5],1:[1,6],2:[2,7],3:[3,4,8,9]}
    print("Start detecting tasks...")
    class_num = len(models)
    all_detect_result = []
    loss_func = nn.MSELoss()
    for tid, test_task in enumerate(scenarios.test_stream):
        detect_result_list = []
        test_loader = get_dataloader(test_task, False)
        for case_id, case in enumerate(test_loader):
            if case_id >= args.test_case_num: break
            case_res = np.zeros(class_num)
            case = [tensor.to(device) for tensor in case]
            _, _, _, _, _, _,V_obs,A_obs,V_tr,A_tr = case
            V_target = V_obs.permute(0,3,1,2)
            A_target = A_obs.squeeze()
            for mid, model in enumerate(models):
                _, recon = model(V_target, A_target)
                recon_loss = loss_func(recon, V_target)
                case_res[mid] = recon_loss.item()
            print("Recon res:", case_res)
            detect_result_list.append(np.argmin(case_res))
        # print(f"[Task {tid}] detect result: {np.bincount(detect_result_list, minlength=class_num)}")
        detect_res = np.bincount(detect_result_list, minlength=class_num)
        all_detect_result.append(detect_res)
        expert_detect_result = []
        for fid in range(len(list(expert_dict.keys()))):
            detect_num = 0
            for ttid in expert_dict[fid]:
                detect_num += detect_res[ttid]
            expert_detect_result.append(detect_num)
        print(f"[Task {tid}] expert detect result: {expert_detect_result}")
    return all_detect_result

def task_predict(args, model, scenarios, device, task_free=False, faes = None, detect_result=None):
    print("Start testing predicting tasks...")
    task_num = len(scenarios.test_stream)

    if task_free and faes is not None:
        model.task_faes = []
        for tid in range(task_num):
            model.task_faes.append(faes[tid])
        model.set_task_detector(use_task_detector=False)
    else:
        model.set_task_detector(use_task_detector=False)

    for tid, test_task in enumerate(scenarios.test_stream):
        print(f"Testing on task {tid}")
        if not task_free:
            eid = model.select_expert(tid)
            res_ade, res_fde = task_test_with_given_expert(model,tid,expert_id=eid,
                                                        test_task=test_task,args=args, )
        elif faes is not None:
            loss_func = nn.MSELoss()
            eval_ade = np.zeros((args.test_times, args.test_case_num))
            eval_fde = np.zeros((args.test_times, args.test_case_num))
            test_loader = DataLoader(test_task.dataset, batch_size=1, shuffle=False, num_workers=16,drop_last = True)
            for i in range(args.test_times):
                if args.deterministic: set_seeds(args.seed+i)
                for idx, case in enumerate(test_loader):
                    if idx >= args.test_case_num: break
                    case_res = np.zeros(len(faes))
                    case = [tensor.to(device) for tensor in case]
                    _, _, _, _, _, _,V_obs,A_obs,V_tr,A_tr = case
                    V_target = V_obs.permute(0,3,1,2)
                    A_target = A_obs.squeeze()
                    for fid, fae in enumerate(faes):
                        _, recon = fae(V_target, A_target)
                        recon_loss = loss_func(recon, V_target)
                        case_res[fid] = recon_loss.item()
                    detect_tid = np.argmin(case_res)
                    detect_eid = model.select_expert(detect_tid)
                    # print(f"Task {tid}, case {idx}, Detected as task {detect_tid},  expert {detect_eid}")
                    case_ade, case_fde = eval_case(case, model, detect_eid)
                    eval_ade[i, idx] = case_ade
                    eval_fde[i, idx] = case_fde
                
            res_ade = np.mean(eval_ade, axis=0)
            res_fde = np.mean(eval_fde, axis=0)
        else:
            eval_ade = np.zeros((args.test_times, args.test_case_num))
            eval_fde = np.zeros((args.test_times, args.test_case_num))
            test_loader = DataLoader(test_task.dataset, batch_size=1, shuffle=False, num_workers=16,drop_last = True)
            for i in range(args.test_times):
                if args.deterministic: set_seeds(args.seed+i)
                for idx, case in enumerate(test_loader):
                    if idx >= args.test_case_num: break
                    eid =  model.select_expert(detect_result[tid][idx])
                    print(f"Task {tid}, case {idx}, Detected as task {detect_result[tid][idx]},  expert {eid}")
                    case_ade, case_fde = eval_case(case, model, eid)
                    eval_ade[i, idx] = case_ade
                    eval_fde[i, idx] = case_fde
                
            res_ade = np.mean(eval_ade, axis=0)
            res_fde = np.mean(eval_fde, axis=0)

        # save_dir="./logs/DEM/result/evaluation")
        print("[Test] columns num:{}, task_{}, ADE:{:.2f}, fde:{:.2f}".format(
                len(model.columns), tid, res_ade.mean(), res_fde.mean()))

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
    