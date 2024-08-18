from argparse import Namespace
import numpy as np
import os

import torch.distributions.multivariate_normal as torchdist
import torch
from torch.utils.data import DataLoader

from utils.metrics import *
from utils.utils import set_seeds

def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def eval_case(case, model,task_id=-1):
    # Set model as eval mode
    model.eval()
    
    # Get case data
    case = [tensor.to(get_device()) for tensor in case]    
    obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
        loss_mask, V_obs, A_obs, V_tr, A_tr = case
    
    num_of_objs = obs_traj_rel.shape[1]
    
    ade_bigls = []
    fde_bigls = []
    
    sample_num = 20
    
    # Use case data to evaluate
    with torch.no_grad():
        V_obs_tmp = V_obs.permute(0,3,1,2)
        V_pred = model(V_obs_tmp, A_obs.squeeze(),task_id)
        # activations = model.get_activation_values(V_obs_tmp, A_obs.squeeze())
        V_pred = V_pred.permute(0,2,3,1)
        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        V_pred,V_tr =  V_pred[:,:num_of_objs,:],V_tr[:,:num_of_objs,:]
        
        # MDN model
        sx = torch.exp(V_pred[:,:,2]) #sx
        sy = torch.exp(V_pred[:,:,3]) #sy
        corr = torch.tanh(V_pred[:,:,4]) #corr

        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).to(get_device())
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]
        
        mvnormal = torchdist.MultivariateNormal(mean,cov)
        
        # Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                 V_x[0,:,:].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                 V_x[-1,:,:].copy())
        
        for n in range(num_of_objs):
            ade_ls[n]=[]
            fde_ls[n]=[]

        for k in range(sample_num):

            V_pred = mvnormal.sample()

            #V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                     V_x[-1,:,:].copy())
            # raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))
            
           # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = [] 
                target = []
                obsrvs = [] 
                number_of = []
                pred.append(V_pred_rel_to_abs[:,n:n+1,:])
                target.append(V_y_rel_to_abs[:,n:n+1,:])
                obsrvs.append(V_x_rel_to_abs[:,n:n+1,:])
                number_of.append(1)

                ade_ls[n].append(ade(pred,target,number_of))
                fde_ls[n].append(fde(pred,target,number_of))
        
        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))
        
    return sum(ade_bigls)/len(ade_bigls), sum(fde_bigls)/len(fde_bigls)

def eval_case_expert_dist(case, model, eid_1, eid_2):
    # Set model as eval mode
    model.eval()
    
    # Get case data
    case = [tensor.to(get_device()) for tensor in case]    
    obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
        loss_mask, V_obs, A_obs, V_tr, A_tr = case
    
    num_of_objs = obs_traj_rel.shape[1]
    
    ade_bigls = []
    fde_bigls = []
    
    sample_num = 20
    
    # Use case data to evaluate
    with torch.no_grad():
        V_obs_tmp = V_obs.permute(0,3,1,2)
        V_pred1 = model(V_obs_tmp, A_obs.squeeze(),eid_1)
        V_pred1 = V_pred1.permute(0,2,3,1)
        V_pred2 = model(V_obs_tmp, A_obs.squeeze(),eid_2)
        V_pred2 = V_pred2.permute(0,2,3,1)
        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred1 = V_pred1.squeeze()
        V_pred2 = V_pred2.squeeze()
        V_pred1 = V_pred1[:,:num_of_objs,:]
        V_pred2 = V_pred2[:,:num_of_objs,:]
        V_tr = V_tr[:,:num_of_objs,:]
        
        # MDN model
        sx1 = torch.exp(V_pred1[:,:,2]) #sx
        sy1 = torch.exp(V_pred1[:,:,3]) #sy
        corr1 = torch.tanh(V_pred1[:,:,4]) #corr
        cov1 = torch.zeros(V_pred1.shape[0],V_pred1.shape[1],2,2).to(get_device())
        cov1[:,:,0,0]= sx1*sx1
        cov1[:,:,0,1]= corr1*sx1*sy1
        cov1[:,:,1,0]= corr1*sx1*sy1
        cov1[:,:,1,1]= sy1*sy1
        mean1 = V_pred1[:,:,0:2]
        mvnormal1 = torchdist.MultivariateNormal(mean1,cov1)
        
        sx2 = torch.exp(V_pred2[:,:,2]) #sx
        sy2 = torch.exp(V_pred2[:,:,3]) #sy
        corr2 = torch.tanh(V_pred2[:,:,4]) #corr
        cov2 = torch.zeros(V_pred2.shape[0],V_pred2.shape[1],2,2).to(get_device())
        cov2[:,:,0,0]= sx2*sx2
        cov2[:,:,0,1]= corr2*sx2*sy2
        cov2[:,:,1,0]= corr2*sx2*sy2
        cov2[:,:,1,1]= sy2*sy2
        mean2 = V_pred2[:,:,0:2]
        mvnormal2 = torchdist.MultivariateNormal(mean2,cov2)
        
        # Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                 V_x[0,:,:].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                 V_x[-1,:,:].copy())
        
        for n in range(num_of_objs):
            ade_ls[n]=[]
            fde_ls[n]=[]

        for k in range(sample_num):

            V_pred1 = mvnormal1.sample()
            V_pred2 = mvnormal2.sample()

            V_pred1_rel_to_abs = nodes_rel_to_nodes_abs(V_pred1.data.cpu().numpy().squeeze().copy(),V_x[-1,:,:].copy())
            V_pred2_rel_to_abs = nodes_rel_to_nodes_abs(V_pred2.data.cpu().numpy().squeeze().copy(),V_x[-1,:,:].copy())
            
            for n in range(num_of_objs):
                pred1 = [] 
                pred2 = []
                target = []
                obsrvs = [] 
                number_of = []
                pred1.append(V_pred1_rel_to_abs[:,n:n+1,:])
                pred2.append(V_pred2_rel_to_abs[:,n:n+1,:])
                target.append(V_y_rel_to_abs[:,n:n+1,:])
                obsrvs.append(V_x_rel_to_abs[:,n:n+1,:])
                number_of.append(1)

                ade_ls[n].append(ade(pred1,pred2,number_of))
                fde_ls[n].append(fde(pred1,pred2,number_of))
        
        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))
        
    return sum(ade_bigls)/len(ade_bigls), sum(fde_bigls)/len(fde_bigls)

def task_val(model,val_loader, args):
    model.eval()
    is_1st_loss = True
    loss_batch = 0
    
    for case_id, case in enumerate(val_loader):    
        # Get data from the loader
        case = [tensor.to(get_device()) for tensor in case]
        
        l = get_graph_loss(model,case)
        if is_1st_loss:
            loss = l
            is_1st_loss = False
        else:
            loss += l
        
        if (case_id+1) % args.batch_size == 0:
            loss = loss / args.batch_size 
            is_1st_loss = True
            loss_batch += loss.item()

    return loss_batch / (int((case_id+1)/args.batch_size))

def task_train(model, optimizer, train_loader: DataLoader, args: Namespace):
    is_1st_loss = True
    loss_batch = 0
    model.train()
    
    for count, case in enumerate(train_loader):
        case = [tensor.to(get_device()) for tensor in case]
        optimizer.zero_grad()
        
        l = get_graph_loss(model,case)
        if is_1st_loss:
            graph_loss = l
            is_1st_loss = False
        else:
            graph_loss += l
        
        if (count+1) % args.batch_size == 0:
            graph_loss = graph_loss / args.batch_size
            is_1st_loss = True
            graph_loss.backward()
            
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            optimizer.step()
            loss_batch += graph_loss.item()
            

    train_loss = loss_batch / (int((count+1)/args.batch_size))

    return model, train_loss

def task_test(model, expert_id, test_loader, args):
    test_times = args.test_times
    test_case_num = args.test_case_num
    eval_ade = np.zeros((test_times,test_case_num))
    eval_fde = np.zeros((test_times,test_case_num))
    for i in range(test_times):
        if args.deterministic:
            set_seeds(args.seed+i)
        dataloader = test_loader
        for idx, case in enumerate(dataloader):
            if idx >= test_case_num:
                break
            else:
                case_ade,case_fde = eval_case(case,model,expert_id)
                eval_ade[i,idx] = case_ade
                eval_fde[i,idx] = case_fde
        del dataloader
    if args.deterministic:
        set_seeds(args.seed)
    return eval_ade,eval_fde

def task_test_expert_dist(model, eid_1, eid_2, test_loader, args):
    test_times = args.test_times
    test_case_num = args.test_case_num
    eval_ade = np.zeros((test_times,test_case_num))
    eval_fde = np.zeros((test_times,test_case_num))
    for i in range(test_times):
        if args.deterministic:
            set_seeds(args.seed+i)
        dataloader = test_loader
        for idx, case in enumerate(dataloader):
            if idx >= test_case_num:
                break
            else:
                case_ade,case_fde = eval_case(case,model,eid_1,eid_2)
                eval_ade[i,idx] = case_ade
                eval_fde[i,idx] = case_fde
        del dataloader
    if args.deterministic:
        set_seeds(args.seed)
    return eval_ade,eval_fde

def task_test_with_given_expert(model, task_id, expert_id, test_task, args, save_dir=None):
    assert expert_id < len(model.columns) # expert id should < column number
    test_loader = DataLoader(test_task.dataset, batch_size=1, shuffle=False,
                             num_workers=16,drop_last = True)
    ade, fde = task_test(model, expert_id, test_loader, args)
            
    # save if save_dir is provided
    if save_dir is not None:    
        os.makedirs(save_dir, exist_ok=True)
        np.save(save_dir+"/ADE-task-{}-exp-{}.npy".format(task_id,expert_id),ade)
        np.save(save_dir+"/FDE-task-{}-exp-{}.npy".format(task_id,expert_id),fde)
    return ade, fde
    
