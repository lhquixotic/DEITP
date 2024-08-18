import numpy as np
import quadprog

import torch

from utils.utils import get_device
from utils.train_eval import get_graph_loss


def gsm_task_train(model, optimizer, train_loader, mem_loader, 
                   cur_tid, grad_elem_num, args):
    is_1st_loss_ = True
    is_1st_loss = True
    loss = 0
    loss_ = 0
    loss_batch = 0
    model.train()
    
    for count, case in enumerate(train_loader):
        '''For previous task'''
        if mem_loader is not None and count % args.batch_size == 0:
            ave_mem = int(args.mem_size / cur_tid)
            grad_mat = torch.zeros((sum(grad_elem_num),cur_tid+1)).cuda()
            grad_mat.data.fill_(0.0)

            for count_, case_ in enumerate(mem_loader):

                case_ = [tensor.to(get_device()) for tensor in case_]
                l_ = get_graph_loss(model, case_)
                
                if is_1st_loss_:
                    loss_ = l_
                    is_1st_loss_ = False
                else:
                    loss_ += l_
                
                if (count_+1) % ave_mem == 0:
                    pre_task_id = int((count_+1)/ave_mem) - 1 
                    loss_ = loss_ / ave_mem
                    is_1st_loss_ = True
                    loss_.backward()

                    if args.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    
                    j = 0
                    para_cnt = 1
                    for params in model.parameters():
                        if para_cnt == 23 or para_cnt == 24 or para_cnt == 31:
                            continue
                        para_cnt += 1
                        if params is not None:
                            if j==0:
                                stpt = 0
                            else:
                                stpt = sum(grad_elem_num[:j])
                            endpt = sum(grad_elem_num[:j+1])

                            grad_mat[stpt:endpt, pre_task_id].data.copy_(
                                params.grad.data.view(-1))
                            j+=1
        
        '''For current task'''
        case = [tensor.to(get_device()) for tensor in case]
        optimizer.zero_grad()
        l = get_graph_loss(model, case)
        if is_1st_loss:
            loss = l
            is_1st_loss = False
        else:
            loss += l
        
        if (count+1) % args.batch_size == 0: # get a minibatch data-
            loss = loss/args.batch_size
            is_1st_loss = True
            loss.backward()
            
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            if mem_loader is not None:
                j = 0
                para_cnt = 1
                for params in model.parameters():
                    if para_cnt == 23 or para_cnt == 24 or para_cnt == 31:
                        continue
                    para_cnt += 1
                    if params is not None:
                        stpt = 0 if j==0 else sum(grad_elem_num[:j])
                        endpt = sum(grad_elem_num[:j+1])
                        grad_mat[stpt:endpt,cur_tid].data.copy_(params.grad.view(-1))
                        j += 1
                
                dot_prod = torch.mm(grad_mat[:,cur_tid].unsqueeze(0),grad_mat[:,:cur_tid+1])
                vector_1 = grad_mat[:,cur_tid].unsqueeze(0).cpu().numpy()
                vector_2 = grad_mat[:,:cur_tid+1].cpu().numpy()
                len1 = np.linalg.norm(vector_1)
                len2 = np.linalg.norm(vector_2)
                cosine_angle = np.dot(vector_1,vector_2)/(len1*len2)

                if (dot_prod < 0).sum() > 0:
                    grad_mat_ = grad_mat
                    mem_grad_np = grad_mat_[:,:cur_tid+1].cpu().t().double().numpy()
                    cur_task_grad_np = grad_mat_[:,cur_tid].unsqueeze(1).cpu().contiguous().view(-1).double().numpy()

                    t = mem_grad_np.shape[0]
                    P = np.dot(mem_grad_np, mem_grad_np.transpose())
                    P = 0.5 * (P + P.transpose()) + np.eye(t) * args.eps
                    q = -np.dot(mem_grad_np, cur_task_grad_np)
                    G_ = np.eye(t)
                    h = np.zeros(t) + args.margin
                    v = quadprog.solve_qp(P, q, G_, h)[0]
                    x = np.dot(v, mem_grad_np) + cur_task_grad_np
                    new_grad = torch.Tensor(x).view(-1)

                    # Copy gradients into params
                    j = 0
                    ct_params = 1
                    for params in model.parameters():
                        if ct_params == 23 or ct_params == 24 or ct_params == 31:
                            continue
                        ct_params += 1

                        if params is not None:
                            stpt = 0 if j==0 else sum(grad_elem_num[:j])
                            endpt = sum(grad_elem_num[:j+1])
                            params.grad.data.copy_(
                                new_grad[stpt:endpt].contiguous().view(params.grad.data.size()))
                            j += 1
            
            optimizer.step()
            
            loss_batch += loss.item()
    
    train_loss = loss_batch / (int((count+1)/args.batch_size))
    return model, train_loss
        