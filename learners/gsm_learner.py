from argparse import Namespace
import os
import logging
import copy
import numpy as np

import torch
import torch.nn as nn

from learners.learner import Learner
from utils.gsm import gsm_task_train
from utils.utils import load_memory_data, save_memory_data, get_dataloader, get_device
from utils.train_eval import task_val, task_test_with_given_expert

class GSMLearner(Learner): 
    def __init__(self, model, scenarios, args: Namespace) -> None:
        super(GSMLearner,self).__init__(model, scenarios, args)
        
        self.mem_size = args.mem_size
        self.pre_mem_data = None
        self.cur_mem_data = None
        # create a path for storing mem_data
        self.mem_path = self.root_path+'/memory/mem_{}'.format(args.mem_size)
        if not os.path.exists(self.mem_path):
            os.makedirs(self.mem_path)
        
    def before_task_learning(self, tid):
        if tid > 0:
            # load memory data
            self.pre_mem_data = load_memory_data(tid, self.mem_path, self.args)
            # load existing model state dict
            self.load_previous_knowledge(tid)
        
    def task_learning(self, tid, train_task, val_task):
        # store memory of current task 
        save_memory_data(tid, train_task, self.mem_path, self.args)
        # learn current task
        self.gsm_task_learning(tid, train_task, val_task)
    
    def after_task_learning(self, tid):
        # load best model
        self.load_best_model()
        self.model.eval()
        
        # test all previous task
        res = dict()
        for id, task in enumerate(self.scenarios.test_stream):
            id += self.args.test_start_task
            if id > tid:
                break
            eid = 0
            ade, fde = task_test_with_given_expert(self.model, id, expert_id=eid, test_task=task,
                                                   args=self.args, save_dir=self.eval_path)
            logging.info("[Test] columns num:{}, task_{}, expert_{}, ADE:{:.2f}, fde:{:.2f}".format(
                len(self.model.columns), id, eid, ade.mean(), fde.mean()))
            res[(eid,id)] = [ade.mean(), fde.mean()]
    
    def gsm_task_learning(self, tid, train_task, val_task):
        self.model = self.model.to(get_device())
        # get dataloaders
        train_loader = get_dataloader(train_task,shuffle=True)
        val_loader = get_dataloader(val_task, shuffle=True)
        ''' load memory data '''
        if self.pre_mem_data is not None:
            mem_loader = get_dataloader(self.pre_mem_data, shuffle=False)
        else:
            mem_loader = None
        
        # initialize the metrics container
        self.metrics['train_loss'] = []
        self.metrics['val_loss'] = []
        self.constant_metrics['min_val_epoch'] = -1
        self.constant_metrics['min_val_loss']  = 9999
        
        # learning process
        optimizer = self.optim_obj(self.model.parameters(), lr= self.args.learning_rate, weight_decay= 0)
        if self.args.use_lrschd:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.2)
        
        ''' calculate each params' elements number ''' 
        grad_elements_num = []
        cnt = 1
        for name, param in self.model.named_parameters():
            if cnt == 23 or cnt == 24 or cnt == 31:
                continue
            grad_elements_num.append(param.data.numel())
            cnt+=1
        # logging.info("grad_elements_num is {}".format(grad_elements_num))

        for ep_id in range(self.args.num_epochs):
            # training
            self.model, train_loss = gsm_task_train(self.model, optimizer, train_loader, mem_loader, 
                                                    tid, grad_elements_num, self.args)
            self.metrics['train_loss'].append(train_loss)
            # validating
            val_loss = task_val(self.model, val_loader, self.args)
            self.metrics['val_loss'].append(val_loss)
                    
            # store best model and extra information
            if torch.isnan(torch.tensor(train_loss)).any():
                logging.warning("[Loss] Train loss contains NaN values.")
            
            if torch.isnan(torch.tensor(val_loss)).any():
                logging.warning("[Loss] Val loss contains NaN values.")
            
            if val_loss < self.constant_metrics['min_val_loss']:
                logging.info("[Loss] Epoch {}, train_loss: {:.4f}, val_loss: {:.4f}".format(ep_id,train_loss,val_loss))
                self.constant_metrics['min_val_loss'] = val_loss
                self.constant_metrics['min_val_epoch'] = ep_id
                self.extra_info = self.get_extra_info(tid)
                self.store_checkpoint(self.model,self.extra_info)
                self.best_model = copy.deepcopy(self.model)
                # early stop if val loss not decreased for long
                
            if ep_id - self.constant_metrics['min_val_epoch'] > self.args.early_stop_epochs:
                break

            if self.args.use_lrschd:
                scheduler.step()
                                
        # save loss
        if self.args.store_loss:
            np_val_loss = np.array(self.metrics['val_loss'])
            np_tra_loss = np.array(self.metrics['train_loss'])
            np.save(self.root_path+"/loss/val_task_{}.npy".format(tid),np_val_loss)
            np.save(self.root_path+"/loss/train_task_{}.npy".format(tid),np_tra_loss)
            with open(self.root_path+"/loss/constant_metrics.txt","a") as file:
                file.write(f"[Task-{tid}] Best model trained in epoch {self.constant_metrics['min_val_epoch']}, loss is {self.constant_metrics['min_val_loss']}\n")    
           