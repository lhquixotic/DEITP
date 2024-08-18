from argparse import Namespace
import os
import logging
import numpy as np
import copy

import torch

from utils.utils import get_dataloader, get_device
from utils.train_eval import task_train, task_val

class Learner():
    def __init__(self, model, scenarios, args:Namespace) -> None:
        self.args = args
        self.optim_obj = getattr(torch.optim, args.optimizer)
        self.model = model
        self.scenarios = scenarios
        self.extra_info = None

        # metrics
        self.metrics = {'train_loss':[], 'val_loss':[]}
        self.constant_metrics = {'min_val_epoch':-1, 'min_val_loss':9999}
        
        # logging
        self.root_path = f"./logs/{self.args.train_method}/{self.args.experiment_name}"
        self.ckpt_fname = None
        self.eval_path = None
        self.best_model = None
        if self.args.debugging:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
    
    def learn_tasks(self):
        for tid, (train_task, val_task, test_task) in enumerate(zip(self.scenarios.train_stream,
                                                                    self.scenarios.val_stream,
                                                                    self.scenarios.test_stream)):
            if train_task.task_id < self.args.train_start_task:
                print("Current task id is {} < {}, skip it.".format(train_task.task_id, self.args.train_start_task))
                continue
            
            assert tid >= self.args.train_start_task
            
            # Create checkpoint and evaluation path
            self.root_path = f"./logs/{self.args.train_method}/{self.args.experiment_name}"
            self.ckpt_fname = self.root_path+"/checkpoint/checkpoint_task_{}.pth".format(tid)
            self.eval_path  = self.root_path+"/evaluation/task_{}".format(tid)
            
            if not os.path.exists(self.eval_path) and not (self.args.no_train and self.args.no_test):
                os.makedirs(self.eval_path)
            
            # Before task learning
            logging.info("*"*30+" Preparing task {} ".format(tid)+"*"*30)
            # init the column if it is empty
            if len(self.model.columns) == 0:
                self.model.add_column()
            self.before_task_learning(tid)
            
            # Task learning
            logging.info("*"*30+"  Learning task {} ".format(tid)+"*"*30)
            self.task_learning(tid, train_task, val_task)
            
            # Store checkpoints
            self.store_checkpoint(self.model, self.extra_info)
            
            # After task learning
            logging.info("*"*30+"   Testing task {} ".format(tid)+"*"*30)
            self.after_task_learning(tid)
            
    def before_task_learning(self, tid):
        pass
    
    def after_task_learning(self, tid):
        pass
    
    def task_learning(self, tid, train_task, val_task):
        self.model = self.model.to(get_device())
        # get dataloaders
        train_loader = get_dataloader(train_task,shuffle=True)
        val_loader = get_dataloader(val_task, shuffle=True)
        
        # initialize the metrics container
        self.metrics['train_loss'] = []
        self.metrics['val_loss'] = []
        self.constant_metrics['min_val_epoch'] = -1
        self.constant_metrics['min_val_loss']  = 9999
        
        # learning process
        optimizer = self.optim_obj(self.model.parameters(), lr= self.args.learning_rate, weight_decay= 0)
        if self.args.use_lrschd:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.2)
        
        for ep_id in range(self.args.num_epochs):
            # training
            self.model, train_loss = task_train(self.model,optimizer,train_loader,self.args)
            self.metrics['train_loss'].append(train_loss)
            # validating
            val_loss = task_val(self.model,val_loader,self.args)
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
            
    def store_checkpoint(self, model, extra_info=None):
        f = dict()
        f['model_state_dict'] = model.state_dict()
        if extra_info is not None:
            for key, value in extra_info.items():
                f[key] = value
        torch.save(f, self.ckpt_fname)
    
    def get_extra_info(self,tid):
        return None
    
    def load_previous_knowledge(self, task_id):
        ckpt_fname = self.root_path+'/checkpoint/checkpoint_task_{}.pth'.format(task_id-1)
        logging.info(f"Load model in {ckpt_fname}")
        assert os.path.exists(ckpt_fname)
        checkpoint = torch.load(ckpt_fname)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
    def load_best_model(self):
        assert os.path.exists(self.ckpt_fname)
        logging.info(f"Load model in {self.ckpt_fname}")
        checkpoint = torch.load(self.ckpt_fname)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    