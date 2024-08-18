from argparse import Namespace
import os
import logging
import copy
import numpy as np

import torch

from learners.learner import Learner
from utils.train_eval import task_test_with_given_expert
from models.familarity_autoencoder import GraphAutoEncoder,train,valid
from utils.utils import get_dataloader

class DEMLearner(Learner):
    def __init__(self, model, scenarios, args: Namespace) -> None:
        super(DEMLearner,self).__init__(model, scenarios, args)
        # set task detector if task label is missing in test
        if args.task_free:
            self.model.set_task_detector(True) 
    
    def before_task_learning(self, tid):
        if self.args.no_test:
            logging.info("No loading and initialization before learning.")
            return
        
        # load previous expert selector knowledge
        if tid>0:
            self.load_previous_knowledge(tid)
            logging.info("Current column number is {}, batch norm para number is {}, expert selector is {}".format(
            len(self.model.columns), len(self.model.batch_norm_para), self.model.expert_selector))
            
            # expand the model and load previous knowledge
            expand_times = len(self.model.expert_selector) - len(self.model.columns) + 1
            for t in range(expand_times):
                if t == expand_times - 1:
                    self.load_previous_knowledge(tid)
                if not self.args.no_train:
                    self.model.add_column()
            
        # initialize the expert selector for current task id
        self.model.expert_selector[len(self.model.columns)-1].append(tid)
        
        task_detector_num = len(self.model.task_faes) if self.args.task_free else 0
        logging.info("Column_num: {}, BN_para_num: {}, Task_detector_num: {}, Expert_selector: {}".format(
            len(self.model.columns), len(self.model.batch_norm_para), task_detector_num, self.model.expert_selector))
        if self.args.task_free:
            assert task_detector_num == tid + 1
        
        # initialize new column with previous parameters
        if len(self.model.columns) >= 2 and self.args.init_new_expert:
            init_expert_id = self.get_initial_expert_id(tid)
            self.model.init_new_expert(copy.deepcopy(self.model.columns[init_expert_id]))

    def after_task_learning(self, tid):
        # whether to test
        if self.args.no_test:
            logging.info("No test after learning.")
            return
        
        # load best model
        self.load_best_model()
        
        # set task_detector True to enable when testing
        if self.args.task_free:
            self.model.set_task_detector(True)
        
        self.model.eval()
        
        # test all previous task
        res = dict()
        for id, task in enumerate(self.scenarios.test_stream):
            id += self.args.test_start_task
            if id > tid:
                break
            self.model.load_batch_norm_para(id)
            eid = self.model.select_expert(id)
            if (eid, id) in res.keys():
                continue
            ade, fde = task_test_with_given_expert(self.model, id, expert_id=eid, test_task=task,
                                                   args=self.args, save_dir=self.eval_path)
            logging.info("[Test] columns num:{}, task_{}, expert_{}, ADE:{:.2f}, fde:{:.2f}".format(
                len(self.model.columns), id, eid, ade.mean(), fde.mean()))
            res[(eid,id)] = [ade.mean(), fde.mean()]
            
    def task_learning(self, tid, train_task, val_task):
        # set task_detector false to use task label when training experts
        self.model.set_task_detector(False)
        
        # learning basic interactive behavior model        
        if not self.args.no_train:
            super().task_learning(tid, train_task, val_task)
            self.model = self.best_model
        
            # update the expert_selector
            self.update_expert_selector(tid)
        else:
            logging.info("No training main model")
        
        # learning FAE for current task
        if self.args.task_free and not self.args.no_fae_train:
            self.fae_learning(tid, train_task, val_task)
            
    def load_previous_knowledge(self, task_id):
        ckpt_fname = self.root_path+'/checkpoint/checkpoint_task_{}.pth'.format(task_id-1)
        logging.info(f"Load model in {ckpt_fname}")
        assert os.path.exists(ckpt_fname)
        checkpoint = torch.load(ckpt_fname, map_location=self.args.device)
        
        # load expert selector
        assert "expert_selector" in checkpoint.keys()
        self.model.expert_selector = checkpoint['expert_selector']
        prev_expert_num = len(checkpoint['expert_selector'])
        
        # load model state dict only when expert number equals to column num        
        if prev_expert_num == len(self.model.columns):
            self.model.load_state_dict(checkpoint['model_state_dict'])
            assert "batch_norm_para" in checkpoint.keys()
            self.model.batch_norm_para = copy.deepcopy(checkpoint["batch_norm_para"])
            self.model.freeze_columns()
            self.model.load_batch_norm_para(task_id-1)

    def get_extra_info(self, tid):
        self.model.save_batch_norm_para(tid)
        extra_info = dict()
        extra_info["batch_norm_para"] = copy.deepcopy(self.model.batch_norm_para)
        extra_info["expert_selector"] = copy.deepcopy(self.model.expert_selector)
        return extra_info
    
    def load_best_model(self):
        assert os.path.exists(self.ckpt_fname), "No checkpoint named {}".format(self.ckpt_fname)
        checkpoint = torch.load(self.ckpt_fname, map_location=self.args.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # load batch_norm parameters
        assert "batch_norm_para" in checkpoint.keys()
        self.model.batch_norm_para = checkpoint["batch_norm_para"]
        # load expert_select dict
        assert "expert_selector" in checkpoint.keys()
        self.model.expert_selector = checkpoint["expert_selector"]
    
    def update_expert_selector(self, tid):
        # load best model
        self.load_best_model()
        
        # test current task with all experts and record results
        res = dict()
        test_task = [task for id, task in enumerate(self.scenarios.test_stream) if id == tid]
        test_task = test_task[0]
        for eid in range(len(self.model.columns)):
            ade, fde = task_test_with_given_expert(self.model, tid, eid, test_task=test_task,
                                                    args=self.args, save_dir=self.eval_path)
            logging.info("[Test] columns num:{}, task_{}, expert_{}, ADE:{:.2f}, fde:{:.2f}".format(
                len(self.model.columns), tid, eid, ade.mean(), fde.mean()))
            res[(eid, tid)] = [ade.mean(), fde.mean()]
            
        # select expert according to ade distance
        selected_eid = len(self.model.columns) - 1
        best_eval = res[(selected_eid, tid)]
        
        # if the last expert is not best
        for key, val in res.items():
            if key[0] == len(self.model.columns) - 1:
                break # break if the model is the last one
            if val[0] - best_eval[0] < self.args.expand_thres:
                selected_eid = key[0]
                best_eval = res[key]

        # if update is needed
        if selected_eid != len(self.model.columns) - 1:
            del(self.model.expert_selector[len(self.model.columns)-1])
            self.model.columns = self.model.columns[:-1]
            logging.info(f"Update expert selector: expert_{selected_eid} -> task_{tid}, current column num is {len(self.model.columns)}")
            self.model.expert_selector[selected_eid].append(tid)
        
        logging.info("Current column number is {}, batch norm para number is {}, expert selector is {}".format(
            len(self.model.columns), len(self.model.batch_norm_para), self.model.expert_selector))
        
        # store the checkpoint
        self.extra_info = self.get_extra_info(tid)
    
    def get_initial_expert_id(self,task_id):
        return -2
    
    def fae_learning(self, tid, train_task, val_task):
        logging.info("[FAE] Starting to train FAE for task_{}".format(tid))
        fae = self.model.task_faes[-1]
        fae.train()
        optimizer = torch.optim.SGD(fae.parameters(), lr=self.args.fae_lr)
        if self.args.use_lrschd:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_sh_rate, gamma=0.2)
    
        self.fae_metrics = {'train_loss':[],'val_loss':[]}
        self.fae_constant_metrics = {'min_val_epoch':-1, 'min_val_loss':9999}
        self.fae_best_model = None
        train_loader = get_dataloader(train_task,shuffle=True)
        val_loader = get_dataloader(val_task, shuffle=True)
        
        for ep_id in range(self.args.num_epochs):
            # training
            fae, train_loss = train(fae, optimizer, train_loader, self.args.batch_size)
            self.fae_metrics['train_loss'].append(train_loss)
            # validating
            val_loss = valid(fae,val_loader,self.args.batch_size)
            self.fae_metrics['val_loss'].append(val_loss)
            if torch.isnan(torch.tensor(train_loss)).any():
                logging.warning("[FAE-Loss] Loss contains NaN values.")
            
            # save best FAE model 
            if val_loss < self.fae_constant_metrics['min_val_loss']:
                logging.info("[FAE-Loss] Epoch {}, train_loss: {:.4f}, val_loss: {:.4f}".format(ep_id,train_loss,val_loss))
                self.fae_constant_metrics['min_val_loss'] = val_loss
                self.fae_constant_metrics['min_val_epoch'] = ep_id
                self.fae_best_model = copy.deepcopy(fae)
                # self.store_checkpoint(self.best_model, self.extra_info)
                # save FAE separately
                os.makedirs("./logs/task_detector/{}/checkpoint".format(self.args.experiment_name), exist_ok=True)
                fae.save_model("./logs/task_detector/{}/checkpoint/FAE_task_{}.pth".format(self.args.experiment_name,tid))
            
            if ep_id - self.fae_constant_metrics['min_val_epoch'] > self.args.early_stop_epochs:
                break
            
            if self.args.use_lrschd:
                scheduler.step()
        
        self.model.task_faes[-1] = self.fae_best_model
        
        # save loss
        if self.args.store_loss:
            np_val_loss = np.array(self.fae_metrics['val_loss'])
            np_tra_loss = np.array(self.fae_metrics['train_loss'])
            np.save(self.root_path+'/loss/fae_val_task_{}.npy'.format(tid), np_val_loss)
            np.save(self.root_path+"/loss/fae_train_task_{}.npy".format(tid), np_tra_loss)
    
    def store_checkpoint(self, model, extra_info=None):
        if self.args.no_train:
            return
        return super().store_checkpoint(model, extra_info)
