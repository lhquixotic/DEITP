from argparse import Namespace
import os
import logging
import copy

import torch

from learners.learner import Learner
from utils.train_eval import task_test_with_given_expert

class PNNLearner(Learner):
    def __init__(self, model, scenarios, args: Namespace) -> None:
        super(PNNLearner,self).__init__(model, scenarios, args)
    
    def before_task_learning(self, tid):
        # expand the model and load previous knowledge
        expand_times = tid - len(self.model.columns) + 1
        for t in range(expand_times):
            if t == expand_times - 1:
                self.load_previous_knowledge(tid)
            self.model.add_column()
        logging.info("Current column number is {}, batch norm para number is {}".format(
            len(self.model.columns), len(self.model.batch_norm_para)))
        
        # initialize new column with previous parameters
        if len(self.model.columns) >= 2 and self.args.init_new_expert:
            self.model.init_new_expert(copy.deepcopy(self.model.columns[-2]))

    def after_task_learning(self, tid):
        # load best model
        self.load_best_model()
        
        # test all previous task
        res = dict()
        for id, task in enumerate(self.scenarios.test_stream):
            id += self.args.test_start_task
            if id > tid:
                break
            self.model.load_batch_norm_para(id)
            ade, fde = task_test_with_given_expert(self.model, id, expert_id=id, test_task=task,
                                                   args=self.args, save_dir=self.eval_path)
            logging.info("[Test] columns num:{}, task_{}, expert_{}, ADE:{:.2f}, fde:{:.2f}".format(
                len(self.model.columns), id, id, ade.mean(), fde.mean()))
            res[(id,0)] = [ade.mean(), fde.mean()]
            
        # store the performance
        with open(self.root_path+"/evaluation/performance.txt", "w") as file:
            for key, val in res.items():
                file.write(f"{key}: {val}\n")
            file.write("*"*40+"\n")
                    
    def load_previous_knowledge(self, task_id):
        ckpt_fname = self.root_path+'/checkpoint/checkpoint_task_{}.pth'.format(task_id-1)
        logging.info(f"Load model in {ckpt_fname}")
        assert os.path.exists(ckpt_fname)
        checkpoint = torch.load(ckpt_fname)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        assert "batch_norm_para" in checkpoint.keys()
        self.model.batch_norm_para = copy.deepcopy(checkpoint["batch_norm_para"])
        self.model.freeze_columns()
        self.model.load_batch_norm_para(task_id-1)

    def get_extra_info(self, tid):
        self.model.save_batch_norm_para(tid)
        extra_info = dict()
        extra_info["batch_norm_para"] = copy.deepcopy(self.model.batch_norm_para)
        return extra_info
    
    def load_best_model(self):
        assert os.path.exists(self.ckpt_fname)
        checkpoint = torch.load(self.ckpt_fname)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        assert "batch_norm_para" in checkpoint.keys()
        self.model.batch_norm_para = checkpoint["batch_norm_para"]
    
