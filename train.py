from email import utils
import os
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__)))

from utils.utils import get_argument_parser, create_log_dirs, save_args, set_seeds
from utils.scenarios import get_continual_scenario_benchmark, get_joint_scenario_benchmark
from models import social_stgcnn, social_stgcnn_pnn, social_stgcnn_dem
from learners import pnn_learner, vanilla_learner, joint_learner, dem_learner, gsm_learner


if __name__ == '__main__':
    args = get_argument_parser()
    create_log_dirs(args)
    save_args(args)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.deterministic: set_seeds(args.seed)
    
    if args.train_method == "Joint":
        # train
        joint_scenario = get_joint_scenario_benchmark(args)
        continual_scenarios = get_continual_scenario_benchmark(args)
        model = social_stgcnn.social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                        output_feat=args.output_size, seq_len=args.obs_seq_len,
                        kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len)
        learner = joint_learner.JointLearner(model, joint_scenario, continual_scenarios, args)
    else:
        scenarios = get_continual_scenario_benchmark(args)
        # determine train method
        if args.train_method == "DEM":
            model_class = social_stgcnn_dem.social_stgcnn_dem
            learner_class = dem_learner.DEMLearner
        if args.train_method == "FineTuning":
            model_class = social_stgcnn.social_stgcnn
            learner_class = vanilla_learner.VanillaLearner
        if args.train_method == "GSM":
            model_class = social_stgcnn.social_stgcnn
            learner_class = gsm_learner.GSMLearner
        if args.train_method == "PNN":
            model_class = social_stgcnn_pnn.social_stgcnn_pnn
            learner_class = pnn_learner.PNNLearner
        
        model = model_class(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                        output_feat=args.output_size, seq_len=args.obs_seq_len,
                        kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len)
        learner = learner_class(model, scenarios, args)
        
    learner.learn_tasks()
