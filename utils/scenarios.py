from easydict import EasyDict as edict
import argparse
import pickle

from torch.utils.data import random_split, ConcatDataset

from utils.data_process import *

class Benchmark():
    def __init__(self,train_datasets, test_datasets, val_datasets=None) -> None:
       self.train_stream = []
       self.test_stream = []
       self.val_stream = []
       self.initialize_streams(self.train_stream, train_datasets)
       self.initialize_streams(self.test_stream,test_datasets)
       if val_datasets is not None:
           self.initialize_streams(self.val_stream,val_datasets)
        
    def initialize_streams(self,stream,datasets):
        for id,dataset in enumerate(datasets):
            task = {"task_id": id, "dataset": dataset}
            stream.append(edict(task))
        
def load_datasets(args: argparse.Namespace):
    # Load continual scenarios sequence
    task_seq = args.task_seq
    start_task_id = min(args.train_start_task, args.test_start_task)
    task_dict = {1:'MA',2:'FT',3:'ZS',4:'EP',5:'SR'}
    dataset_ids = [int(task_id) for task_id in task_seq.split("-")]
    dataset_ids = [dataset_id for task_id,dataset_id in enumerate(dataset_ids) if task_id >= start_task_id]
    print("[Scenario] Task sequence:",dataset_ids)
 
    train_datasets = []
    val_datasets = []
    test_datasets  = []
    for dataset_id in dataset_ids:
        print("*" * 40 + " " + "Dataset: " +str(dataset_id)+"-" + task_dict[dataset_id] +" " + "*"*40)
        dataset_filename = "./data/original/" + str(dataset_id)+"-" + task_dict[dataset_id]+"/"
        dir_names = ['train/','val/','test/']
        if args.is_demo:
            dir_names = ['tinydemo/','tinydemo/','tinydemo/']
        train_dir, val_dir, test_dir = [dataset_filename+dir_name for dir_name in dir_names]
        
        # load pkl if exists
        pro_dataset_fname = "./data/processed/"+ str(dataset_id)+"-" + task_dict[dataset_id]+"/" 
        if args.is_demo:
            pro_dataset_fname = "./data/processed_demo/"+ str(dataset_id)+"-" + task_dict[dataset_id]+"/"
        if not os.path.exists(pro_dataset_fname):
            os.makedirs(pro_dataset_fname)
        pro_file_names = ["train.pkl", "val.pkl", "test.pkl"]
        pro_train_file, pro_val_file, pro_test_file = [pro_dataset_fname + pro_file_name for pro_file_name in pro_file_names]
        # Train Dataset
        if os.path.exists(pro_train_file):
            with open(pro_train_file, "rb") as file:
                train_data = pickle.load(file)
            print("Loaded {}, length is {}.".format(pro_train_file, len(train_data)))
        else:
            train_data = TrajectoryDataset(train_dir,
                                        obs_len=args.obs_seq_len,
                                        pred_len=args.pred_seq_len,
                                        skip=1,
                                        norm_lap_matr=True)
            with open(pro_train_file, "wb") as file:
                pickle.dump(train_data, file)
        # Validation Dataset
        if os.path.exists(pro_val_file):
            with open(pro_val_file, "rb") as file:
                val_data = pickle.load(file)
            print("Loaded {}, length is {}".format(pro_val_file, len(val_data)))
        else:
            val_data = TrajectoryDataset(val_dir,
                                        obs_len=args.obs_seq_len,
                                        pred_len=args.pred_seq_len,
                                        skip=1,
                                        norm_lap_matr=True)
            with open(pro_val_file, "wb") as file:
                pickle.dump(val_data, file)
        # Test Dataset  
        if os.path.exists(pro_test_file):
            with open(pro_test_file, "rb") as file:
                test_data = pickle.load(file)
            print("Loaded {}, length is {}".format(pro_test_file, len(test_data)))
        else:     
            test_data = TrajectoryDataset(test_dir,
                                        obs_len=args.obs_seq_len,
                                        pred_len=args.pred_seq_len,
                                        skip=1,
                                        norm_lap_matr=True)
            with open(pro_test_file, "wb") as file:
                pickle.dump(test_data, file)

        train_datasets.append(train_data)
        test_datasets.append(test_data)
        val_datasets.append(val_data)
    return train_datasets, val_datasets, test_datasets

def get_continual_scenario_benchmark(args: argparse.Namespace):
    train_datasets, val_datasets, test_datasets = load_datasets(args)
    # split datasets to different tasks
    if args.datasets_split_num > 1:
        train_datasets = split_datasets(train_datasets,args.datasets_split_num)
        val_datasets = split_datasets(val_datasets,args.datasets_split_num)
        test_datasets = split_datasets(test_datasets,args.datasets_split_num)    
    print("-"*80)
    print("Loaded datasets completed, {} tasks in total.".format(len(train_datasets)))
    benchmark = Benchmark(train_datasets,test_datasets,val_datasets=val_datasets)
    return benchmark

def get_joint_scenario_benchmark(args:argparse.Namespace):
    train_datasets, val_datasets, test_datasets = load_datasets(args)
    # concat datasets into one 
    com_train_dataset = [ConcatDataset(train_datasets)]
    com_val_dataset = [ConcatDataset(val_datasets)]
    com_test_dataset = [ConcatDataset(test_datasets)]
    print("-"*80)
    print("Loaded datasets completed, {} tasks in total, train start from task_{}.".format(len(com_test_dataset),args.train_start_task))
    benchmark = Benchmark(com_train_dataset, com_test_dataset, val_datasets=com_val_dataset)
    return benchmark

def split_datasets(datasets, split_num):
    datasets_num = len(datasets)
    mid_datasets = []
    for dataset in datasets:
        dataset_len = len(dataset)
        split_size = int(dataset_len/split_num)
        split_sizes = [split_size] * (split_num-1) + [dataset_len - split_size*(split_num-1)]
        split_dataset = random_split(dataset,split_sizes)
        mid_datasets.append(split_dataset)
    split_datasets = []
    for j in range(split_num):
        for i in range(datasets_num):
            split_datasets.append(mid_datasets[i][j])
    return split_datasets