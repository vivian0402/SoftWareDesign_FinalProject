import argparse
import logging
import os
import shutil
import sys
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import train_test_split
import CTBert
import warnings
from torchsummary import summary

from CTBert.load_pretrain_data import load_labeled_classify_data

dev = 'cuda'
warnings.filterwarnings("ignore")

# set random seed
CTBert.random_seed(42)

def log_config(args):
    """
    log Configuration information, specifying the saving path of output log file, etc
    :return: None
    """
    dataset_name = args.data
    exp_dir = 'search_{}_{}'.format(dataset_name, time.strftime("%Y%m%d-%H%M%S"))
    exp_log_dir = Path('pretrain_logs') / exp_dir
    # save argss
    setattr(args, 'exp_log_dir', exp_log_dir)

    if not os.path.exists(exp_log_dir):
        os.mkdir(exp_log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(exp_log_dir / 'log.txt')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

def parse_args():
    parser = argparse.ArgumentParser(description='CT-BERT-supCL-pretrain')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help='local rank passed from distributed launcher')
    parser.add_argument("--data_args", type=str, default="/home/gslu/small_clean_pretrain_data/clean_labeled_dataset", help="load_data's path")
    parser.add_argument("--save_model", type=str, default="./DS_pretrain", help="save_model's path")
    parser.add_argument("--num_data", type=int, default=3, help="num of the pretain datasets")
    parser.add_argument("--ds_config_path", type=str, default="./ds_config_lgs.json", help="read path about ds_config_lgs.json")

    parser.add_argument("--is_supervised", type=int, default=1, help=" if take supervised CL")
    parser.add_argument("--coresize", type=int, default=10000, help=" the size of coreset")
    parser.add_argument("--fe_limit", type=int, default=50, help=" fe_limit")
    parser.add_argument("--vocab_freeze", type=int, default=1, help=" vocab_freeze")
    args = parser.parse_args()
    return args

_args = parse_args()
# log_config(_args)
# logging.info(f'args : {_args}')
###############   choice dataset and device   ###################
pretrain_dataset = [
                'credit-g',
                'credit-approval',
                # 'dresses-sales',
                # 'adult',
                # 'cylinder-bands',
                # 'telco-customer-churn',
                # 'data/IO',
                # 'data/IC',
                # 'data/BM',
                # 'data/ST',
            ]
cal_device = dev
cpt = './checkpoint-pretrain-openml'
# 10 datasets
# Allset, cat_cols, num_cols, bin_cols = transtab.load_data(pretrain_dataset)
# trainset = []
# valset = []
# for X, y in Allset:
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y, shuffle=True)
#     trainset.append((X_train, y_train))
#     valset.append((X_val, y_val))


# openml big datasets
allset, trainset, valset, cat_cols, num_cols, bin_cols  = \
    load_labeled_classify_data(label_data_path=_args.data_args, coresize=_args.coresize, limit=_args.num_data, fe_limit=_args.fe_limit)


# ###############    pretrain    ################
model_arg = {
    'num_partition' : 2,
    'overlap_ratio' : 0.5,
    'num_attention_head' : 8,
    'num_layer' : 3,
}
logging.info(model_arg)
model, collate_fn = CTBert.build_contrastive_learner(
    cat_cols, num_cols, bin_cols,
    supervised=False, # if take supervised CL
    num_partition=model_arg['num_partition'], # num of column partitions for pos/neg sampling
    overlap_ratio=model_arg['overlap_ratio'], # specify the overlap ratio of column partitions during the CL
    device=cal_device,
    hidden_dropout_prob=0.2,
    num_attention_head=model_arg['num_attention_head'],
    num_layer=model_arg['num_layer'],
    vocab_freeze=True,

    # hidden_dim=768,
    # ffn_dim=1536,
    # projection_dim=256,
)
# total_params = sum(p.numel() for p in model.parameters())
# summary(model.encoder, input_size=[(100, 128), (100,)])
training_arguments = {
    'num_epoch': 300,
    'batch_size':256,
    'lr':5e-5,
    'eval_metric':'val_loss',
    'eval_less_is_better':True,
    'output_dir':cpt,
    'patience':5,
    # 'num_workers':24
}
logging.info(training_arguments)
if os.path.isdir(training_arguments['output_dir']):
    shutil.rmtree(training_arguments['output_dir'])
CTBert.train(model, trainset, valset, collate_fn=collate_fn, use_deepspeed=False, **training_arguments)
