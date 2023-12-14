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


from CTBert.load_pretrain_data import load_all_data

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
    exp_log_dir = Path('Log') / exp_dir
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
    parser = argparse.ArgumentParser(description='CT-BERT-mask-pretrain')
    parser.add_argument('--data', type=str, default="pretrain", help='task')
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()
    # # config.py ---> args's arri.
    # search_dataset_info = OPENML_DATACONFIG[args.data]
    # for key, value in search_dataset_info.items():
    #     setattr(args, key, value)
    return args

_args = parse_args()
log_config(_args)
logging.info(f'args : {_args}')
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
# allset, trainset, valset, cat_cols, num_cols, bin_cols = transtab.load_openml_data(limit=10)
trainset, valset, cat_cols, num_cols, bin_cols = load_all_data(
    label_data_path='/home/gslu/small_clean_pretrain_data/clean_labeled_dataset',
    unlabel_data_path='/home/gslu/small_clean_pretrain_data/clean_unlabeled_dataset',
    limit=10,
)

# ###############    pretrain    ################
model_arg = {
    'mlm_probability' : 0.35,
    'num_attention_head' : 8,
    'num_layer' : 3,
}
logging.info(model_arg)
model = CTBert.build_mask_features_learner(
    cat_cols, num_cols, bin_cols,
    mlm_probability=model_arg['mlm_probability'],
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
    'num_epoch': 500,
    'batch_size':256,
    'lr':3e-4,
    'eval_metric':'val_loss',
    'eval_less_is_better':True,
    'output_dir':cpt,
    'device':cal_device,
    'patience':5,
}
logging.info(training_arguments)
if os.path.isdir(training_arguments['output_dir']):
    shutil.rmtree(training_arguments['output_dir'])
CTBert.train(model, trainset, valset, use_deepspeed=False, **training_arguments)
