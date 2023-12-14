import argparse
import logging
import os
import shutil

import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import train_test_split
from CTBert.dataset_openml import load_single_data_all, Feature_type_recognition


import CTBert

import warnings
warnings.filterwarnings("ignore")

# set random seed
CTBert.random_seed(42)

cal_device = 'cuda'

def log_config(args):
    """
    log Configuration information, specifying the saving path of output log file, etc
    :return: None
    """
    dataset_name = args.data
    exp_dir = 'search_{}_{}'.format(dataset_name, time.strftime("%Y%m%d-%H%M%S"))
    exp_log_dir = Path('fintune') / exp_dir
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
    parser = argparse.ArgumentParser(description='CT-BERT-finetune')
    parser.add_argument('--data', type=str, default="finetune", help='task')
    args = parser.parse_args()
    return args

_args = parse_args()
log_config(_args)


df = pd.read_csv('/home/gslu/task_data/task_data_new.csv')
cpt = './mask_v5_0dot7/epoch_50_valloss_0.09760503503600243'


skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
auto_feature_type = Feature_type_recognition()
all_res = {}
for index, table_info in df.iterrows():
    task = table_info['file_name']

    logging.info(f'Start========>{task}_DataSet==========>')
    # table_file = '/home/gslu/CC18_meaning/data/' + task
    table_file = '/home/gslu/task_data/data/' + task
    X, y, cat_cols, num_cols, bin_cols = load_single_data_all(table_file, table_info['target'], auto_feature_type)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    num_class = len(y.value_counts())
    logging.info(f'num_class : {num_class}')
    cat_cols = [cat_cols]
    num_cols = [num_cols]
    bin_cols = [bin_cols]
    idd = 0
    score_list = []
    for trn_idx, val_idx in skf.split(X, y):
        CTBert.random_seed(42)
        idd += 1
        cpt1 = f'./temp_models/checkpoint-finetune'
        train_data = X.loc[trn_idx]
        train_label = y[trn_idx]
        X_test = X.loc[val_idx]
        y_test = y[val_idx]
        X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=0, stratify=train_label, shuffle=True)
        model = CTBert.build_classifier(
            checkpoint=cpt,

            device=cal_device,
            num_class=num_class,
            num_layer=4,
            hidden_dropout_prob=0.3,
            vocab_freeze=True,
        )
        model.update({'cat':cat_cols, 'num':num_cols, 'bin':bin_cols})
        training_arguments = {
            'num_epoch':300,
            'batch_size':64,
            'lr':3e-4,
            'eval_metric':'auc',
            'eval_less_is_better':False,
            'output_dir':cpt1,
            'patience':20,
            'num_workers':0,
            'device':cal_device,
            'flag':1
        }
        logging.info(training_arguments)
        if os.path.isdir(training_arguments['output_dir']):
            shutil.rmtree(training_arguments['output_dir'])
        CTBert.train(model, (X_train, y_train), (X_val, y_val), **training_arguments)

        ypred = CTBert.predict(model, X_test)
        ans = CTBert.evaluate(ypred, y_test, metric='auc', num_class=num_class)
        score_list.append(ans[0])
        logging.info(f'Test_Score_{idd}===>{task}_DataSet==> {ans[0]}')
    all_res[task] = np.mean(score_list)
    logging.info(f'Test_Score_5_fold===>{task}_DataSet==> {np.mean(score_list)}')

mean_list = []
for key in all_res:
    logging.info(f'meaning_5_fold=>{all_res[key]}=>{key}')
    mean_list.append(all_res[key])
logging.info(f'meaning all data=>{np.mean(mean_list)}')