import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
import CTBert
import warnings


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
    exp_log_dir = Path('SoftwareDesign_FinalProject/Log') / exp_dir
    # save argss
    setattr(args, 'exp_log_dir', exp_log_dir)

    os.makedirs(exp_log_dir, exist_ok=True)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(exp_log_dir / 'log.txt')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

def parse_args():
    parser = argparse.ArgumentParser(description='CT-BERT-mask-pretrain')
    parser.add_argument('--data', type=str, default="pretrain", help='task')
    args = parser.parse_args()
    return args

_args = parse_args()
log_config(_args)
logging.info(f'args : {_args}')
###############   choice dataset and device   ###################

cal_device = dev
cpt = './SoftwareDesign_FinalProject/checkpoint-pretrain-openml'



# openml big datasets
trainset, valset, cat_cols, num_cols, bin_cols = load_all_data(
    label_data_path='/home/vivian/SoftwareDesign_FinalProject/pretrain_dataset/data_label',
    unlabel_data_path='/home/vivian/SoftwareDesign_FinalProject/pretrain_dataset/data_unlabel',
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
)

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
