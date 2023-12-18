import argparse
import logging
import os
import shutil
import CTBert
import warnings

from CTBert.load_pretrain_data import load_labeled_classify_data

dev = 'cuda'
warnings.filterwarnings("ignore")

# set random seed
CTBert.random_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='CT-BERT-supCL-pretrain')
    parser.add_argument("--data_args", type=str, default="/home/vivian/SoftwareDesign_FinalProject/pretrain_dataset/data_label", help="load_data's path")
    args = parser.parse_args()
    return args

_args = parse_args()

cal_device = dev
cpt = './SoftwareDesign_FinalProject/checkpoint-pretrain-openml'


# openml big datasets
trainset, valset, cat_cols, num_cols, bin_cols  = \
    load_labeled_classify_data(label_data_path=_args.data_args)


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
    vocab_freeze=True
)

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
