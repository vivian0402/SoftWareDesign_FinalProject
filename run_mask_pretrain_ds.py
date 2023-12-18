import argparse
import logging
import os
import deepspeed
import CTBert
from loguru import logger
from CTBert.load_pretrain_data import load_all_data


# set random seed
CTBert.random_seed(42)

print('=========================>start running main')

def parse_args():
    parser = argparse.ArgumentParser(description='CT-BERT-mask-pretrain-ds')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument("--lable_data_args", type=str, default="/home/vivian/AIGO/ct-bert-main/pretrain_dataset/data_label", help="load_data's path")
    parser.add_argument("--unlable_data_args", type=str, default="/home/vivian/AIGO/ct-bert-main/pretrain_dataset/data_unlabel", help="load_data's path")
    parser.add_argument("--save_model", type=str, default="./mask_v7", help="save_model's path")
    parser.add_argument("--num_data", type=int, default=2000, help="num of the pretain datasets")
    parser.add_argument("--log_path", type=str, default="mask_log_v7.txt", help="read path about ds_config.json")

    parser.add_argument("--num_layer", type=int, default=4, help="num_layer")
    parser.add_argument("--mlm_probability", type=float, default=0.35, help="num_layer")
    parser.add_argument("--num_attention_head", type=int, default=8, help="num_attention_head")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="hidden_dropout_prob")

    parser.add_argument("--num_epoch", type=int, default=200, help="num_epoch")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--patience", type=int, default=5, help="patience")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

print('=========================>start deepspeed init')

deepspeed.init_distributed()

print('=========================>start set arg')

_args = parse_args()
dev = f'cuda:{_args.local_rank}'
logger.info(f'dev:{dev}')
cal_device = dev

if "OMPI_COMM_WORLD_RANK" in os.environ:
    # mpi env
    my_rank = int(os.getenv("OMPI_COMM_WORLD_RANK"))
elif "RANK" in os.environ:
    # torch distributed env
    my_rank = int(os.getenv("RANK"))
else:
    my_rank = 0

log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

logger_config = {
    "handlers": [
        {
            "sink": _args.log_path,
            "level": log_level,
            "colorize": True,
            "format": "[rank {extra[rank]}] [{time}] [{level}] {message}",
        },
    ],
    "extra": {"rank": my_rank},
}
logger.configure(**logger_config)

print('=========================>start read data')

trainset, valset, cat_cols, num_cols, bin_cols = load_all_data(
    label_data_path=_args.lable_data_args,
    unlabel_data_path=_args.unlable_data_args,
    limit=_args.num_data,
)

print('=========================>start init model and train')

model = CTBert.build_mask_features_learner(
    cat_cols, num_cols, bin_cols,
    mlm_probability=_args.mlm_probability,
    device=cal_device,
    hidden_dropout_prob=_args.hidden_dropout_prob,
    num_attention_head=_args.num_attention_head,
    num_layer=_args.num_layer,

    vocab_freeze=True,
)

training_arguments = {
    'num_epoch': _args.num_epoch,
    'batch_size':_args.batch_size,
    'lr':_args.lr,
    'eval_metric':'val_loss',
    'eval_less_is_better':True,
    'output_dir':_args.save_model,
    'patience':_args.patience,
    'num_workers':0,
}
logging.info(training_arguments)

CTBert.train(model, trainset, valset, use_deepspeed=True, cmd_args=_args, **training_arguments)