import argparse
from run_file import BaseRunFile
import deepspeed

def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-tt", "--task_type", type=str, required=True, choices=['pretrain_CL_ds', 'pretrain_mask_ds', 'pretrain_mask', 'fintune', 'scratch'], help="Type of task")
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.task_type == 'pretrain_mask_ds':
        parser = deepspeed.add_config_arguments(argparse.ArgumentParser())
    else:
        parser = argparse.ArgumentParser()

    parser.add_argument("-tt", "--task_type", type=str, required=True, choices=['pretrain_CL_ds', 'pretrain_mask_ds', 'pretrain_mask', 'fintune', 'scratch'], help="Type of task")
    parser.add_argument("-mt", "--model_type", type=str, required=True, choices=['regression', 'classify'], help="Type of model")
    parser.add_argument("-cs", '--checkpoint_save', type=str, required=True, help='checkpoint save folder name') 
    parser.add_argument("-em", '--eval_metric', type=str, required=True, choices=['acc', 'auc', 'mse', 'r2', 'rae', 'val_loss'], help='data_path')
    
    parser.add_argument("-di", '--task_data_info', type=str, help='data info')
    parser.add_argument("-d", '--task_dataset', type=str, help='task_data_path')
    parser.add_argument("-pld", '--pretrain_label_dataset', type=str, help='pretrain_label_data_path')
    parser.add_argument("-pud", '--pretrain_unlabel_dataset', type=str, help='pretrain_unlabel_data_path')
    
    parser.add_argument("-cl", '--checkpoint_load', type=str, default="/home/vivian/SoftwareDesign_FinalProject/CT-BERT-v1/CT-BERT-v1/CT-BERT-v1-CL", help='data_path')
    parser.add_argument("-lp", "--log_path", type=str, default="mask_log_v7.txt", help="read path about ds_config.json")
    parser.add_argument("-nl", "--num_layer", type=int, default=4, help="num_layer")
    parser.add_argument("-mp", "--mlm_probability", type=float, default=0.35, help="num_layer")
    parser.add_argument("-nah", "--num_attention_head", type=int, default=8, help="num_attention_head")
    parser.add_argument("-hdp", "--hidden_dropout_prob", type=float, default=0.3, help="hidden_dropout_prob")
    parser.add_argument("-e", "--num_epoch", type=int, default=300, help="num_epoch")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("-p", "--patience", type=int, default=5, help="patience")
    parser.add_argument("-lr", "--lr", type=float, default=3e-4, help="learning rate")
    args = parser.parse_args()

    run_file = BaseRunFile(args)
    run_file.create_run_file()

if __name__ == '__main__':
    main()
