# SoftWareDesign_FinalProject
112-1 SoftWare Design - Desing Pattern Refactor final project

This task is to reconstruct the code. The original file source is: https://github.com/Grason-Lu/ct-bert

Environment: linux

***You need to write the information of the input dataset in a CSV file."

Execute command string example: $ python main.py -tt "pretrain_CL_ds" -mt "classify" -cs "checkpoint-pretrain-CL" -em "val_loss" -pld "/home/vivian/SoftwareDesign_FinalProject/pretrain_dataset/data_label" -nl 3 -nah 8 -hdp 0.2 -e 10 -b 256 -p 5 -lr 5e-5

Instruction:

-tt:"Type of task", type=str, required=True, choices=['pretrain_CL_ds', 'pretrain_mask_ds', 'pretrain_mask', 'fintune', 'scratch']

-mt: "Type of model", type=str, required=True, choices=['regression', 'classify']

-cs: 'checkpoint save folder name', type=str, required=True

-em: 'eval metric', type=str, required=True, choices=['acc', 'auc', 'mse', 'r2', 'rae', 'val_loss']


-di: 'data info', type=str

-d: 'task_data_path', type=str

-pld: 'pretrain_label_data_path', type=str

-pud: 'pretrain_unlabel_data_path', type=str


-cl: 'data_path', type=str, default="/home/vivian/SoftwareDesign_FinalProject/CT-BERT-v1/CT-BERT-v1/CT-BERT-v1-CL" (load from:https://github.com/Grason-Lu/ct-bert)

-lp: "read path about ds_config.json", type=str, default="mask_log_v7.txt"

-nl: "num_layer", type=int, default=4

-mp: "mlm_probability", type=float, default=0.35

-nah: "num_attention_head", type=int, default=8

-hdp: "hidden_dropout_prob", type=float, default=0.3

-e: "num_epoch", type=int, default=300

-b: "batch_size", type=int, default=64

-p: "patience", type=int, default=5

-lr: "learning rate", type=float, default=3e-4
