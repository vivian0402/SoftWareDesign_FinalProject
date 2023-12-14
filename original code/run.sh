CUDA_VISIBLE_DEVICES=0 python -u run_scratch.py --data sup_scratch > print_out/sup_scratch.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 python -u run_finetune.py --data finetune > print_out/finetune.txt 2>&1 &

deepspeed --master_port 29400 --num_gpus=4 run_mask_pretrain_ds.py --deepspeed_config ds_config_yc.json > print_out/pretrain.txt 2>&1 &
