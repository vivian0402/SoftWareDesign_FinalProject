import logging
import os
import sys
import time
from pathlib import Path
import warnings
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from CTBert.data_loader import DataLoader
from CTBert.train import BaseTrainer
from CTBert.evaluator import Evaluator
from CTBert .CTBert import BaseCTBertModel
from sklearn.model_selection import train_test_split
import shutil
import numpy as np
import deepspeed
import random
import torch

class BaseRunFile:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda'
        self.log_config()
        if self.args.task_data_info is not None:
            self.df = pd.read_csv(self.args.task_data_info)
        self.skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        self.checkpoint_save = f'./save_info/temp_models/{self.args.checkpoint_save}'
        self.task_type = args.task_type
        self.model_type = args.model_type
        self.random_seed(42)
        self.evaluator = Evaluator()
        warnings.filterwarnings("ignore")    

    def create_run_file(self):
        if self.args.task_type == 'pretrain_CL_ds':
            run_file = PretrainCLDeepSeed(self.args)
        elif self.args.task_type == 'pretrain_mask_ds':
            run_file = PretrainMaskDeepSeed(self.args)
        elif self.args.task_type == 'pretrain_mask':
            run_file = PretrainMask(self.args)
        elif self.args.task_type == 'fintune':
            run_file = Finetune(self.args)
        elif self.args.task_type == 'scratch':
            run_file = Scratch(self.args)
        else:
            raise ValueError(f"Unknown task: {self.args.task_type}")
        
        run_file.run()

    def log_config(self):
        log_name = self.args.task_type
        exp_dir = f'{log_name}_{time.strftime("%Y%m%d-%H%M%S")}'
        exp_log_dir = Path('save_info/Log') / exp_dir
        setattr(self.args, 'exp_log_dir', exp_log_dir)
        os.makedirs(exp_log_dir, exist_ok=True)
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(exp_log_dir / 'log.txt')
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
    
    def random_seed(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

class PretrainCLDeepSeed(BaseRunFile):
    def __init__(self, args):
        super().__init__(args)
        self.training_arguments = {
            'num_epoch': self.args.num_epoch,
            'batch_size':self.args.batch_size,
            'lr':self.args.lr,
            'eval_metric_name':self.args.eval_metric,
            'eval_less_is_better':True,
            'output_dir':self.checkpoint_save,
            'patience':self.args.patience,
        }

    def run(self):
        self.log_config()

        if self.args.pretrain_label_dataset is None:
            print("please input pretrain_label_dataset")
            return
        else:
            try:
                data_loader = DataLoader(
                    task_type=self.args.task_type,
                    label_data_path=self.args.pretrain_label_dataset,
                    limit=10,
                    is_classify=True
                )
                trainset, valset, cat_cols, num_cols, bin_cols = data_loader.create_loader()
            except Exception as e:
                print(f"An error occurred while loading data: {e}")
                return

        self.model_params = {
            'categorical_columns': cat_cols,
            'numerical_columns': num_cols,
            'binary_columns': bin_cols,
            'device': self.device,
            'hidden_dropout_prob': self.args.hidden_dropout_prob,
            'num_layer': self.args.num_layer,
            'vocab_freeze': True,
            'task_type': self.args.task_type,
            'model_type': self.args.model_type,
            'num_attention_head':self.args.num_attention_head,
            'projection_dim':128,
            'supervised':False,
            'num_partition':2,
            'overlap_ratio':0.5,
            'ignore_duplicate_cols':True
        }
        logging.info(self.model_params)
        model= BaseCTBertModel(**self.model_params)
        model, collate_fn = model.create_model()

        logging.info(self.training_arguments)
        if os.path.isdir(self.training_arguments['output_dir']):
            shutil.rmtree(self.training_arguments['output_dir'])
        trainer = BaseTrainer(model, trainset, valset, collate_fn=collate_fn, use_deepspeed=False, **self.training_arguments)
        trainer.create_trainer()

class PretrainMaskDeepSeed(BaseRunFile):
    def __init__(self, args):
        super().__init__(args)
        self.training_arguments = {
            'num_epoch': self.args.num_epoch,
            'batch_size':self.args.batch_size,
            'lr':self.args.lr,
            'eval_metric_name':'val_loss',
            'eval_less_is_better':True,
            'output_dir':self.checkpoint_save,
            'patience':self.args.patience,
            'num_workers':0,
        }

    def configure_logging(self):
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        log_format = "[%(asctime)s] [%(levelname)s] [%(message)s]"
        logging.basicConfig(level=log_level, format=log_format, filename=self.args.log_path, filemode='a')

        if "OMPI_COMM_WORLD_RANK" in os.environ or "RANK" in os.environ:
            mpi_rank = os.getenv("OMPI_COMM_WORLD_RANK") or os.getenv("RANK")
            formatter = logging.Formatter(f"[rank {mpi_rank}] " + log_format)
            for handler in logging.root.handlers:
                handler.setFormatter(formatter)

    def run(self):
        deepspeed.init_distributed()

        self.configure_logging()

        if self.args.pretrain_label_dataset is None and self.args.pretrain_unlabel_dataset is None:
            print("please input pretrain_label_dataset & pretrain_unlabel_dataset")
            return
        else:
            try:
                data_loader = DataLoader(
                    task_type=self.args.task_type,
                    label_data_path=self.args.pretrain_label_dataset,
                    unlabel_data_path=self.args.pretrain_unlabel_dataset,
                    limit=10,
                )
                trainset, valset, cat_cols, num_cols, bin_cols = data_loader.create_loader()
            except Exception as e:
                print(f"An error occurred while loading data: {e}")
        
        self.model_params = {
            'categorical_columns': cat_cols,
            'numerical_columns': num_cols,
            'binary_columns': bin_cols,
            'device': self.device,
            'mlm_probability':self.args.mlm_probability,
            'hidden_dropout_prob': self.args.hidden_dropout_prob,
            'num_layer': self.args.num_layer,
            'vocab_freeze': True,
            'task_type': self.args.task_type,
            'model_type': self.args.model_type,
            'num_attention_head':self.args.num_attention_head,
            'projection_dim':128
        }
        logging.info(self.model_params)
        model = BaseCTBertModel(**self.model_params)
        model = model.create_model()

        logging.info(self.training_arguments)
        trainer = BaseTrainer(model, trainset, valset, use_deepspeed=True, cmd_args=self.args, **self.training_arguments)
        trainer.create_trainer()
        
class PretrainMask(BaseRunFile):
    def __init__(self, args):
        super().__init__(args)
        self.training_arguments = {
            'num_epoch':self.args.num_epoch,
            'batch_size':self.args.batch_size,
            'lr':self.args.lr,
            'eval_metric_name':self.args.eval_metric,
            'eval_less_is_better':True,
            'output_dir':self.checkpoint_save,
            'device':self.device,
            'patience':self.args.patience,
        }

    def run(self):
        self.log_config()

        if self.args.pretrain_label_dataset is None and self.args.pretrain_unlabel_dataset is None:
            print("please input pretrain_label_dataset & pretrain_unlabel_dataset")
            return
        else:
            try:
                data_loader = DataLoader(
                    task_type=self.args.task_type,
                    label_data_path=self.args.pretrain_label_dataset,
                    unlabel_data_path=self.args.pretrain_unlabel_dataset,
                    limit=10,
                )
                trainset, valset, cat_cols, num_cols, bin_cols = data_loader.create_loader()
            except Exception as e:
                print(f"An error occurred while loading data: {e}")
                
        self.model_params = {
            'categorical_columns': cat_cols,
            'numerical_columns': num_cols,
            'binary_columns': bin_cols,
            'device': self.device,
            'mlm_probability':self.args.mlm_probability,
            'hidden_dropout_prob': self.args.hidden_dropout_prob,
            'num_layer': self.args.num_layer,
            'vocab_freeze': True,
            'task_type': self.args.task_type,
            'model_type': self.args.model_type,
            'num_attention_head':self.args.num_attention_head,
            'projection_dim':128
        }
        logging.info(self.model_params)
        model = BaseCTBertModel(**self.model_params)
        model = model.create_model()

        logging.info(self.training_arguments)
        if os.path.isdir(self.training_arguments['output_dir']):
            shutil.rmtree(self.training_arguments['output_dir'])
        trainer = BaseTrainer(model, trainset, valset, use_deepspeed=False, **self.training_arguments)
        trainer.create_trainer()
        
class Finetune(BaseRunFile):
    def __init__(self, args):
        super().__init__(args)
        
        self.training_arguments = {
            'num_epoch':self.args.num_epoch,
            'batch_size':self.args.batch_size,
            'lr':self.args.lr,
            'eval_metric_name':self.args.eval_metric,
            'eval_less_is_better':False,
            'output_dir':self.checkpoint_save,
            'patience':self.args.patience,
            'num_workers':0,
            'device':self.device,
            'flag':1
        }

    def run(self):
        self.log_config()

        all_res = {}
        for index, table_info in self.df.iterrows():
            task = table_info['file_name']

            logging.info(f'Start========>{task}_DataSet==========>')
            table_file = self.args.task_dataset + task
            data_loader = DataLoader(
                    task_type=self.args.task_type,
                    task_target=table_info['target'],
                    task_data_path=table_file
                )
            X, y, cat_cols, num_cols, bin_cols = data_loader.create_loader() 
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

            num_class = len(y.value_counts())
            logging.info(f'num_class : {num_class}')
            cat_cols = [cat_cols]
            num_cols = [num_cols]
            bin_cols = [bin_cols]
            idd = 0
            score_list = []
            for trn_idx, val_idx in self.skf.split(X, y):
                self.random_seed(42)
                idd += 1
                train_data = X.loc[trn_idx]
                train_label = y[trn_idx]
                X_test = X.loc[val_idx]
                y_test = y[val_idx]
                X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=0, stratify=train_label, shuffle=True)
                
                self.model_params = {
                    'categorical_columns': cat_cols,
                    'numerical_columns': num_cols,
                    'binary_columns': bin_cols,
                    'checkpoint': self.args.checkpoint_load,
                    'device': self.device,
                    'num_class': num_class,
                    'num_layer': self.args.num_layer,
                    'hidden_dropout_prob': self.args.hidden_dropout_prob,
                    'vocab_freeze': True,
                    'task_type': self.args.task_type,
                    'model_type': self.args.model_type,
                }

                model = BaseCTBertModel(**self.model_params)
                model = model.create_model()

                model.update({'cat':cat_cols, 'num':num_cols, 'bin':bin_cols})
                
                logging.info(self.training_arguments)
                if os.path.isdir(self.training_arguments['output_dir']):
                    shutil.rmtree(self.training_arguments['output_dir'])
                trainer = BaseTrainer(model, (X_train, y_train), (X_val, y_val), **self.training_arguments)
                trainer.create_trainer()
                
                ypred = self.evaluator.predict(model, X_test)
                ans = self.evaluator.evaluate(ypred, y_test, metric='auc', num_class=num_class)
                score_list.append(ans[0])
                logging.info(f'Test_Score_{idd}===>{task}_DataSet==> {ans[0]}')
            all_res[task] = np.mean(score_list)
            logging.info(f'Test_Score_5_fold===>{task}_DataSet==> {np.mean(score_list)}')

        mean_list = []
        for key in all_res:
            logging.info(f'meaning_5_fold=>{all_res[key]}=>{key}')
            mean_list.append(all_res[key])
        logging.info(f'meaning all data=>{np.mean(mean_list)}')

class Scratch(BaseRunFile):
    def __init__(self, args):
        super().__init__(args)
        self.training_arguments = {
            'num_epoch':self.args.num_epoch,
            'batch_size':self.args.batch_size,
            'lr':self.args.lr,
            'eval_metric_name':self.args.eval_metric,
            'eval_less_is_better':False,
            'output_dir':self.checkpoint_save,
            'patience':self.args.patience,
            'num_workers':0,
            'device':self.device,
            'flag':1
        }

    def run(self):
        self.log_config()

        all_res = {}
        for index, table_info in self.df.iterrows():
            task = table_info['file_name']

            logging.info(f'Start========>{task}_DataSet==========>')
            table_file = self.args.task_dataset + task
            data_loader = DataLoader(
                    task_type=self.args.task_type,
                    task_target=table_info['target'],
                    task_data_path=table_file
                )
            X, y, cat_cols, num_cols, bin_cols = data_loader.create_loader() 
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

            num_class = len(y.value_counts())
            logging.info(f'num_class : {num_class}')
            cat_cols = [cat_cols]
            num_cols = [num_cols]
            bin_cols = [bin_cols]
            idd = 0
            score_list = []
            for trn_idx, val_idx in self.skf.split(X, y):
                self.random_seed(42)
                idd += 1
                train_data = X.loc[trn_idx]
                train_label = y[trn_idx]
                X_test = X.loc[val_idx]
                y_test = y[val_idx]
                X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=0, stratify=train_label, shuffle=True)
                self.model_params = {
                    'categorical_columns': cat_cols,
                    'numerical_columns': num_cols,
                    'binary_columns': bin_cols,
                    'device': self.device,
                    'num_class': num_class,
                    'num_layer': self.args.num_layer,
                    'hidden_dropout_prob': self.args.hidden_dropout_prob,
                    'vocab_freeze': True,
                    'use_bert': True,
                    'task_type': self.args.task_type,
                    'model_type': self.args.model_type,
                }
                model = BaseCTBertModel(**self.model_params)
                model = model.create_model()
                
                logging.info(self.training_arguments)
                if os.path.isdir(self.training_arguments['output_dir']):
                    shutil.rmtree(self.training_arguments['output_dir'])
                trainer = BaseTrainer(model, (X_train, y_train), (X_val, y_val), **self.training_arguments)
                trainer.create_trainer()
                
                ypred = self.evaluator.predict(model, X_test)
                ans = self.evaluator.evaluate(ypred, y_test, metric='auc', num_class=num_class)
                score_list.append(ans[0])
                logging.info(f'Test_Score_{idd}===>{task}_DataSet==> {ans[0]}')
            all_res[task] = np.mean(score_list)
            logging.info(f'Test_Score_5_fold===>{task}_DataSet==> {np.mean(score_list)}')

        mean_list = []
        for key in all_res:
            logging.info(f'meaning_5_fold=>{all_res[key]}=>{key}')
            mean_list.append(all_res[key])
        logging.info(f'meaning all data=>{np.mean(mean_list)}')