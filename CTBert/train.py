import os
import math
import time
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm.autonotebook import trange
from . import constants
from .evaluator import Evaluator
from .trainer_utils import SupervisedTrainCollator, TrainDataset
import logging
import deepspeed
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup
)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, output_dir='ckpt', trace_func=print, less_is_better=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = output_dir
        self.trace_func = trace_func
        self.less_is_better = less_is_better

    def __call__(self, val_loss, model):
        if self.patience < 0: # no early stop
            self.early_stop = False
            return
        
        if self.less_is_better:
            score = val_loss
        else:    
            score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.path, constants.WEIGHTS_NAME))
        self.val_loss_min = val_loss

class BaseTrainer:
    def __init__(self, model, train_set_list, test_set_list=None, collate_fn=None, use_deepspeed=False, **kwargs):
        self.model = model
        self.train_set_list = [train_set_list] if isinstance(train_set_list, tuple) else train_set_list
        self.test_set_list = [test_set_list] if isinstance(test_set_list, tuple) else test_set_list
        self.use_deepspeed = use_deepspeed

        self.regression_task = kwargs.get('regression_task', False)
        self.output_dir = kwargs.get('output_dir', './ckpt')
        self.num_epoch = kwargs.get('num_epoch', 10)
        self.batch_size = kwargs.get('batch_size', 64)
        self.lr = kwargs.get('lr', 1e-4)
        self.weight_decay = kwargs.get('weight_decay', 0)
        self.patience = kwargs.get('patience', 5)
        self.eval_batch_size = kwargs.get('eval_batch_size', 256)
        self.warmup_ratio = kwargs.get('warmup_ratio', None)
        self.warmup_steps = kwargs.get('warmup_steps', None)
        self.balance_sample = kwargs.get('balance_sample', False)
        self.load_best_at_last = kwargs.get('load_best_at_last', True)
        self.eval_metric_name = kwargs.get('eval_metric_name', 'auc')
        self.eval_less_is_better = kwargs.get('eval_less_is_better', False)
        self.flag = kwargs.get('flag', 0)
        self.num_workers = kwargs.get('num_workers', 0)
        self.device = kwargs.get('device', None)
        self.ignore_duplicate_cols = kwargs.get('ignore_duplicate_cols', True)
        
        self.optimizer = None
        self.lr_scheduler = None
        self.collate_fn = collate_fn if collate_fn is not None else SupervisedTrainCollator(
            categorical_columns=model.categorical_columns, numerical_columns=model.numerical_columns, binary_columns=model.binary_columns, ignore_duplicate_cols=self.ignore_duplicate_cols)
        self.testloader_list = [
            self._build_dataloader((testset, index), self.eval_batch_size, collator=self.collate_fn, num_workers=self.num_workers, shuffle=False)
            for index, testset in enumerate(self.test_set_list)
        ] if self.test_set_list else None
        self.early_stopping = EarlyStopping(output_dir=self.output_dir, patience=self.patience, verbose=False, less_is_better=self.eval_less_is_better)
        self.num_training_steps = self.get_num_train_steps()
        self.evaluator = Evaluator()
        self.kwargs = kwargs

        self.args = {
            'num_epoch': self.num_epoch,
            'batch_size': self.batch_size,
            'eval_batch_size': self.eval_batch_size,
            'lr': self.lr,
            'weight_decay':self.weight_decay,
            'patience':self.patience,
            'warmup_ratio':self.warmup_ratio,
            'warmup_steps':self.warmup_steps,
            'output_dir':self.output_dir,
            'collate_fn':self.collate_fn,
            'num_workers':self.num_workers,
            'balance_sample':self.balance_sample,
            'load_best_at_last':self.load_best_at_last,
            'ignore_duplicate_cols':self.ignore_duplicate_cols,
            'eval_less_is_better':self.eval_less_is_better,
            'flag':self.flag,
            'num_training_steps': self.num_training_steps,
            'regression_task':self.regression_task,
            'device':self.device,
            'eval_metric_name': self.eval_metric_name
        }

        self.TYPE_TO_SCHEDULER_FUNCTION = {
            'linear': get_linear_schedule_with_warmup,
            'cosine': get_cosine_schedule_with_warmup,
            'cosine_with_restarts': get_cosine_with_hard_restarts_schedule_with_warmup,
            'polynomial': get_polynomial_decay_schedule_with_warmup,
            'constant': get_constant_schedule,
            'constant_with_warmup': get_constant_schedule_with_warmup,
        }

    def make_path(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    def _build_dataloader(self, trainset, batch_size, collator, num_workers, shuffle=True):
        trainloader = DataLoader(
            TrainDataset(trainset),
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            )
        return trainloader
    
    def get_scheduler(self, name):
        name = name.lower()
        schedule_func = self.TYPE_TO_SCHEDULER_FUNCTION[name]
        num_warmup_steps = self.get_warmup_steps(self.num_training_steps)
        num_training_steps=self.num_training_steps

        if name == 'constant':
            return schedule_func(self.optimizer)
        
        if num_warmup_steps is None:
            raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

        if name == 'constant_with_warmup':
            return schedule_func(self.optimizer, num_warmup_steps=num_warmup_steps)
        
        if num_training_steps is None:
            raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

        return schedule_func(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    def create_scheduler(self):
        self.lr_scheduler = self.get_scheduler(
            'cosine',
            num_training_steps=self.num_training_steps,
        )
        return self.lr_scheduler

    def get_num_train_steps(self):
        total_step = 0
        for trainset in self.train_set_list:
            x_train, _ = trainset
            total_step += np.ceil(len(x_train) / self.batch_size)
        total_step *= self.num_epoch
        return total_step

    def get_warmup_steps(self):
        warmup_steps = (
            self.args['warmup_steps'] if self.args['warmup_steps'] is not None else math.ceil(self.num_training_steps * self.args['warmup_ratio'])
        )
        return warmup_steps
    
    def save_model(self, output_path): 
        if output_path is None:
            print('no path assigned for save mode, default saved to ./ckpt/model.pt !')
            output_path = './ckpt'

        if not os.path.exists(output_path): os.makedirs(output_path, exist_ok=True)
        logging.info(f'saving model checkpoint to {output_path}')
        self.model.save(output_path)
        self.collate_fn.save(output_path)

        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(output_path, constants.OPTIMIZER_NAME))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_path, constants.SCHEDULER_NAME))

        if self.args is not None:
            train_args = {}
            for k,v in self.args.items():
                if isinstance(v, int) or isinstance(v, str) or isinstance(v, float):
                    train_args[k] = v
            with open(os.path.join(output_path, constants.TRAINING_ARGS_NAME), 'w', encoding='utf-8') as f:
                f.write(json.dumps(train_args))
            
    def get_parameter_names(self, model, forbidden_layer_types):
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in self.get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        result += list(model._parameters.keys())
        return result

    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = self.get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args['weight_decay'],
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args['lr'])
        if self.use_deepspeed:
            return optimizer_grouped_parameters
    
    def change_device(self, data, dev):
        for key in data:
            if data[key] is not None:
                data[key] = data[key].to(dev)
        return data
    
    def warm_up(self):
        if self.args['warmup_ratio'] is not None or self.args['warmup_steps'] is not None:
            num_train_steps = self.args['num_training_steps']
            logging.info(f'set warmup training in initial {num_train_steps} steps')
            self.create_scheduler(num_train_steps, self.optimizer)
    
    def evaluate(self, model, testloader_list):
        model.eval()
        eval_res_list = []
        pred_all = None
        for dataindex in range(len(testloader_list)):
            y_test, pred_list, loss_list = [], [], []
            for data in testloader_list[dataindex]:
                y_test.append(data[1])
                with torch.no_grad():
                    logits, loss = model(data[0], data[1], table_flag=dataindex)
                if loss is not None:
                    loss_list.append(loss.item())
                if logits is not None:
                    if self.regression_task:
                        pred_list.append(logits.detach().cpu().numpy())
                    elif logits.shape[-1] == 1: # binary classification
                        pred_list.append(logits.sigmoid().detach().cpu().numpy())
                    else: # multi-class classification
                        pred_list.append(torch.softmax(logits,-1).detach().cpu().numpy())
            if len(pred_list)>0:
                pred_all = np.concatenate(pred_list, 0)
                if logits.shape[-1] == 1:
                    pred_all = pred_all.flatten()
            if self.args['eval_metric_name'] == 'val_loss':
                eval_res = np.mean(loss_list)
            else:
                y_test = pd.concat(y_test, 0)
                if self.regression_task:
                    eval_res = self.evaluator.get_eval_metric_fn(self.eval_metric_name)(y_test, pred_all)
                else:
                    eval_res = self.evaluator.get_eval_metric_fn(self.eval_metric_name)(y_test, pred_all, self.model.num_class)
            eval_res_list.append(eval_res)
        return eval_res_list
    
    def create_trainer(self):
        if self.use_deepspeed:
            trainer = Trainer_ds(**self.__dict__)
        else:
            trainer = Trainer(**self.__dict__)
        return trainer.train()

class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainloader_list = [self._build_dataloader((trainset,dataindex), self.batch_size, collator=self.collate_fn, num_workers=self.num_workers) for dataindex,trainset in enumerate(self.train_set_list)]
        self.make_path(self.output_dir)

    def train(self):
        self.create_optimizer()
        self.warm_up()
        start_time = time.time()
        for epoch in trange(self.args['num_epoch'], desc='Epoch'):
            ite = 0
            train_loss_all = 0
            self.model.train()
            for dataindex in range(len(self.trainloader_list)):
                print(f'epoch {epoch} + data {dataindex}')
                for data in self.trainloader_list[dataindex]:
                    self.optimizer.zero_grad()
                    for key in data[0]:
                        if isinstance(data[0][key], list):
                            for i in range(len(data[0][key])):
                                data[0][key][i] = self.change_device(data[0][key][i], self.device)
                        else:
                            data[0] = self.change_device(data[0], self.device)
                        break
                    if data[1] is not None:
                        data[1] = torch.tensor(data[1].values).to(self.device)
                    logits, loss = self.model(data[0], data[1], table_flag=dataindex)
                    loss.backward()
                    self.optimizer.step()
                    train_loss_all += loss.item()
                    ite += 1
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

            if self.test_set_list is not None and epoch%5==0:
                eval_res_list = self.evaluate(self.model, self.testloader_list)
                eval_res = np.mean(eval_res_list)
                print('epoch: {}, test {}: {:.6f}'.format(epoch, self.args['eval_metric_name'], eval_res))
                self.early_stopping(-eval_res, self.model)
                if self.early_stopping.early_stop:
                    print('early stopped')
                    break
                logging.info('epoch: {}, train loss: {:.4f}, test {}: {:.6f}, lr: {:.6f}, spent: {:.1f} secs'.format(epoch, train_loss_all, self.args['eval_metric_name'], eval_res, self.optimizer.param_groups[0]['lr'], time.time()-start_time))
            else:
                logging.info('epoch: {}, train loss: {:.4f}, lr: {:.6f}, spent: {:.1f} secs'.format(epoch, train_loss_all, self.optimizer.param_groups[0]['lr'], time.time()-start_time))

        if os.path.exists(self.output_dir):
            if self.test_set_list is not None:
                # load checkpoints
                logging.info(f'load best at last from {self.output_dir}')
                state_dict = torch.load(os.path.join(self.output_dir, constants.WEIGHTS_NAME), map_location='cpu')
                self.model.load_state_dict(state_dict)
            self.save_model(self.output_dir)

        logging.info('training complete, cost {:.1f} secs.'.format(time.time()-start_time))

class Trainer_ds(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmd_args=None
        
        parameters = self.create_optimizer()
        logging.info(f'deepspeed init start')
        self.model_engine, _, _, _ = deepspeed.initialize(model=self.model, model_parameters=parameters, args=self.cmd_args)
        logging.info(f'deepspeed init finish')
        self.trainloader_list = []
        self.trainsampler_list = []
        for index, trainset in enumerate(self.train_set_list):
            train_data = TrainDataset((trainset, index))
            train_sampler = DistributedSampler(train_data)
            self.trainsampler_list.append(train_sampler)
            trainloader = DataLoader(train_data, collate_fn=self.collate_fn, batch_size=self.batch_size, sampler=train_sampler)
            self.trainloader_list.append(trainloader)
        logging.info(f'deepspeed dataload finish')

        if self.model_engine.local_rank == 0:
            self.make_path(self.output_dir)   
        
    def train(self):
        self.warm_up()
        start_time = time.time()
        for epoch in trange(self.args['num_epoch'], desc='Epoch'):
            ite = 0
            train_loss_all = 0
            for dataindex in range(len(self.trainloader_list)):
                for data in self.trainloader_list[dataindex]:
                    data = list(data)
                    for key in data[0]:
                        if isinstance(data[0][key], list):
                            for i in range(len(data[0][key])):
                                data[0][key][i] = self.change_device(data[0][key][i], self.model_engine.local_rank)
                        else:
                            data[0] = self.change_device(data[0], self.model_engine.local_rank)
                        break
                    if data[1] is not None:
                        data[1] = torch.tensor(data[1].values).to(self.model_engine.local_rank)
                    logits, loss = self.model_engine(data[0], data[1], table_flag=dataindex)
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                    train_loss_all += loss.item()
                    ite += 1
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
            
            if self.test_set_list is not None and (epoch+1)%10==0:
                eval_res_list = self.evaluate(self.model, self.testloader_list)
                eval_res = np.mean(eval_res_list)
                logging.info('epoch: {}, test {}: {:.6f}'.format(epoch+1, self.args['eval_metric_name'], eval_res))
                model_save_path = self.output_dir + '/epoch_' + str(epoch+1) + '_' + 'valloss_' + str(eval_res)
                self.save_model(model_save_path)
                logging.info('epoch: {}, train loss: {:.4f}, test {}: {:.6f}, lr: {:.6f}, spent: {:.1f} secs'.format(epoch+1, train_loss_all, self.args['eval_metric_name'], eval_res, self.optimizer.param_groups[0]['lr'], time.time()-start_time))
            else:
                logging.info('epoch: {}, train loss: {:.4f}, lr: {:.6f}, spent: {:.1f} secs'.format(epoch+1, train_loss_all, self.optimizer.param_groups[0]['lr'], time.time()-start_time))
        logging.info('training complete, cost {:.1f} secs.'.format(time.time()-start_time))