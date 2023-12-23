from collections import defaultdict
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

class Evaluator:  
    def predict(self, 
        clf, 
        x_test,
        y_test=None,
        return_loss=False,
        eval_batch_size=256,
        table_flag=0,
        regression_task=False,
        ):
        clf.eval()
        pred_list, loss_list = [], []
        for i in range(0, len(x_test), eval_batch_size):
            bs_x_test = x_test.iloc[i:i+eval_batch_size]
            with torch.no_grad():
                logits, loss = clf(bs_x_test, y_test, table_flag=table_flag)
            
            if loss is not None:
                loss_list.append(loss.item())
            
            if regression_task:
                pred_list.append(logits.detach().cpu().numpy())
            elif logits.shape[-1] == 1: # binary classification
                pred_list.append(logits.sigmoid().detach().cpu().numpy())
            else: # multi-class classification
                pred_list.append(torch.softmax(logits,-1).detach().cpu().numpy())
        pred_all = np.concatenate(pred_list, 0)
        if logits.shape[-1] == 1:
            pred_all = pred_all.flatten()

        if return_loss:
            avg_loss = np.mean(loss_list)
            return avg_loss
        else:
            return pred_all

    def evaluate(self, ypred, y_test, metric='auc', num_class=2, seed=123, bootstrap=False):
        np.random.seed(seed)
        eval_fn = self.get_eval_metric_fn(metric)
        res_list = []
        stats_dict = defaultdict(list)
        if bootstrap:
            for i in range(10):
                sub_idx = np.random.choice(np.arange(len(ypred)), len(ypred), replace=True)
                sub_ypred = ypred[sub_idx]
                sub_ytest = y_test.iloc[sub_idx]
                try:
                    sub_res = eval_fn(sub_ytest, sub_ypred)
                except ValueError:
                    print('evaluation went wrong!')
                stats_dict[metric].append(sub_res)
            for key in stats_dict.keys():
                stats = stats_dict[key]
                alpha = 0.95
                p = ((1-alpha)/2) * 100
                lower = max(0, np.percentile(stats, p))
                p = (alpha+((1.0-alpha)/2.0)) * 100
                upper = min(1.0, np.percentile(stats, p))
                print('{} {:.2f} mean/interval {:.4f}({:.2f})'.format(key, alpha, (upper+lower)/2, (upper-lower)/2))
                if key == metric: res_list.append((upper+lower)/2)
        else:
            res = eval_fn(y_test, ypred, num_class)
            res_list.append(res)
        return res_list

    def get_eval_metric_fn(self, eval_metric):
        fn_dict = {
            'acc': self.acc_fn,
            'auc': self.auc_fn,
            'mse': self.mse_fn,
            'r2':self.r2_fn,
            'rae':self.rae_fn,
            'val_loss': None,
        }
        return fn_dict[eval_metric]

    def acc_fn(self, y, p, num_class=2):
        if num_class==2:
            y_p = (p >= 0.5).astype(int)
        else:
            y_p = np.argmax(p, -1)
        return accuracy_score(y, y_p)

    def auc_fn(self, y, p, num_class=2):
        if num_class > 2:
            return roc_auc_score(y, p, multi_class='ovo')
        else:
            return roc_auc_score(y, p)

    def mse_fn(self, y, p):
        return mean_squared_error(y, p)

    def r2_fn(self, y, p):
        y = y.values
        return r2_score(y, p)

    def rae_fn(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_true = y_true.values
        up = np.abs(y_pred - y_true).sum()
        down = np.abs(y_true.mean() - y_true).sum()
        score = 1 - up / down
        return score