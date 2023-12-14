import os
import pdb
import datetime
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

class Feature_type_recognition():
    def __init__(self):
        self.df = None
    
    def get_data_type(self, col):
        if 'std' in self.df[col].describe():
            if self.df[col].nunique() < 15:
                return 'cat'
            return 'num'
        else:
            return 'cat'

    def fit(self, df):
        self.df = df.infer_objects()
        self.num = []
        self.cat = []
        self.bin = []
        for col in self.df.columns:
            cur_type = self.get_data_type(col)
            if (cur_type == 'num'):
                self.num.append(col)
            elif (cur_type == 'cat'):
                self.cat.append(col)
            else:
                raise RuntimeError('error feature type!')
        return self.cat, self.bin, self.num
    
    def check_class(self, data_path):
        self.df = pd.read_csv(data_path)
        
        target_type = self.get_data_type(self.df.columns.tolist()[-1])
        if target_type == 'cat':
            return True
        else:
            return False

def check_col_name_meaning(table_file, target, threshold=2):
    df = pd.read_csv(table_file)
    col_names = df.columns.values.tolist()
    col_names.remove(target)
    res = False
    good_cnt = 0
    one_cnt = 0
    for name in col_names:
        if len(name) <= 1:
            one_cnt += 1
            if one_cnt*2 >= len(col_names):
                return False
        if not name[-1].isdigit():
            good_cnt += 1
            if good_cnt>threshold or good_cnt==len(col_names):
                res = True
    return res

def get_col_type(col):
    if 'std' in col.describe():
        if col.nunique() < 15:
            return 'cat'
        return 'num'
    else:
        return 'cat'

def load_single_classify_data(table_file, auto_feature_type, seed=42, coresize=10000, fe_limit=50):
    if os.path.exists(table_file):
        print(f'load from local data dir {table_file}')
        df = pd.read_csv(table_file)

        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        is_classify = True if get_col_type(df[df.columns.tolist()[-1]]) == 'cat' else False

        if is_classify:
            target = df.columns.tolist()[-1]

            value_counts = df[target].value_counts()
            unique_values = value_counts[value_counts == 1].index
            df = df[~df[target].isin(unique_values)]
            y = df[target]
            # X = df.drop([target], axis=1)

            if (X.shape[0] > coresize):
                sample_ratio = (coresize / X.shape[0])
                X, _, y, _ = train_test_split(X, y, train_size=sample_ratio, random_state=seed, stratify=y, shuffle=True)

            # if (X.shape[1] > 50):
            #     forest = RandomForestClassifier()
            #     forest.fit(X, y)
            #     importances = forest.feature_importances_
            #     cols = X.columns.tolist()
            #     indices = np.argsort(importances)[::-1]
            #     final_fe = []
            #     for ix in indices[:fe_limit]:
            #         final_fe.append(cols[ix])
            #     # print(final_fe)
            #     X = X[final_fe]

            y = LabelEncoder().fit_transform(y.values)
            y = pd.Series(y, index=X.index)
        else:
            raise Exception("data must be classify")

        all_cols = [col.lower() for col in X.columns.tolist()]
        X.columns = all_cols
        attribute_names = all_cols

        # divide cat/bin/num feature
        cat_cols, bin_cols, num_cols = auto_feature_type.fit(X)
        if len(cat_cols) > 0:
            X[cat_cols] = X[cat_cols].apply(lambda x: x.astype(str).str.lower())
            # X[cat_cols] = X[cat_cols].astype(str)
        X = X[bin_cols + num_cols + cat_cols]

        # split train/val
        if is_classify:
            train_dataset, test_dataset, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y, shuffle=True)
        else:
            raise Exception("data must be classify")
        
        assert len(attribute_names) == len(cat_cols) + len(bin_cols) + len(num_cols)
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y == 1).sum() / len(y)))
        return (X, y), (train_dataset, y_train), (test_dataset, y_test), cat_cols, num_cols, bin_cols
    else:
        raise RuntimeError('no such data!')


def load_single_data(table_file, auto_feature_type, is_label=False, is_classify=False, seed=42, core_size=10000):
    if os.path.exists(table_file):
        print(f'load from local data dir {table_file}')
        df = pd.read_csv(table_file)

        if is_classify:
            target = df.columns.tolist()[-1]

            value_counts = df[target].value_counts()
            unique_values = value_counts[value_counts == 1].index
            df = df[~df[target].isin(unique_values)]

            y = df[target]
            X = df.drop([target], axis=1)

            if (X.shape[0] > core_size):
                sample_ratio = (core_size / X.shape[0])
                X, _, y, _ = train_test_split(X, y, train_size=sample_ratio, random_state=seed, stratify=y, shuffle=True)

            y = LabelEncoder().fit_transform(y.values)
            y = pd.Series(y, index=X.index)
        else:
            X = df
            if df.shape[0] > core_size:
                X = df.sample(n=core_size, random_state=seed)
            if is_label:
                target = df.columns.tolist()[-1]
                X = X.drop([target], axis=1)
            y = None

        all_cols = [col.lower() for col in X.columns.tolist()]
        X.columns = all_cols
        attribute_names = all_cols

        if X.shape[1] > 1000:
            raise RuntimeError('too much features!')
        
        # divide cat/bin/num feature
        cat_cols, bin_cols, num_cols = auto_feature_type.fit(X)
        if len(cat_cols) > 0:
            for col in cat_cols: 
                X[col].fillna(X[col].mode()[0], inplace=True)       
            X[cat_cols] = X[cat_cols].apply(lambda x: x.astype(str).str.lower())
        if len(num_cols) > 0:
            for col in num_cols: 
                X[col].fillna(X[col].mode()[0], inplace=True)       
            X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])
        
        # split train/val
        if is_classify:
            train_dataset, test_dataset, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y, shuffle=True)
        else:
            train_dataset, test_dataset = train_test_split(X, test_size=0.2, random_state=seed, shuffle=True)
            y_train = None
            y_test = None
    
        assert len(attribute_names) == len(cat_cols) + len(bin_cols) + len(num_cols)
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols)))
        return (train_dataset,y_train), (test_dataset,y_test), cat_cols, num_cols, bin_cols
    else:
        raise RuntimeError('no such data!')

def load_all_data(label_data_path=None, 
                  unlabel_data_path=None, 
                  seed=42, limit=10000, core_size=10000):
    
    num_col_list, cat_col_list, bin_col_list = [], [], []
    train_list, val_list = [], []
    auto_feature_type = Feature_type_recognition()

    label_data_list = os.listdir(label_data_path)
    unlabel_data_list = os.listdir(unlabel_data_path)

    label_cnt = 0
    unlabel_cnt = 0
    
    for data in tqdm(label_data_list, desc='load label data'):
        if data[-3:]=='csv':
            data_path = label_data_path + os.sep + data

            try:
                trainset, valset, cat_cols, num_cols, bin_cols = \
                load_single_data(data_path, auto_feature_type=auto_feature_type, is_label=False, seed=seed, core_size=core_size)
            except:
                continue

            label_cnt += 1
            num_col_list.append(num_cols)
            cat_col_list.append(cat_cols)
            bin_col_list.append(bin_cols)
            train_list.append(trainset)
            val_list.append(valset)

            # 快速测试用的，只用前100个数据集
            if len(train_list) > limit-1:
                break
    

    # for data in tqdm(unlabel_data_list, desc='load unlabel data'):
    #     if data[-3:]=='csv':
    #         data_path = unlabel_data_path + os.sep + data
        
    #         try:
    #             trainset, valset, cat_cols, num_cols, bin_cols = \
    #             load_single_data(data_path, auto_feature_type=auto_feature_type, seed=seed, core_size=core_size)
    #         except:
    #             continue

    #         unlabel_cnt += 1
    #         num_col_list.append(num_cols)
    #         cat_col_list.append(cat_cols)
    #         bin_col_list.append(bin_cols)
    #         train_list.append(trainset)
    #         val_list.append(valset)
            
    #         # 快速测试用的，只用前100个数据集
    #         if len(train_list) >= limit-1:
    #             break

    print(f'useful tab ===========> {len(train_list)}')
    print(f'label tab ===========> {label_cnt}')
    print(f'unlabel tab ===========> {unlabel_cnt}')
    
    return train_list, val_list, cat_col_list, num_col_list, bin_col_list


def load_labeled_classify_data(label_data_path=None, 
                               seed=42, limit=10000):
    
    num_col_list, cat_col_list, bin_col_list = [], [], []
    train_list, val_list= [], []
    auto_feature_type = Feature_type_recognition()

    label_data_list = os.listdir(label_data_path)
    
    for data in tqdm(label_data_list, desc='load label data'):
        if data[-3:] == 'csv':
            data_path = label_data_path + os.sep + data
            if not auto_feature_type.check_class(data_path):
                continue
            
            try:
                trainset, valset, cat_cols, num_cols, bin_cols = \
                load_single_data(data_path, auto_feature_type=auto_feature_type, is_classify=True, seed=seed)
            except:
                continue

            # trainset, valset, cat_cols, num_cols, bin_cols = \
            #     load_single_data(data_path, auto_feature_type=auto_feature_type, is_classify=True, seed=seed)

            num_col_list.append(num_cols)
            cat_col_list.append(cat_cols)
            bin_col_list.append(bin_cols)
            train_list.append(trainset)
            val_list.append(valset)

            # 快速测试用的，只用前100个数据集
            if len(train_list) >= limit-1:
                break

    print(f'useful tab ===========> {len(train_list)}')
    return train_list, val_list, cat_col_list, num_col_list, bin_col_list


if __name__ == '__main__':
    # load_labeled_classify_data('/home/gslu/small_clean_pretrain_data/clean_labeled_dataset')
    
    trainset, valset, cat_cols, num_cols, bin_cols = load_all_data(
        label_data_path='/home/gslu/small_clean_pretrain_data/clean_labeled_dataset',
        unlabel_data_path='/home/gslu/small_clean_pretrain_data/clean_unlabeled_dataset',
        limit=2000,
        core_size=10000,
    )

