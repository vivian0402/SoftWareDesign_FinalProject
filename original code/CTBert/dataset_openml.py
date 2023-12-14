import os
import pdb
import datetime
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import openml
from loguru import logger


class Feature_type_recognition():
    def __init__(self):
        self.df = None
    
    def detect_TIMESTAMP(self, col):
        try:
            ts_min = int(float(self.df.loc[~(self.df[col] == '') & (self.df[col].notnull()), col].min()))
            ts_max = int(float(self.df.loc[~(self.df[col] == '') & (self.df[col].notnull()), col].max()))
            datetime_min = datetime.datetime.utcfromtimestamp(ts_min).strftime('%Y-%m-%d %H:%M:%S')
            datetime_max = datetime.datetime.utcfromtimestamp(ts_max).strftime('%Y-%m-%d %H:%M:%S')
            if datetime_min > '2000-01-01 00:00:01' and datetime_max < '2030-01-01 00:00:01' and datetime_max > datetime_min:
                return True
        except:
            return False

    def detect_DATETIME(self, col):
        is_DATETIME = False
        if self.df[col].dtypes == object or str(self.df[col].dtypes) == 'category':
            is_DATETIME = True
            try:
                pd.to_datetime(self.df[col])
            except:
                is_DATETIME = False
        return is_DATETIME
    
    def get_data_type(self, col):
        if self.detect_DATETIME(col):
            return 'cat'
        if self.detect_TIMESTAMP(col):
            return 'cat'
        if self.df[col].dtypes == object or self.df[col].dtypes == bool or str(self.df[col].dtypes) == 'category':
            # if self.df[col].nunique() == 2:
            #     return 'bin'
            return 'cat'
        if 'int' in str(self.df[col].dtype) or 'float' in str(self.df[col].dtype):
            if self.df[col].nunique() < 15:
                return 'cat'
            return 'num'

    def fit(self, df):
        self.df = df
        self.num = []
        self.cat = []
        self.bin = []
        for col in self.df.columns:
            cur_type = self.get_data_type(col)
            if (cur_type == 'num'):
                self.num.append(col)
            elif (cur_type == 'cat'):
                self.cat.append(col)
            elif (cur_type == 'bin'):
                self.bin.append(col)
            else:
                raise RuntimeError('error feature type!')
        return self.cat, self.bin, self.num


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


def load_openml_data(dataset_config=None, encode_cat=False, seed=42, limit=400):
    num_col_list, cat_col_list, bin_col_list = [], [], []
    all_list, train_list, val_list= [], [], []
    auto_feature_type = Feature_type_recognition()

    # data_info = pd.read_csv('/home/gslu/openml_data/data_info_v1.csv')
    data_info = pd.read_csv('/home/gslu/openml_meaning/data_info_v1.csv')

    error_cnt = 0
    big_cnt = 0
    max_len = 0

    good_col_name = []
    bad_col_name = []
    use_col_name = []
    task_tables = []

    # with open('/home/gslu/tableGPT_yc/task_table_meaning.txt','r') as f:
    #     for x in f.readlines():
    #         task_tables.append(x.strip())

    with tqdm(total=data_info.shape[0], desc='traverse data') as pbar:
        for index, table_info in data_info.iterrows():
            if table_info['mode'] == 'regression':
                pbar.update(1)
                continue

            if 'BNG' in table_info['file_name']:
                pbar.update(1)
                continue

            # if table_info['file_name'] in task_tables:
            #     pbar.update(1)
            #     continue
            
            if table_info['file_name']=='1686_CIFAR_10.csv' or table_info['file_name']=='1687_Internet-Advertisements.csv':
                pbar.update(1)
                continue

            table_file = table_info['file_name']
            # table_file = '/home/gslu/openml_data/data_v1'+ os.sep + table_file
            table_file = '/home/gslu/openml_meaning/data_v1'+ os.sep + table_file

            try:
                df = pd.read_csv(table_file)
                memory_use = df.memory_usage().sum()/(1024**2)
                if memory_use >= 10:
                    big_cnt += 1
                    pbar.update(1)
                    continue
            except:
                pbar.update(1)
                continue

            try:
                if check_col_name_meaning(table_file, table_info['target']):
                    good_col_name.append(str(pd.read_csv(table_file).columns.values.tolist())+' = '+table_info['file_name']+'\n')
                else:
                    bad_col_name.append(str(pd.read_csv(table_file).columns.values.tolist())+' = '+table_info['file_name']+'\n')
                    pbar.update(1)
                    continue
            except:
                pbar.update(1)
                continue

            try:
                allset, trainset, valset, cat_cols, num_cols, bin_cols = \
                load_single_data(table_file, target=table_info['target'], auto_feature_type=auto_feature_type, dataset_config=dataset_config, encode_cat=encode_cat, seed=seed)
            except:
                error_cnt += 1
                pbar.update(1)
                continue

            max_len = max(max_len, allset[0].shape[0])

            num_col_list.append(num_cols)
            cat_col_list.append(cat_cols)
            bin_col_list.append(bin_cols)
            all_list.append(allset)
            train_list.append(trainset)
            val_list.append(valset)

            tmp = pd.read_csv(table_file)
            # if len(tmp[table_info['target']].unique()) == 2:
            use_col_name.append(str(tmp.columns.values.tolist())+' = '+table_info['file_name']+'\n')
            # 快速测试用的，只用前100个数据集
            if len(all_list) > limit:
                break

            pbar.update(1)

        print(f'useful tab ===========> {len(all_list)}')
        print(f'error tab ===========> {error_cnt}')
        print(f'big tab ===========> {big_cnt}')
        print(f'max_len ===========> {max_len}')

        with open('./good_col_name.txt','w') as file:
            for col_names in good_col_name:
                file.write(col_names)
        with open('./bad_col_name.txt','w') as file:
            for col_names in bad_col_name:
                file.write(col_names)
        with open('./use_col_name.txt','w') as file:
            for col_names in use_col_name:
                file.write(col_names)

        return all_list, train_list, val_list, cat_col_list, num_col_list, bin_col_list

def load_single_data(table_file, target, auto_feature_type, dataset_config=None, encode_cat=False, seed=123):
    if os.path.exists(table_file):
        print(f'load from local data dir {table_file}')
        df = pd.read_csv(table_file)

        # Delete the sample whose label count is 1 or label is nan
        count_num = list(df[target].value_counts())
        count_value = list(df[target].value_counts().index)
        delete_index = []
        for i,cnt in enumerate(count_num):
            if cnt <= 1:
                index = df.loc[df[target]==count_value[i]].index.to_list()
                delete_index.extend(index)
        df.drop(delete_index, axis=0, inplace=True)
        df.dropna(axis=0, subset=[target], inplace=True)

        y = df[target]
        X = df.drop([target], axis=1)
        all_cols = [col.lower() for col in X.columns.tolist()]
        X.columns = all_cols
        attribute_names = all_cols

        # divide cat/bin/num feature
        cat_cols, bin_cols, num_cols = auto_feature_type.fit(X)
        
        # encode target label
        y = LabelEncoder().fit_transform(y.values)
        y = pd.Series(y, index=X.index)

    # start processing features
    # process num
    if len(num_cols) > 0:
        for col in num_cols: 
            X[col].fillna(X[col].mode()[0], inplace=True)
        X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        for col in cat_cols: 
            X[col].fillna(X[col].mode()[0], inplace=True)
        # process cate
        if encode_cat:
            X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])
        else:
            X[cat_cols] = X[cat_cols].astype(str)

    if len(bin_cols) > 0:
        for col in bin_cols: 
            X[col].fillna(X[col].mode()[0], inplace=True)
        X[bin_cols] = X[bin_cols].astype(str).applymap(lambda x: 1 if x.lower() in ['yes','true','1','t'] else 0).values        
        for col in bin_cols:
            if X[col].nunique() <= 1:
                raise RuntimeError('bin feature process error!')
    
    X = X[bin_cols + num_cols + cat_cols]
    if len(X.columns) < 3:
        raise RuntimeError('column number is too few!')

    # split train/val
    train_dataset, test_dataset, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y, shuffle=True)

    assert len(attribute_names) == len(cat_cols) + len(bin_cols) + len(num_cols)
    print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y==1).sum()/len(y)))
    return (X,y), (train_dataset,y_train), (test_dataset, y_test), cat_cols, num_cols, bin_cols

def get_multiview(data, cat_cols):
    X, y = data
    X_num = X.copy(deep=True)
    X_num[cat_cols] = OrdinalEncoder().fit_transform(X_num[cat_cols])

    return (X, X_num, y)




def load_one_openml_data(table_file):
    # data_info = pd.read_csv('/home/gslu/openml_data/data_info_v1.csv')
    data_info = pd.read_csv('/home/gslu/openml_meaning/data_info_v1.csv')
    data_index = data_info[data_info['file_name']==table_file].index.to_list()
    assert len(data_index)==1
    table_info = data_info.loc[data_index[0]]
    # table_file = '/home/gslu/openml_data/data_v1'+ os.sep + table_file
    table_file = '/home/gslu/openml_meaning/data_v1'+ os.sep + table_file
    auto_feature_type = Feature_type_recognition()
    return load_single_data_all(table_file=table_file, target=table_info['target'], auto_feature_type=auto_feature_type)


def load_single_data_all(table_file, target, auto_feature_type, dataset_config=None, encode_cat=False, seed=123):
    if os.path.exists(table_file):
        print(f'load from local data dir {table_file}')
        df = pd.read_csv(table_file)

        # Delete the sample whose label count is 1 or label is nan
        count_num = list(df[target].value_counts())
        count_value = list(df[target].value_counts().index)
        delete_index = []
        for i,cnt in enumerate(count_num):
            if cnt <= 1:
                index = df.loc[df[target]==count_value[i]].index.to_list()
                delete_index.extend(index)
        df.drop(delete_index, axis=0, inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(axis=0, subset=[target], inplace=True)

        y = df[target]
        X = df.drop([target], axis=1)
        all_cols = [col.lower() for col in X.columns.tolist()]
        X.columns = all_cols
        attribute_names = all_cols

        # divide cat/bin/num feature
        cat_cols, bin_cols, num_cols = auto_feature_type.fit(X)
        
        # encode target label
        y = LabelEncoder().fit_transform(y.values)
        y = pd.Series(y, index=X.index)
    else:
        raise RuntimeError('no such data file!')

    # start processing features
    # process num
    if len(num_cols) > 0:
        for col in num_cols: 
            X[col].fillna(X[col].mode()[0], inplace=True)
        X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        for col in cat_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        # process cate
        if encode_cat:
            X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])
        else:
            # X[cat_cols] = X[cat_cols].astype(str).str.lower()
            X[cat_cols] = X[cat_cols].apply(lambda x: x.astype(str).str.lower()) 

    if len(bin_cols) > 0:
        for col in bin_cols: 
            X[col].fillna(X[col].mode()[0], inplace=True)
        X[bin_cols] = X[bin_cols].astype(str).applymap(lambda x: 1 if x.lower() in ['yes','true','1','t'] else 0).values        
        for col in bin_cols:
            if X[col].nunique() <= 1:
                raise RuntimeError('bin feature process error!')
    
    X = X[bin_cols + num_cols + cat_cols]

    # split train/val
    # train_dataset, test_dataset, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y, shuffle=True)

    assert len(attribute_names) == len(cat_cols) + len(bin_cols) + len(num_cols)
    print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y==1).sum()/len(y)))
    return X, y, cat_cols, num_cols, bin_cols


def load_regression_data(table_file, encode_cat=False):
    if os.path.exists(table_file):
        print(f'load from local data dir {table_file}')
        df = pd.read_csv(table_file)
        target = df.columns.tolist()[-1]

        # Delete the sample whose label count is 1 or label is nan
        y = df[target]
        X = df.drop([target], axis=1)
        all_cols = [col.lower() for col in X.columns.tolist()]
        X.columns = all_cols
        attribute_names = all_cols

        # divide cat/bin/num feature
        auto_feature_type = Feature_type_recognition()
        cat_cols, bin_cols, num_cols = auto_feature_type.fit(X)
        
        # encode target label
        y = pd.Series(y, index=X.index)
    else:
        raise RuntimeError('no such data file!')

    # start processing features
    # process num
    if len(num_cols) > 0:
        for col in num_cols: 
            X[col].fillna(X[col].mode()[0], inplace=True)
        X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        for col in cat_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        # process cate
        if encode_cat:
            X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])
        else:
            X[cat_cols] = X[cat_cols].astype(str)

    if len(bin_cols) > 0:
        for col in bin_cols: 
            X[col].fillna(X[col].mode()[0], inplace=True)
        X[bin_cols] = X[bin_cols].astype(str).applymap(lambda x: 1 if x.lower() in ['yes','true','1','t'] else 0).values        
        for col in bin_cols:
            if X[col].nunique() <= 1:
                raise RuntimeError('bin feature process error!')
    
    X = X[bin_cols + num_cols + cat_cols]

    # split train/val
    # train_dataset, test_dataset, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y, shuffle=True)

    assert len(attribute_names) == len(cat_cols) + len(bin_cols) + len(num_cols)
    print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y==1).sum()/len(y)))
    return X, y, cat_cols, num_cols, bin_cols



if __name__ == '__main__':
    load_openml_data(limit=1000)