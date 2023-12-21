import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from tqdm import tqdm

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

class DataLoader:
    def __init__(self, is_classify = False, label_data_path = None, unlabel_data_path = None, 
                 task_data_path = None, is_label = False, limit=10000, task_type = None, task_target = None):
        self.task_type = task_type
        self.auto_feature_type = Feature_type_recognition()
        self.seed = 42
        self.core_size = 10000
        self.is_classify = is_classify
        self.label_data_path = label_data_path
        self.unlabel_data_path = unlabel_data_path
        self.task_data_path = task_data_path
        self.is_label = is_label
        self.limit = limit
        self.task_target=task_target
    
    def create_loader(self, **kwargs):
        if self.task_type == 'pretrain_CL_ds' or self.task_type == 'pretrain_mask_ds' or self.task_type == 'pretrain_mask':
            loader = LoadPretrainAllData(
                is_classify=self.is_classify,
                label_data_path=self.label_data_path,
                unlabel_data_path=self.unlabel_data_path,
                task_data_path=self.task_data_path,
                is_label=self.is_label,
                limit=self.limit)
        elif self.task_type == 'fintune' or self.task_type == 'scratch':
            loader = LoadTaskSingleData(
                task_data_path = self.task_data_path,
                task_target=self.task_target
            )
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        return loader.run()
    
    def read_file(self, file):
        if os.path.exists(file):
            print(f'load from local data dir {file}')
            df = pd.read_csv(file)  
            return df
        else:
            raise RuntimeError('no such data!')

    def devide_feature(self, X, encode_cat=False):
        cat_cols, bin_cols, num_cols = self.auto_feature_type.fit(X)
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
                X[cat_cols] = X[cat_cols].apply(lambda x: x.astype(str).str.lower()) 

        if len(bin_cols) > 0:
            for col in bin_cols: 
                X[col].fillna(X[col].mode()[0], inplace=True)
            X[bin_cols] = X[bin_cols].astype(str).applymap(lambda x: 1 if x.lower() in ['yes','true','1','t'] else 0).values        
            for col in bin_cols:
                if X[col].nunique() <= 1:
                    raise RuntimeError('bin feature process error!')
        return X, cat_cols, bin_cols, num_cols

class LoadPretrainAllData(DataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def process_classification_task(self, df):
        target = df.columns.tolist()[-1]
        value_counts = df[target].value_counts()
        unique_values = value_counts[value_counts == 1].index
        df = df[~df[target].isin(unique_values)]

        y = df[target]
        X = df.drop([target], axis=1)
        if X.shape[0] > self.core_size:
            sample_ratio = self.core_size / X.shape[0]
            X, _, y, _ = train_test_split(X, y, train_size=sample_ratio, random_state=self.seed, stratify=y, shuffle=True)
        y = LabelEncoder().fit_transform(y.values)
        y = pd.Series(y, index=X.index)
        return X, y
    
    def process_regression_task(self, df):
        X = df
        if df.shape[0] > self.core_size:
            X = df.sample(n=self.core_size, random_state=self.seed)
        if self.is_label:
            target = df.columns.tolist()[-1]
            X = X.drop([target], axis=1)
        y = None
        return X, y
    
    def load_pretrain_single_data(self, file):
        df = self.read_file(file)

        if self.is_classify:
            X, y = self.process_classification_task(df)
        else:
            X, y = self.process_regression_task(df)

        all_cols = [col.lower() for col in X.columns.tolist()]
        X.columns = all_cols
        attribute_names = all_cols

        if X.shape[1] > 1000:
            raise RuntimeError('too much features!')
        
        # divide cat/bin/num feature
        X, cat_cols, bin_cols, num_cols = self.devide_feature(X)
        
        # split train/val
        if self.is_classify:
            train_dataset, test_dataset, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed, stratify=y, shuffle=True)
        else:
            train_dataset, test_dataset = train_test_split(X, test_size=0.2, random_state=self.seed, shuffle=True)
            y_train = None
            y_test = None
    
        assert len(attribute_names) == len(cat_cols) + len(bin_cols) + len(num_cols)
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols)))
        return (train_dataset,y_train), (test_dataset,y_test), cat_cols, num_cols, bin_cols

    def process_data_path(self, path, desc, num_col_list, cat_col_list, bin_col_list, train_list, val_list):
        cnt = 0
        data_path_list = os.listdir(path)
        for data in tqdm(data_path_list, desc=desc):
            if data[-3:]=='csv':
                data_path = path + os.sep + data
                try:
                    trainset, valset, cat_cols, num_cols, bin_cols = \
                    self.load_pretrain_single_data(data_path)
                except:
                    print(f"Can not load {data_path}")
                    continue
                cnt += 1
                num_col_list.append(num_cols)
                cat_col_list.append(cat_cols)
                bin_col_list.append(bin_cols)
                train_list.append(trainset)
                val_list.append(valset)

                if len(train_list) > self.limit-1:
                    break
        return num_col_list, cat_col_list, bin_col_list, train_list, val_list, cnt

    def run(self):
        num_col_list, cat_col_list, bin_col_list = [], [], []
        train_list, val_list = [], []
        label_cnt = 0
        unlabel_cnt = 0

        if self.label_data_path is None and self.unlabel_data_path is None:
            raise Exception("Did not input data!")

        if self.label_data_path is not None:
            num_col_list, cat_col_list, bin_col_list, train_list, val_list, label_cnt = \
            self.process_data_path(
                path=self.label_data_path, 
                desc='load label data', 
                num_col_list=num_col_list, 
                cat_col_list=cat_col_list, 
                bin_col_list=bin_col_list, 
                train_list=train_list, 
                val_list=val_list
            )
        
        if self.unlabel_data_path is not None:
            num_col_list, cat_col_list, bin_col_list, train_list, val_list, unlabel_cnt = \
            self.process_data_path(
                path=self.unlabel_data_path, 
                desc='load unlabel data', 
                num_col_list=num_col_list, 
                cat_col_list=cat_col_list, 
                bin_col_list=bin_col_list, 
                train_list=train_list, 
                val_list=val_list
            )

        print(f'useful tab ===========> {len(train_list)}')
        print(f'label tab ===========> {label_cnt}')
        print(f'unlabel tab ===========> {unlabel_cnt}')
        return train_list, val_list, cat_col_list, num_col_list, bin_col_list

class LoadTaskSingleData(DataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encode_cat=False

    def run(self):
        df = self.read_file(self.task_data_path)

        # Delete the sample whose label count is 1 or label is nan
        count_num = list(df[self.task_target].value_counts())
        count_value = list(df[self.task_target].value_counts().index)
        delete_index = []
        for i,cnt in enumerate(count_num):
            if cnt <= 1:
                index = df.loc[df[self.task_target]==count_value[i]].index.to_list()
                delete_index.extend(index)
        df.drop(delete_index, axis=0, inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(axis=0, subset=[self.task_target], inplace=True)
        y = df[self.task_target]
        X = df.drop([self.task_target], axis=1)

        all_cols = [col.lower() for col in X.columns.tolist()]
        X.columns = all_cols
        attribute_names = all_cols
        
        # encode target label
        y = LabelEncoder().fit_transform(y.values)
        y = pd.Series(y, index=X.index)

        # divide cat/bin/num feature
        X, cat_cols, bin_cols, num_cols = self.devide_feature(X, self.encode_cat)
        X = X[bin_cols + num_cols + cat_cols]

        assert len(attribute_names) == len(cat_cols) + len(bin_cols) + len(num_cols)
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y==1).sum()/len(y)))
        return X, y, cat_cols, num_cols, bin_cols


