import pandas as pd
import os
from tqdm import tqdm
from shutil import copy

from dataset_openml import check_col_name_meaning

openml_yc_path = '/home/yech/openml_data/data_info_v1.csv'
openml_meaning_path = '/home/yech/openml_meaning/data_info_v1.csv'
openml_yc_dir = '/home/yech/openml_data/data_v1'
openml_meaning_dir = '/home/yech/openml_meaning/data_v1'


def merge():
    openml_yc_datainfo = pd.read_csv(openml_yc_path)
    openml_meaning_datainfo = pd.read_csv(openml_meaning_path)
    meaning_id_set = set(openml_meaning_datainfo['id_openml'].astype('int64').values.tolist())

    cnt = 0
    with tqdm(total=openml_yc_datainfo.shape[0], desc='traverse yc data') as pbar:
        for index, table_info in openml_yc_datainfo.iterrows():
            id = int(table_info['id_openml'])
            if id not in meaning_id_set:
                table_file = openml_yc_dir + os.sep + table_info['file_name']
                if check_col_name_meaning(table_file, table_info['target']):
                    cnt += 1
                    # copy file
                    # copy(table_file, openml_meaning_dir)
                    # concat data_info
                    openml_meaning_datainfo.loc[openml_meaning_datainfo.shape[0]] = table_info.values.tolist()
            pbar.update(1)
    
    # openml_meaning_datainfo.to_csv(openml_meaning_path, index=False)
    print(f'add table number : {cnt}')

def same_mode():
    openml_yc_datainfo = pd.read_csv(openml_yc_path)
    openml_meaning_datainfo = pd.read_csv(openml_meaning_path)
    meaning_id_set = set(openml_meaning_datainfo['id_openml'].astype('int64').values.tolist())

    print(openml_meaning_datainfo.loc[1, 'mode'])

    all_cnt = 0
    error_cnt = 0
    openml_meaning_datainfo[['id_openml']] = openml_meaning_datainfo[['id_openml']].astype(int)
    with tqdm(total=openml_yc_datainfo.shape[0], desc='traverse yc data') as pbar:
        for index, table_info in openml_yc_datainfo.iterrows():
            id = int(table_info['id_openml'])
            if id in meaning_id_set:
                all_cnt += 1
                data_index = openml_meaning_datainfo[openml_meaning_datainfo['id_openml']==id].index.to_list()
                assert len(data_index)==1
                meaning_table_info = openml_meaning_datainfo.loc[data_index[0]]
                if table_info['mode'] != meaning_table_info['mode']:
                    openml_meaning_datainfo.loc[data_index[0], 'mode'] = table_info['mode']
                    error_cnt += 1
                    # print(meaning_table_info['file_name'])
            pbar.update(1)

    print(all_cnt, error_cnt)
    # openml_meaning_datainfo.to_csv(openml_meaning_path, index=False)


if __name__ == '__main__':
    same_mode()
