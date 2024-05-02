#将csv文件生成dataset

import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import os
import math

current_path=os.getcwd()
csv_applib_path=os.path.join(current_path,'original dataset','app_lib.csv')
csv_app_path=os.path.join(current_path,'original dataset','app.csv')
csv_lib_path=os.path.join(current_path,'original dataset','lib.csv')



class app(Dataset):
    def __init__(self, app_annotations_file,lib_annotation_file, interaction,transform=None):
        self.app_labels = pd.read_csv(app_annotations_file,header=None)
        self.lib_labels=pd.read_csv(lib_annotation_file,header=None)
        self.interaction=interaction
        self.transform=transform
    
    
    def load(self):
        #['app','lib']
        df=pd.read_csv(self.interaction,sep=',',names=['app','lib'],usecols=['app','lib'])
        return df

    def __len__(self):
        return len(self.interaction)
    
    def __getitem__(self, idx):
        app=self.interaction.iloc[idx,0]
        lib=self.interaction.iloc[idx,1]
        # if self.transform:
        #     app=self.app_labels.iloc[]
        return app, lib
    
    def app_size(self):
        return len(self.app_labels[0].unique())

    def lib_size(self):
        return len(self.lib_labels[0].unique())
    
    def app_mapping(self):
        dict={x:i for i,x in enumerate(self.app_labels[1].unique())}
        return dict
    
    def lib_mapping(self):
        dict={x:i for i,x in enumerate(self.lib_labels[1].unique())}
        return dict


def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict

def create_pair(user_list):
    pair = []
    for user, item_list in enumerate(user_list):
        pair.extend([(user, item) for item in item_list])
    return pair

def create_app_list(df, app_size):
    app_list = [list() for u in range(app_size)]
    for row in df.itertuples():
        app_list[row[0]].append(row[1])
    return app_list

def split_train_test(df, user_size, test_size=0.2, time_order=False):
    """Split a dataset into `train_user_list` and `test_user_list`.
    Because it needs `user_list` for splitting dataset as `time_order` is set,
    Returning `user_list` data structure will be a good choice."""
    # TODO: Handle duplicated items
    if not time_order:
        test_idx = np.random.choice(len(df), size=int(len(df) * test_size))
        train_idx = list(set(range(len(df))) - set(test_idx))
        test_df = df.loc[test_idx].reset_index(drop=True)
        train_df = df.loc[train_idx].reset_index(drop=True)
        test_user_list = create_app_list(test_df, app_size)
        train_user_list = create_app_list(train_df, app_size)
    else:
        total_user_list = create_app_list(df, app_size)
        train_user_list = [None] * len(user_list)
        test_user_list = [None] * len(user_list)
        for user, item_list in enumerate(total_user_list):
            # Choose latest item
            item_list = sorted(item_list, key=lambda x: x[0])
            # Split item
            test_item = item_list[math.ceil(len(item_list) * (1 - test_size)):]
            train_item = item_list[:math.ceil(len(item_list) * (1 - test_size))]
            # Register to each user list
            test_user_list[user] = test_item
            train_user_list[user] = train_item
    # Remove time
    test_user_list = [list(map(lambda x: x[1], l)) for l in test_user_list]
    train_user_list = [list(map(lambda x: x[1], l)) for l in train_user_list]
    return train_user_list, test_user_list

dataset=app(csv_app_path,csv_lib_path,csv_applib_path)

# train_size=int(len(dataset) * 0.8)
# validate_size=0
# test_size=len(dataset) - validate_size - train_size
# train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(dataset,
#                                                         [train_size, validate_size, test_size])

# train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True)
# test_dataloader=DataLoader(test_dataset,batch_size=64, shuffle=True)
# app,lib=next(iter(train_dataloader))
# print(len(train_dataset))
# print(len(test_dataset))
# print(app.size())
# print(lib.size())

df, app_mapping = convert_unique_idx(dataset.load(), 0)
df, lib_mapping = convert_unique_idx(df, 1)
print('Complete assigning unique index to app and lib')

app_size = len(df[0].unique())
lib_size = len(df[1].unique())
print('app size:'+str(app_size))
print('lib_size:'+str(lib_size))

train_app_list,test_app_list=split_train_test(df,app_size,test_size=0.2)
train_pair = create_pair(train_app_list)
print('Complete spliting items for training and testing')

appRec_dataset={'app_size':app_size,'lib_size':lib_size,
         'app_mapping':app_mapping,'lib_mapping':lib_mapping,
         'train_app_list': train_app_list, 'test_app_list': test_app_list,
        'train_pair': train_pair}

pickle_dirname=os.path.join(current_path,'preprocessed')
os.makedirs(pickle_dirname,exist_ok=True)
with open(os.path.join(pickle_dirname,'appRec.pickle'),'wb') as f:
    pickle.dump(appRec_dataset,f,protocol=pickle.HIGHEST_PROTOCOL)