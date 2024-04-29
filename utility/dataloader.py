import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import os

current_path=os.getcwd()
csv_applib_path=os.path.join(current_path,'original dataset','app_lib.csv')
csv_app_path=os.path.join(current_path,'original dataset','app.csv')
csv_lib_path=os.path.join(current_path,'original dataset','lib.csv')



class app(Dataset):
    def __init__(self, app_annotations_file,lib_annotation_file, interaction,transform=None):
        self.app_labels = pd.read_csv(app_annotations_file,header=None)
        self.lib_labels=pd.read_csv(lib_annotation_file,header=None)
        self.interaction=pd.read_csv(interaction,header=None)
        self.transform=transform

    def __len__(self):
        return len(self.interaction)
    
    def __getitem__(self, idx):
        app=self.interaction.iloc[idx,0]
        lib=self.interaction.iloc[idx,1]
        # if self.transform:
        #     app=self.app_labels.iloc[]
        return app, lib
    



dataset=app(csv_app_path,csv_lib_path,csv_applib_path)
train_size=int(len(dataset) * 0.8)
validate_size=0
test_size=len(dataset) - validate_size - train_size
train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                        [train_size, validate_size, test_size])

train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=64, shuffle=True)
app,lib=next(iter(train_dataloader))
print(len(train_dataset))
print(len(test_dataset))
print(app.size())
print(lib.size())
