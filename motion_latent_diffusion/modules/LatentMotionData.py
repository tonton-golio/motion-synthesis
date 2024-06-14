import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from utils import print_header
# transform
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

import numpy as np

class NoScaler:
    def fit(self, data):
        return self
    def transform(self, data):
        return data
    def fit_transform(self, data):
        return data
    def inverse_transform(self, data):
        return data

class LatentMotionData(pl.LightningDataModule):
    def __init__(self, z_train, z_val, z_test, texts_train, texts_val, texts_test, file_num_train, file_num_val, file_num_test, batch_size=32, **kwargs):
        super().__init__()
        

        self.batch_size = batch_size
        self.latent_dim = z_train.shape[-1]
        self.z_limit = kwargs.get('z_limit', 5.0)
        self.scale = kwargs.get('scale', False)

        self.z_train, self.texts_train, self.file_num_train = self.prepare_data_(z_train, texts_train, file_num_train)
        self.z_val, self.texts_val, self.file_num_val = self.prepare_data_(z_val, texts_val, file_num_val)
        self.z_test, self.texts_test, self.file_num_test = self.prepare_data_(z_test, texts_test, file_num_test)

        self.scaler = self.fit_scaler(self.z_train) if self.scale else NoScaler()
        self.z_train = self.transform_data(self.z_train, self.scaler)
        self.z_val = self.transform_data(self.z_val, self.scaler)
        self.z_test = self.transform_data(self.z_test, self.scaler)

        self.z_train = torch.tensor(self.z_train).float().to('mps')
        self.z_val = torch.tensor(self.z_val).float().to('mps')
        self.z_test = torch.tensor(self.z_test).float().to('mps')

        self.file_num_train = torch.tensor(self.file_num_train).long()
        self.file_num_val = torch.tensor(self.file_num_val).long()
        self.file_num_test = torch.tensor(self.file_num_test).long()

        for z in [self.z_train, self.z_val, self.z_test]:
            self.print_stats(z)

        print('z_train shape:', self.z_train.shape)
        print('texts_train shape:', self.texts_train.shape)
        print('file_num_train shape:', self.file_num_train.shape)

    def prepare_data_(self, z, texts, file_num):
        
        # repeat data for each text
        z = z.repeat(3, 1).cpu().numpy()
        texts = texts.view(-1, texts.shape[-1])
        file_num = file_num.repeat(3).cpu().numpy()
        
        # remove outliers        
        bad_idx = (np.abs(z)>self.z_limit).max(axis=1).astype(bool)
        print(f"Removing {bad_idx.sum()} outliers")

        z, texts, file_num = z[~bad_idx], texts[~bad_idx], file_num[~bad_idx]
        return z, texts, file_num

    def fit_scaler(self, data):
        scaler = StandardScaler()
        return scaler.fit(data)

    def transform_data(self, data, scaler):
        return scaler.transform(data)

    def print_stats(self, data):
        print(f"""
        z shape: {data.shape}
        z mean: {data.mean()}
        z std: {data.std()}
        z min: {data.min()}
        z max: {data.max()}
        """)
   
    def setup(self, stage=None):
        
        # split data into train, val, test
        self.dataset_train = TensorDataset(self.z_train, self.texts_train, self.file_num_train)
        self.dataset_val = TensorDataset(self.z_val, self.texts_val, self.file_num_val)
        self.dataset_test = TensorDataset(self.z_test, self.texts_test, self.file_num_test)


    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False)
    