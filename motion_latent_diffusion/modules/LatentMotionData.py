import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
try:
    from utils import print_header
except:
    from motion_latent_diffusion.utils import print_header
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
        # Diffusion expects ~unit-variance inputs, so standardize latents by default
        # (the toy-2D test shows DDIM is unstable on unnormalized data).
        self.scale = kwargs.get('scale', True)
        self.tiny = kwargs.get('tiny', False)

        print('self scale:', self.scale)

        self.z_train, self.texts_train, self.file_num_train = self.prepare_data_(z_train, texts_train, file_num_train)
        self.z_val, self.texts_val, self.file_num_val = self.prepare_data_(z_val, texts_val, file_num_val)
        self.z_test, self.texts_test, self.file_num_test = self.prepare_data_(z_test, texts_test, file_num_test)

        self.scaler = self.fit_scaler(self.z_train) if self.scale else NoScaler()
        self.z_train = self.transform_data(self.z_train, self.scaler)
        self.z_val = self.transform_data(self.z_val, self.scaler)
        self.z_test = self.transform_data(self.z_test, self.scaler)

        # Keep tensors on CPU; Lightning moves batches to the right device. (The
        # original hardcoded .to('mps'), breaking non-Apple machines and eagerly
        # pinning the whole dataset to the GPU.)
        self.z_train = torch.tensor(self.z_train).float()
        self.z_val = torch.tensor(self.z_val).float()
        self.z_test = torch.tensor(self.z_test).float()

        self.file_num_train = torch.tensor(self.file_num_train).long()
        self.file_num_val = torch.tensor(self.file_num_val).long()
        self.file_num_test = torch.tensor(self.file_num_test).long()

        for z in [self.z_train, self.z_val, self.z_test]:
            self.print_stats(z)

        print('z_train shape:', self.z_train.shape)
        print('texts_train shape:', self.texts_train.shape)
        print('file_num_train shape:', self.file_num_train.shape)

    def prepare_data_(self, z, texts, file_num):

        # Each motion has 3 captions: texts is (N, 3, cond_dim). Flattening gives
        # [t0a,t0b,t0c, t1a,...], so the latents/file_nums must be INTERLEAVED to
        # match ([z0,z0,z0, z1,...]). The original used .repeat(3,1) which TILES
        # ([z0..zN, z0..zN, ...]) and paired every latent with the wrong caption
        # (deep-review D-LatentMotionData).
        n_caps = texts.shape[1] if texts.dim() == 3 else 1
        z = z.repeat_interleave(n_caps, dim=0).cpu().numpy()
        texts = texts.reshape(-1, texts.shape[-1])
        file_num = file_num.repeat_interleave(n_caps).cpu().numpy()
        assert z.shape[0] == texts.shape[0] == file_num.shape[0], (
            f"latent/text/file_num misaligned: {z.shape[0]}, {texts.shape[0]}, {file_num.shape[0]}"
        )

        # remove outliers
        bad_idx = (np.abs(z)>self.z_limit).max(axis=1).astype(bool)
        print(f"Removing {bad_idx.sum()} outliers")

        z, texts, file_num = z[~bad_idx], texts[~bad_idx], file_num[~bad_idx]
        if self.tiny:
            z, texts, file_num = z[:self.tiny], texts[:self.tiny], file_num[:self.tiny]
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
    