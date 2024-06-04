import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from utils import print_header

class LatentMotionData(pl.LightningDataModule):
    def __init__(self, z, texts, batch_size=32):
        super().__init__()
        # print('z shape:', z.shape)  # this is (n, z_dim)
        # print('texts shape:', texts.shape) # this is (n, 3, text_len) -> 3 is for the 3 different texts
        self.z = z.repeat(3, 1)  # repeat z for each text
        self.texts = texts.view(-1, texts.shape[-1])  # flatten texts
        # print('z shape:', self.z.shape)  # this is (n*3, z_dim)
        # print('texts shape:', self.texts.shape) # this is (n*3, text_len)
        self.batch_size = batch_size
        self.latent_dim = z.shape[-1]

        print_header('LatentMotionData')
        print('\tz shape:', self.z.shape)  # this is (n*3, z_dim)
        min_z = torch.min(self.z, dim=0).values
        max_z = torch.max(self.z, dim=0).values
        # for i in range(self.latent_dim):
        #     print(f'\t\tz_dim {i}: min={min_z[i]:.2f}, max={max_z[i]:.2f}')


        # instead of printing all, we just wanna print the mean and std of min and max
        print(f'\t\tz_dim mean: min={torch.mean(min_z):.2f}, max={torch.mean(max_z):.2f}')

    def setup(self, stage=None):
        
        # split data into train, val, test
        dataset = TensorDataset(self.z, self.texts)

        # train/val/test split
        n = len(dataset)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        n_test = n - n_train - n_val

        train_data, val_data, test_data = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
    