import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

from sklearn.preprocessing import StandardScaler

class LatentSpaceDataModule(pl.LightningDataModule):
    def __init__(self, X, y, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

        self.X = X  # (N, D)
        self.y = y

        # normalize X
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X.detach().numpy())
        self.X = torch.tensor(self.X).float()

        # inverse transform
        # self.X = self.scaler.inverse_transform(self.X)
        
        self.train_prc = 0.8
        self.val_prc = 0.1
        self.test_prc = 0.1

    def setup(self, stage=None):
        indices = torch.randperm(len(self.X)).tolist()
        train_end = int(self.train_prc * len(self.X))
        val_end = train_end + int(self.val_prc * len(self.X))
        X_train, X_val, X_test = (
            self.X[indices[:train_end]],
            self.X[indices[train_end:val_end]],
            self.X[indices[val_end:]],
        )
        y_train, y_val, y_test = (
            self.y[indices[:train_end]],
            self.y[indices[train_end:val_end]],
            self.y[indices[val_end:]],
        )
    
        self.X_train = X_train.clone().detach()
        self.X_val = X_val.clone().detach()
        self.X_test = X_test.clone().detach()
        self.y_train = y_train.clone().detach().unsqueeze(1)
        self.y_val = y_val.clone().detach().unsqueeze(1)
        self.y_test = y_test.clone().detach().unsqueeze(1)

        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.val_dataset = TensorDataset(self.X_val, self.y_val)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, persistent_workers=True)

