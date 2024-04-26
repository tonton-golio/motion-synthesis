# MNIST dataset
# lightning data module
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

from sklearn.preprocessing import StandardScaler

import numpy as np

class BoolTransform(torch.nn.Module):
    """A custom PyTorch transform that converts tensors to boolean and then to float."""
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.bool().float()
    

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, verbose=True, **kwargs):
        super().__init__()
        self.batch_size = kwargs.get("BATCH_SIZE", 256)
        self.path = '/Users/tonton/Documents/motion-synthesis/mnist_latent_diffusion/'
        self.shuffle = kwargs.get("SHUFFLE", True)
        # transforms
        self.rotation = kwargs.get("ROTATION", 0)
        self.scale = kwargs.get("SCALE", 0)
        self.translate = (kwargs.get("TRANSLATE_X", 0), kwargs.get("TRANSLATE_Y", 0))
        self.shear = kwargs.get("SHEAR", 0)
        self.normalize = (kwargs.get("NORMALIZE_MEAN", 0.1307), kwargs.get("NORMALIZE_STD", 0.3081))
        self.bool = kwargs.get("BOOL", False)
        self.no_normalize = kwargs.get("NO_NORMALIZE", False)

        if verbose: self.print_params()

    def print_params(self):
        print('DataModule params:')
        print("\tbatch_size:", self.batch_size)
        print("\tpath:", self.path)
        print("\trotation:", self.rotation)
        print("\tscale:", self.scale)
        print("\ttranslate:", self.translate)
        print("\tshear:", self.shear)
        print("\tnormalize:", self.normalize)
        print("\tbool:", self.bool)
        print("\tno_normalize:", self.no_normalize)

    def compose_transfoms(self):
        """
        Compose transforms for MNIST dataset.
        """
        
        transform_lst = [transforms.ToTensor()]
        if self.rotation != 0 or self.scale != 0 or self.translate != (0, 0) or self.shear != 0:
            transform_lst.append(transforms.RandomAffine(degrees=self.rotation, 
                                                         translate=self.translate, 
                                                            shear=self.shear,
                                                         scale=(1-self.scale, 1+self.scale), 
                                                         fill=0))
        if self.bool:
            transform_lst.append(BoolTransform())
        elif self.no_normalize:
            pass
        else:
            transform_lst.append(transforms.Normalize(*self.normalize))
            
            # instead do 0 to 1 
        return transforms.Compose(transform_lst)
    
    def setup(self, stage=None, max_samples=None):
        """
        Get dataloaders for MNIST dataset.
        """   
        transforms = self.compose_transfoms()

        data_train = MNIST(root=self.path, train=True, download=False, transform=transforms)
        self.data_test = MNIST(root=self.path, train=False, download=False, transform=transforms)
        print('len before', len(data_train), len(self.data_test))
        if max_samples is not None:
            
            data_train = torch.utils.data.Subset(data_train, torch.arange(max_samples))
            self.data_test = torch.utils.data.Subset(self.data_test, torch.arange(max_samples))
            print('len after',len(data_train), len(self.data_test))

        # split train into train and val
        train_size = int(0.8 * len(data_train))
        val_size = len(data_train) - train_size
        self.data_train, self.data_val = random_split(data_train, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=False, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=2, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=2, persistent_workers=True)
     
    def show_batch(self, return_fig=False):
        dataiter = next(iter(self.train_dataloader()))
        images, labels = dataiter

        img = make_grid(images)
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(img.permute(1, 2, 0))
        if return_fig:
            return plt
        else:
            plt.show()


def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

class LatentSpaceDataModule(pl.LightningDataModule):
    def __init__(self, X, y, batch_size=64, scale=False):
        super().__init__()
        self.batch_size = batch_size

        self.X = X  # (N, D)
        # y = y.unsqueeze(1)  # (N, 1)
        # make it into an index tensor
        # print('y', y.shape, y)
        self.y = torch.nn.functional.one_hot(y.long(), num_classes=10).float()#.squeeze().to('mps')

        # normalize X
        if scale:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X.detach().numpy())
            self.X = torch.tensor(self.X).float()
        else:
            self.X = self.X.float()
            self.scaler = None

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
        self.y_train = y_train.clone().detach()#.unsqueeze(1)
        self.y_val = y_val.clone().detach()#.unsqueeze(1)
        self.y_test = y_test.clone().detach()#.unsqueeze(1)


        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.val_dataset = TensorDataset(self.X_val, self.y_val)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, persistent_workers=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, persistent_workers=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, persistent_workers=True, drop_last=True)

