# MNIST dataset
# lightning data module
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class BoolTransform(torch.nn.Module):
    """A custom PyTorch transform that converts tensors to boolean and then to float."""
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.bool().float()
    

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, verbose=True, **kwargs):
        super().__init__()
        self.batch_size = kwargs.get("BATCH_SIZE", 256)
        self.path = '/Users/tonton/Documents/motion-synthesis/mnist_latent_diffusion/'

        # transforms
        self.rotation = kwargs.get("ROTATION", 0)
        self.scale = kwargs.get("SCALE", 0)
        self.translate = (kwargs.get("TRANSLATE_X", 0), kwargs.get("TRANSLATE_Y", 0))
        self.shear = kwargs.get("SHEAR", 0)
        self.normalize = (kwargs.get("NORMALIZE_MEAN", 0.1307), kwargs.get("NORMALIZE_STD", 0.3081))
        self.bool = kwargs.get("BOOL", False)

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
        else:
            transform_lst.append(transforms.Normalize(*self.normalize))
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
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=2, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=2, persistent_workers=True)
     
    def show_batch(self):
        dataiter = next(iter(self.train_dataloader()))
        images, labels = dataiter

        img = make_grid(images)
        plt.imshow(img.permute(1, 2, 0))
        plt.show()

