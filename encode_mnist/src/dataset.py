# MNIST dataset
# lightning

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from config import dotdict

class Bool(torch.nn.Module):
    def __init__(self):
        super(Bool, self).__init__()

    def forward(self, tensor : torch.Tensor) -> torch.Tensor:

        return tensor.bool().float()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, kwargs):
        super().__init__()
        self.batch_size = kwargs.get("batch_size", 256)
        self.include_digits = kwargs.get("_include_digits", [0,1,2,3,4,5,6,7,8,9])
        self.path = kwargs.get("path", '../../data/other_data')
        self.transforms = kwargs.get("transforms", dotdict({
            'rotate_degrees' : 0,
            'scale' : 0,
            'translate' : (0, 0),
            'shear' : 0,
            'normalize' : (0.1307, 0.3081),
            'bool' : True,
        }))

    def setup(self, stage=None):
        """
        Get dataloaders for MNIST dataset.
        """
        transform_lst = [transforms.ToTensor()]
        if self.transforms.rotate_degrees != 0 or self.transforms.scale != 0 or self.transforms.translate != (0, 0) or self.transforms.shear != 0:
            transform_lst.append(transforms.RandomAffine(degrees=self.transforms.rotate_degrees, 
                                                         translate=self.transforms.translate, 
                                                            shear=self.transforms.shear,
                                                         scale=(1-self.transforms.scale, 1+self.transforms.scale), 
                                                         fill=0))
        if self.transforms.bool:
            transform_lst.append(Bool())
        else:
            transform_lst.append(transforms.Normalize(*self.transforms.normalize))


        transform = transforms.Compose(transform_lst)

        data_train = torchvision.datasets.MNIST(root=self.path, train=True, download=False, transform=transform)
        data_test = torchvision.datasets.MNIST(root=self.path, train=False, download=False, transform=transform)

        # filter out digits
        if self.include_digits != [0,1,2,3,4,5,6,7,8,9]:
            data_train = torch.utils.data.Subset(data_train, indices=[i for i in range(len(data_train)) if data_train[i][1] in self.include_digits])
            data_test = torch.utils.data.Subset(data_test, indices=[i for i in range(len(data_test)) if data_test[i][1] in self.include_digits])

        # split train into train and val
        train_size = int(0.8 * len(data_train))
        val_size = len(data_train) - train_size
        self.data_train, self.data_val = torch.utils.data.random_split(data_train, [train_size, val_size])
        self.data_test = data_test

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=2, persistent_workers=True)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=1, persistent_workers=True)
    

# test it
if __name__ == "__main__":
    # small check

    # dm = MNISTDataModule()
    # dm.setup()
    # for batch in dm.train_dataloader:
    #     print(batch[0].shape)
    #     break
    # for batch in dm.val_dataloader:
    #     print(batch[0].shape)
    #     break
    # for batch in dm.test_dataloader:
    #     print(batch[0].shape)
    #     break
    # print('done')

    # larger check (check the effect of transforms)
    batch_size = 1024
    dm_noTransform = MNISTDataModule(transforms = {'rotate_degrees' : 0, 'distortion_scale' : 0, 'translate' : (0, 0)}, batch_size=batch_size)
    dm_noTransform.setup()
    dm_transform = MNISTDataModule(transforms = {'rotate_degrees' : 10, 'distortion_scale' : 0.4, 'translate' : (0.2, 0.2)}, batch_size=batch_size)
    dm_transform.setup()

    batch_noTransform = next(iter(dm_noTransform.train_dataloader()))
    batch_transform = next(iter(dm_transform.train_dataloader()))

    grid_noTransform = torchvision.utils.make_grid(batch_noTransform[0][:32], nrow=8)
    grid_transform = torchvision.utils.make_grid(batch_transform[0][:32], nrow=8)

    plt.figure(figsize=(15, 15))
    plt.subplot(3, 2, 1)
    plt.imshow(grid_noTransform.permute(1, 2, 0))
    plt.title('No transform')

    plt.subplot(3, 2, 2)
    plt.imshow(grid_transform.permute(1, 2, 0))
    plt.title('With transform')
    

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # PCA
    PC_noTransform = PCA(n_components=2).fit_transform(batch_noTransform[0].view(batch_noTransform[0].shape[0], -1))
    PC_transform = PCA(n_components=2).fit_transform(batch_transform[0].view(batch_transform[0].shape[0], -1))

    plt.subplot(3, 2, 3)
    plt.scatter(PC_noTransform[:, 0], PC_noTransform[:, 1], c=batch_noTransform[1], cmap='tab10')
    plt.title('No transform')
    plt.colorbar()

    plt.subplot(3, 2, 4)
    plt.scatter(PC_transform[:, 0], PC_transform[:, 1], c=batch_transform[1], cmap='tab10')
    plt.title('With transform')
    plt.colorbar()


    # t-SNE
    TSNE_noTransform = TSNE(n_components=2).fit_transform(batch_noTransform[0].view(batch_noTransform[0].shape[0], -1))
    TSNE_transform = TSNE(n_components=2).fit_transform(batch_transform[0].view(batch_transform[0].shape[0], -1))

    plt.subplot(3, 2, 5)
    plt.scatter(TSNE_noTransform[:, 0], TSNE_noTransform[:, 1], c=batch_noTransform[1], cmap='tab10')
    plt.title('No transform')
    plt.colorbar()

    plt.subplot(3, 2, 6)
    plt.scatter(TSNE_transform[:, 0], TSNE_transform[:, 1], c=batch_transform[1], cmap='tab10')
    plt.title('With transform')
    plt.colorbar()


    plt.tight_layout()

    plt.show()