import importlib.util
import os


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import pytorch_lightning as pl
from sklearn.manifold import TSNE
import glob
import umap

from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl
import math
from tqdm import tqdm


import sys; sys.path += ['/Users/tonton/Documents/motion-synthesis']
from encode_mnist.src.model import Autoencoder
from encode_mnist.src.dataset import MNISTDataModule
from encode_mnist.src.config import config as cfg_ae

class AutoencoderUtility:
    def __init__(self, path: str, N: int = 1000, latent_dim: int = -1):
        self.path = path
        print("path:", path)
        if latent_dim == -1:
            latent_dim = cfg_ae.MODEL.latent_dim
        self.latent_dim = latent_dim
        self.N = N

        self.checkpoint_path = self.get_latest_checkpoint()

    def setup(self):
        self.autoencoder = self.load_autoencoder()
        # self.data, self.z, self.decoded = self.process_data()
        self.X, self.y, self.z, self.reconstruction = self.process_data()
        self.projector, self.projection = self.make_projection()
        self.decoder = self.get_decorder()

    def get_latest_checkpoint(self):
        return max(
            glob.glob(os.path.join(self.path, "checkpoints", "*")), key=os.path.getctime
        )

    def load_autoencoder(self):
        cfg_ae.MODEL.model_path = self.checkpoint_path
        cfg_ae.MODEL.load_model = True
        

        autoencoder = Autoencoder(cfg_ae.MODEL)
        # state_dict = torch.load(checkpoint_path, map_location="mps")
        # autoencoder.load_state_dict(state_dict["state_dict"])
        return autoencoder
    
    def get_decorder(self):
        # return the function to decode
        return self.autoencoder.decode

    def process_data(self):
        cfg_ae.DATA.batch_size = self.N
        data_module = MNISTDataModule(cfg_ae.DATA)
        # data_module.setup(stage='test')
        # data_loader = data_module.test_dataloader()
        data_module.setup()
        data_loader = data_module.train_dataloader()
        data = next(iter(data_loader))

        self.autoencoder.eval()
        with torch.no_grad():
            x, y = data
            # print('x:', x.shape, x[:1])
            # print('y:', y.shape, y[:1])
            mu, logvar = self.autoencoder.encode(x)
            # print('mu:', mu.shape, mu[:1])
            # print('logvar:', logvar.shape, logvar[:1])
            z = self.autoencoder.reparameterize(mu, logvar)
            # print('z:', z.shape, z[:1])
            reconstructed = self.autoencoder.decode(z)
            # print('decoded:', decoded.shape, decoded[:1])

        return x, y, z, reconstructed

    def make_projection(self):
        # we use umap, because it is parametric and ca thus be used to project new data
        reducer = umap.UMAP()
        projector = reducer.fit(self.z.cpu().detach().numpy())
        projection = projector.transform(self.z.cpu().detach().numpy())
        return projector, projection

    def plot(self):
        fig = plt.figure(figsize=(15, 6))
        gs = GridSpec(3, 6, figure=fig)

        # add axes
        ax_prodj = fig.add_subplot(gs[:, 2:])
        ax_digits = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        ax_recons = [fig.add_subplot(gs[i, 1]) for i in range(3)]

        # Digit and reconstruction plots
        for i, (ax_d, ax_r) in enumerate(zip(ax_digits, ax_recons)):

            ax_d.imshow(self.X[i].squeeze(), cmap="gray")
            ax_d.set_title(f"Digit {i}")
            ax_d.axis("off")

            ax_r.imshow(self.reconstruction[i].squeeze(), cmap="gray")
            ax_r.set_title(f"Recon {i}")
            ax_r.axis("off")

        scatter = ax_prodj.scatter(
            self.projection[:, 0],
            self.projection[:, 1],
            c=self.y.cpu().detach().numpy(),
            cmap="tab10",
            alpha=0.5,
            s=2,
        )
        ax_prodj.set_xlabel("z1")
        ax_prodj.set_ylabel("z2")
        ax_prodj.set_title("Latent Space")
        fig.colorbar(scatter, ax=ax_prodj, orientation="vertical")

        plt.tight_layout()


def get_latent_space(latent_dim=8, N=100, V=86, plot=False):
    
    utility = AutoencoderUtility(
            path=f"../../encode_mnist/tb_logs/MNISTAutoencoder/version_{V}/",
            N=N,
            latent_dim=latent_dim,
        )
    save_path = f"../saved_latent/" + '-'.join(utility.checkpoint_path.split("/")[-2:]).split(".")[0]
    # check if saved data exists and load it
    if os.path.exists(save_path):
        print(f"Loading latent space from {save_path}")
        X = torch.load(os.path.join(save_path, "X.pt"))
        y = torch.load(os.path.join(save_path, "y.pt"))
        z = torch.load(os.path.join(save_path, "z.pt"))
        reconstruction = torch.load(os.path.join(save_path, "reconstruction.pt"))
        projector = torch.load(os.path.join(save_path, "projector.pt"))
        projection = torch.load(os.path.join(save_path, "projection.pt"))
        decoder = torch.load(os.path.join(save_path, "decoder.pt"))
        return X, y, z, reconstruction, projector, projection, decoder
    else:
        
        utility.setup()
        X, y, z, reconstruction = utility.X, utility.y, utility.z, utility.reconstruction
        projector, projection = utility.projector, utility.projection
        decoder = utility.get_decorder()

    #  save the data in ../saved_latent

        os.makedirs(save_path, exist_ok=True)
        for k, v in zip('X y z reconstruction projector projection decoder'.split(), 
                        [X, y, z, reconstruction, projector, projection, decoder]):
            torch.save(v, os.path.join(save_path, f"{k}.pt"))
        


    if plot:
        utility.plot()

    # delete the utility object to free up memory
    del utility
    # del Autoencoder
    # del MNISTDataModule
    return X, y, z, reconstruction, projector, projection, decoder

class LatentSpaceDataModule(pl.LightningDataModule):
    def __init__(self, X, y, batch_size=64):
        super().__init__()
        train_prc, val_prc, test_prc = 0.8, 0.1, 0.1
        indices = torch.randperm(len(X)).tolist()
        train_end = int(train_prc * len(X))
        val_end = train_end + int(val_prc * len(X))
        X_train, X_val, X_test = (
            X[indices[:train_end]],
            X[indices[train_end:val_end]],
            X[indices[val_end:]],
        )
        y_train, y_val, y_test = (
            y[indices[:train_end]],
            y[indices[train_end:val_end]],
            y[indices[val_end:]],
        )
        
        # self.X_train = torch.tensor(X_train)
        # self.X_val = torch.tensor(X_val)
        # self.X_test = torch.tensor(X_test)
        # self.y_train = torch.tensor(y_train).unsqueeze(1)
        # self.y_val = torch.tensor(y_val).unsqueeze(1)
        # self.y_test = torch.tensor(y_test).unsqueeze(1)

        self.X_train = X_train.clone().detach()
        self.X_val = X_val.clone().detach()
        self.X_test = X_test.clone().detach()
        self.y_train = y_train.clone().detach().unsqueeze(1)
        self.y_val = y_val.clone().detach().unsqueeze(1)
        self.y_test = y_test.clone().detach().unsqueeze(1)



        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.val_dataset = TensorDataset(self.X_val, self.y_val)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, persistent_workers=True)


if __name__ == "__main__":
    BATCH_SIZE = 64
    X, y, z, reconstruction, projector, projection, decoder = get_latent_space()
    dm = LatentSpaceDataModule(z, y, batch_size=BATCH_SIZE)
    dm.setup()

    print("dm:", dm)
    print("dm.train_dataloader:", dm.train_dataloader())
    # get first batch and print shape
    x, y = next(iter(dm.train_dataloader()))
    print("x:", x.shape)
    print("y:", y.shape)

    # decode
    decoded = decoder(z)
    plt.imshow(decoded[0].squeeze().detach().numpy(), cmap="gray")
    plt.show()