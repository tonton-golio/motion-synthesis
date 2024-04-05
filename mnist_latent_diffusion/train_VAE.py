from VAE.dataset import MNISTDataModule
from VAE.model import VAE
from loss import VAE_Loss
import yaml
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import os

import torch

def save_for_diffusion(model, data_loaders, save_path):
    """
    Save:
        'encoder': 'encoder.pt',
        'decoder': 'decoder.pt',
        'latent' : 'z.pt',
        'labels' : 'y.pt',
        'projection' : 'projection.pt',
        'reconstruction' : 'reconstruction.pt',
        'projector' : 'projector.pt',
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    latent = []
    labels = []
    x = []
    x_hat = []

    for data_loader in data_loaders:
        for batch in data_loader:
            x_, y_ = batch
            x.append(x_)
            labels.append(y_)
            x_hat_, z, mu, logvar = model(x_)
            latent.append(z)
            x_hat.append(x_hat_)

    latent = torch.cat(latent, dim=0)  # maybe detach
    labels = torch.cat(labels, dim=0)
    x = torch.cat(x, dim=0)
    x_hat = torch.cat(x_hat, dim=0)

    import umap
    import numpy as np
    reducer = umap.UMAP()
    projection = reducer.fit_transform(latent.cpu().detach().numpy())
    
    torch.save(model, f'{save_path}/model.pth')
    torch.save(latent, f'{save_path}/z.pt')
    torch.save(labels, f'{save_path}/y.pt')
    torch.save(projection, f'{save_path}/projection.pt')
    torch.save(x, f'{save_path}/x.pt')
    torch.save(x_hat, f'{save_path}/x_hat.pt')
    torch.save(reducer, f'{save_path}/projector.pt')

    plt.figure()
    plt.scatter(projection[:, 0], projection[:, 1], c=labels.cpu().numpy(), cmap='tab10', alpha=0.5, s=4)
    plt.colorbar()
    plt.savefig(f'{save_path}/projection.png')


if __name__ == "__main__":
    logger = TensorBoardLogger("logs/", name="MNIST_VAE")

    # make logs directory
    if not os.path.exists(logger.log_dir):
        os.makedirs(logger.log_dir)

    # cp config file to the log directory
    shutil.copyfile('configs/config_VAE.yaml', f"{logger.log_dir}/config_VAE.yaml")
    
    # Load config file and use it to create the data module
    with open('configs/config_VAE.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Create the data module
    dm = MNISTDataModule(**config['VAE']['DATA'], verbose=False)
    dm.setup()

    # set up loss
    criteria = VAE_Loss(config['VAE']['LOSS'])

    # Create the model
    model = VAE(criteria, **config['VAE']['MODEL'])

    # load
    # if config['VAE']['MODEL']['load']:
        
    #     model = VAE.load_from_checkpoint(config['VAE']['MODEL']['load'])

    # train the model
    trainer = Trainer(logger=logger, **config['VAE']['TRAINER'])
    trainer.fit(model, dm)

    # test
    trainer.test(model, datamodule=dm)
    res = model.on_test_epoch_end()
    logger.log_hyperparams(model.hparams, {'unscaled mse test loss' : res} )

    # make latent space
    print('saving latent space')
    dataloaders = [dm.test_dataloader(), dm.train_dataloader(), dm.val_dataloader()]
    save_for_diffusion(model, dataloaders, logger.log_dir+'/saved_latent')
