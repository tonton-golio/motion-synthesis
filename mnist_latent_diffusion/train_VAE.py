from VAE.dataset import MNISTDataModule
from VAE.model import VAE2 as VAE
from loss import VAE_Loss
import yaml
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import os

import torch


def print_scientific(x):
    return "{:.2e}".format(x)

def plotUMAP(latent, labels, latent_dim, KL_weight,  save_path, show=False):
    import umap
    reducer = umap.UMAP()
    projection = reducer.fit_transform(latent.cpu().detach().numpy())
    
    fig = plt.figure()
    plt.scatter(projection[:, 0], projection[:, 1], c=labels.cpu().numpy(), cmap='tab10', alpha=0.5, s=4)
    plt.colorbar()
    plt.title(f'UMAP projection of latent space (LD={latent_dim}, KL={print_scientific(KL_weight)})')
    
    if save_path is not None:
        plt.savefig(f'{save_path}/projection_LD{latent_dim}_KL{print_scientific(KL_weight)}.png')
    
        return projection, reducer
    elif show:
        plt.show()
    return fig

def prep_save(model, data_loaders, enable_y=False):
    latent, labels = list(), list()
    for data_loader in data_loaders:
        for batch in data_loader:
            x_, y_ = batch
            _, z, _, _ = model(x_, y_) if enable_y else model(x_)
            labels.append(y_)
            latent.append(z)

    latent = torch.cat(latent, dim=0)  # maybe detach
    labels = torch.cat(labels, dim=0)

    return latent, labels

    
def save_for_diffusion(save_path, model, **kwargs):
    """
    Save:
        'model' : 'model.pth',
        'latent' : 'z.pt',
        'labels' : 'y.pt',
        'projection' : 'projection.pt',
        'reconstruction' : 'reconstruction.pt',
        'projector' : 'projector.pt',
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(model, f'{save_path}/model.pth')

    for k, v in kwargs.items():
        torch.save(v, f'{save_path}/{k}.pt')

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
    dm = MNISTDataModule(**config['TRAIN']['DATA'], verbose=False)
    dm.setup()

    # set up loss
    criteria = VAE_Loss(config['TRAIN']['LOSS'])

    # Create the model
    model = VAE(criteria, **config['TRAIN']['MODEL'])

    # load
    # if config['VAE']['MODEL']['load']:
        
    #     model = VAE.load_from_checkpoint(config['VAE']['MODEL']['load'])

    # train the model
    trainer = Trainer(logger=logger, **config['TRAIN']['TRAINER'])
    trainer.fit(model, dm)

    # test
    trainer.test(model, datamodule=dm)
    res = model.on_test_epoch_end()
    logger.log_hyperparams(model.hparams, {'unscaled mse test loss' : res[0]} )

    # make latent space
    print('Preparing latent space')
    dataloaders = [dm.test_dataloader(), dm.train_dataloader(), dm.val_dataloader()]
    
    
    KL_weight = config['TRAIN']['LOSS']['DIVERGENCE_KL']

    latent, labels = prep_save(model, dataloaders)
    latent_dim = torch.prod(torch.tensor(latent.shape[1:]))
    print('latent shape:', latent.shape)
    # reshape latent, as -1, 9
    latent = latent.view(-1, latent_dim)

    projection, reducer = plotUMAP(latent, labels, latent_dim, KL_weight, logger.log_dir, show=False)
    if input('save for diffusion? [y/n]') == 'y':
        save_for_diffusion(save_path=logger.log_dir+'/saved_latent',
                            model = model, 
                            z = latent,
                            y = labels,
                            projection = projection,
                            # x_hat = x_hat,
                            # x = x,
                            projector = reducer,
                            )
                            