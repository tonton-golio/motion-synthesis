from LatentDiffusion.dataset import LatentSpaceDataModule
from LatentDiffusion.model import LatentDiffusionModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import os
import torch
import yaml
from loss import VAE_Loss


if __name__ == "__main__":
    logger = TensorBoardLogger("logs/", name="LatentDiffusion")

    # make logs directory
    if not os.path.exists(logger.log_dir):
        os.makedirs(logger.log_dir)

    # cp config file to the log directory
    shutil.copyfile('configs/config_LatentDiffusion.yaml', f"{logger.log_dir}/hparams.yaml")

    # Load config file and use it to create the data module
    with open('configs/config_LatentDiffusion.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # load latent space
    path = f"logs/MNIST_VAE/version_{config['DIFFUSE']['DATA']['V']}/saved_latent/"
    z = torch.load(path + 'z.pt')
    autoencoder = torch.load(path + 'model.pth').to(torch.device('mps'))
    y = torch.load(path + 'y.pt')

    # set up data module
    dm = LatentSpaceDataModule(z, y, batch_size=config['DIFFUSE']['DATA']['BATCH_SIZE'])
    dm.setup()
    scaler = dm.scaler
    torch.save(scaler, f'{logger.log_dir}/scaler.pth')

    # loss
    criteria = VAE_Loss(config['DIFFUSE']['LOSS'])

    # set up model
    model = LatentDiffusionModel(autoencoder=autoencoder, 
                                 scaler=scaler,
                                criteria=criteria,  
                                 **config['DIFFUSE']['MODEL'])
    
    # load
    

    if config['DIFFUSE']['MODEL']['LOAD']:
        name, num = logger.log_dir.split('/')[-1].split('_')
        log_dir = '/'.join(logger.log_dir.split('/')[:-1])



        model = LatentDiffusionModel.load_from_checkpoint(
            # logger.log_dir + '/' 
            log_dir + '/version_' + str(int(num)-1) + '/checkpoints/' +
            config['DIFFUSE']['MODEL']['LOAD'])
        print("LOADed model")

    # train
    trainer = pl.Trainer(logger=logger, **config['DIFFUSE']['TRAINER'])
    trainer.fit(model, dm)
    epochs_trained = trainer.current_epoch

    # test
    res = trainer.test(model, dm.test_dataloader())

    # logging hyperparameters
    logger.log_hyperparams(model.hparams, metrics=res[0])
    print("done")

    # save model
    torch.save(model, f'{"/".join(logger.log_dir.split("/")[:-1])}//model.pth')

    print("model saved")
