# This file Is our main run file for the application.

# models: VAE, imageDiffusion, latentDiffusion
# modes: train, build (over_fit single batch), inference, optuna
import yaml, argparse
import matplotlib.pyplot as plt
import torch.nn as nn
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from utils import load_config, manual_config_log, get_ckpt
from modules.dataModules import MNISTDataModule


if __name__ == "__main__":
    # add arguments for model and mode
    parser = argparse.ArgumentParser(description='Run the model')
    parser.add_argument('--model', type=str, default='VAE', help='Model to run')
    parser.add_argument('--mode', type=str, default='train', help='Mode to run')
    args = parser.parse_args()
    print(args.model, args.mode)
    assert args.model in ['VAE', 'imageDiffusion', 'latentDiffusion']  # assert valid
    assert args.mode in ['train', 'build', 'inference', 'optuna']

    if args.model == 'VAE':
        from modules.VAE import VAE2 as VAE
        from modules.loss import VAE_Loss
        from scripts.VAE.train import train as train_VAE
        from scripts.VAE.optuna import optuna_sweep as optuna_VAE

        config = load_config('VAE')
        dm = MNISTDataModule(**config[args.mode.upper()]['DATA'], verbose=False)
        dm.setup()
        logger = TensorBoardLogger("logs/VAE/", name=f"{args.mode}")
        criteria = VAE_Loss(config[args.mode.upper()]['LOSS'])

        if args.mode == 'build':
            # overfit a single sample
            model = VAE(criteria, **config['BUILD']['MODEL'])
            trainer = Trainer(logger=logger, **config['BUILD']['TRAINER'])
            trainer.fit(model, dm)

        elif args.mode == 'train':
            manual_config_log(logger.log_dir, cp_file='configs/config_VAE.yaml')
            train_VAE(dm, criteria, config, logger, VAE, save_latent=True)

        elif args.mode == 'optuna': optuna_VAE(VAE, dm, config)
        
    elif args.model == 'imageDiffusion':
        from mnist_latent_diffusion.modules.imageDiffusion import ImageDiffusionModule, _calculate_FID_SCORE
        config = load_config('Diffusion')

        if args.mode == 'train':
            dm = MNISTDataModule(**config[args.mode.upper()]['DATA'], verbose=False)
            dm.setup()

            # if args.mode == 'build':
            plModule = ImageDiffusionModule(criteria=nn.MSELoss(), **config['TRAIN']['MODEL'])

            logger = pl.loggers.TensorBoardLogger("logs/imageDiffusion", name="train")
            trainer = pl.Trainer(max_epochs=400,
                                logger=logger,)
            trainer.fit(plModule, dm)
        elif args.mode == 'inference':

            
            parent_log_dir = 'logs/imageDiffusion/train/'
            checkpoint = get_ckpt(parent_log_dir)

            plModule = ImageDiffusionModule.load_from_checkpoint(checkpoint)

            x_t, hist, y = plModule.model.sampling(1, clipped_reverse_diffusion=False, y=True, device='mps', tqdm_disable=False)

            fig, ax = plt.subplots(1, 1, figsize=(5, 6))
            ax.imshow(x_t.squeeze().detach().cpu().numpy(), cmap='gray')
            ax.set_title(f'Sample from model, with label y={y.item()}')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.show()

    elif args.model == 'latentDiffusion':
        
        from modules.latentDiffusion import LatentDiffusionModel
        if args.mode == 'train':
            from scripts.latentDiffusion.train import train as train_LatentDiffusion
            train_LatentDiffusion()
        
        elif args.mode == 'inference':
            print('Inference')
            parent_log_dir = 'logs/latentDiffusion/train/'
            checkpoint = get_ckpt(parent_log_dir)
            
    else: raise NotImplementedError
