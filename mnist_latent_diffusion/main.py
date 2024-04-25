


# This file Is our main run file for the application.

# models: VAE, Diffusion, LatentDiffusion
# modes: train, build, inference, optuna

## In build mode: we want to overfit just a single sample.

import argparse
from modules.data_modules import MNISTDataModule
from modules.VAE_model import VAE2 as VAE
from modules.loss import VAE_Loss
import yaml
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import os
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch

# add arguments for model and mode
parser = argparse.ArgumentParser(description='Run the model')
parser.add_argument('--model', type=str, default='VAE', help='Model to run')
parser.add_argument('--mode', type=str, default='train', help='Mode to run')

args = parser.parse_args()
assert args.model in ['VAE', 'Diffusion', 'LatentDiffusion']  # assert valid
assert args.mode in ['train', 'build', 'inference', 'optuna']


def load_config(name):
    with open(f'configs/config_{name}.yaml', 'r') as file:
        return yaml.safe_load(file)
    

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

    # make covariance matrix of latent space
    cov = torch.cov(latent.T)
    cov_fig = plt.figure()
    plt.imshow(cov.cpu().detach().numpy())
    plt.colorbar()
    plt.title('Covariance matrix of latent space')
    plt.savefig(f'{logger.log_dir}/covariance_matrix.png')
    plt.close(cov_fig)
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
    if args.model == 'VAE':
        config = load_config('VAE')
        # VAE accepts train, build, optuna
        dm = MNISTDataModule(**config[args.mode.upper()]['DATA'], verbose=False)
        dm.setup()
        logger = TensorBoardLogger("logs/VAE/", name=f"{args.mode}")
        criteria = VAE_Loss(config['TRAIN']['LOSS'])

        if args.mode == 'train':
            if not os.path.exists(logger.log_dir): os.makedirs(logger.log_dir)
            shutil.copyfile('configs/config_VAE.yaml', f"{logger.log_dir}/config_VAE.yaml")

            # train
            model = VAE(criteria, **config['TRAIN']['MODEL'])
            trainer = Trainer(logger=logger, **config['TRAIN']['TRAINER'])
            trainer.fit(model, dm)

            # test
            trainer.test(model, datamodule=dm)
            res = model.on_test_epoch_end()
            logger.log_hyperparams(model.hparams, {'unscaled mse test loss' : res[0]} )

            # save
            dataloaders = [dm.test_dataloader(), dm.train_dataloader(), dm.val_dataloader()]
            KL_weight = config['TRAIN']['LOSS']['DIVERGENCE_KL']
            latent, labels = prep_save(model, dataloaders)
            latent_dim = torch.prod(torch.tensor(latent.shape[1:]))
            latent = latent.view(-1, latent_dim)

            projection, reducer = plotUMAP(latent, labels, latent_dim, KL_weight, logger.log_dir, show=False)
            if input('save for diffusion? [y/n]') == 'y':
                save_for_diffusion(save_path=logger.log_dir+'/saved_latent', model = model, z = latent, y = labels, projection = projection, projector = reducer,  )

        elif args.mode == 'build':
            # overfit a single sample
            model = VAE(criteria, **config['BUILD']['MODEL'])
            trainer = Trainer(logger=logger, **config['BUILD']['TRAINER'])
            trainer.fit(model, dm)

        elif args.mode == 'optuna':
            def objective(trial: optuna.trial.Trial) -> float:
                # these are our sweep parameters
                ld = trial.suggest_categorical("latent_dim", [4, 6, 8, 10])

                acts = ['tanh', 'sigmoid', 'softsign', 'leaky_relu', 'ReLU', 'elu']
                #act = trial.suggest_categorical("activation", acts)
                klDiv = trial.suggest_float("klDiv", 1e-6, 1e-4, log=True)
                
                # activation func
                #config['OPTUNA']['MODEL']['activation'] = act

                # set up loss function
                loss_dict = {
                    'RECONSTRUCTION_L2': 1,
                    'DIVERGENCE_KL': klDiv,
                }
                criteria = VAE_Loss(loss_dict)

                # instantiate the logger
                logger = TensorBoardLogger("logs/", name="mnistVAEoptuna_sigmoid_outact")
                
                # Create the model
                model = VAE(criteria, **config['OPTUNA']['MODEL'], LATENT_DIM=ld)

                trainer = Trainer(
                    logger=logger,
                    **config['OPTUNA']['TRAINER'],
                    enable_checkpointing=False,
                    # val_check_interval=0.5,
                    callbacks=[PyTorchLightningPruningCallback(trial, monitor="test_loss")],
                )

                trainer.fit(model, dm)

                res = trainer.test(model, dm.test_dataloader())
                print('res:', res   )
                # logging hyperparameters
                logger.log_hyperparams(model.hparams, metrics=res[0])    

                # manual logging in hparams.yaml
                hparams = {k: v for k, v in model.hparams.items()}
                ## append the test loss, and sweep parameters and model.metric_res
                hparams['mse_us_tst'] = res[0]['test_unscaled_RECONSTRUCTION_L2']
                hparams['latent_dim'] = ld
                hparams['klDivWeight'] = klDiv
                for k, v in res[0].items():
                    hparams[k] = v
                
                with open(logger.log_dir + '/hparams.yaml', 'w') as file:
                    yaml.dump(hparams, file)

                return res[0]['test_unscaled_RECONSTRUCTION_L2']

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=config['OPTUNA']['OPTIMIZE']['n_trials'], timeout=config['OPTUNA']['OPTIMIZE']['timeout'] )
            print('best trial:', study.best_trial.params)
        else:
            raise NotImplementedError
        
    elif args.model == 'Diffusion':
        pass

    elif args.model == 'LatentDiffusion':
        

        pass


    else:
        raise NotImplementedError