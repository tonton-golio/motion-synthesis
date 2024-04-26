


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

from utils import load_config, plotUMAP, prep_save, save_for_diffusion
if __name__ == "__main__":
    # add arguments for model and mode
    parser = argparse.ArgumentParser(description='Run the model')
    parser.add_argument('--model', type=str, default='VAE', help='Model to run')
    parser.add_argument('--mode', type=str, default='train', help='Mode to run')

    args = parser.parse_args()
    assert args.model in ['VAE', 'Diffusion', 'LatentDiffusion']  # assert valid
    assert args.mode in ['train', 'build', 'inference', 'optuna']

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
            latent, labels = prep_save(model, dataloaders, enable_y=True, log_dir=logger.log_dir)
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