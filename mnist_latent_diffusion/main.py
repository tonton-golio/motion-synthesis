# This file Is our main run file for the application.

# models: VAE, imageDiffusion, latentDiffusion
# modes: train, build (over_fit single batch), inference, optuna
import yaml, argparse
import matplotlib.pyplot as plt
import torch.nn as nn
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from utils import load_config, manual_config_log, get_ckpt, print_dict
from modules.dataModules import MNISTDataModule
import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback


def optuna_sweep(MODEL, dm, config):
    def objective(trial: optuna.trial.Trial) -> float:
        # these are our sweep parameters
        # ld = trial.suggest_categorical("latent_dim", [4, 6, 8, 10])

        # acts = ['tanh', 'sigmoid', 'softsign', 'leaky_relu', 'ReLU', 'elu']
        # #act = trial.suggest_categorical("activation", acts)
        # klDiv = trial.suggest_float("klDiv", 1e-6, 1e-4, log=True)

        # what should we sweep?
        # learning rate, batch_size, timesteps, Noise_schedule_method
        # epsilon if noise_schedule_method is cosine
        # beta_start, beta_end, if noise_schedule_method is linear or square
        settings = dict(
            # learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            # batch_size = trial.suggest_categorical("batch_size", [256, 1024]),
            TIMESTEPS = trial.suggest_categorical("TIMESTEPS", [100, 200]),
            NOISE_SCHEDULE_METHOD = trial.suggest_categorical("NOISE_SCHEDULE_METHOD", ['cosine', 'linear', 'square']),
            EPSILON = trial.suggest_float("EPSILON", 1e-3, 1e-1, log=True),
            BETA_START = trial.suggest_float("BETA_START", 1e-6, 1e-3, log=True),
            BETA_END = trial.suggest_float("BETA_END", 1e-4, 1e-1, log=True),
        )
        

        # change config
        for k, v in settings.items():
            config['OPTUNA']['MODEL'][k] = v

        print_dict(config['OPTUNA'])
        
        criteria = nn.MSELoss()
        # criteria = nn.CrossEntropyLoss()

        # instantiate the logger
        logger = TensorBoardLogger("logs/imageDiffusion/", name="optuna")

        # check if SAMPLE_EVERY is smaller than max_epochs

        if config['OPTUNA']['TRAINER']['max_epochs'] <= config['OPTUNA']['MODEL']['SAMPLE_EVERY']:
            print('max_epochs should be greater than SAMPLE_EVERY')
            config['OPTUNA']['MODEL']['SAMPLE_EVERY'] = config['OPTUNA']['TRAINER']['max_epochs'] - 1

        # Create the model
        model = MODEL(criteria, **config['OPTUNA']['MODEL'])

        # instantiate the trainer
        trainer = Trainer(
            logger=logger,
            **config['OPTUNA']['TRAINER'],
            enable_checkpointing=False,
            # val_check_interval=0.5,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss"),
                        # pl.callbacks.ModelCheckpoint(monitor='val_loss'),
                        pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss')
                    ]
        )

        # fit the model
        trainer.fit(model, dm)

        # hyperparameters = dict(
        #     TIMESTEPS = config['OPTUNA']['MODEL']['TIMESTEPS'],
        #     NOISE_SCHEDULE_METHOD = config['OPTUNA']['MODEL']['NOISE_SCHEDULE_METHOD'],
        #     EPSILON = config['OPTUNA']['MODEL']['EPSILON'],
        #     BETA_START = config['OPTUNA']['MODEL']['BETA_START'],
        #     BETA_END = config['OPTUNA']['MODEL']['BETA_END'],
        # )

        # test the model
        # trainer.test(model, dm.test_dataloader())
        stage = 'test'
        val_loss = trainer.callback_metrics[f'{stage}_loss'].item()
        fid = trainer.callback_metrics[f'{stage}_fid'].item()
        diversity = trainer.callback_metrics[f'{stage}_diversity'].item()
        multi_modality = trainer.callback_metrics[f'{stage}_multi_modality'].item()

        # logging hyperparameters
        logger.log_hyperparams(config['OPTUNA']['MODEL'], 
                               metrics={
                                       'hp_metric': val_loss,
                                       'val_loss_final': val_loss,
                                       'fid_final': fid,
                                       'diversity_final': diversity,
                                        'multi_modality_final': multi_modality
                                 })


        return val_loss
    
 
    pruner = optuna.pruners.MedianPruner() # median pruner
        # create the study
    # storage_location_base = 
    study = optuna.create_study(direction="minimize",
                                study_name='latentDiffusion',
                                storage='sqlite:///latentDiffusion.db',
                                load_if_exists=True)
                                # pruner=pruner)
                                
    # study.optimize(objective, n_trials=100)
    study.optimize(objective, 
                   n_trials=config['OPTUNA']['OPTIMIZE']['n_trials'], 
                   timeout=config['OPTUNA']['OPTIMIZE']['timeout'],
                   show_progress_bar=True)
                    

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

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
        from scripts.VAE.train import train, test
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
            model, trainer = train(dm, criteria, config, logger, VAE)
            test(dm, trainer, model, logger, config, save_latent=True)

        elif args.mode == 'optuna': 
            optuna_VAE(VAE, dm, config)
        
    elif args.model == 'imageDiffusion':
        from modules.imageDiffusion import ImageDiffusionModule, _calculate_FID_SCORE
        
        
        config = load_config('Diffusion', verbose=False)

        if args.mode == 'build':
            print('Build mode not implemented for imageDiffusion')
            pass

        elif args.mode == 'train':
            print_dict(config['TRAIN'])
            dm = MNISTDataModule(**config[args.mode.upper()]['DATA'], verbose=False)
            dm.setup()

            # if args.mode == 'build':\
            config['TRAIN']['MODEL']['BATCH_SIZE'] = config['TRAIN']['DATA']['BATCH_SIZE']
            plModule = ImageDiffusionModule(criteria=nn.MSELoss(), **config['TRAIN']['MODEL'])

            # load from checkpoint
            # ckpt_path = None
            # if config['TRAIN']['TRAINER'].get('continue_training', False):
            parent_log_dir = 'logs/imageDiffusion/train/'
            checkpoint = get_ckpt(parent_log_dir)
            ckpt_path = checkpoint.get('ckpt_path', None) if not checkpoint is None else None

            logger = pl.loggers.TensorBoardLogger("logs/imageDiffusion", name="train")
            trainer = pl.Trainer(logger=logger, **config['TRAIN']['TRAINER'])
            trainer.fit(plModule, dm, ckpt_path=ckpt_path)

        elif args.mode == 'inference':
            parent_log_dir = 'logs/imageDiffusion/train/'
            checkpoint = get_ckpt(parent_log_dir, config_name='hparams.yaml')
            print(checkpoint)
            # import sys
            # sys.exit(0)
            plModule = ImageDiffusionModule.load_from_checkpoint(checkpoint['ckpt_path'])
            plModule.eval()

            import yaml
            with open(checkpoint['config_path'], 'r') as file:
                hparams = yaml.safe_load(file)

            clipped_reverse_diffusion = hparams.get('CLIPPED_REVERSE_DIFFUSION', False)

            count = 0
            x_t_All, hist_all, y_all = plModule.model.sampling(20, clipped_reverse_diffusion=clipped_reverse_diffusion, y=True, device='mps', tqdm_disable=False)
            while True:
                # ask user for input
                print('Enter a number between 0 and 9')
                y = int(input('y: '))
                #select index, random where it fits
                matches = torch.where(y_all == y)[0]
                if len(matches) == 0:
                    print('No matches found for y')
                    continue
                idx = matches[torch.randint(0, len(matches), (1,))].item()
                # x_t, hist, y = plModule.model.sampling(20, clipped_reverse_diffusion=False, y=True, device='mps', tqdm_disable=False)
                x_t = x_t_All[idx]
                hist = hist_all[idx]
                y = y_all[idx]

                fig, ax = plt.subplots(1, 1, figsize=(5, 6))
                ax.imshow(x_t.squeeze().detach().cpu().numpy(), cmap='gray')
                ax.set_title(f'Sample from model, with label y={y.item()}')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.show()

                count += 1
                if count > 3:
                    break

        elif args.mode == 'optuna':
            dm = MNISTDataModule(**config[args.mode.upper()]['DATA'], verbose=False)
            dm.setup()
            optuna_sweep(ImageDiffusionModule, dm, config)

    elif args.model == 'latentDiffusion':
        
        from modules.latentDiffusion import LatentDiffusionModule
        if args.mode == 'train':
            from scripts.latentDiffusion.train import train as train_LatentDiffusion
            train_LatentDiffusion()
        
        elif args.mode == 'inference':
            print('Inference')
            parent_log_dir = 'logs/latentDiffusion/train/'
            checkpoint = get_ckpt(parent_log_dir)
            print(checkpoint)
            # hparams = yaml.safe_load(open(checkpoint['config_path'], 'r'))

            path = f"logs/VAE/train/version_{checkpoint['version_num']}/saved_latent/"
            autoencoder = torch.load(path + 'model.pth').to(torch.device('mps'))
            projector = torch.load(path + 'projector.pt')
            config = load_config('LatentDiffusion')
            print(config)
            model = LatentDiffusionModule(autoencoder=autoencoder, 
                                 scaler=None,   # TODO,
                                criteria=None,
                                classifier=None,
                                projector=projector,
                                projection=None,
                                labels=None,
                                 **config['DIFFUSE']['MODEL'])
            
            model = LatentDiffusionModule.load_from_checkpoint(checkpoint['ckpt_path'])
            model.eval()

            # criteria = VAE_Loss(config['DIFFUSE']['LOSS'])

            # import yaml
            # with open(checkpoint['config_path'], 'r') as file:
            #     hparams = yaml.safe_load(file)

            # clipped_reverse_diffusion = hparams.get('CLIPPED_REVERSE_DIFFUSION', False)

            count = 0
            x_t_All, hist_all, y_all = model.sampling(20, device='mps', tqdm_disable=False)

            while True:
                # ask user for input
                print('Enter a number between 0 and 9')
                y = int(input('y: '))
                #select index, random where it fits
                matches = torch.where(y_all == y)[0]
                if len(matches) == 0:
                    print('No matches found for y')
                    continue
                idx = matches[torch.randint(0, len(matches), (1,))].item()
                # x_t, hist, y = plModule.model.sampling(20, clipped_reverse_diffusion=False, y=True, device='mps', tqdm_disable=False)
                x_t = x_t_All[idx]
                hist = hist_all[idx]
                y = y_all[idx]

                fig, ax = plt.subplots(1, 1, figsize=(5, 6))
                ax.imshow(x_t.squeeze().detach().cpu().numpy(), cmap='gray')
                ax.set_title(f'Sample from model, with label y={y.item()}')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.show()

                count += 1
                if count > 3:
                    break


        elif args.mode == 'build':
            from scripts.latentDiffusion.train import train as train_LatentDiffusion
            train_LatentDiffusion(build_mode=True)

    else: raise NotImplementedError
