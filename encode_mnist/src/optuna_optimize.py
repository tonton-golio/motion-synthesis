# optuna optimizer for pytorch lightning for mnist
from model import Autoencoder, activation_dict
from dataset import MNISTDataModule
import config
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from config import config as cfg
import sys; sys.path += ['/Users/tonton/Documents/motion-synthesis']
from global_utils import unpack_dict, pretty_print_config
import yaml

# to view logs: tensorboard --logdir=tb_logs
datamodule = MNISTDataModule(cfg.DATA)


def objective(trial: optuna.trial.Trial) -> float:
    # cfg.MODEL.latent_dim = trial.suggest_int("latent_dim", 6, 14)
    # cfg.MODEL.activation = trial.suggest_categorical("activation", activation_dict.keys())
    cfg.MODEL.LOSS.klDiv = trial.suggest_float("klDiv", 0.000001, 0.001, log=True)

    logger = TensorBoardLogger("../tb_logs_kl_sweep", name="MNISTAutoencoder")
    pretty_print_config(cfg)
    
    model = Autoencoder(cfg.MODEL)

    trainer = Trainer(
        logger=logger,
        max_epochs=cfg.TRAINER.max_epochs,
        #log_every_n_steps = 100,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="mse_us_tst")],
    )

    trainer.fit(model, datamodule)

    res = trainer.test(model, datamodule.test_dataloader())
    
    # logging hyperparameters
    logger.log_hyperparams(unpack_dict(cfg, prefix=""), metrics=res[0])
    print('res:', res)
    

    # manual logging
    hparams = unpack_dict(cfg, prefix="")
    hparams['mse_us_tst'] = res[0]['mse_us_tst']

    with open(logger.log_dir + '/hparams.yaml', 'w') as file:
        yaml.dump(hparams, file)

    return res[0]['mse_us_tst']


def main():
    
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200, timeout=60*60*3)
    print('best trial:', study.best_trial.params)
    


if __name__ == '__main__':
    main()