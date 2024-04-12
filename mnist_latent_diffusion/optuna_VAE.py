# optuna optimizer for pytorch lightning for mnist
from VAE.model import activation_dict
from VAE.dataset import MNISTDataModule
from loss import VAE_Loss
from VAE.model import VAE

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import yaml

# Load config file and use it to create the data module
with open('configs/config_VAE.yaml', 'r') as file:
    config = yaml.safe_load(file)

# set up data module
dm = MNISTDataModule(**config['OPTUNA']['DATA'], verbose=False)
dm.setup()


def objective(trial: optuna.trial.Trial) -> float:
    # these are our sweep parameters
    ld = trial.suggest_categorical("latent_dim", [4, 6, 8, 10, 12, 14])

    acts = ['tanh', 'sigmoid', 'softsign', 'leaky_relu', 'ReLU', 'elu']
    act = trial.suggest_categorical("activation", acts)
    klDiv = trial.suggest_float("klDiv", 1e-6, 1e-4, log=True)
    
    # activation func
    config['OPTUNA']['MODEL']['activation'] = act

    # set up loss function
    loss_dict = {
        'RECONSTRUCTION_L2': 1,
        'DIVERGENCE_KL': klDiv,
    }
    criteria = VAE_Loss(loss_dict)

    # instantiate the logger
    logger = TensorBoardLogger("logs/", name="mnistVAEoptuna")
    
    # Create the model
    model = VAE(criteria, **config['OPTUNA']['MODEL'], LATENT_DIM=ld)

    trainer = Trainer(
        logger=logger,
        **config['OPTUNA']['TRAINER'],
        enable_checkpointing=False,
        val_check_interval=0.5,
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

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200, timeout=60*60*3)
    print('best trial:', study.best_trial.params)
    
if __name__ == '__main__':
    main()