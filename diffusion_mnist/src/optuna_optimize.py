import optuna
from optuna.integration import PyTorchLightningPruningCallback
from dataset import *
from model import LatentDiffusionModel
import pytorch_lightning as pl
import config
from pytorch_lightning.loggers import TensorBoardLogger
import yaml



# Objective function to be optimized
def objective(trial: optuna.trial.Trial) -> float:
    # Define Optuna hyperparameters here
    config.LEARNING_RATE = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config.HIDDEN_DIM = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512, 1024])
    # config.NHIDDEN = trial.suggest_categorical("nhidden", [4, 8,])
    # Updated to use suggest_float for uniform distribution
    # config.DP_RATE = trial.suggest_float("dp_rate", 0.0, 0.5)
    config.TIMESTEPS = trial.suggest_categorical("timesteps", [50, 100, 200, 400])
    config.TIME_EMBEDDING_DIM = trial.suggest_categorical("time_embedding_dim", [8, 16])
    config.TARGET_EMBEDDING_DIM = trial.suggest_categorical("target_embedding_dim", [4, 8, 16])
    config.EPSILON = trial.suggest_float("epsilon", 1e-3, 1e-0, log=True)


    hparams = {k: v for k, v in config.__dict__.items()}# if not k.startswith("__")}
    
    for k, v in hparams.items():
        print(f"{k}: {v}")

    

    # Load latent space
    # Data setup


    # Model instantiation with Optuna suggested hyperparameters
    model = LatentDiffusionModel(decoder=decoder, **hparams)

    # Logger setup
    logger = TensorBoardLogger("../tb_logs2", name="LatentDiffusionMNIST")

    # Trainer setup with PyTorch Lightning Pruning Callback
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.N_EPOCHS,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")], # Adjust monitor metric as needed
    )

    # Fit model
    trainer.fit(model, dm)

    # Here we assume val_loss is the metric to minimize
    val_loss = trainer.callback_metrics["val_loss"].item()
    epochs_trained = trainer.current_epoch

    
    hparams['EPOCHS'] = epochs_trained

    # write hparams manually
    hparams['val_loss'] = val_loss
    path = logger.log_dir
    with open(f"{path}/hparams.yaml", "w") as file:
        yaml.dump(hparams, file)

    del model, trainer, logger, dm, X, y, z, reconstruction, projector, projection, decoder


    return val_loss

if __name__ == '__main__':

    X, y, z, reconstruction, projector, projection, decoder = get_latent_space(
        config.LATENT_DIM, config.N, config.__V, plot=False
    )
    dm = LatentSpaceDataModule(z, y, batch_size=config.BATCH_SIZE)
    dm.setup()
    
    # optuna.logging.set_verbosity(optuna.logging.WARNING)  # Adjust logging level if needed
    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=100, timeout=600)
    # print("Best trial:", study.best_trial.params)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=60 * 30)
    print("Best trial:", study.best_trial.params)
