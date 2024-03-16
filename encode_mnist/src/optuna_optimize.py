# optuna optimizer for pytorch lightning for mnist
from model import Autoencoder, activation_dict
from dataset import MNISTDataModule
import config
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# to view logs: tensorboard --logdir=tb_logs
datamodule = MNISTDataModule(
        batch_size=config.BATCH_SIZE,
        # include_digits=config.__INCLUDE_DIGITS,
        transforms=dict(
            rotate_degrees=config.TRANSFORM_ROTATE_DEGREES,
            distortion_scale=config.TRANSFORM_DISTORTION_SCALE,
            translate=(config.TRANSFORM_TRANSLATEx, config.TRANSFORM_TRANSLATEy),
        ),
    )


def objective(trial: optuna.trial.Trial) -> float:
    # config.LATENT_DIM = trial.suggest_int("latent_dim", 6, 14)
    config.LEARNING_RATE = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    config.ACTIVATION = trial.suggest_categorical("activation", activation_dict.keys())
    # config.TRANSFORM_ROTATE_DEGREES = trial.suggest_float("rotate_degrees", 0.0, 8.0)
    # config.TRANSFORM_DISTORTION_SCALE = trial.suggest_float("distortion_scale", 0.0, .01)
    hparams = {k: v for k, v in config.__dict__.items() if not k.startswith("__")}
    logger = TensorBoardLogger("../tb_logs_new_model", name="MNISTAutoencoder")
    print('hparams:', hparams)
    
    loss_weights = {
        "mse": config.MSE_LOSS,
        'klDiv': config.KL_LOSS,
        "l1": config.L1_LOSS,
        # "klDiv": config.KL_LOSS,
        # "manhattan": 0.,
    }
    hparams["LOSS_DICT"] = loss_weights


    model = Autoencoder(hparams)

    trainer = Trainer(
        logger=logger,
        max_epochs=config.EPOCHS,
        #log_every_n_steps = 100,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="mse_unscaled_val")],
    )

    trainer.fit(model, datamodule)

    res = trainer.test(model, datamodule.test_dataloader())
    print('res:', res)

    # logger.log_metrics({'mse_unscaled_test': res})
    # logger.log_hyperparams(hparams)
    del hparams['LOSS_DICT']

    logger.log_hyperparams(
        params=hparams, 
        metrics={"mse_unscaled_test": res[0]['mse_unscaled_test']}
        )
    
    # # # manual dump
    import yaml
    hparams['mse_unscaled_test'] = res[0]['mse_unscaled_test']
    with open(logger.log_dir + '/hparams.yaml', 'w') as file:
         yaml.dump(hparams, file)
    


    del model
    del trainer
    del logger


    # # del datamodule
    # # more clean up
    import gc
    gc.collect() # to clear memory

    # close the open files
    import resource
    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


    return res[0]['mse_unscaled_test']

def main():
    
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200, timeout=60*60*3)
    print('best trial:', study.best_trial.params)
    


if __name__ == '__main__':
    main()