# trainer (pytorch lightning) for mnist
from model import Autoencoder
from dataset import MNISTDataModule
from config import config as cfg
from config import upack_dict as unpack_dict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
# to view logs: tensorboard --logdir=tb_logs

def pretty_print_config(config):
    print('cfg:',)
    for k, v in unpack_dict(config, prefix="").items():
        print(k.ljust(33), ':', v)
    print()

if __name__ == "__main__":
    logger = TensorBoardLogger("../tb_logs", name="MNISTAutoencoder")
    pretty_print_config(cfg)

    datamodule = MNISTDataModule(cfg.DATA)
    
    model = Autoencoder(cfg.MODEL)

    trainer = Trainer(
        logger=logger,
        max_epochs=cfg.TRAINER.max_epochs,
    )

    trainer.fit(model, datamodule)
    epochs_trained = trainer.current_epoch
    res = trainer.test(model, datamodule.test_dataloader())
    
    # logging hyperparameters
    logger.log_hyperparams(unpack_dict(cfg, prefix=""), metrics=res[0])
