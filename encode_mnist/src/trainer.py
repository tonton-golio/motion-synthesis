# trainer (pytorch lightning) for mnist
from model import Autoencoder
from dataset import MNISTDataModule
from config import config as cfg
import sys; sys.path += ['/Users/tonton/Documents/motion-synthesis']
from global_utils import unpack_dict, pretty_print_config
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
# to view logs: tensorboard --logdir=tb_logs

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
