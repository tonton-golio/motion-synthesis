from dataset import *
from model import LatentDiffusionModel
import pytorch_lightning as pl
from config import config as cfg
from pytorch_lightning.loggers import TensorBoardLogger

import sys; sys.path += ['/Users/tonton/Documents/motion-synthesis']
from encode_mnist.src.config import config as cfg_ae
from global_utils import unpack_dict, pretty_print_config

if __name__ == "__main__":
    logger = TensorBoardLogger("../tb_logs3", name="LatentDiffusionMNIST")
    X, y, z, reconstruction, projector, projection, decoder = get_latent_space(
        cfg_ae.MODEL.latent_dim, cfg.DATA.N, cfg.DATA.V, plot=False)
    dm = LatentSpaceDataModule(z, y, batch_size=cfg.DATA.batch_size)
    dm.setup()

    pretty_print_config(cfg)
    cfg.MODEL.decoder = decoder
    cfg.MODEL.latent_dim = cfg_ae.MODEL.latent_dim

    model = LatentDiffusionModel(
        latent_dim=cfg.MODEL.latent_dim,
        hidden_dim=cfg.MODEL.hidden_dim,
        nhidden=cfg.MODEL.n_hidden,
        timesteps=cfg.MODEL.timesteps,
        time_embedding_dim=cfg.MODEL.time_embedding_dim,
        target_embedding_dim=cfg.MODEL.target_embedding_dim,
        epsilon=cfg.MODEL.epsilon,
        dp_rate=cfg.MODEL.dropout,
        lr=cfg.MODEL.learning_rate,
        decoder=decoder,
        noise_multiplier=cfg.MODEL.noise_multiplier,
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg.TRAINER.max_epochs,
    )
    trainer.fit(model, dm)
    epochs_trained = trainer.current_epoch

    # test
    res = trainer.test(model, dm.test_dataloader())

    # logging hyperparameters
    logger.log_hyperparams(unpack_dict(cfg, prefix=""), metrics=res[0])
    print("done")
