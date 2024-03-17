from model import TransformerMotionAutoencoder
from dataset import MotionDataModule
import config
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
# import summarywriter
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import DeviceStatsMonitor

import torch
# to view logs: tensorboard --logdir=tb_logs

if __name__ == "__main__":
    logger = TensorBoardLogger("../tb_logs", name="TransformerMotionAutoencoder")
    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
    #     #schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=20),
    # )
    # writer = SummaryWriter('tb_logs')

    datamodule = MotionDataModule(
        config.__FILE_LIST_PATHS, 
        config.__MOTION_PATH, 
        config.SEQ_LEN, 
        config.BATCH_SIZE,
        config.__N_WORKERS
    )

    # make dict of hyperparameters
    model = TransformerMotionAutoencoder(
        input_length=config.SEQ_LEN,
        input_dim=66,
        hidden_dim=config.HIDDEN_DIM,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        dropout=config.DROPOUT,
        latent_dim=config.LATENT_DIM,
        loss_function=config.LOSS_FUNCTION,
        learning_rate=config.LEARNING_RATE,
        optimizer=config.OPTIMIZER,
        kl_weight=config.KL_WEIGHT,
        save_animations=config.__SAVE_ANIMATIONS,
    )

    trainer = Trainer(
        # profiler=profiler,
        logger=logger,
        accelerator=config.__ACCERLATOR,
        max_epochs=config.EPOCHS,
        devices=config.__DEVICES,
        precision=config.__PRECISION,
        fast_dev_run=config.__FAST_DEV_RUN,
        # callbacks=[DeviceStatsMonitor()],
        log_every_n_steps = 16,
        
        #gpus=config.__N_GPUS,
        
    )

    trainer.fit(model, datamodule)
    test_loss = trainer.test(model, datamodule)
    hparams = {k: v for k, v in config.__dict__.items() if not k.startswith("__")}
    logger.log_hyperparams(
        hparams,
        metrics=dict(test_loss=test_loss[0]['test_loss']),
    )

    