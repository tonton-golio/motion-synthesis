from model import TransformerMotionAutoencoder
from dataset import MotionDataModule
from config import config as cfg
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from global_utils import unpack_dict, pretty_print_config
# to view logs: tensorboard --logdir=tb_logs

if __name__ == "__main__":
    logger = TensorBoardLogger("../tb_logs2", name="TransformerMotionAutoencoder")
    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
    #     #schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=20),
    # )
    pretty_print_config(cfg)

    datamodule = MotionDataModule(cfg.DATA)

    # make dict of hyperparameters
    model = TransformerMotionAutoencoder(cfg.MODEL)

    trainer = Trainer(
        # profiler=profiler,
        logger = logger,
        accelerator = cfg.TRAINING.accelerator,
        max_epochs = cfg.TRAINING.max_epochs,
        devices = cfg.TRAINING.devices,
        precision=cfg.TRAINING.precision,
        fast_dev_run=cfg.TRAINING.fast_dev_run,
        # callbacks=[DeviceStatsMonitor()],
        log_every_n_steps = 16,
    )

    trainer.fit(model, datamodule)
    res = trainer.test(model, datamodule)
    logger.log_hyperparams(unpack_dict(cfg, prefix=""), metrics=res[0])
    