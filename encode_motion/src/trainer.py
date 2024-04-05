from model import TransformerMotionAutoencoder

# from modelnew import TransformerMotionAutoencoder_Concatenated, TransformerMotionAutoencoder_Chunked
from dataset import MotionDataModule
from config import config as cfg

# from config_chunked import config as cfg
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from global_utils import unpack_dict, pretty_print_config

# to view logs: tensorboard --logdir=tb_logs
import glob

if __name__ == "__main__":
    logger = TensorBoardLogger("../tb_logs5", name="TransformerMotionAutoencoder")
    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
    #     #schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=20),
    # )
    pretty_print_config(cfg.MODEL)
    # check if config.MODEL._checkpoint_path is latest if so look it up
    if cfg.MODEL._checkpoint_path == "latest" and cfg.MODEL.load:
        # SEE RUNS
        print("looking for latest checkpoint in ", logger.log_dir)
        num = logger.log_dir.split("_")[-1]
        print("num: ", num)
        folder = logger.log_dir.split("version_")[0] + "version_" + str(int(num) - 1)
        print("folder: ", folder)
        check_point_folder = folder + "/checkpoints"
        print(check_point_folder)

        checkpoints = glob.glob(f"{check_point_folder}/*.ckpt")
        if len(checkpoints) > 0:
            cfg.MODEL._checkpoint_path = max(
                checkpoints, key=lambda x: int(x.split("=")[-1].split(".")[0])
            )
            print("checkpoint path: ", cfg.MODEL._checkpoint_path)

    datamodule = MotionDataModule(cfg.DATA)

    # make dict of hyperparameters
    cfg.MODEL.seq_len = cfg.DATA.seq_len
    model = TransformerMotionAutoencoder(cfg.MODEL)

    trainer = Trainer(
        # profiler=profiler,
        logger=logger,
        accelerator=cfg.TRAINING.accelerator,
        max_epochs=cfg.TRAINING.max_epochs,
        devices=cfg.TRAINING.devices,
        precision=cfg.TRAINING.precision,
        fast_dev_run=cfg.TRAINING.fast_dev_run,
        # callbacks=[DeviceStatsMonitor()],
        # log_every_n_steps = 60,
    )
    epochs_trained = trainer.callback_metrics.get("epoch", 0)
    trainer.fit(model, datamodule)
    res = trainer.test(model, datamodule)
    logger.log_hyperparams(unpack_dict(cfg.MODEL, prefix=""), metrics=res[0])

    import yaml

    cfg.MODEL.metrics = res[0]
    cfg.MODEL.epochs_trained = epochs_trained

    with open(logger.log_dir + "/hparams.yaml", "w") as file:
        yaml.dump(unpack_dict(cfg.MODEL, prefix=""), file)
