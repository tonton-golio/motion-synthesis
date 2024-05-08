
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from utils import load_config, get_ckpts
import glob


def model_selector(model_name='VAE1'):
    if model_name == 'VAE1':
        from modules.motion_VAE_1 import TransformerMotionAutoencoder as VAE
        config = load_config('motion_VAE1')
        from modules.data_modules import MotionDataModule1 as DM
    elif model_name == 'VAE2':
        from modules.motion_VAE_2 import TransformerMotionAutoencoder_Chunked as VAE
        config = load_config('motion_VAE2')
        from modules.data_modules import MotionDataModule1 as DM
    elif model_name == 'VAE3':
        from modules.motion_VAE_3_text import TransformerMotionAutoencoder as VAE
        config = load_config('motion_VAE3')
        from modules.data_modules import MotionDataModule2 as DM
    elif model_name == 'VAE4':
        from modules.motion_VAE_4 import TransformerMotionVAE as VAE
        config = load_config('motion_VAE4')
        from modules.data_modules import MotionDataModule1 as DM
    elif model_name == 'VAE5':
        from modules.motion_VAE_5 import TransformerMotionVAE as VAE
        config = load_config('motion_VAE5')
        from modules.data_modules import MotionDataModule1 as DM

    return config, VAE, DM

def train(model_name='VAE1'):
    cfg, VAE, DM = model_selector(model_name)
    logger = TensorBoardLogger("logs/VAE/", name="train")
    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
    #     #schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=20),
    # )
    print(cfg)
    # check if config.MODEL._checkpoint_path is latest if so look it up
    if cfg['MODEL']['load']:
        checkpoints = get_ckpts(logger.log_dir)
        checkpoint = checkpoints[cfg['MODEL']['_checkpoint_path']]
        cfg['MODEL']['_checkpoint_path'] = checkpoint['path']
      
    datamodule = DM(**cfg['DATA'])



    # make dict of hyperparameters
    cfg['MODEL']['seq_len'] = cfg['DATA']['seq_len']
    model = VAE(**cfg['MODEL'])

    trainer = Trainer(
        # profiler=profiler,
        logger=logger,
        accelerator=cfg['TRAINING']['accelerator'],
        max_epochs=cfg['TRAINING']['max_epochs'],
        devices=cfg['TRAINING']['devices'],
        precision=cfg['TRAINING']['precision'],
        fast_dev_run=cfg['TRAINING']['fast_dev_run'],
        enable_checkpointing=False,
        # log_every_n_steps = 60,
    )
    epochs_trained = trainer.callback_metrics.get("epoch", 0)
    trainer.fit(model, datamodule)
    res = trainer.test(model, datamodule)
    # logger.log_hyperparams(model.hparams, {"final test": res[0]})
    import yaml

    # cfg.MODEL.metrics = res[0]
    # cfg.MODEL.epochs_trained = epochs_trained

    cfg["MODEL"]["metrics"] = res[0]
    cfg["MODEL"]["epochs_trained"] = epochs_trained

    with open(logger.log_dir + "/hparams.yaml", "w") as file:
        yaml.dump(cfg, file)
