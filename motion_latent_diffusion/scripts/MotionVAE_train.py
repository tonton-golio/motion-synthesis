
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.profilers import PyTorchProfiler
from utils import load_config, get_ckpts
from utils import prep_save, plotUMAP, save_for_diffusion
import matplotlib.pyplot as plt
import torch
import yaml, os
from modules.MotionVAE import MotionVAE as VAE
from modules.MotionData import MotionDataModule1 as DM


def train(model_name='VAE1', build=False):
    cfg= load_config('motion_VAE', mode='TRAIN', model_type=model_name[3:])
    # cfg = cfg['TRAIN'] if not build else cfg['BUILD']
    print(cfg)

    logger = TensorBoardLogger(f"logs/MotionVAE/{model_name}/", name="train" if not build else "build")
    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
    #     #schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=20),
    # )
    
    
    # check if config.MODEL._checkpoint_path is latest if so look it up
    if cfg['MODEL']['load']:
        checkpoints = get_ckpts('/'.join(logger.log_dir.split('/')[:-1]))
        checkpoint = checkpoints[cfg['MODEL']['_checkpoint_path']]
        cfg['MODEL']['_checkpoint_path'] = checkpoint['path']
        print(f"Loading checkpoint from {cfg['MODEL']['_checkpoint_path']}")
      
    datamodule = DM(**cfg['DATA'])

    # make dict of hyperparameters
    cfg['MODEL']['seq_len'] = cfg['DATA']['seq_len']
    model = VAE(model_name, verbose = False if not build else True, **cfg['MODEL'])

    trainer = Trainer(
        # profiler=profiler,
        logger=logger,
        **cfg['TRAINER']
    )
    epochs_trained = trainer.callback_metrics.get("epoch", 0)
    trainer.fit(model, datamodule, 
                ckpt_path=cfg['MODEL']['_checkpoint_path'] if cfg['MODEL']['load'] else None)
    
    print('i will test now, please wait (did we already test?)')
    res = trainer.test(model, datamodule)
    # logger.log_hyperparams(model.hparams, {"final test": res[0]})
    

    # cfg.MODEL.metrics = res[0]
    # cfg.MODEL.epochs_trained = epochs_trained

    cfg["MODEL"]["metrics"] = res[0]
    cfg["MODEL"]["epochs_trained"] = epochs_trained

    if not build:
        with open(logger.log_dir + "/hparams.yaml", "w") as file:
            yaml.dump(cfg, file)

    return datamodule, trainer, model, logger, cfg

def test(dm , trainer, model, logger, config, save_latent=False):
    # test
    res = trainer.test(model, datamodule=dm)
    print(res)
    logger.log_hyperparams(model.hparams, res[0])
    print(config)
    # save_latent
    if save_latent:
        dataloaders = [dm.test_dataloader(), dm.train_dataloader(), dm.val_dataloader()]
        KL_weight = config['MODEL']['LOSS']['DIVERGENCE_KL']
        latent, texts = prep_save(model, dataloaders, enable_y=False, log_dir=logger.log_dir)
        print(latent)
        print(latent.shape)
        latent_dim = torch.prod(torch.tensor(latent.shape[1:]))
        print('latent_dim:', latent_dim )
        latent = latent.view(-1, latent_dim)

        projection, reducer = plotUMAP(latent, latent_dim, KL_weight, logger.log_dir, show=False, max_points=5000)
        
        save_for_diffusion(save_path=logger.log_dir+'/saved_latent', model = model, z = latent, projection = projection, projector = reducer, texts=texts )
