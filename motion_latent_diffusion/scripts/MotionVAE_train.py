
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.profilers import PyTorchProfiler
from utils import load_config, get_ckpts, get_ckpt, print_header
from utils import prep_save, plotUMAP, save_for_diffusion
import matplotlib.pyplot as plt
import torch
import yaml, os
from modules.MotionVAE import MotionVAE as VAE
from modules.MotionData import MotionDataModule1 as DM


def train(model_name='VAE1', build=False):
    cfg= load_config('motion_VAE', mode='TRAIN', model_type=model_name[3:])

    logger = TensorBoardLogger(f"logs/MotionVAE/{model_name}/", name="train" if not build else "build")
    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
    #     #schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=20),
    # )
    
    ckpt = None
    if cfg['FIT']['load_checkpoint'] and not build:
        path = logger.log_dir.split("version_")[0]
        ckpt = get_ckpt(path)
      
    datamodule = DM(**cfg['DATA'])

    # make dict of hyperparameters
    cfg['MODEL']['seq_len'] = cfg['DATA']['seq_len']
    model = VAE(model_name, verbose = False if not build else True, **cfg['MODEL'])

    new_path = 'logs/MotionVAE/VAE1/train/version_89/checkpoints/epoch=299-step=38700_renamed.ckpt'
    cpkt_loaded = torch.load(new_path, map_location='mps')
    model.load_state_dict(cpkt_loaded)


    print_header(f"Training {model_name}")
    trainer = Trainer(
        # profiler=profiler,
        logger=logger,
        **cfg['TRAINER']
    )
    epochs_trained = trainer.callback_metrics.get("epoch", 0)
    trainer.fit(model, datamodule, 
                ckpt_path=ckpt,
    )
    
    model.eval()
    res = trainer.test(model, datamodule)
    logger.log_hyperparams(model.hparams, res[0])

    # cfg.MODEL.metrics = res[0]
    # cfg.MODEL.epochs_trained = epochs_trained

    cfg["MODEL"]["metrics"] = res[0]
    cfg["MODEL"]["epochs_trained"] = epochs_trained

    if not build:
        with open(logger.log_dir + "/hparams.yaml", "w") as file:
            yaml.dump(cfg, file)

    return datamodule, trainer, model, logger, cfg

def test(dm , trainer, model, logger, config, save_latent=False):

    # clean up
    del trainer
    model.eval()
    if save_latent:
        
        KL_weight = config['MODEL']['LOSS']['DIVERGENCE_KL']
        # dataloaders = [dm.test_dataloader(), dm.train_dataloader(), dm.val_dataloader()]
        # latent, texts = prep_save(model, dataloaders, enable_y=False, log_dir=logger.log_dir)

        latent_train, texts_train, action_group_train, action_train = prep_save(model, dm.train_dataloader(), log_dir=logger.log_dir)
        latent_test, texts_test, action_group_test, action_test = prep_save(model, dm.test_dataloader(), log_dir=logger.log_dir)
        latent_val, texts_val, action_group_val, action_val = prep_save(model, dm.val_dataloader(), log_dir=logger.log_dir)


        latent_dim = torch.prod(torch.tensor(latent_train.shape[1:]))
        latent_train = latent_train.view(-1, latent_dim)
        latent_test = latent_test.view(-1, latent_dim)
        latent_val = latent_val.view(-1, latent_dim)

        projection, reducer = plotUMAP(latent_test, latent_dim, KL_weight, logger.log_dir, labels=action_group_test, show=False, max_points=5000)
        
        save_for_diffusion(save_path=logger.log_dir+'/saved_latent', 
                           model = model, 
                           projection = projection, 
                           projector = reducer, 
                           latent_train = latent_train,
                            latent_test = latent_test,
                            latent_val = latent_val,
                            texts_train = texts_train,
                            texts_test = texts_test,
                            texts_val = texts_val,
                            action_group_train = action_group_train,
                            action_group_test = action_group_test,
                            action_group_val = action_group_val,
                            action_train = action_train,
                            action_test = action_test,
                            action_val = action_val,
                            )
