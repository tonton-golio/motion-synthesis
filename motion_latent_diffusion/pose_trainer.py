from utils_pose import *
from modules.pose_VAE import LinearPoseAutoencoder, NodeLevelGNNAutoencoder
from modules.data_modules import PoseDataModule

from utils import load_config
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

from torch_geometric.loader import DataLoader as DataLoader_geometric


cfg = config = load_config('pose_VAE')['TRAIN']

print(cfg)
if __name__ == "__main__":
    logger = TensorBoardLogger("../tb_logs2", name="TransformerMotionAutoencoder")
    datamodule = PoseDataModule(**cfg['DATA'])
    
    # if config['checkpoint_path'] == "latest" and config['load']:
    #     # SEE RUNS
    #     print("looking for latest checkpoint in ", logger.log_dir)
    #     num = logger.log_dir.split("_")[-1]
    #     print("num: ", num)
    #     folder = logger.log_dir.split("version_")[0] + "version_" + str(int(num) - 1)
    #     print("folder: ", folder)
    #     check_point_folder = folder + "/checkpoints"
    #     print(check_point_folder)

    #     checkpoints = glob.glob(f"{check_point_folder}/*.ckpt")
    #     if len(checkpoints) > 0:
    #         config.checkpoint_path = max(
    #             checkpoints, key=lambda x: int(x.split("=")[-1].split(".")[0])
    #         )
    #         print("checkpoint path: ", config['checkpoint_path'] )

   
    if config['MODEL']['MODEL_TYPE'] == "linear":
        model = LinearPoseAutoencoder(
            input_dim=66,
            hidden_dims=config.HIDDEN_DIMS,
            latent_dim=config.LATENT_DIM,
            loss_function=config.LOSS_FUNCTION,
            learning_rate=config.LEARNING_RATE,
            optimizer=config.OPTIMIZER,
            kl_weight=config.KL_WEIGHT,
        )

    elif config['MODEL']['MODEL_TYPE'] == "graph":
        model = NodeLevelGNNAutoencoder(
            model_name = 'graph',
            c_hidden = config.CHANNELS_HIDDEN,
            c_in = 3,
            c_out = config.CHANNELS_OUT,
            latent_dim=config.LATENT_DIM,
            num_layers=config.N_LAYERS,
            lr=config.LEARNING_RATE,
            optimizer=config.OPTIMIZER,
            loss_weights = {
                'kl_div': config.KL_WEIGHT,
                'mse': 1,
            },
            dp_rate=config.DROPOUT,
            checkpoint = config.checkpoint_path,
            load=config.load
        )
    elif config['MODEL']['MODEL_TYPE'] == "conv":
        pass

    else:
        raise ValueError("MODEL_TYPE not recognized")

    trainer = Trainer(
        # profiler=profiler,
        logger=logger,
        #accelerator=config.__ACCERLATOR,
        max_epochs=config['EPOCHS'],
        #devices=config.__DEVICES,
        precision=config['__PRECISION'],
        fast_dev_run=config['__FAST_DEV_RUN'],
        # callbacks=[DeviceStatsMonitor()],
        log_every_n_steps = 50,
        
        #gpus=config.__N_GPUS,
        
    )

    train_val_loss = trainer.fit(model, datamodule)
    test_loss = trainer.test(model, datamodule)
    hparams = {k: v for k, v in config.__dict__.items() if not k.startswith("__")}
    logger.log_hyperparams(
        hparams,
        metrics=dict(test_loss=test_loss[0]['test_loss']),
    )

    


