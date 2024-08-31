
import argparse
from utils import load_config
import torch
from utils import load_config, unpack_nested_dict, get_ckpts
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from modules.PoseVAE import PoseVAE
from modules.PoseData import PoseDataModule
import sys
sys.path.append('/Users/tonton/Documents/motion-synthesis/')
for p in sys.path:
    print(p)



if __name__ == "__main__":
    # add arguments for model and mode
    parser = argparse.ArgumentParser(description='Run the model')
    parser.add_argument('--model_name', type=str, default='VAE', help='Model to run')
    parser.add_argument('--mode', type=str, default='train', help='Mode to run')
    args = parser.parse_args()
    assert args.model_name.upper() in ['VAE1', 'VAE4', 'VAE5', 'VAE6', # MotionVAE
                          'LD_VAE1', 'LD_VAE4', 'LD_VAE5', # Latent Diffusion
                            'POSELINEAR', 'POSEGRAPH', 'POSECONV'  # PoseVAE
                          ]
    assert args.mode.lower() in ['train', 'build', 'inference', 'optuna']

    if args.model_name[:3] == 'VAE':
        if args.mode == 'train':
            # train the model
            from scripts.MotionVAE_train import train
            datamodule, trainer, model, logger, cfg = train(model_name=args.model_name)

            # test the model
            from scripts.MotionVAE_train import test
            test(datamodule, trainer, model, logger, cfg, save_latent=True)
        elif args.mode == 'build':
            from scripts.MotionVAE_train import train
            train(model_name=args.model_name, build=True)

    elif args.model_name[:2] == 'LD':
        if args.mode == 'train':
            from motion_latent_diffusion.scripts.MotionLD_train import train
            
            # 
            train(VAE_version=args.model_name.split('_')[1])
        elif args.mode == 'inference':
            from motion_latent_diffusion.scripts.MotionLD_train import inference
            inference(VAE_version=args.model_name.split('_')[1])

    elif args.model_name[:4] == 'POSE':
        # if args.mode = 'train', train the model
        model_name = args.model_name[4:]
    
        cfg = load_config('pose_VAE', mode='TRAIN', model_type=model_name)
        logger = TensorBoardLogger("logs/PoseVAE", name=model_name)
        datamodule = PoseDataModule(**cfg['DATA'])
        datamodule.setup('stage')
        ckpt = None
        if cfg['FIT']['load_checkpoint']:
            path = logger.log_dir.split("version_")[0]
            ckpt = get_ckpts(path)

        # if model_name == "LINEAR":  model = PoseVAE(model_name, **cfg['MODEL'])
        # elif model_name == "GRAPH": model = PoseVAE(model_name, **cfg['MODEL'])
        # elif model_name == "CONV":  model = PoseVAE(model_name, **cfg['MODEL'])
        # else: raise ValueError("MODEL not recognized")
        test_video = datamodule.test_video
        model = PoseVAE(model_name, test_video=test_video, **cfg['MODEL'])

        trainer = Trainer(
            logger=logger,
            **cfg['TRAINER'])

        train_val_loss = trainer.fit(model, datamodule, ckpt_path=ckpt)
        trainer.test(model, datamodule)
        test_loss = model.test_losses
        test_loss = torch.stack(test_loss).mean(0).item()

        hparams = unpack_nested_dict(cfg)
        
        logger.log_hyperparams(
            hparams,
            metrics=dict(test_loss=test_loss),
        )

