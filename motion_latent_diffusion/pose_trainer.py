from utils_pose import *
from modules.pose_VAE import PoseVAE
from modules.pose_data import PoseDataModule
import argparse
from utils import load_config, unpack_nested_dict, get_ckpts
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger



if __name__ == "__main__":
    # add arguments for model and mode
    parser = argparse.ArgumentParser(description='Run the model')
    parser.add_argument('--model', type=str, default='LINEAR', help='Model to run')
    args = parser.parse_args()
    model_type = args.model.upper()
    assert model_type in ['LINEAR', 'GRAPH', 'CONV' ]  # assert valid
    

    cfg = load_config('pose_VAE', mode='TRAIN', model_type=model_type)
    logger = TensorBoardLogger("logs/PoseVAE", name=model_type)
    datamodule = PoseDataModule(**cfg['DATA'])
    
    if cfg['FIT']['load_checkpoint']:
        path = logger.log_dir.split("version_")[0]
        ckpts = get_ckpts(path)

        for i, ckpt in enumerate(ckpts):
            print(i, ckpt)

        try:
            ckpt_num = int(input("Enter checkpoint number: "))
            ckpt = ckpts[ckpt_num]['path']
        except:
            ckpt = None
    else: ckpt = None

    if model_type == "LINEAR":  model = PoseVAE(model_type, **cfg['MODEL'])
    elif model_type == "GRAPH": model = PoseVAE(model_type, **cfg['MODEL'])
    elif model_type == "CONV":  model = PoseVAE(model_type, **cfg['MODEL'])
    else: raise ValueError("MODEL not recognized")

    trainer = Trainer(
        # profiler=profiler,
        logger=logger,
        **cfg['TRAINER'])

    train_val_loss = trainer.fit(model, datamodule, ckpt_path=ckpt)
    trainer.test(model, datamodule)
    test_loss = model.test_losses
    test_loss = torch.stack(test_loss).mean(0).item()
    print(test_loss)

    hparams = unpack_nested_dict(cfg)
    
    print(hparams)
    logger.log_hyperparams(
        hparams,
        metrics=dict(test_loss=test_loss),
    )

