
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.profilers import PyTorchProfiler
from utils import load_config, get_ckpts
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import yaml


def print_scientific(x):
    return "{:.2e}".format(x)

def plotUMAP(latent, latent_dim, KL_weight,  save_path, show=False, max_points=5000):
    import umap
    print('\n\nPLotting UMAP...')
    if latent.shape[0] > max_points:
        idx = torch.randperm(latent.shape[0])[:max_points]
        latent = latent[idx]
        # labels = labels[idx]

    print(f'latent shape: {latent.shape}')

    reducer = umap.UMAP()
    projection = reducer.fit_transform(latent.cpu().detach().numpy())
    
    fig = plt.figure()
    plt.scatter(projection[:, 0], projection[:, 1], 
                #c=labels.cpu().numpy(), cmap='tab10', 
                alpha=0.5, s=4)
    plt.colorbar()
    plt.title(f'UMAP projection of latent space (LD={latent_dim}, KL={print_scientific(KL_weight)})')
    
    if save_path is not None:
        plt.savefig(f'{save_path}/projection_LD{latent_dim}_KL{print_scientific(KL_weight)}.png')
    
        return projection, reducer
    elif show:
        plt.show()
    return fig

def prep_save(model, data_loaders, enable_y=False, log_dir=None):
    latent, texts = list(), list()
    for data_loader in data_loaders:
        for batch in tqdm(data_loader):
            x_, text = batch
            print(x_.shape)
            z = model.encode(x_)
            print('z.shape:', z.shape)
            latent.append(z)
            texts.append(text)

    latent = torch.cat(latent, dim=0)  # maybe detach
    texts = torch.cat(texts, dim=0)

    # make covariance matrix of latent space
    # cov = torch.cov(latent.T)
    # cov_fig = plt.figure()
    # plt.imshow(cov.cpu().detach().numpy())
    # plt.colorbar()
    # plt.title('Covariance matrix of latent space')
    # plt.savefig(f'{log_dir}/covariance_matrix.png')
    # plt.close(cov_fig)
    return latent, texts
    
def save_for_diffusion(save_path, model, **kwargs):
    """
    Save:
        'model' : 'model.pth',
        'latent' : 'z.pt',
        'labels' : 'y.pt',
        'projection' : 'projection.pt',
        'reconstruction' : 'reconstruction.pt',
        'projector' : 'projector.pt',
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(model, f'{save_path}/model.pth')

    num_params = sum(p.numel() for p in model.parameters())
    with open(f'{save_path}/num_params.txt', 'w') as file:
        file.write(f'Number of parameters: {num_params}')

    for k, v in kwargs.items():
        torch.save(v, f'{save_path}/{k}.pt')



def model_selector(model_name='VAE1'):
    from motion_latent_diffusion.modules.motion_VAE import MotionVAE as VAE
    from modules.data_modules import MotionDataModule1 as DM
    config = load_config(f'motion_{model_name}')
    return config, VAE, DM

def train(model_name='VAE1', build=False):
    cfg, VAE, DM = model_selector(model_name)

    logger = TensorBoardLogger(f"logs/{model_name}/", name="train" if not build else "build")
    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
    #     #schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=20),
    # )
    
    cfg = cfg['TRAIN'] if not build else cfg['BUILD']
    print(cfg)
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
