import yaml, os, shutil
import matplotlib.pyplot as plt
import torch.nn as nn
import torch


activation_dict = {
    # soft step
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'softsign': nn.Softsign(),

    # rectifiers
    'leaky_relu': nn.LeakyReLU(),
    'ReLU': nn.ReLU(),
    'elu': nn.ELU(),
    'swish': nn.SiLU(),

    # identity
    'None': nn.Identity(),
    None: nn.Identity(),
}

def get_ckpt(parent_log_dir = 'logs/imageDiffusion/train/', config_name='config_VAE.yaml', return_all=False):
    # find available checkpoints
    
    checkpoints = {}
    for root, dirs, files in os.walk(parent_log_dir):
        for file in files:
            if file.endswith(".ckpt"):
                cp_name = file.split('_')[0]
                version_num = root.split('/')[-2].split('_')[-1]

                ckpt_path = os.path.join(root, file)
                config_path = os.path.join('/'.join(root.split('/')[:-1]), config_name)

                checkpoints[version_num] = {
                    'ckpt_path': ckpt_path,
                    'config_path': config_path
                }
                

    print(checkpoints)
    if return_all:
        return checkpoints

    checkpoint = checkpoints[input('Enter checkpoint name: ')]
    return checkpoint

def dict_merge(dct, merge_dct):
    """Recursively merge two dictionaries, dct takes precedence over merge_dct."""
    for k, v in merge_dct.items():
        if k in dct and isinstance(dct[k], dict):
            dict_merge(dct[k], v)  # merge dicts recursively
        elif k in dct:
            pass  # skip, same key already in dct
        else:
            dct[k] = v
    return dct

def load_config(name):
    
    if '.yaml' in name:
        full_name = name
    else:
        full_name = f'configs/config_{name}.yaml'

    with open(full_name, 'r') as file:
        cfg =  yaml.safe_load(file)
    
    # check if BASE in cfg, if so, append the BASE config to other configs
    if 'BASE' in cfg:
        base_cfg = cfg['BASE']
        cfg.pop('BASE')

        for key in cfg:
            cfg[key] = dict_merge(cfg[key], base_cfg)

    return cfg

def manual_config_log(log_dir, cp_file='configs/config_VAE.yaml'):
                if not os.path.exists(log_dir): os.makedirs(log_dir)
                shutil.copyfile(cp_file, f"{log_dir}/{cp_file.split('/')[-1]}")
    

def print_scientific(x):
    return "{:.2e}".format(x)

def plotUMAP(latent, labels, latent_dim, KL_weight,  save_path, show=False, max_points=5000):
    import umap
    
    if latent.shape[0] > max_points:
        idx = torch.randperm(latent.shape[0])[:max_points]
        latent = latent[idx]
        labels = labels[idx]

    reducer = umap.UMAP()
    projection = reducer.fit_transform(latent.cpu().detach().numpy())
    
    fig = plt.figure()
    plt.scatter(projection[:, 0], projection[:, 1], c=labels.cpu().numpy(), cmap='tab10', alpha=0.5, s=4)
    plt.colorbar()
    plt.title(f'UMAP projection of latent space (LD={latent_dim}, KL={print_scientific(KL_weight)})')
    
    if save_path is not None:
        plt.savefig(f'{save_path}/projection_LD{latent_dim}_KL{print_scientific(KL_weight)}.png')
    
        return projection, reducer
    elif show:
        plt.show()
    return fig

def prep_save(model, data_loaders, enable_y=False, log_dir=None):
    latent, labels = list(), list()
    for data_loader in data_loaders:
        for batch in data_loader:
            x_, y_ = batch
            _, z, _, _ = model(x_, y_) if enable_y else model(x_)
            labels.append(y_)
            latent.append(z)

    latent = torch.cat(latent, dim=0)  # maybe detach
    labels = torch.cat(labels, dim=0)

    # make covariance matrix of latent space
    cov = torch.cov(latent.T)
    cov_fig = plt.figure()
    plt.imshow(cov.cpu().detach().numpy())
    plt.colorbar()
    plt.title('Covariance matrix of latent space')
    plt.savefig(f'{log_dir}/covariance_matrix.png')
    plt.close(cov_fig)
    return latent, labels

    
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

    for k, v in kwargs.items():
        torch.save(v, f'{save_path}/{k}.pt')

