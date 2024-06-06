import yaml, os, shutil
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import streamlit as st

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

def get_ckpt(parent_log_dir = 'logs/imageDiffusion/train/', config_name='config_VAE.yaml', return_all=False, with_streamlit=False):
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
                    'config_path': config_path,
                    'version_num': version_num,
                }
    
    # sort by version number
    checkpoints = dict(sorted(checkpoints.items(), key=lambda item: int(item[0])))
    for k, v in checkpoints.items():
        if with_streamlit:
            pass#st.write(f"{k}: {v['ckpt_path']}")
        else:
            print(f"{k}: {v['ckpt_path']}")
    if return_all:
        return checkpoints
    if with_streamlit:
        choice = st.selectbox('Select checkpoint', list(checkpoints.keys()))
        
    else:
        choice = input('Enter checkpoint idx/key: ')
        # checkpoint = checkpoints[input('Enter checkpoint idx/key: ')]

    

    ckpt =  checkpoints.get(choice, None)
    
    
    return ckpt

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

def print_dict(d, indent=0, num_spaces=4):
    for key, value in d.items():
        print(' ' * (indent * num_spaces) + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1, num_spaces)
        else:
            print(': ' + str(value))

def load_config(name, verbose=True):
    
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

    if verbose:
        print_dict(cfg)


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

    num_params = sum(p.numel() for p in model.parameters())
    with open(f'{save_path}/num_params.txt', 'w') as file:
        file.write(f'Number of parameters: {num_params}')

    for k, v in kwargs.items():
        torch.save(v, f'{save_path}/{k}.pt')



# load latent space from VAE
# find saved latent vectors

import os
import matplotlib.pyplot as plt
import glob
import torch

def find_saved_latent(path = f"logs/VAE/train/"):
    """
    Find saved latent vectors from VAE training
    """

    VAE_data = {}
    for version in os.listdir(path):
        version_num = version.split('_')[-1]
        contents = os.listdir(f"{path}{version}")
        base_path = os.path.join(path, version, )

        if 'saved_latent' in contents:
            cfg_file = None  # get config file
            for file in contents:
                if 'config' in file and file.endswith('.yaml'):
                    cfg_file = file
                    break
            
            projection = None  # get projection image
            for file in contents:
                if 'projection' in file and file.endswith('.png'):
                    projection = file
                    break

            checkpoints = glob.glob(f"{base_path}/checkpoints/*")  # check for checkpoints
            saved_latent = os.listdir(os.path.join(base_path, 'saved_latent'))  # open saved_latent and check whats inside

            VAE_data[version_num] = {
                'saved_latent' : saved_latent,
                'paths' : {
                    'config' : os.path.join(base_path, cfg_file),
                    'saved_latent' : os.path.join(base_path, 'saved_latent'),
                    'projection' : os.path.join(base_path, projection),
                    'checkpoints' : checkpoints,
                    'log' : base_path,
                },
                'contents' : contents
            }

    return VAE_data

def show_saved_latent_info(data, return_fig=False):

    saved_latent_info = {}

    for version, info in data.items():
        saved_latent = info['saved_latent']
        saved_latent_info[version] = {
            'num_files' : len(saved_latent),
            'size' : None,
            'min' : None,
            'max' : None,
            'std_dev' : None,
            'projection' : None
        }

        for file in saved_latent:
            # get size of file
            # get min and max values
            # get std dev
            pass

        # projection_image = plt.imread(info['paths']['projection'])
        saved_latent_info[version]['projection'] = info['paths']['projection']

    fig, ax = plt.subplots(2, len(data), figsize=(20, 10))
    if len(data) == 1:
        ax = ax.reshape(2, 1)
    for i, (version, info) in enumerate(data.items()):
        ax[0, i].imshow(plt.imread(info['paths']['projection']))
        ax[0, i].set_title(f"Version {version}")
        ax[0, i].axis('off')

        ax[1, i].text(0.5, 0.5, f"Num Files: {saved_latent_info[version]['num_files']}", ha='center', va='center')
        ax[1, i].axis('off')
    if return_fig:
        return fig

    plt.show()

def latent_picker():
    data = find_saved_latent()
    # print(data)

    # if the user has difficulty picking a version, show info
    ## info to show: projection image, config file, saved_latent vectors (how many?, how big?, min/max values?, std dev?, etc.)
    ## also show the checkpoint files

    show_saved_latent_info(data)

    # ask user for input of version number
    print("Please enter the version number you would like to use: ")
    for version in data.keys():
        print(f"\t{version}")
    version = input('Version: ')

    return data[version], version

def load_latent(data_version):
    path = data_version['paths']['saved_latent']
    z = torch.load(path + '/z.pt')#.to(torch.device('mps'))
    y = torch.load(path + '/y.pt')#.to(torch.device('mps'))
    autoencoder = torch.load(path + '/model.pth').to(torch.device('mps'))
    projector = torch.load(path + '/projector.pt')
    projection = torch.load(path + '/projection.pt')

    # load checkpoint
    # checkpoint = torch.load(data_version['paths']['checkpoints'][0])
    # autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])

    return z, y , autoencoder, projector, projection

 