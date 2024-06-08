import numpy as np
import torch
import torch.nn as nn
import os, glob, yaml
from os.path import join as pjoin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import subprocess
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# general utils

def print_dict(d, indent=0, num_spaces=4, title=None):
    if title is not None:
        print(title)
    for key, value in d.items():
        print(' ' * (indent * num_spaces) + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1, num_spaces)
        else:
            print(': ' + str(value))

def print_header(title='Title', small=False):
    if not small:
        print()
        print('=' * 80)
        print(title)
    
    else:
        print()
        print('-' * 80)
        print(title)

def get_ckpts(log_dir):
    """
    Get all checkpoints in the log directory
    """
    folders = glob.glob(f"{log_dir}/*") # like: version_0, ...
    ckpts = {}
    count = {'success': 0, 'fail': 0}
    for folder in folders:
        
        try:
            path = glob.glob(f"{folder}/checkpoints/*.ckpt")[0]
            file = path.split('/')[-1]
            config = glob.glob(f"{folder}/*.yaml")
            epoch = int(file.split('-')[0].split('=')[-1])
            step = int(file.split('=')[2].split('.')[0])

            version = folder.split('/')[-1]
            version_num = int(version.split('_')[-1])
            ckpts[version_num] = {
                'epoch' : epoch,
                'step' : step,
                'path' : path,
                'cfg_path' : config,
            }
            count['success'] += 1
        except:
            count['fail'] += 1


    if count['success'] == 0:
        ckpts['latest'] = {'path': None, 'cfg_path': None, 'epoch': 0, 'step': 0}
    else:
        ckpts['latest'] = ckpts[max(ckpts.keys(), key=int)]
    
    return ckpts

def get_ckpt(path, verbose=True):

    if verbose: print_header('Checkpoints')
    ckpts = get_ckpts(path)
    # sort by key
    # ckpts = dict(sorted(ckpts.items()))
    print("Available checkpoints:")
    for k, v in ckpts.items():
        print(k, {k: v for k, v in v.items() if 'path' not in k})

    try:
        ckpt_num = int(input("Enter checkpoint number: "))
        print(ckpt_num)
        print(ckpts[ckpt_num])
        ckpt = ckpts[ckpt_num]['path']
        print("Loading checkpoint:", ckpt)
    except:
        ckpt = None
        print("No checkpoint loaded")
    return ckpt

def dict_merge(dct, merge_dct):
    """
    Recursively merge two dictionaries, dct takes precedence over merge_dct.
    """
    for k, v in merge_dct.items():
        if k in dct and isinstance(dct[k], dict):
            dict_merge(dct[k], v)  # merge dicts recursively
        elif k in dct:
            pass  # skip, same key already in dct
        else:
            dct[k] = v
    return dct

def load_config(name, mode=None, model_type=None, verbose=True):
    """
    Load config file and return the config dictionary

    Args:
    - name (str): name of the config file
    - mode (str): mode like 'TRAIN', 'BUILD', 'INFERENCE'
    - model_type (str): model type like 'CONV', 'LINEAR', 'GRAPH'
    """
    if verbose: print_header('Config')
    if verbose: print(f"Loading config for {name}, mode: {mode}, model_type: {model_type}")
    # load config file
    full_name = name if '.yaml' in name else f'configs/config_{name}.yaml'
    with open(full_name, 'r') as file:
        cfg =  yaml.safe_load(file)

    # check if BASE in cfg, if so, append the BASE config to other configs
    if 'BASE' in cfg:
        if verbose:
            print('Merging with base')
        base_cfg = cfg['BASE']
        cfg.pop('BASE')
        for key in cfg: 
            cfg[key] = dict_merge(cfg[key], base_cfg)

    # check if mode is specified, if so, only return that mode
    if mode is not None:
        if verbose:
            print(f"Returning config for mode: {mode}")
        cfg = cfg[mode]

    if model_type is not None:
        # delete other models except base and model_type
        to_pop = []
        for key in cfg:
            if 'MODEL' in key and model_type not in key and 'BASE' not in key:
                # cfg.pop(key)
                to_pop.append(key)

        if verbose: print(f"Popping entries: {to_pop}")
        for key in to_pop:
            cfg.pop(key)

        # merge with base
        if 'MODEL_BASE' in cfg:
            if verbose:
                print('Merging with base')
            model_base_cfg = cfg['MODEL_BASE']
            cfg.pop('MODEL_BASE')
            to_pop = []
            for key in cfg:
                
                if f'MODEL_{model_type}' in key: 
                    temp = dict_merge(cfg[key], model_base_cfg)

                if 'MODEL' in key: 
                    to_pop.append(key)
            
            if verbose: print(f"Popping entries: {to_pop}")
            for key in to_pop:
                cfg.pop(key)

            cfg['MODEL'] = temp


    if verbose: print_dict(cfg, title='Final config')
    return cfg

def unpack_nested_dict(d, unpacked=None, prefix=''):
    if unpacked is None:
        unpacked = {}
    for k, v in d.items():
        # print(f"prefix: {prefix}")
        if isinstance(v, dict):
            unpacked = unpack_nested_dict(v, unpacked, prefix=f"{prefix}{k}_")
        else:
            unpacked[f"{prefix}{k}"] = v
    return unpacked

# nn utils
activation_dict = {
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "elu": nn.ELU(),
    "swish": nn.SiLU(),
    "mish": nn.Mish(),
    "softplus": nn.Softplus(),
    "softsign": nn.Softsign(),
    "softmax": nn.Softmax(),
    "softmin": nn.Softmin(),
    "softshrink": nn.Softshrink(),}

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# logging utils (plotting)
def plot_3d_pose(data, index, ax=None):
    """
    Plot a single 3D pose.

    Args:
    - data (np.array): 3D pose data (seq_len, joints_num, 3)
    - index (int): index of the frame to plot
    """

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

    kinematic_chain = [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    ]

    colors = [
        "red",
        "blue",
        "black",
        "red",
        "blue",
    ]

    for i, (chain, color) in enumerate(zip(kinematic_chain, colors)):
        ax.plot3D(
            data[index, chain, 0],
            data[index, chain, 1],
            data[index, chain, 2],
            linewidth=4.0 if i < 5 else 2.0,
            color=color,
        )

    plt.axis("off")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    return ax

def plot_xzPlane(ax, minx, maxx, miny, minz, maxz):
    ## Plot a plane XZ
    verts = [
        [minx, miny, minz],
        [minx, miny, maxz],
        [maxx, miny, maxz],
        [maxx, miny, minz],
    ]
    xz_plane = Poly3DCollection([verts])
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
    ax.add_collection3d(xz_plane)

def init_3d_plot(ax, fig, title, radius=2):
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_ylim3d([0, radius])
    ax.set_zlim3d([0, radius])
    # print(title)
    title_sp = title.split(" ")
    if len(title_sp) > 10:
        title = "\n".join([" ".join(title_sp[:10]), " ".join(title_sp[10:])])
    fig.suptitle(title, fontsize=20)
    ax.grid(b=False)

def plot_trajec(trajec, index, ax):
    ax.plot3D(
        trajec[:index, 0] - trajec[index, 0],
        np.zeros_like(trajec[:index, 0]),
        trajec[:index, 1] - trajec[index, 1],
        linewidth=1.0,
        color="blue",
    )

def plot_3d_motion_animation(
    data,
    title,
    figsize=(10, 10),
    fps=20,
    radius=2,
    save_path="test.mp4",
    velocity=False,
    save_path_2=None,):
    #     matplotlib.use('Agg')
    data = data.copy().reshape(len(data), -1, 3)  # (seq_len, joints_num, 3)

    # cut tail, if equal
    # print(data.shape)
    for i in range(data.shape[0]-1):
        # if i > 100:
            if (data[i] == data[i+1]).all():
                data = data[:i]
                break
    # print(data.shape)

    if velocity:
        data = np.cumsum(data, axis=0)

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111, projection="3d")
    init_3d_plot(ax, fig, title, radius)
    MINS, MAXS = data.min(axis=0).min(axis=0), data.max(axis=0).max(axis=0)

    data[:, :, 1] -= MINS[1]  # height offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]  # centering
    data[..., 2] -= data[:, 0:1, 2]  # centering

    def update(index):
        # ax.dist = 7.5
        def do_it_all(data, index, ax):
            ax.view_init(elev=120, azim=-90)
            plot_xzPlane(
                ax,
                MINS[0] - trajec[index, 0],
                MAXS[0] - trajec[index, 0],
                0,
                MINS[2] - trajec[index, 1],
                MAXS[2] - trajec[index, 1],
            )

            if index > 1:
                plot_trajec(trajec, index, ax)

            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_ylim3d([0, radius])
            ax.set_zlim3d([0, radius])

            plot_3d_pose(data, index, ax)

        ax.clear()
        do_it_all(data, index, ax)

    ani = FuncAnimation(
        fig, update, frames=data.shape[0], interval=100 / fps, repeat=False
    )
    ani.save(save_path, fps=fps)
    if save_path_2:
        ani.save(save_path_2, fps=fps)

    plt.close()

def plot_3d_motion_frames_single(data, title, axes, nframes=5, radius=2):
    data = data.copy().reshape(len(data), -1, 3)  # (seq_len, joints_num, 3)

    # cut tail, if equal
    # print(data.shape)
    for i in range(data.shape[0]-1):
        if (data[i] == data[i+1]).all():
            data = data[:i]
            break
    # print(data.shape)

    # init(ax, fig, title, radius)
    MINS, MAXS = data.min(axis=0).min(axis=0), data.max(axis=0).max(axis=0)

    data[:, :, 1] -= MINS[1]  # height offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]  # centering
    data[..., 2] -= data[:, 0:1, 2]  # centering

    # frames to plot
    frames_to_plot = np.linspace(0, len(data) - 1, nframes, dtype=int)
    axes.flatten()[0].set_ylabel(title)
    for ax, index in zip(axes.flatten(), frames_to_plot):
        # ax.clear()
        ax.view_init(elev=120, azim=-90)
        plot_xzPlane(
            ax,
            MINS[0] - trajec[index, 0],
            MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1],
            MAXS[2] - trajec[index, 1],
        )
        if index > 1:
            plot_trajec(trajec, index, ax)

        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        plot_3d_pose(data, index, ax)

def plot_3d_motion_frames_multiple(
    data_multiple,
    titles,
    nframes=5,
    radius=2,
    figsize=(10, 10),
    return_array=False, velocity=False):
    fig, axes = plt.subplots(
        len(data_multiple), nframes, figsize=figsize, subplot_kw={"projection": "3d"}
    )
    for i, data in enumerate(data_multiple):
        try:
            if velocity:
                data = np.cumsum(data, axis=0)
            plot_3d_motion_frames_single(data, titles[i], axes[i], nframes, radius)
        except:
            print(f"Error plotting {titles[i]}")
    if return_array:
        plt.savefig("tmp.png")
        X = plt.imread("tmp.png")
        plt.close()

        # delete the file

        os.remove("tmp.png")

        return torch.tensor(X).permute(2, 0, 1)

# VAE training
def print_scientific(x):
    return "{:.2e}".format(x)

def plotUMAP(latent, latent_dim, KL_weight,  save_path, labels=None, show=False, max_points=5000):
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
    if labels is not None:  
        plt.scatter(projection[:, 0], projection[:, 1], 
                    c=labels.cpu().numpy(), 
                    alpha=0.5, s=4)
    else:
        plt.scatter(projection[:, 0], projection[:, 1], 
                    #c=labels.cpu().numpy(), cmap='tab10', 
                    alpha=0.5, s=4)
    
    plt.title(f'UMAP projection of latent space (LD={latent_dim}, KL={print_scientific(KL_weight)})')
    
    if save_path is not None:
        plt.savefig(f'{save_path}/projection_LD{latent_dim}_KL{print_scientific(KL_weight)}.png')
    
        return projection, reducer
    elif show:
        plt.show()
    return fig

def prep_save(model, data_loader, log_dir=None):
    latent, texts = list(), list()
    action_groups, actions = list(), list()
    for batch in tqdm(data_loader):
        x_, text, action_group, action = batch  
        z = model.encode(x_).squeeze()
        latent.append(z.detach())
        texts.append(text.detach())
        action_groups.append(action_group)
        actions.append(action)

    latent = torch.cat(latent, dim=0)
    texts = torch.cat(texts, dim=0)
    action_groups = torch.cat(action_groups, dim=0)
    actions = torch.cat(actions, dim=0)

    return latent, texts, action_groups, actions
    
def save_for_diffusion(save_path, model=None, **kwargs):
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

    if model is not None:
        torch.save(model, f'{save_path}/model.pth')
        num_params = sum(p.numel() for p in model.parameters())
        with open(f'{save_path}/num_params.txt', 'w') as file:
            file.write(f'Number of parameters: {num_params}')

    for k, v in kwargs.items():
        print(f'Saving {k}...')
        torch.save(v, f'{save_path}/{k}.pt')


# data utils
def pad_crop(data, length=420):
    # Use numpy to handle padding and truncating efficiently
    if data.shape[0] < length:
        pad_cropped = np.pad(data, ((0, length - data.shape[0]), (0, 0), (0, 0)), mode='edge')
    else:
        pad_cropped = data[:length]
    return pad_cropped


# text mapping
def translate(txt, idx2word):
    return ' '.join([idx2word[i.item()] for i in txt])

def translate_inv(txt, word2idx, max_len=250):
    enc =  [word2idx[i] for i in txt.split()]
    return torch.tensor(enc + [0] * (max_len - len(enc)))

def test_translate(texts=None, txt = 'a person is walking', word2idx=None, idx2word=None):
    print(f"""
    Testing the translation functions:
        (on input text)
            txt          : {txt}
            dec(enc(txt)): {translate(translate_inv(txt, word2idx), idx2word)}""")
    
    if texts is not None:
        # print('               Does the encoder work as expected: ',
        #       (texts[0][0].detach().cpu() == translate_inv(translate(texts[0][0], idx2word), word2idx)).all())
        # print('                  Example:     ',translate(texts[0][0], idx2word))

        print(f"""
        (on texts[0][0])
            txt          : {translate(texts[0][0], idx2word)}
            dec(enc(txt)): {translate(translate_inv(translate(texts[0][0], idx2word), word2idx), idx2word)}""")


# testing the functions
if __name__ == '__main__':
    path = "../stranger_repos/HumanML3D/joints/000000.npy"
    motion = np.load(path)

    ax = plot_3d_pose(motion, 12) 
    plot_xzPlane(ax, -.1, .1, 0, -.1, .1)
    plt.show()