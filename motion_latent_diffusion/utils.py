import numpy as np
import torch
import os

# animation
from os.path import join as pjoin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import yaml
import torch.nn as nn
import glob

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
    # 'bent_identity': nn.BentIdentity(),
    # 'gaussian': nn.Gaussian(),
    "softmax": nn.Softmax(),
    "softmin": nn.Softmin(),
    "softshrink": nn.Softshrink(),
    # 'sinc': nn.Sinc(),
}

def get_ckpts(log_dir):
    print('log_dir:', log_dir)
    folders = glob.glob(f"{log_dir}/*") # like: version_0, ...
    print(folders)
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
            # print(f"Failed to get checkpoint for {folder}")

    print(ckpts)
    print(count)
    if count['success'] == 0:
        ckpts['latest'] = {'path': None, 'cfg_path': None, 'epoch': 0, 'step': 0}
    else:
        ckpts['latest'] = ckpts[max(ckpts.keys(), key=int)]
    

    return ckpts

def load_config(name):
    with open(f'configs/config_{name}.yaml', 'r') as file:
        return yaml.safe_load(file)
    
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

def plot_3d_pose(data, index, ax=None):
    """Plot a 3D pose."""

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


def init(ax, fig, title, radius=2):
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
    save_path_2=None,
):
    #     matplotlib.use('Agg')
    data = data.copy().reshape(len(data), -1, 3)  # (seq_len, joints_num, 3)

    # cut tail, if equal
    print(data.shape)
    for i in range(data.shape[0]-1):
        if (data[i] == data[i+1]).all():
            data = data[:i]
            break
    print(data.shape)

    if velocity:
        data = np.cumsum(data, axis=0)

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111, projection="3d")
    init(ax, fig, title, radius)
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
    print(data.shape)
    for i in range(data.shape[0]-1):
        if (data[i] == data[i+1]).all():
            data = data[:i]
            break
    print(data.shape)

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
    return_array=False,
    velocity=False,
):
    fig, axes = plt.subplots(
        len(data_multiple), nframes, figsize=figsize, subplot_kw={"projection": "3d"}
    )
    for i, data in enumerate(data_multiple):
        if velocity:
            data = np.cumsum(data, axis=0)
        plot_3d_motion_frames_single(data, titles[i], axes[i], nframes, radius)
    if return_array:
        plt.savefig("tmp.png")
        X = plt.imread("tmp.png")
        plt.close()

        # delete the file

        os.remove("tmp.png")

        return torch.tensor(X).permute(2, 0, 1)


if __name__ == "__main__":
    path = "../data/HumanML3D/joints/000000.npy"
    motion = np.load(path)
    print("motion shape: ", motion.shape)
    # plot_3d_motion_animation(motion, 'test', radius=1.4, save_path='test.mp4')
    motion1, motion2 = motion[: len(motion) // 2], motion[len(motion) // 2 :]
    motion_stacked = [motion1, motion2]
    X = plot_3d_motion_frames_multiple(
        motion_stacked, nframes=5, radius=1.4, figsize=(20, 10), return_array=True
    )
    print(X.shape)
