import numpy as np

from os.path import join as pjoin
import torch.nn.functional as F
import torch

# logging
import os
import glob
import shutil
import pickle
import yaml

# animation
import matplotlib
from os.path import join as pjoin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3


import seaborn as sns
iris = sns.load_dataset("iris")
import os
import glob
import subprocess

def get_file_names(path):
    return glob.glob(path + "/*.png")

def file_name_sort(filenames):
    nums = []
    for name in filenames:
        digits = ''
        for char in name:
            if char.isdigit():
                digits += char

        nums.append(int(digits))

    return [x for _,x in sorted(zip(nums,filenames))]

# Write file names to a text file for ffmpeg
def write_to_file(file_names, output_file, time_per_frame=0.05, time_at_end=1.0):
    with open(output_file, 'w') as file:
        for name in file_names:
            file.write(f"file '{name}'\n")
            file.write(f"duration {time_per_frame}\n")

        # time at end
        file.write(f"file '{file_names[-1]}'\n")
        file.write(f"duration {time_at_end}\n")
            
# Create a gif using ffmpeg
def create_gif(input_path, output_path, size=(32, 64), time_per_frame=0.02):
    file_names = get_file_names(input_path)
    file_names = file_name_sort(file_names)[:-1]
    temp_file = "file_list.txt"
    # for f in file_names:
    #     print(f)    
    write_to_file(file_names, temp_file, time_per_frame)
    command = f'ffmpeg -f concat -safe 0 -i {temp_file} -vf "fps=10,scale={size[0]}:{size[1]}:flags=lanczos" -c:v gif {output_path} -y'
    subprocess.run(command, shell=True)
    os.remove(temp_file)  # Clean up the temporary file


import numpy as np
import torch
import os

# animation
from os.path import join as pjoin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_3d_pose(data, index, ax = None):
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

    colors = ["red", "blue", "black", "red", "blue", ]

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

def plot_3d_motion_animation(data, title, figsize=(10, 10), fps=20, radius=2, save_path='test.mp4'):
    
    #     matplotlib.use('Agg')
    data = data.copy().reshape(len(data), -1, 3)  # (seq_len, joints_num, 3)
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
            plot_xzPlane(ax,
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

        
    ani = FuncAnimation(fig, update, frames=data.shape[0], interval=100 / fps, repeat=False)
    ani.save(save_path, fps=fps)

def plot_3d_motion_frames_single(data, title,  axes,  nframes=5, radius=2):
    data = data.copy().reshape(len(data), -1, 3)  # (seq_len, joints_num, 3)
    
    # init(ax, fig, title, radius)
    # MINS, MAXS = data.min(axis=0).min(axis=0), data.max(axis=0).max(axis=0)

    # data[:, :, 1] -= MINS[1]  # height offset
    # trajec = data[:, 0, [0, 2]]

    # data[..., 0] -= data[:, 0:1, 0]  # centering
    # data[..., 2] -= data[:, 0:1, 2]  # centering

    # frames to plot
    frames_to_plot = np.linspace(0, len(data)-1, nframes, dtype=int)
    axes.flatten()[0].set_ylabel(title)
    for (ax, index) in zip(axes.flatten(), frames_to_plot):
        # ax.clear()
        ax.view_init(elev=120, azim=-90)
        # plot_xzPlane(ax,
        #     MINS[0] - trajec[index, 0],
        #     MAXS[0] - trajec[index, 0],
        #     0,
        #     MINS[2] - trajec[index, 1],
        #     MAXS[2] - trajec[index, 1],
        # )
        # if index > 1:
        #         plot_trajec(trajec, index, ax)

        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        plot_3d_pose(data, index, ax)
        
def plot_3d_motion_frames_multiple(data_multiple, titles, nframes=5, radius=2, figsize=(10, 10), return_array=False):
    fig, axes = plt.subplots(len(data_multiple), nframes, figsize=figsize, subplot_kw={'projection': '3d'})
    for i, data in enumerate(data_multiple):
        plot_3d_motion_frames_single(data, titles[i], axes[i], nframes, radius)
    if return_array:
        
        plt.savefig('tmp.png')
        X = plt.imread('tmp.png')
        plt.close()

        # delete the file
        
        #os.remove('tmp.png')

        return torch.tensor(X).permute(2, 0, 1).requires_grad_(False)

if __name__ == '__main__':
    path = '../data/HumanML3D/joints/000000.npy'
    motion = np.load(path)
    print('motion shape: ', motion.shape)
    # plot_3d_motion_animation(motion, 'test', radius=1.4, save_path='test.mp4')
    motion1, motion2 = motion[:len(motion)//2], motion[len(motion)//2:]
    motion_stacked = [motion1, motion2]
    X = plot_3d_motion_frames_multiple(motion_stacked, nframes=5, radius=1.4, figsize=(20,10), return_array=True)
    print(X.shape)