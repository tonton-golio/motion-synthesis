import streamlit as st
import numpy as np
import os, sys, glob
import matplotlib.pyplot as plt
# path join 
from os.path import join as pjoin
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import os

# Intro and title
"""
# Motion Latent Diffusion


"""
tab_names = [
    'LD_VAE1',
]
tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}



def find_logs(path = f"../motion_latent_diffusion/logs/MotionLD/VAE1/", cfg_name='hparams'):    
    """
    Find saved latent vectors from VAE training
    """

    VAE_data = {}
    bad_versions = []
    # st.write(os.listdir(path))
    for version in os.listdir(path):
        if not os.path.isdir(f"{path}{version}"):
            continue
        version_num = version.split('_')[-1]
        contents = os.listdir(f"{path}{version}")
        base_path = os.path.join(path, version, )
        # st.write(contents)
        
        # st.write(f"Found saved latent vectors for version {version_num}")
        cfg_file = None  # get config file
        for file in contents:
            if cfg_name in file and file.endswith('.yaml'):
                cfg_file = file
                break
        
        video_paths = []  # get projection image
        for file in contents:
            # if 'projection' in file and file.endswith('.png'):
            #     projection = file
            #     break
            if file.endswith('.mp4'):
                video_paths.append(file)
        if not video_paths:
            # st.write(f"No videos found for version {version_num}"   )
            bad_versions.append(version_num)
            continue

        checkpoints = glob.glob(f"{base_path}/checkpoints/*")  # check for checkpoints
        
        VAE_data[version_num] = {
            # 'paths' : {
                'config' : os.path.join(base_path, cfg_file),
                'video_paths_dirty' : [os.path.join(base_path, video_path) for video_path in video_paths if 'dirty' in video_path],
                'video_paths_clean' : [os.path.join(base_path, video_path) for video_path in video_paths if 'clean' in video_path],
                # 'video_paths_s
                'checkpoints' : checkpoints,
                'log' : base_path,
            # },
            # 'contents' : contents
        }
    if bad_versions:
        # sort
        bad_versions = sorted(bad_versions, key=lambda x: int(x.split('-')[-1]))
        st.write(f"Bad versions: {bad_versions}")

    return VAE_data


with tabs['LD_VAE1']:
    ld_data = find_logs()
    st.write(ld_data)

    idx = st.selectbox("Select version", list(ld_data.keys()))

    st.write(f"Selected version: {idx}")

    cols = st.columns((1, 1, 1))
    with cols[0]:
        # st.write(f"Config file: {ld_data[idx]['config']}")
        # show a video
        st.video(ld_data[idx]['video_paths_dirty'][0])
        
        
        
        # show losses

    def load_log_scalar(logdir, tag='train/loss'):
        """
        Load the loss from tensorboard log
        """
        ea = event_accumulator.EventAccumulator(logdir)
        ea.Reload()
        tags = ea.Tags()['scalars']
        if tag not in tags:
            'try one of these:', tags
            raise ValueError(f"Tag {tag} not found in {logdir}")
        
        return ea.Scalars(tag)
    with cols[1]:
        scalars = {
            k : load_log_scalar(ld_data[str(idx)]['log'], tag=k) for k in ['train_loss_step', 'val_loss', 'train_loss_epoch']
        }

         # plot all scalars
        fig, axes = plt.subplots(1, len(scalars), figsize=(5*len(scalars), 5))
        for ax, (k, v) in zip(axes, scalars.items()):
            ax.plot([x.step for x in v], [x.value for x in v])
            ax.set_title(k)

        st.pyplot(fig)
    


st.divider()

# open latent diffusion model in inference mode

