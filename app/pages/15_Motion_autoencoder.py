import streamlit as st
import numpy as np
import os, sys, glob
import matplotlib.pyplot as plt
# path join 
from os.path import join as pjoin
from pathlib import Path

# Intro and title
"""
# Motion Autoencoder


"""
tab_names = [
    'VAE1',
]
tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}


with tabs['VAE1']:
    st.write("""
    Our first model, VAE1, is a transformer based VAE which uses linear layers for compression and decompression.
             """)
    



def find_saved_latent(path = f"logs/VAE/train/", cfg_name='hparams'):    
    """
    Find saved latent vectors from VAE training
    """

    VAE_data = {}
    # st.write(os.listdir(path))
    for version in os.listdir(path):
        if not os.path.isdir(f"{path}{version}"):
            continue
        version_num = version.split('_')[-1]
        contents = os.listdir(f"{path}{version}")
        base_path = os.path.join(path, version, )
        # st.write(contents)
        if 'saved_latent' in contents:
            # st.write(f"Found saved latent vectors for version {version_num}")
            cfg_file = None  # get config file
            for file in contents:
                if cfg_name in file and file.endswith('.yaml'):
                    cfg_file = file
                    break
            
            projection = None  # get projection image
            for file in contents:
                if 'projection' in file and file.endswith('.png'):
                    projection = file
                    break
            if projection is None:
                continue

            checkpoints = glob.glob(f"{base_path}/checkpoints/*")  # check for checkpoints
            saved_latent = os.listdir(os.path.join(base_path, 'saved_latent'))  # open saved_latent and check whats inside

            VAE_data[version_num] = {
                'saved_latent' : saved_latent,
                'paths' : {
                    'config' : os.path.join(base_path, cfg_file),
                    'saved_latent' : os.path.join(base_path, 'saved_latent'),
                    'projection' : os.path.join(base_path, projection) if projection else None,
                    'checkpoints' : checkpoints,
                    'log' : base_path,
                },
                'contents' : contents
            }

    return VAE_data

VAE_data = find_saved_latent('../motion_latent_diffusion/logs/MotionVAE/VAE1/train/')
idx = st.selectbox('Select VAE model', list(VAE_data.keys()))
VAE_data[str(idx)]


# now we want to show the loss,
# the projection of the latent space


# show the projection
for entry in VAE_data:
    VAE_data[entry]['image'] = plt.imread(VAE_data[entry]['paths']['projection'])
    st.image(VAE_data[entry]['image'])


# load the loss
from tensorboard.backend.event_processing import event_accumulator
import os

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


# # "VELOCITY_L2_trn"
# 1:"MOTION_L2_trn"
# 2:"MOTIONRELATIVE_L2_trn"
# 3:"DIVERGENCE_KL_trn"
scalars = {
    k : load_log_scalar(VAE_data[str(idx)]['paths']['log'], tag=k) for k in ['MOTION_L2_trn', 'MOTIONRELATIVE_L2_trn', 'DIVERGENCE_KL_trn']
}

scalars
# config
from motion_latent_diffusion.utils import load_config
cfg = load_config(VAE_data[str(idx)]['paths']['config'])
cfg


# get num param from num_params.txt
with open(f"{VAE_data[str(idx)]['paths']['saved_latent']}/num_params.txt", 'r') as file:
    num_params = file.read()

num_params


# view recon_latest.mp4

st.video(f"{VAE_data[str(idx)]['paths']['log']}/recon_latest.mp4")
