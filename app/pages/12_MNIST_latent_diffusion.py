import streamlit as st
import matplotlib.pyplot as plt
import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator
import os
import numpy as np
import torch
# from utils import latent_picker, load_latent
from mnist_latent_diffusion.utils import load_latent, find_saved_latent
# Intro and title
"""
# MNIST latent diffusion
"""

tab_names = [
    'Select Latent space',
    'Noise schedule set-up',
    'Network',
    'Losses & Metrics',
    'Inference',
]

tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}

with tabs['Select Latent space']:
    from mnist_latent_diffusion.utils import find_saved_latent
    VAE_data = find_saved_latent(path = f"mnist_latent_diffusion/logs/VAE/train/")
    # sort it
    VAE_data = dict(sorted(VAE_data.items(), key=lambda item: int(item[0])))
    with st.sidebar:
        st.write("Select latent space:")
        selected_latent = st.selectbox("", list(VAE_data.keys()))

    outer_cols = st.columns(2)

    for i, entry in enumerate(VAE_data):
        # im_idx = 0 if i % 2 == 0 else 2
        # text_idx = 1 if i % 2 == 0 else 3
        # VAE_data[entry]['image'] = plt.imread(VAE_data[entry]['paths']['projection'])
        # outer_cols[im_idx].image(VAE_data[entry]['paths']['projection'])
        # outer_cols[text_idx].write(f"Latent space: {entry}")
        with outer_cols[i%2]:
            innner_cols = st.columns(2)

            innner_cols[0].image(VAE_data[entry]['paths']['projection'])
            
            with innner_cols[1]:
                st.write(f"Latent space:", entry)
                # VAE_data[entry]

                # what do i actually want to show here?
                # Number of parameters
                # KL loss
                # reconstruction loss
                # dropout rate
                # training time
                # latent dimension

                # load events.out.tfevents.1716932121.Antons-MacBook-Pro-3.local.94642.0 to get training time
                path = VAE_data[entry]['paths']['log']
                ea = event_accumulator.EventAccumulator(path)
                ea.Reload()

                # get all tags
                tags = ea.Tags()
                # tags

                try:
                    mse_test = ea.Scalars('MSE (test, unscaled)')[0].value
                    mse_test_rounded = round(mse_test, 4)
                    st.write(f"MSE (test, unscaled):", mse_test_rounded)
                except:
                    st.write("MSE (test, unscaled) not found")

                train_loss = ea.Scalars('train_loss')

                training_time = train_loss[-1].wall_time - train_loss[0].wall_time
                st.write('Training time:', round(training_time/60, 2), 'minutes')
  

                # load config file
                from mnist_latent_diffusion.utils import load_config
                config = load_config(VAE_data[entry]['paths']['config'])['TRAIN']
                with st.expander('Config'):
                    st.write(config)


with tabs['Noise schedule set-up']:
    pass


with tabs['Inference']:
    if not 'idx' in st.session_state:
        st.session_state.idx = 'None'
    from mnist_latent_diffusion.modules.latentDiffusion import LatentDiffusionModule
    from mnist_latent_diffusion.utils import get_ckpt
    parent_log_dir = 'mnist_latent_diffusion/logs/latentDiffusion/train/'
    # checkpoint = get_ckpt(parent_log_dir, config_name='hparams.yaml', with_streamlit=True)

    def find_latent_diffusion(path):
        LD_data = {}
        folders = [i for i in os.listdir(path) if i.startswith('version_')]
        
        folders = sorted(folders, key=lambda x: int(x.split('_')[-1]))
        
        for version in folders:
            version_num = version.split('_')[-1]
            contents = os.listdir(f"{path}{version}")
            base_path = os.path.join(path, version, )
            if len(contents) < 3:
                
                version_num, contents, base_path
                continue
            import glob
            checkpoints = glob.glob(f"{base_path}/checkpoints/*.ckpt") 
            
            # contents

            if 'version.txt' in contents:
                    # Add an "else" statement here
                    autoencoder_version = open(f"{base_path}/version.txt", 'r').read()
            else:
                autoencoder_version = None

            if not autoencoder_version is None:
                VAE_path = find_saved_latent(f"mnist_latent_diffusion/logs/VAE/train/")
                VAE_ckpt_path = VAE_path[autoencoder_version]['paths']['checkpoints'][0]
            else:
                VAE_ckpt_path = None
            LD_data[version_num] = dict(
                paths = {
                    'log': base_path,
                    'config' : os.path.join(base_path, 'hparams.yaml'),
                    'checkpoints': checkpoints,
                },
                scalar = torch.load(f"{base_path}/scaler.pth") if 'scaler.pth' in contents else None,
                version_num = version_num,
                VAE_version = autoencoder_version,
                VAE_ckpt_path = VAE_ckpt_path,

    
            )
                # 'ckpt_path': checkpoint[0] if len(checkpoint) > 0 else None,
                # 'config_path': os.path.join(base_path, 'hparams.yaml'),
                # 'version_num': version_num,
                # 'base_path': base_path,

                

        return LD_data
    


    saved_latent_data = find_latent_diffusion('mnist_latent_diffusion/logs/latentDiffusion/train/')
    'saved_latent_data', saved_latent_data.keys()
    idx = st.selectbox('Select latent diffusion model', list(saved_latent_data.keys()))
    # idx = "79"
    # saved_latent_data[idx]

    ckpt_LD = saved_latent_data[idx]['paths']['checkpoints'][0]
    # ckpt_LD
    VAE_data  = find_saved_latent(path = f"mnist_latent_diffusion/logs/VAE/train/")
    data_version = VAE_data[saved_latent_data[idx]['VAE_version']]
    # data_version
    # autoencoder = torch.load(saved_latent_data[idx]['VAE_ckpt_path'])
    scalar = saved_latent_data[idx]['scalar']
    
    # autoencoder
    z, y,  autoencoder, projector, projection = load_latent(data_version)
    
    
    model = LatentDiffusionModule(autoencoder=autoencoder, 
                                 scaler=scalar,
                                criteria=None,
                                classifier=None,
                                projector=projector,
                                projection=projection,
                                labels=y,
                                
                                 **config['MODEL'])

    def load_model_and_get_samples(checkpoint):

        plModule = LatentDiffusionModule.load_from_checkpoint(checkpoint)
        plModule.eval()

        import yaml
        with open(checkpoint['config_path'], 'r') as file:
            hparams = yaml.safe_load(file)

        clipped_reverse_diffusion = hparams.get('CLIPPED_REVERSE_DIFFUSION', False)

        count = 0
        x_t_All, hist_all, y_all = plModule.model.sampling(20, clipped_reverse_diffusion=clipped_reverse_diffusion, y=True, device='mps', tqdm_disable=False)

        return x_t_All, hist_all, y_all
    


    if 'x_t_All' not in st.session_state or st.session_state.idx != idx:
        st.session_state.idx = idx
        st.session_state.x_t_All, st.session_state.hist_all, st.session_state.y_all = load_model_and_get_samples(ckpt_LD)

    cols = st.columns([1, 1])

    with cols[0]:

        y = st.select_slider('Select label', options=list(range(10)))  # ask user for input
        matches = torch.where(st.session_state.y_all == y)[0]  #select index, random where it fits
        if len(matches) == 0:
            st.write('No matches found for y')
    with cols[1]:
        if len(matches) > 0:
            idx = matches[torch.randint(0, len(matches), (1,))].item()
            # x_t, hist, y = plModule.model.sampling(20, clipped_reverse_diffusion=False, y=True, device='mps', tqdm_disable=False)
            # x_t = x_t_All[idx]
            # hist = hist_all[idx]
            # y = y_all[idx]

            x_t = st.session_state.x_t_All[idx]
            hist = st.session_state.hist_all[idx]
            y = st.session_state.y_all[idx]

            fig, ax = plt.subplots(1, 1, figsize=(5, 6))
            ax.imshow(x_t.squeeze().detach().cpu().numpy(), cmap='gray')
            ax.set_title(f'Sample from model, with label y={y.item()}')
            ax.set_xticks([])
            ax.set_yticks([])
            
            st.pyplot(fig)
            plt.close(fig)