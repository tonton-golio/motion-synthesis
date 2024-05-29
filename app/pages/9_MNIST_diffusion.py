import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mnist_latent_diffusion.modules.dataModules import MNISTDataModule
import torch
import torch.nn as nn
import matplotlib.image as mpimg
import os
import torchviz
from home import embed_pdf

from subpages.mnist.noise_schedule import mnist_noise_schedule_setup

# Intro and title
"""
# MNIST Diffusion (pixel space)
"""

tab_names = [
    'Noise schedule set-up',
    'Network',
    'Losses & Metrics',
    'Inference',
]

tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}


# setting up the data module and the variance schedule
with st.sidebar:  # parameters for the data module
        
        params_preset = True
        if params_preset:
            params_data = dict(
                BATCH_SIZE = 16,
                ROTATION = 30,
                SCALE = 0.4,
                TRANSLATE_X = 0.1,
                TRANSLATE_Y = 0.1,
                SHEAR = 0.1,

                NORMALIZE_MEAN = 0., 
                NORMALIZE_STD = 1.,

                BOOL = False,
                NO_NORMALIZE = False
            )
        else:
            'Data module parameters:'
            params_data = dict(
                BATCH_SIZE = 16,
                ROTATION = st.slider('rotation', 0, 90, 30),
                SCALE = st.slider('scale', 0., 1., 0.4),
                TRANSLATE_X = st.slider('translate x', 0., 1., 0.1),
                TRANSLATE_Y = st.slider('translate y', 0., 1., 0.1),
                SHEAR = st.slider('shear', 0., 1., 0.1),

                NORMALIZE_MEAN = st.slider('normalize mean', 0., 1., 0.,), 
                NORMALIZE_STD = st.slider('normalize std', 0.01, 1., 1.),

                BOOL = st.checkbox('bool'),
                NO_NORMALIZE = st.checkbox('no normalize')
            )
            '---'


def get_images():
    dm = MNISTDataModule(verbose=False, **params_data)
    dm.setup()
    num_images = 10
    ims = [dm.data_train[i][0].squeeze() for i in range(num_images)]
    return ims

if 'ims' not in st.session_state:
    st.session_state.ims = get_images()

ims = st.session_state.ims

# Noise schedule
with tabs['Noise schedule set-up']:
    mnist_noise_schedule_setup(ims)

# Network
with tabs['Network']:
    from mnist_latent_diffusion.modules.imageDiffusion import Unet
    def make_graph(model):
        # check if graph.png exists
        if not 'graph.pdf' in os.listdir():
            model.eval()
            x = torch.randn(1, 1, 28, 28)
            y = model(x, torch.tensor([10]), torch.tensor([9]))
            g = torchviz.make_dot(y.mean(), params=dict(model.named_parameters()))

            # Render the Graphviz output and save as a file
            g.render(filename='graph', format='pdf', cleanup=True, )

        embed_pdf('graph.pdf')
        
    model = Unet(timesteps=100, time_embedding_dim=10, dim_mults=[2,4])
    num_params = sum(p.numel() for p in model.parameters())

    cols = st.columns([1, 3])
    with cols[0]:
        """
        We employ a U-Net architecture, with sequential convolution \& down-sampling blocks, with skipped connections.
        """
        'Number of parameters:', num_params

    with cols[1]:
        # show graph of Unet
        make_graph(model)


# Losses
with tabs['Losses & Metrics']:
    import tensorboard as tb
    from tensorboard.backend.event_processing import event_accumulator

    def plot_scalar(ea, tag = 'val_loss'):
        # get val_loss
        scalar = np.array(ea.Scalars(tag))
        scalar_dict = {x.step: x.value for x in scalar}
        
        # plot val_loss
        fig, ax = plt.subplots()
        ax.plot(scalar_dict.keys(), scalar_dict.values())
        ax.set_title(tag)
        
        st.pyplot(fig)
        plt.close(fig)

    path = '../mnist_latent_diffusion/logs/imageDiffusion/train'
    folders = sorted(os.listdir(path), key=lambda x: int(x.split('_')[1]))[::-1]

    cols = st.columns([1, 1, 1])
    with cols[0]:
        folder = st.selectbox('Select folder', folders)
        files = os.listdir(os.path.join(path, folder))
        files

        # to open a file like: "events.out.tfevents.1716030374.Antons-MacBook-Pro-3.local.40050.0"
        ea = event_accumulator.EventAccumulator(os.path.join(path, folder))
        ea.Reload()

        # get all tags
        tags = ea.Tags()
        tags
        

    with cols[1]:
        
        plot_scalar(ea, 'val_loss')
        plot_scalar(ea, 'train_loss')

    with cols[2]:
        plot_scalar(ea, 'fid')

        im = ea.Images('hist')
        # show image
        if im:
            im = im[0]
            from io import BytesIO
            from PIL import Image
            import matplotlib.pyplot as plt

            image = Image.open(BytesIO(im.encoded_image_string))
            fig, ax = plt.subplots()
            plt.imshow(image)
            plt.axis('off')
            st.pyplot(fig)


        
# Inference
with tabs['Inference']:
    if not 'checkpoint_num' in st.session_state:
        st.session_state.checkpoint_num = 'None'
    from mnist_latent_diffusion.modules.imageDiffusion import ImageDiffusionModule
    from mnist_latent_diffusion.utils import get_ckpt
    parent_log_dir = '../mnist_latent_diffusion/logs/imageDiffusion/train/'
    checkpoint = get_ckpt(parent_log_dir, config_name='hparams.yaml', with_streamlit=True)

    def load_model_and_get_samples(checkpoint):

        plModule = ImageDiffusionModule.load_from_checkpoint(checkpoint['ckpt_path'])
        plModule.eval()

        import yaml
        with open(checkpoint['config_path'], 'r') as file:
            hparams = yaml.safe_load(file)

        clipped_reverse_diffusion = hparams.get('CLIPPED_REVERSE_DIFFUSION', False)

        count = 0
        x_t_All, hist_all, y_all = plModule.model.sampling(20, clipped_reverse_diffusion=clipped_reverse_diffusion, y=True, device='mps', tqdm_disable=False)

        return x_t_All, hist_all, y_all
    checkpoint

    if 'x_t_All' not in st.session_state or st.session_state.checkpoint_num != checkpoint['version_num']:
        st.session_state.checkpoint_num = checkpoint['version_num']
        st.session_state.x_t_All, st.session_state.hist_all, st.session_state.y_all = load_model_and_get_samples(checkpoint)

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