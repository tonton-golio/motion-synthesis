import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from mnist_latent_diffusion.modules.dataModules import MNISTDataModule
import torch
import torch.nn as nn
import matplotlib.image as mpimg
import os
import torchviz
from home import embed_pdf

# from app.subpages.noise_schedule import mnist_noise_schedule_setup

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
num_images = st.slider('Number of samples', 1, 1000, 20, 99)
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


def get_images(num_images = 10):
    dm = MNISTDataModule(verbose=False, **params_data)
    dm.setup()
    
    ims = [dm.data_train[i][0].squeeze() for i in range(num_images)]
    labs = [dm.data_train[i][1] for i in range(num_images)] 
    return ims, labs



if 'ims' not in st.session_state or 'labs' not in st.session_state:
    ims, labs = get_images(num_images)
    st.session_state.ims = ims
    st.session_state.labs = labs

ims = st.session_state.ims
labs = st.session_state.labs

# Noise schedule
# with tabs['Noise schedule set-up']:
#     mnist_noise_schedule_setup(ims)

# Network
with tabs['Network']:
    from mnist_latent_diffusion.modules.imageDiffusion import Unet
    try:
        def make_graph(model):
            # check if graph.png exists
            fname = 'assets_produced/9_MNIST_diffusion/graph'
            if not fname in os.listdir():
                model.eval()
                x = torch.randn(1, 1, 28, 28)
                y = model(x, torch.tensor([10]), torch.tensor([9]))
                g = torchviz.make_dot(y.mean(), params=dict(model.named_parameters()))
                # Render the Graphviz output and save as a file
                g.render(filename=fname, format='pdf', cleanup=True, )

            embed_pdf(fname + '.pdf')
            
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
    except Exception as e:
        pass


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
        plot_scalar(ea, 'val_fid')

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

    def load_model_and_get_samples(checkpoint, num_samples=20):

        plModule = ImageDiffusionModule.load_from_checkpoint(checkpoint['ckpt_path'])
        plModule.eval()

        import yaml
        with open(checkpoint['config_path'], 'r') as file:
            hparams = yaml.safe_load(file)

        clipped_reverse_diffusion = hparams.get('CLIPPED_REVERSE_DIFFUSION', False)
        x_t_All, hist_all, y_all = plModule.model.sampling(num_samples, clipped_reverse_diffusion=clipped_reverse_diffusion, y=True, device='mps', tqdm_disable=False)

        return x_t_All, hist_all, y_all
    

    
    checkpoint

    
    if True:#st.button('Load model and get samples'):    
        num_samples = num_images
        if 'x_t_All' not in st.session_state or st.session_state.checkpoint_num != checkpoint['version_num']:
            st.session_state.checkpoint_num = checkpoint['version_num']
            st.session_state.x_t_All, st.session_state.hist_all, st.session_state.y_all = load_model_and_get_samples(checkpoint, num_samples)
        if st.button('Reload samples'):
            st.session_state.checkpoint_num = checkpoint['version_num']
            st.session_state.x_t_All, st.session_state.hist_all, st.session_state.y_all = load_model_and_get_samples(checkpoint, num_samples)
            ims, labs = get_images(num_images)
            st.session_state.ims = ims
            st.session_state.labs = labs

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
                # hist = st.session_state.hist_all[idx]
                y = st.session_state.y_all[idx]

                fig, ax = plt.subplots(1, 1, figsize=(5, 6))
                ax.imshow(x_t.squeeze().detach().cpu().numpy(), cmap='gray')
                ax.set_title(f'Sample from model, with label y={y.item()}')
                ax.set_xticks([])
                ax.set_yticks([])
                
                st.pyplot(fig)
                plt.close(fig)

                fig, ax = plt.subplots(1, 1, figsize=(5, 6))
                st.write(st.session_state.x_t_All.shape)

                from subpages.metrics_diffusion import _calculate_FID_SCORE, get_div


                st.write('---')
                st.write('FID')
                fid = _calculate_FID_SCORE(torch.tensor(np.array(ims)).to('cpu'), st.session_state.x_t_All.clone().detach().cpu())
                st.write(f'FID: {fid}')

                st.write('---')
                st.write('Diversity')
                div = get_div(st.session_state.x_t_All.clone().detach().cpu(), st.session_state.y_all.clone().detach().cpu())
                st.write(div)

                st.write('---')
                st.write('Multimodality')
                class_diversity = get_div(st.session_state.x_t_All.clone().detach().cpu(), st.session_state.y_all.clone().detach().cpu(), method='class')
                st.write(class_diversity)
                Multimodality = np.mean(list(class_diversity.values()))
                st.write(f'Multimodality: {Multimodality}')

                

                st.write('---')
                # now measure it all for the original images
                st.write('Original images')
                st.write(len(ims))
                st.write('---')
                st.write('FID')
                fid = _calculate_FID_SCORE(torch.tensor(np.array(ims)).to('cpu'), torch.tensor(np.array(ims)).to('cpu'))
                st.write(f'FID: {fid}')

                st.write('---')
                st.write('Diversity')
                div = get_div(torch.tensor(np.array(ims)).to('cpu'), torch.tensor(labs).to('cpu'))
                st.write(div)

                st.write('---')
                st.write('Multimodality')
                class_diversity_original = get_div(torch.tensor(np.array(ims)).to('cpu'), torch.tensor(labs).to('cpu'), method='class')
                st.write(class_diversity)
                Multimodality = np.mean(list(class_diversity_original.values()))
                st.write(f'Multimodality: {Multimodality}')

                # plot class_diversity for both original and generated
                fig, ax = plt.subplots(1, 1, figsize=(6, 3))
                ax.bar(class_diversity.keys(), class_diversity.values(), color='purple', alpha=0.4, label='Generated')
                
                ax.bar(class_diversity_original.keys(), class_diversity_original.values(), color='orangered', alpha=0.4, label='Original', zorder=0)
                ax.set_xlabel('Class label')
                ax.set_ylabel('Diversity')
                ax.set_title('Per-class diversity')
                ax.set_ylim(0, max(max(class_diversity.values()), max(class_diversity_original.values()))+1.5)
                ax.legend(ncol=2)
                st.pyplot(fig)
                plt.close(fig)