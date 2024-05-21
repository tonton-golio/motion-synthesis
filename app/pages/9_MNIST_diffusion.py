import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mnist_latent_diffusion.modules.dataModules import MNISTDataModule
import torch
import torch.nn as nn
import math
from matplotlib import gridspec

# make st.session_state available




# Intro and title
"""
# MNIST Diffusion (pixel space)
"""

tab_names = [
    'Noise schedule',
]

tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}

# setting up the data module and the variance schedule
    
with st.sidebar:  # parameters for the data module
        'Data module parameters:'
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

      

class VarianceSchedule(nn.Module):

    def __init__(self, timesteps, method="cosine", **kwargs):
        super(VarianceSchedule, self).__init__()
        self.timesteps = timesteps

        if method == "cosine":
            # st.write('using cosine, with epsilon:', kwargs.get("epsilon", 0.008))
            betas = self._cosine_variance_schedule(timesteps, epsilon=kwargs.get("epsilon", 0.008))
        elif method == "linear":
            betas = self._linear_variance_schedule(timesteps, 
                                                beta_start=kwargs.get("beta_start", 1e-4),
                                                beta_end=kwargs.get("beta_end", 0.02))
        elif method == "square":
            betas = self._sqr_variance_schedule(timesteps, 
                                                beta_start=kwargs.get("beta_start", 1e-4),
                                                beta_end=kwargs.get("beta_end", 0.02))
        else:
            raise NotImplementedError

    
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = (
            torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)
            ** 2
        )
        betas = torch.clip(1 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        betas = betas / torch.max(betas)
        
        return betas
    
    def _linear_variance_schedule(self, timesteps, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, steps=timesteps + 1, dtype=torch.float32) + .02
        betas = torch.clip(betas[1:] , 0.0, 0.999)
        return betas
    
    def _sqr_variance_schedule(self, timesteps, beta_start=1e-4, beta_end=0.02):
        steps = torch.linspace(beta_start**0.5, beta_end**0.5, steps=timesteps + 1, dtype=torch.float32)
        f_t = steps**2
        betas = torch.clip(f_t[1:] , 0.0, 0.999)
        return betas

    
    def forward(self, x, noise, t, clip=True):

        x_t = self.sqrt_alphas_cumprod[t] * x + self.sqrt_one_minus_alphas_cumprod[t] * noise
        
        if clip:
            x_t = torch.clip(x_t, 0.0, 1.0)
        return x_t
    

def get_images():
    dm = MNISTDataModule(verbose=False, **params_data)
    dm.setup()
    num_images = 10
    ims = [dm.data_train[i][0].squeeze() for i in range(num_images)]
    return ims

if 'ims' not in st.session_state:
    st.session_state.ims = get_images()

ims = st.session_state.ims

with st.sidebar:
    'Noise schedule parameters:'
    params_schedule = dict(
        method = st.selectbox('method', ['cosine', 'linear', 'square']),
        timesteps = st.slider('timesteps', 100, 1000, 100),
    )
    if params_schedule['method'] == 'cosine':
        params_schedule['epsilon'] = st.select_slider('epsilon', options=np.logspace(-14, -1, 9))
    else:
        params_schedule['beta_start'] = st.select_slider('beta_start', 
                                                        #[0.00001, 0.0001, 0.0001]
                                                        np.logspace(-5, -1, 5)
                                                        )
        params_schedule['beta_end'] = st.select_slider('beta_end', options=np.logspace(-4, -2, 11))

VS = VarianceSchedule(**params_schedule)

# Noise schedule
with tabs['Noise schedule']:

    

    def make_grid(ds, nrow=3, ncols=8):
        ims = np.array([ds[i][0].squeeze() for i in range(nrow * ncols)])
        fig, axes = plt.subplots(nrow, ncols, figsize=(2*ncols, 2*nrow))
        for ax, im in zip(axes.flatten(), ims):
            ax.imshow(im, cmap='gray_r')
            ax.axis('off')
        st.pyplot(fig)

    def plot_distribution(ds):
        ims = np.array([ds[i][0].squeeze() for i in range(30)])
        vals = ims.flatten()
        fig, ax = plt.subplots(1, 1, figsize=(16, 4))
        ax.hist(vals, bins=100)
        st.pyplot(fig)

    def plot_noising_schedule(ims, VS):

        fig, ax = plt.subplots(len(ims), 10, figsize=(10, 6))
        if len(ims) == 1: ax = ax[None,:]
        ts = np.linspace(0, params_schedule['timesteps']-1, 10, dtype=int)
        for i, t in enumerate(ts):
            if i == 0:
                ax[0,i].set_title(f"t={t} / {params_schedule['timesteps']-1}")
            else:
                ax[0,i].set_title(f"{t}")
            for j, im in enumerate(ims):
                noise = torch.randn_like(im)

                im_noisy = VS(im, noise, t)
                ax[j,i].imshow(im_noisy, cmap='gray_r')
                ax[j,i].axis('off')
        st.pyplot(fig)

    def shannon_entropy_2d(im, plot=False, title1='image'):
        im_arr = np.array(im)
        # im_arr.shape
        im_derivative = np.gradient(im_arr)
        # im_derivative[0].shape
        im_derivative = im_derivative[0]**2 + im_derivative[1]**2
        im_derivative /= im_derivative.sum()
        
        H = -np.sum(im_derivative[im_derivative > 0] * np.log2(im_derivative[im_derivative > 0]))

        if plot:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(f"Shannon entropy: {H:.2f}")
            im1 = ax[0].imshow(im_arr, cmap='gray_r')
            ax[0].set_title(title1)
            # plt.colorbar(im1, ax=ax[0])
            ax[0].axis('off')
            ax[1].set_title('Spatial derivative (second order, 2d)')
            im2 = ax[1].imshow(im_derivative, cmap='gray_r')
            # plt.colorbar(im2, ax=ax[1])
            ax[1].axis('off')

            dist_diff = im_derivative.flatten()
            ax[2].hist(dist_diff, bins=100)
            ax[2].set_title('derivative distribution')
            ax[2].set_yscale('log')
            st.pyplot(fig)

        return H

    def plot_noising_schedule2(ims, VS):
        nt = 8
        fig, ax = plt.subplots(5, nt, figsize=(22, 9))
        ts = np.linspace(0, params_schedule['timesteps']-1, nt, dtype=int)
        for i, t in enumerate(ts):
            for j, im in enumerate(ims):
                noise = torch.randn_like(im)

                im_noisy = VS(im, noise, t)
                S = shannon_entropy_2d(im_noisy)
                if j == 0:
                    ax[j,i].set_title(f"t={t} / {params_schedule['timesteps']-1}, H={S:.2f}", color='red')
                else:
                    ax[j,i].set_title(f"H={S:.2f}", color='red')

                ax[j,i].imshow(im_noisy, cmap='gray')
                ax[j,i].axis('off')

        fig.set_facecolor('black')
        st.pyplot(fig)


    

    # get some samples from the data module
    

    # plot_noising_schedule(ims[:1], VS)

    # shannon_entropy_2d(ims[0], plot=True)
    plot_noising_schedule2(ims[:5], VS)
    

    def shannon_entropy_across_time_plot(ims, num_timesteps = 20, num_noise_samples=10):
        metadata = dict(
            num_images = len(ims),
            num_timesteps = num_timesteps,
            num_noise_samples = num_noise_samples,
        )

        ts = np.linspace(0, params_schedule['timesteps']-1, num_timesteps, dtype=int)
        
        res = {}

        for t in ts:
            res[t] = []
            for im in ims:
                noise = torch.randn_like(ims[0])
                im_noisy = VS(im, noise, t)
                res[t].append(shannon_entropy_2d(im_noisy))

            # convert to mean and std
            res[t] = np.array(res[t])
            res[t] = res[t].mean(), res[t].std()

        # lets calculate it for pure noise
        res_pure_noise = []
        for i in range(num_noise_samples):
            pure_noise = torch.randn_like(ims[0])
            res_pure_noise.append(shannon_entropy_2d(pure_noise))

        res_pure_noise = np.array(res_pure_noise)
        res_pure_noise = res_pure_noise.mean(), res_pure_noise.std()

        # lets calculate it for clean images
        res_clean = []
        for im in ims:
            res_clean.append(shannon_entropy_2d(im))

        res_clean = np.array(res_clean)
        res_clean = res_clean.mean(), res_clean.std()

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.errorbar(list(res.keys()), [r[0] for r in res.values()], yerr=[r[1] for r in res.values()], label='noisy images', lw=2)

        ax.axhline(res_pure_noise[0], color='red', linestyle='--', label='pure noise (mean)')
        ax.fill_between([0, params_schedule['timesteps']], res_pure_noise[0]-res_pure_noise[1], res_pure_noise[0]+res_pure_noise[1], color='red', alpha=0.5, label='pure noise (std)')

        ax.axhline(res_clean[0], color='green', linestyle='--', label='clean images (mean)')
        ax.fill_between([0, params_schedule['timesteps']], res_clean[0]-res_clean[1], res_clean[0]+res_clean[1], color='green', alpha=0.5, label='clean images (std)')

        ax.set_xlabel('Timestep', color='white')
        ax.set_ylabel(r'$H(X)$', color='white')
        fig.suptitle('2d Shannon entropy, $H$ spanning the noise schedule', color='white')
        ax.set_yticks(ax.get_yticks(), ax.get_yticks(), color='white')
        fig.set_facecolor('black')
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white', bbox_to_anchor=(1., -.25), 
                  ncol=3,
                  #loc='upper left'
                  )
        ax.set_facecolor('black')
        plt.grid()

        return fig, metadata
        
    
    num_images=100
    
    fig, metadata = shannon_entropy_across_time_plot(ims, num_timesteps=30, num_noise_samples=10)

    cols = st.columns((2,3))

    with cols[0]:
        metadata
    
    cols[1].pyplot(fig)