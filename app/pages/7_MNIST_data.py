import streamlit as st

# Intro and title
"""
# MNIST Data
"""

tab_names = [
    'Transformations',
]

tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}

# Transformations
with tabs['Transformations']:
    import sys
    sys.path.append('..')
    import streamlit as st
    from mnist_latent_diffusion.modules.dataModules import MNISTDataModule

    import matplotlib.pyplot as plt
    import numpy as np

    import torch
    import torch.nn as nn
    import math
    from matplotlib import gridspec
        
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
        fig, ax = plt.subplots(5, 10, figsize=(10, 6))
        ts = np.linspace(0, params_schedule['timesteps']-1, 10, dtype=int)
        for i, t in enumerate(ts):
            for j, im in enumerate(ims):
                noise = torch.randn_like(im)

                im_noisy = VS(im, noise, t)
                S = shannon_entropy_2d(im_noisy)

                ax[j,i].set_title(f"H={S:.2f}")

                ax[j,i].imshow(im_noisy, cmap='gray_r')
                ax[j,i].axis('off')
        st.pyplot(fig)


    with st.sidebar:  # parameters for the data module
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

    # title and intro
    """
    # MNIST Diffusion: Data and noise schedule

    Source: https://stats.stackexchange.com/questions/235270/entropy-of-an-image

    ---
    ### Transformations
    Here you see our MNIST dataset with applied transformations. So what are good transformations. We want to introduce greater variation in our dataset, such that it learns to subtract noise regardsless of sclaing, rotation, translation, and shear.

    To set good values for these parameters, we try to set them as high as possible, without obfuscating the original image excessively.


    """

    dm = MNISTDataModule(verbose=False, **params_data)
    dm.setup()
    # VS = VarianceSchedule(**params_schedule)
    make_grid(dm.data_train)
    """
    Since we will be adding noise, sampled from a standard Gaussian distribution $\mathcal{N}(0,1)$, we want make sure pixels values are appropriate. An approate pixel value is between 0 and 1. We can check this by plotting the distribution of pixel values in the dataset.
    """
    plot_distribution(dm.data_train)

    # look at the distribution now that we have applied transformations

    # embedding UMAP
    import umap
    n_umap = 500
    ims = [dm.data_train[i][0].squeeze() for i in range(n_umap)]
    labs = [dm.data_train[i][1] for i in range(n_umap)]
    ims = np.array(ims).reshape(n_umap, -1)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(ims)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(embedding[:, 0], embedding[:, 1], c=labs, cmap='tab10')
    st.pyplot(fig)


    '---'

    ims = [dm.data_train[i][0].squeeze() for i in range(5)]

    