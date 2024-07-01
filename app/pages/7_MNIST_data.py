import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import umap
import sys
sys.path.append('..')
from mnist_latent_diffusion.modules.dataModules import MNISTDataModule

from utils_app import VarianceSchedule
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
    
        
    def make_grid(ds, nrow=3, ncols=8):
        ims = np.array([ds[i][0].squeeze() for i in range(nrow * ncols)])
        fig, axes = plt.subplots(nrow, ncols, figsize=(2*ncols, 2*nrow))
        for ax, im in zip(axes.flatten(), ims):
            ax.imshow(im, cmap='gray_r')
            ax.axis('off')
        st.pyplot(fig)

    def plot_distribution(ims):
        
        vals = ims.flatten()
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.hist(vals, bins=100)
        ax.set_xlabel('Pixel value')
        ax.set_ylabel('Frequency')
        ax.set_yscale('log')
        fig.suptitle('Pixel value distribution')
        plt.tight_layout()
        plt.close(fig)
        return fig
    
    with st.sidebar:  # parameters for the data module
        params_data = dict(
            BATCH_SIZE = 16,
            ROTATION = st.slider('rotation', 0, 90, 0),
            SCALE = st.slider('scale', 0., 1., 0.0),
            TRANSLATE_X = st.slider('translate x', 0., 1., 0.0),
            TRANSLATE_Y = st.slider('translate y', 0., 1., 0.0),
            SHEAR = st.slider('shear', 0., 1., 0.0),

            NORMALIZE_MEAN = st.slider('normalize mean', 0., 1., 0.5,), 
            NORMALIZE_STD = st.slider('normalize std', 0.01, 1., .5),

            BOOL = st.checkbox('bool'),
            NO_NORMALIZE = st.checkbox('no normalize')
        )


    # title and intro
    """
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
    
    # embedding UMAP
    with st.sidebar:
        n_samples = st.select_slider('number of samples', options=(2**np.arange(8, 12)).astype(int))
        ims = np.array([dm.data_train[i][0].squeeze() for i in range(n_samples)])
        labs = [dm.data_train[i][1] for i in range(n_samples)]

    cols = st.columns(2)
    with cols[0]:
        fig = plot_distribution(ims)
        st.pyplot(fig)

    
    ims = np.array(ims).reshape(n_samples, -1)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(ims)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(embedding[:, 0], embedding[:, 1], c=labs, cmap='tab10')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('UMAP embedding of MNIST dataset')
    plt.tight_layout()
    cols[1].pyplot(fig)
    
    """
    Beware that the pixel value extremes, limits us to the kind of activation if any we may choose to apply at the end of our network.

    For example, if we choose to use a sigmoid activation function, we will have to normalize the pixel values to be between 0 and 1. If we choose to use a tanh activation function, we will have to normalize the pixel values to be between -1 and 1.

    So for sigmoid, we could sent normalize mean to 0.0 and normalize std to 1.0. For tanh, we could set normalize mean to 0.5 and normalize std to 0.5.
    """

    n_samples = 3
    ims_n = ims[:n_samples].reshape(n_samples, 28, 28)
    ims_n.shape
    import torch
    ims_n = torch.tensor(ims_n, dtype=torch.float32)
    
    T = 8
    noised = []
    vs = VarianceSchedule(T)
    for im in ims_n:
        tmp = []
        for t in range(T):
            if t == 0:
                tmp.append(im)
            else:
                tmp.append(vs(im, t))
        noised.append(torch.stack(tmp)  )

    noised = torch.stack(noised)
    noised.shape

    fig, axes = plt.subplots(n_samples, T, figsize=(2*T, 2*n_samples))
    for i, ax in enumerate(axes):
        for j, a in enumerate(ax):
            a.imshow(noised[i, j], cmap='gray_r', vmin=0, vmax=1)
            a.axis('off')

    plt.tight_layout()
    st.pyplot(fig)
    

    
