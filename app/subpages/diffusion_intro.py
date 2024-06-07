import streamlit as st
import torch
import matplotlib.pyplot as plt
import math
from torch import nn
from torchvision import transforms
import numpy as np

from utils import load_or_save_fig, VarianceSchedule, kl_score
deactivate = True


    

def prep_image(img_path):
    img = plt.imread(img_path)[:, :, 0]
    img = torch.tensor(img, dtype=torch.float32)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=.5)
    ])

    img = transform(img).squeeze()
    return img


def plot_variance_schedule_image_series(img, vs, noise_type='uniform', kl=False):
    T = vs.timesteps
    fig_imgs, axes_imgs = plt.subplots(1, T+1, figsize=(10, 5))
   
    noised = {i+1:    vs(img, i, noise=None, clip=False, noise_type=noise_type) for i in range(T)}
    noised[0] = img

    for i in range(T+1):
        if kl:
            kl = kl_score(noised[i])
            axes_imgs[i].set_title(f'$x_{i}$, KL: {kl:.2f}')
        else:
            axes_imgs[i].set_title(f'$x_{i}$')
        
        axes_imgs[i].imshow(noised[i], cmap='gray')
        axes_imgs[i].axis('off')
        
        print(noised[i].shape)
        print(f'$x_{i}$' + f'done, min: {noised[i].min()}, max: {noised[i].max()}')

    plt.tight_layout()

    return fig_imgs

@load_or_save_fig('assets_produced/3_diffusion_theory/variance_schedule_image_series_uniform.png', deactivate=deactivate)
def plot_variance_schedule_image_series_uniform(img, vs):
    return plot_variance_schedule_image_series(img, vs, noise_type='uniform')

@load_or_save_fig('assets_produced/3_diffusion_theory/variance_schedule_image_series_normal.png', deactivate=deactivate)
def plot_variance_schedule_image_series_normal(img, vs):
    return plot_variance_schedule_image_series(img, vs, noise_type='normal')



def plot_variance_schedule_hists(img, vs, noise_type='normal'):
    T = vs.timesteps    
    # apply noise
    noised = {i+1:    vs(img, i, noise=None, clip=False, noise_type=noise_type) for i in range(T)}
    noised[0] = img

    x = torch.linspace(-1, 1, steps=1000)
    kde = torch.exp(-x**2 /.20 )/(2*math.pi)**.5
    kde *= 1/kde.max()/16

    # make histograms
    fixed_bins = torch.linspace(-1, 1, steps=51)
    hists = [np.histogram(noised[i].flatten(), bins=fixed_bins) for i in range(T+1)]

    # normalize histograms
    for i in range(T+1):
        hists[i] = (hists[i][0] / hists[i][0].sum(), hists[i][1])
    
    fig_hist, axes_hist = plt.subplots(1, T+1, figsize=(10, 3), sharey=True, sharex=True)
    fig_hist, axes_hist = plt.subplots(1, T+1, figsize=(10, 3), sharey=True, sharex=True)
    for i in range(T+1):        
        axes_hist[i].stairs(hists[i][0], hists[i][1],
                            fill=True, alpha=0.3, color='orangered',lw=1, label='histogram', edgecolor='black')
        # axes_hist[i].set_yscale('log')
        axes_hist[i].set_title(f'$x_{i}$')
        
        #axes_hist[i].plot(x, kde, label='KDE', color='blue', lw=1)
        axes_hist[i].set_xlim(-1, 1)
        axes_hist[i].set_yticks([.05])
        axes_hist[i].set_xticks([-1, 0, 1])

    fig_hist.suptitle('Histograms of pixel values from noised sample, $x_t$')
    plt.tight_layout()

    return fig_hist

@load_or_save_fig('assets_produced/3_diffusion_theory/variance_schedule_histograms_normal.png', deactivate=deactivate)
def plot_variance_schedule_hists_normal(img, vs):
    return plot_variance_schedule_hists(img, vs, noise_type='normal')

@load_or_save_fig('assets_produced/3_diffusion_theory/variance_schedule_histograms_uniform.png', deactivate=deactivate)
def plot_variance_schedule_hists_uniform(img, vs):
    return plot_variance_schedule_hists(img, vs, noise_type='uniform')


def diffusion_intro():


    T = 5
    vs = VarianceSchedule(T, method="cosine", epsilon=0.08)

    img_path = 'assets/example_images/cat.png'

    # fig_imgs, fig_hist = plot_variance_schedule(img_path, vs)
    img = prep_image(img_path)
    fig_imgs = plot_variance_schedule_image_series_normal(img, vs)
    fig_hist = plot_variance_schedule_hists_normal(img, vs)
    
    st.write("""
    To learn the reverse diffusion process, we need first to define the forward process. 


    """)
    st.pyplot(fig_imgs)
    st.caption('a sample image, shown without noise on the left, with noise added with timestep increasing rightwards')

    st.write("""
    The iterative sample distortion process is defined by normal offsets. We can consider this a linear combination of a sample with a noise vector;
    """)
    st.write(r"""
    $$
        x_{t+1} =a x_t + b \mathcal{N}(\mu, \sigma^2).
    $$

    With the noise characterized by the population-mean $\mu$ and -variance $\sigma^2$, we distort the sample, but without shifting the distribution.

    """)
    st.pyplot(fig_hist)
    st.caption('histograms of the noised samples, with timestep increasing rightwards. The histograms are shown in log scale and share axes and bins')


    st.write(r"""
             $a$ and $b$ are determined by the variance schedule which we denote $\beta_t$, a time dependent function in the range [0,1]. Regardless of the specific schedule, we obtain $a$ and $b$ by
             $$
                \alpha_t \equiv 1 - \beta_t\\
                \bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i\\
                a = \sqrt{\bar{\alpha}_t}, \qquad
                b = \sqrt{1 - \bar{\alpha}_t}
             $$
             """)
    

    
    st.write(r"""
    For the plots above, we have employ a `cosine` variance schedule, which is a common choice. Other choices include `linear` and `square`. These describe $\beta_t, t=1,2,\ldots,T$.
             """)
    

    st.divider()
    st.write("""
            Now with uniform noise, we obtain the following histograms:
            """)
    fig_imgs = plot_variance_schedule_image_series_uniform(img, vs)
    fig_hist = plot_variance_schedule_hists_uniform(img, vs)
    st.pyplot(fig_imgs)
    st.pyplot(fig_hist)
    
    

