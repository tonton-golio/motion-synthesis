import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import sys
import streamlit as st
import math
from utils_app import load_or_save_fig, VarianceSchedule, kl_score

deactivate = False

# PARTICLE DIFFUSION
def get_x_and_x_boxed(n, d, T, box_lim=3):
    x = np.zeros((n, T, d))
    x_boxed = np.zeros((n, T, d))
    for i in range(1, T):
        x[:, i, ] = x[:, i-1] + np.random.randn(n, d)

        x_new = x_boxed[:, i-1] + np.random.randn(n, d)*.02
        x_new[x_new > box_lim] = box_lim
        x_new[x_new < -box_lim] = -box_lim
        x_boxed[:, i, ] = x_new
    return x, x_boxed

def sci_num_print(t):
    def sci_num(x):
        return "{:.0e}".format(x)
    
    s = sci_num(t)
    a = s.split('e')[0]
    b = int(s.split('e')[1].split('+')[1])

    if b == 0:   return a
    elif b == 4: return r"$t\rightarrow\infty$"
    else:        return f"$t={10}^{b}$"

def plot_particle_diffusion(x, x_boxed, ts, ax, colors = ['orangered', 'purple']):

    for i, t in enumerate(ts[:-1]):
        ax[i].scatter(x_boxed[:, t, 0], x_boxed[:, t, 1], alpha=0.2, s=3, c=colors[0])
        ax[i].set_title(f"{sci_num_print(t)}")
        ax[i].set_xticks([])
        # ax[i].axis('equal')

        ax[-(i+1)].scatter(x[:, t, 0], x[:, t, 1], alpha=0.2, s=3, c=colors[1])
        ax[-(i+1)].set_title(f"{sci_num_print(t)}")
        ax[-(i+1)].set_xticks([])
        # ax[i].axis('equal')

    ax[4].scatter(x_boxed[:, 0, 0], x_boxed[:, 0, 1], alpha=0.2, s=6, c='black', marker='x')
    ax[4].set_title(f"t=0")
    for i in range(4):
        ax[i].set_xlim(-3.5, 3.5)
        ax[i].set_ylim(-3.5, 3.5)

    for i in range(4, 9):
        ax[i].set_xlim(-350, 350)
        ax[i].set_ylim(-350, 350)
        
    for axi in ax:
        axi.set_xticks([])
        axi.set_yticks([])

    ax[1].set_xlabel(" "*24+"Diffusion in a box", labelpad=-135, fontsize=14)
    ax[6].set_xlabel(" "*23+"Unconstrained diffusion", labelpad=-135, fontsize=14)

# IMAGE DIFFUSION
def prep_image(img_path, size=(128, 128)):
    img = plt.imread(img_path)[:, :, 0]
    img = torch.tensor(img, dtype=torch.float32)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=.5)
    ])

    img = transform(img).squeeze()
    return img

def plot_variance_schedule_image_series(img, vs, noise_type='uniform', kl=False, ax=None):
    T = vs.timesteps
    
   
    noised = {i+1:    vs(img, i, clip=False, noise_type=noise_type) for i in range(T)}
    noised[0] = img

    for i in range(T+1):
        if kl:
            kl = kl_score(noised[i])
            # ax[i].set_title(f'$x_{i}$, KL: {kl:.2f}')
        else:
            pass
            #ax[i].set_title(f'$x_{i}$')
        
        ax[i].imshow(noised[i], cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        
        # print(noised[i].shape)
        # print(f'$x_{i}$' + f'done, min: {noised[i].min()}, max: {noised[i].max()}')

    plt.tight_layout()

    return ax

def plot_variance_schedule_hists(img, vs, noise_type='normal', ax=None):
    T = vs.timesteps    
    # apply noise
    noised = {i+1:    vs(img, i, clip=False, noise_type=noise_type) for i in range(T)}
    noised[0] = img

    x = torch.linspace(-1, 1, steps=1000)
    kde = torch.exp(-x**2 /.20 )/(2*math.pi)**.5
    kde *= 1/kde.max()/16

    # make histograms
    fixed_bins = torch.linspace(-3, 3, steps=51)
    hists = [np.histogram(noised[i].flatten(), bins=fixed_bins) for i in range(T+1)]

    # normalize histograms
    for i in range(T+1):
        hists[i] = (hists[i][0] / hists[i][0].sum(), hists[i][1])

    for i in range(T+1):        
        ax[i].stairs(hists[i][0], hists[i][1],
                            fill=True, alpha=0.3, color='red',lw=1, label='histogram', edgecolor='black')
        # axes_hist[i].set_yscale('log')
        # ax[i].set_title(f'$x_{i}$')
        
        #axes_hist[i].plot(x, kde, label='KDE', color='blue', lw=1)
        # ax[i].set_xlim(-1.15, 1.15)
        ax[i].set_yticks([])
        ax[i].set_xticks([])


# MAIN
@load_or_save_fig(savepath="assets_produced/3_diffusion_theory/diffusion_constriction_plot", deactivate=deactivate)
def diffusion_constriction_plot():
    ## PARTICLE DIFFUSION
    n, d, T = 1000, 2, 10001
    box_lim = 3
    x, x_boxed = get_x_and_x_boxed(n, d, T, box_lim)

    # set up Ts to plot
    ts = np.logspace(0, 4, 5, dtype=int)[::-1]
    ts[-1] =0

    ## IMAGE DIFFUSION
    T = 4
    vs = VarianceSchedule(T, method="cosine", epsilon=0.08)
    img_path = 'assets/example_images/cat.png'
    img = prep_image(img_path)


    fig, ax = plt.subplots(3, 9, figsize=(12, 5))

    # make ax[2, :] share x and y
    for i in range(9-1):
        ax[2, i].sharex(ax[2, i+1])
        ax[2, i].sharey(ax[2, i+1])

    plot_particle_diffusion(x, x_boxed, ts, ax[0], colors=['orangered', 'purple'])
    plot_variance_schedule_image_series(img, vs, noise_type='normal', kl=False, ax=ax[1, 4:])
    plot_variance_schedule_image_series(img, vs, noise_type='uniform', kl=False, ax=ax[1, :5][::-1])
    plot_variance_schedule_hists(img, vs, noise_type='normal', ax=ax[2, 4:])
    plot_variance_schedule_hists(img, vs, noise_type='uniform', ax=ax[2, :5][::-1])

    ax[0, 0].set_ylabel("Particle\n diffusion", fontsize=14, rotation=0, labelpad=50, loc='center')
    ax[1, 0].set_ylabel("Image\n diffusion", fontsize=14, rotation=0, labelpad=50)
    ax[2, 0].set_ylabel("Pixel\n  distribution", fontsize=14, rotation=0, labelpad=50)

    # fig.suptitle("               Constricted diffusion vs. unconstricted diffusion", fontsize=16)

    plt.tight_layout()

    return fig

def diffusion_intro():

    fig = diffusion_constriction_plot()

   
    st.write("""
    To learn the reverse diffusion process, we need first to define the forward process. 
    

    """)
    st.pyplot(fig)
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
    # st.pyplot(fig_hist)
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
    # fig_imgs = plot_variance_schedule_image_series_uniform(img, vs)
    # fig_hist = plot_variance_schedule_hists_uniform(img, vs)
    # st.pyplot(fig_imgs)
    # st.pyplot(fig_hist)
    
    

