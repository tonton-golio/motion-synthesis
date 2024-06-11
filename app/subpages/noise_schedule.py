import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math

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

def plot_noising_schedule2(ims, VS, params_schedule):
    nt = 8
    fig, ax = plt.subplots(len(ims), nt, figsize=(22, 9))
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

            ax[j,i].imshow(im_noisy, cmap='gray_r')
            ax[j,i].axis('off')

    # fig.set_facecolor('black')
    st.pyplot(fig)

def shannon_entropy_across_time_plot(ims, params_schedule, VS, num_timesteps = 20, num_noise_samples=10, dark_mode=False):
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

    label_color = 'white' if dark_mode else 'black'

    ax.set_xlabel('Timestep', color=label_color)
    ax.set_ylabel(r'$H(X)$', color=label_color)
    fig.suptitle('2d Shannon entropy, $H$ spanning the noise schedule', color=label_color)
    ax.set_yticks(ax.get_yticks(), ax.get_yticks(), color=label_color)
    
    ax.legend(facecolor='black' if dark_mode else 'white',
                edgecolor='white', 
                labelcolor=label_color, bbox_to_anchor=(1., -.25), 
                ncol=3,
                #loc='upper left'
                )
    fig.set_facecolor('black' if dark_mode else 'white')
    ax.set_facecolor('black' if dark_mode else 'white')
    plt.grid()

    return fig, metadata

def mnist_noise_schedule_setup(ims):
    cols = st.columns((2,3))

    with cols[0]:
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
        fig, metadata = shannon_entropy_across_time_plot(ims, params_schedule, VS,
                                                         num_timesteps=30, num_noise_samples=10)
        metadata = {**metadata, **params_schedule}
        st.write(metadata)
        

    with cols[1]:
        plot_noising_schedule2(ims[:3], VS, params_schedule)
        
        st.pyplot(fig)

import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from app.utils_app import load_or_save_fig

from app.utils_app import kl_score, VarianceSchedule
from subpages.diffusion_intro import prep_image

deactivate = False

class VarianceSchedule123(nn.Module):

    def __init__(self, timesteps, method="cosine", **kwargs):
        super(VarianceSchedule123, self).__init__()
        self.timesteps = timesteps

        if method == "cosine":
            # st.write('using cosine, with epsilon:', kwargs.get("epsilon", 0.008))
            betas = self._cosine_variance_schedule(timesteps, epsilon=kwargs.get("epsilon", 0.008))
        elif method == "linear":
            betas = self._linear_variance_schedule(timesteps, 
                                                beta_start=kwargs.get("beta_start", 1e-5),
                                                beta_end=kwargs.get("beta_end", .01))
        elif method == "square":
            betas = self._sqr_variance_schedule(timesteps, 
                                                beta_start=kwargs.get("beta_start", 1e-4),
                                                beta_end=kwargs.get("beta_end", .1))
        else:
            raise NotImplementedError

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _cosine_variance_schedule(self, timesteps, epsilon=0.08):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = (
            torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)
            ** 2
        )
        betas = torch.clip(1 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        # betas = betas / torch.max(betas)
        return betas
    
    def _linear_variance_schedule(self, timesteps, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, steps=timesteps + 1, dtype=torch.float32)
        betas = torch.clip(betas[1:] , 0.0, 0.999)
        return betas
    
    def _sqr_variance_schedule(self, timesteps, beta_start=1e-4, beta_end=0.02):
        steps = torch.linspace(beta_start**0.5, beta_end**0.5, steps=timesteps + 1, dtype=torch.float32)
        f_t = steps**2
        betas = torch.clip(f_t[1:] , 0.0, 0.999)
        return betas


    def forward(self, x, t, clip=False):
        A, B = self.alphas_cumprod[t], self.sqrt_one_minus_alphas_cumprod[t]
        noise = torch.randn_like(x)*.1+.5
        # x_t = self.sqrt_alphas_cumprod[t] * x + self.sqrt_one_minus_alphas_cumprod[t] * noise
        x_t = A * x + B * noise
        if clip:
            x_t = torch.clip(x_t, 0.0, 1.0)
        return x_t
    
def shannon_entropy_2d(im, plot=False, title1='image'):
    im_arr = np.array(im)
    if len(im_arr.shape) == 3:
        im_arr = im_arr.mean(axis=2)
    # print(im_arr.shape)
    im_derivative = np.gradient(im_arr)
    # im_derivative[0].shape
    im_derivative = im_derivative[0]**2 + im_derivative[1]**2
    im_derivative /= im_derivative.sum()
    
    H = -np.sum(im_derivative[im_derivative != 0] * np.log2(im_derivative[im_derivative != 0]))

    # H tilde
    H_tilde = H / np.log2(2/3 * im_arr.size)
    H = H_tilde
    H = 1-((1-H) ** .25)
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
        # st.pyplot(fig)

    return H

@load_or_save_fig("assets_produced/3_Diffusion_theory/noise_level_over_time.png", deactivate=deactivate)
def plot_noise_levels(schedules, colors = ["red", "orange", "salmon"], linestyles = ["-", "--", "-."]):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for i, m in enumerate(schedules.keys()):
        ax.plot(schedules[m].sqrt_one_minus_alphas_cumprod,
                label=f"{m}", 
                color=colors[i],
                linestyle=linestyles[i])
        

    ax.set_title("Variance schedule for different methods ($\\sqrt{1-\\bar\\alpha_i}$)")
    ax.set_xlabel("Timesteps")
    ax.grid()
    ax.set_ylabel("Noise level $\\sqrt{1-\\bar\\alpha_i}$")
    ax.legend(loc='lower right', ncol=1)
    plt.tight_layout()
    return fig

@load_or_save_fig("assets_produced/3_Diffusion_theory/noise_grid.png", deactivate=False)
def plot_noising_grid(T, schedules, example_img):

    n_imgs = 6

    ts = torch.linspace(0, T-1, n_imgs, dtype=torch.int32)
    noised = {m : {
        t.item() : schedules[m](example_img, t) for t in ts} for m in schedules.keys()}

    # Example usage
    fig, ax = plt.subplots(3, n_imgs, figsize=(12, 7))
    fig.suptitle("Variance schedule for different methods", fontsize=17)

    for i, m in enumerate(schedules.keys()):
        for j, t in enumerate(ts):
            im = noised[m][t.item()]
            kl = kl_score(im)
            im = im.detach().numpy()
            H = shannon_entropy_2d(im)
            
            ax[i,j].imshow(im, cmap='gray')
            ax[i,j].set_title(f"t={t}, KL={kl:.2f}", fontsize=10)
            ax[i,j].axis("off")

        ax[i,0].text(-30, 80, m, fontsize=15, fontweight='bold', rotation=90)

    plt.tight_layout()
    return fig

@load_or_save_fig("assets_produced/3_Diffusion_theory/entropy_across_time.png", deactivate=deactivate)
def noise_across_time(T, schedules, example_img, colors = ["red", "orange", "salmon"], linestyles = ["-", "--", "-."]):
    example_img = example_img[0]
    data = {m : [] for m in schedules.keys()}
    ts = torch.linspace(0, T-1, 16, dtype=torch.int32)
    for m in schedules.keys():
        for t in ts:
            im = schedules[m](example_img, t).squeeze()
            # st.write(im.shape)
            #.permute(1, 2, 0).detach().numpy()
            H = shannon_entropy_2d(im)
            data[m].append(H)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for m in schedules.keys():
        ax.plot(ts, data[m], label=m, color=colors.pop(0), linestyle=linestyles.pop(0))


    noise_H = []
    for _ in range(10):
        noise = np.random.randn(*example_img.shape)*.15+.5
        H = shannon_entropy_2d(noise)
        noise_H.append(H)

    noise_H_mu, noise_H_std = np.mean(noise_H), np.std(noise_H)
    st.write(f"Pure noise: {noise_H_mu:.2f} ± {noise_H_std:.2f}")
    ax.axhline(noise_H_mu, color='black', linestyle='-', label='pure noise')
    ax.fill_between(ts, noise_H_mu - noise_H_std, noise_H_mu + noise_H_std, color='black', alpha=0.2)
    ax.set_title("Shannon entropy across time")

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Shannon entropy")
    ax.grid()
    ax.legend()
    plt.tight_layout()
    return fig



# Example usage
def NoiseScheduleDemo():
    """
    Produces two plots:
        1. Plot of the noise levels for different variance schedules
        2. Grid of images noised with different variance schedules
    """
    
    
    T = 641
    methods = ["linear", "square", "cosine"]
    schedules = {m : VarianceSchedule(T, epsilon=0.00008, 
                                    beta_start=1e-6, 
                                    beta_end=0.01, 
                                    method=m) 
                                    for m in methods}
    img_path = "assets/example_images/cat.png"
    example_img = prep_image(img_path)
    # st.write(example_img.shape)
    # st.write(example_img)
    colors = ["red", "orange", "salmon"]
    linestyles = ["-", "--", "-."]

    fig1 = plot_noise_levels(schedules, colors, linestyles)
    fig2 = plot_noising_grid(T, schedules, example_img)
    fig3 = noise_across_time(T, schedules, example_img, colors, linestyles)

    # noise  = torch.randn_like(example_img)*.1+.5
    # fig4, ax = plt.subplots(4, 3, figsize=(12, 12))
    # ax = ax.flatten()

    # fig5, ax5 = plt.subplots(1, 1, figsize=(6, 3))


    # ax[0].hist(noise.flatten(), bins=30, label='pure noise', alpha=0.5)
    # ax[-1].hist(noise.flatten(), bins=30, label='pure noise', alpha=0.5)
    
    # ax[0].hist(example_img.flatten(), bins=30, label='cat', alpha=0.5)
    # ax[0].legend()
    # ax[0].set_yscale('log')
    # ax[0].set_title('Histogram of pixel values')

    # for i, t in enumerate([0, 10, 100, 200, 300, 400, 500, 600]):
    #     caaet = schedules['cosine'](example_img, t).squeeze().permute(1, 2, 0).detach().numpy()
    #     ax[i+1].hist(caaet.flatten(), bins=30, label=f'linear, t={t}', alpha=0.5)
    #     ax5.hist(caaet.flatten(), bins=30, label=f'cosine, t={t}', alpha=0.5)
    #     ax5.set_yscale('log')
    # plt.tight_layout()
    # ax[0].grid()

    # st.pyplot(fig5)
    


    return fig1, fig2, fig3
    

