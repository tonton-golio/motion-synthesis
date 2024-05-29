import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

class VarianceSchedule(nn.Module):

    def __init__(self, timesteps, method="cosine", **kwargs):
        super(VarianceSchedule, self).__init__()
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


    def forward(self, x, t, clip=True):
        A, B = self.alphas_cumprod[t], self.sqrt_one_minus_alphas_cumprod[t]
        noise = torch.randn_like(x)
        # x_t = self.sqrt_alphas_cumprod[t] * x + self.sqrt_one_minus_alphas_cumprod[t] * noise
        x_t = (A * x + B * noise)/(A + B)
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

# Example usage
def NoiseScheduleDemo():
    """
    Produces two plots:
        1. Plot of the noise levels for different variance schedules
        2. Grid of images noised with different variance schedules
    """
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
    
    def plot_noising_grid(schedules, example_img):

        n_imgs = 8

        ts = torch.linspace(0, T-1, n_imgs, dtype=torch.int32)
        noised = {m : {
            t.item() : schedules[m](example_img, t) for t in ts} for m in schedules.keys()}

        # Example usage
        fig, ax = plt.subplots(3, n_imgs, figsize=(8, 6))
        fig.suptitle("Variance schedule for different methods")

        for i, m in enumerate(schedules.keys()):
            for j, t in enumerate(ts):
                im = noised[m][t.item()].squeeze().permute(1, 2, 0).detach().numpy()
                H = shannon_entropy_2d(im)
                ax[i,j].imshow(im)
                ax[i,j].set_title(f"t={t}, H={H:.2f}", fontsize=8)
                ax[i,j].axis("off")

            ax[i,0].text(-200, 800, m, fontsize=12, fontweight='bold', rotation=90)

        plt.tight_layout()
        return fig

    def noise_across_time(schedules, example_img, colors = ["red", "orange", "salmon"], linestyles = ["-", "--", "-."]):
        data = {m : [] for m in schedules.keys()}
        ts = torch.linspace(0, T-1, 20, dtype=torch.int32)
        for m in schedules.keys():
            for t in ts:
                im = schedules[m](example_img, t).squeeze().permute(1, 2, 0).detach().numpy()
                H = shannon_entropy_2d(im)
                data[m].append(H)
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        for m in schedules.keys():
            ax.plot(ts, data[m], label=m, color=colors.pop(0), linestyle=linestyles.pop(0))

        # now calc for pure noise
        noise_H = []
        for i in range(10):
            noise = torch.randn_like(example_img)
            H = shannon_entropy_2d(noise)
            noise_H.append(H)

        noise_H_mu, noise_H_std = np.mean(noise_H), np.std(noise_H)
        ax.axhline(noise_H_mu, color='black', linestyle='-', label='pure noise')
        ax.fill_between(ts, noise_H_mu - noise_H_std, noise_H_mu + noise_H_std, color='black', alpha=0.2)
        ax.set_title("Shannon entropy across time")

        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Shannon entropy")
        ax.grid()
        ax.legend()
        plt.tight_layout()
        return fig
        

    T = 641
    methods = ["linear", "square", "cosine"]
    schedules = {m : VarianceSchedule(T, epsilon=0.008, 
                                    beta_start=1e-6, 
                                    beta_end=0.02, 
                                    method=m) 
                                    for m in methods}
    example_img = plt.imread("assets/example_images/cat.png")
    example_img = torch.tensor(example_img).permute(2, 0, 1).float()
    # st.write(example_img.shape)
    # st.write(example_img)
    colors = ["red", "orange", "salmon"]
    linestyles = ["-", "--", "-."]

    fig1 = plot_noise_levels(schedules, colors, linestyles)
    fig2 = plot_noising_grid(schedules, example_img)
    fig3 = noise_across_time(schedules, example_img, colors, linestyles)

    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)

