import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from subpages.diffusion_intro import plot_variance_schedule_image_series, plot_variance_schedule_hists, prep_image
from utils_app import kl_score, VarianceSchedule, load_or_save_fig
import torch.nn as nn
import math

deactivate = True


  
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

def write_scientific_notation(x):
    # x = f"{x:.2e}"
    # x = r"{x}".format(x=x)
    # x = x.replace("1.00", "")
    # x = x.replace(".00", r"\\times")
    # x = x.replace("e-0", r"10^{-")
    # x = x.replace("e+0", r"10^{")
    # x = x.replace("e", r"10^{")
    # x += "}"
    return x

# @load_or_save_fig("assets_produced/3_Diffusion_theory/noise_level_over_time.png", deactivate=deactivate)
def plot_noise_levels(schedules, colors = ["red", "orange", "salmon"], linestyles = ["-", "--", "-."]):
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i, m in enumerate(schedules.keys()):
        label = f"{m}"
        epsilon = write_scientific_notation(schedules[m].epsilon)
        beta_start = write_scientific_notation(schedules[m].beta_start)
        beta_end = write_scientific_notation(schedules[m].beta_end)
        st.write(epsilon, beta_start, beta_end)
        epsilon = r'8*10^{-5}'
        beta_start = r'10^{-6}'
        beta_end = r'10^{-2}'
        if m == "cosine":
            label += f"  ($\\epsilon={epsilon}$)"
        elif m == "linear":
            label += f"   ($\\beta_0={beta_start}$, $\\beta_T={beta_end}$)"
        elif m == "square":

            label += f" ($\\beta_0={beta_start}$, $\\beta_T={beta_end}$)"

        ax.plot(schedules[m].sqrt_one_minus_alphas_cumprod,
                label=label,
                color=colors[i],
                linestyle=linestyles[i])
        

    ax.set_title("Variance schedule for different methods")
    ax.set_xlabel("Timestep")
    ax.grid()
    ax.set_ylabel("Noise level: $\\sqrt{1-\\bar\\alpha_i}$")
    ax.legend(loc='lower right', ncol=1)
    plt.tight_layout()
    return fig

# @load_or_save_fig("assets_produced/3_Diffusion_theory/noise_grid.png", deactivate=deactivate)
def plot_noising_grid(T, schedules, example_img):

    n_imgs = 6

    ts = torch.linspace(0, T-1, n_imgs, dtype=torch.int32)
    noised = {m : {
        t.item() : schedules[m](example_img, t) for t in ts} for m in schedules.keys()}

    # Example usage
    fig, ax = plt.subplots(3, n_imgs, figsize=(10, 6))
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

# @load_or_save_fig("assets_produced/3_Diffusion_theory/entropy_across_time.png", deactivate=deactivate)
def noise_across_time(T, schedules, example_img, colors = ["red", "orange", "salmon"], linestyles = ["-", "--", "-."]):
    example_img = example_img[0]
    data = {m : [] for m in schedules.keys()}
    ts = torch.linspace(0, T-1, 16, dtype=torch.int32)
    for m in schedules.keys():
        for t in ts:
            im = schedules[m](example_img, t).squeeze()
            # st.write(im.shape)
            #.permute(1, 2, 0).detach().numpy()
            KL = kl_score(im)
            data[m].append(KL)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for m in schedules.keys():
        ax.plot(ts, data[m], label=m, color=colors.pop(0), linestyle=linestyles.pop(0))


    ax.set_title("Kullback-Liebler score across noise schedule")
    ax.set_xlabel("Time-index")
    ax.set_ylabel("KL divergence from $\mathcal{N}(0, 1)$")
    ax.grid()
    ax.legend()
    plt.tight_layout()
    return fig



def noise_schedule_page():
    st.write("""        
    In the literature we see reference to a couple different base functions for the noise schedule. Here we compare three different base functions: linear, square, and cosine. Here we present a tool for visualizing the noise schedule and the noise level in an image across time.
    """)
    cols = st.columns(2)
    with cols[0]:

        st.write(r"""
        The base function is manipulated to yield the cumulative product:
        $$
            \alpha_t  =1-\beta_t, \quad\rightarrow\quad
            \bar\alpha_t = \prod_{i=1}^t \alpha_i
        $$
        """)
        st.write(r"""
        The noise level in an image is then given by:
        $$
        \sqrt{1-\bar\alpha_t}
        $$

        The cosine noise schedule is parameterized by $\epsilon$ and the base-function takes the form:

        $$
        \beta_t = \cos\left(\frac{\pi}{2}
                    \cdot
                    \frac{t/T + \epsilon}{1+\epsilon}\right).
        $$
        The linear and square noise schedules are parameterized by their start and end values, $\beta_0$ and $\beta_T$ respectively.
        """)
        st.write(r"""
        **A good noise schedule** is one that has a balance between: noising slowly enough that the model can learn, and noising quickly enough that the model can generalize.
        """)
    with cols[0]:
        sub_cols = st.columns(2)
        with sub_cols[0]:
            beta_start_linear = st.select_slider("Beta start (linear)", options=[1e-6, 1e-5, 1e-4, 1e-3], value=1e-6)
            beta_start_square = st.select_slider("Beta start (square)", options=[1e-6, 1e-5, 1e-4, 1e-3], value=1e-6)
            epsilon = st.select_slider("Epsilon", options=[0.00008, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01], value=0.00008)
        with sub_cols[1]:
            beta_end_linear = st.select_slider("Beta end (linear)", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.01)
            beta_end_square = st.select_slider("Beta end (square)", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.01)

    with cols[1]:
        img_path = 'assets/example_images/penguin.png'
        img = prep_image(img_path)
        T = 641

        schedules = {
            "linear" : VarianceSchedule(T, beta_start=beta_start_linear, beta_end=beta_end_linear, method="linear"),
            "square" : VarianceSchedule(T, beta_start=beta_start_square, beta_end=beta_end_square, method="square"),
            "cosine" : VarianceSchedule(T, epsilon=epsilon, method="cosine")
        }

        colors = ["red", "orange", "salmon"]
        linestyles = ["-", "--", "-."]

        fig1 = plot_noise_levels(schedules, colors, linestyles)
        fig2 = plot_noising_grid(T, schedules, img)
        fig3 = noise_across_time(T, schedules, img, colors, linestyles)
        st.pyplot(fig1)
        st.pyplot(fig2)
        st.pyplot(fig3)
        
