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

class VarianceSchedule(nn.Module):

    def __init__(self, timesteps, method="cosine", **kwargs):
        super(VarianceSchedule, self).__init__()
        self.timesteps = timesteps

        if method == "cosine":
            st.write('using cosine, with epsilon:', kwargs.get("epsilon", 0.008))
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
VS = VarianceSchedule(**params_schedule)
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

'''
What makes a good noise schedule?
* We dont want a bunch of pure noise images at the end of the schedule
* But we do want some pure noise images, such that we can diffuse from pure noise.
* If we see the shadow of the original image, even in the highest noise case, Then the model struggels when given pure noise.

*So its a goldiloch situation!*
'''

# Example usage
# fig, ax = plt.subplots(2, 6, figsize=(12, 4))
fig = plt.figure(figsize=(8, 6))
fig.suptitle("Variance schedule for different methods")

n_imgs = 5
T = 200
methods = ["cosine", "linear", "square"]
colors = ["red", "blue", "green"]

schedules = {m : VarianceSchedule(T, epsilon=0.00008, method=m) for m in methods}
gs = gridspec.GridSpec(len(methods), n_imgs + 1, width_ratios=[3] + [1]*n_imgs)
ax_coor = [fig.add_subplot(gs[:1, 0]), fig.add_subplot(gs[1:, 0])]
im_ax = [[fig.add_subplot(gs[i, j+1]) for i in range(len(methods))] for j in range(n_imgs)]


for i, m in enumerate(methods):
    ax_coor[0].plot(schedules[m].betas, label=f"{m}", color=colors[i], linestyle="--")
    ax_coor[1].plot(schedules[m].sqrt_one_minus_alphas_cumprod, label=f"{m}", color=colors[i])


ax_coor[0].set_title("Base function: $\\beta$")
ax_coor[0].set_xlabel("Timesteps")
ax_coor[1].set_title("Noise: $\\sqrt{1 - \\bar\\alpha_i}$")
ax_coor[1].set_xlabel("Timesteps")
# ax_coor[0].legend(ncol=1)
ax_coor[1].legend()
ax_coor[0].grid()
ax_coor[1].grid()

example_img = plt.imread("assets/example_img.png")
example_img = torch.tensor(example_img).permute(2, 0, 1).unsqueeze(0).float()
noise = torch.randn_like(example_img)

ts = torch.linspace(0, T-1, n_imgs, dtype=torch.int32)

noised = {m : [schedules[m](example_img, noise, t) for t in ts] for m in methods}

for j, m in enumerate(methods):
    for i, t in enumerate(ts):
        im_ax[i][j].imshow(noised[m][i][0].permute(1, 2, 0).detach().numpy())
        im_ax[i][j].set_title(f"t={t}")
        im_ax[i][j].axis("off")

    im_ax[0][j].text(-200, 800, m, fontsize=12, fontweight='bold', rotation=90)


plt.tight_layout()
# plt.savefig("assets/variance_schedule_demo.png")

st.pyplot(fig)





plot_noising_schedule(ims[:1], VS)

"""
We see our noise schedule, but how do we determine whether it adds noise too fast, to slow, or just right?

We want the last image to be pure noise!

To identify whether an image is pure noise, we use the 2d formulation of the Shannon Entropy as per (https://arxiv.org/pdf/1609.01117.pdf). The way to do that is:
1. Compute the spatial derivative of the image (along both x and y axis)
2. Combine each spatial derivative to obtain the 2d derivative
3. Compute the Shannon entropy of the 2d derivative, in the normal fashion
    $$
    H = -\sum p(x) \log_2 p(x)
    $$

where $p(x)$ is the probability of pixel value $x$ in the 2d derivative. The more complete formulation from the paper is:
$$
H(\\nabla f)  = -\sum_{j=1}^J\sum_{i=1}^I p_{i,j} \log_2(p_{i,j})
$$
"""

shannon_entropy_2d(ims[0], plot=True)

'''
Now the metric works, and we want to see how the Shannon entropy changes as we add noise to the image. We plot the Shannon entropy for each image in the schedule.
'''
# check if image is pure noise
# if pure noise: the shannon entropy should be very high
# we use 2d finite difference to obtain the derivative as per: 

plot_noising_schedule2(ims[:5], VS)

"""Now to figure out what value of Shannon entropy corresponds to pure noisem, we calculate the metric for pure noise images:"""

pure_noise = torch.randn_like(ims[0])
shannon_entropy_2d(pure_noise, plot=True, title1='pure noise')

"""
We find that for noiseless images, that is, images corresponding to T = 0, the measured Shannon entropy (2d) is around 6.5 to 7.5. When we take an image of pure noise, and compute the Shannon entropy, we see values around 8.8. This means that we can use the Shannon entropy as a metric to determine if an image is pure noise or not.
"""


"""
---
# lets do some statistics
For a range of $t\in T$, we want to compute the Shannon entropy of the images. We can then make a errror bar plot of the results.
"""

num_images = 1000
nt = 20

ts = np.linspace(0, params_schedule['timesteps']-1, nt, dtype=int)
ims = [dm.data_train[i][0].squeeze() for i in range(num_images)]
res = {}
for t in ts:
    res[t] = []
    for i in range(num_images):
        noise = torch.randn_like(ims[0])
        im_noisy = VS(ims[0], noise, t)
        res[t].append(shannon_entropy_2d(im_noisy))

    # convert to mean and std
    res[t] = np.array(res[t])
    res[t] = res[t].mean(), res[t].std()

# lets calculate it for pure noise
res_pure_noise = []
for i in range(10):
    pure_noise = torch.randn_like(ims[0])
    res_pure_noise.append(shannon_entropy_2d(pure_noise))

res_pure_noise = np.array(res_pure_noise)
res_pure_noise = res_pure_noise.mean(), res_pure_noise.std()

# lets calculate it for clean images
res_clean = []
for i in range(num_images):
    res_clean.append(shannon_entropy_2d(ims[i]))

res_clean = np.array(res_clean)
res_clean = res_clean.mean(), res_clean.std()

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.errorbar(list(res.keys()), [r[0] for r in res.values()], yerr=[r[1] for r in res.values()], label='noisy images')

ax.axhline(res_pure_noise[0], color='red', linestyle='--', label='pure noise (mean)')
ax.fill_between([0, params_schedule['timesteps']], res_pure_noise[0]-res_pure_noise[1], res_pure_noise[0]+res_pure_noise[1], color='red', alpha=0.5, label='pure noise (std)')

ax.axhline(res_clean[0], color='green', linestyle='--', label='clean images (mean)')
ax.fill_between([0, params_schedule['timesteps']], res_clean[0]-res_clean[1], res_clean[0]+res_clean[1], color='green', alpha=0.5, label='clean images (std)')

ax.set_xlabel('t')
ax.set_ylabel('Shannon entropy')
fig.suptitle('2d Shannon entropy across noise schedule')
plt.legend()
plt.grid()
st.pyplot(fig)
