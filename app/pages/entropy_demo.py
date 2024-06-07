import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.append('../')


from utils import load_or_save_fig


# Vector Entropy
def normalize(v, ord=1):
    norm = np.linalg.norm(v, ord=ord)
    if norm == 0: 
       return v
    return v / norm

def grayscale(im):
    if im.shape[-1] == 3:
        weights = np.array([0.2989, 0.5870, 0.1140])
    elif im.shape[-1] == 4:
        weights = np.array([0.2989, 0.5870, 0.1140, 0])
    return np.dot(im, weights)

def resize(im, size):
    return np.array(Image.fromarray(im).resize(size))

def shannon_entropy_1d(v, norm_ord=None):
    if norm_ord is not None:
        v = normalize(v, ord=norm_ord)
    return -np.sum(v[v>0] * np.log2(v[v>0]))

def shannon_entropy_2d(im, return_all=False):
    im_arr = np.array(im)
    # im_arr.shape
    im_derivative = np.gradient(im_arr)
    # im_derivative[0].shape
    im_derivative = im_derivative[0]**2 + im_derivative[1]**2
    im_derivative /= im_derivative.sum()
    
    # H = -np.sum(im_derivative[im_derivative > 0] * np.log2(im_derivative[im_derivative > 0]))
    H = shannon_entropy_1d(im_derivative)
    if return_all:
        return H, im_derivative
    return H

@load_or_save_fig('assets_produced/3_diffusion_theory/vector_entropy.png')
def vector_entropy_demo(vector_length = 8):
    vl = vector_length 
    A = normalize(np.random.rand(vl))
    B = normalize(np.random.rand(vl)**20)
    # max(A), max(B)
    def make_plot():

        fig, ax = plt.subplots(1, 2, figsize=(12, 1), gridspec_kw={'width_ratios': [1, 1]})
        vmin = 0
        ax[0].imshow(A.reshape(1, -1), aspect='auto', vmin=vmin, vmax=1, cmap='Greys')
        rightplot = ax[1].imshow(B.reshape(1, -1), aspect='auto', vmin=vmin, vmax=1, cmap='Greys')
        # plt.colorbar(rightplot, ax=ax[1])

        ax[0].set_title(r'$H(\bar{a})$='+f'{shannon_entropy_1d(A):.2f}')
        ax[1].set_title(r'$H(\bar{b})$='+f'{shannon_entropy_1d(B):.2f}')

        for a, V in zip(ax, [A, B]):
            #a.set_xticks([])
            a.set_yticks([])
            a.set_xlabel('Vector component')

            # annotate the vector components with values
            for i, val in enumerate(V):
                a.text(i, 0, f'{val:.2f}', ha='center', va='center', color='white' if val > .5 else 'black')
                
        # plt.tight_layout()
        plt.close()
        return fig
    
    fig = make_plot()
    return fig

def image_entropy_demo(ims):
    st.write(r"""
    ### Image Entropy
    Given an image, we can calculate the entropy of the distribution of values in the image.
             
    $$
        H(\nabla f)  = -\sum_{j=1}^J\sum_{i=1}^I p_{i,j} \log_2(p_{i,j})
    $$

    The entropy of an image is calculated by taking the spatial derivative of the image, and calculating the entropy of the distribution of values in the derivative image.
    """)
    cols = st.columns(len(ims))
    for im, col in zip(ims, cols):
        H, im_derivative = shannon_entropy_2d(im, return_all=True)
        H_1d = shannon_entropy_1d(im.flatten())

        fig, ax = plt.subplots(2, 2, figsize=(8, 4))
        fig.suptitle(f"Shannon entropy: {H:.2f}, Naive 1d entropy: {H_1d:.2f}")

        # image
        im1 = ax[0,0].imshow(im, cmap='gray_r')
        # plt.colorbar(im1, ax=ax[0])
        ax[0,0].set_title('image')
        ax[0,0].axis('off')

        # derivative image
        ax[1,0].set_title('Spatial derivative (second order, 2d)')
        im2 = ax[1,0].imshow(im_derivative, cmap='gray_r')
        # plt.colorbar(im2, ax=ax[1])
        ax[1,0].axis('off')

        # image distribution
        dist = im.flatten()
        ax[0,1].hist(dist, bins=100)
        ax[0,1].set_title('image distribution')
        ax[0,1].set_yscale('log')

        # derivative distribution
        dist_diff = im_derivative.flatten()
        ax[1,1].hist(dist_diff, bins=100)
        ax[1,1].set_title('derivative distribution')
        ax[1,1].set_yscale('log')
        plt.tight_layout()
        col.pyplot(fig)


fig = vector_entropy_demo(10)
r"""
To measure how 

### Vector Entropy
Given two vectors of unit length (manhattan norm), we can calculate the entropy of the distribution of values in the vector.

A vector, $\bar{a}$, with values distributed uniformly, has maximum entropy, while a vector, $\bar{b}$, with all values concentrated in a single component has minimum entropy.
"""



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
"""Now to figure out what value of Shannon entropy corresponds to pure noisem, we calculate the metric for pure noise images:"""


"""
We find that for noiseless images, that is, images corresponding to T = 0, the measured Shannon entropy (2d) is around 6.5 to 7.5. When we take an image of pure noise, and compute the Shannon entropy, we see values around 8.8. This means that we can use the Shannon entropy as a metric to determine if an image is pure noise or not.
"""


"""
---
# lets do some statistics
For a range of $t\in T$, we want to compute the Shannon entropy of the images. We can then make a errror bar plot of the results.
"""


st.pyplot(fig)



st.divider()
im_4 = grayscale(plt.imread('assets/example_images/4.png'))
im_cat = grayscale(plt.imread('assets/example_images/cat.png'))
im_cat = resize(im_cat, (im_4.shape[0], im_4.shape[1]))
im_noise = np.random.rand(im_4.shape[0], im_4.shape[1])
image_entropy_demo([im_4, im_cat, im_noise])


widths = (2**np.arange(4, 10)).astype(int)
heights = (2**np.arange(4, 10)).astype(int)
entropies = np.zeros((len(widths), len(heights)))
area_entropy = []
for i, w in enumerate(widths):
    for j, h in enumerate(heights):
        if abs(i-j) < 200:    
            im = np.random.rand(h, w)
            entropies[i, j] = shannon_entropy_2d(im)
            area = w*h
            area_entropy.append((area, entropies[i, j]))
area_entropy = np.array(area_entropy)   
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
imshow1 = ax.imshow(entropies, cmap='viridis')
plt.colorbar(imshow1, ax=ax)
ax.set_xticks(np.arange(len(widths)))
ax.set_xticklabels(widths)
ax.set_yticks(np.arange(len(heights)))
ax.set_yticklabels(heights)
ax.set_xlabel('Width')
ax.set_ylabel('Height')
ax.set_title('Entropy of random images')

cols = st.columns(2)
cols[0].pyplot(fig)

with cols[1]:
    """
    How does the entropy of an image scale with the size of the image? Lets investigate, and check if its just related to the area or also shape.

    It looks like it just depends on the area of the image, and not the shape.

    Ok, so how does it depend on the area?

    Fitting log(area) to entropy, we get:
    $$
        H \sim \log_2(a\cdot area)
    $$
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.scatter(area_entropy[:,0], area_entropy[:,1])
    
    # fit a line
    func = lambda x, a: np.log2(a*x)
    x = np.linspace(area_entropy[:,0].min(), area_entropy[:,0].max(), 100)
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(func, area_entropy[:,0], area_entropy[:,1])
    y = func(x, *popt)
    a = popt[0]
    a
    f"""
    $$
    H \sim \log_2({a:.3f}\cdot area)
    $$
    """
    ax.plot(x, y, color='red', linestyle='--', )#label=f'log({a}x)+{b}')
    R_squared = 1 - np.sum((area_entropy[:,1] - func(area_entropy[:,0], *popt))**2) / np.sum((area_entropy[:,1] - np.mean(area_entropy[:,1]))**2)
    ax.text(0.5, 0.1, f'$R^2$={R_squared:.6f}', transform=ax.transAxes)
    ax.set_xscale('log')    
    ax.set_xlabel('Area')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy vs Area')
    st.pyplot(fig)

with cols[0]:
    r"""
    We will approximate this as:
    $$
    H \sim \log_2\left(
        \frac{2}{3}\cdot area
        \right)
    $$


    Now with an approximation of the scaling of entropy as measured in this method, we can normalize our entropy with respedct to the area of the image.
    """
    r"""
    $$
        \tilde{H} = H / \log_2\left(
        \frac{2}{3}\cdot area
        \right)
    $$
    
    """
    area  = area_entropy[:,0]
    H = area_entropy[:,1]
    H_tilde = H / np.log2(2/3 * area)


    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.scatter(area_entropy[:,0], H_tilde, alpha=0.5)
    # ax.set_xscale('log')
    ax.set_xlabel('Area')
    ax.set_ylabel('Entropy / $\log(area)$')
    ax.set_title('Entropy normalized by area')
    st.pyplot(fig)


    """
    It looks like there is still an exponential relationship

    """
    exp_func = lambda x, a, b, c, k: a*np.exp(k*(-x-b))+c
    p0 = [
        10, # a
        .1, # b
        .99, # c
        0.000001 # k
    ]
    # popt, pcov = curve_fit(exp_func, area, H_tilde, p0=p0)
    # y = exp_func(x, *popt)
    # y_p0 = exp_func(x, *p0)
    # a, b, c, k= popt
    # popt
    
    # # fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # # ax.scatter(area, H_tilde)
    # # ax.plot(x, y, color='red', linestyle='--', )#label=f'log({a}x)+{b}')
    # # # ax.plot(x, y_p0, color='green', linestyle='--', )#label=f'log({a}x)+{b}')
    # # R_squared = 1 - np.sum((H_tilde - exp_func(area, *popt))**2) / np.sum((H_tilde - np.mean(H_tilde))**2)
    # # ax.text(0.5, 0.1, f'$R^2$={R_squared:.6f}', transform=ax.transAxes)
    # # ax.set_xlabel('Area')
    # # ax.set_ylabel('Entropy / $\log(area)$')
    # # ax.set_title('Entropy normalized by area')
    # # st.pyplot(fig)
    

r"""
We obtain a final approximation of the entropy scaling, and can thus express our image-entropy as:
$$
    \tilde{H}  =
    \frac{-1}{\log_2\left(
        \frac{2}{3}\cdot area
        \right)}
    \cdot
    \sum_{j=1}^J\sum_{i=1}^I p_{i,j} \log_2(p_{i,j}) 
$$
"""

st.divider()

r"""
Applying this to our example image, resized to 512x512, 256x256, and 128x128, with 0, 50, and 100% noise, we get the following results:
"""

im_cat = grayscale(plt.imread('assets/example_images/cat.png'))
st.write('im cat size:', im_cat.shape)
im_cat_512 = resize(im_cat, (512, 512))
im_cat_256 = resize(im_cat, (256, 256))
im_cat_128 = resize(im_cat, (128, 128))
st.write('im cat 512 size:', im_cat_512.shape)
st.write('im cat 256 size:', im_cat_256.shape)
st.write('im cat 128 size:', im_cat_128.shape)

def mix_with_noise(im, noise_fraction):
    min_im = 0#im.min()
    max_im = 1#im.max()
    noisy_im = im * (1-noise_fraction) + np.random.randn(*im.shape) * noise_fraction
    noisy_im = np.clip(noisy_im, min_im, max_im)
    return noisy_im

fig, ax = plt.subplots(3, 3, figsize=(8, 8))
fig.suptitle('Entropy of images with noise')
for i, im in enumerate([im_cat_512, im_cat_256, im_cat_128]):

    for j, noise_fraction in enumerate([0, 0.5, 1]):
        noisy_im = mix_with_noise(im, noise_fraction)
        H = shannon_entropy_2d(noisy_im)
        ax[i,j].imshow(noisy_im, cmap='gray_r')
        ax[i,j].set_title(f'H={H:.2f}')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        if j ==0:
            ax[i,j].set_ylabel(f'{im.shape[0]}x{im.shape[1]}')
plt.tight_layout()
st.pyplot(fig)





