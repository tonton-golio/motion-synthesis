import streamlit as st
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.append('../')
from mnist_latent_diffusion.modules.dataModules import MNISTDataModule
from matplotlib import gridspec

st.set_page_config(
    page_title="Diffusion Theory",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Intro and title
"""
# Diffusion Theory


"""
tab_names = [
    'Vector/image entropy',
    'Noise Schedule',
    'Metrics',
    'Time Embedding',
]
tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}

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

def vector_entropy_demo(vector_length = 8):
    vl = vector_length 
    A = normalize(np.random.rand(vl))
    B = normalize(np.random.rand(vl)**20)
    # max(A), max(B)
    def make_plot():

        fig, ax = plt.subplots(1, 2, figsize=(12, 1), gridspec_kw={'width_ratios': [1, 1.2]})
        vmin = 0
        ax[0].imshow(A.reshape(1, -1), aspect='auto', vmin=vmin, vmax=1, cmap='Greys')
        rightplot = ax[1].imshow(B.reshape(1, -1), aspect='auto', vmin=vmin, vmax=1, cmap='Greys')
        plt.colorbar(rightplot, ax=ax[1])

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
    
    r"""
    ### Vector Entropy
    Given two vectors of unit length (manhattan norm), we can calculate the entropy of the distribution of values in the vector.

    A vector, $\bar{a}$, with values distributed uniformly, has maximum entropy, while a vector, $\bar{b}$, with all values concentrated in a single component has minimum entropy.
    """
    
    fig = make_plot()
    st.pyplot(fig)

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

with tabs['Vector/image entropy']:
    vector_entropy_demo(10)
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
    



# Noise Schedule
with tabs['Noise Schedule']:
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

        # H tilde
        H_tilde = H / np.log2(2/3 * im_arr.size)
        H = H_tilde
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
    ### Noise schedule
    """
    dm = MNISTDataModule(verbose=False, **params_data)
    dm.setup()
    VS = VarianceSchedule(**params_schedule)
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
    ims = [dm.data_train[i][0].squeeze() for i in range(5)]
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

    example_img = plt.imread("assets/example_images/cat.png")
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


# Metrics
with tabs['Metrics']:

    import sys
    # sys.path.append('../')
    import streamlit as st
    from mnist_latent_diffusion.modules.dataModules import MNISTDataModule
    import numpy as np
    import matplotlib.pyplot as plt
    from torcheval.metrics import FrechetInceptionDistance as FID
    import torch
    from torch import Tensor

    import seaborn as sns
    import pandas as pd
    import umap
    from sklearn.preprocessing import StandardScaler

    ## Diffusion
    def matrix_norm(A, B):
        return np.mean([np.linalg.norm(a-b) for a, b in zip(A, B)])

    def get_div(data, labels, method='all'):
        """
        method can be either 'all' or 'class'
        """
        if method == 'all':
            ims1, ims2 = data[:len(data)//2], data[len(data)//2:]
            return matrix_norm(ims1, ims2)
        
        elif method == 'class':
            classes = np.unique(labels)
            res = {}
            for class_ in classes:
                ims = [data[i] for i in range(len(data)) if labels[i] == class_]
                ims1, ims2 = ims[:len(ims)//2], ims[len(ims)//2:]
                res[int(class_)] = matrix_norm(ims1, ims2)
            return res
        
        else:
            raise ValueError('method must be either "all" or "class"')

    def _calculate_frechet_distance(
            mu1: Tensor,
            sigma1: Tensor,
            mu2: Tensor,
            sigma2: Tensor,
        ) -> Tensor:
            """
            Calculate the Frechet Distance between two multivariate Gaussian distributions.

            Args:
                mu1 (Tensor): The mean of the first distribution.
                sigma1 (Tensor): The covariance matrix of the first distribution.
                mu2 (Tensor): The mean of the second distribution.
                sigma2 (Tensor): The covariance matrix of the second distribution.

            Returns:
                tensor: The Frechet Distance between the two distributions.
            """

            # Compute the squared distance between the means
            mean_diff = mu1 - mu2
            mean_diff_squared = mean_diff.square().sum(dim=-1)

            # Calculate the sum of the traces of both covariance matrices
            trace_sum = sigma1.trace() + sigma2.trace()

            # Compute the eigenvalues of the matrix product of the real and fake covariance matrices
            sigma_mm = torch.matmul(sigma1, sigma2)
            eigenvals = torch.linalg.eigvals(sigma_mm)

            # Take the square root of each eigenvalue and take its sum
            sqrt_eigenvals_sum = eigenvals.sqrt().real.sum(dim=-1)

            # Calculate the FID using the squared distance between the means,
            # the sum of the traces of the covariance matrices, and the sum of the square roots of the eigenvalues
            fid = mean_diff_squared + trace_sum - 2 * sqrt_eigenvals_sum

            return fid

    def _calculate_FID_SCORE(images_real, images_synthetic):
        images_real = images_real.view(images_real.size(0), -1).T
        images_synthetic = images_synthetic.view(images_synthetic.size(0), -1).T
        images_real.shape, images_synthetic.shape
        mu_real = images_real.mean(dim=1)
        sigma_real = torch.cov(images_real)

        mu_synthetic = images_synthetic.mean(dim=1)
        sigma_synthetic = torch.cov(images_synthetic)
        return _calculate_frechet_distance(mu_real, sigma_real, mu_synthetic, sigma_synthetic)


    # title and introduction
    """
    # Metrics

    Here we present the metrics we will use for evaluating our VAE and Diffusion models. These are typically used in the literature, and thus serve as a common point of reference!

    ---
    """

    ###### SECTION 1: Diffusion metrics ######
    """
    ## Diffusion metrics
    """
    tabs_diffusion = st.tabs(["Diversity (DIV)", "MultiModality (MM)", "Frechet Inception Distance (FID)"])


    with tabs_diffusion[0]: # DIV
        # title and introduction
        """
        Source: https://arxiv.org/pdf/2212.04048.pdf

        The diversity metric (DIV) is a measure of similarity between two subsets of our data $X$ and $X'$. DIV is calculated by taking the average pairwise Frobenius norm between feature matricies.
        """
        st.latex(r'''
        \begin{equation}
        DIV = \frac{1}{N} \sum_{i=1}^{N} \left|\left| X_i - X'_i \right|\right|_2
        \end{equation}
        ''')

        """
        So what we're going to do now, to test if the metric is sensible; we will measure the diversity for our total 10,000 samples spanning across classes, as well as for individual classes. Naturally; we expect the diversity whithin a class to be lower than the diversity across classes. 

        """

        dm = MNISTDataModule(verbose=False,ROTATION=0)
        dm.setup()

        N = 800

        ims = [dm.data_train[i][0].squeeze() for i in range(N)]
        labs = [dm.data_train[i][1] for i in range(N)]

        # calc DIV for all
        div_all = get_div(ims, labs, method='all')
        div_class = get_div(ims, labs, method='class')

        fig, ax = plt.subplots(1, 1, figsize=(7, 3))
        ax.bar(div_class.keys(), div_class.values(), label='DIV per class', alpha=0.7)
        ax.set_xticks(list(div_class.keys()))
        ax.axhline(div_all, color='red', linestyle='--', label='DIV all samples')
        ax.set_xlabel('Class label')
        ax.set_ylabel('Diversity')  
        ax.set_ylim(min(div_class.values())-2, max(div_class.values())+2)
        ax.set_title('Diversity across classes')
        ax.legend(loc='lower right')
        st.pyplot(fig)

    with tabs_diffusion[1]: # MM
        """
        Multimodality is a very similar metric to that of DIV, except we calculate it per class, and extract the average, i.e., we calculate the average pairwise Frobenius norm between feature matricies for each class, and then take the average across all classes.
        """
        st.latex(r'''
        \begin{equation}
            MM = \frac{1}{J_m\times X_m} \sum_{j=1}^{J_m} \sum_{i=1}^{X_m}  \left|\left| X_{j,i} - X'_{j,i} \right|\right|_2
        \end{equation}
        ''')

        """
        where $J_m$ is the number of classes, and $X_m$ is the number of samples within each class. 
        """

        MM = np.mean(list(get_div(ims, labs, method='class').values()))
        MM


    with tabs_diffusion[2]: # FID
        # title and introduction
        """
        source: https://arxiv.org/pdf/1706.08500.pdf

        The Frechet Inception Distance (FID) is a measure of similarity between two sets of images. It is based on the Frechet Distance, which is a measure of similarity between two multivariate Gaussian distributions. We measure how unlike generated images are to real images.

        If we are working with higher dimensional data, we can use the exact same technique.

        **Can we backpropagate through the FID?**

        No, because the FID is a measure of similarity between two sets of images. It is not a differentiable function.
        """

        cat = plt.imread('assets/example_images/cat.png')[:,:,:3]
        cat_noisy = cat + np.random.rand(*cat.shape)*0.1
        cat_NOISY = cat + np.random.rand(*cat.shape)*1


        fig, ax = plt.subplots(1, 3, figsize=(12, 5))

        ax[0].imshow(cat);       ax[0].set_title('Cat')
        ax[1].imshow(cat_noisy); 
        ax[2].imshow(cat_NOISY); 

        _ = [axi.axis('off') for axi in ax]



        # FID
        cat_t = torch.tensor(cat, dtype=torch.float32).permute(2, 0, 1)
        cat_noisy_t = torch.tensor(cat_noisy, dtype=torch.float32).permute(2, 0, 1)
        cat_NOISY_t = torch.tensor(cat_NOISY, dtype=torch.float32).permute(2, 0, 1)

        cat_t = torch.clip(cat_t, 0.00001, .9999)
        cat_noisy_t = torch.clip(cat_noisy_t, 0.000001, .9999)
        cat_NOISY_t = torch.clip(cat_NOISY_t, 0.000001, .9999)

        cat_t = torch.stack([cat_t, cat_t])
        cat_noisy_t = torch.stack([cat_noisy_t, cat_noisy_t])
        cat_NOISY_t = torch.stack([cat_NOISY_t, cat_NOISY_t])

        fid = FID()
        fid.update(cat_t, is_real=True)
        fid.update(cat_noisy_t, is_real=False)
        fid_score = fid.compute()

        ax[1].set_title('Noisy cat, FID: {:.2f}'.format(fid_score))


        fid = FID()
        fid.update(cat_t, is_real=True)
        fid.update(cat_NOISY_t, is_real=False)
        fid_score = fid.compute()
        ax[2].set_title('NOISY cat, FID: {:.2f}'.format(fid_score))
        st.pyplot(fig)

        '---'

        """
        ## Math behind FID

        The Frechet Distance is a measure of similarity between two multivariate Gaussian distributions. 

        We take our images, and flatten them into vectors. We then look at each each index in our vector, and determine the mean value corresponding to that index. We then calculate the covariance matrix for pixel values across all images.

        We take the difference of the means... (note, when we say mean, we mean for each pixel in the image) and square it, and sum the values. This gives us the squared distance between the means.

        We add the sum of the trace sum of the covariance matrices. And we subtract twice the sum of the square roots of the eigenvalues of the matrix product of the covariance matrices.

        In math notation, because this is getting too convoluted: We have defined the vectors and matrices: $\mu_1, \sigma_1, \mu_2,$ and $\sigma_2$. (notice mu are vectors, sigma are matrices)
        """

        st.latex(r"""
        \begin{align*}
            \delta_{\mu} = \mu_1 - \mu_2\\
            \Sigma_{mm} = \sigma_1 \sigma_2\\
            \lambda_i = \text{eigenvalues}(\Sigma_{mm})\\
            F = \delta_{\mu}^2 + \text{Tr}(\sigma_1 + \sigma_2) - 2 \sum_{i=1}^n \sqrt{\lambda_i}
        \end{align*}
        """)



        # example usage
        images_real = torch.rand(100, 32, 32)
        images_synthetic = torch.rand(10, 32, 32)

        fid_score = _calculate_FID_SCORE(images_real, images_synthetic)
        fid_score

# Time Embedding
with tabs['Time Embedding']:
    import streamlit as st
    import matplotlib.pyplot as plt
    import umap
    import numpy as np

    def sinusoidal_time_embedding(time_indices, dimension):
        """
        Generates sinusoidal embeddings for a sequence of time indices.
        
        Args:
        time_indices (np.array): A 1D numpy array of time indices.
        dimension (int): The dimensionality of the embeddings.
        
        Returns:
        np.array: A 2D numpy array where each row represents the sinusoidal embedding of a time index.
        """
        assert dimension % 2 == 0, "Dimension must be even."
        
        # Initialize the positions matrix
        position = time_indices[:, np.newaxis]  # Convert to a column vector
        
        # Compute the divisors for each dimension
        div_term = np.exp(np.arange(0, dimension, 2) * -(np.log(10000.0) / dimension))
        
        # Calculate sinusoidal embeddings
        embeddings = np.zeros((len(time_indices), dimension))
        embeddings[:, 0::2] = np.sin(position * div_term)  # Apply sine to even indices
        embeddings[:, 1::2] = np.cos(position * div_term)  # Apply cosine to odd indices
        
        return embeddings

    # Example usage
    time_indices = np.arange(0, 512)
    dimension = 32
    embeddings = sinusoidal_time_embedding(time_indices, dimension)
    umap_embeddings = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='cosine').fit_transform(embeddings)



    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    fig.suptitle('Sinusoidal Time Embeddings')

    imsh = ax[0].imshow(embeddings, cmap='RdYlBu', aspect='auto')

    plt.colorbar(imsh, ax=ax[0], label='Embedding Value')
    ax[0].set_xlabel('Embedding Dimension')
    ax[0].set_ylabel('Time Index')
    ax[0].set_title('Sinusoidal Embeddings')

    scat = ax[1].scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=time_indices, cmap='RdYlBu')
    ax[1].set_title('UMAP Projection')
    plt.colorbar(scat, ax=ax[1], label='Time Index')

    plt.tight_layout()
    # plt.savefig('assets/sinusoidal_time_embeddings.png')


    # title and introduction
    """
    # Time embedding
    """

    st.pyplot(fig)


    """
    embeddings

    # make time embedding mlp
    import torch
    import torch.nn as nn
    class TimeEmbeddingMLP(nn.Module):
        def __init__(self, timesteps, hidden_dim, output_dim, num_layers, dropout):
            super(TimeEmbeddingMLP, self).__init__()
            self.embedding = nn.Embedding(timesteps, hidden_dim)
            self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
            self.output_layer = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x)
                x = torch.relu(x)
                x = self.dropout(x)
            x = self.output_layer(x)
            return x
        
    # Example usage

    model = TimeEmbeddingMLP(timesteps=512, hidden_dim=32, output_dim=32, num_layers=2, dropout=0.1)
    t = torch.tensor([0, 1]).long()
    output = model(t)

    # train to match sinusoidal embeddings
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    y = torch.tensor(embeddings).float()
    x = torch.arange(0, 512).long()

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TimeEmbeddingMLP(timesteps=512, hidden_dim=64, output_dim=32, num_layers=3, dropout=0.1)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1000):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    """


