import streamlit as st
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.append('../')
from mnist_latent_diffusion.modules.dataModules import MNISTDataModule
from matplotlib import gridspec


from app.utils_app import load_or_save_fig

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
    'Introduction',
    'Measuring noise-level',
    'Noise Schedule',
    'Metrics',
    'Time Embedding',
    #'CLIP',
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

@load_or_save_fig('assets_produced/3_diffusion_theory/vector_entropy.png')
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


with tabs['Introduction']:
    from subpages.diffusion_intro import diffusion_intro
    diffusion_intro()


with tabs['Measuring noise-level']:

    from subpages.measure_noise_level import measure_noise_level_page
    measure_noise_level_page()


# Noise Schedule
with tabs['Noise Schedule']:
    import torch
    import torch.nn as nn
    import math
    from matplotlib import gridspec

    from subpages.diffusion_intro import plot_variance_schedule_image_series, plot_variance_schedule_hists, prep_image

    from app.utils_app import VarianceSchedule

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


    img_path = 'assets/example_images/cat.png'

    # fig_imgs, fig_hist = plot_variance_schedule(img_path, vs)
    img = prep_image(img_path)
    vs = VarianceSchedule(6, epsilon=0.08, method='cosine')
    fig_imgs = plot_variance_schedule_image_series(img, vs, kl=True, noise_type='normal')
    st.pyplot(fig_imgs)
    
    """
    Above the images along the noise schedule, I've noted the KL divergence between the noised image and the noise distribution. This is a measure of how much the noised image differs from the original image.
    """

    
    if False:
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

    if False:
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

    if False:
        plot_noising_schedule(ims[:1], VS)

    

    if False:
        shannon_entropy_2d(ims[0], plot=True)

    '''
    Now the metric works, and we want to see how the Shannon entropy changes as we add noise to the image. We plot the Shannon entropy for each image in the schedule.
    '''
    # check if image is pure noise
    # if pure noise: the shannon entropy should be very high
    # we use 2d finite difference to obtain the derivative as per: 

    if False:
        plot_noising_schedule2(ims[:5], VS)

    


    if False:
        pure_noise = torch.randn_like(ims[0])
        shannon_entropy_2d(pure_noise, plot=True, title1='pure noise')

    if False:
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


with tabs['Noise Schedule']:
    from subpages.noise_schedule import NoiseScheduleDemo
    fig1, fig2, fig3 = NoiseScheduleDemo()
    st.pyplot(fig1)
    st.pyplot(fig2)
    # st.pyplot(fig4)
    st.pyplot(fig3)


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
    from subpages.time_embedding import time_embedding_page
    time_embedding_page()



