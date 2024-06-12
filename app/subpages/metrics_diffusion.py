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


def metrics_diffusion_page():
    # title and introduction
    st.write("""
    Here we present the metrics we will use for evaluating our Diffusion models. These are typically used in the literature, and thus serve as a common point of reference!
    """)
    st.divider()

    # Diversity
    st.write("""
    #### Diversity (DIV)
    The diversity metric (DIV) is a measure of similarity between two subsets of our data $X$ and $X'$. DIV is calculated by taking the average pairwise Frobenius norm between feature matricies. Source: https://arxiv.org/pdf/2212.04048.pdf
    """)
    cols = st.columns(2)
    with cols[0]:
        st.latex(r'''
        \begin{equation}
        DIV = \frac{1}{N} \sum_{i=1}^{N} \left|\left| X_i - X'_i \right|\right|_2
        \end{equation}
        ''')
        st.write("""
        So what we're going to do now, to test if the metric is sensible; we will measure the diversity for our total 10,000 samples spanning across classes, as well as for individual classes. Naturally; we expect the diversity whithin a class to be lower than the diversity across classes. 

        """)
    with cols[1]:
        dm = MNISTDataModule(verbose=False,ROTATION=0)
        dm.setup()

        N = 800

        ims = [dm.data_train[i][0].squeeze() for i in range(N)]
        labs = [dm.data_train[i][1] for i in range(N)]

        # calc DIV for all
        div_all = get_div(ims, labs, method='all')
        div_class = get_div(ims, labs, method='class')

        fig, ax = plt.subplots(1, 1, figsize=(7, 3))
        ax.bar(div_class.keys(), div_class.values(), 
               label='DIV per class', alpha=0.7, color='purple')
        ax.set_xticks(list(div_class.keys()))
        ax.axhline(div_all, color='red', linestyle='--', label='DIV all samples')
        ax.set_xlabel('Class label')
        ax.set_ylabel('Diversity')  
        ax.set_ylim(min(div_class.values())-2, max(div_class.values())+2)
        ax.set_title('Diversity across classes')
        ax.legend(loc='lower right')
        st.pyplot(fig)


    st.divider()
    st.write("""
    #### MultiModality (MM)
    Multimodality is a very similar metric to that of DIV, except we calculate it per class, and extract the average, i.e., we calculate the average pairwise Frobenius norm between feature matricies for each class, and then take the average across all classes.
    """)
    st.latex(r'''
    \begin{equation}
        MM = \frac{1}{J_m\times X_m} \sum_{j=1}^{J_m} \sum_{i=1}^{X_m}  \left|\left| X_{j,i} - X'_{j,i} \right|\right|_2
    \end{equation}
    ''')

    st.write("""
    where $J_m$ is the number of classes, and $X_m$ is the number of samples within each class. 
    """)

    MM = np.mean(list(get_div(ims, labs, method='class').values()))
    st.code(MM)

    st.divider()
    st.write("""
    #### Frechet Inception Distance (FID)
    
    source: https://arxiv.org/pdf/1706.08500.pdf

    The Frechet Inception Distance (FID) is a measure of similarity between two sets of images. It is based on the Frechet Distance, which is a measure of similarity between two multivariate Gaussian distributions. We measure how unlike generated images are to real images.

    If we are working with higher dimensional data, we can use the exact same technique.

    **Can we backpropagate through the FID?**

    No, because the FID is a measure of similarity between two sets of images. It is not a differentiable function.
    """)

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

   

    st.write("""
    ## Math behind FID

    The Frechet Distance is a measure of similarity between two multivariate Gaussian distributions. 

    We take our images, and flatten them into vectors. We then look at each each index in our vector, and determine the mean value corresponding to that index. We then calculate the covariance matrix for pixel values across all images.

    We take the difference of the means... (note, when we say mean, we mean for each pixel in the image) and square it, and sum the values. This gives us the squared distance between the means.

    We add the sum of the trace sum of the covariance matrices. And we subtract twice the sum of the square roots of the eigenvalues of the matrix product of the covariance matrices.

    In math notation, because this is getting too convoluted: We have defined the vectors and matrices: $\mu_1, \sigma_1, \mu_2,$ and $\sigma_2$. (notice mu are vectors, sigma are matrices)
    """)

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
    st.code(fid_score)