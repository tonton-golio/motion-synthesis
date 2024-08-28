import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from sklearn.metrics import pairwise_distances
from utils_app import load_or_save_fig
# Title and intro
"""
# Autoencoding Theory

"""

tab_names = [
    'General',
        # under general, discuss the central method
        # talk about intrinsic dimensionality
        # show a schema of an autoencoder
        # discuss the latent space
        # reparametrization trick
        
    'Reconstruction Error (RE)',
    'KL Divergence (KL)',
    'Inverse spatial entropy (ISE)'
]
tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}

# General
with tabs['General']:
    cols = st.columns(2)
    with cols[0]:
        st.write("""
        Auto-encoders, as the name suggests are encoding networks which associate the output with the input. As such, autoencoders are an approximation of the identity function for a population of data.

        Seeing as these networks are encoding a sample, we may consider them dimensionality reduction algorithms. The latent space is a compressed representation of the input data.     
        """)
    with cols[1]:
        st.image('app/assets/5_Autoencoding_theory/AutoEncoderSchema.png', caption='Autoencoder schema', use_column_width=True)

# Reconstruction Error
with tabs['Reconstruction Error (RE)']:
    st.write(r"""
    The primary measure for how well our network is able to encode and decode a sample, is the reconstruction error. The reconstruction error is the difference between the input and the output of the network. 
    
    We may measure reconstruction error in a variety of ways, such as a distanced based error, a likelihood based error, or a perceptual error.

    ### Distance based error
    Distance based error is the most common form of reconstruction error. The most common distance metric is the mean squared error (MSE). The MSE is the average of the squared differences between the input and the output.
    $$
    \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2
    $$
    where $x_i$ is the input and $\hat{x}_i$ is the output of the network.

    ### Likelihood based error
    Likelihood based error is the negative log likelihood of the output given the input. This is the error used in variational autoencoders (VAEs). The likelihood based error is the average of the log likelihood of the output given the input.
    $$
    \text{NLL} = -\frac{1}{n} \sum_{i=1}^{n} \log p(x_i | \hat{x}_i)
    $$
    where $p(x_i | \hat{x}_i)$ is the likelihood of the output given the input.

    ### Perceptual error
    Perceptual error is the error of the output given the input as perceived. We may employ an adversarial network to determine the perceptual error.

    """)

# KL Divergence
with tabs['KL Divergence (KL)']:
    # here we explore how the kl divergence loss works for autoencoders
    

    def kl_divergence(p, q):
        return np.sum(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))


    def kl_divergence_loss(mu, log_var):
        return -0.5 * np.sum(1 + log_var - np.square(mu) - np.exp(log_var), axis=-1)

    @load_or_save_fig('app/assets_produced/5_Autoencoding_theory/kl_divergence_contour.png')
    def make_contour_plot(low_alpha=-np.pi, high_alpha=np.pi, low_beta=-np.pi, high_beta=np.pi):
        n, n_ = 100, 10

        KL = np.zeros((n, n))
        log_var_scale = np.linspace(-2,2, n)
        mu_scale = np.linspace(-1, 1, n)
        for i, v in enumerate(log_var_scale):
            for j, m in enumerate(mu_scale):
                mu = np.random.rand(n)# * alpha
                log_var = np.random.rand(n)# * beta
                
                KL[i, j] = kl_divergence_loss(m*mu, v*log_var)
        
        
        n_beta, n_alpha = KL.shape
        

        fig, ax = plt.subplots()
        plt.contourf(KL, cmap='inferno', levels=20)
        plt.colorbar()
        plt.xlabel('mu scale')
        plt.ylabel('log_var scale')
        plt.xticks([0, n_beta - 1], [low_beta, high_beta])
        plt.yticks([0, n_alpha - 1], [low_alpha, high_alpha])
        plt.xticks([0, (n_beta - 1)/2, n_beta - 1], ['-1', 0, '1'])
        plt.yticks([0, n_alpha - 1], [0, '$2$'])
        # equal aspect ratio
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        return fig

    @load_or_save_fig('app/assets_produced/5_Autoencoding_theory/kl_divergence_dist.png')
    def dist_plot(n):
        mu = np.random.rand(n)# * alpha
        log_var = np.random.rand(n)# * beta
        dist = np.random.normal(mu, np.exp(log_var / 2))
        rand = np.random.rand(n)
        fig = plt.figure()
        shared_bins = np.linspace(-2, 2, 30)
        counts_dist, _ = np.histogram(dist, bins=shared_bins)
        counts_rand, _ = np.histogram(rand, bins=shared_bins)

        plt.bar(shared_bins[:-1], counts_dist, width=shared_bins[1] - shared_bins[0], alpha=0.5, label='dist')
        plt.bar(shared_bins[:-1], counts_rand, width=shared_bins[1] - shared_bins[0], alpha=0.5, label='rand')

        kl_divergence(counts_dist / n, counts_rand / n,  )
        plt.legend()
        return fig


    '''
    KL divergence is a measure of how one probability distribution differs from a second probability distribution. In the context of autoencoders, the KL divergence loss force the latentspace to consists of near univariate gaussians. 

    The term punishes symertrically with respect to the mean. The term punishes more for larger variances.
    '''
    fig1 = make_contour_plot()
    fig2 = dist_plot(n = 1000)
    
    cols = st.columns(2)
    with cols[0]:
        st.write('KL divergence loss as a function of mu and log_var')
        st.pyplot(fig1)
    with cols[1]:
        st.write('???')
        st.pyplot(fig2)


# Metrics
with tabs['Inverse spatial entropy (ISE)']:

    ## VAE
    def get_space_fullness(X, N=10, n_runs=3):
        def single_run(X, N, low=0, high=1):
            x = np.random.uniform(low, high, (N, X.shape[1]))
            # distance to nearest
            
            dist = pairwise_distances(X, x, metric='manhattan').min(axis=0)
            # plt.figure()
            # plt.hist(dist, bins=50)
            # mean, median = np.mean(dist), np.median(dist)
            # plt.axvline(mean, label='mean', c='r')
            # plt.axvline(median, label='median', ls='--', c='r')
            # plt.legend()
            # plt.show()
            return np.median(dist)
        
        # scale the data: now everything is 0-1
        X = X.copy()
        X -= X.min(axis=0)
        X /= X.max(axis=0)

        vals = []
        for i in range(n_runs):
            vals.append(single_run(X, N))

        return (np.mean(vals), np.std(vals)), X
    
    @load_or_save_fig('app/assets_produced/5_Autoencoding_theory/umap_space_inverse_entropy.png')
    def make_space_full_plot(n_sample_points=1000):
        N = n_sample_points  # number of sampling points


        # data
        data = sns.load_dataset('iris').drop('species', axis=1)
        X1 = np.random.rand(data.shape[0], data.shape[1])

        #umap
        reducer = umap.UMAP()
        X = reducer.fit_transform(data)
        X1 = reducer.fit_transform(X1)

        space_fullness, X = get_space_fullness(X, N, n_runs=3)
        space_fullness1, X1 = get_space_fullness(X1, N, n_runs=3)

        # print(f'space fullness: {space_fullness[0]:.2f} ± {space_fullness[1]:.2f}')
        # print(f'space fullness1: {space_fullness1[0]:.2f} ± {space_fullness1[1]:.2f}')


        #plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        ax[0].scatter(X[:, 0], X[:, 1], c='black', s=10, marker='x')
        ax[0].set_title(r'iris, inverse entropy: $S^{-1}$' +'= {:.2f} ± {:.2f}'.format(space_fullness[0], space_fullness[1]))

        ax[1].scatter(X1[:, 0], X1[:, 1], s=10, c='black', marker='x')
        ax[1].set_title(r'random , inverse entropy: $S^{-1}$' +'={:.2f} ± {:.2f}'.format(space_fullness1[0], space_fullness1[1]))

        # plt.savefig('assets/umap_space_inverse_entropy.png')
        return fig
    
    fig = make_space_full_plot()
    st.pyplot(fig)  
