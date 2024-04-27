import streamlit as st


# PCA explanation

import numpy as np
import matplotlib.pyplot as plt

'# Principal Component Analysis (PCA)'

def PCA_demo(n_samples=100, plot=True):
    # Generate data
    # np.random.seed(1)

    # Generate 2D data
    mu_vec = np.array([0, .5])
    cov_matrix = np.array([[.7, -3], [0, 1]]) 
    x = np.random.multivariate_normal(mu_vec, cov_matrix, n_samples).T

    # Compute covariance matrix and eigen values
    cov_measured = np.cov(x)  # measure covariance
    eig_vals, eig_vecs = np.linalg.eig(cov_measured)  # diagonalize

    # sort eigen values in descending order
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # show eigenvectors
    
    if plot:
        mean_x1 = np.mean(x, axis=1)  # compute mean vector for plotting

        l = length_factor = 6  # length factors, to make plot nicer
        l2 = length_factor /2

        fig = plt.figure(figsize=(6.5, 6.5))
        plt.scatter(x[0], x[1], marker='x', color='red', alpha=0.4, label='samples')

        plt.plot([mean_x1[0]-l*eig_pairs[0][1][0], mean_x1[0] + l * eig_pairs[0][1][0]], 
                [mean_x1[1]-l*eig_pairs[0][1][1], mean_x1[1] + l * eig_pairs[0][1][1]],
                    color='black', linewidth=3, label='Principal Axis 1')  
        plt.plot([mean_x1[0]-l*eig_pairs[1][1][0], mean_x1[0] + l * eig_pairs[1][1][0]],
                    [mean_x1[1]-l2*eig_pairs[1][1][1], mean_x1[1] + l2 * eig_pairs[1][1][1]],
                        color='darkgrey', linewidth=3, label='Principal Axis 2')  

        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.legend(loc='lower left')

        # plt.savefig('../assets/PCA_demonstration.png')
        return fig
    
    return eig_pairs


cols = st.columns((2, 2))
fig = PCA_demo(n_samples=420)
cols[0].write("""
              PCA is a technique for linear dimensionality reduction. It can be explained in two ways; 1. more intuitive, 2. more mathematical.

              1. *Intuitive explanation*: We place the first principal axis, $\\text{A}_1$, along the direction in our data cloud, which contains the greatest variance (see the black line on the plot). Then in the space orthogonal to $\\text{A}_1$, we repeat the process, see the grey line.

              2. *Mathematical explanation*: Principal axes are the eigenvectors of the covariance matrix of the data. The eigenvalues of the covariance matrix represent the variance along the principal axes.
            
              We define some data $X in \mathbb{R}^{n \times m}$, where $n$ is the number of samples and $m$ is the number of features. We compute the covariance between features
                $$
              \\text{cov}(X) = \mathbb{E}[(X - \mathbb{E}[X])(X - \mathbb{E}[X])^T].
              $$
              The eigenvectors of $C$ are the principal axes, and the eigenvalues are the variance along these axes.

              """)

cols[1].pyplot(fig)