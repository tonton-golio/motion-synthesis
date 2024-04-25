import streamlit as st


# PCA explanation

import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(1)

# Generate 2D data
mu_vec1 = np.array([0, .5])
cov_mat1 = np.array([[.7, -3], [0, 1]])
x1_samples = np.random.multivariate_normal(mu_vec1, cov_mat1, 1000)

# plot
fig = plt.figure(figsize=(5.5, 5.5))
plt.scatter(x1_samples[:, 0], x1_samples[:, 1], marker='x', color='red', alpha=0.4, label='samples')

# look for dimension of most variance
# compute mean vector
mean_x1 = np.mean(x1_samples, axis=0)

# compute covariance matrix
cov_mat1 = np.cov(x1_samples.T)

# compute eigen values and eigen vectors
eig_vals, eig_vecs = np.linalg.eig(cov_mat1)

# sort eigen values in descending order
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# show eigenvectors
l = length_factor = 6
l2 = length_factor /2
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
st.pyplot(fig)