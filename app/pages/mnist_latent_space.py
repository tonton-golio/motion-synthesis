

import streamlit as st


# title and introduction
"""
# MNIST Latent Space

show the following:
* mnist latent space embedded in 2D
* reconstruction from differnt part of latent space
* reconstruction from traversal of latent space
* metrics

"""

# try loading it back
import torch
import matplotlib.pyplot as plt

path = 'logs/MNIST_VAE/version_105/saved_latent/'
device = torch.device('mps')
z = torch.load(path + 'z.pt').to(device)
autoencoder = torch.load(path + 'model.pth').to(device)
# x = torch.load(path + 'x.pt').to(device)
y = torch.load(path + 'y.pt').to(device)
projector = torch.load(path + 'projector.pt')
projection = torch.load(path + 'projection.pt')

z.shape, y.shape
# plot the latent space
fig = plt.figure()
plt.scatter(projection[:, 0], #.detach().cpu().numpy(),
            projection[:, 1], #.detach().cpu().numpy(),
              c=y.detach().cpu().numpy(),
              cmap='tab10',
              alpha=0.5)

plt.colorbar()
st.pyplot(fig)