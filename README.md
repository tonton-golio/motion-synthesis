
# Rapid Motion Synthesis (RaMoS)

A pipeline for training a motion synthesis AI. Application of reverse diffusion on a latent embedding, for diverse outputs with minimal compute requirements.

## Contents

### MNIST latent diffusion pipeline

The repo contains a latent diffusion pipeline implemented for the MNIST dataset. The pipeline is implemented in PyTorch, using PyTorch Lightning for training and tensorboard for logging. The pipeline is implemented in a modular fashion, with the following components:

- A variational autoencoder (VAE) for learning a latent embedding of the input data
- A reverse diffusion model for generating diverse outputs from noisy latent embeddings

*Implementing the pipeline for the simple, clean and well-known dataset, let me become comfortable with the; diffusion model, PyTorch Lightning and the general pipeline structuring*

### Motion data preprocessing and analysis

The AMASS dataset is processed using the HUMAN3DML library, and then trimmed to exlude sequences including high velocity joints. Additionally, the data is trimmed to exclude uncommon motions like ``tapdance`` and ``backflip``. The data is then analyzed to find the most common motions, and the distribution of motion lengths.


### Pose latent embedding

We implement a variational autoencoder (VAE) for single pose frames. With this we explore the efficacy of using a graph neural network for pose embedding. The reason for using a graph based neural network, is to inform the network about the connection between joints, thereby enabling sensible convolutions on the pose data.

### Motion latent embedding

We implement a variational autoencoder (VAE) for motion sequences. The VAE is trained on the AMASS dataset, and the latent space is analyzed to find the most common motions, and the distribution of motion lengths. We start with 200 frames of each 22 joints expressed with 3D coordinates, and perform dimensionality reduction from 13200 to 256 dimensions.

### Motion latent diffusion

not implemented yet