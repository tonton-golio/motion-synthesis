import streamlit as st

# title and introduction

"""
#Unet Architecture

What is unet: Unet is neural network based around the convolution and pooling operations in a encoder-decoder architecture \cite{https://arxiv.org/pdf/1505.04597.pdf}. Through the use of skip connections, the model is able to retain exact spatial information, while also being able to learn high level features. In the original purpose of the model, they used it to segment electron microscopy images of neuronal structures. The iterative downsampling, allow the model to identify a region of interest, while the skipped connections allow the model to understand exactly where the identified region is located.

"""
st.image('assets/unet.png', caption='Unet architecture', use_column_width=True)



"""
Despite originally being proposed for biomedical image segmentation, the model has seen huge success in a variety of fields. [imagen https://arxiv.org/abs/2205.11487, dalle2 https://arxiv.org/abs/2204.06125, stable-diffusion https://arxiv.org/abs/2112.10752].


## Specifics of the model
The model takes an image's input. And produces an image as output. The input is passed through two convolutional layers. Which does not Decrease the image size other than. Other than cutting the edges. The images then pass through a max pooling layer, which downsamples along both dimensions by a factor 2. These two steps, along with an activation function, make up a block in the unit architecture. A block passes a copy of its output Across to the symmetrical brother. Was passing its output to yet another block In the encoder. The symmetric paths across the structure is the. skip connection. In the original paper They perform five of these blocks. But that's up for change Depending on the image sizes you're working with. 


## Why use Unet
We use our unit architecture. Because we'll be working with the Mnist data set, which are images there just makes sense But also to let ourselves inspire when we move on to do the Diffusion and encoding of motion. Although we won't be using convolutional layers, the skip connections may still be relevant As per (https://arxiv.org/abs/2212.04048)


## How to add time and target to the unet structure
We call extra inputs like time and target *Conditioning*. We concat these at every block of the model as is done in (https://arxiv.org/pdf/2112.10752.pdf)
"""