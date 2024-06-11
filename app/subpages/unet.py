import streamlit as st

# title and introduction
# st.set_page_config(layout="wide")
def render():
    """
    ### Unet Architecture
    """

    cols = st.columns((1, 2))
        
    with cols[0]:
        st.write(
        """
        Unet is neural network based around the convolution and pooling operations in a encoder-decoder architecture, [1: unet paper](https://arxiv.org/pdf/1505.04597.pdf). The compressive nature of the network, does well to extract large scale structures in the data. A drawback of a typical encoder-decoder architecture however, is the loss of spatial resolution. Unet mitigates this via use of skip connections between layers in the encoder and decoder respectively which are of similar size, letting the model retain exact spatial information, while also being able to learn high level features. 
        
        In the original purpose of the model, they used it to segment electron microscopy images of neuronal structures. The iterative downsampling, allow the model to identify a region of interest, while the skipped connections allow the model to understand exactly where the identified region is located.

        Despite originally being proposed for biomedical image segmentation, the model has seen huge success in image generation tasks. Some examples include:
        [2: imagen](https://arxiv.org/abs/2205.11487),
        [3: dalle2](https://arxiv.org/abs/2204.06125),
        [4: stable-diffusion](https://arxiv.org/abs/2112.10752)
        """)
    with cols[1]:
        # st.image('assets/1_NN_fundamentals_and_architectures/unet.png', caption='Unet architecture', use_column_width=True)
        im_ = 'https://www.researchgate.net/profile/Lu-Xu-45/publication/336327037/figure/fig1/AS:832209418215424@1575425595547/The-schematic-diagram-of-U-Net-structure.png'
        st.image(im_, caption='Unet architecture', use_column_width=True)
    cols = st.columns((2, 1,1))

    with cols[0]:
        
        st.write("""
        #### Specifics of the model
        The model takes an image's input. And produces an image as output. The input is passed through two convolutional layers. Which does not Decrease the image size other than. Other than cutting the edges. The images then pass through a max pooling layer, which downsamples along both dimensions by a factor 2. These two steps, along with an activation function, make up a block in the unit architecture. A block passes a copy of its output Across to the symmetrical brother. Was passing its output to yet another block In the encoder. The symmetric paths across the structure is the. skip connection. In the original paper They perform five of these blocks. But that's up for change Depending on the image sizes you're working with. 
        """)
    with cols[1]:
        st.write("""
        #### Why use Unet
        We use our unit architecture. Because we'll be working with the Mnist data set, which are images there just makes sense But also to let ourselves inspire when we move on to do the Diffusion and encoding of motion. Although we won't be using convolutional layers, the skip connections may still be relevant As per (https://arxiv.org/abs/2212.04048)
        """)
        
    with cols[2]:   
        st.write("""
        #### How to add time and target to the unet structure
        We call extra inputs like time and target *Conditioning*. We concat these at every block of the model as is done in (https://arxiv.org/pdf/2112.10752.pdf)
        """)
