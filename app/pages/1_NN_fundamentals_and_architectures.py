import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from utils import load_or_save_fig

# Title and intro
"""
# Neural Network Fundamentals & Architectures

"""

tab_names = [
    'Activation Functions',
    'Graph Neural Networks',
    'Transformer',
    'U-Net',
]

tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}

with st.sidebar:
    pass
    darkmode = st.checkbox('Dark mode', value=False)
    enable_transformer = st.checkbox('Enable Transformer', value=False)

# Activation Functions
with tabs['Activation Functions']:


    @load_or_save_fig('assets_produced/1_NN_fundamentals_and_architectures/activation_functions.png', deactivate=False, darkmode=darkmode)
    def activation_grid(darkmode=False):
        """
        Create a grid of activation function plots.

        Parameters:
        - darkmode: Boolean to enable dark mode

        Returns:
        - fig: Matplotlib figure
        """
        # activation functions
        act_funcs = {
            'Soft step' : {
                'tanh': lambda x: np.tanh(x),
                'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
                'softsign': lambda x: x / (1 + np.abs(x)),
            },
            'rectifier': {
                'relu': lambda x: np.maximum(0, x),
                'leaky_relu': lambda x: np.maximum(0.1 * x, x),
                'elu': lambda x: np.maximum(0.1 * (np.exp(x) - 1), x),
                'swish': lambda x: x * 1 / (1 + np.exp(-x)),
                'mish': lambda x: x * np.tanh(np.log(1 + np.exp(x))),
                'softplus': lambda x: np.log(1 + np.exp(x)),
            },
        }
        act_funcs['all'] = {**act_funcs['Soft step'], **act_funcs['rectifier']}  # combine all
        x = np.linspace(-3, 3, 100)

        fig, axes = plt.subplots(3, 3, figsize=(6, 6), sharex=True, sharey=True)
        color = 'black' if darkmode else 'purple'

        for (name, func), ax in zip(act_funcs['all'].items(), axes.flatten()):
            ax.plot(x, func(x), label=name, color=color, lw=3, alpha=0.7)
            
            if darkmode:
                ax.set_facecolor('lightgrey')
                ax.set_title(name, color='white')
                ax.grid(color='white', alpha=0.6)
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                fig.set_facecolor('black')
            else:
                ax.set_title(name)
                ax.grid()

            ax.set_ylim(-1.5, 1.5)
        plt.tight_layout()
        return fig

    # Title and intro
    """
    ### Activation Functions
    Activation functions are used to introduce non-linearity to a neural network.
    """
    
    cols = st.columns((2,3))
    cols[0].write("""
    Activation function introduce non-linearity to a neural network. Some common ones are displayed on the right.
                
    **A series of linear functions, remains a linear function.** As such, the network will fail to learn the generator function. For example,
                
    If we construct $f: y=x^2+z^2$, and let $x$ and $z$ be the randomly sampled input, with ground truth $y$, the network will fail to learn the generator function. (as $y$ is not linear in $x$ and $z$).
                
    *I wonder if these create different latent spaces.*
    """)


    fig = activation_grid(darkmode=darkmode)
    cols[1].pyplot(fig)



# Transformer
with tabs['Transformer']:
    if enable_transformer:
        from subpages.transformer_shakespeare import render
        render()
    
# U-Net
with tabs['U-Net']:
    import streamlit as st

    # title and introduction

    """
    ### Unet Architecture

    What is unet: Unet is neural network based around the convolution and pooling operations in a encoder-decoder architecture \cite{https://arxiv.org/pdf/1505.04597.pdf}. Through the use of skip connections, the model is able to retain exact spatial information, while also being able to learn high level features. In the original purpose of the model, they used it to segment electron microscopy images of neuronal structures. The iterative downsampling, allow the model to identify a region of interest, while the skipped connections allow the model to understand exactly where the identified region is located.

    """
    st.image('assets/1_NN_fundamentals_and_architectures/unet.png', caption='Unet architecture', use_column_width=True)



    """
    Despite originally being proposed for biomedical image segmentation, the model has seen huge success in a variety of fields. [imagen https://arxiv.org/abs/2205.11487, dalle2 https://arxiv.org/abs/2204.06125, stable-diffusion https://arxiv.org/abs/2112.10752].


    ## Specifics of the model
    The model takes an image's input. And produces an image as output. The input is passed through two convolutional layers. Which does not Decrease the image size other than. Other than cutting the edges. The images then pass through a max pooling layer, which downsamples along both dimensions by a factor 2. These two steps, along with an activation function, make up a block in the unit architecture. A block passes a copy of its output Across to the symmetrical brother. Was passing its output to yet another block In the encoder. The symmetric paths across the structure is the. skip connection. In the original paper They perform five of these blocks. But that's up for change Depending on the image sizes you're working with. 


    ## Why use Unet
    We use our unit architecture. Because we'll be working with the Mnist data set, which are images there just makes sense But also to let ourselves inspire when we move on to do the Diffusion and encoding of motion. Although we won't be using convolutional layers, the skip connections may still be relevant As per (https://arxiv.org/abs/2212.04048)


    ## How to add time and target to the unet structure
    We call extra inputs like time and target *Conditioning*. We concat these at every block of the model as is done in (https://arxiv.org/pdf/2112.10752.pdf)
    """
