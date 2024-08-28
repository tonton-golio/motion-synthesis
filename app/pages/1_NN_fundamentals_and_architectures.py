import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from utils_app import load_or_save_fig

st.set_page_config(page_title='Neural Network Fundamentals & Architectures', layout='wide')
# Title and intro
"""
# Neural Network Fundamentals & Architectures

"""

tab_names = [
    'Activation Functions',
    #'Graph Neural Networks',
    'Transformer',
    'U-Net',
    'CLIP',
]

tabs = {name: tab for name, tab in zip(tab_names, st.tabs(tab_names))}

with st.sidebar:
    pass
    darkmode = st.checkbox('Dark mode', value=False)


# Activation Functions
with tabs['Activation Functions']:


    @load_or_save_fig('app/assets_produced/1_NN_fundamentals_and_architectures/activation_functions.png', deactivate=False, darkmode=darkmode)
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
    
    cols = st.columns((3,2))
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
    # if enable_transformer:
    from subpages.transformer_shakespeare import render
    render()
    
# U-Net
with tabs['U-Net']:
    from subpages.unet import render
    render()

# CLIP
with tabs['CLIP']:
    from subpages.CLIP import render
    render()