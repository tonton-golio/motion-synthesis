import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Title and intro
"""
# Activation Functions
Activation functions are used to introduce non-linearity to a neural network.
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


def activation_grid(act_funcs, x, darkmode=True):
    fig, axes = plt.subplots(3, 3, figsize=(6, 6), sharex=True, sharey=True)
    if darkmode:
        fig.patch.set_facecolor('black')
        fig.patch.set_alpha(0.)
        color = 'white'
    else:
        color = 'purple'

    for (name, func), ax in zip(act_funcs.items(), axes.flatten()):
        ax.plot(x, func(x), label=name, color=color, lw=6, alpha=0.3)
        ax.plot(x, func(x), label=name, color=color, lw=3, alpha=0.3, ls='-')
        ax.plot(x, func(x), label=name, color=color, lw=1, alpha=0.3, ls='-')
        
        
        if darkmode: 
            ax.set_facecolor('grey')
            ax.set_title(name, color='white')
            ax.grid(color='white', alpha=0.6)
            ax.set_xticks([-1,0,1], [-1,0,1], color='white')
            ax.set_yticks([-1,0,1], [-1,0,1], color='white')
        else:
            ax.set_title(name)
            ax.grid()
        ax.set_ylim(-1.5, 1.5)
    plt.tight_layout()
    return fig

def plot_functions(act_funcs, x, darkmode=True, **kwargs):
    fig, ax = plt.subplots(figsize=(5, 3))
    if darkmode:
        'dark'
        # plt.style.use('dark_background')
    for name, f in act_funcs.items():
        ax.plot(x, f(x), label=name)
    ax.legend(ncol=kwargs.get('ncol', 1))#, bbox_to_anchor=kwargs.get('bbox', (1, 1)))
    ax.set_ylim(kwargs.get('ylim', (-1.3, 1.3)))
    
    ax.grid()
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    x = np.linspace(-3, 3, 100)
    fig = activation_grid(act_funcs['all'], x)
    cols = st.columns((2,3))
    cols[0].write("""
    Activation function introduce non-linearity to a neural network. Some common ones are displayed on the right.
                  
    **A series of linear functions, remains a linear function.** As such, the network will fail to learn the generator function. For example,
                  
    If we construct $f: y=x^2+z^2$, and let $x$ and $z$ be the randomly sampled input, with ground truth $y$, the network will fail to learn the generator function. (as $y$ is not linear in $x$ and $z$).
                  
    *I wonder if these create different latent spaces.*
    """)
    cols[1].pyplot(fig)

    # # cols = st.columns(2)
    # fig = plot_functions(act_funcs['Soft step'], x)
    # cols[1].pyplot(fig)

    # fig = plot_functions(act_funcs['rectifier'], x, ncol=2, ylim=(-1, 3), )
    # cols[1].pyplot(fig)
