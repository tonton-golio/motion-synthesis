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

act_funcs['all'] = {**act_funcs['Soft step'], **act_funcs['rectifier']}


def activation_grid(act_funcs, x):
    fig, axes = plt.subplots(3, 3, figsize=(9, 6), sharex=True, sharey=True)
    for (name, func), ax in zip(act_funcs.items(), axes.flatten()):
        ax.plot(x, func(x), label=name)
        ax.set_title(name)
        ax.grid()
        ax.set_ylim(-1.5, 1.5)
    plt.tight_layout()
    return fig

def plot_functions(act_funcs, x):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, f in act_funcs.items():
        ax.plot(x, f(x), label=name)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    x = np.linspace(-5, 5, 100)
    fig = activation_grid(act_funcs['all'], x)
    st.pyplot(fig)

    cols = st.columns(2)
    fig = plot_functions(act_funcs['Soft step'], x)
    cols[0].pyplot(fig)

    fig = plot_functions(act_funcs['rectifier'], x)
    cols[1].pyplot(fig)
    
