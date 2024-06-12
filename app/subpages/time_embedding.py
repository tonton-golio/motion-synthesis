import streamlit as st
import matplotlib.pyplot as plt
import umap
import numpy as np
from utils_app import load_or_save_fig

deactivate = False

def sinusoidal_time_embedding(time_indices, dimension):
    """
    Generates sinusoidal embeddings for a sequence of time indices.
    
    Args:
    time_indices (np.array): A 1D numpy array of time indices.
    dimension (int): The dimensionality of the embeddings.
    
    Returns:
    np.array: A 2D numpy array where each row represents the sinusoidal embedding of a time index.
    """
    assert dimension % 2 == 0, "Dimension must be even."
    
    # Initialize the positions matrix
    position = time_indices[:, np.newaxis]  # Convert to a column vector
    
    # Compute the divisors for each dimension
    div_term = np.exp(np.arange(0, dimension, 2) * -(np.log(10000.0) / dimension))
    
    # Calculate sinusoidal embeddings
    embeddings = np.zeros((len(time_indices), dimension))
    embeddings[:, 0::2] = np.sin(position * div_term)  # Apply sine to even indices
    embeddings[:, 1::2] = np.cos(position * div_term)  # Apply cosine to odd indices
    
    return embeddings

@load_or_save_fig("assets_produced/3_Diffusion_theory/sine_time_embed_demo.png", deactivate=deactivate)
def sine_time_embed_demo(T=512, D=32):
    # Example usage
    time_indices = np.arange(0, T)
    dimension = D
    embeddings = sinusoidal_time_embedding(time_indices, dimension)
    umap_embeddings = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='cosine').fit_transform(embeddings)


    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    fig.suptitle('Sinusoidal Time Embeddings')


    # 'RdYlBu'
    cmap = 'plasma_r'


    imsh = ax[0].imshow(embeddings, 
                        cmap=cmap,
                        aspect='auto')

    plt.colorbar(imsh, ax=ax[0], label='Embedding value')
    ax[0].set_xlabel('Embedding index')
    ax[0].set_ylabel('Time index')
    ax[0].set_title('Sinusoidal embeddings')

    scat = ax[1].scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=time_indices, cmap=cmap, s=5)
    ax[1].set_title('UMAP projection of embeddings')
    plt.colorbar(scat, ax=ax[1], label='Time Index')
    ax[1].axis('off')

    plt.tight_layout()
    
    plt.close()
    return fig


def time_embedding_page():
    # title and introduction
    st.write("""
    If we wish to inform a network of a timestep along with our sample, we must consider how to pass this information. The simplest technique, is to concatenate the timestep to the input sample. However, recieving this information in such a condensed format, forces the model to use multiple layers to expand this information.
             
    So we want a great embedding of our time-index. This embedding should obey the following properties:
             
    * time step t should be more similar to time-step $t+1$, than to time-step $t+10$.
    * and we want the embedding space to be fully occupied by the range of time-steps.
             
    A method, achieving these two goals, often used in the literature, is a sinusoidal embedding.
    """
    )   
    st.write(r"""
    Given a time index $t \in [0, T]$, we can generate a sinusoidal embedding, $E$, of dimension $D$ as follows:

    $$
        E(t, i) = \begin{cases}
            \sin\left(\frac{t}{10000^{2i/D}}\right) & \text{if i is even} \\
            \cos\left(\frac{t}{10000^{2i/D}}\right) & \text{if i is odd}
        \end{cases},
    $$
    in which, $i$ is the dimension index, and $D$ is the dimensionality of the embedding.

    
    """)
    fig = sine_time_embed_demo()
    st.pyplot(fig)

    st.write("""
             As is demonstrated in the right panel in the plot above, in which the sinusoidal embeddings are projected to a 2D space using UMAP, we see that consecutive time-steps are close to each other.

             However, in the left panel, we notice that not all indicies (especially the latter-most) are not fully utilized. Notice, for example, the last embedding index for which the value is the same across the time-index. This is a waste of the embedding space, and we could have used a lower dimensionality.
             """)

    st.divider()

    st.write("""#### Learning the time embedding""")
    cols = st.columns((1,2))
    
    with cols[0]:
        st.write("""
        To mitigate the slight under-utilization of the sinusoidal embedding, 
        embeddings OpenAI suggests to learn the time embedding using a small MLP.
                 
        A simple MLP is shown on the right.
                 
        Before plugging our time-MLP into our model, we suggest pre-training it to match the sinusoidal embeddings. This saves training time, as the parameters involved in the sinusoidal embeddings are already known, and serve as a good initialization.
        """)

    with cols[1]:
        st.code(r"""
        # make time embedding mlp
        import torch
        import torch.nn as nn
        class TimeEmbeddingMLP(nn.Module):
            def __init__(self, T, d_hidden, d_out, n_layers):
                super(self).__init__()
                self.emb = nn.Embedding(T, d_hidden)
                self.layers = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for _ in range(num_layers)])
                self.out = nn.Linear(d_hidden, d_out)
                
            def forward(self, x):
                x = self.emb(x)
                for layer in self.layers:
                    x = layer(x)
                    x = torch.relu(x)
                x = self.out(x)
                return x
        """)
