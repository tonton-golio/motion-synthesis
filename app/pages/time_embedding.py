import streamlit as st
import matplotlib.pyplot as plt
import umap
import numpy as np

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

# Example usage
time_indices = np.arange(0, 512)
dimension = 32
embeddings = sinusoidal_time_embedding(time_indices, dimension)
umap_embeddings = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='cosine').fit_transform(embeddings)



fig, ax = plt.subplots(1, 2, figsize=(8, 3))
fig.suptitle('Sinusoidal Time Embeddings')

imsh = ax[0].imshow(embeddings, cmap='RdYlBu', aspect='auto')

plt.colorbar(imsh, ax=ax[0], label='Embedding Value')
ax[0].set_xlabel('Embedding Dimension')
ax[0].set_ylabel('Time Index')
ax[0].set_title('Sinusoidal Embeddings')

scat = ax[1].scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=time_indices, cmap='RdYlBu')
ax[1].set_title('UMAP Projection')
plt.colorbar(scat, ax=ax[1], label='Time Index')

plt.tight_layout()
# plt.savefig('assets/sinusoidal_time_embeddings.png')


# title and introduction
"""
# Time embedding
"""

st.pyplot(fig)


"""
embeddings

# make time embedding mlp
import torch
import torch.nn as nn
class TimeEmbeddingMLP(nn.Module):
    def __init__(self, timesteps, hidden_dim, output_dim, num_layers, dropout):
        super(TimeEmbeddingMLP, self).__init__()
        self.embedding = nn.Embedding(timesteps, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        return x
    
# Example usage

model = TimeEmbeddingMLP(timesteps=512, hidden_dim=32, output_dim=32, num_layers=2, dropout=0.1)
t = torch.tensor([0, 1]).long()
output = model(t)

# train to match sinusoidal embeddings
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

y = torch.tensor(embeddings).float()
x = torch.arange(0, 512).long()

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = TimeEmbeddingMLP(timesteps=512, hidden_dim=64, output_dim=32, num_layers=3, dropout=0.1)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1000):
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
"""