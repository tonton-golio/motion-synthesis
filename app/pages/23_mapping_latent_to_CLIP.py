import streamlit as st


"""
# Mapping Latent to CLIP

*Can our latent vectors be mapped to CLIP in a generalizable manner?*

If so, the CLIP embeddings contain the distinctions also present in the latent space.

This means our diffusion model should be able to use the CLIP embeddings to generate motion sequences with the desired descriptions.
"""

st.divider()


r"""
To test this, we consider the shapes of our latent vectors and the CLIP embeddings.
$$
    \text{latent} \in \mathbb{R}^{1024} \quad \text{CLIP} \in \mathbb{R}^{512}
$$

We setup a network of linear layers, and watch if val loss decreases.
"""
import torch
import torch.nn as nn

class MappingLatentToCLIP:
    def __init__(self, latent_dim=1024, clip_dim=512):
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, clip_dim)
        )

    def forward(self, latent):
        return self.model(latent)

    def train(self, latent, clip):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for i in range(1000):
            optimizer.zero_grad()
            pred = self.forward(latent)
            loss = criterion(pred, clip)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {i}, Loss {loss.item()}")

        return loss.item()
