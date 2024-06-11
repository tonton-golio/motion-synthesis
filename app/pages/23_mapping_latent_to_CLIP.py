import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


"""
# Mapping CLIP to Latent (because we have multiple descriptions for each latent vector)

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

class CosineSimilarity(nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def forward(self, x, y):
        return torch.nn.functional.cosine_similarity(x, y, dim=-1)

class MappingCLIPToLatent(nn.Module):
    def __init__(self, latent_dim=1024, clip_dim=512, n_layers=2, hidden_dim=512, act=nn.ReLU()):
        super(MappingCLIPToLatent, self).__init__()
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.act = act
        self.dropout = nn.Dropout(0.2)

        self.model = nn.Sequential(
            nn.Linear(clip_dim, self.hidden_dim),

            *[nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                self.act,
                self.dropout
            ) for _ in range(self.n_layers)],
            nn.Linear(self.hidden_dim, latent_dim)
        )

    def forward(self, clip):
        return self.model(clip)

    def train(self, train_loader, val_loader, n_epochs=6):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        self.losses = {k: [] for k in ['train', 'val']}
        for e in range(n_epochs):
            running_loss = 0
            for b, (clip, latent) in enumerate(train_loader):
                optimizer.zero_grad()
                pred_clip = self.forward(clip)
                loss = criterion(pred_clip, latent)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # print(f"Loss {b}/{len(dataloader)}: {loss.item()}")
            print(f"Epoch {e}: {running_loss/b}")
            self.losses['train'].append(running_loss/b)

            with torch.no_grad():
                running_val_loss = 0
                for b, (clip, latent) in enumerate(val_loader):
                    pred_clip = self.forward(clip)
                    loss = criterion(pred_clip, latent)
                    running_val_loss += loss.item()
                print(f"Val Loss {e}: {running_val_loss/b}")
                self.losses['val'].append(running_val_loss/b)







base_path = '../motion_latent_diffusion/logs/MotionVAE/VAE1/train/version_107/saved_latent/'    
paths = {
    f'latent_{stage}': base_path + f'latent_{stage}.pt' for stage in ['train', 'val', 'test']
}
paths.update({
    f'clip_{stage}': base_path + f'clip_{stage}.pt' for stage in ['train', 'val', 'test']
})

def loader(stage, n=10, device = 'mps', batch_size=64):
    latent = torch.load(paths[f'latent_{stage}'])[:n]
    clip = torch.load(paths[f'clip_{stage}'])[:n]
    latent = latent.unsqueeze(1).repeat(1, 3, 1).view(-1, latent.shape[-1])
    clip = clip.view(-1, clip.shape[-1])

    
    dataset = TensorDataset(clip.to(device), latent.to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


n = 10000
device = 'mps'
val_loader = loader('val', n=n, device=device)
train_loader = loader('train', n=n, device=device)


# we need to copy latent 3 times along a new axis at 1


model = MappingCLIPToLatent(n_layers=5).to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
st.write(f"Model has {num_params} parameters")
if st.selectbox("Train", [False, True]):
    model.train(train_loader, val_loader, n_epochs=10)

# fig = plt.figure()
# plt.plot(model.losses, lw=100)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Loss over training")

# dark theme
if st.sidebar.checkbox("Dark Theme"):
    plt.style.use('dark_background')
else:
    plt.style.use('default')
fig, ax = plt.subplots()
ax.plot(model.losses['train'], label='train')
ax.plot(model.losses['val'], label='val')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Loss over training")
ax.legend()

cols = st.columns(2)
with cols[1]:
    st.pyplot(fig)

if False:
    with cols[0]:
        

        st.write("## Model")
        from torchviz import make_dot
        x = torch.randn(1, 512).to(device)
        y = model.forward(x)
        path = "assets_produced/23_mapping_latent_to_CLIP_model/MappingLatentToCLIP.png"
        make_dot(y, 
                params=dict(model.model.named_parameters()), 
                show_attrs=True, show_saved=True
                ).render("assets_produced/23_mapping_latent_to_CLIP_model/MappingLatentToCLIP", 
                        format="png",
                        cleanup=True)
        st.image("assets_produced/23_mapping_latent_to_CLIP_model/MappingLatentToCLIP.png")

