import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import plot_3d_motion_frames_multiple, plot_3d_motion_animation, plot_3d_motion_frames_multiple
from glob import glob


def get_loss_function(loss_function, kl_weight=0.1, **kwargs):
    if loss_function == "L1Loss":
        return nn.L1Loss()
    elif loss_function == "MSELoss":
        return nn.MSELoss()
    elif loss_function == "SmoothL1Loss":
        return nn.SmoothL1Loss()

    elif loss_function == "MSELoss + KL":
        # this should be the sum of the reconstruction loss and the KL divergence
        mse = nn.MSELoss()
        kl = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return lambda x, y, mu, logvar: mse(x, y) + kl_weight * kl(mu, logvar)

    raise ValueError(f"Loss function {loss_function} not found")

def get_optimizer(model, optimizer_name, learning_rate):
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer")
    return optimizer


class TransformerMotionAutoencoder(pl.LightningModule):
    def __init__(
        self,
        input_length,
        input_dim,
        hidden_dim=512,
        n_layers=2,
        n_heads=6,
        dropout=0.01,
        latent_dim=256,
        loss_function="MSELoss",
        learning_rate=2 * 1e-4,
        optimizer="AdamW",
        kl_weight=0.1,
        save_animations=False,
    ):
        super(TransformerMotionAutoencoder, self).__init__()
        self.input_length = input_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.optimizer = optimizer
        self.save_animations = save_animations
        self.loss_function = get_loss_function(loss_function, kl_weight=kl_weight)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.input_dim,
                nhead=self.n_heads,
                dim_feedforward=self.hidden_dim,
                dropout=self.dropout,
                batch_first=True,
            ),
            num_layers=self.n_layers,
        )

        self.fc1_enc = nn.Linear(self.input_dim * self.input_length, self.hidden_dim)
        self.fc2_enc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_mu_enc = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_logvar_enc = nn.Linear(self.hidden_dim, latent_dim)

        self.fc1_dec = nn.Linear(latent_dim, self.hidden_dim)
        self.fc2_dec = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3_dec = nn.Linear(self.hidden_dim, self.input_dim * self.input_length)

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.input_dim,
                nhead=self.n_heads,
                dim_feedforward=self.hidden_dim,
                dropout=self.dropout,
                batch_first=True,
            ),
            num_layers=self.n_layers,
        )
        self.fc_out1 = nn.Linear(
            self.input_dim * self.input_length, self.input_dim * self.input_length
        )

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(-1, self.input_length, self.input_dim)
        x = self.transformer_encoder(x)
        x = x.view(-1, self.input_dim * self.input_length)
        x = F.relu(self.fc1_enc(x))
        x = F.relu(self.fc2_enc(x))
        mu = self.fc_mu_enc(x)
        logvar = self.fc_logvar_enc(x)

        z = self.reparametrize(mu, logvar)

        x = F.relu(self.fc1_dec(z))
        x = F.relu(self.fc2_dec(x))
        x = F.relu(self.fc3_dec(x))

        x = x.view(-1, self.input_length, self.input_dim)
        x = self.transformer_decoder(x, x)

        x = x.view(-1, self.input_dim * self.input_length)
        x = self.fc_out1(x)
        return x.view(-1, self.input_length, self.input_dim // 3, 3), mu, logvar

    def training_step(self, batch, batch_idx):
        loss, recon, x = self._common_step(batch, batch_idx)

        if batch_idx == 0:
            im_arr = plot_3d_motion_frames_multiple([recon.cpu().detach().numpy(), x.cpu().detach().numpy()], ["recon", "true"], 
                                                    nframes=5, radius=2, figsize=(20,8), return_array=True)
            print(im_arr.shape)
            self.logger.experiment.add_image("recon_vs_true", im_arr, global_step=self.global_step)
        self.log("train_loss", loss, prog_bar=True)

        # if first batch, make a grid of the first 8 images in sequence

        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon, x = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, recon, x = self._common_step(batch, batch_idx)
        #self.log("test_loss", loss)
        # we want to add test loss final to the tensorboard
        self.log("test_loss", loss, on_epoch=True, on_step=False)

        if batch_idx == 0 and self.save_animations:
            print("Saving animations")
            folders = glob("../tb_logs/TransformerMotionAutoencoder/version_*")
            # sort
            folders = sorted(folders, key=lambda x: int(x.split("_")[-1]))
            folder = folders[-1]
            plot_3d_motion_animation(recon[0].cpu().detach().numpy(), "recon", figsize=(10, 10), fps=20, radius=2, save_path=f"{folder}/recon.mp4")


        return loss

    def _common_step(self, batch, batch_idx):
        x = batch
        recon, mu, logvar = self(x)
        loss = self.loss_function(x, recon, mu=mu, logvar=logvar)
        return loss, recon, x

    def configure_optimizers(self):
        # this is also where we would put the scheduler
        return get_optimizer(self, self.optimizer, self.lr)
