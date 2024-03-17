import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from tqdm import tqdm

import torchvision

class SimpleModel(nn.Module):
    # a simple model takes (batchsize, latent dim),
    # performs linear layers
    # and returns (batchsize, latent dim)

    def __init__(
        self,
        latent_dim,
        hidden_dim,
        nhidden=5,
        timesteps=1000,
        time_embedding_dim=64,
        target_embedding_dim=5,
        dp_rate=0.1,
    ):
        super(SimpleModel, self).__init__()
        self.time_embedding_dim = time_embedding_dim
        self.target_embedding_dim = target_embedding_dim
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)
        self.target_embedding = nn.Embedding(10, target_embedding_dim)
        # dropout
        self.dropout = nn.Dropout(dp_rate)

        self.fc1 = nn.Linear(
            latent_dim + time_embedding_dim + target_embedding_dim, hidden_dim
        )

        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(nhidden)]
        )

        self.fc_hidden2conv = nn.Linear(hidden_dim, 32 * 7 * 7)
        # make transpose convolutional layers
        self.conv1 = nn.ConvTranspose2d(
            32, 16, 3, stride=2, padding=1, output_padding=1
        )
        self.conv2 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)

        self.fc2 = nn.Sequential(
            nn.Linear(28 * 28, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x, y, t):
        # print(x.shape, t.shape)
        # print(x, t)
        # print('x', x.shape, 't', t.shape, 'y', y.shape)
        t = self.time_embedding(t)  # embed time
        y = self.target_embedding(y).view(
            y.shape[0], self.target_embedding_dim
        )  # embed target
        # print('x', x.shape, 't', t.shape, 'y', y.shape)
        x = torch.cat([x, t, y], dim=-1)
        # x = torch.cat([x, t], dim=-1)
        # print(x.shape, t.shape)
        x = nn.LeakyReLU()(self.fc1(x))
        x = self.dropout(x)
        for layer in self.fc_hidden:
            # x = torch.relu(layer(x))
            x = nn.LeakyReLU()(layer(x))
            x = self.dropout(x)

        # conv
        x = nn.LeakyReLU()(self.fc_hidden2conv(x))
        x = x.view(-1, 32, 7, 7)
        x = nn.LeakyReLU()(self.conv1(x))
        x = nn.LeakyReLU()(self.conv2(x))
        # print('x', x.shape)
        x = x.view(-1, 28 * 28)

        x = self.fc2(x)

        return x


class LatentDiffusion(nn.Module):
    def __init__(
        self,
        latent_dim=8,
        hidden_dim=64,
        nhidden=3,
        timesteps=1000,
        time_embedding_dim=64,
        target_embedding_dim=5,
        epsilon=0.008,
        dp_rate=0.1,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = latent_dim

        betas = self._cosine_variance_schedule(timesteps, epsilon)
        # print('betas', betas.shape)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        # print('alphas_cumprod', alphas_cumprod.shape)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        self.model = SimpleModel(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            nhidden=nhidden,
            timesteps=timesteps,
            time_embedding_dim=time_embedding_dim,
            target_embedding_dim=target_embedding_dim,
            dp_rate=dp_rate,
        )

    def forward(self, x, y, noise):
        # x:NCHW
        t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
        # print('t from the LatentDIffusion forward', t)

        x_t = self._forward_diffusion(x, t, noise)

        # print('x_t', x_t.shape, )
        # print('t', t.shape)
        pred_noise = self.model(x_t, y, t)
        # print('pred_noise', pred_noise.shape)

        return pred_noise

    @torch.no_grad()
    def sampling(self, n_samples, clipped_reverse_diffusion=True, device="cuda"):
        x_t = torch.randn(
            (n_samples, self.in_channels, self.image_size, self.image_size)
        ).to(device)
        for i in tqdm(range(self.timesteps - 1, -1, -1), desc="Sampling"):
            noise = torch.randn_like(x_t).to(device)
            t = torch.tensor([i for _ in range(n_samples)]).to(device)

            if clipped_reverse_diffusion:
                x_t = self._reverse_diffusion_with_clip(x_t, t, noise)
            else:
                x_t = self._reverse_diffusion(x_t, t, noise)

        x_t = (x_t + 1.0) / 2.0  # [-1,1] to [0,1]

        return x_t

    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = (
            torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)
            ** 2
        )
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)

        return betas

    def _forward_diffusion(self, x_0, t, noise):
        # print('x_0', x_0.shape)
        # print('noise', noise.shape)
        # print('t', t)

        assert x_0.shape == noise.shape
        # print('self.sqrt_alphas_cumprod.gather(t)', self.sqrt_alphas_cumprod.gather(0,t).shape)
        # q(x_{t}|x_{t-1})

        A = self.sqrt_alphas_cumprod.gather(0, t).unsqueeze(1)
        B = self.sqrt_one_minus_alphas_cumprod.gather(0, t).unsqueeze(1)
        # print('A', A.shape)
        # print('B', B.shape)
        # print('noise', noise.shape)
        # print('x_0', x_0.shape)
        return A * x_0 + B * noise

    @torch.no_grad()
    def _reverse_diffusion(self, x_t, y, t, noise):
        """
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        """
        pred = self.model(x_t, y, t)

        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1)  # ,1,1)
        # print('alpha_t', alpha_t.shape)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(
            x_t.shape[0], 1
        )  # ,1,1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1)  # ,1,1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(
            -1, t
        ).reshape(
            x_t.shape[0], 1
        )  # ,1,1)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred
        )

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(
                x_t.shape[0], 1
            )  # ,1,1)
            std = torch.sqrt(
                beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            )
        else:
            std = 0.0

        return mean + std * noise


# make pl model
class LatentDiffusionModel(pl.LightningModule):
    def __init__(
        self,
        decoder,
        **kwargs,
    ):
        super().__init__()
        self.lr = kwargs.get("lr", 0.001)
        self.model = LatentDiffusion(
            latent_dim=kwargs.get("latent_dim", 8),
            hidden_dim=kwargs.get("hidden_dim", 64),
            nhidden=kwargs.get("nhidden", 5),
            timesteps=kwargs.get("timesteps", 100),
            time_embedding_dim=kwargs.get("time_embedding_dim", 8),
            epsilon=kwargs.get("epsilon", 0.008),
            target_embedding_dim=kwargs.get("target_embedding_dim", 8),
            dp_rate=kwargs.get("dp_rate", 0.1),
        )
        self.noise_multiplier = kwargs.get("noise_multiplier", 3.0)
        # self.save_hyperparameters()
        self.decoder = decoder

    def forward(self, data):
        x, y = data
        noise = torch.randn_like(x) * self.noise_multiplier
        # print('x', x.shape)
        # print('noise', noise.shape)
        return self.model(x, y, noise), noise

    def training_step(self, batch, batch_idx):
        pred_noise, noise = self.forward(batch)
        loss = nn.functional.mse_loss(pred_noise, noise)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        pred_noise, noise = self.forward(batch)
        loss = nn.functional.mse_loss(pred_noise, noise)


        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        if batch_idx == 0 and self.decoder is not None:
            with torch.no_grad():
                # make image by decoding latent space
                x, y = batch
                # print(x.shape, y.shape)
                pred_clean = x-pred_noise
                x_dirty = x + noise
                raw_reconstruction = self.decoder(x_dirty.detach().cpu())
                reconstruction = self.decoder(pred_clean.detach().cpu())
                # print('reconstruction', reconstruction.shape)

                raw_and_recon = torch.cat([raw_reconstruction[:8], reconstruction[:8], ])

                grid = torchvision.utils.make_grid(raw_and_recon[:16], nrow=8, normalize=True)
                self.logger.experiment.add_image('top: noisy input, bot: reconstruction', grid, global_step=self.global_step)

        return loss

    def test_step(self, batch, batch_idx):
        pred_noise, noise = self.forward(batch)
        loss = nn.functional.mse_loss(pred_noise, noise) / self.noise_multiplier
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    model = LatentDiffusionModel(
        latent_dim=8,
        hidden_dim=64,
        nhidden=5,
        timesteps=1000,
        time_embedding_dim=8,
        target_embedding_dim=8,
        epsilon=0.008,
        dp_rate=0.1,
        decoder=None
    )
    n_samples = 2
    sample = torch.randn(n_samples, 8)  # x_0
    print("sample", sample.shape, sample)

    # print('model', model)
    # t = torch.randint(0, 1000, (n_samples,))
    y = torch.randint(0, 10, (n_samples,))
    print("y", y.shape, y)

    out = model((sample, y))
    print("out", out)
