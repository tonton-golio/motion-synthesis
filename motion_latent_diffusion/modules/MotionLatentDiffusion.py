import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from tqdm import tqdm
import torchvision


from utils import plot_3d_motion_animation, translate
import matplotlib.pyplot as plt

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
        target_size=10054,
        dp_rate=0.1,
        verbose=False
    ):
        super(SimpleModel, self).__init__()
        self.verbose = verbose
        self.time_embedding_dim = time_embedding_dim
        self.target_embedding_dim = target_embedding_dim
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)
        self.target_embedding = nn.Embedding(target_size+1, target_embedding_dim, sparse=False)
        # dropout
        self.dropout = nn.Dropout(dp_rate)

        input_dim = latent_dim + time_embedding_dim + 250*4
        out_put_dim = latent_dim
        self.fc1 = nn.Linear(
            input_dim, hidden_dim
        )
        
        # input_dim = 11036
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(nhidden)]
        )

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
        )

        # text transformer encoder
        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=target_embedding_dim,
                nhead=10,
                dim_feedforward=512,
                dropout=0.1,
                activation='relu',
                batch_first=True
            ),
            num_layers=3
        )
        self.text_linear1 = nn.Linear(target_embedding_dim, 4)

    def forward(self, x, y, t):
        if self.verbose:
            print('x', x.shape)
            print('y', y.shape)
            print('t', t.shape)

        t = self.time_embedding(t)  # (batchsize, time_embedding_dim)
        if self.verbose:
            print('t', t.shape)
        y = self.target_embedding(y)
        if self.verbose:
            print('y (after target embedding)', y.shape)

        y = self.text_transformer(y)
        if self.verbose:
            print('y', y.shape)
        y = self.text_linear1(y)
        y = nn.Flatten()(y)
        if self.verbose:
            print('y (done)', y.shape)
            print('t', t.shape)
            print('x', x.shape)
        

        x = torch.cat([x, t, y], dim=1)
        if self.verbose:
            print('after cat: x', x.shape)
        x = self.fc1(x)
        if self.verbose:
            print('x', x.shape)
        for layer in self.fc_hidden:
            # x = torch.relu(layer(x))
            x = nn.LeakyReLU()(layer(x))
            x = self.dropout(x)

        x = self.fc2(x)
        return x

class LatentDiffusionModel(nn.Module):
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
        verbose=False
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
            verbose=verbose
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
class MotionLatentDiffusion(pl.LightningModule):
    def __init__(
        self,
        decode,
        idx2word,
        verbose = False,
        **kwargs,
    ):
        super().__init__()
        self.verbose = verbose
        self.idx2word = idx2word
        self.lr = kwargs.get("lr", 0.001)
        self.model = LatentDiffusionModel(
            latent_dim=kwargs.get("latent_dim", 8),
            hidden_dim=kwargs.get("hidden_dim", 64),
            nhidden=kwargs.get("nhidden", 5),
            timesteps=kwargs.get("timesteps", 100),
            time_embedding_dim=kwargs.get("time_embedding_dim", 8),
            epsilon=kwargs.get("epsilon", 0.008),
            target_embedding_dim=kwargs.get("target_embedding_dim", 8),
            dp_rate=kwargs.get("dp_rate", 0.1),
            verbose=verbose
        )
        self.noise_multiplier = kwargs.get("noise_multiplier", .1)
        # self.save_hyperparameters()
        self.decode = decode

    def forward(self, data):
        x, y = data
        if self.verbose:
            print('x', x.shape)
            print('y', y.shape)
        noise = torch.randn_like(x) * self.noise_multiplier
        return self.model(x, y, noise), noise
    
    def _reverse_diffusion(self, x_t, y, t):
        noise = torch.randn_like(x_t)
        return self.model._reverse_diffusion(x_t, y, t, noise)

    def training_step(self, batch, batch_idx):
        pred_noise, noise = self.forward(batch)
        # print('pred_noise', pred_noise.shape)
        loss = nn.functional.mse_loss(pred_noise, noise)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        pred_noise, noise = self.forward(batch)
        loss = nn.functional.mse_loss(pred_noise, noise)


        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        if batch_idx == 0 and self.decode is not None:
            with torch.no_grad():
                # make image by decoding latent space
                x, y = batch
                # print(x.shape, y.shape)
                pred_clean = x-pred_noise
                x_dirty = x + noise

                raw_reconstruction = self.decode(x_dirty)
                reconstruction = self.decode(pred_clean)
                # print('reconstruction', reconstruction.shape)

                # raw_and_recon = torch.cat([raw_reconstruction[:8], reconstruction[:8], ])

                # grid = torchvision.utils.make_grid(raw_and_recon[:16], nrow=8, normalize=True)
                # self.logger.experiment.add_image('top: noisy input, bot: reconstruction', grid, global_step=self.global_step)

                plot_3d_motion_animation(raw_reconstruction[0].cpu().detach().numpy(), translate(y[0], self.idx2word), 
                                     figsize=(10, 10), fps=20, radius=2, save_path=f"recon_dirty.mp4", velocity=False)
                plt.close()

                plot_3d_motion_animation(reconstruction[0].cpu().detach().numpy(), translate(y[0], self.idx2word), 
                                     figsize=(10, 10), fps=20, radius=2, save_path=f"recon_clean.mp4", velocity=False)
                plt.close()

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
