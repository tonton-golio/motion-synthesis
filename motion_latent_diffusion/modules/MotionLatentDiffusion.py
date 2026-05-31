"""Latent-space conditional diffusion over motion-VAE latents.

Rebuilt from the broken original (deep-review B08/B09/B10): epsilon-prediction
training, the correct DDPM/DDIM reverse process (in modules/diffusion.py), unit
variance noise (no ``noise_multiplier``), classifier-free guidance, and no
hardcoded ``device='mps'``.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

try:
    from motion_latent_diffusion.modules.diffusion import GaussianDiffusion
except ImportError:
    from modules.diffusion import GaussianDiffusion


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(self, t):
        return self.pe[t]


def _mlp(in_dim, hidden, out_dim, nhidden=2, act=nn.SiLU, dropout=0.0):
    layers = [nn.Linear(in_dim, hidden), act()]
    for _ in range(nhidden):
        layers += [nn.Linear(hidden, hidden), act()]
        if dropout:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class LatentDenoiser(nn.Module):
    """epsilon predictor: (x_t, t, cond) -> eps, with a learned null condition for CFG."""

    def __init__(self, latent_dim, cond_dim=512, hidden_dim=512, time_dim=64,
                 nhidden=4, timesteps=1000, dropout=0.1):
        super().__init__()
        self.time_embed = SinusoidalPositionalEmbedding(time_dim, max_len=timesteps)
        self.time_mlp = _mlp(time_dim, time_dim * 2, time_dim, nhidden=1, dropout=0.0)
        self.cond_mlp = _mlp(cond_dim, cond_dim, time_dim, nhidden=2, dropout=dropout)
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))  # CFG unconditional token
        self.backbone = _mlp(latent_dim + 2 * time_dim, hidden_dim, latent_dim,
                             nhidden=nhidden, dropout=dropout)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x_t, t, cond):
        te = self.time_mlp(self.time_embed(t))
        ce = self.cond_mlp(cond)
        return self.backbone(torch.cat([x_t, ce, te], dim=-1))


class MotionLatentDiffusion(pl.LightningModule):
    def __init__(self, decode=None, projector=None, projection=None, scaler=None,
                 verbose=False, **kwargs):
        super().__init__()
        self.verbose = verbose
        self.lr = kwargs.get("lr", 1e-4)
        self.latent_dim = kwargs.get("latent_dim", 256)
        # CLIP text features are 512-d; accept either name.
        self.cond_dim = kwargs.get("cond_dim", kwargs.get("target_embedding_dim", 512))
        self.timesteps = kwargs.get("timesteps", 1000)
        self.cfg_dropout = kwargs.get("cfg_dropout", 0.1)
        self.cfg_scale = kwargs.get("cfg_scale", 2.0)
        self.sample_steps = kwargs.get("sample_steps", 50)
        self.clip_x0 = kwargs.get("clip_x0", 3.0)

        self.diffusion = GaussianDiffusion(
            timesteps=self.timesteps, schedule=kwargs.get("schedule", "cosine"))
        self.model = LatentDenoiser(
            latent_dim=self.latent_dim,
            cond_dim=self.cond_dim,
            hidden_dim=kwargs.get("hidden_dim", 512),
            time_dim=kwargs.get("time_embedding_dim", 64),
            nhidden=kwargs.get("nhidden", 4),
            timesteps=self.timesteps,
            dropout=kwargs.get("dp_rate", 0.1),
        )

        # kept for visualization (guarded); not required for training correctness
        self.decode = decode
        self.projector = projector
        self.projection = projection
        self.scaler = scaler

    # ---- conditioning helpers ----
    def _drop_cond(self, cond):
        """Randomly replace a fraction of conditions with the null token (CFG train)."""
        if self.cfg_dropout <= 0:
            return cond
        mask = torch.rand(cond.shape[0], device=cond.device) < self.cfg_dropout
        null = self.model.null_cond.to(cond.dtype).expand_as(cond)
        return torch.where(mask.unsqueeze(-1), null, cond)

    def _guided_fn(self, cond, scale):
        null = self.model.null_cond.expand(cond.shape[0], -1)

        def fn(x_t, t):
            eu = self.model(x_t, t, null)
            if scale == 0:
                return eu
            ec = self.model(x_t, t, cond)
            return eu + scale * (ec - eu)
        return fn

    # ---- train / val / test ----
    def _step(self, batch, stage):
        x, y = batch[0], batch[1]
        y = self._drop_cond(y)
        loss = self.diffusion.training_loss(lambda x_t, t: self.model(x_t, t, y), x)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    @torch.no_grad()
    def sample(self, cond, scale=None, steps=None, eta=0.0):
        """Generate latents conditioned on ``cond`` (B, cond_dim) via guided DDIM."""
        scale = self.cfg_scale if scale is None else scale
        steps = self.sample_steps if steps is None else steps
        shape = (cond.shape[0], self.latent_dim)
        return self.diffusion.ddim_sample(
            self._guided_fn(cond, scale), shape, device=cond.device,
            steps=steps, eta=eta, clip_x0=self.clip_x0)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.timesteps)
        return {"optimizer": opt, "lr_scheduler": sch}
