"""Model-agnostic Gaussian diffusion (DDPM + DDIM) with eps-prediction.

This replaces the broken diffusion math in MotionLatentDiffusion.py (deep-review
B08/B10): the original early-returned the raw network output (all reverse-posterior
math was dead code), trained an x0-regressor but "sampled" by iterating it, and
scaled the noise by a constant ``noise_multiplier`` that breaks the
variance-preserving schedule.

Here the forward process uses UNIT-variance noise, the model predicts epsilon, and
sampling uses the correct ancestral DDPM posterior or deterministic DDIM. The class
is decoupled from any particular network — ``model_fn`` is a callable
``(x_t, t) -> eps_pred`` — so it is unit-tested on a toy 2D mixture
(tests/test_diffusion_toy_2d.py) independently of the motion VAE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    acp = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    acp = acp / acp[0]
    betas = 1 - acp[1:] / acp[:-1]
    return betas.clamp(1e-8, 0.999)


def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000, schedule: str = "cosine"):
        super().__init__()
        self.timesteps = timesteps
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(schedule)

        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = torch.cat([torch.ones(1), acp[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("alphas_cumprod_prev", acp_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(acp))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - acp))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / acp))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / acp - 1.0))

        # true posterior q(x_{t-1} | x_t, x_0)
        posterior_var = betas * (1.0 - acp_prev) / (1.0 - acp)
        self.register_buffer("posterior_variance", posterior_var)
        self.register_buffer("posterior_log_var", torch.log(posterior_var.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(acp_prev) / (1.0 - acp))
        self.register_buffer("posterior_mean_coef2", (1.0 - acp_prev) * torch.sqrt(alphas) / (1.0 - acp))

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        """Gather a[t] and reshape to broadcast over a tensor of ``shape``."""
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))

    # ---- forward process ----
    def q_sample(self, x0, t, noise):
        return (self._extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
                + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    def predict_x0_from_eps(self, x_t, t, eps):
        return (self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps)

    def q_posterior_mean(self, x0, x_t, t):
        return (self._extract(self.posterior_mean_coef1, t, x_t.shape) * x0
                + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)

    # ---- training ----
    def training_loss(self, model_fn, x0, noise=None):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)  # UNIT variance (no noise_multiplier)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = model_fn(x_t, t)
        return F.mse_loss(eps_pred, noise)

    # ---- DDPM ancestral sampling ----
    @torch.no_grad()
    def p_sample(self, model_fn, x_t, t, clip_x0=None):
        eps = model_fn(x_t, t)
        x0 = self.predict_x0_from_eps(x_t, t, eps)
        if clip_x0 is not None:
            x0 = x0.clamp(-clip_x0, clip_x0)
        mean = self.q_posterior_mean(x0, x_t, t)
        noise = torch.randn_like(x_t)
        # no noise at t == 0 (per-element mask, not all-or-nothing)
        nonzero = (t != 0).float().reshape(t.shape[0], *([1] * (x_t.dim() - 1)))
        var = self._extract(self.posterior_variance, t, x_t.shape)
        return mean + nonzero * torch.sqrt(var) * noise

    @torch.no_grad()
    def p_sample_loop(self, model_fn, shape, device, clip_x0=None):
        x_t = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x_t = self.p_sample(model_fn, x_t, t, clip_x0=clip_x0)
        return x_t

    # ---- DDIM sampling ----
    @torch.no_grad()
    def ddim_sample(self, model_fn, shape, device, steps=50, eta=0.0, clip_x0=None):
        times = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        x_t = torch.randn(shape, device=device)
        for i in range(steps):
            t = torch.full((shape[0],), int(times[i].item()), device=device, dtype=torch.long)
            eps = model_fn(x_t, t)
            x0 = self.predict_x0_from_eps(x_t, t, eps)
            if clip_x0 is not None:
                x0 = x0.clamp(-clip_x0, clip_x0)
            acp_t = self._extract(self.alphas_cumprod, t, x_t.shape)
            if i < steps - 1:
                t_prev = torch.full((shape[0],), int(times[i + 1].item()), device=device, dtype=torch.long)
                acp_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)
            else:
                acp_prev = torch.ones_like(acp_t)
            sigma = eta * torch.sqrt(((1 - acp_prev) / (1 - acp_t)) * (1 - acp_t / acp_prev))
            noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
            x_t = (acp_prev.sqrt() * x0
                   + (1 - acp_prev - sigma ** 2).clamp(min=0).sqrt() * eps
                   + sigma * noise)
        return x_t
