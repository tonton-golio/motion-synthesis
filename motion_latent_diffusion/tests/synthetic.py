"""Synthetic motion data for CPU unit tests (no dataset / GPU required)."""

import math
import torch


def make_synthetic_motion(B=4, T=16, njoints=22, seed=0, device="cpu"):
    """Smooth, per-sample-distinct (B, T, njoints, 3) joint trajectories.

    Each joint follows a base offset plus low-frequency sinusoids, giving
    temporally-smooth motion that a VAE should be able to overfit quickly.
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 2 * math.pi, T).view(1, T, 1, 1)
    base = torch.randn(B, 1, njoints, 3, generator=g)
    freqs = torch.randint(1, 4, (B, 1, njoints, 3), generator=g).float()
    phase = torch.rand(B, 1, njoints, 3, generator=g) * 2 * math.pi
    amp = 0.3 * torch.rand(B, 1, njoints, 3, generator=g)
    motion = base + amp * torch.sin(freqs * t + phase)
    return motion.to(device)


def small_model(latent_dim=64, latent_size=1, seq_len=16, nlayers=3, nhead=4, **kw):
    """A small MotionVAE_MLD for fast CPU tests."""
    from motion_latent_diffusion.modules.motion_vae_mld import MotionVAE_MLD

    return MotionVAE_MLD(
        latent_dim=latent_dim,
        latent_size=latent_size,
        seq_len=seq_len,
        input_dim=66,
        njoints=22,
        nhead=nhead,
        ff_transformer=128,
        nlayers_transformer=nlayers,
        dropout=0.0,
        transformer_activation="gelu",
        **kw,
    )
