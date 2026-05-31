"""Tests for eval metrics and the rebuilt latent diffusion LightningModule."""

import math
import torch

from motion_latent_diffusion.modules import eval_metrics as M
from motion_latent_diffusion.modules.MotionLatentDiffusion import MotionLatentDiffusion
from motion_latent_diffusion.modules.preprocessing import _yaw_rotation
from .synthetic import make_synthetic_motion


# ---- eval metrics ----
def test_metrics_identity_zero():
    x = make_synthetic_motion(2, 16)
    assert M.mpjpe(x, x).item() == 0.0
    assert M.velocity_mpjpe(x, x).item() == 0.0
    assert M.pa_mpjpe(x, x).item() < 1e-4


def test_pa_mpjpe_rigid_invariant():
    x = make_synthetic_motion(1, 16, seed=2)
    R = _yaw_rotation(torch.tensor(0.7))
    moved = x @ R.T + torch.tensor([1.0, -2.0, 0.5])  # global rotation + translation
    assert M.mpjpe(moved, x).item() > 0.1            # raw error is large
    assert M.pa_mpjpe(moved, x).item() < 1e-3        # Procrustes removes it


def test_foot_skate_static_is_low():
    # A clip whose feet never move should have ~zero skating.
    x = make_synthetic_motion(1, 16, seed=3)
    feet = list(M.DEFAULT_FOOT_JOINTS)
    x[:, :, feet, :] = x[:, 0:1, feet, :]  # freeze feet across time
    assert M.foot_skate_ratio(x).item() < 1e-6


# ---- latent diffusion ----
LD_KW = dict(latent_dim=16, cond_dim=8, hidden_dim=128, time_embedding_dim=32,
             nhidden=3, timesteps=100, cfg_dropout=0.1, cfg_scale=2.0,
             sample_steps=30, clip_x0=3.0, lr=2e-3, dp_rate=0.0)


def _latent_batch(n=64, cond_dim=8, latent_dim=16, seed=0):
    """Two condition clusters -> two latent clusters (normalized)."""
    g = torch.Generator().manual_seed(seed)
    cls = torch.randint(0, 2, (n,), generator=g)
    # class 0 -> negative latent (cond -1); class 1 -> positive latent (cond +1)
    centers = torch.tensor([[-2.0], [2.0]]).repeat(1, latent_dim)
    z = centers[cls] + 0.3 * torch.randn(n, latent_dim, generator=g)
    z = (z - z.mean(0)) / z.std(0)  # standardize (mirrors StandardScaler)
    cond = torch.where(cls.unsqueeze(1).bool(),
                       torch.ones(n, cond_dim), -torch.ones(n, cond_dim))
    return z.float(), cond.float(), cls


def test_latent_diffusion_trains_and_samples():
    torch.manual_seed(0)
    m = MotionLatentDiffusion(**LD_KW).train()
    z, cond, cls = _latent_batch()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    first = last = None
    for step in range(400):
        loss = m._step((z, cond), "train")
        if step == 0:
            first = loss.item()
        last = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert last < 0.7 * first, f"diffusion loss did not drop: {first:.3f} -> {last:.3f}"

    m.eval()
    cond_pos = torch.ones(32, 8)
    out = m.sample(cond_pos, scale=2.0)
    assert out.shape == (32, 16) and torch.isfinite(out).all()


def test_latent_diffusion_conditioning_separates():
    """Samples conditioned on +1 vs -1 must land in different latent regions."""
    torch.manual_seed(0)
    m = MotionLatentDiffusion(**LD_KW).train()
    z, cond, cls = _latent_batch(n=128)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for _ in range(600):
        loss = m._step((z, cond), "train")
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    s_pos = m.sample(torch.ones(64, 8), scale=2.0).mean().item()
    s_neg = m.sample(-torch.ones(64, 8), scale=2.0).mean().item()
    assert s_pos > s_neg, f"conditioning did not separate clusters ({s_pos:.2f} vs {s_neg:.2f})"
