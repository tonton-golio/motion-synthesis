"""Integration tests for the MotionVAE LightningModule on synthetic CPU data."""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from motion_latent_diffusion.modules.MotionVAE import MotionVAE
from .synthetic import make_synthetic_motion

T, B, D = 16, 4, 64
MODEL_KW = dict(
    seq_len=T, latent_dim=D, latent_size=1, nlayers_transformer=3,
    ff_transformer=128, nhead=4, dropout=0.0, learning_rate=1e-3,
    w_foot=0.0, kl_beta=1e-4, kl_warmup_epochs=0, kl_anneal_epochs=1,
)


def _batch(n=B, t=T, full=True):
    motion = make_synthetic_motion(n, t, seed=7)
    lengths = torch.full((n,), t if full else t - 3)
    text = torch.zeros(n, 512)
    ag = torch.zeros(n, dtype=torch.long)
    action = torch.zeros(n, dtype=torch.long)
    file_num = torch.arange(n)
    return (motion, lengths, text, ag, action, file_num)


def test_module_builds_and_step_runs():
    m = MotionVAE("VAEMLD", **MODEL_KW)
    res = m._common_step(_batch())
    assert torch.isfinite(res["total_loss"])
    for k in ("POS", "VEL", "FOOT", "KL"):
        assert k in res["losses_unscaled"]


def test_module_overfits_one_batch():
    torch.manual_seed(0)
    m = MotionVAE("VAEMLD", **MODEL_KW).train()
    batch = _batch()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    first = last = None
    for step in range(200):
        res = m._common_step(batch)
        if step == 0:
            first = res["losses_unscaled"]["POS"].item()
        last = res["losses_unscaled"]["POS"].item()
        opt.zero_grad()
        res["total_loss"].backward()
        opt.step()
    assert last < 0.3 * first, f"module did not learn: {first:.4f} -> {last:.4f}"


def test_trainer_fast_dev_run():
    """Full Lightning lifecycle (train/val/optim/logging) must not crash."""
    m = MotionVAE("VAEMLD", save_animations_freq=-1, **MODEL_KW)
    ds = TensorDataset(*_batch(n=8))
    dl = DataLoader(ds, batch_size=4)
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, fast_dev_run=True,
        enable_checkpointing=False, logger=False, enable_progress_bar=False,
    )
    trainer.fit(m, dl, dl)


def test_decode_wrapper_accepts_lengths():
    """The wrapper decode(z, lengths) must not crash (deep-review B04)."""
    m = MotionVAE("VAEMLD", **MODEL_KW).eval()
    motion = make_synthetic_motion(B, T)
    lengths = torch.full((B,), T)
    z = m.encode(motion, lengths)
    recon = m.decode(z, lengths)
    assert recon.shape == (B, T, 22, 3)
