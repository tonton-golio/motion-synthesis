"""Core correctness tests for the rewritten MotionVAE_MLD ("the compression").

The headline test is ``test_per_sample_independence`` — it is the assertion that
would have caught the original ``z.view(16, bs, 16)`` batch-scramble (B03) on day
one.
"""

import math
import torch

from motion_latent_diffusion.modules import loss as loss_mod
from .synthetic import make_synthetic_motion, small_model


def test_shapes():
    B, T, J, C, D, m = 4, 16, 22, 3, 64, 1
    model = small_model(latent_dim=D, latent_size=m, seq_len=T).eval()
    x = make_synthetic_motion(B, T)
    lengths = torch.full((B,), T)
    z, lengths_out, mu, logvar = model.encode(x, lengths)
    assert z.shape == (B, m, D)
    assert mu.shape == logvar.shape == (B, m, D)
    recon = model.decode(z, lengths)
    assert recon.shape == (B, T, J, C)
    assert torch.isfinite(z).all() and torch.isfinite(recon).all()


def test_latent_size_7():
    B, T, D, m = 3, 16, 64, 7
    model = small_model(latent_dim=D, latent_size=m, seq_len=T).eval()
    x = make_synthetic_motion(B, T)
    z, _, mu, logvar = model.encode(x)
    assert z.shape == mu.shape == logvar.shape == (B, m, D)
    assert model.decode(z).shape == (B, T, 22, 3)


def _encode_decode_det(model, x, lengths=None):
    """Deterministic round-trip: use mu (no reparam noise), dropout off."""
    z, lengths_out, mu, logvar = model.encode(x, lengths)
    return model.decode(mu, lengths if lengths is not None else lengths_out)


def test_per_sample_independence():
    """Permuting the batch order must permute the outputs identically, and a
    single sample decoded alone must equal its slice from the batched decode.

    With the old ``view(16, bs, 16)`` decode this FAILS (cross-sample mixing).
    """
    B, T, D = 4, 16, 64
    torch.manual_seed(0)
    model = small_model(latent_dim=D, seq_len=T).eval()
    x = make_synthetic_motion(B, T, seed=1)

    with torch.no_grad():
        out_full = _encode_decode_det(model, x)  # (B, T, 22, 3)

        perm = torch.tensor([2, 0, 3, 1])
        out_perm = _encode_decode_det(model, x[perm])
        assert torch.allclose(out_perm, out_full[perm], atol=1e-5), (
            "batch permutation changed per-sample outputs -> cross-sample leakage"
        )

        for b in range(B):
            out_single = _encode_decode_det(model, x[b:b + 1])
            assert torch.allclose(out_single, out_full[b:b + 1], atol=1e-5), (
                f"sample {b} decoded alone != its slice from the batched decode"
            )


def test_kl_batch_invariance():
    mu = torch.randn(8, 1, 32)
    logvar = torch.randn(8, 1, 32)
    kl8 = loss_mod.kl_divergence(mu, logvar)
    kl16 = loss_mod.kl_divergence(mu.repeat(2, 1, 1), logvar.repeat(2, 1, 1))
    assert torch.allclose(kl8, kl16, atol=1e-4), "KL must be batch-size invariant"
    # KL of N(mu,sigma) vs N(0,I) at mu=0, logvar=0 is exactly 0; in general >= 0.
    zero = loss_mod.kl_divergence(torch.zeros(4, 1, 16), torch.zeros(4, 1, 16))
    assert abs(zero.item()) < 1e-5
    assert kl8.item() >= 0.0


def test_gradient_flow():
    B, T, D = 4, 16, 64
    model = small_model(latent_dim=D, seq_len=T).train()
    crit = loss_mod.MotionVAELoss(w_foot=1.0)
    x = make_synthetic_motion(B, T)
    lengths = torch.full((B,), T)
    recon, z, mu, logvar = model(x, lengths)
    total, _, _ = crit(recon, x, mu, logvar, lengths, beta=1e-4)
    total.backward()

    named = dict(model.named_parameters())
    for name in [
        "skel_enc.weight",
        "global_motion_token",
        "final_layer.weight",
    ]:
        g = named[name].grad
        assert g is not None, f"no grad for {name}"
        assert torch.isfinite(g).all(), f"non-finite grad for {name}"
        assert g.abs().sum() > 0, f"zero grad for {name}"
    # The encoder distribution tokens must receive gradient from the recon loss,
    # proving the latent path is actually wired into reconstruction.
    assert named["global_motion_token"].grad.abs().sum() > 0


def test_overfit_one_batch():
    """The VAE must be able to represent a single batch of motion: reconstruction
    drops sharply while KL stays finite and non-collapsed.
    """
    B, T, D = 4, 16, 96
    torch.manual_seed(0)
    model = small_model(latent_dim=D, seq_len=T, nlayers=3).train()
    crit = loss_mod.MotionVAELoss(w_vel=0.5, w_foot=0.0)
    x = make_synthetic_motion(B, T, seed=2)
    lengths = torch.full((B,), T)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    pos_losses, kls = [], []
    for _ in range(300):
        recon, z, mu, logvar = model(x, lengths)
        total, _, unscaled = crit(recon, x, mu, logvar, lengths, beta=1e-4)
        pos_losses.append(unscaled["POS"].item())
        kls.append(unscaled["KL"].item())
        opt.zero_grad()
        total.backward()
        opt.step()

    assert pos_losses[-1] < 0.1 * pos_losses[0], (
        f"recon did not drop enough: {pos_losses[0]:.4f} -> {pos_losses[-1]:.4f}"
    )
    assert all(math.isfinite(k) for k in kls), "KL became non-finite"
    assert kls[-1] > 1e-3, "KL collapsed to ~0 (latent unused / posterior collapse)"
    assert kls[-1] < 1e4, "KL exploded"


def test_lengths_masking():
    """Garbage in the padded tail must not change the masked reconstruction loss."""
    B, T, L, D = 2, 16, 10, 64
    model = small_model(latent_dim=D, seq_len=T).eval()
    crit = loss_mod.MotionVAELoss(w_foot=0.0)
    gt = make_synthetic_motion(B, T, seed=3)
    lengths = torch.full((B,), L)
    with torch.no_grad():
        recon, z, mu, logvar = model(gt, lengths)

    recon_a = recon.clone()
    recon_b = recon.clone()
    recon_b[:, L:] = 999.0  # corrupt the padded region only
    la, _, _ = crit(recon_a, gt, mu, logvar, lengths, beta=0.0)
    lb, _, _ = crit(recon_b, gt, mu, logvar, lengths, beta=0.0)
    assert torch.allclose(la, lb, atol=1e-5), "padded frames leaked into the loss"
