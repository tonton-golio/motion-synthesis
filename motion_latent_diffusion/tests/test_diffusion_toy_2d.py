"""Toy 2D diffusion test — verifies the rebuilt DDPM/DDIM math independently of
the motion VAE by recovering a known distribution (a 4-Gaussian mixture).

Guards deep-review B08 (sampling must use the real reverse process, not iterate an
x0-regressor) and B10 (forward noise must be unit-variance; a noise_multiplier
would break variance preservation).
"""

import math
import torch
import torch.nn as nn

from motion_latent_diffusion.modules.diffusion import GaussianDiffusion

CENTERS = torch.tensor([[2.0, 2.0], [-2.0, 2.0], [2.0, -2.0], [-2.0, -2.0]])


def sample_mixture(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(CENTERS), (n,), generator=g)
    return CENTERS[idx] + 0.25 * torch.randn(n, 2, generator=g)


class ToyEps(nn.Module):
    def __init__(self, dim=2, hidden=128, t_dim=32, timesteps=200, n_classes=0):
        super().__init__()
        self.time_emb = nn.Embedding(timesteps, t_dim)
        self.cls_emb = nn.Embedding(n_classes + 1, t_dim) if n_classes else None
        in_dim = dim + t_dim + (t_dim if n_classes else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t, y=None):
        feats = [x, self.time_emb(t)]
        if self.cls_emb is not None:
            if y is None:  # unconditional -> null class index
                y = torch.full((x.shape[0],), self.cls_emb.num_embeddings - 1,
                               device=x.device, dtype=torch.long)
            feats.append(self.cls_emb(y))
        return self.net(torch.cat(feats, dim=-1))


def test_variance_preservation():
    """q_sample at the last timestep must have ~unit variance (no noise_multiplier)."""
    diff = GaussianDiffusion(timesteps=200, schedule="cosine")
    x0 = sample_mixture(4096)
    t = torch.full((4096,), 199, dtype=torch.long)
    x_t = diff.q_sample(x0, t, torch.randn_like(x0))
    assert abs(x_t.var().item() - 1.0) < 0.1, f"variance not preserved: {x_t.var().item():.3f}"


def test_ddim_recovers_mixture():
    torch.manual_seed(0)
    diff = GaussianDiffusion(timesteps=200, schedule="cosine")
    model = ToyEps(timesteps=200)
    raw = sample_mixture(8192, seed=1)
    # Diffusion expects ~unit-variance inputs (the real pipeline normalizes latents
    # with StandardScaler); train in normalized space and de-normalize samples.
    mean, std = raw.mean(0), raw.std(0)
    data = (raw - mean) / std
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    model.train()
    for step in range(2500):
        idx = torch.randint(0, data.shape[0], (256,))
        loss = diff.training_loss(lambda x_t, t: model(x_t, t), data[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    samples = diff.ddim_sample(lambda x_t, t: model(x_t, t), (2000, 2),
                               device="cpu", steps=50, eta=0.0, clip_x0=3.0)
    samples = samples * std + mean

    assert torch.isfinite(samples).all()
    assert (samples.mean(0) - raw.mean(0)).abs().max() < 0.35
    assert (samples.std(0) - raw.std(0)).abs().max() < 0.40
    # every one of the 4 known modes must be covered
    for c in CENTERS:
        near = ((samples - c).norm(dim=1) < 0.7).sum().item()
        assert near > 20, f"mode {c.tolist()} under-covered ({near} samples)"


def test_conditional_cfg_sharpens():
    """Classifier-free guidance must pull samples toward the conditioned mode."""
    torch.manual_seed(0)
    diff = GaussianDiffusion(timesteps=200, schedule="cosine")
    model = ToyEps(timesteps=200, n_classes=4)
    # class k lives at CENTERS[k]
    def sample_labeled(n, seed):
        g = torch.Generator().manual_seed(seed)
        y = torch.randint(0, 4, (n,), generator=g)
        x = CENTERS[y] + 0.25 * torch.randn(n, 2, generator=g)
        return x, y
    raw, labels = sample_labeled(8192, 1)
    mean, std = raw.mean(0), raw.std(0)
    data = (raw - mean) / std
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    model.train()
    for step in range(4000):
        idx = torch.randint(0, data.shape[0], (256,))
        xb, yb = data[idx], labels[idx].clone()
        # CFG training: drop the condition 10% of the time -> null class
        yb[torch.rand(256) < 0.1] = 4  # null index
        loss = diff.training_loss(lambda x_t, t: model(x_t, t, yb), xb)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    target = 0  # condition on class 0 -> CENTERS[0] = (2,2)
    n = 1000
    y_cond = torch.full((n,), target, dtype=torch.long)
    y_null = torch.full((n,), 4, dtype=torch.long)

    def guided(scale):
        def fn(x_t, t):
            ec = model(x_t, t, y_cond)
            eu = model(x_t, t, y_null)
            return eu + scale * (ec - eu)
        return fn

    def sample(scale):
        return diff.ddim_sample(guided(scale), (n, 2), device="cpu",
                                steps=50, eta=0.0, clip_x0=3.0) * std + mean

    d_uncond = (sample(0.0) - CENTERS[target]).norm(dim=1).mean().item()   # scale 0 = null
    s_cond = sample(1.0)                                                   # scale 1 = conditional
    d_cond = (s_cond - CENTERS[target]).norm(dim=1).mean().item()
    frac_mod = ((sample(2.0) - CENTERS[target]).norm(dim=1) < 0.7).float().mean().item()

    # Conditioning must steer sampling strongly toward the conditioned mode...
    assert d_cond < 0.5 * d_uncond, f"conditioning did not steer (uncond={d_uncond:.2f}, cond={d_cond:.2f})"
    # ...and moderate guidance (w=2) must keep most samples on-target.
    # (NB: very large guidance, e.g. w=3, over-extrapolates eps and is unstable —
    # a known CFG phenomenon; production sampling should clip x0 and use w~2-2.5.)
    assert frac_mod > 0.7, f"moderate-guidance samples drifted off the conditioned mode ({frac_mod:.2f})"
