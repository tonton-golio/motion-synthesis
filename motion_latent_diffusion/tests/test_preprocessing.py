import torch

from motion_latent_diffusion.modules import preprocessing as P
from motion_latent_diffusion.modules.preprocessing import _yaw_rotation
from .synthetic import make_synthetic_motion


def test_remove_root_translation():
    m = make_synthetic_motion(1, 16)[0]  # (T,J,3)
    out = P.remove_root_translation(m)
    # frame-0 root XZ is now at origin; height preserved
    assert torch.allclose(out[0, P.ROOT_JOINT, [0, 2]], torch.zeros(2), atol=1e-6)
    assert torch.allclose(out[0, P.ROOT_JOINT, 1], m[0, P.ROOT_JOINT, 1])
    # relative geometry within a frame is unchanged (pure translation)
    assert torch.allclose(out[5] - out[5, P.ROOT_JOINT], m[5] - m[5, P.ROOT_JOINT], atol=1e-5)


def test_canonicalize_heading_is_yaw_invariant():
    m = make_synthetic_motion(1, 16, seed=4)[0]
    canon = P.canonicalize_heading(m)
    # Apply an arbitrary global yaw to the input; canonical form must be identical.
    R = _yaw_rotation(torch.tensor(1.1))
    m_rot = m @ R.T
    canon_rot = P.canonicalize_heading(m_rot)
    assert torch.allclose(canon, canon_rot, atol=1e-4), "heading not removed"


def test_normalize_roundtrip():
    motions = [make_synthetic_motion(1, 16, seed=s)[0] for s in range(5)]
    mean, std = P.compute_norm_stats(motions)
    assert mean.shape == (3,) and std.shape == (3,)
    m = motions[0]
    rt = P.denormalize(P.normalize(m, mean, std), mean, std)
    assert torch.allclose(rt, m, atol=1e-5)
    # normalized data is roughly standardized per axis across the corpus
    allnorm = torch.cat([P.normalize(m, mean, std).reshape(-1, 3) for m in motions])
    assert allnorm.mean(0).abs().max() < 0.2
    assert (allnorm.std(0) - 1.0).abs().max() < 0.2


def test_preprocess_clip():
    m = make_synthetic_motion(1, 10, seed=5)[0]  # T=10
    mean, std = P.compute_norm_stats([make_synthetic_motion(1, 10, seed=s)[0] for s in range(4)])
    padded, L = P.preprocess_clip(m, seq_len=16, mean=mean, std=std,
                                  canonicalize=True, normalize_data=True)
    assert padded.shape == (16, 22, 3)
    assert L == 10
    assert torch.allclose(padded[10:], torch.zeros_like(padded[10:]))  # zero pad
    # frame-0 root XZ at origin after root removal (pre-normalization invariant
    # only holds in raw space; here we just sanity-check finiteness)
    assert torch.isfinite(padded).all()


def test_pad_to_length():
    m = make_synthetic_motion(1, 10)[0]  # T=10
    padded, L = P.pad_to_length(m, 16)
    assert padded.shape == (16, 22, 3) and L == 10
    assert torch.allclose(padded[10:], torch.zeros_like(padded[10:]))  # zero pad, not edge
    assert torch.allclose(padded[:10], m)
    cropped, L2 = P.pad_to_length(m, 6)
    assert cropped.shape == (6, 22, 3) and L2 == 6
