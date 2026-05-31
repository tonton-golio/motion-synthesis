"""Lightweight reconstruction / generation metrics for 3D joint motion.

These give an objective signal without the full text-to-motion (Guo et al.)
evaluator: per-joint position error, Procrustes-aligned error, velocity error, and
a foot-skating ratio. Inputs are (B, T, J, 3) or (T, J, 3) tensors in metric space
(de-normalize first).
"""

import torch

from .loss import DEFAULT_FOOT_JOINTS


def _as_btj3(x):
    return x.unsqueeze(0) if x.dim() == 3 else x


def mpjpe(pred, gt):
    """Mean Per-Joint Position Error (mean L2 over joints, frames, batch)."""
    pred, gt = _as_btj3(pred), _as_btj3(gt)
    return (pred - gt).norm(dim=-1).mean()


def velocity_mpjpe(pred, gt):
    """MPJPE on temporal velocity (first difference along frames)."""
    pred, gt = _as_btj3(pred), _as_btj3(gt)
    return mpjpe(pred[:, 1:] - pred[:, :-1], gt[:, 1:] - gt[:, :-1])


def pa_mpjpe(pred, gt):
    """Procrustes-Aligned MPJPE: best rigid (rotation+translation+scale) fit of
    pred onto gt per sample, then MPJPE. Invariant to global pose/orientation.
    """
    pred, gt = _as_btj3(pred), _as_btj3(gt)
    B = pred.shape[0]
    out = []
    for b in range(B):
        P = pred[b].reshape(-1, 3)  # (N,3)
        G = gt[b].reshape(-1, 3)
        muP, muG = P.mean(0), G.mean(0)
        Pc, Gc = P - muP, G - muG
        # rotation via SVD of cross-covariance
        H = Pc.T @ Gc
        U, S, Vt = torch.linalg.svd(H)
        d = torch.sign(torch.det(Vt.T @ U.T))
        D = torch.diag(torch.tensor([1.0, 1.0, d], device=P.device, dtype=P.dtype))
        R = Vt.T @ D @ U.T
        # scale
        var_P = (Pc ** 2).sum()
        scale = (S * torch.tensor([1.0, 1.0, d], device=P.device, dtype=P.dtype)).sum() / (var_P + 1e-8)
        P_aligned = scale * (Pc @ R.T) + muG
        out.append((P_aligned - G).norm(dim=-1).mean())
    return torch.stack(out).mean()


def foot_skate_ratio(motion, feet=DEFAULT_FOOT_JOINTS, height_thresh=0.05, speed_thresh=0.01):
    """Fraction of foot-frames that are in ground contact (low height) yet moving.

    A lower ratio is better; static-foot ground contacts should not slide.
    """
    motion = _as_btj3(motion)
    feet = list(feet)
    foot = motion[:, :, feet, :]              # (B, T, F, 3)
    speed = (foot[:, 1:] - foot[:, :-1]).norm(dim=-1)  # (B, T-1, F)
    height = foot[:, 1:, :, 1]                # (B, T-1, F) y-coordinate
    height = height - motion[..., 1].amin()   # floor at the lowest point seen
    in_contact = height < height_thresh
    sliding = in_contact & (speed > speed_thresh)
    denom = in_contact.sum().clamp(min=1)
    return sliding.sum().float() / denom
