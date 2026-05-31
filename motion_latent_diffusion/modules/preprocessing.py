"""Motion preprocessing: root-translation removal, heading canonicalization,
per-axis normalization, and length-aware padding.

These address deep-review bugs B14 (no normalization; unbounded root translation
dominates the loss) and B18 (edge-padding replicates the last frame, biasing the
model to "freeze"). All functions are pure and operate on torch tensors so they
can be unit-tested on synthetic data without the HumanML3D dataset.

Joint layout (22-joint HumanML3D/SMPL): 0 = pelvis (root), 1/2 = left/right hip,
16/17 = left/right shoulder, 7/8 = ankles, 10/11 = feet.
"""

import torch

ROOT_JOINT = 0
LEFT_HIP, RIGHT_HIP = 1, 2
LEFT_SHOULDER, RIGHT_SHOULDER = 16, 17


def remove_root_translation(motion: torch.Tensor) -> torch.Tensor:
    """Subtract the frame-0 root XZ from every joint/frame (keep height Y).

    motion: (T, J, 3) or (B, T, J, 3). Removes unbounded global ground placement
    so it cannot dominate the reconstruction gradient, while preserving vertical
    position and all relative motion.
    """
    if motion.dim() == 3:
        offset = motion[0:1, ROOT_JOINT:ROOT_JOINT + 1, :].clone()  # (1,1,3)
    elif motion.dim() == 4:
        offset = motion[:, 0:1, ROOT_JOINT:ROOT_JOINT + 1, :].clone()  # (B,1,1,3)
    else:
        raise ValueError("motion must be (T,J,3) or (B,T,J,3)")
    offset[..., 1] = 0.0  # keep height
    return motion - offset


def _yaw_rotation(angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrix about the vertical (Y) axis for angle (radians)."""
    c, s = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    # rotate the XZ plane; Y unchanged
    return torch.stack([
        torch.stack([c, zero, s], dim=-1),
        torch.stack([zero, one, zero], dim=-1),
        torch.stack([-s, zero, c], dim=-1),
    ], dim=-2)  # (..., 3, 3)


def _facing_angle(frame: torch.Tensor) -> torch.Tensor:
    """Yaw angle of the body's forward direction at a single frame.

    frame: (J, 3). Forward is cross(up, across-hips+across-shoulders), projected
    to the ground plane; returns atan2(forward_x, forward_z).
    """
    across = (frame[RIGHT_HIP] - frame[LEFT_HIP]) + (frame[RIGHT_SHOULDER] - frame[LEFT_SHOULDER])
    across = across / (across.norm() + 1e-8)
    up = torch.tensor([0.0, 1.0, 0.0], device=frame.device, dtype=frame.dtype)
    forward = torch.linalg.cross(across, up)  # in the XZ plane
    return torch.atan2(forward[0], forward[2] + 1e-8)


def canonicalize_heading(motion: torch.Tensor) -> torch.Tensor:
    """Rotate the whole clip about Y so frame 0 faces a canonical direction.

    motion: (T, J, 3). Removes arbitrary global heading (HumanML3D normally does
    this), making the VAE's job heading-invariant.
    """
    assert motion.dim() == 3, "canonicalize_heading expects a single clip (T,J,3)"
    angle = -_facing_angle(motion[0])  # rotate facing back to 0
    R = _yaw_rotation(angle)  # (3,3)
    return motion @ R.T


def compute_norm_stats(motions):
    """Per-axis (x, y, z) mean and std over all frames/joints of all clips.

    motions: iterable of (T, J, 3) tensors (already root-removed / canonicalized).
    Returns (mean (3,), std (3,)). Per-axis (NOT per-of-66-dims) so limb geometry
    is preserved.
    """
    flat = torch.cat([m.reshape(-1, 3) for m in motions], dim=0)  # (N, 3)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp(min=1e-6)
    return mean, std


def normalize(motion: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (motion - mean) / std


def denormalize(motion: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return motion * std + mean


def preprocess_clip(motion: torch.Tensor, seq_len: int, mean=None, std=None,
                    canonicalize: bool = True, normalize_data: bool = True):
    """Full per-clip pipeline: root-removal -> heading canon -> normalize -> zero-pad.

    Returns (padded (seq_len, J, 3), true_length). Normalization is applied to the
    valid frames before zero-padding so padded frames are exact zeros (and are
    excluded from the loss via the returned length).
    """
    motion = remove_root_translation(motion)
    if canonicalize:
        motion = canonicalize_heading(motion)
    if normalize_data and mean is not None and std is not None:
        motion = normalize(motion, mean, std)
    return pad_to_length(motion, seq_len)


def pad_to_length(motion: torch.Tensor, length: int):
    """Zero-pad or crop a clip to ``length`` frames; return (padded, true_length).

    Unlike edge-padding (which replicates the last frame and teaches the model to
    freeze), padded frames are zeros and excluded from the loss via the returned
    true length.
    """
    T = motion.shape[0]
    if T >= length:
        return motion[:length], length
    pad = torch.zeros((length - T,) + tuple(motion.shape[1:]),
                      dtype=motion.dtype, device=motion.device)
    return torch.cat([motion, pad], dim=0), T
