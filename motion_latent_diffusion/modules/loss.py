"""Loss functions for the motion / pose VAEs.

Key corrections over the original (see deep-review bugs B01, B06):
  * ``kl_divergence`` is now the standard per-sample KL: sum over latent dims,
    MEAN over the batch. It is therefore batch-size invariant, so the KL weight
    (beta) is interpretable and no longer a ~1e-8 fudge factor compensating for a
    bare ``torch.sum`` over batch*latent.
  * ``BatchSequenceMSELoss`` handles (B, T, ...) tensors and masks padded frames
    using true sequence lengths, and is actually reachable from the new motion
    loss path.
  * ``MotionVAELoss`` implements four orthogonal, mean-reduced terms in
    normalized space (position, velocity, foot-contact, KL) with a runtime
    ``beta`` for KL warm-up/annealing.
"""

import torch
from torch import nn
from torch.nn import functional as F


# Default foot joint indices for the 22-joint HumanML3D / SMPL-style skeleton
# (left/right ankle and foot). Used by the foot-skate regularizer.
DEFAULT_FOOT_JOINTS = (7, 8, 10, 11)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    """Standard Gaussian KL D(q(z|x) || N(0, I)), per-sample, mean over batch.

    mu, logvar: (B, ...latent dims...). Returns a scalar that does NOT scale with
    batch size or latent dimensionality count beyond the genuine per-sample KL.
    """
    kld = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # (B, ...)
    if free_bits > 0.0:
        # Floor each latent dimension so no single dimension can fully collapse.
        kld = torch.clamp(kld, min=free_bits)
    kld = kld.flatten(start_dim=1).sum(dim=1)  # (B,) sum over latent dims
    return kld.mean()  # mean over batch -> batch-invariant scalar


def _masked_frame_mean(per_frame: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of a per-frame loss (B, T') over valid frames, then over the batch.

    per_frame: (B, T') already reduced over feature dims.
    mask:      (B, T') bool, True = valid frame.
    """
    mask = mask.to(per_frame.dtype)
    valid = mask.sum(dim=1).clamp(min=1.0)  # (B,) avoid div-by-zero
    loss_per_seq = (per_frame * mask).sum(dim=1) / valid  # (B,)
    return loss_per_seq.mean()


class BatchSequenceMSELoss(nn.Module):
    """Length-masked reconstruction loss over (B, T, ...) tensors.

    ``kind`` selects 'mse' or 'smooth_l1' (Huber); MoCap joints are spiky so
    smooth_l1 is the default for the motion VAE.
    """

    def __init__(self, kind: str = "smooth_l1", beta: float = 0.1):
        super().__init__()
        self.kind = kind
        self.beta = beta

    def _elementwise(self, preds, trues):
        if self.kind == "mse":
            return F.mse_loss(preds, trues, reduction="none")
        if self.kind == "smooth_l1":
            return F.smooth_l1_loss(preds, trues, reduction="none", beta=self.beta)
        if self.kind == "l1":
            return F.l1_loss(preds, trues, reduction="none")
        raise ValueError(f"unknown reconstruction kind '{self.kind}'")

    def forward(self, preds, trues, lengths=None):
        # preds, trues: (B, T, ...). Reduce over all feature dims -> (B, T).
        err = self._elementwise(preds, trues)
        feat_dims = tuple(range(2, err.dim()))
        per_frame = err.sum(dim=feat_dims) if feat_dims else err  # (B, T)

        if lengths is None:
            return per_frame.mean()

        device = per_frame.device
        T = per_frame.size(1)
        if not torch.is_tensor(lengths):
            lengths = torch.as_tensor(lengths, device=device)
        lengths = lengths.to(device=device)
        mask = torch.arange(T, device=device).expand(per_frame.size(0), T) < lengths.unsqueeze(1)
        return _masked_frame_mean(per_frame, mask)


def velocity(x: torch.Tensor) -> torch.Tensor:
    """First temporal difference along the frame axis (dim=1)."""
    return x[:, 1:] - x[:, :-1]


def foot_skate_loss(recon, gt, lengths=None, feet=DEFAULT_FOOT_JOINTS,
                    contact_speed: float = 0.02) -> torch.Tensor:
    """Penalize predicted foot velocity where the GT foot is (near-)static.

    recon, gt: (B, T, J, 3). A foot is "in contact" at frame t when its GT speed
    is below ``contact_speed`` (in the same units as the normalized data). We
    penalize the predicted foot speed at those frames (classic foot-skating).
    """
    feet = list(feet)
    gt_v = velocity(gt[:, :, feet, :])      # (B, T-1, F, 3)
    rec_v = velocity(recon[:, :, feet, :])  # (B, T-1, F, 3)
    gt_speed = gt_v.norm(dim=-1)            # (B, T-1, F)
    contact = (gt_speed < contact_speed).to(rec_v.dtype)  # (B, T-1, F)
    skate = (rec_v.norm(dim=-1) * contact)  # (B, T-1, F)
    per_frame = skate.sum(dim=-1)           # (B, T-1) sum over feet
    if lengths is None:
        return per_frame.mean()
    device = per_frame.device
    Tm1 = per_frame.size(1)
    lengths = torch.as_tensor(lengths, device=device) if not torch.is_tensor(lengths) else lengths.to(device)
    # velocity at t is valid only if frame t and t+1 are both valid -> need length-1
    vmask = torch.arange(Tm1, device=device).expand(per_frame.size(0), Tm1) < (lengths.unsqueeze(1) - 1)
    return _masked_frame_mean(per_frame, vmask)


class MotionVAELoss(nn.Module):
    """Four-term motion VAE loss in normalized space.

    total = w_pos*L_pos + w_vel*L_vel + w_foot*L_foot + beta*L_kl

    ``beta`` is supplied per-call to support KL warm-up / annealing.
    """

    def __init__(self, w_pos=1.0, w_vel=0.5, w_foot=1.0, free_bits=0.0,
                 recon_kind="smooth_l1", smooth_beta=0.1, feet=DEFAULT_FOOT_JOINTS):
        super().__init__()
        self.w_pos = w_pos
        self.w_vel = w_vel
        self.w_foot = w_foot
        self.free_bits = free_bits
        self.feet = feet
        self.recon = BatchSequenceMSELoss(kind=recon_kind, beta=smooth_beta)

    def forward(self, recon, gt, mu, logvar, lengths=None, beta=1.0):
        l_pos = self.recon(recon, gt, lengths)
        l_vel = self.recon(velocity(recon), velocity(gt),
                           (lengths - 1) if lengths is not None else None)
        l_foot = (foot_skate_loss(recon, gt, lengths, self.feet)
                  if self.w_foot > 0 else recon.new_zeros(()))
        l_kl = kl_divergence(mu, logvar, self.free_bits)

        unscaled = {"POS": l_pos, "VEL": l_vel, "FOOT": l_foot, "KL": l_kl}
        scaled = {
            "POS": self.w_pos * l_pos,
            "VEL": self.w_vel * l_vel,
            "FOOT": self.w_foot * l_foot,
            "KL": beta * l_kl,
        }
        total = scaled["POS"] + scaled["VEL"] + scaled["FOOT"] + scaled["KL"]
        return total, scaled, unscaled


class VAE_Loss(nn.Module):
    """Generic weighted-term VAE loss (kept for the Pose/MNIST callers).

    Only the KL term math is corrected here (batch-invariant); the weighted-dict
    interface is preserved for backward compatibility.
    """

    def __init__(self, loss_weights):
        super().__init__()
        self.loss_weights = loss_weights
        self.sequence_mse_loss = BatchSequenceMSELoss(kind="mse")

    def forward(self, loss_data):
        total_loss = 0.0
        losses_unscaled = {}
        losses_scaled = {}
        for k, v in loss_data.items():
            name, method = k.rsplit("_", 1)
            lengths = v["lengths"] if "lengths" in v else None
            weight = self.loss_weights.get(k, 0)
            if weight == 0:
                continue

            if method == "L2":
                if lengths is not None:
                    losses_unscaled[k] = self.sequence_mse_loss(v["rec"], v["true"], lengths)
                else:
                    losses_unscaled[k] = F.mse_loss(v["rec"], v["true"], reduction="mean")
            elif method == "L1":
                losses_unscaled[k] = F.l1_loss(v["rec"], v["true"], reduction="mean")
            elif method == "KL":
                losses_unscaled[k] = kl_divergence(v["mu"], v["logvar"])
            elif method == "BCE":
                losses_unscaled[k] = F.binary_cross_entropy(v["rec"], v["true"], reduction="mean")
            else:
                raise ValueError(f"Invalid loss method '{method}' for component '{k}'.")

            losses_scaled[k] = weight * losses_unscaled[k]
            total_loss = total_loss + losses_scaled[k]

        return total_loss, losses_scaled, losses_unscaled

    def kl_divergence(self, mu, logvar):  # kept as a method for callers that use it directly
        return kl_divergence(mu, logvar)
