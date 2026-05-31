"""
MotionVAE_MLD — a correct, ACTOR/MLD-style transformer VAE for motion compression.

This replaces the VAE1/VAE4/VAE5/VAE6 cascade in MotionVAE.py, whose `decode`
reshaped the latent with `z.view(16, bs, 16)` and silently scrambled samples
across the batch axis (see the deep-review bug B03-view-scramble).

Design (see the project improvement plan, section 2):
  * Encoder: skeleton-embed each frame, prepend `2*latent_size` learnable
    distribution tokens (ACTOR), run a Transformer encoder with frame padding
    masked out, and read mu/logvar ONLY from the distribution-token outputs.
  * Decoder: cross-attend `seq_len` positional query tokens to the latent
    tokens (Perceiver-IO / MLD style), then project to joint coordinates.

Invariants that make this correct, asserted by tests/test_per_sample_independence.py:
  * The batch axis is always the leading axis and is never folded into another
    axis by any reshape. The only `reshape`/`view` calls touch the *feature*
    axis (66 -> 22x3) or are no-ops, never the batch or time axes.
  * decode(z) for a single sample equals the corresponding slice of decode(z)
    for a batch containing that sample.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Union


def lengths_to_mask(lengths: Union[List[int], torch.Tensor],
                    device: torch.device,
                    max_len: Optional[int] = None) -> torch.Tensor:
    """Boolean mask (B, max_len) where True marks *valid* (non-padded) frames."""
    if not torch.is_tensor(lengths):
        lengths = torch.as_tensor(lengths, device=device)
    lengths = lengths.to(device=device)
    max_len = int(max_len) if max_len is not None else int(lengths.max().item())
    rng = torch.arange(max_len, device=device).expand(len(lengths), max_len)
    return rng < lengths.unsqueeze(1)


class PositionalEncoding(nn.Module):
    """Standard additive sinusoidal positional encoding, batch_first (B, S, D)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, d_model)
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class MotionVAE_MLD(nn.Module):
    """Transformer VAE compressing (B, T, njoints, 3) motion to (B, latent_size, latent_dim).

    For ``latent_size == 1`` the latent is (B, 1, latent_dim); squeeze axis 1
    (with ``z.squeeze(1)``, never argument-less ``squeeze()``) at the boundary to
    a flat diffusion model.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        latent_size: int = 1,
        seq_len: int = 160,
        input_dim: int = 66,
        njoints: int = 22,
        nhead: int = 4,
        ff_transformer: int = 1024,
        nlayers_transformer: int = 9,
        dropout: float = 0.1,
        transformer_activation: str = "gelu",
        **kwargs,
    ):
        super().__init__()
        self.verbose = False
        self.latent_dim = latent_dim
        self.latent_size = latent_size
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.njoints = njoints
        assert input_dim == njoints * 3, (
            f"input_dim ({input_dim}) must equal njoints*3 ({njoints * 3}) for the "
            "xyz representation; use a custom head for the 263-d HumanML3D vector."
        )

        # ---- Encoder ----
        self.skel_enc = nn.Linear(input_dim, latent_dim)
        self.global_motion_token = nn.Parameter(
            torch.randn(2 * latent_size, latent_dim)
        )
        self.encoder_pos = PositionalEncoding(
            latent_dim, max_len=seq_len + 2 * latent_size + 1
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=ff_transformer,
            dropout=dropout,
            activation=transformer_activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=nlayers_transformer,
            norm=nn.LayerNorm(latent_dim),
        )

        # ---- Decoder ----
        self.decoder_pos = PositionalEncoding(latent_dim, max_len=seq_len + 1)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=ff_transformer,
            dropout=dropout,
            activation=transformer_activation,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=nlayers_transformer,
            norm=nn.LayerNorm(latent_dim),
        )
        self.final_layer = nn.Linear(latent_dim, input_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _resolve_lengths(self, lengths, batch_size, device):
        if lengths is None:
            return torch.full((batch_size,), self.seq_len, device=device, dtype=torch.long)
        if not torch.is_tensor(lengths):
            lengths = torch.as_tensor(lengths, device=device)
        return lengths.to(device=device).long()

    def encode(self, x: torch.Tensor, lengths=None):
        """x: (B, T, njoints, 3) or (B, T, input_dim) -> z, lengths, mu, logvar.

        z, mu, logvar have shape (B, latent_size, latent_dim).
        """
        batch_size = x.shape[0]
        # Reshape touches ONLY the feature axis (njoints*3 -> input_dim). Batch & time untouched.
        x = x.reshape(batch_size, self.seq_len, self.input_dim)
        device = x.device
        lengths = self._resolve_lengths(lengths, batch_size, device)

        h = self.skel_enc(x)  # (B, T, D)
        tokens = self.global_motion_token.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 2m, D)
        xseq = torch.cat([tokens, h], dim=1)  # (B, 2m + T, D)
        xseq = self.encoder_pos(xseq)

        frame_valid = lengths_to_mask(lengths, device, self.seq_len)  # (B, T) True=valid
        token_valid = torch.ones(
            batch_size, 2 * self.latent_size, dtype=torch.bool, device=device
        )
        key_padding_mask = ~torch.cat([token_valid, frame_valid], dim=1)  # True=pad/ignore

        enc = self.encoder(xseq, src_key_padding_mask=key_padding_mask)
        dist = enc[:, : 2 * self.latent_size, :]  # (B, 2m, D) — only the token outputs
        mu = dist[:, : self.latent_size, :]
        logvar = dist[:, self.latent_size:, :]
        z = self.reparametrize(mu, logvar)
        return z, lengths, mu, logvar

    def decode(self, z: torch.Tensor, lengths=None) -> torch.Tensor:
        """z: (B, latent_size, latent_dim) -> (B, T, njoints, 3)."""
        batch_size = z.shape[0]
        device = z.device
        lengths = self._resolve_lengths(lengths, batch_size, device)
        frame_valid = lengths_to_mask(lengths, device, self.seq_len)  # (B, T)

        queries = torch.zeros(batch_size, self.seq_len, self.latent_dim, device=device)
        queries = self.decoder_pos(queries)  # learnable-free positional query tokens
        out = self.decoder(
            tgt=queries,
            memory=z,
            tgt_key_padding_mask=~frame_valid,
        )  # (B, T, D)
        out = self.final_layer(out)  # (B, T, input_dim)
        # Zero padded frames (non-inplace, autograd-safe — unlike `out[~mask] = 0`).
        out = out * frame_valid.unsqueeze(-1).to(out.dtype)
        feats = out.reshape(batch_size, self.seq_len, self.njoints, 3)
        return feats

    def forward(self, x: torch.Tensor, lengths=None):
        z, lengths, mu, logvar = self.encode(x, lengths)
        recon = self.decode(z, lengths)
        return recon, z, mu, logvar
