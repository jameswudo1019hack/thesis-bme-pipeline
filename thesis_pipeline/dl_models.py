"""Olsen 2020 BiGRU for per-second sleep-disordered-breathing detection (Aim 2 DL arm).

Architecture (Olsen et al., SLEEP 2020;43(5):zsz276, Table 2, C = 128, k = {1, 2}):

    input (B, 1200, c_in)                       5 min at 4 Hz
    SoftMinMax (per window, per channel)        q05 / q95, fp32, outside autocast
    block k = 1, 2:
        BiGRU(128) -> BiGRU(128)                 256-wide output
        MaxPool(2)                               time / 2
        BatchNorm(256)                           after the pool (declared adaptation A3)
        ReLU -> Dropout(0.5)
    head: Linear(256 -> 512) -> ReLU -> Linear(512 -> 1)
    output (B, 300) logits, one per second

Trainable parameters: 1,123,841 for c_in = 2 (each extra input channel adds 768).
The paper's 1,125,378 is this plus 1,024 BN running statistics plus a second output
unit (a 2-unit softmax is equivalent to one logit with BCE).

Declared adaptations (design section 1.2): PyTorch cuDNN-style GRU biases (A1), a
single-logit head with BCEWithLogits (A2), BN after MaxPool (A3), He-normal fan-in
initialisation for input-to-hidden and Linear weights with orthogonal hidden-to-hidden
weights and zero biases (A4).

``SoftMinMax`` is Olsen's per-segment soft min-max normalisation. It always runs in
fp32 with autocast disabled (``torch.nanquantile`` rejects fp16 inputs; we do not use
it anyway, see ``masked_quantiles``) and it excludes padding **by index** through a
``valid`` mask, never by value, so a zero-filled channel inside the night still counts.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

__all__ = ["masked_quantiles", "SoftMinMax", "OlsenBiGRU", "count_parameters"]


def masked_quantiles(
    x: torch.Tensor, valid: torch.Tensor, qs: Sequence[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantiles of ``x`` along time over the ``valid`` samples only.

    Parameters
    ----------
    x : (B, T, C) float tensor
    valid : (B, T) bool tensor; True for samples inside the recording
    qs : quantile levels in [0, 1]

    Returns
    -------
    (Q, n_valid) where Q is (len(qs), B, C) and n_valid is (B,). Uses the same
    'linear' interpolation as ``torch.quantile`` (position q * (n - 1)). Windows
    with no valid sample return 0. Implemented with a sort so it works on CPU,
    CUDA and MPS without ``nanquantile``.
    """
    if x.dim() != 3:
        raise ValueError(f"x must be (B, T, C); got {tuple(x.shape)}")
    B, T, C = x.shape
    valid = valid.to(torch.bool)
    inf = torch.full((), float("inf"), dtype=x.dtype, device=x.device)
    xs, _ = torch.sort(torch.where(valid.unsqueeze(-1), x, inf), dim=1)
    n = valid.sum(dim=1)  # (B,)
    last = (n - 1).clamp(min=0)
    out = []
    for q in qs:
        pos = float(q) * last.to(x.dtype)
        lo = torch.floor(pos)
        frac = (pos - lo).view(B, 1)
        lo_i = lo.long()
        hi_i = torch.minimum(lo_i + 1, last)
        v_lo = torch.gather(xs, 1, lo_i.view(B, 1, 1).expand(B, 1, C)).squeeze(1)
        v_hi = torch.gather(xs, 1, hi_i.view(B, 1, 1).expand(B, 1, C)).squeeze(1)
        diff = torch.where(hi_i.view(B, 1) == lo_i.view(B, 1), torch.zeros_like(v_lo), v_hi - v_lo)
        out.append(v_lo + diff * frac)
    Q = torch.stack(out)
    Q = torch.where((n > 0).view(1, B, 1), Q, torch.zeros_like(Q))
    return Q, n


class SoftMinMax(nn.Module):
    """Per-window, per-channel (x - q_lo) / (q_hi - q_lo) over non-padding samples.

    * fp32 always, autocast disabled inside.
    * ``valid`` (B, T) marks samples inside the recording; padding is excluded from
      the quantiles and set to 0 in the output.
    * A channel with q_hi - q_lo < ``eps`` in a window is set to 0 (flat or missing).
    * Channels in ``skip_channels`` (SpO2, which has a fixed affine in stage B) are
      passed through unchanged except that padding is zeroed.
    * No clipping (Olsen states none).
    """

    def __init__(
        self,
        q_lo: float = 0.05,
        q_hi: float = 0.95,
        eps: float = 1e-6,
        skip_channels: Sequence[int] = (),
    ) -> None:
        super().__init__()
        if not 0.0 <= q_lo < q_hi <= 1.0:
            raise ValueError("need 0 <= q_lo < q_hi <= 1")
        self.q_lo = float(q_lo)
        self.q_hi = float(q_hi)
        self.eps = float(eps)
        self.skip_channels = tuple(int(c) for c in skip_channels)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            B, T, C = x.shape
            if valid is None:
                valid = torch.ones(B, T, dtype=torch.bool, device=x.device)
            valid = valid.to(device=x.device, dtype=torch.bool)
            Q, _ = masked_quantiles(x, valid, (self.q_lo, self.q_hi))
            lo, hi = Q[0], Q[1]
            rng = hi - lo
            flat = rng < self.eps
            denom = torch.where(flat, torch.ones_like(rng), rng)
            y = (x - lo.unsqueeze(1)) / denom.unsqueeze(1)
            y = torch.where(flat.unsqueeze(1), torch.zeros_like(y), y)
            if self.skip_channels:
                keep = torch.zeros(C, dtype=torch.bool, device=x.device)
                keep[list(self.skip_channels)] = True
                y = torch.where(keep.view(1, 1, C), x, y)
            y = y * valid.unsqueeze(-1).to(y.dtype)
        return y

    def extra_repr(self) -> str:
        return f"q_lo={self.q_lo}, q_hi={self.q_hi}, eps={self.eps}, skip={self.skip_channels}"


class _Block(nn.Module):
    """BiGRU x n_gru -> MaxPool(2) -> BatchNorm -> ReLU -> Dropout."""

    def __init__(self, d_in: int, hidden: int, n_gru: int, p_drop: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            d_in, hidden, num_layers=n_gru, bidirectional=True, batch_first=True
        )
        self.pool = nn.MaxPool1d(2)
        self.bn = nn.BatchNorm1d(2 * hidden)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        h, _ = self.gru(x)
        h = h.transpose(1, 2)  # (B, 2H, T)
        h = self.pool(h)
        h = self.bn(h)
        h = torch.relu(h)
        h = self.drop(h)
        return h.transpose(1, 2)  # (B, T/2, 2H)


class OlsenBiGRU(nn.Module):
    """Olsen 2020 Table 2 model. ``forward(x, valid) -> logits (B, T // 4)``.

    Parameters
    ----------
    c_in : number of input channels (2 for M = [RR, EDR], 4 for P4)
    hidden : GRU units per direction (128 in the paper; smaller only in unit tests)
    n_gru : BiGRU layers per block (2)
    n_blocks : blocks (2)
    p_drop : dropout (0.5)
    skip_norm_channels : channel indices that bypass SoftMinMax (SpO2)
    """

    def __init__(
        self,
        c_in: int = 2,
        hidden: int = 128,
        n_gru: int = 2,
        n_blocks: int = 2,
        p_drop: float = 0.5,
        skip_norm_channels: Sequence[int] = (),
        normalise: bool = True,
    ) -> None:
        super().__init__()
        self.c_in = int(c_in)
        self.hidden = int(hidden)
        self.n_blocks = int(n_blocks)
        self.norm = SoftMinMax(skip_channels=skip_norm_channels) if normalise else None
        blocks = []
        d = self.c_in
        for _ in range(n_blocks):
            blocks.append(_Block(d, hidden, n_gru, p_drop))
            d = 2 * hidden
        self.blocks = nn.ModuleList(blocks)
        self.fc1 = nn.Linear(2 * hidden, 4 * hidden)
        self.fc2 = nn.Linear(4 * hidden, 1)
        self.reset_parameters()

    @property
    def downsample(self) -> int:
        return 2 ** self.n_blocks

    def reset_parameters(self) -> None:
        """He-normal (fan_in) input and Linear weights, orthogonal recurrent weights
        per gate, zero biases (declared adaptation A4)."""
        for name, p in self.named_parameters():
            if "bn" in name:
                continue  # BatchNorm keeps weight = 1, bias = 0
            if "bias" in name:
                nn.init.zeros_(p)
            elif "weight_hh" in name:
                h = p.shape[1]
                for g in range(p.shape[0] // h):
                    nn.init.orthogonal_(p.data[g * h:(g + 1) * h])
            elif "weight" in name:
                nn.init.kaiming_normal_(p, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[-1] != self.c_in:
            raise ValueError(f"expected {self.c_in} channels, got {x.shape[-1]}")
        if self.norm is not None:
            x = self.norm(x, valid)
        for b in self.blocks:
            x = b(x)
        h = torch.relu(self.fc1(x))
        return self.fc2(h).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count (excludes BN running statistics)."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
