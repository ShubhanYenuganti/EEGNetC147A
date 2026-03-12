"""
Lightweight EEG Transformer with Spatial-Temporal Attention
===========================================================

Architecture for BCI Competition IV Dataset 2a:
  - 22 channels, 1000 timepoints (4s @ 250Hz), 4 classes

Design philosophy:
  CNN front-end (inspired by EEGNet) extracts local features, then separate
  spatial and temporal attention mechanisms capture which channels (electrodes)
  and which time points matter most for classification.

Pipeline:
  1. Temporal Conv      → learn frequency-specific filters across time
  2. Separable Conv     → refine combined spatial-temporal features
  3. Spatial Attention   → multi-head attention over channels (which electrodes?)
  4. Temporal Attention  → multi-head attention over time (which moments?)
  5. Classifier         → linear projection to class logits

v2 changes (tuned for 201-trial small data regime):
  - d_model: 32 → 16        (reduce overfitting)
  - spatial/temporal_heads: 4 → 2  (proportional to d_model)
  - ff_dim: 64 → 32         (smaller feedforward)
  - trans_dropout: 0.3 → 0.5  (much heavier — model was overfitting by epoch 25)
  - fc_dropout: 0.5 → 0.7     (heavier classifier regularisation)
  - cnn_dropout: 0.4 → 0.5    (heavier CNN regularisation)
  - label_smoothing in loss    (handled in run script, not model)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding — sinusoidal, no learned parameters
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding from Vaswani et al. (2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Multi-head attention with extractable weights
# ---------------------------------------------------------------------------
class MultiHeadAttentionWithWeights(nn.Module):
    """Standard multi-head attention that returns attention weights."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = query.shape

        Q = self.W_q(query).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)

        output = self.W_o(context)
        return output, attn_weights


# ---------------------------------------------------------------------------
# Transformer encoder block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Single transformer encoder block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttentionWithWeights(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, attn_weights


# ---------------------------------------------------------------------------
# Lightweight EEG Transformer
# ---------------------------------------------------------------------------
class EEGTransformer(nn.Module):
    """
    Lightweight Transformer with separate spatial and temporal attention
    for EEG motor imagery classification.

    Args:
        n_classes       : number of output classes (default: 4)
        n_channels      : number of EEG channels (default: 22 for BCI-IV-2a)
        n_timepoints    : number of time samples (default: 1000 = 4s @ 250Hz)
        sfreq           : sampling frequency in Hz (default: 250)
        F1              : number of temporal filters (default: 8)
        D               : depth multiplier for depthwise conv (default: 2)
        cnn_dropout     : dropout after CNN blocks (default: 0.5)
        d_model         : transformer model dimension (default: 16)
        spatial_heads   : attention heads for spatial transformer (default: 2)
        temporal_heads  : attention heads for temporal transformer (default: 2)
        spatial_layers  : number of spatial transformer blocks (default: 1)
        temporal_layers : number of temporal transformer blocks (default: 1)
        ff_dim          : feedforward hidden dimension (default: 32)
        trans_dropout   : dropout in transformer blocks (default: 0.5)
        fc_dropout      : dropout before classifier (default: 0.7)
    """

    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        n_timepoints: int = 1000,
        sfreq: int = 250,
        # CNN hyper-parameters
        F1: int = 8,
        D: int = 2,
        cnn_dropout: float = 0.5,
        # Transformer hyper-parameters (v2: smaller, heavier dropout)
        d_model: int = 16,
        spatial_heads: int = 2,
        temporal_heads: int = 2,
        spatial_layers: int = 1,
        temporal_layers: int = 1,
        ff_dim: int = 32,
        trans_dropout: float = 0.5,
        # Classifier
        fc_dropout: float = 0.7,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.d_model = d_model

        F2 = F1 * D

        # ── CNN Block 1: temporal filtering (preserve spatial dim) ───────
        temporal_kernel = sfreq // 2
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                1, F1,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(cnn_dropout),
        )

        # ── CNN Block 2: separable temporal convolution ──────────────────
        sep_kernel = sfreq // 8
        self.sep_conv = nn.Sequential(
            nn.Conv2d(
                F1, F2,
                kernel_size=(1, sep_kernel),
                padding=(0, sep_kernel // 2),
                groups=F1,
                bias=False,
            ),
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(cnn_dropout),
        )

        # ── Compute CNN output shape ─────────────────────────────────────
        cnn_channels, cnn_spatial, cnn_temporal = self._cnn_output_shape()

        # ── Spatial projection ───────────────────────────────────────────
        self.spatial_proj = nn.Linear(cnn_channels * cnn_temporal, d_model)
        self.spatial_pos = SinusoidalPositionalEncoding(d_model, max_len=n_channels, dropout=trans_dropout)

        # ── Spatial Transformer ──────────────────────────────────────────
        self.spatial_transformer = nn.ModuleList([
            TransformerBlock(d_model, spatial_heads, ff_dim, trans_dropout)
            for _ in range(spatial_layers)
        ])

        # ── Temporal projection ──────────────────────────────────────────
        self.temporal_proj = nn.Linear(cnn_channels * cnn_spatial, d_model)
        self.temporal_pos = SinusoidalPositionalEncoding(d_model, max_len=512, dropout=trans_dropout)

        # ── Temporal Transformer ─────────────────────────────────────────
        self.temporal_transformer = nn.ModuleList([
            TransformerBlock(d_model, temporal_heads, ff_dim, trans_dropout)
            for _ in range(temporal_layers)
        ])

        # ── Classifier head ──────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(fc_dropout),
            nn.Linear(d_model * 2, n_classes),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _cnn_output_shape(self) -> tuple[int, int, int]:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_timepoints)
            out = self.sep_conv(self.temporal_conv(dummy))
            _, C, S, T = out.shape
            return C, S, T

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Args:
            x: (batch, 1, n_channels, n_timepoints) or (batch, n_channels, n_timepoints)
            return_attention: if True, return dict with spatial and temporal weights

        Returns:
            logits: (batch, n_classes)
            attention_dict (optional): {
                'spatial': (batch, n_heads, n_channels, n_channels),
                'temporal': (batch, n_heads, n_timesteps, n_timesteps),
            }
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        B = x.size(0)

        # ── CNN feature extraction (preserve spatial dim) ────────────────
        x = self.temporal_conv(x)
        x = self.sep_conv(x)

        _, C, S, T = x.shape

        # ── Spatial attention path ───────────────────────────────────────
        x_spatial = x.permute(0, 2, 1, 3).reshape(B, S, C * T)
        x_spatial = self.spatial_proj(x_spatial)
        x_spatial = self.spatial_pos(x_spatial)

        spatial_attn_weights = None
        for block in self.spatial_transformer:
            x_spatial, spatial_attn_weights = block(x_spatial)

        spatial_out = x_spatial.mean(dim=1)

        # ── Temporal attention path ──────────────────────────────────────
        x_temporal = x.permute(0, 3, 1, 2).reshape(B, T, C * S)
        x_temporal = self.temporal_proj(x_temporal)
        x_temporal = self.temporal_pos(x_temporal)

        temporal_attn_weights = None
        for block in self.temporal_transformer:
            x_temporal, temporal_attn_weights = block(x_temporal)

        temporal_out = x_temporal.mean(dim=1)

        # ── Fuse and classify ────────────────────────────────────────────
        fused = torch.cat([spatial_out, temporal_out], dim=-1)
        logits = self.classifier(fused)

        if return_attention:
            return logits, {
                "spatial": spatial_attn_weights,
                "temporal": temporal_attn_weights,
            }
        return logits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        _, attn_dict = self.forward(x, return_attention=True)
        return attn_dict["spatial"]

    def get_temporal_attention(self, x: torch.Tensor) -> torch.Tensor:
        _, attn_dict = self.forward(x, return_attention=True)
        return attn_dict["temporal"]


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print(" EEG Transformer v2 (small-data tuned) — Sanity Check")
    print("=" * 65)

    model = EEGTransformer()
    x = torch.randn(8, 1, 22, 1000)

    logits = model(x)
    print(f"Input shape:        {tuple(x.shape)}")
    print(f"Output shape:       {tuple(logits.shape)}")
    print(f"Trainable params:   {model.count_parameters():,}")
    assert logits.shape == (8, 4)

    logits2, attn_dict = model(x, return_attention=True)
    print(f"Spatial attention:  {tuple(attn_dict['spatial'].shape)}")
    print(f"Temporal attention: {tuple(attn_dict['temporal'].shape)}")

    assert attn_dict["spatial"].shape[2] == 22

    x_3d = torch.randn(4, 22, 1000)
    logits3 = model(x_3d)
    assert logits3.shape == (4, 4)

    print("-" * 65)
    print("✓ All sanity checks passed")
    print("=" * 65)

    print("\n── Layer breakdown ──")
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:25s}  {n_params:>8,} params")
