"""
Lightweight EEG Transformer with Spatial-Temporal Attention
===========================================================

Architecture for BCI Competition IV Dataset 2a:
  - 22 channels, 1000 timepoints (4s @ 250Hz), 4 classes

Design philosophy:
  CNN front-end (matching EEGNet/CNN+GRU) extracts spatial-temporal features,
  then separate spatial and temporal attention mechanisms capture which channels
  and which time points matter most for classification.

Key difference from standard transformer:
  - Matched CNN front-end (F1=16 temporal filters, depthwise spatial, separable)
  - Dual attention paths: spatial (over filter groups) + temporal (over timesteps)
  - Max-norm constraints for regularization (matching EEGNet paper)
  - Attention pooling for explainability

Pipeline:
  1. Temporal Conv      → frequency-specific filters across time
  2. Depthwise Conv     → spatial (channel) mixing per temporal filter
  3. Separable Conv     → refine combined spatial-temporal features
  4. Reshape + Project  → prepare for dual attention paths
  5. Spatial Attention   → which spatial filter groups matter?
  6. Temporal Attention  → which time windows matter?
  7. Fuse + Classify    → concatenate and project to logits

For attention visualization (C3/C4 analysis):
  The spatial attention weights show which depthwise filter groups the model
  focuses on. To map back to electrodes, examine the depthwise conv weights
  (each group corresponds to a spatial filter over the 22 channels).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Max-norm weight constraint (mirrors EEGNet paper)
# ---------------------------------------------------------------------------
class MaxNormConstraint:
    """Clips the L2 norm of a parameter tensor to `max_norm` in-place."""
    
    def __init__(self, param: nn.Parameter, max_norm: float, dim: int = 0):
        self.param = param
        self.max_norm = max_norm
        self.dim = dim

    def __call__(self):
        with torch.no_grad():
            norms = self.param.norm(2, dim=self.dim, keepdim=True).clamp(min=1e-8)
            scale = (norms / self.max_norm).clamp(min=1.0)
            self.param.div_(scale)


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from Vaswani et al. (2017)."""

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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Multi-head Attention with Extractable Weights
# ---------------------------------------------------------------------------
class MultiHeadAttentionWithWeights(nn.Module):
    """Multi-head attention that returns attention weights for analysis."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

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
        """
        Args:
            query, key, value: (batch, seq_len, d_model)
            
        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, n_heads, seq_len, seq_len)
        """
        B, T, _ = query.shape

        # Project and reshape for multi-head attention
        Q = self.W_q(query).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, heads, T, d_k)
        K = self.W_k(key).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, heads, T, T)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # (B, heads, T, d_k)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, d_model)

        output = self.W_o(context)
        return output, attn_weights


# ---------------------------------------------------------------------------
# Transformer Encoder Block
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
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            x: (batch, seq_len, d_model) — updated representation
            attn_weights: (batch, n_heads, seq_len, seq_len)
        """
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, attn_weights


# ---------------------------------------------------------------------------
# Lightweight EEG Transformer v3
# ---------------------------------------------------------------------------
class EEGTransformer(nn.Module):
    """
    Lightweight Transformer with separate spatial and temporal attention
    for EEG motor imagery classification.

    CNN front-end matches EEGNet/CNN+GRU (temporal → depthwise spatial →
    separable), then dual transformer paths attend over spatial filters
    and time steps separately.

    Args:
        n_classes       : number of output classes (default: 4)
        n_channels      : number of EEG channels (default: 22)
        n_timepoints    : number of time samples (default: 1000)
        sfreq           : sampling frequency in Hz (default: 250)
        F1              : number of temporal filters (default: 16) ← MATCHED TO CNN-GRU
        D               : depth multiplier for depthwise conv (default: 2)
        cnn_dropout     : dropout after CNN blocks (default: 0.4)
        d_model         : transformer model dimension (default: 64)
        spatial_heads   : attention heads for spatial transformer (default: 4)
        temporal_heads  : attention heads for temporal transformer (default: 4)
        spatial_layers  : number of spatial transformer blocks (default: 2)
        temporal_layers : number of temporal transformer blocks (default: 2)
        ff_dim          : feedforward hidden dimension (default: 256)
        trans_dropout   : dropout in transformer blocks (default: 0.3)
        fc_dropout      : dropout before classifier (default: 0.5)
    """

    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        n_timepoints: int = 1000,
        sfreq: int = 250,
        # CNN hyper-parameters (matched to CNN-GRU Round 3)
        F1: int = 16,
        D: int = 2,
        cnn_dropout: float = 0.4,
        # Transformer hyper-parameters (tuned to match CNN-GRU capacity)
        d_model: int = 64,
        spatial_heads: int = 4,
        temporal_heads: int = 4,
        spatial_layers: int = 2,
        temporal_layers: int = 2,
        ff_dim: int = 256,
        trans_dropout: float = 0.3,
        # Classifier
        fc_dropout: float = 0.5,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.d_model = d_model
        self.F1 = F1
        self.D = D

        F2 = F1 * D  # 32 for default F1=16

        # ── CNN Block 1: temporal + depthwise spatial (EEGNet-matching) ───
        temporal_kernel = sfreq // 2  # 125 for 250 Hz
        self.cnn_block1 = nn.Sequential(
            # Temporal conv — frequency-specific features
            nn.Conv2d(
                1, F1,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
            # Depthwise spatial conv — learns channel (electrode) mixing
            nn.Conv2d(
                F1, F2,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False,
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),   # 1000 → 250
            nn.Dropout(cnn_dropout),
        )

        # ── CNN Block 2: separable conv (EEGNet-matching) ────────────────
        sep_kernel = sfreq // 8  # 32 for 250 Hz
        self.cnn_block2 = nn.Sequential(
            # Depthwise temporal conv
            nn.Conv2d(
                F2, F2,
                kernel_size=(1, sep_kernel),
                padding=(0, (sep_kernel - 1) // 2),
                groups=F2,
                bias=False,
            ),
            # Pointwise conv
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),   # 250 → ~31
            nn.Dropout(cnn_dropout),
        )

        # ── Compute CNN output shape ─────────────────────────────────────
        cnn_filters, cnn_temporal = self._cnn_output_shape()

        # ── Spatial Attention Path ───────────────────────────────────────
        # Each of the F2 spatial filters gets a feature vector (its temporal profile)
        # Attention learns which spatial filter groups matter
        self.spatial_proj = nn.Linear(cnn_temporal, d_model, bias=False)
        self.spatial_pos = SinusoidalPositionalEncoding(
            d_model, max_len=F2, dropout=trans_dropout
        )
        self.spatial_transformer = nn.ModuleList([
            TransformerBlock(d_model, spatial_heads, ff_dim, trans_dropout)
            for _ in range(spatial_layers)
        ])

        # ── Temporal Attention Path ──────────────────────────────────────
        # Each time step gets a feature vector (all F2 filter activations at that time)
        # Attention learns which time windows matter
        self.temporal_proj = nn.Linear(cnn_filters, d_model, bias=False)
        self.temporal_pos = SinusoidalPositionalEncoding(
            d_model, max_len=512, dropout=trans_dropout
        )
        self.temporal_transformer = nn.ModuleList([
            TransformerBlock(d_model, temporal_heads, ff_dim, trans_dropout)
            for _ in range(temporal_layers)
        ])

        # ── Classifier: fuse spatial + temporal summaries ────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(fc_dropout),
            nn.Linear(d_model * 2, n_classes),
        )

        # ── Max-norm constraints (mirrors EEGNet paper Table 2) ──────────
        self._dw_constraint = MaxNormConstraint(
            self.cnn_block1[2].weight, max_norm=1.0, dim=0
        )
        self._cls_constraint = MaxNormConstraint(
            self.classifier[2].weight, max_norm=0.25, dim=1
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _cnn_output_shape(self) -> tuple[int, int]:
        """Get (n_filters, n_timesteps) after CNN blocks."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_timepoints)
            out = self.cnn_block2(self.cnn_block1(dummy))
            # out shape: (1, F2, 1, T')
            _, F2, _, T = out.shape
            return F2, T

    def _init_weights(self):
        """Xavier uniform init for conv/linear; constant for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def apply_constraints(self):
        """Call after every optimizer.step() to enforce max-norm constraints."""
        self._dw_constraint()
        self._cls_constraint()

    def apply_max_norm_(self) -> None:
        """Alias for apply_constraints() — called by train.py after every optimizer step."""
        self.apply_constraints()

    def get_depthwise_weights(self) -> torch.Tensor:
        """
        Extract depthwise spatial conv weights for electrode mapping.

        Returns:
            weights: (F2, 1, n_channels, 1) — each filter's spatial pattern
            To visualize which electrodes a spatial filter focuses on,
            plot weights[filter_idx, 0, :, 0] as a topographic map.
        """
        # Depthwise conv is the 3rd layer in cnn_block1 (index 2)
        return self.cnn_block1[2].weight.data

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
                'spatial': (batch, n_heads, F2, F2) — attention over spatial filters
                'temporal': (batch, n_heads, T', T') — attention over time steps
            }
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        B = x.size(0)

        # ── CNN feature extraction ───────────────────────────────────────
        x = self.cnn_block1(x)   # (B, F2, 1, T/4)
        x = self.cnn_block2(x)   # (B, F2, 1, T/32)

        _, F2, _, T = x.shape
        x = x.squeeze(2)         # (B, F2, T)

        # ── Spatial Attention Path ───────────────────────────────────────
        # Each spatial filter gets its temporal profile as feature vector
        # x_spatial: (B, F2, T) → (B, F2, d_model)
        x_spatial = self.spatial_proj(x)              # (B, F2, d_model)
        x_spatial = self.spatial_pos(x_spatial)

        spatial_attn_weights = None
        for block in self.spatial_transformer:
            x_spatial, spatial_attn_weights = block(x_spatial)

        spatial_out = x_spatial.mean(dim=1)           # (B, d_model)

        # ── Temporal Attention Path ──────────────────────────────────────
        # Each time step gets all filter activations as feature vector
        # x_temporal: (B, F2, T) → (B, T, F2) → (B, T, d_model)
        x_temporal = x.permute(0, 2, 1)              # (B, T, F2)
        x_temporal = self.temporal_proj(x_temporal)   # (B, T, d_model)
        x_temporal = self.temporal_pos(x_temporal)

        temporal_attn_weights = None
        for block in self.temporal_transformer:
            x_temporal, temporal_attn_weights = block(x_temporal)

        temporal_out = x_temporal.mean(dim=1)         # (B, d_model)

        # ── Fuse and Classify ────────────────────────────────────────────
        fused = torch.cat([spatial_out, temporal_out], dim=-1)  # (B, 2*d_model)
        logits = self.classifier(fused)               # (B, n_classes)

        if return_attention:
            return logits, {
                "spatial": spatial_attn_weights,       # (B, heads, F2, F2)
                "temporal": temporal_attn_weights,     # (B, heads, T', T')
            }
        return logits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial attention weights.

        To map back to electrodes:
            spatial_weights = model.get_spatial_attention(x)  # (B, heads, F2, F2)
            depthwise_w = model.get_depthwise_weights()       # (F2, 1, 22, 1)
            # Combine: high-attention filters × their spatial patterns → electrode importance
            
        Returns:
            spatial_weights: (batch, n_heads, F2, F2)
        """
        _, attn_dict = self.forward(x, return_attention=True)
        return attn_dict["spatial"]

    def get_temporal_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract temporal attention weights.

        Each time step corresponds to ~32ms of EEG (after 32× downsampling).
        To map to real time: time_sec = step_index * 32 / 250 + 2.0
        (2.0s offset for the MI window start).
        
        Returns:
            temporal_weights: (batch, n_heads, T', T')
        """
        _, attn_dict = self.forward(x, return_attention=True)
        return attn_dict["temporal"]


# ---------------------------------------------------------------------------
# Sanity Check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print(" EEG Transformer v3 (matched CNN+GRU front-end) — Sanity Check")
    print("=" * 65)

    model = EEGTransformerV3()
    x = torch.randn(8, 1, 22, 1000)

    # Forward pass
    logits = model(x)
    print(f"Input shape:        {tuple(x.shape)}")
    print(f"Output shape:       {tuple(logits.shape)}")
    print(f"Trainable params:   {model.count_parameters():,}")
    assert logits.shape == (8, 4), f"Unexpected output shape: {logits.shape}"

    # Attention weights
    logits2, attn_dict = model(x, return_attention=True)
    print(f"Spatial attention:  {tuple(attn_dict['spatial'].shape)}")
    print(f"Temporal attention: {tuple(attn_dict['temporal'].shape)}")

    # Depthwise weights for electrode mapping
    dw = model.get_depthwise_weights()
    print(f"Depthwise weights:  {tuple(dw.shape)}")

    # Verify attention sums to 1
    spatial_sum = attn_dict["spatial"].sum(dim=-1)
    assert torch.allclose(spatial_sum, torch.ones_like(spatial_sum), atol=1e-4), \
        "Spatial attention weights don't sum to 1"
    temporal_sum = attn_dict["temporal"].sum(dim=-1)
    assert torch.allclose(temporal_sum, torch.ones_like(temporal_sum), atol=1e-4), \
        "Temporal attention weights don't sum to 1"

    # 3-D input convenience
    x_3d = torch.randn(4, 22, 1000)
    logits3 = model(x_3d)
    assert logits3.shape == (4, 4), "3-D input handling failed"

    print("-" * 65)
    print("✓ All sanity checks passed")
    print("=" * 65)

    print("\n── Layer breakdown ──")
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:25s}  {n_params:>8,} params")