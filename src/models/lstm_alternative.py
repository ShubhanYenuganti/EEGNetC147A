"""
Pure LSTM for EEG motor imagery classification — 128 Hz / ERS pipeline.

Architecture for BCI Competition IV Dataset 2a:
    - 22 channels, 256 timepoints (2s @ 128Hz), 4 classes

Design philosophy:
  No CNN front-end. The LSTM operates on the raw EEG channels after a
  non-learnable temporal subsampling step (AvgPool1d, 4×) that reduces
  the sequence length from 256 to 64. An input projection expands the
  channel space before the LSTM, and temporal attention pooling
  aggregates the bidirectional hidden states.

Pipeline:
  1. Permute (B, 22, 256) → (B, 22, 256)   -- pool over time axis
  2. AvgPool1d(4)  (B, 22, 256) → (B, 22, 64)   -- non-learnable downsample
  3. Permute (B, 22, 64) → (B, 64, 22)     -- sequence-first for LSTM
  4. Input projection  Linear(22→16) + LayerNorm + ELU
  5. proj_dropout (0.2)  -- regularizes projected representation before LSTM
  6. Bidirectional LSTM, 1 layer, hidden_size=32
  7. lstm_out_dropout (0.5)  -- aggressively regularizes recurrent output
  8. LayerNorm on LSTM output (B, 64, 64)
  9. Temporal attention pooling  (B, 64, 64) → (B, 64)
  10. Classifier  Dropout + Linear(64, n_classes)

Design rationale:
  - AvgPool1d(4): reduces 256→64 timesteps with zero parameters; removes
    high-frequency noise and makes the LSTM sequence length tractable for
    small EEG datasets (~288 training trials per subject).
  - 1-layer BiLSTM, hidden_size=32: ~12.7K parameters in the recurrent
    block (vs. ~128K for the original 2-layer hidden_size=64 design),
    dramatically reducing the capacity available for memorizing noise.
  - proj_dim=16: lightweight input projection appropriate for a 64-step
    input after pooling.
  - LayerNorm throughout (not BatchNorm): no running stats, no cross-subject
    distribution leak — correct for the ERS-normalised 128 Hz pipeline.
  - Attention pooling: same two-layer soft-attention as cnn_gru_alternative,
    learns to weight the 64 timesteps by relevance.
  - MaxNormConstraint on classifier weight (max_norm=0.25): consistent with
    alternative_eegnet.py and cnn_lstm_alternative.py.

Parameter count (~15K):
  Input projection  Linear(22→16) + LayerNorm    ~  374
  LSTM layer 1      (16→32, bidir, 1 layer)      ~ 12,672
  LayerNorm(64)                                       128
  Attention         (64→24→1)                    ~  1,560
  Classifier        Linear(64→4)                     260
  Total                                          ~ 14,994
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention pooling — identical to cnn_gru_alternative.py
# ---------------------------------------------------------------------------
class TemporalAttentionPool(nn.Module):
    """
    Soft attention over the time dimension of LSTM output.

    Given H in (batch, T, D), computes:
        e_t  = tanh(W * h_t + b)       -> (batch, T, attn_dim)
        a_t  = softmax(v * e_t)        -> (batch, T)
        c    = sum a_t * h_t           -> (batch, D)
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 24):
        super().__init__()
        self.project = nn.Linear(hidden_dim, attn_dim, bias=True)
        self.context = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim)  -- attention-weighted summary
            weights: (batch, seq_len)     -- attention distribution
        """
        e = torch.tanh(self.project(h))           # (B, T, attn_dim)
        scores = self.context(e).squeeze(-1)       # (B, T)
        weights = F.softmax(scores, dim=-1)        # (B, T)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, D)
        return context, weights


# ---------------------------------------------------------------------------
# MaxNorm constraint — consistent with cnn_lstm_alternative.py
# ---------------------------------------------------------------------------
class MaxNormConstraint:
    """Clips the L2 norm of a parameter tensor to `max_norm` in-place."""

    def __init__(self, param: nn.Parameter, max_norm: float, dim: int = 0):
        self.param    = param
        self.max_norm = max_norm
        self.dim      = dim

    def __call__(self):
        with torch.no_grad():
            norms = self.param.norm(2, dim=self.dim, keepdim=True).clamp(min=1e-8)
            scale = (norms / self.max_norm).clamp(min=1.0)
            self.param.div_(scale)


# ---------------------------------------------------------------------------
# Pure LSTM model
# ---------------------------------------------------------------------------
class LSTM(nn.Module):
    """
    Pure bidirectional LSTM for EEG motor imagery — 128 Hz / ERS pipeline.

    Args:
        n_classes     : number of output classes (default: 4)
        n_channels    : number of EEG channels (default: 22)
        n_timepoints  : number of time samples (default: 256 = 2s @ 128Hz)
        pool_factor   : AvgPool1d stride for temporal subsampling (default: 4)
        proj_dim      : input projection dimension (default: 16)
        hidden_size   : LSTM hidden size per direction (default: 32)
        num_layers    : number of stacked LSTM layers (default: 1)
        attn_dim      : attention projection size (default: 24)
        dropout_rate  : dropout probability (inter-layer + classifier, default: 0.5)
        proj_dropout  : dropout after input projection, before LSTM (default: 0.2)
        lstm_out_dropout : dropout after LSTM output, before LayerNorm (default: 0.5)
        bidirectional : use bidirectional LSTM (default: True)
    """

    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        n_timepoints: int = 256,
        pool_factor: int = 4,
        proj_dim: int = 24,
        hidden_size: int = 48,
        num_layers: int = 1,
        attn_dim: int = 24,
        dropout_rate: float = 0.5,
        proj_dropout: float = 0.2,
        lstm_out_dropout: float = 0.4,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.n_channels    = n_channels
        self.n_timepoints  = n_timepoints
        self.pool_factor   = pool_factor
        self.hidden_size   = hidden_size
        self.bidirectional = bidirectional

        # -- Non-learnable temporal subsampling ------------------------------
        # AvgPool1d reduces sequence length from n_timepoints → n_timepoints/pool_factor
        # (256 → 64 with default pool_factor=4). Zero parameters.
        self.time_pool = nn.AvgPool1d(kernel_size=pool_factor, stride=pool_factor)

        # -- Input projection: expand 22 channels → proj_dim before LSTM ----
        self.input_proj = nn.Sequential(
            nn.Linear(n_channels, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ELU(),
        )
        self.proj_drop     = nn.Dropout(proj_dropout)
        self.lstm_out_drop = nn.Dropout(lstm_out_dropout)

        # -- Bidirectional LSTM ----------------------------------------------
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # -- LayerNorm on LSTM output (no running stats) ---------------------
        self.lstm_norm = nn.LayerNorm(lstm_out_dim)

        # -- Temporal attention pooling (same as cnn_gru_alternative) --------
        self.attention = TemporalAttentionPool(lstm_out_dim, attn_dim)

        # -- Classifier head -------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_out_dim, n_classes),
        )

        self._cls_constraint = MaxNormConstraint(
            self.classifier[1].weight, max_norm=0.25, dim=1
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_channels, n_timepoints)  -- channels-first from dataloader
            return_attention: if True, also return attention weights

        Returns:
            logits: (batch, n_classes)
            attn_weights (optional): (batch, n_timepoints // pool_factor)
        """
        # Drop channel-singleton dimension if present (4-D input from CNN pipeline)
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, 1, C, T) -> (B, C, T)

        # -- Non-learnable temporal subsampling ------------------------------
        # x is (B, C, T) — AvgPool1d operates over the last dimension
        x = self.time_pool(x)   # (B, 22, 256) → (B, 22, 64)

        # Permute channels-first → sequence-first for LSTM
        x = x.permute(0, 2, 1)  # (B, 22, 64) → (B, 64, 22)

        # -- Input projection ------------------------------------------------
        x = self.input_proj(x)   # (B, 64, proj_dim)
        x = self.proj_drop(x)    # dropout before LSTM

        # -- LSTM ------------------------------------------------------------
        x, _ = self.lstm(x)      # (B, 64, lstm_out_dim)
        x = self.lstm_out_drop(x)  # dropout after LSTM, before LayerNorm
        x = self.lstm_norm(x)

        # -- Attention pooling -----------------------------------------------
        context, attn_weights = self.attention(x)  # (B, lstm_out_dim), (B, 64)

        # -- Classification --------------------------------------------------
        logits = self.classifier(context)  # (B, n_classes)

        if return_attention:
            return logits, attn_weights
        return logits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def apply_max_norm_(self) -> None:
        """Apply MaxNorm constraint to classifier weight. Called by train_128.py."""
        self._cls_constraint()


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print(" Pure LSTM (128 Hz / ERS pipeline) — Sanity Check")
    print("=" * 65)

    model = LSTM()
    x = torch.randn(8, 22, 256)

    pooled_len = 256 // model.pool_factor  # 64

    # Forward pass (3-D input — standard from dataloader)
    logits = model(x)
    print(f"Input shape:        {tuple(x.shape)}")
    print(f"Output shape:       {tuple(logits.shape)}")
    print(f"Trainable params:   {model.count_parameters():,}")
    assert logits.shape == (8, 4), f"Unexpected output shape: {logits.shape}"

    # Forward with attention weights
    logits2, attn_w = model(x, return_attention=True)
    print(f"Attention weights:  {tuple(attn_w.shape)}")
    assert attn_w.shape == (8, pooled_len), f"Attention shape mismatch: {attn_w.shape}"
    assert torch.allclose(attn_w.sum(dim=-1), torch.ones(8), atol=1e-5), (
        "Attention weights don't sum to 1"
    )

    # apply_max_norm_ (called by train_128.py after each optimizer step)
    model.apply_max_norm_()
    cls_weight = model.classifier[1].weight
    row_norms = cls_weight.norm(2, dim=1)
    assert (row_norms <= 0.25 + 1e-5).all(), (
        f"MaxNorm constraint failed: max norm = {row_norms.max():.4f}"
    )
    print(f"MaxNorm check:      passed (max row norm = {row_norms.max():.4f})")

    # 4-D input (in case caller passes (B, 1, C, T))
    x_4d = torch.randn(4, 1, 22, 256)
    logits3 = model(x_4d)
    assert logits3.shape == (4, 4), "4-D input handling failed"
    print(f"4-D input:          passed")

    print("-" * 65)
    print("All sanity checks passed")
    print("=" * 65)

    # -- Architecture summary ------------------------------------------------
    print("\n-- Layer breakdown --")
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:20s}  {n_params:>8,} params")
