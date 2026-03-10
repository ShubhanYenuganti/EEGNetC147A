"""
CNN+LSTM hybrid architecture for EEG motor imagery classification.
Based on: Bashivan et al. (2016) - https://arxiv.org/abs/1511.06448

Architecture overview:
  1. Temporal conv  — learns frequency-specific features along time axis
  2. Depthwise spatial conv — learns which electrode combinations matter
  3. LSTM           — models long-range temporal dependencies
  4. Classifier     — linear layer to 4 classes

Input:  (batch, 22, 1000)  — 22 EEG channels, 1000 timepoints (4s @ 250Hz)
Output: (batch, 4)         — logits for 4 motor imagery classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    """
    EEGNet architecture.

    Args:
        n_classes     : number of output classes (default: 4)
        n_channels    : number of EEG channels (default: 22)
        n_timepoints  : number of time samples (default: 1000)
        F1            : number of temporal filters (default: 32)
        F2            : number of pointwise filters = F1 * D (default: 64)
        lstm_hidden   : LSTM hidden state size (default : 128)
        lstm_layers   : number of LSTM layers (default: 2)
        dropout_rate  : dropout probability (default: 0.5)
    """

    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        n_timepoints: int = 1000,
        F1: int = 8,
        F2: int = 16,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        dropout_rate: float = 0.7,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.lstm_hidden = lstm_hidden

        # -------------------------------------------------------------------
        # Stage 1: Temporal convolution
        # Kernel (1, 25) slides along time axis only.
        # Learns short-range temporal patterns (frequency content).
        # padding=(0, 12) keeps the time dimension at 1000.
        # -------------------------------------------------------------------
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU()
        )

        # -------------------------------------------------------------------
        # Stage 2: Depthwise spatial convolution
        # Kernel (n_channels, 1) collapses  the electrode dimension.
        # groups=F1 means each temporal filter gets its own spatial filter
        # (depthwise - no cross-filter mixing yet).
        # Followed by pooling to reduce time from 100 -> 250.
        # -------------------------------------------------------------------

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F2, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(dropout_rate),
        )

        # -------------------------------------------------------------------
        # Stage 3: LSTM
        # Input: (batch, seq_len=250, input_size=F2).
        # Output: we take only the last hidden state -> (batch, lstm_hidden)
        # batch_first=True so input/output are (batch, seq, features)
        # dropout between LSTM layers when lstm_layers > 1
        # -------------------------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=F2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0
        )

        # -------------------------------------------------------------------
        # Classifier
        # -------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden, n_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, n_channels, n_timepoints)  — or (batch, 1, n_channels, n_timepoints)
        Returns:
            logits: (batch, n_classes)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # CNN stages     
        x = self.temporal_conv(x) # (batch, F1, 22, 1000)
        x = self.spatial_conv(x)    # (batch, F2, 1, 250)

        # Reshape for LSTM: (batch, F2, 1, 250) → (batch, 250, F2)
        x = x.squeeze(2)            # (batch, F2, 250)
        x = x.permute(0, 2, 1)     # (batch, 250, F2)

        # LSTM: take only the last timestep's output
        x, _ = self.lstm(x)         # (batch, 250, lstm_hidden)
        x = x[:, -1, :]            # (batch, lstm_hidden)

        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# sanity check 
if __name__ == "__main__":
    model = CNNLSTM()
    x = torch.randn(8, 22, 1000)    # batch of 8
    out = model(x)
    print(f"Input:      {tuple(x.shape)}")
    print(f"Output:     {tuple(out.shape)}")
    print(f"Parameters: {model.count_parameters():,}")
    assert out.shape == (8, 4), "Unexpected output shape"
    print("✓ CNN+LSTM sanity check passed")