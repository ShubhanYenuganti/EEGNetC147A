"""
EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs
Based on: Lawhern et al. (2018) - https://arxiv.org/abs/1611.08024

Architecture for BCI Competition IV Dataset 2a:
  - 22 channels, 1000 timepoints (4s @ 250Hz), 4 classes
  - ~2K parameters (lightweight baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet architecture.

    Args:
        n_classes     : number of output classes (default: 4)
        n_channels    : number of EEG channels (default: 22 for BCI-IV-2a)
        n_timepoints  : number of time samples (default: 1000 = 4s @ 250Hz)
        sfreq         : sampling frequency in Hz (default: 250)
        F1            : number of temporal filters (default: 8)
        D             : depth multiplier for depthwise conv (default: 2)
        F2            : number of pointwise filters = F1 * D (default: 16)
        dropout_rate  : dropout probability (default: 0.5)
    """

    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        n_timepoints: int = 1000,
        sfreq: int = 250,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        F2 = F1 * D  # enforce F2 = F1 * D

        # Temporal conv: kernel = sfreq // 2  (captures ~2 cycles at lowest freq)
        temporal_kernel = sfreq // 2  # 125 for 250 Hz
        self.block1 = nn.Sequential(
            # Temporal filter — learns frequency-specific features
            nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial filter — learns spatial (channel) patterns
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # downsample: 1000 → 250
            nn.Dropout(dropout_rate),
        )

        sep_kernel = sfreq // 8  # 32 for 250 Hz
        self.block2 = nn.Sequential(
            # Depthwise conv
            nn.Conv2d(F2, F2, kernel_size=(1, sep_kernel), padding=(0, sep_kernel // 2), groups=F2, bias=False),
            # Pointwise conv
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),  # downsample: 250 → 31
            nn.Dropout(dropout_rate),
        )

        # Compute flattened size dynamically
        self._flat_size = self._get_flat_size()
        self.classifier = nn.Linear(self._flat_size, n_classes)

    def _get_flat_size(self) -> int:
        """Pass a dummy tensor to compute the flattened feature size."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_timepoints)
            out = self.block2(self.block1(dummy))
            return int(out.view(1, -1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, n_channels, n_timepoints)  — or (batch, n_channels, n_timepoints)
        Returns:
            logits: (batch, n_classes)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)     
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# sanity check 
if __name__ == "__main__":
    model = EEGNet()
    x = torch.randn(8, 1, 22, 1000)    # batch of 8
    out = model(x)
    print(f"Input:      {tuple(x.shape)}")
    print(f"Output:     {tuple(out.shape)}")
    print(f"Parameters: {model.count_parameters():,}")
    assert out.shape == (8, 4), "Unexpected output shape"
    print("✓ EEGNet sanity check passed")