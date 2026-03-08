"""
Dummy model for pipeline testing only.
Has the correct input/output shape but won't learn anything meaningful.
Delete once real models are implemented.

Input:  (batch_size, 22, 1000)
Output: (batch_size, 4)
"""

import torch.nn as nn


class Dummy(nn.Module):
    def __init__(self, n_channels: int = 22, n_classes: int = 4):
        super().__init__()
        self.fc = nn.Linear(n_channels * 1000, n_classes)

    def forward(self, x):
        # x: (batch_size, 22, 1000)
        return self.fc(x.flatten(1))  # (batch_size, 4)





