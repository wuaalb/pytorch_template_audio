import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_chan):
        super().__init__()
        self.conv = nn.Conv1d(n_chan, n_chan, 5, padding=2)

    def forward(self, x):
        return self.conv(x)
