import torch.nn as nn


class ReidProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
