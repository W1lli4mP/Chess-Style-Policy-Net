import torch.nn as nn
import torch

class ContextMLP(nn.Module):
    def __init__(self):
        super().__init__()

    # (15,) -> (64,)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass