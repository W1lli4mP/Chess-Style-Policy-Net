import torch.nn as nn
import torch

CONTEXT_DIM = 15
CONTEXT_EMBEDDING_DIM = 64
class ContextMLP(nn.Module):
    def __init__(
        self,
        context_dim: int = CONTEXT_DIM,
        hidden_dim: int = 64,
        context_embedding_dim: int = CONTEXT_EMBEDDING_DIM
    ):
        super().__init__()

        self.hidden_layers = nn.Sequential(
            # hidden 1
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(inplace=True),

            # hidden 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # no activation
        self.output = nn.Linear(hidden_dim, context_embedding_dim)

    # (batch_size, 15) -> (batch_size, 64)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(x)
        x = self.output(x)

        return x