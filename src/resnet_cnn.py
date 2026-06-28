import torch
import torch.nn as nn
from residual_block import ResidualBlock
from board_encoder import encode_board

"""
usage
x = encode_board()
chess_nn = ChessResNet()
chess_nn.forward(x) for inference
"""

class ChessResNet(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 64,
        num_res_blocks: int = 4
    ):
        super().__init__()

        # input board tensor into projection layer before feeding into res blocks
        #? dimensionality expansion (18 -> 64)
        self.projection = nn.Sequential(
            nn.Conv2d(kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(hidden_channels) for _ in range(num_res_blocks)
            ]
        )

        # fc 256
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 8 * 8, 256),
            nn.ReLU(inplace=True)
        )

        self.linear = nn.Linear(hidden_channels * 8 * 8, 256)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.projection(x)
        self.res_blocks(x)
        
        # flatten 4096 (from 64x8x8)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = self.relu(x)
        
        return x