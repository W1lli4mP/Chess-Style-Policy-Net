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
        in_channels: int = 18,
        hidden_channels: int = 64,
        num_res_blocks: int = 4,
        board_embedding_dim: int = 256
    ):
        super().__init__()

        # input board tensor into projection layer before feeding into res blocks
        #? dimensionality expansion
        # (batch_size, 18, 8, 8) -> (batch_size, 64, 8, 8)
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(hidden_channels) for _ in range(num_res_blocks)
            ]
        )

        # fc 256
        # (batch_size, 64, 8, 8) -> (batch_size, 4096) -> (batch_size, 256)
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(hidden_channels * 8 * 8, board_embedding_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.projection(x)
        self.res_blocks(x)
        self.fc(x)
        
        return x