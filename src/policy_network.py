from resnet_cnn import ChessResNet
from context_mlp import ContextMLP
from board_encoder import ChessPositionDataset
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.board_nn = ChessResNet()
        self.context_nn = ContextMLP()

        # self.policy_head = ...
        # self.movetime_head = ...
    
    def forward(self, board_batch: torch.Tensor, context_batch: torch.Tensor) -> torch.Tensor:
        shared_embedding = self.get_shared_embedding(
            board_batch,
            context_batch
        )

        # feed shared embedding to both policy and move-time heads
        # policy out: num_moves from UCI vocab
        # move-time out: 5 defined time buckets

        pass

    def get_shared_embedding(self, board_batch: torch.Tensor, context_batch: torch.Tensor) -> torch.Tensor:
        x_board = board_batch
        board_embedding = self.board_nn.forward(x_board)

        x_context = context_batch
        context_embedding = self.board_nn.forward(x_context)

        shared_embedding = torch.cat(
            [board_embedding, context_embedding],
            dim=1
        )

        return shared_embedding
        
#! temp
if __name__ == "__main__":
    policy_net = PolicyNetwork()

    dataset = ChessPositionDataset("data/processed/training_positions.parquet")

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    # process each batch
    for board_batch, context_batch in loader:
        policy_net.forward(board_batch, context_batch)