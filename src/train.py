import torch.nn as nn
from torch.utils.data import DataLoader

from policy_network import PolicyNetwork
from chess_position_dataset import ChessPositionDataset
from move_vocab import MOVE_VOCAB_SIZE

EPOCHS = 50

policy_net = PolicyNetwork(MOVE_VOCAB_SIZE)

dataset = ChessPositionDataset("data/processed/training_positions.parquet")

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

loss_function = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    policy_net.train()

    total_loss = 0

    for board_inputs, context_inputs in loader:
        # from board, context branches
        policy_pred, move_time_pred = policy_net(board_inputs, context_inputs)

        #! need to extract targets
        # loss = loss_function(y, y_bar)
        # loss.backward()
        # total_loss += loss.item()

    print(f"Epoch {epoch}: loss = {total_loss / len(loader):.4f}")