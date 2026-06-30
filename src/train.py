import torch.nn as nn
from torch.utils.data import DataLoader

from policy_network import PolicyNetwork
from chess_position_dataset import ChessPositionDataset
from move_vocab import MOVE_VOCAB_SIZE

#* hyperparams
EPOCHS = 50
# controls how much move-time loss matters compared with policy loss
MOVE_TIME_LOSS_WEIGHT = 0.2

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

    total_loss = 0.0
    total_policy_loss = 0.0
    total_move_time_loss = 0.0

    for (
        board_inputs,
        context_inputs,
        move_target,
        move_time_target,
        move_time_mask
    ) in loader:
        # from board, context branches
        policy_pred, move_time_pred = policy_net(board_inputs, context_inputs)

        policy_loss = loss_function(
            policy_pred,
            move_target
        )

        move_time_loss = loss_function(
            move_time_pred,
            move_time_target
        )

        # aggregate policy and move-time loss before backpropagation
        loss = (
            policy_loss + MOVE_TIME_LOSS_WEIGHT * move_time_loss
        )
        loss.backward()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_move_time_loss += move_time_loss.item()
    
    num_batches = len(loader)

    print(
        f"Epoch {epoch}: "
        f"loss={total_loss / num_batches:.4f}, "
        f"policy_loss={total_policy_loss / num_batches:.4f}, "
        f"move_time_loss={total_move_time_loss / num_batches:.4f}"
    )