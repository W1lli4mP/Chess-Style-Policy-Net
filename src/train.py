import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from pathlib import Path

from policy_network import PolicyNetwork
from chess_position_dataset import ChessPositionDataset
from move_vocab import MOVE_VOCAB_SIZE

CHECKPOINT_PATH = Path("models/latest_policy_net.pt")
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

#* hyperparams
EPOCHS = 30
# controls how much move-time loss matters compared with policy loss
MOVE_TIME_LOSS_WEIGHT = 0.2

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

policy_net = PolicyNetwork(MOVE_VOCAB_SIZE).to(device)

print(f"Training on {device}")

dataset = ChessPositionDataset("data/processed/training_positions.parquet")

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

num_batches = len(loader)

optimiser = torch.optim.AdamW(
    policy_net.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

policy_loss_function = nn.CrossEntropyLoss()

#? move-time requires multiple losses; not scalar
move_time_loss_function = nn.CrossEntropyLoss(
    reduction="none"
)

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.perf_counter()

    policy_net.train()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_move_time_loss = 0.0

    # batch indexing to track current batches processed
    for batch_index, (
        board_inputs,
        context_inputs,
        move_targets,
        move_time_targets,
        move_time_masks
    ) in enumerate(loader, start=1):
        board_inputs = board_inputs.to(device)
        context_inputs = context_inputs.to(device)
        move_targets = move_targets.to(device)
        move_time_targets = move_time_targets.to(device)
        move_time_masks = move_time_masks.to(device)

        # removes gradients from prev batch
        optimiser.zero_grad()

        # from board, context branches
        policy_logits, move_time_logits = policy_net(board_inputs, context_inputs)

        policy_loss = policy_loss_function(
            policy_logits,
            move_targets
        )

        move_time_losses = move_time_loss_function(
            move_time_logits,
            move_time_targets
        )

        # apply mask
        if move_time_masks.any():
            move_time_loss = move_time_losses[
                move_time_masks
            ].mean()
        else:
            move_time_loss = move_time_logits.sum() * 0.0

        # aggregate policy and move-time loss before backpropagation
        loss = (
            policy_loss + MOVE_TIME_LOSS_WEIGHT * move_time_loss
        )

        loss.backward()

        # updates model weights
        optimiser.step()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_move_time_loss += move_time_loss.item()
    
        if batch_index % 100 == 0:
            elapsed = time.perf_counter() - epoch_start

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"batch {batch_index}/{num_batches} | "
                f"loss {loss.item():.4f} | "
                f"elapsed {elapsed:.1f}s",
                flush=True
            )

    elapsed = time.perf_counter() - epoch_start

    print(
        f"Epoch {epoch}: "
        f"loss={total_loss / num_batches:.4f}, "
        f"policy_loss={total_policy_loss / num_batches:.4f}, "
        f"move_time_loss={total_move_time_loss / num_batches:.4f}, "
        f"time={elapsed:.1f}s",
        flush=True
    )

    # overwrite saved model with latest one per epoch
    #? in case training process gets interrupted
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": policy_net.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "move_vocab_size": MOVE_VOCAB_SIZE,
            "move_time_loss_weight": MOVE_TIME_LOSS_WEIGHT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        CHECKPOINT_PATH,
    )

    print(f"Saved checkpoint after epoch {epoch}")