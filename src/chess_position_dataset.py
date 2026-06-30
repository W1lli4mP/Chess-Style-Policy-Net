from torch.utils.data import Dataset
from pathlib import Path
import chess
import pandas as pd
from board_encoder import encode_board
from context_encoder import encode_context
from move_vocab import MOVE_TO_ID
import torch

class ChessPositionDataset(Dataset):
    def __init__(self, parquet_path: str):
        parquet_path = Path(parquet_path)

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Parquet file does not exist: {parquet_path.resolve()}"
            )

        self.rows = pd.read_parquet(parquet_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[index]

        board = chess.Board(row["fen_before_move"])
        board_tensor = encode_board(board)

        context_tensor = encode_context(row)

        # construct move target + validation
        uci_move = row["uci_move"]

        try:
            move_id = MOVE_TO_ID[uci_move]
        except KeyError as e:
            raise ValueError(
                f"Move {uci_move!r} is not present in the move vocabulary"
            ) from e

        move_target = move_id

        # construct move-time target
        has_time_target = row["has_move_time_target"]

        #? long dtype since cross entropy expects a 64-bit int tensor
        move_time_target = torch.tensor(
            row["move_time_bucket"]
            if has_time_target else 0,
            dtype=torch.long
        )

        move_time_mask = torch.tensor(
            has_time_target,
            dtype=torch.bool
        )

        return (
            board_tensor,
            context_tensor,
            move_target,
            move_time_target,
            move_time_mask
        )