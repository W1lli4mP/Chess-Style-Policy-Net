from torch.utils.data import Dataset
from pathlib import Path
import chess
import pandas as pd
from board_encoder import encode_board
from context_encoder import encode_context
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[index]

        board = chess.Board(row["fen_before_move"])
        board_tensor = encode_board(board)

        context_tensor = encode_context(row)

        return board_tensor, context_tensor