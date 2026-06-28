from torch.utils.data import Dataset
from pathlib import Path
import chess
import pandas as pd
from board_encoder import encode_board

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

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]

        board = chess.Board(row["fen_before_move"])
        board_tensor = encode_board(board)

        return board_tensor