from pathlib import Path
import pandas as pd
import chess.pgn

INPUT_PATH = Path("data/processed/games_raw.parquet")
OUTPUT_PATH = Path("data/processed/training_positions.parquet")

def load_games() -> pd.DataFrame:
    return pd.read_parquet(INPUT_PATH)

def parse_game_pgn(pgn: str) -> str:
    pass

def build_training_dataset() -> None:
    games_df = load_games()

    pass

if __name__ == "__main__":
    build_training_dataset()