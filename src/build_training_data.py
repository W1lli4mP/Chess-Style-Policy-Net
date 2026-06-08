from pathlib import Path
import pandas as pd
import chess.pgn
import io

INPUT_PATH = Path("data/processed/games_raw.parquet")
OUTPUT_PATH = Path("data/processed/training_positions.parquet")

def load_games() -> pd.DataFrame:
    return pd.read_parquet(INPUT_PATH)

def parse_game_pgn(pgn: str) -> chess.pgn.Game | None:
    # convert pgn into usable python-chess game object

    if not isinstance(pgn, str) or not pgn.strip():
        return None
    
    pgn_stream = io.StringIO(pgn)
    game = chess.pgn.read_game(pgn_stream)

    if game is None or game.errors:
        return None

    return game

def build_training_dataset() -> None:
    games_df = load_games()

    #* board branch
    # construct relevant board information via the dataset's PGNs
    for _, game_row in games_df.iterrows():
        game = parse_game_pgn(game_row["pgn"])

        if game is None:
            continue

    #* context branch
    # add non-board info here

if __name__ == "__main__":
    build_training_dataset()