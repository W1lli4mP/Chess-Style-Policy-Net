from pathlib import Path
import pandas as pd
import chess.pgn
import io

INPUT_PATH = Path("data/processed/games_raw.parquet")
OUTPUT_PATH = Path("data/processed/training_positions.parquet")

def load_games() -> pd.DataFrame:
    return pd.read_parquet(INPUT_PATH)

def get_move_time_bucket() -> str:
    pass

def parse_game_pgn(pgn: str) -> chess.pgn.Game | None:
    # convert pgn into usable python-chess game object

    if not isinstance(pgn, str) or not pgn.strip():
        return None
    
    pgn_stream = io.StringIO(pgn)
    game = chess.pgn.read_game(pgn_stream)

    if game is None or game.errors:
        return None

    return game

def extract_board_fields(board: chess.Board) -> dict:
    return {
        "fen_before_move": board.fen(),
    }

def extract_context_fields(
    game_row: pd.Series,
    board: chess.Board,
    base_time_seconds: int,
    increment_seconds: int,
    my_clock: float | None,
    opponent_clock: float | None,
) -> dict:
    """
        return all context vector fields:
            time_class /
            base_time_seconds /
            increment_seconds /
            my_clock_seconds /
            opponent_clock_seconds /
            my_clock_to_base_ratio /
            opponent_clock_to_base_ratio /
            ply /
            fullmove_number /
            my_colour /
            rating_diff /
            opponent_rating /
            my_rating /
    """

    #! add recent move time history in future

    my_clock_to_base_ratio = (
        my_clock / base_time_seconds
        if my_clock is not None and base_time_seconds > 0
        else None
    )

    opponent_clock_to_base_ratio = (
        opponent_clock / base_time_seconds
        if opponent_clock is not None and base_time_seconds > 0
        else None
    )

    return {
        "time_class": game_row["time_class"],
        "base_time_seconds": base_time_seconds,
        "increment_seconds": increment_seconds,

        "my_clock_seconds": my_clock,
        "opponent_clock_seconds": opponent_clock,
        "my_clock_to_base_ratio": my_clock_to_base_ratio,
        "opponent_clock_to_base_ratio": opponent_clock_to_base_ratio,

        "ply": board.ply(),
        "fullmove_number": board.fullmove_number,

        "my_colour": game_row["my_colour"],
        "rating_diff": game_row["rating_diff"],
        "opponent_rating": game_row["opponent_rating"],
        "my_rating": game_row["my_rating"],
    }

def extract_target_fields(
    move: chess.Move,
    move_time_seconds: float | None
) -> dict:
    return {
        "uci_move": move.uci(),
        "move_time_seconds": move_time_seconds,
        "move_time_bucket": get_move_time_bucket(move_time_seconds)
    }

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