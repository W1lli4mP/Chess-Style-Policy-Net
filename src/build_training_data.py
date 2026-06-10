from pathlib import Path
import pandas as pd
import chess.pgn
import io

INPUT_PATH = Path("data/processed/games_raw.parquet")
OUTPUT_PATH = Path("data/processed/training_positions.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_games() -> pd.DataFrame:
    return pd.read_parquet(INPUT_PATH)

def get_move_time_bucket(
    move_time_seconds: float | None
) -> int | None:
    if move_time_seconds is None:
        return None

    if move_time_seconds < 0.3:
        return 0
    if move_time_seconds < 1.0:
        return 1
    if move_time_seconds < 3.0:
        return 2
    if move_time_seconds < 8.0:
        return 3
    
    return 4

def parse_game_pgn(pgn: str) -> chess.pgn.Game | None:
    # convert pgn into usable python-chess game object

    if not isinstance(pgn, str) or not pgn.strip():
        return None
    
    pgn_stream = io.StringIO(pgn)
    game = chess.pgn.read_game(pgn_stream)

    if game is None or game.errors:
        return None

    return game

def parse_time_control(time_control: str) -> tuple[int, float]:
    # <base-seconds> | <plus> | <increment>
    if "+" in time_control:
        base, increment = time_control.split("+", maxsplit=1)
        return int(base), float(increment)

    return int(time_control), 0

def extract_training_rows(
    game: chess.pgn.Game,
    game_row: pd.Series
) -> list[dict]:
    """
        process an entire chess game and return a list of training row dicts containing
        all information for the board, context and target branches

        clock nuance:
            clock_before_move is an input feature
            move_time_seconds is the target
            clock_after_move is only used in preprocessing to calculate the target
    """

    training_rows = []
    board = game.board()

    base_time_seconds, increment_seconds = parse_time_control(
        game_row["time_control"]
    )

    # initialise to starting times
    white_clock = float(base_time_seconds)
    black_clock = float(base_time_seconds)

    for node in game.mainline():
        move = node.move
        moving_colour = board.turn

        clock_before_move = (
            white_clock if moving_colour == chess.WHITE else black_clock
        )

        clock_after_move = node.clock()

        move_time_seconds = None

        if clock_after_move is not None:
            raw_move_time = (
                clock_before_move +
                increment_seconds -
                clock_after_move
            )

            # clip negative times
            if raw_move_time < -0.1:
                move_time_seconds = None
            else:
                move_time_seconds = max(0.0, raw_move_time)

        # verify turn
        my_turn = (
            board.turn == chess.WHITE
            and game_row["my_colour"] == "white"
        ) or (
            board.turn == chess.BLACK
            and game_row["my_colour"] == "black"
        )

        # record moves only if they are my turn
        if my_turn:
            my_clock = clock_before_move
            
            #* before my move, opponent's stored clock is
            #* their clock after their previous move
            opponent_clock = (
                black_clock
                if moving_colour == chess.WHITE
                else white_clock
            )

            # construct training row with all relevant info
            training_row = {
                "game_uuid": game_row["uuid"],
                "my_username": game_row["my_username"],
                "end_time": game_row["end_time"],

                **extract_board_fields(board),

                **extract_context_fields(
                    game_row=game_row,
                    board=board,
                    base_time_seconds=base_time_seconds,
                    increment_seconds=increment_seconds,
                    my_clock=my_clock,
                    opponent_clock=opponent_clock
                ),

                **extract_target_fields(
                    move=move,
                    move_time_seconds=move_time_seconds
                )
            }

            training_rows.append(training_row)

        if clock_after_move is not None:
            if moving_colour == chess.WHITE:
                white_clock = clock_after_move
            else:
                black_clock = clock_after_move

        # update the board after recording an entire training row
        board.push(move)

    return training_rows

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
        "move_time_bucket": get_move_time_bucket(move_time_seconds),
        "has_move_time_target": move_time_seconds is not None
    }

def build_training_dataset() -> None:
    games_df = load_games()

    all_training_rows = []
    skipped_games = 0

    for _, game_row in games_df.iterrows():
        game = parse_game_pgn(game_row["pgn"])

        if game is None:
            skipped_games += 1
            continue

        # retrieve all training rows and store into parquet
        # catch invalid values to avoid terminating the entire build
        try:
            game_training_rows = extract_training_rows(
                game=game,
                game_row=game_row
            )
        except (TypeError, ValueError) as error:
            skipped_games += 1
            print(
                f"Skipping game {game_row['uuid']}: "
                f"invalid time control {game_row['time_control']!r} ({error})"
            )
            continue

        all_training_rows.extend(game_training_rows)

    if not all_training_rows:
        raise RuntimeError("No training rows were created")

    training_df = pd.DataFrame(all_training_rows)

    training_df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print(f"Games loaded: {len(games_df)}")
    print(f"Games skipped: {skipped_games}")
    print(f"Training positions completed: {len(training_df)}")
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    build_training_dataset()