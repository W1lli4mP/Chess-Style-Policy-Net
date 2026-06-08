import json
import pandas as pd
from pathlib import Path

from config import USERNAMES

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_month_file(path: Path, source_username: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    games = data.get("games", [])

    for game in games:
        game["source_username"] = source_username
    
    return games

def load_user_games(username: str) -> list[dict]:
    username = username.lower()
    user_dir = RAW_DIR / username

    if not user_dir.exists():
        raise FileNotFoundError(f"Could not find folder: {user_dir}")

    all_games = []

    for path in sorted(user_dir.glob("games_*.json")):
        games = load_month_file(path, username)
        all_games.extend(games)
        print(f"Loaded {len(games)} games from {path.name}")
    
    return all_games

def clean_games_dataframe(df: pd.DataFrame, diagnostics: bool = False) -> pd.DataFrame:
    df = df.copy()

    # flatten nested player dicts
    # df["white"] and df["black"] are series' so extracting their fields and adding a prefix
    # allows their fields to be exposed and access more easily
    white_df = pd.json_normalize(df["white"]).add_prefix("white_")
    black_df = pd.json_normalize(df["black"]).add_prefix("black_")

    # remove old series
    df = pd.concat(
        [df.drop(columns=["white", "black"]), white_df, black_df],
        axis=1
    )

    # normalise usernames to be in lowercase
    usernames = { u.lower() for u in USERNAMES }

    df["white_username"] = df["white_username"].astype("string").str.lower()
    df["black_username"] = df["black_username"].astype("string").str.lower()

    required_columns = [
        "pgn",
        "time_control",
        "end_time",
        "rated",
        "uuid",
        "initial_setup",
        "fen",
        "rules",
        "time_class",
        "white_rating",
        "white_result",
        "white_username",
        "black_rating",
        "black_result",
        "black_username"
    ]

    # partition each condition to diagnose individually

    valid_required = df[required_columns].notna().all(axis=1)

    valid_time_class = df["time_class"].isin(["bullet", "blitz", "rapid"])

    valid_rules = df["rules"].eq("chess")

    white_is_known = df["white_username"].isin(usernames)
    black_is_known = df["black_username"].isin(usernames)

    # track recognised players separately for diagnosis
    neither_player_known = ~white_is_known & ~black_is_known
    both_players_known = white_is_known & black_is_known
    exactly_one_known = white_is_known ^ black_is_known

    mask = (
        valid_required &
        valid_time_class &
        valid_rules &
        exactly_one_known
    )

    #! optional diagnostics for tracking why games were dropped
    if diagnostics:
        print("Missing required value:", (~valid_required).sum())
        print("Invalid time class:", (~valid_time_class).sum())
        print("Rules not chess:", (~valid_rules).sum())

        print("Games where neither player is recognised:", neither_player_known.sum())
        print("Games where both players are recognised:", both_players_known.sum())

        # restrict diagnostics to games with the correct time control
        # NOTE: change for testing outside time control
        target_class = valid_time_class

        print(
            "Target games missing required values:",
            (target_class & ~valid_required).sum()
        )

        print(
            "Target games with non-chess rules:",
            (target_class & ~valid_rules).sum()
        )

        print(
            "Target games with unrecognised usernames:",
            (target_class & neither_player_known).sum()
        )

        print(
            "Target games with both players recognised:",
            (target_class & both_players_known).sum()
        )

        #* populate diagnostic df with boolean algebra
        diagnostic_df = df.loc[target_class].copy()

        diagnostic_df["missing_required"] = (
            ~valid_required.loc[target_class]
        )
        diagnostic_df["invalid_rules"] = (
            ~valid_rules.loc[target_class]
        )
        diagnostic_df["neither_player_known"] = (
            neither_player_known.loc[target_class]
        )
        diagnostic_df["both_players_known"] = (
            both_players_known.loc[target_class]
        )

        failure_columns = [
            "missing_required",
            "invalid_rules",
            "neither_player_known",
            "both_players_known"
        ]

        failed = diagnostic_df.loc[
            diagnostic_df[failure_columns].any(axis=1)
        ]

        print("\nREJECTED BULLET, BLITZ AND RAPID GAMES:")

        print(
            failed[
                [
                    "uuid",
                    "source_username",
                    "time_class",
                    "rules",
                    "white_username",
                    "black_username",
                    *failure_columns
                ]
            ].to_string(index=False)
        )

        missing_counts = (
            df.loc[target_class, required_columns]
                .isna()
                .sum()
                .sort_values(ascending=False)
        )

        print("\nMissing required fields:")
        print(missing_counts[missing_counts > 0])

    # update df with the mask
    df = df.loc[mask].copy()

    # add useful fields to help processing parquets later
    is_white = df["white_username"].isin(usernames)

    df["my_colour"] = is_white.map({
        True: "white",
        False: "black"
    })

    df["my_username"] = df["white_username"].where(
        is_white,
        df["black_username"]
    )

    df["my_rating"] = df["white_rating"].where(
        is_white,
        df["black_rating"]
    )

    df["opponent_rating"] = df["black_rating"].where(
        is_white,
        df["white_rating"]
    )

    df["rating_diff"] = df["my_rating"] - df["opponent_rating"]

    df = df.drop_duplicates(subset=["uuid"])
    df = df.sort_values(["source_username", "end_time"])

    return df.reset_index(drop=True)

def build_games_dataset() -> None:
    all_games = []

    for username in USERNAMES:
        user_games = load_user_games(username)
        all_games.extend(user_games)

    df = pd.DataFrame(all_games)

    print(df.head())
    print(df.columns)
    print(f"Loaded {len(df)} total games")

    ## actually filter/clean the data here...
    df_clean = clean_games_dataframe(df, True)

    print("\nRemaining games:", len(df_clean))

    output_path = PROCESSED_DIR / "games_raw.parquet"
    df_clean.to_parquet(output_path, index=False)

    print(f"Saved {output_path}")

if __name__ == "__main__":
    build_games_dataset()