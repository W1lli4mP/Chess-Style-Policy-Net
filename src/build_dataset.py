import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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

def clean_data() -> None:
    all_games = []

    for username in USERNAMES:
        user_games = load_user_games(username)
        all_games.extend(user_games)

    df = pd.DataFrame(all_games)

    print(df.head())
    print(df.columns)
    print(f"Loaded {len(df)} total games")

    ## testing
    print(f'There are {len(df[df["time_class"] == "bullet"])} bullet games')
    print(f'There are {len(df[df["time_class"] == "blitz"])} blitz games')
    print(f'There are {len(df[df["time_class"] == "rapid"])} rapid games')
    print(f'There are {len(df[df["time_class"] == "daily"])} daily games')

    print(f'Sum: {len(df[df["time_class"] == "bullet"]) + len(df[df["time_class"] == "blitz"]) + len(df[df["time_class"] == "rapid"]) + len(df[df["time_class"] == "daily"])}')


    """
    dataframe -> arrow table -> parquet
    
    example
    df = pd.DataFrame(...)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, "example.parquet")
    """

    ## actually filter/clean the data here...

    output_path = PROCESSED_DIR / "games_raw.parquet"
    df.to_parquet(output_path, index=False)

    print(f"Saved {output_path}")

if __name__ == "__main__":
    clean_data()