import json
import time
import requests
from pathlib import Path

from config import USERNAMES

USERNAMES = USERNAMES
ARCHIVES_URL = """https://api.chess.com/pub/player/{username}/games/archives"""

# chess.com's API recommend a recognisable User Agent to avoid error 403s
HEADERS = {
    "User-Agent": "Chess-Style-Policy-Net/0.1",
    "Accept": "application/json"
}

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def construct_archive_url(username: str) -> str:
    return ARCHIVES_URL.format(username=username)

## helper
def get_json(url: str) -> dict:
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response.json()

def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_games(username) -> None:
    # case does not matter so standardise for file naming
    username = username.lower()

    user_raw_dir = RAW_DIR / username
    user_raw_dir.mkdir(parents=True, exist_ok=True)

    ## GET archives
    archives_url = construct_archive_url(username)
    archives = get_json(archives_url)

    # save JSON
    archives_output_path = Path("../data/archives.json")
    save_json(archives, archives_output_path)
    
    ## GET monthly game info
    for archive_url in archives["archives"]:
        # example of the URL structure
        # https://api.chess/pub/player/nerf_ee/games/2023/01
        parts = archive_url.rstrip("/").split("/")
        year = parts[-2]
        month = parts[-1]

        output_path = user_raw_dir / f"games_{year}_{month}.json"

        # skip already downloaded files
        if output_path.exists():
            print(f"Skipping existing {output_path.resolve()}")
            continue

        month_data = get_json(archive_url)
        save_json(month_data, output_path)
        
        print(f"Saved {output_path}")

        # adding a delay to not overload the chess.com API
        time.sleep(1)

def download_games() -> None:
    for username in USERNAMES:
        print(f"Downloading games for {username}")
        get_games(username)

if __name__ == "__main__":
    download_games()