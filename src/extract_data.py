import json
import time
import requests
from pathlib import Path

USERNAME = "nerf_ee"
ARCHIVES_URL = f"https://api.chess.com/pub/player/{USERNAME}/games/archives"

# chess.com's API recommend a recognisable User Agent to avoid error 403s
HEADERS = {
    "User-Agent": "Chess-Style-Policy-Net/0.1",
    "Accept": "application/json"
}

RAW_DIR = Path("../data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

## helper
def get_json(url):
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response.json()

## GET archives
def get_games():
    archives = get_json(ARCHIVES_URL)

    # save JSON
    output_path = Path("../data/archives.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(archives, f, indent=2)
    
    ## GET monthly game info
    for archive_url in archives["archives"]:
        month_data = get_json(archive_url)

        # example of the URL structure
        # https://api.chess/pub/player/nerf_ee/games/2023/01
        parts = archive_url.rstrip("/").split("/")
        year = parts[-2]
        month = parts[-1]

        output_path = RAW_DIR / f"games_{year}_{month}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(month_data, f, indent=2)
        
        print(f"Saved {output_path}")

        # adding a delay to not overload the chess.com API
        time.sleep(1)

get_games()