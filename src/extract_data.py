import json
import requests
from pathlib import Path

USERNAME = "nerf_ee"
ARCHIVES_URL = f"https://api.chess.com/pub/player/{USERNAME}/games/archives"

# chess.com's API recommend a recognisable User Agent to avoid error 403s
HEADERS = {
    "User-Agent": "Chess-Style-Policy-Net/0.1",
    "Accept": "application/json"
}

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
        pass

get_games()