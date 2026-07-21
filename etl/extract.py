import json
import requests

from config.settings import (
    BASE_URL,
    PREMIER_LEAGUE,
    HEADERS,
    MATCHES_FILE,
)

def fetch_matches():
    url = f"{BASE_URL}/competitions/{PREMIER_LEAGUE}/matches"

    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        raise Exception(
            f"API Error {response.status_code}: {response.text}"
        )

    data = response.json()

    MATCHES_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(MATCHES_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print(f"✅ Saved matches to:\n{MATCHES_FILE}")

    return data


if __name__ == "__main__":
    fetch_matches()