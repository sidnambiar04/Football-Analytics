import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

headers = {
    "X-Auth-Token": API_KEY
}

url = "https://api.football-data.org/v4/competitions/PL/matches"

response = requests.get(url, headers=headers)

print("Status Code:", response.status_code)

if response.status_code == 200:

    data = response.json()

    # Create folder if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)

    with open("data/raw/premier_league_matches.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print("Raw dataset saved successfully!")

else:
    print(response.text)