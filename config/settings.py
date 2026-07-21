import os
from pathlib import Path
from dotenv import load_dotenv

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# API Configuration
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.football-data.org/v4"

# Competition Codes
PREMIER_LEAGUE = "PL"

# Season
SEASON = "2025"

# Data Paths
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "PL" / "2025-2026"
MATCHES_FILE = RAW_DATA_DIR / "matches.json"

# Headers
HEADERS = {
    "X-Auth-Token": API_KEY
}