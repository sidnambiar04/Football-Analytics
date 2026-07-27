import pandas as pd
from pathlib import Path
import re

# ============================
# Paths
# ============================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "football_matches_master.csv"

# ============================
# League Name Mapping
# ============================

LEAGUE_MAP = {
    "PremierLeague": "Premier League",
    "LaLiga": "La Liga",
    "Bundesliga": "Bundesliga",
    "SerieA": "Serie A",
    "Ligue1": "Ligue 1"
}

# ============================
# Football Columns to Keep
# ============================

KEEP_COLUMNS = [
    "League",
    "Season",
    "Div",
    "Date",
    "Time",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HTR",
    "Referee",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR"
]

# ============================
# Store all DataFrames
# ============================

all_dataframes = []

print("=" * 60)
print("Combining Football Datasets")
print("=" * 60)

# ============================
# Read every league folder
# ============================

for league_folder in RAW_DATA_DIR.iterdir():

    if not league_folder.is_dir():
        continue

    folder_name = league_folder.name
    league_name = LEAGUE_MAP.get(folder_name, folder_name)

    print(f"\nLeague : {league_name}")

    for csv_file in sorted(league_folder.glob("*.csv")):

        print(f"Reading : {csv_file.name}")

        # Read CSV
        df = pd.read_csv(csv_file)

        # ----------------------------
        # Add League Column
        # ----------------------------
        df["League"] = league_name

        # ----------------------------
        # Extract Season
        # Example:
        # PL22-23.csv
        # LL24-25.csv
        # ----------------------------

        match = re.search(r'(\d{2})-(\d{2})', csv_file.stem)

        if match:
            start = match.group(1)
            end = match.group(2)
            season = f"20{start}-20{end}"
        else:
            season = "Unknown"

        df["Season"] = season

        # ----------------------------
        # Add Missing Columns
        # ----------------------------

        for column in KEEP_COLUMNS:
            if column not in df.columns:
                df[column] = None

        # ----------------------------
        # Keep only required columns
        # ----------------------------

        df = df[KEEP_COLUMNS]

        all_dataframes.append(df)

# ============================
# Merge All DataFrames
# ============================

master_df = pd.concat(all_dataframes, ignore_index=True)

# ============================
# Remove Duplicate Rows
# ============================

master_df.drop_duplicates(inplace=True)

# ============================
# Reset Index
# ============================

master_df.reset_index(drop=True, inplace=True)

# ============================
# Save Master Dataset
# ============================

master_df.to_csv(OUTPUT_FILE, index=False)

# ============================
# Summary
# ============================

print("\n" + "=" * 60)
print("MASTER DATASET CREATED")
print("=" * 60)

print(f"Total Matches : {len(master_df)}")
print(f"Total Columns : {len(master_df.columns)}")

print("\nMatches per League")

print(master_df["League"].value_counts())

print("\nSaved to:")

print(OUTPUT_FILE)

print("=" * 60)