import os
import glob
import pandas as pd

RAW_FOLDER = "data/raw/player_stats"
OUTPUT_FOLDER = "data/processed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

league_folders = [
    "PremierLeague",
    "LaLiga",
    "Bundesliga",
    "SerieA",
    "Ligue1"
]

all_data = []

print("=" * 60)
print("Combining Player Statistics...")
print("=" * 60)

for league in league_folders:

    folder = os.path.join(RAW_FOLDER, league)

    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    for file in csv_files:

        filename = os.path.basename(file)

        print(f"Reading {filename}")

        # -----------------------------
        # Read FBref CSV
        # -----------------------------
        df = pd.read_csv(
            file,
            skiprows=3,
            header=1,
            engine="python",
            on_bad_lines="skip"
        )

        # ----------------------------------------
        # Remove footer ("Provided by FBref...")
        # ----------------------------------------
        df = df[df["Rk"].notna()]
        df = df[df["Rk"] != "Provided by <a href"]

        # ----------------------------------------
        # Remove duplicated header rows if present
        # ----------------------------------------
        df = df[df["Rk"] != "Rk"]

        # ----------------------------------------
        # Remove Matches hyperlink column
        # ----------------------------------------
        if "Matches" in df.columns:
            df.drop(columns=["Matches"], inplace=True)

        # ----------------------------------------
        # Remove last ID column (-9999)
        # ----------------------------------------
        last_col = df.columns[-1]

        if last_col == "-9999":
            df.drop(columns=[last_col], inplace=True)

        # ----------------------------------------
        # Add League
        # ----------------------------------------
        df["League"] = league

        # ----------------------------------------
        # Add Season
        # ----------------------------------------
        season = filename[-9:-4]
        df["Season"] = season

        all_data.append(df)

# --------------------------------------------
# Combine
# --------------------------------------------
master_df = pd.concat(all_data, ignore_index=True)

# Remove duplicate rows
master_df.drop_duplicates(inplace=True)

# Reset index
master_df.reset_index(drop=True, inplace=True)

# Save
output_file = os.path.join(
    OUTPUT_FOLDER,
    "player_master.csv"
)

master_df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("Player Dataset Created Successfully")
print("=" * 60)

print(f"Rows    : {len(master_df)}")
print(f"Columns : {len(master_df.columns)}")
print(f"Saved   : {output_file}")