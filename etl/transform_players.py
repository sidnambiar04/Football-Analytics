import pandas as pd
import os

INPUT_FILE = "data/processed/player_master.csv"
OUTPUT_FILE = "data/processed/player_master_clean.csv"

print("=" * 60)
print("Transforming Player Dataset")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

# ----------------------------------------------------
# Remove duplicate rows
# ----------------------------------------------------
df.drop_duplicates(inplace=True)

# ----------------------------------------------------
# Rename columns
# FBref exports duplicate names because of multi-level headers
# ----------------------------------------------------
new_columns = [
    "Rank",
    "Player",
    "Nation",
    "Position",
    "Club",
    "Age",
    "BirthYear",
    "MatchesPlayed",
    "Starts",
    "Minutes",
    "NinetyMinutes",
    "Goals",
    "Assists",
    "GoalContributions",
    "NonPenaltyGoals",
    "PenaltyGoals",
    "PenaltyAttempts",
    "YellowCards",
    "RedCards",
    "GoalsPer90",
    "AssistsPer90",
    "GoalContributionsPer90",
    "NonPenaltyGoalsPer90",
    "GoalContributionsPer90_NP",
    "League",
    "Season"
]

df.columns = new_columns

# ----------------------------------------------------
# Convert numeric columns
# ----------------------------------------------------
numeric_columns = [
    "Age",
    "BirthYear",
    "MatchesPlayed",
    "Starts",
    "Minutes",
    "NinetyMinutes",
    "Goals",
    "Assists",
    "GoalContributions",
    "NonPenaltyGoals",
    "PenaltyGoals",
    "PenaltyAttempts",
    "YellowCards",
    "RedCards",
    "GoalsPer90",
    "AssistsPer90",
    "GoalContributionsPer90",
    "NonPenaltyGoalsPer90",
    "GoalContributionsPer90_NP"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------------------------------------------
# Remove rows with missing player names
# ----------------------------------------------------
df = df[df["Player"].notna()]

# ----------------------------------------------------
# Remove rows with missing club names
# ----------------------------------------------------
df = df[df["Club"].notna()]

# ----------------------------------------------------
# Remove rows with missing league
# ----------------------------------------------------
df = df[df["League"].notna()]

# ----------------------------------------------------
# Reset index
# ----------------------------------------------------
df.reset_index(drop=True, inplace=True)

# ----------------------------------------------------
# Save cleaned dataset
# ----------------------------------------------------
os.makedirs("data/processed", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nTransformation Completed Successfully!")

print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\nColumn Names:\n")
print(df.columns.tolist())

print("\nFirst Five Rows:\n")
print(df.head())

print(f"\nSaved to: {OUTPUT_FILE}")