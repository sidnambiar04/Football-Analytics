import pandas as pd
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "football_matches_cleaned.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "pre_match_random_forest.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded")
print("Rows:", len(df))


# ============================================================
# DATE FEATURES
# ============================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["match_month"] = df["date"].dt.month
df["match_dayofweek"] = df["date"].dt.dayofweek


# ============================================================
# PRE-MATCH FEATURES
# ============================================================

features = [
    "league",
    "season",
    "div",
    "hometeam",
    "awayteam",
    "referee",
    "kickoff_hour",
    "time_of_day",
    "match_month",
    "match_dayofweek"
]

target = "fulltimeresult"


X = df[features].copy()
y = df[target].copy()


# ============================================================
# REMOVE INVALID TARGET ROWS
# ============================================================

valid_rows = y.notna()

X = X.loc[valid_rows]
y = y.loc[valid_rows]


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = [
    "league",
    "season",
    "div",
    "hometeam",
    "awayteam",
    "referee",
    "time_of_day"
]

numeric_features = [
    "kickoff_hour",
    "match_month",
    "match_dayofweek"
]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# TRAIN
# ============================================================

print("\nTraining pre-match Random Forest...")

pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# EVALUATION
# ============================================================

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========================================")
print("PRE-MATCH MODEL RESULTS")
print("========================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(
    f"Location: {MODEL_PATH}"
)