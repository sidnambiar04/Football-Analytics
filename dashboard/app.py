import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "best_random_forest.pkl"
MATCH_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "football_matches_cleaned.csv"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Football Analytics Dashboard",
    page_icon="⚽",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_match_data():
    df = pd.read_csv(MATCH_DATA_PATH)

    # Convert date column if available
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


model = load_model()
matches = load_match_data()


# ============================================================
# HEADER
# ============================================================

st.title("⚽ Football Analytics & Match Prediction")
st.markdown(
    """
    An interactive football analytics dashboard for exploring match data,
    team performance and machine-learning predictions.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Match Prediction",
        "Team Performance",
        "Model Explainability",
        "Model Metrics",
        "Drift Detection"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Project Overview")

    st.markdown(
        """
        ### Football Analytics System

        This dashboard brings together the major components developed
        throughout the Applied Data Science experiments:

        - Football match data analysis
        - Machine-learning based match outcome prediction
        - Team and player performance analysis
        - Model explainability
        - Model evaluation
        - Data drift monitoring
        - Responsible AI reporting
        """
    )

    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Matches", f"{len(matches):,}")

    if "league" in matches.columns:
        col2.metric(
            "Leagues",
            matches["league"].nunique()
        )
    else:
        col2.metric("Leagues", "N/A")

    if "season" in matches.columns:
        col3.metric(
            "Seasons",
            matches["season"].nunique()
        )
    else:
        col3.metric("Seasons", "N/A")

    if "hometeam" in matches.columns:
        col4.metric(
            "Teams",
            len(
                set(matches["hometeam"].dropna())
                |
                set(matches["awayteam"].dropna())
            )
        )
    else:
        col4.metric("Teams", "N/A")

    st.subheader("Match Data")

    st.dataframe(
        matches.head(10),
        use_container_width=True
    )


# ============================================================
# MATCH PREDICTION
# ============================================================

elif page == "Match Prediction":

    st.header("⚽ Match Prediction")
    
    st.subheader("Model Input")

    col1, col2 = st.columns(2)

    with col1:

        league = st.selectbox(
            "League",
            sorted(matches["league"].dropna().unique())
        )

        season = st.selectbox(
            "Season",
            sorted(matches["season"].dropna().unique())
        )

        div = st.text_input(
            "Division",
            value="D1"
        )

        teams = sorted(
            set(matches["hometeam"].dropna())
            |
            set(matches["awayteam"].dropna())
        )

        hometeam = st.selectbox(
            "Home Team",
            teams
        )

        awayteam = st.selectbox(
            "Away Team",
            teams
        )

        referee = st.text_input(
            "Referee",
            value="Unknown"
        )

    with col2:

        kickoff_hour = st.number_input(
            "Kickoff Hour",
            min_value=0.0,
            max_value=23.0,
            value=19.0,
            step=1.0
        )

        time_of_day = st.selectbox(
            "Time of Day",
            [
                "Morning",
                "Afternoon",
                "Evening / Night"
            ]
        )

        shot_accuracy_away = st.number_input(
            "Away Shot Accuracy",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.01
        )

        total_shots = st.number_input(
            "Total Shots",
            min_value=0.0,
            value=20.0,
            step=1.0
        )

        total_shots_on_target = st.number_input(
            "Total Shots on Target",
            min_value=0.0,
            value=8.0,
            step=1.0
        )

        total_fouls = st.number_input(
            "Total Fouls",
            min_value=0.0,
            value=20.0,
            step=1.0
        )

        total_corners = st.number_input(
            "Total Corners",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

        match_month = st.number_input(
            "Match Month",
            min_value=1,
            max_value=12,
            value=5,
            step=1
        )

        match_dayofweek = st.number_input(
            "Match Day of Week",
            min_value=0,
            max_value=6,
            value=5,
            step=1
        )

    if st.button("Predict Match Outcome", type="primary"):

        input_data = pd.DataFrame(
            [{
                "league": league,
                "season": season,
                "div": div,
                "hometeam": hometeam,
                "awayteam": awayteam,
                "referee": referee,
                "kickoff_hour": kickoff_hour,
                "time_of_day": time_of_day,
                "shot_accuracy_away": shot_accuracy_away,
                "total_shots": total_shots,
                "total_shots_on_target": total_shots_on_target,
                "total_fouls": total_fouls,
                "total_corners": total_corners,
                "match_month": match_month,
                "match_dayofweek": match_dayofweek
            }]
        )

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        probability_dict = {
            str(cls): float(prob)
            for cls, prob in zip(
                model.classes_,
                probabilities
            )
        }

        result_labels = {
            "H": "Home Win",
            "D": "Draw",
            "A": "Away Win"
        }

        predicted_label = result_labels.get(
            str(prediction),
            "Unknown"
        )

        st.success(
            f"Predicted Outcome: **{predicted_label}**"
        )

        st.subheader("Prediction Probabilities")

        pcol1, pcol2, pcol3 = st.columns(3)

        pcol1.metric(
            "Home Win",
            f"{probability_dict.get('H', 0) * 100:.2f}%"
        )

        pcol2.metric(
            "Draw",
            f"{probability_dict.get('D', 0) * 100:.2f}%"
        )

        pcol3.metric(
            "Away Win",
            f"{probability_dict.get('A', 0) * 100:.2f}%"
        )


# ============================================================
# TEAM PERFORMANCE
# ============================================================

elif page == "Team Performance":

    st.title("⚽ Team Performance")

    st.write(
        "Analyse historical team performance using the cleaned football match dataset."
    )

    # -----------------------------
    # Team selection
    # -----------------------------

    leagues = sorted(matches["league"].dropna().unique())

    selected_league = st.selectbox(
        "Select League",
        leagues
    )

    league_df = matches[matches["league"] == selected_league].copy()

    teams = sorted(
        set(league_df["hometeam"].dropna().unique())
        | set(league_df["awayteam"].dropna().unique())
    )

    selected_team = st.selectbox(
        "Select Team",
        teams
    )

    # -----------------------------
    # Filter selected team matches
    # -----------------------------

    team_home = league_df[
        league_df["hometeam"] == selected_team
    ].copy()

    team_away = league_df[
        league_df["awayteam"] == selected_team
    ].copy()

    matches_played = len(team_home) + len(team_away)

    # -----------------------------
    # Calculate results
    # -----------------------------

    wins = 0
    draws = 0
    losses = 0
    goals_scored = 0
    goals_conceded = 0

    for _, row in team_home.iterrows():

        goals_scored += row["fulltimehomegoals"]
        goals_conceded += row["fulltimeawaygoals"]

        if row["fulltimeresult"] == "H":
            wins += 1
        elif row["fulltimeresult"] == "D":
            draws += 1
        elif row["fulltimeresult"] == "A":
            losses += 1

    for _, row in team_away.iterrows():

        goals_scored += row["fulltimeawaygoals"]
        goals_conceded += row["fulltimehomegoals"]

        if row["fulltimeresult"] == "A":
            wins += 1
        elif row["fulltimeresult"] == "D":
            draws += 1
        elif row["fulltimeresult"] == "H":
            losses += 1

    win_percentage = (
        (wins / matches_played) * 100
        if matches_played > 0
        else 0
    )

    avg_goals = (
        goals_scored / matches_played
        if matches_played > 0
        else 0
    )

    # -----------------------------
    # Metrics
    # -----------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Matches", matches_played)
    col2.metric("Wins", wins)
    col3.metric("Draws", draws)
    col4.metric("Losses", losses)
    col5.metric("Win %", f"{win_percentage:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Goals Scored",
            int(goals_scored)
        )

    with col2:
        st.metric(
            "Goals Conceded",
            int(goals_conceded)
        )

    st.metric(
        "Average Goals / Match",
        f"{avg_goals:.2f}"
    )

    # -----------------------------
    # Home vs Away
    # -----------------------------

    st.subheader("🏠 Home vs Away Performance")

    home_wins = len(
        team_home[team_home["fulltimeresult"] == "H"]
    )

    home_draws = len(
        team_home[team_home["fulltimeresult"] == "D"]
    )

    home_losses = len(
        team_home[team_home["fulltimeresult"] == "A"]
    )

    away_wins = len(
        team_away[team_away["fulltimeresult"] == "A"]
    )

    away_draws = len(
        team_away[team_away["fulltimeresult"] == "D"]
    )

    away_losses = len(
        team_away[team_away["fulltimeresult"] == "H"]
    )

    performance_df = pd.DataFrame({
        "Location": ["Home", "Away"],
        "Wins": [home_wins, away_wins],
        "Draws": [home_draws, away_draws],
        "Losses": [home_losses, away_losses]
    })

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------
    # Recent Matches
    # -----------------------------

    st.subheader("📅 Recent Matches")

    recent_home = team_home[
        [
            "date",
            "hometeam",
            "awayteam",
            "fulltimehomegoals",
            "fulltimeawaygoals",
            "fulltimeresult"
        ]
    ].copy()

    recent_away = team_away[
        [
            "date",
            "hometeam",
            "awayteam",
            "fulltimehomegoals",
            "fulltimeawaygoals",
            "fulltimeresult"
        ]
    ].copy()

    recent_matches = pd.concat(
        [recent_home, recent_away],
        ignore_index=True
    )

    recent_matches["date"] = pd.to_datetime(
        recent_matches["date"],
        errors="coerce"
    )

    recent_matches = recent_matches.sort_values(
        "date",
        ascending=False
    )

    st.dataframe(
        recent_matches.head(10),
        use_container_width=True,
        hide_index=True
    )

#============================================================

# ============================================================
# MODEL EXPLAINABILITY
# ============================================================
elif page == "Model Explainability":

    st.title("🔍 Model Explainability")

    st.write(
        "This section explains the pre-match Random Forest model "
        "using feature importance grouped by the original input features."
    )

    # ---------------------------------------------------------
    # Load PRE-MATCH model
    # ---------------------------------------------------------

    explain_model = joblib.load(
        BASE_DIR / "model" / "pre_match_random_forest.pkl"
    )

    preprocessor = explain_model.named_steps["preprocessor"]
    rf_model = explain_model.named_steps["model"]

    # ---------------------------------------------------------
    # Original feature groups
    # ---------------------------------------------------------

    original_features = [
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

    transformed_features = (
        preprocessor.get_feature_names_out()
    )

    importances = rf_model.feature_importances_

    # ---------------------------------------------------------
    # Map encoded features back to original features
    # ---------------------------------------------------------

    grouped_importance = {}

    for original_feature in original_features:

        total_importance = 0

        for feature_name, importance in zip(
            transformed_features,
            importances
        ):

            clean_name = feature_name.replace(
                "num__",
                ""
            ).replace(
                "cat__",
                ""
            )

            if clean_name == original_feature:
                total_importance += importance

            elif clean_name.startswith(
                original_feature + "_"
            ):
                total_importance += importance

        grouped_importance[original_feature] = (
            total_importance
        )

    importance_df = pd.DataFrame({
        "Feature": list(grouped_importance.keys()),
        "Importance": list(grouped_importance.values())
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    # Convert to percentage
    importance_df["Importance (%)"] = (
        importance_df["Importance"] * 100
    ).round(2)

    # ---------------------------------------------------------
    # Global Feature Importance
    # ---------------------------------------------------------

    st.subheader("📊 Global Feature Importance")

    st.write(
        "Feature importance shows how much each original input "
        "feature contributes to the Random Forest's decision-making."
    )

    st.bar_chart(
        importance_df.set_index("Feature")[
            "Importance (%)"
        ]
    )

    st.dataframe(
        importance_df[
            ["Feature", "Importance (%)"]
        ],
        use_container_width=True,
        hide_index=True
    )

    # ---------------------------------------------------------
    # Interpretation
    # ---------------------------------------------------------

    top_feature = importance_df.iloc[0]

    st.info(
        f"The most influential original feature in the model "
        f"is **{top_feature['Feature']}**, with approximately "
        f"**{top_feature['Importance (%)']:.2f}%** of the "
        f"Random Forest feature importance."
    )

    # ---------------------------------------------------------
    # Model information
    # ---------------------------------------------------------

    st.subheader("ℹ️ Explainability Notes")

    st.markdown(
        """
        **Model:** Pre-match Random Forest

        **Prediction classes:** Home Win (H), Draw (D), Away Win (A)

        **Features used:** Only information available before the match.

        **Important:** Categorical variables such as teams, leagues,
        seasons and referees are internally converted into multiple
        one-hot encoded variables. These encoded variables are grouped
        back into their original feature categories above.

        Therefore, the dashboard reports importance for meaningful
        football features instead of displaying hundreds of individual
        encoded categories.
        """
    )

#============================================================
# MODEL METRICS
# ============================================================

elif page == "Model Metrics":

    st.title("📈 Model Metrics")

    st.write(
        "Evaluation metrics for the pre-match Random Forest "
        "model using the historical football match dataset."
    )

    # ---------------------------------------------------------
    # Prepare evaluation dataset
    # ---------------------------------------------------------

    metrics_df = pd.read_csv(
        BASE_DIR / "data" / "processed" / "football_matches_cleaned.csv"
    )

    metrics_df["date"] = pd.to_datetime(
        metrics_df["date"],
        errors="coerce"
    )

    # Calendar features
    metrics_df["match_month"] = (
        metrics_df["date"].dt.month
    )

    metrics_df["match_dayofweek"] = (
        metrics_df["date"].dt.dayofweek
    )

    # Remove rows with missing target/features
    metrics_df = metrics_df.dropna(
        subset=[
            "fulltimeresult",
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
    )

    # ---------------------------------------------------------
    # Features and target
    # ---------------------------------------------------------

    metric_features = [
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

    X = metrics_df[metric_features]

    y = metrics_df["fulltimeresult"]

    # ---------------------------------------------------------
    # Same 80/20 stratified split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ---------------------------------------------------------
    # Load pre-match model
    # ---------------------------------------------------------

    metric_model = joblib.load(
        BASE_DIR / "model" / "pre_match_random_forest.pkl"
    )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    y_pred = metric_model.predict(X_test)

    # ---------------------------------------------------------
    # Calculate metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # ---------------------------------------------------------
    # Display metric cards
    # ---------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )

    st.divider()

    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------

    st.subheader("🔢 Confusion Matrix")

    labels = ["A", "D", "H"]

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Away",
            "Actual Draw",
            "Actual Home"
        ],
        columns=[
            "Predicted Away",
            "Predicted Draw",
            "Predicted Home"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    # ---------------------------------------------------------
    # Classification Report
    # ---------------------------------------------------------

    st.subheader("📋 Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=[
            "Away Win",
            "Draw",
            "Home Win"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    # ---------------------------------------------------------
    # Dataset information
    # ---------------------------------------------------------

    st.subheader("ℹ️ Evaluation Information")

    st.write(
        f"**Total historical matches:** {len(metrics_df)}"
    )

    st.write(
        f"**Training samples:** {len(X_train)}"
    )

    st.write(
        f"**Testing samples:** {len(X_test)}"
    )

    st.write(
        "**Split:** 80% training / 20% testing"
    )

    st.write(
        "**Split method:** Stratified train-test split "
        "with random_state = 42"
    )

    st.info(
        "These metrics evaluate the pre-match model using only "
        "features that are available before a football match."
    )

# ============================================================
# DRIFT DETECTION
# ============================================================

elif page == "Drift Detection":

    st.title("📉 Drift Detection")

    st.write(
        "This section compares two periods of historical football data "
        "to identify changes in numerical feature distributions."
    )

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    drift_df = pd.read_csv(
        BASE_DIR / "data" / "processed" / "football_matches_cleaned.csv"
    )

    drift_df["date"] = pd.to_datetime(
        drift_df["date"],
        errors="coerce"
    )

    drift_df = drift_df.dropna(
        subset=["date"]
    )

    # Sort chronologically
    drift_df = drift_df.sort_values("date")

    # ---------------------------------------------------------
    # Create reference and comparison datasets
    # ---------------------------------------------------------

    split_index = len(drift_df) // 2

    reference_df = drift_df.iloc[:split_index].copy()

    comparison_df = drift_df.iloc[split_index:].copy()

    # ---------------------------------------------------------
    # Display periods
    # ---------------------------------------------------------

    st.subheader("📅 Comparison Periods")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Reference Matches",
            len(reference_df)
        )

        st.write(
            f"From **{reference_df['date'].min().date()}** "
            f"to **{reference_df['date'].max().date()}**"
        )

    with col2:

        st.metric(
            "Comparison Matches",
            len(comparison_df)
        )

        st.write(
            f"From **{comparison_df['date'].min().date()}** "
            f"to **{comparison_df['date'].max().date()}**"
        )

    st.divider()

    # ---------------------------------------------------------
    # Numerical features
    # ---------------------------------------------------------

    numerical_features = [
        "kickoff_hour",
        "total_goals",
        "goal_difference",
        "total_cards",
        "total_shots",
        "total_shots_on_target",
        "total_fouls",
        "total_corners",
        "shot_accuracy_home",
        "shot_accuracy_away"
    ]

    # ---------------------------------------------------------
    # KS Test
    # ---------------------------------------------------------

    from scipy.stats import ks_2samp

    drift_results = []

    for feature in numerical_features:

        reference_values = pd.to_numeric(
            reference_df[feature],
            errors="coerce"
        ).dropna()

        comparison_values = pd.to_numeric(
            comparison_df[feature],
            errors="coerce"
        ).dropna()

        if len(reference_values) > 0 and len(comparison_values) > 0:

            statistic, p_value = ks_2samp(
                reference_values,
                comparison_values
            )

            drift_results.append({
                "Feature": feature,
                "KS Statistic": statistic,
                "p-value": p_value,
                "Drift Detected": (
                    "Yes"
                    if p_value < 0.05
                    else "No"
                )
            })

    drift_results_df = pd.DataFrame(
        drift_results
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    st.subheader("🔬 Statistical Drift Analysis")

    st.write(
        "The Kolmogorov–Smirnov test compares the distributions "
        "of each numerical feature between the reference and "
        "comparison periods."
    )

    st.dataframe(
        drift_results_df.style.format({
            "KS Statistic": "{:.4f}",
            "p-value": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # ---------------------------------------------------------
    # Overall drift
    # ---------------------------------------------------------

    drift_count = (
        drift_results_df["Drift Detected"] == "Yes"
    ).sum()

    total_features = len(
        drift_results_df
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Features Tested",
            total_features
        )

    with col2:

        st.metric(
            "Features With Drift",
            drift_count
        )

    if drift_count == 0:

        st.success(
            "No statistically significant distribution drift "
            "was detected at the 5% significance level."
        )

    else:

        st.warning(
            f"{drift_count} out of {total_features} numerical "
            "features show statistically significant distribution change."
        )

    # ---------------------------------------------------------
    # Feature distribution comparison
    # ---------------------------------------------------------

    st.subheader("📊 Feature Distribution Comparison")

    selected_feature = st.selectbox(
        "Select a feature",
        numerical_features
    )

    reference_plot = reference_df[
        selected_feature
    ].dropna()

    comparison_plot = comparison_df[
        selected_feature
    ].dropna()

    distribution_df = pd.DataFrame({
        "Reference": reference_plot.reset_index(drop=True),
        "Comparison": comparison_plot.reset_index(drop=True)
    })

    st.line_chart(
        distribution_df
    )

    # ---------------------------------------------------------
    # Interpretation
    # ---------------------------------------------------------

    st.subheader("ℹ️ Interpretation")

    st.markdown(
        """
        **KS Statistic:** Measures the maximum difference between
        the two feature distributions.

        **p-value:** Indicates whether the observed difference is
        statistically significant.

        **Drift threshold:** A p-value below **0.05** is treated
        as evidence of statistically significant distribution change.

        A detected drift does not automatically mean that the model
        has become inaccurate. It indicates that the input data
        distribution has changed and should be investigated further.
        """
    )