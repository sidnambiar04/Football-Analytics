from etl.extract import fetch_matches
from etl.transform import transform_matches
from etl.load import load_data


def run_pipeline():

    print("\n========== FOOTBALL ETL ==========\n")

    print("Step 1 : Extracting...")
    fetch_matches()

    print("Step 2 : Transforming...")
    competition, teams, matches = transform_matches()

    print("Step 3 : Loading...")
    load_data(
        competition,
        teams,
        matches
    )

    print("\n========== PIPELINE COMPLETE ==========\n")


if __name__ == "__main__":
    run_pipeline()