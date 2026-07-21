from database.db import get_connection


def load_data(competition, teams, matches):
    conn = get_connection()

    if conn is None:
        return

    cursor = conn.cursor()

    try:
        # -----------------------------
        # Competition
        # -----------------------------
        cursor.execute("""
            INSERT INTO raw_data.competitions
            (competition_id, code, name, country, current_season)

            VALUES (%s,%s,%s,%s,%s)

            ON CONFLICT (competition_id)

            DO UPDATE SET
                code = EXCLUDED.code,
                name = EXCLUDED.name,
                country = EXCLUDED.country,
                current_season = EXCLUDED.current_season;
        """,
        (
            competition["competition_id"],
            competition["code"],
            competition["name"],
            competition["country"],
            competition["current_season"]
        ))

        # -----------------------------
        # Teams
        # -----------------------------
        for team in teams:

            cursor.execute("""
                INSERT INTO raw_data.teams
                (
                    team_id,
                    competition_id,
                    team_name,
                    short_name,
                    tla,
                    crest_url,
                    venue,
                    founded
                )

                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)

                ON CONFLICT (team_id)

                DO UPDATE SET
                    competition_id = EXCLUDED.competition_id,
                    team_name = EXCLUDED.team_name,
                    short_name = EXCLUDED.short_name,
                    tla = EXCLUDED.tla,
                    crest_url = EXCLUDED.crest_url,
                    venue = EXCLUDED.venue,
                    founded = EXCLUDED.founded;
            """,
            (
                team["team_id"],
                team["competition_id"],
                team["team_name"],
                team["short_name"],
                team["tla"],
                team["crest_url"],
                team["venue"],
                team["founded"]
            ))

        # -----------------------------
        # Matches
        # -----------------------------
        for match in matches:

            cursor.execute("""
                INSERT INTO raw_data.matches
                (
                    match_id,
                    competition_id,
                    season,
                    match_date,
                    home_team_id,
                    away_team_id,
                    home_score,
                    away_score,
                    winner,
                    status,
                    matchday
                )

                VALUES
                (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)

                ON CONFLICT (match_id)

                DO UPDATE SET
                    competition_id = EXCLUDED.competition_id,
                    season = EXCLUDED.season,
                    match_date = EXCLUDED.match_date,
                    home_team_id = EXCLUDED.home_team_id,
                    away_team_id = EXCLUDED.away_team_id,
                    home_score = EXCLUDED.home_score,
                    away_score = EXCLUDED.away_score,
                    winner = EXCLUDED.winner,
                    status = EXCLUDED.status,
                    matchday = EXCLUDED.matchday;
            """,
            (
                match["match_id"],
                match["competition_id"],
                match["season"],
                match["match_date"],
                match["home_team_id"],
                match["away_team_id"],
                match["home_score"],
                match["away_score"],
                match["winner"],
                match["status"],
                match["matchday"]
            ))

        conn.commit()

        print("\n================================")
        print("✅ Data Loaded Successfully")
        print("================================")

        print(f"Competition : 1")
        print(f"Teams       : {len(teams)}")
        print(f"Matches     : {len(matches)}")

    except Exception as e:
        conn.rollback()
        print(e)

    finally:
        cursor.close()
        conn.close()