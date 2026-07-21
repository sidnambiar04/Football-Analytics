import json
from config.settings import MATCHES_FILE


def transform_matches():
    # Load raw JSON
    with open(MATCHES_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Competition
    competition = {
        "competition_id": data["competition"].get("id"),
        "code": data["competition"].get("code"),
        "name": data["competition"].get("name"),
        "country": "England",  # Football-data v4 no longer returns 'area'
        "current_season": data.get("filters", {}).get("season"),
    }

    teams = {}
    matches = []

    for match in data.get("matches", []):

        home = match.get("homeTeam", {})
        away = match.get("awayTeam", {})

        # Home Team
        teams[home.get("id")] = {
            "team_id": home.get("id"),
            "competition_id": competition["competition_id"],
            "team_name": home.get("name"),
            "short_name": home.get("shortName"),
            "tla": home.get("tla"),
            "crest_url": home.get("crest"),
            "venue": home.get("venue"),
            "founded": home.get("founded"),
        }

        # Away Team
        teams[away.get("id")] = {
            "team_id": away.get("id"),
            "competition_id": competition["competition_id"],
            "team_name": away.get("name"),
            "short_name": away.get("shortName"),
            "tla": away.get("tla"),
            "crest_url": away.get("crest"),
            "venue": away.get("venue"),
            "founded": away.get("founded"),
        }

        score = match.get("score", {})
        full_time = score.get("fullTime", {})

        matches.append({
            "match_id": match.get("id"),
            "competition_id": competition["competition_id"],
            "season": data.get("filters", {}).get("season"),
            "match_date": match.get("utcDate"),
            "home_team_id": home.get("id"),
            "away_team_id": away.get("id"),
            "home_score": full_time.get("home"),
            "away_score": full_time.get("away"),
            "winner": score.get("winner"),
            "status": match.get("status"),
            "matchday": match.get("matchday"),
        })

    return competition, list(teams.values()), matches


if __name__ == "__main__":

    competition, teams, matches = transform_matches()

    print("\n========== TRANSFORMATION SUMMARY ==========\n")

    print(f"Competition : {competition['name']}")
    print(f"Country     : {competition['country']}")
    print(f"Season      : {competition['current_season']}")

    print(f"\nTotal Teams   : {len(teams)}")
    print(f"Total Matches : {len(matches)}")

    if teams:
        print("\nSample Team")
        print(teams[0])

    if matches:
        print("\nSample Match")
        print(matches[0])