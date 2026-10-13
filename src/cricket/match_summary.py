
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DELIVERIES_PATH = PROJECT_ROOT / "data" / "processed" / "deliveries.csv"


def generate_match_summary_data(
    match_id: str,
    innings: int | None = None,
) -> dict:
    """
    Calculate a verified summary from historical ball-by-ball data.

    This function does not use an LLM to calculate statistics.
    """
    if not DELIVERIES_PATH.exists():
        raise FileNotFoundError(
            f"Deliveries dataset not found: {DELIVERIES_PATH}"
        )

    df = pd.read_csv(
        DELIVERIES_PATH,
        dtype={"match_id": str},
    )

    match = df[df["match_id"] == str(match_id)].copy()

    if match.empty:
        raise ValueError(f"No deliveries found for match ID {match_id}")

    if innings is not None:
        match = match[match["innings"] == innings].copy()

    if match.empty:
        raise ValueError(
            f"No deliveries found for innings {innings} "
            f"of match {match_id}"
        )

    match = match.sort_values(
        ["innings", "over", "delivery_in_over"]
    )

    innings_summaries = []

    for innings_number, data in match.groupby("innings", sort=True):
        batting_team = str(data["batting_team"].iloc[0])

        total_runs = int(data["total_runs"].sum())
        wickets = int(data["wicket"].fillna(False).astype(bool).sum())

        legal_balls = 0
        for _, delivery in data.iterrows():
            try:
                extras = json.loads(delivery["extras_json"])
                extras = extras if isinstance(extras, dict) else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                extras = {}

            # Wides and no-balls do not count as legal deliveries.
            if "wides" not in extras and "noballs" not in extras:
                legal_balls += 1

        overs = f"{legal_balls // 6}.{legal_balls % 6}"

        batter_stats = (
            data.groupby("batter", dropna=True)
            .agg(
                runs=("batter_runs", "sum"),
                deliveries=("batter_runs", "count"),
            )
            .sort_values("runs", ascending=False)
        )

        top_batters = [
            {
                "player": str(player),
                "runs": int(row["runs"]),
                "balls": int(row["deliveries"]),
            }
            for player, row in batter_stats.head(3).iterrows()
        ]

        bowling_stats = []

        for bowler, bowler_data in data.groupby("bowler", dropna=True):
            wickets_taken = 0

            for _, delivery in bowler_data.iterrows():
                try:
                    details = json.loads(delivery["wicket_details_json"])
                    details = details if isinstance(details, list) else []
                except (TypeError, ValueError, json.JSONDecodeError):
                    details = []

                for dismissal in details:
                    kind = str(dismissal.get("kind", "")).lower()

                    # Exclude dismissals that are not credited to the bowler.
                    if kind not in {
                        "retired hurt",
                        "obstructing the field",
                        "run out",
                        "retired out",
                        "handled the ball",
                    }:
                        wickets_taken += 1

            bowling_stats.append(
                {
                    "player": str(bowler),
                    "runs_conceded": int(bowler_data["total_runs"].sum()),
                    "wickets": wickets_taken,
                }
            )

        bowling_stats.sort(
            key=lambda item: (
                item["wickets"],
                -item["runs_conceded"],
            ),
            reverse=True,
        )

        innings_summaries.append(
            {
                "innings": int(innings_number),
                "batting_team": batting_team,
                "runs": total_runs,
                "wickets": wickets,
                "overs": overs,
                "legal_balls": legal_balls,
                "top_batters": top_batters,
                "top_bowlers": bowling_stats[:3],
            }
        )

    return {
        "match_id": str(match_id),
        "innings_requested": innings,
        "innings_summaries": innings_summaries,
    }


if __name__ == "__main__":
    summary = generate_match_summary_data("1001349")
    print(json.dumps(summary, indent=2))
