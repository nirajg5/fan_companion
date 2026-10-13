
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.cricket.turning_points import detect_turning_points
from src.cricket.historical_match import load_match_deliveries


DATA_PATH = Path("data/processed/deliveries.csv")



def build_delivery_states(match_id: str) -> dict[str, Any]:
    """Build second-innings states for one completed historical T20 match."""
    columns = [
        "match_type",
        "teams",
        "innings",
        "batting_team",
        "over",
        "delivery_in_over",
        "total_runs",
        "extras_json",
        "wicket",
    ]

    match = load_match_deliveries(match_id)

    missing_columns = [
        column for column in columns if column not in match.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Delivery dataset is missing required columns: {missing_columns}"
        )

    match = match[
        match["match_type"].str.upper() == "T20"
    ].copy()

    if match.empty:
        raise ValueError(f"T20 match {match_id} was not found.")

    first_innings = match[match["innings"] == 1]
    second_innings = match[match["innings"] == 2].copy()

    if first_innings.empty or second_innings.empty:
        raise ValueError("Both innings are required for this analysis.")

    first_total = int(first_innings["total_runs"].sum())
    target = first_total + 1

    second_innings = second_innings.sort_values(
        ["over", "delivery_in_over"]
    )

    runs = 0
    wickets = 0
    legal_balls = 0
    states: list[dict[str, Any]] = []

    for _, row in second_innings.iterrows():
        extras = json.loads(row["extras_json"] or "{}")
        total_runs = int(row["total_runs"])

        runs += total_runs

        if bool(row["wicket"]):
            wickets += 1

        if "wides" not in extras and "noballs" not in extras:
            legal_balls += 1

        states.append({
            "runs": runs,
            "wickets": wickets,
            "legal_balls": legal_balls,
            "target": target,
            "over": int(row["over"]),
            "delivery": int(row["delivery_in_over"]),
            "event": (
                f"{total_runs} run(s)"
                + (" and a wicket" if bool(row["wicket"]) else "")
            ),
            "description": (
                f"{row['batting_team']} scored {total_runs} "
                "run(s) on this recorded delivery."
            ),
        })

        if runs >= target or wickets >= 10 or legal_balls >= 120:
            break

    return {
        "match_id": str(match_id),
        "data_source": "Cricsheet historical replay",
        "teams": str(match["teams"].iloc[0]),
        "target": target,
        "delivery_states": states,
    }

def analyze_historical_turning_points(
    match_id: str,
    threshold: float = 10.0,
) -> dict[str, Any]:
    """Detect significant win-probability changes in a historical chase."""
    replay = build_delivery_states(match_id)

    points = detect_turning_points(
        delivery_states=replay["delivery_states"],
        threshold=threshold,
        max_overs=20,
    )

    return {
        "match_id": replay["match_id"],
        "data_source": replay["data_source"],
        "teams": replay["teams"],
        "target": replay["target"],
        "deliveries_analyzed": len(replay["delivery_states"]),
        "turning_points": points,
    }


if __name__ == "__main__":
    result = analyze_historical_turning_points("1001349")

    print(f"Match: {result['teams']}")
    print(f"Source: {result['data_source']}")
    print(f"Target: {result['target']}")
    print(f"Deliveries analyzed: {result['deliveries_analyzed']}")
    print(f"Turning points found: {len(result['turning_points'])}")

    for point in result["turning_points"]:
        print(point)
