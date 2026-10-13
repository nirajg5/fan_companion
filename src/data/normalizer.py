
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "t20s_male_json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "deliveries.csv"


def find_match_files():
    """Find all historical match JSON files."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    return sorted(DATASET_DIR.rglob("*.json"))


def extract_deliveries(match, match_id):
    """Extract delivery-level information from one match."""
    rows = []
    info = match.get("info", {})

    for innings_number, innings in enumerate(
        match.get("innings", []), start=1
    ):
        batting_team = innings.get("team")
        overs = innings.get("overs", [])

        for over_index, over in enumerate(overs):
            for delivery_index, delivery in enumerate(
                over.get("deliveries", []), start=1
            ):
                batter = delivery.get("batter")
                bowler = delivery.get("bowler")
                non_striker = delivery.get("non_striker")
                runs = delivery.get("runs", {})
                extras = delivery.get("extras", {})
                wickets = delivery.get("wickets", [])

                total_runs = runs.get("total", 0)
                batter_runs = runs.get("batter", 0)
                extra_runs = runs.get("extras", 0)

                rows.append({
                    "match_id": match_id,
                    "match_type": info.get("match_type"),
                    "teams": " vs ".join(info.get("teams", [])),
                    "innings": innings_number,
                    "batting_team": batting_team,
                    "over": over_index + 1,
                    "delivery_in_over": delivery_index,
                    "batter": batter,
                    "bowler": bowler,
                    "non_striker": non_striker,
                    "batter_runs": batter_runs,
                    "extra_runs": extra_runs,
                    "total_runs": total_runs,
                    "extras_json": json.dumps(extras),
                    "wicket": bool(wickets),
                    "wicket_details_json": json.dumps(wickets),
                })

    return rows


def main():
    files = find_match_files()

    if not files:
        raise RuntimeError(
            f"No JSON files found in {DATASET_DIR}"
        )

    rows = []
    failed_files = []

    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as file:
                match = json.load(file)

            # Use the dataset filename as a stable local identifier.
            match_id = file_path.stem

            rows.extend(extract_deliveries(match, match_id))

        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            failed_files.append((str(file_path), str(exc)))

    if not rows:
        raise RuntimeError(
            "No deliveries were extracted. Check the JSON schema."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Normalization completed.")
    print("Files discovered:", len(files))
    print("Files with errors:", len(failed_files))
    print("Matches represented:", df["match_id"].nunique())
    print("Delivery rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("Output:", OUTPUT_FILE)

    if failed_files:
        print("\nFirst five file errors:")
        for path, error in failed_files[:5]:
            print(path, "->", error)


if __name__ == "__main__":
    main()
