
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "t20s_male_json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "win_probability_dataset.csv"

MAX_OVERS = 20
MAX_BALLS = MAX_OVERS * 6


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_innings_total(innings: dict) -> int:
    return sum(
        delivery.get("runs", {}).get("total", 0)
        for over in innings.get("overs", [])
        for delivery in over.get("deliveries", [])
    )


def build_chase_examples(match: dict, match_id: str) -> list[dict]:
    """Build second-innings states labelled by the actual match winner."""
    info = match.get("info", {})
    innings_list = match.get("innings", [])
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")

    # Restrict the initial dataset to standard completed T20 matches.
    if info.get("match_type") != "T20":
        return []

    if len(innings_list) != 2 or not winner:
        return []

    if outcome.get("method") == "D/L":
        return []

    first_innings, second_innings = innings_list

    if (
        first_innings.get("team") == second_innings.get("team")
        or winner not in info.get("teams", [])
    ):
        return []

    target = get_innings_total(first_innings) + 1
    if target <= 1:
        return []

    batting_team = second_innings.get("team")
    label = int(winner == batting_team)

    runs = 0
    wickets = 0
    legal_balls = 0
    examples = []

    for over in second_innings.get("overs", []):
        for delivery in over.get("deliveries", []):
            balls_remaining = MAX_BALLS - legal_balls

            # Record the situation BEFORE this delivery.
            # The label is the eventual match result.
            if runs < target and wickets < 10 and balls_remaining > 0:
                runs_required = target - runs

                examples.append({
                    "match_id": match_id,
                    "innings": 2,
                    "runs": runs,
                    "wickets": wickets,
                    "legal_balls": legal_balls,
                    "balls_remaining": balls_remaining,
                    "target": target,
                    "runs_required": runs_required,
                    "run_rate": (
                        runs / (legal_balls / 6)
                        if legal_balls > 0 else 0.0
                    ),
                    "required_run_rate": (
                        runs_required / (balls_remaining / 6)
                    ),
                    "batting_team_won": label,
                })

            # Apply the actual historical delivery.
            runs_data = delivery.get("runs", {})
            extras = delivery.get("extras", {})

            runs += int(runs_data.get("total", 0))
            wickets += len(delivery.get("wickets", []))

            # Wides and no-balls do not consume a legal ball.
            if "wides" not in extras and "noballs" not in extras:
                legal_balls += 1

            # Stop once the innings reaches a terminal state.
            if (
                runs >= target
                or wickets >= 10
                or legal_balls >= MAX_BALLS
            ):
                break

        if (
            runs >= target
            or wickets >= 10
            or legal_balls >= MAX_BALLS
        ):
            break

    # Avoid including states from innings with no recorded deliveries.
    return examples


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    files = sorted(DATASET_DIR.rglob("*.json"))
    if not files:
        raise RuntimeError("No historical JSON files found.")

    all_examples = []
    failed_files = 0

    for path in files:
        try:
            match = load_json(path)
            all_examples.extend(
                build_chase_examples(match, path.stem)
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            failed_files += 1
            print(f"Skipped {path.name}: {exc}")

    if not all_examples:
        raise RuntimeError(
            "No training examples created. Check the dataset schema "
            "and match eligibility conditions."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = list(all_examples[0].keys())
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_examples)

    match_ids = {row["match_id"] for row in all_examples}
    wins = sum(row["batting_team_won"] for row in all_examples)
    losses = len(all_examples) - wins

    print("\nDataset preparation completed.")
    print("JSON files discovered:", len(files))
    print("Files with errors:", failed_files)
    print("Matches represented:", len(match_ids))
    print("Training examples:", len(all_examples))
    print("Winning-state labels:", wins)
    print("Losing-state labels:", losses)
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
