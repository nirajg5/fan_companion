
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "t20s_male_json"


def list_match_files():
    """Find all historical match JSON files."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    return sorted(DATASET_DIR.rglob("*.json"))


def load_match(file_path):
    """Load one match from a JSON file."""
    with Path(file_path).open("r", encoding="utf-8") as file:
        match = json.load(file)

    if not isinstance(match, dict):
        raise ValueError("Expected a JSON object.")

    return match


def inspect_match(match):
    """Print basic match and delivery information."""
    info = match.get("info", {})
    innings = match.get("innings", [])

    print("Match type:", info.get("match_type"))
    print("Teams:", info.get("teams"))
    print("Innings:", len(innings))

    for index, inning in enumerate(innings, start=1):
        overs = inning.get("overs", [])
        deliveries = sum(
            len(over.get("deliveries", []))
            for over in overs
        )

        print(f"Innings {index}: {inning.get('team')}")
        print("Overs recorded:", len(overs))
        print("Deliveries recorded:", deliveries)


if __name__ == "__main__":
    files = list_match_files()

    if not files:
        print("No JSON match files found.")
    else:
        print("Sample file:", files[0])
        match = load_match(files[0])
        inspect_match(match)
