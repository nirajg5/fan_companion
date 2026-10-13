
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "win_probability.joblib"


def load_model():
    """Load the trained model artifact."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run the training script first."
        )

    artifact = joblib.load(MODEL_PATH)

    if "pipeline" not in artifact or "features" not in artifact:
        raise ValueError("Invalid model artifact.")

    return artifact


def predict_win_probability(
    runs: int,
    wickets: int,
    legal_balls: int,
    target: int,
    max_overs: int = 20,
) -> dict:
    """
    Estimate the chasing team's win probability.

    This function supports standard T20 chases only.
    """
    if not 0 <= wickets <= 10:
        raise ValueError("Wickets must be between 0 and 10.")

    if max_overs <= 0:
        raise ValueError("max_overs must be positive.")

    max_balls = max_overs * 6

    if not 0 <= legal_balls <= max_balls:
        raise ValueError("Invalid number of legal balls.")

    if runs < 0 or target <= 0:
        raise ValueError("Runs cannot be negative; target must be positive.")

    if wickets == 10 or legal_balls == max_balls or runs >= target:
        chasing_team_wins = runs >= target and wickets < 10

        return {
            "win_probability": 100.0 if chasing_team_wins else 0.0,
            "status": "completed",
            "method": "deterministic_terminal_state",
        }

    balls_remaining = max_balls - legal_balls
    runs_required = target - runs

    run_rate = runs / (legal_balls / 6) if legal_balls else 0.0
    required_run_rate = runs_required / (balls_remaining / 6)

    features = {
        "runs": runs,
        "wickets": wickets,
        "legal_balls": legal_balls,
        "balls_remaining": balls_remaining,
        "target": target,
        "runs_required": runs_required,
        "run_rate": run_rate,
        "required_run_rate": required_run_rate,
    }

    artifact = load_model()
    pipeline = artifact["pipeline"]
    expected_features = artifact["features"]

    input_df = pd.DataFrame(
        [[features[name] for name in expected_features]],
        columns=expected_features,
    )

    probability = float(pipeline.predict_proba(input_df)[0][1])

    return {
        "win_probability": round(probability * 100, 2),
        "status": "in_progress",
        "method": "machine_learning",
        "interpretation": (
            "Estimated probability that the chasing team wins, "
            "assuming a standard T20 chase."
        ),
    }


if __name__ == "__main__":
    example = predict_win_probability(
        runs=120,
        wickets=4,
        legal_balls=90,
        target=180,
    )

    print(example)
