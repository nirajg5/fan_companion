
from typing import Any

from src.cricket.scenario_simulator import simulate_scenario
from src.cricket.predict_win_probability import predict_win_probability


def analyze_scenario(
    historical_deliveries: list[dict[str, Any]],
    innings: int,
    scenario: list[str],
    target: int,
    max_overs: int = 20,
) -> dict[str, Any]:
    """
    Compare the estimated win probability before and after
    a hypothetical delivery sequence.
    """

    simulation = simulate_scenario(
        historical_deliveries=historical_deliveries,
        innings=innings,
        scenario=scenario,
        target=target,
        max_overs=max_overs,
    )

    before = simulation["before"]
    after = simulation["after"]

    before_prediction = predict_win_probability(
        runs=before["runs"],
        wickets=before["wickets"],
        legal_balls=before["legal_balls"],
        target=target,
        max_overs=max_overs,
    )

    after_prediction = predict_win_probability(
        runs=after["runs"],
        wickets=after["wickets"],
        legal_balls=after["legal_balls"],
        target=target,
        max_overs=max_overs,
    )

    before_probability = before_prediction["win_probability"]
    after_probability = after_prediction["win_probability"]

    return {
        "scenario": scenario,
        "before": {
            "runs": before["runs"],
            "wickets": before["wickets"],
            "overs": before["overs"],
            "win_probability": before_probability,
        },
        "after": {
            "runs": after["runs"],
            "wickets": after["wickets"],
            "overs": after["overs"],
            "win_probability": after_probability,
        },
        "probability_change_percentage_points": round(
            after_probability - before_probability, 2
        ),
        "simulation": simulation,
        "warning": (
            "The probability change is conditional on the "
            "hypothetical deliveries occurring. Model accuracy "
            "and calibration have not yet been established."
        ),
    }


if __name__ == "__main__":
    import json

    sample_deliveries = [
        {
            "innings": 2,
            "total_runs": 1,
            "extras_json": json.dumps({}),
            "wicket": False,
        }
    ]

    result = analyze_scenario(
        historical_deliveries=sample_deliveries,
        innings=2,
        scenario=["six", "four", "wicket"],
        target=180,
    )

    print(json.dumps(result, indent=2))
