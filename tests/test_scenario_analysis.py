
import json

from src.cricket.scenario_analysis import analyze_scenario


def test_scenario_analysis_returns_probability_comparison():
    history = [
        {
            "innings": 2,
            "total_runs": 1,
            "extras_json": json.dumps({}),
            "wicket": False,
        }
    ]

    result = analyze_scenario(
        historical_deliveries=history,
        innings=2,
        scenario=["six", "four", "wicket"],
        target=180,
    )

    assert result["before"]["runs"] == 1
    assert result["after"]["runs"] == 11
    assert result["after"]["wickets"] == 1

    assert 0 <= result["before"]["win_probability"] <= 100
    assert 0 <= result["after"]["win_probability"] <= 100

    assert isinstance(
        result["probability_change_percentage_points"], float
    )

    # Hypothetical events must not alter the original input.
    assert len(history) == 1
