
import json
import pytest

from src.cricket.scenario_simulator import simulate_scenario


def make_delivery(runs, extras=None, wicket=False):
    return {
        "innings": 2,
        "total_runs": runs,
        "extras_json": json.dumps(extras or {}),
        "wicket": wicket,
    }


def test_six_four_wicket_scenario():
    history = [make_delivery(1)]

    result = simulate_scenario(
        historical_deliveries=history,
        innings=2,
        scenario=["six", "four", "wicket"],
        target=180,
    )

    assert result["before"]["runs"] == 1
    assert result["after"]["runs"] == 11
    assert result["after"]["wickets"] == 1
    assert result["runs_added"] == 10
    assert result["wickets_added"] == 1
    assert len(history) == 1  # Original data is unchanged.


def test_wide_does_not_consume_a_legal_ball():
    result = simulate_scenario(
        historical_deliveries=[],
        innings=2,
        scenario=["wide"],
    )

    assert result["after"]["runs"] == 1
    assert result["after"]["legal_balls"] == 0


def test_unknown_event_is_rejected():
    with pytest.raises(ValueError, match="Unknown scenario"):
        simulate_scenario(
            historical_deliveries=[],
            innings=2,
            scenario=["magic_ball"],
        )


def test_target_reached_stops_remaining_events():
    history = [make_delivery(9)]

    result = simulate_scenario(
        historical_deliveries=history,
        innings=2,
        scenario=["six", "four"],
        target=15,
    )

    assert result["after"]["runs"] == 15
    assert result["applied_events"] == ["six"]
    assert result["events_skipped"] == 1



def test_no_ball_does_not_consume_a_legal_ball():
    result = simulate_scenario(
        historical_deliveries=[],
        innings=2,
        scenario=["no_ball"],
    )

    assert result["after"]["runs"] == 1
    assert result["after"]["legal_balls"] == 0
    assert result["applied_events"] == ["no_ball"]


def test_scenario_does_not_modify_original_delivery():
    history = [make_delivery(2, extras={"byes": 1})]
    original = json.loads(json.dumps(history))

    simulate_scenario(
        historical_deliveries=history,
        innings=2,
        scenario=["six", "wicket"],
        target=180,
    )

    assert history == original
