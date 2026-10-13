
import json
from src.cricket.match_state import calculate_match_state


def delivery(runs, extras=None, wicket=False, innings=2):
    return {
        "innings": innings,
        "total_runs": runs,
        "extras_json": json.dumps(extras or {}),
        "wicket": wicket,
    }


def test_basic_score_and_run_rate():
    deliveries = [
        delivery(1),
        delivery(4),
        delivery(0),
        delivery(2),
        delivery(1),
        delivery(6),
    ]

    state = calculate_match_state(
        deliveries, innings=2, target=180
    )

    assert state["runs"] == 14
    assert state["legal_balls"] == 6
    assert state["overs"] == "1.0"
    assert state["run_rate"] == 14.0
    assert state["runs_required"] == 166
    assert state["balls_remaining"] == 114


def test_wide_does_not_count_as_legal_ball():
    deliveries = [
        delivery(1, extras={"wides": 1}),
        delivery(4),
    ]

    state = calculate_match_state(deliveries, innings=2)

    assert state["runs"] == 5
    assert state["legal_balls"] == 1
    assert state["overs"] == "0.1"


def test_wicket_count():
    deliveries = [
        delivery(0),
        delivery(1, wicket=True),
        delivery(2),
    ]

    state = calculate_match_state(deliveries, innings=2)

    assert state["runs"] == 3
    assert state["wickets"] == 1
    assert state["legal_balls"] == 3


def test_first_innings_is_not_counted():
    deliveries = [
        delivery(50, innings=1),
        delivery(10, innings=2),
    ]

    state = calculate_match_state(deliveries, innings=2)

    assert state["runs"] == 10
    assert state["legal_balls"] == 1
