
from src.cricket.turning_points import detect_turning_points


def test_requires_at_least_two_states():
    assert detect_turning_points([]) == []
    assert detect_turning_points([
        {
            "runs": 50,
            "wickets": 2,
            "legal_balls": 30,
            "target": 180,
        }
    ]) == []


def test_rejects_invalid_threshold():
    import pytest

    with pytest.raises(ValueError):
        detect_turning_points([], threshold=101)


def test_state_changes_have_valid_probabilities():
    states = [
        {
            "runs": 100,
            "wickets": 2,
            "legal_balls": 60,
            "target": 180,
            "event": "Previous delivery",
        },
        {
            "runs": 106,
            "wickets": 2,
            "legal_balls": 61,
            "target": 180,
            "event": "Six",
        },
    ]

    points = detect_turning_points(states, threshold=0)

    assert isinstance(points, list)

    for point in points:
        assert 0 <= point["previous_win_probability"] <= 100
        assert 0 <= point["current_win_probability"] <= 100
        assert point["direction"] in ["increased", "decreased"]


def test_detects_probability_change_at_exact_threshold(monkeypatch):
    from src.cricket import turning_points

    probabilities = iter([40.0, 50.0])

    def fake_predict(**kwargs):
        return {"win_probability": next(probabilities)}

    monkeypatch.setattr(
        turning_points,
        "predict_win_probability",
        fake_predict,
    )

    states = [
        {
            "runs": 100, "wickets": 2, "legal_balls": 60,
            "target": 180, "event": "Previous delivery",
        },
        {
            "runs": 106, "wickets": 2, "legal_balls": 61,
            "target": 180, "event": "Six",
        },
    ]

    points = turning_points.detect_turning_points(
        states, threshold=10.0
    )

    assert len(points) == 1
    assert points[0]["change_percentage_points"] == 10.0
    assert points[0]["direction"] == "increased"
    assert points[0]["event"] == "Six"


def test_ignores_probability_change_below_threshold(monkeypatch):
    from src.cricket import turning_points

    probabilities = iter([40.0, 49.99])

    monkeypatch.setattr(
        turning_points,
        "predict_win_probability",
        lambda **kwargs: {
            "win_probability": next(probabilities)
        },
    )

    states = [
        {"runs": 100, "wickets": 2, "legal_balls": 60, "target": 180},
        {"runs": 106, "wickets": 2, "legal_balls": 61, "target": 180},
    ]

    assert turning_points.detect_turning_points(
        states, threshold=10.0
    ) == []


def test_detects_decrease_and_preserves_chronological_order(monkeypatch):
    from src.cricket import turning_points

    probabilities = iter([80.0, 65.0, 80.0])

    monkeypatch.setattr(
        turning_points,
        "predict_win_probability",
        lambda **kwargs: {
            "win_probability": next(probabilities)
        },
    )

    states = [
        {
            "runs": 100, "wickets": 2, "legal_balls": 60,
            "target": 180, "over": 10, "delivery": 1,
            "event": "First state",
        },
        {
            "runs": 100, "wickets": 3, "legal_balls": 61,
            "target": 180, "over": 10, "delivery": 2,
            "event": "Wicket",
        },
        {
            "runs": 106, "wickets": 3, "legal_balls": 62,
            "target": 180, "over": 10, "delivery": 3,
            "event": "Six",
        },
    ]

    points = turning_points.detect_turning_points(
        states, threshold=10.0
    )

    assert len(points) == 2
    assert points[0]["event"] == "Wicket"
    assert points[0]["change_percentage_points"] == -15.0
    assert points[0]["direction"] == "decreased"
    assert points[1]["event"] == "Six"
    assert points[1]["change_percentage_points"] == 15.0
    assert points[1]["direction"] == "increased"

