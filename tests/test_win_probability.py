
import pytest

from src.cricket.predict_win_probability import predict_win_probability


@pytest.mark.parametrize(
    "inputs",
    [
        {"runs": 100, "wickets": 11, "legal_balls": 60, "target": 180},
        {"runs": 100, "wickets": 3, "legal_balls": 121, "target": 180},
        {"runs": 100, "wickets": 3, "legal_balls": 60, "target": 0},
        {"runs": -1, "wickets": 3, "legal_balls": 60, "target": 180},
    ],
)
def test_invalid_inputs_raise_value_error(inputs):
    with pytest.raises(ValueError):
        predict_win_probability(**inputs)


def test_completed_chase_returns_100_percent():
    result = predict_win_probability(
        runs=180,
        wickets=5,
        legal_balls=100,
        target=180,
    )

    assert result["win_probability"] == 100.0
    assert result["status"] == "completed"


def test_all_out_before_target_returns_zero_percent():
    result = predict_win_probability(
        runs=150,
        wickets=10,
        legal_balls=110,
        target=180,
    )

    assert result["win_probability"] == 0.0
    assert result["status"] == "completed"
