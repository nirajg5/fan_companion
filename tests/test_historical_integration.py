
import pytest

from src.cricket.historical_turning_points import (
    analyze_historical_turning_points,
)


def test_cricsheet_match_turning_point_pipeline():
    """Verify the real historical data-to-turning-points pipeline."""
    try:
        result = analyze_historical_turning_points(
            match_id="1001349",
            threshold=10.0,
        )
    except FileNotFoundError as exc:
        pytest.skip(f"Historical dataset unavailable: {exc}")

    assert result["match_id"] == "1001349"
    assert result["data_source"] == "Cricsheet historical replay"
    assert result["target"] == 169
    assert result["deliveries_analyzed"] > 0

    points = result["turning_points"]
    assert points, "Expected at least one detected turning point."

    for point in points:
        assert 0 <= point["previous_win_probability"] <= 100
        assert 0 <= point["current_win_probability"] <= 100

        expected_change = round(
            point["current_win_probability"]
            - point["previous_win_probability"],
            2,
        )
        assert point["change_percentage_points"] == expected_change

        if expected_change > 0:
            assert point["direction"] == "increased"
        elif expected_change < 0:
            assert point["direction"] == "decreased"

    indices = [point["state_index"] for point in points]
    assert indices == sorted(indices)
