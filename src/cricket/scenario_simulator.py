
import json
from typing import Any

from src.cricket.match_state import calculate_match_state


SCENARIOS = {
    "dot": {"total_runs": 0, "extras": {}, "wicket": False},
    "single": {"total_runs": 1, "extras": {}, "wicket": False},
    "four": {"total_runs": 4, "extras": {}, "wicket": False},
    "six": {"total_runs": 6, "extras": {}, "wicket": False},
    "wicket": {"total_runs": 0, "extras": {}, "wicket": True},
    "wide": {"total_runs": 1, "extras": {"wides": 1}, "wicket": False},
    "no_ball": {
        "total_runs": 1,
        "extras": {"noballs": 1},
        "wicket": False,
    },
}


def simulate_scenario(
    historical_deliveries: list[dict[str, Any]],
    innings: int,
    scenario: list[str],
    target: int | None = None,
    max_overs: int = 20,
) -> dict[str, Any]:
    """
    Apply hypothetical outcomes to a copy of historical deliveries.

    Historical deliveries must contain the selected innings in
    chronological order. The scenario uses SCENARIOS keys.
    """
    unknown = [event for event in scenario if event not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenario events: {unknown}")

    # Copy each record so the source data is never modified.
    simulated_deliveries = [dict(row) for row in historical_deliveries]

    before = calculate_match_state(
        simulated_deliveries,
        innings=innings,
        target=target,
        max_overs=max_overs,
    )

    if before["wickets"] >= 10:
        raise ValueError("Cannot simulate: the innings has ended.")

    if before["legal_balls"] >= max_overs * 6:
        raise ValueError("Cannot simulate: the over limit has been reached.")

    if target is not None and before["runs"] >= target:
        raise ValueError("Cannot simulate: the target has already been reached.")

    applied_events = []

    for event_name in scenario:
        current = calculate_match_state(
            simulated_deliveries,
            innings=innings,
            target=target,
            max_overs=max_overs,
        )

        # Stop when the innings reaches a terminal state.
        if current["wickets"] >= 10:
            break
        if current["legal_balls"] >= max_overs * 6:
            break
        if target is not None and current["runs"] >= target:
            break

        event = SCENARIOS[event_name]

        simulated_deliveries.append({
            "innings": innings,
            "total_runs": event["total_runs"],
            "extras_json": json.dumps(event["extras"]),
            "wicket": event["wicket"],
        })
        applied_events.append(event_name)

    after = calculate_match_state(
        simulated_deliveries,
        innings=innings,
        target=target,
        max_overs=max_overs,
    )

    return {
        "before": before,
        "after": after,
        "requested_scenario": scenario,
        "applied_events": applied_events,
        "events_skipped": len(scenario) - len(applied_events),
        "runs_added": after["runs"] - before["runs"],
        "wickets_added": after["wickets"] - before["wickets"],
    }
