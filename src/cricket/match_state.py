
from typing import Any


def calculate_match_state(
    deliveries: list[dict[str, Any]],
    innings: int,
    target: int | None = None,
    max_overs: int = 20,
) -> dict[str, Any]:
    """
    Calculate a simplified T20 match state from ordered deliveries.

    Each delivery should contain:
        innings, total_runs, extras_json, wicket

    Optional:
        wicket_details_json

    `deliveries` must contain only the selected innings and must
    be ordered chronologically.
    """
    if max_overs <= 0:
        raise ValueError("max_overs must be positive.")

    if innings < 1:
        raise ValueError("innings must be at least 1.")

    runs = 0
    wickets = 0
    legal_balls = 0
    max_balls = max_overs * 6

    for delivery in deliveries:
        if delivery.get("innings") != innings:
            continue

        extras = delivery.get("extras_json", {})

        # CSV-loaded values may be JSON strings.
        if isinstance(extras, str):
            import json
            extras = json.loads(extras) if extras else {}

        if not isinstance(extras, dict):
            extras = {}

        total_runs = int(delivery.get("total_runs", 0) or 0)
        runs += total_runs

        if delivery.get("wicket", False):
            wickets += 1

        # Wides and no-balls are illegal deliveries.
        # Byes and leg-byes do not make a delivery illegal.
        is_legal = "wides" not in extras and "noballs" not in extras

        if is_legal:
            legal_balls += 1

    overs_completed = legal_balls // 6
    balls_in_current_over = legal_balls % 6

    overs_display = (
        f"{overs_completed}.{balls_in_current_over}"
    )

    run_rate = (
        runs / (legal_balls / 6)
        if legal_balls > 0
        else 0.0
    )

    balls_remaining = max(0, max_balls - legal_balls)

    runs_required = None
    required_run_rate = None

    if target is not None:
        runs_required = max(0, target - runs)

        if balls_remaining > 0 and runs < target:
            required_run_rate = (
                runs_required / (balls_remaining / 6)
            )
        elif runs >= target:
            required_run_rate = 0.0

    return {
        "innings": innings,
        "runs": runs,
        "wickets": wickets,
        "legal_balls": legal_balls,
        "overs": overs_display,
        "run_rate": round(run_rate, 2),
        "target": target,
        "runs_required": runs_required,
        "balls_remaining": balls_remaining,
        "required_run_rate": (
            round(required_run_rate, 2)
            if required_run_rate is not None
            else None
        ),
    }
