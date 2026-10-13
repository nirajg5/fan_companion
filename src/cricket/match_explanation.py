
def explain_match_situation(state: dict) -> dict:
    """
    Generate a simple explanation from a calculated match state.

    Expected fields:
        runs, wickets, overs, balls_remaining,
        target, runs_required, run_rate, required_run_rate
    """
    required_fields = [
        "runs",
        "wickets",
        "overs",
        "balls_remaining",
        "target",
        "runs_required",
        "run_rate",
        "required_run_rate",
    ]

    missing = [key for key in required_fields if key not in state]
    if missing:
        raise ValueError(f"Missing match-state fields: {missing}")

    runs = state["runs"]
    wickets = state["wickets"]
    overs = state["overs"]
    balls_remaining = state["balls_remaining"]
    target = state["target"]
    runs_required = state["runs_required"]
    run_rate = state["run_rate"]
    required_run_rate = state["required_run_rate"]

    if wickets >= 10:
        summary = (
            f"The innings has ended at {runs} runs "
            f"for {wickets} wickets."
        )
    elif target is not None and runs >= target:
        summary = (
            f"The chasing team has reached its target of {target} runs."
        )
    elif target is None:
        summary = (
            f"The team has scored {runs}/{wickets} in {overs} overs. "
            f"Its current run rate is {run_rate:.2f} runs per over."
        )
    else:
        summary = (
            f"The chasing team has scored {runs}/{wickets} in "
            f"{overs} overs. It needs {runs_required} more runs "
            f"from {balls_remaining} balls."
        )

    if target is None or runs >= target or wickets >= 10:
        chase_assessment = None
    elif balls_remaining == 0:
        chase_assessment = "The scheduled overs have been exhausted."
    elif required_run_rate is None:
        chase_assessment = "Required run rate is unavailable."
    else:
        difference = required_run_rate - run_rate

        if difference <= 0:
            chase_assessment = (
                "The current scoring rate is sufficient for the "
                "remaining target, if maintained."
            )
        elif difference <= 2:
            chase_assessment = (
                "The required run rate is slightly above the "
                "current scoring rate."
            )
        elif difference <= 5:
            chase_assessment = (
                "The chasing team needs to increase its scoring rate."
            )
        else:
            chase_assessment = (
                "The chase requires a substantial increase in scoring rate."
            )

    return {
        "summary": summary,
        "chase_assessment": chase_assessment,
        "current_run_rate": run_rate,
        "required_run_rate": required_run_rate,
        "runs_required": runs_required,
        "balls_remaining": balls_remaining,
    }


if __name__ == "__main__":
    example_state = {
        "runs": 120,
        "wickets": 4,
        "overs": "15.0",
        "balls_remaining": 30,
        "target": 180,
        "runs_required": 60,
        "run_rate": 8.0,
        "required_run_rate": 12.0,
    }

    result = explain_match_situation(example_state)

    print(result["summary"])
    print(result["chase_assessment"])
