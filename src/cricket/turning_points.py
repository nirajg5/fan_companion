
from typing import Any

from src.cricket.predict_win_probability import (
    predict_win_probability,
)


def detect_turning_points(
    delivery_states: list[dict[str, Any]],
    threshold: float = 10.0,
    max_overs: int = 20,
) -> list[dict[str, Any]]:
    """
    Detect large changes in estimated chasing-team win probability.

    Each state must contain:
        runs, wickets, legal_balls, target

    Optional:
        over, delivery, event, description

    States must be ordered chronologically and represent the
    match immediately after each delivery.
    """
    if threshold < 0 or threshold > 100:
        raise ValueError("threshold must be between 0 and 100.")

    if len(delivery_states) < 2:
        return []

    predictions = []

    for state in delivery_states:
        prediction = predict_win_probability(
            runs=int(state["runs"]),
            wickets=int(state["wickets"]),
            legal_balls=int(state["legal_balls"]),
            target=int(state["target"]),
            max_overs=max_overs,
        )

        predictions.append(prediction["win_probability"])

    turning_points = []

    for index in range(1, len(delivery_states)):
        previous_probability = predictions[index - 1]
        current_probability = predictions[index]

        change = round(
            current_probability - previous_probability, 2
        )

        if abs(change) < threshold:
            continue

        state = delivery_states[index]

        turning_points.append({
            "state_index": index,
            "over": state.get("over"),
            "delivery": state.get("delivery"),
            "event": state.get("event", "Unknown event"),
            "description": state.get("description", ""),
            "previous_win_probability": previous_probability,
            "current_win_probability": current_probability,
            "change_percentage_points": change,
            "direction": (
                "increased" if change > 0 else "decreased"
            ),
        })

    return turning_points


if __name__ == "__main__":
    # Demonstration only: these are example match states,
    # not real historical predictions.
    example_states = [
        {
            "over": "18.0",
            "delivery": 1,
            "runs": 145,
            "wickets": 5,
            "legal_balls": 108,
            "target": 180,
            "event": "Dot ball",
            "description": "The chasing team needs 35 runs.",
        },
        {
            "over": "18.1",
            "delivery": 2,
            "runs": 151,
            "wickets": 5,
            "legal_balls": 109,
            "target": 180,
            "event": "Six",
            "description": "A six reduces the required runs.",
        },
    ]

    points = detect_turning_points(
        example_states,
        threshold=10.0,
    )

    for point in points:
        print(point)
