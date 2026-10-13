
import pandas as pd

from src.cricket.match_state import calculate_match_state


PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "deliveries.csv"


def load_deliveries() -> pd.DataFrame:
    """Load normalized historical deliveries."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Delivery dataset not found: {CSV_PATH}. "
            "Run the normalizer first."
        )

    return pd.read_csv(CSV_PATH)



def load_match_deliveries(match_id: str) -> pd.DataFrame:
    """Load only the rows belonging to one historical match."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Delivery dataset not found: {CSV_PATH}. "
            "Run the normalizer first."
        )

    match_id = str(match_id)

    # Read in chunks to avoid loading the full dataset into memory.
    chunks = []

    for chunk in pd.read_csv(
        CSV_PATH,
        dtype={"match_id": str},
        chunksize=50_000,
    ):
        matched = chunk[chunk["match_id"] == match_id]

        if not matched.empty:
            chunks.append(matched)

    if not chunks:
        raise ValueError(f"Match not found: {match_id}")

    return pd.concat(chunks, ignore_index=True)



def get_recorded_delivery_interval(
    match_id: str,
    innings: int,
    replay_position: int,
    start_over: int | None = None,
    end_over: int | None = None,
) -> pd.DataFrame:
    """Retrieve recorded deliveries within a selected replay interval."""

    deliveries = load_match_deliveries(str(match_id))

    innings_rows = deliveries[
        deliveries["innings"] == int(innings)
    ].copy()

    if innings_rows.empty:
        raise ValueError(
            f"Innings {innings} not found for match {match_id}."
        )

    innings_rows = innings_rows.sort_values(
        ["over", "delivery_in_over"]
    ).reset_index(drop=True)

    if replay_position < 0:
        raise ValueError("Replay position cannot be negative.")

    # Replay positions count recorded delivery rows, not legal balls.
    replay_position = min(replay_position, len(innings_rows))
    available = innings_rows.iloc[:replay_position].copy()

    if start_over is not None:
        available = available[
            available["over"] >= int(start_over)
        ]

    if end_over is not None:
        available = available[
            available["over"] <= int(end_over)
        ]

    if (
        start_over is not None
        and end_over is not None
        and start_over > end_over
    ):
        raise ValueError("Start over cannot exceed end over.")

    return available.reset_index(drop=True)


def get_historical_match_state(
    match_id: str,
    innings: int,
    target: int | None = None,
) -> dict:
    """Calculate the state at the end of a historical innings."""
    df = load_deliveries()

    match_rows = df[df["match_id"].astype(str) == str(match_id)]
    if match_rows.empty:
        raise ValueError(f"Match not found: {match_id}")

    innings_rows = match_rows[match_rows["innings"] == innings]
    if innings_rows.empty:
        raise ValueError(
            f"Innings {innings} not found for match {match_id}"
        )

    records = innings_rows.to_dict(orient="records")

    state = calculate_match_state(
        deliveries=records,
        innings=innings,
        target=target,
    )

    state["match_id"] = str(match_id)
    state["batting_team"] = innings_rows.iloc[0]["batting_team"]

    return state


if __name__ == "__main__":
    df = load_deliveries()

    match_id = str(df.iloc[0]["match_id"])
    innings = int(df.iloc[0]["innings"])

    result = get_historical_match_state(match_id, innings)

    print("Historical match state")
    for key, value in result.items():
        print(f"{key}: {value}")
