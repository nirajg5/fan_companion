
import os
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone

from src.cricket.match_summary import generate_match_summary_data


load_dotenv()

INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "matchpulse-summaries",
)
NAMESPACE = "match-summaries"

# Must produce 1024-dimensional vectors to match your index.
EMBEDDING_MODEL = "multilingual-e5-large"


_pinecone_client = None
_pinecone_index = None


def get_pinecone_client() -> Pinecone:
    """Initialize the Pinecone client once."""
    global _pinecone_client

    if _pinecone_client is None:
        api_key = os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY is missing from your .env file."
            )

        _pinecone_client = Pinecone(api_key=api_key)

    return _pinecone_client


def get_pinecone_index():
    """Connect to the existing Pinecone index."""
    global _pinecone_index

    if _pinecone_index is None:
        _pinecone_index = get_pinecone_client().Index(INDEX_NAME)

    return _pinecone_index


def build_summary_text(summary: dict[str, Any]) -> str:
    """Convert verified statistics into searchable summary text."""
    lines = [
        f"Historical cricket match ID: {summary['match_id']}",
        "Innings summaries:",
    ]

    for innings in summary["innings_summaries"]:
        lines.append(
            f"Innings {innings['innings']}: "
            f"{innings['batting_team']} scored "
            f"{innings['runs']}/{innings['wickets']} "
            f"in {innings['overs']} overs."
        )

        for batter in innings["top_batters"]:
            lines.append(
                f"Top batter: {batter['player']}, "
                f"{batter['runs']} runs from {batter['balls']} balls."
            )

        for bowler in innings["top_bowlers"]:
            lines.append(
                f"Bowler: {bowler['player']}, "
                f"{bowler['wickets']} wickets, "
                f"{bowler['runs_conceded']} runs conceded."
            )

    return "\n".join(lines)


def generate_embedding(text: str) -> list[float]:
    """Generate a hosted embedding without loading a local ML model."""
    pc = get_pinecone_client()

    response = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=[text],
        parameters={"input_type": "passage", "truncate": "END"},
    )

    if not response.data:
        raise RuntimeError("Pinecone returned no embedding.")

    embedding = response.data[0].values

    if len(embedding) != 1024:
        raise ValueError(
            f"Expected 1024 embedding dimensions, "
            f"received {len(embedding)}."
        )

    return list(embedding)


def store_match_summary(match_id: str) -> dict[str, Any]:
    """Calculate, embed, and store a historical match summary."""
    summary = generate_match_summary_data(match_id)
    summary_text = build_summary_text(summary)

    embedding = generate_embedding(summary_text)

    index = get_pinecone_index()

    response = index.upsert(
        vectors=[
            {
                "id": f"match-summary-{match_id}",
                "values": embedding,
                "metadata": {
                    "match_id": str(match_id),
                    "source_type": "historical_match_summary",
                    "summary_text": summary_text,
                },
            }
        ],
        namespace=NAMESPACE,
    )

    return {
        "match_id": str(match_id),
        "status": "stored",
        "namespace": NAMESPACE,
        "embedding_dimension": len(embedding),
        "upserted_count": response.upserted_count,
        "summary_text": summary_text,
    }


if __name__ == "__main__":
    result = store_match_summary("1001349")

    print("\nMatch summary stored successfully.")
    print(f"Match ID: {result['match_id']}")
    print(f"Embedding dimensions: {result['embedding_dimension']}")
    print(f"Vectors uploaded: {result['upserted_count']}")
    print(f"Namespace: {result['namespace']}")
    print("\nSummary:")
    print(result["summary_text"])
