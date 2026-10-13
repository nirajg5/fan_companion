
import os

from dotenv import load_dotenv
from groq import Groq

from src.cricket.match_summary import generate_match_summary_data
from src.cricket.summary_embeddings import build_summary_text
from src.cricket.summary_retriever import search_match_summaries


load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"


def generate_natural_language_summary(
    match_id: str,
    question: str | None = None,
) -> dict:
    """
    Generate a match summary using verified match statistics,
    Pinecone-retrieved context, and Groq.
    """
    # 1. Calculate statistics directly from historical deliveries.
    summary_data = generate_match_summary_data(match_id)
    verified_summary = build_summary_text(summary_data)

    # 2. Retrieve relevant context from Pinecone.
    retrieved_matches = search_match_summaries(
        query=question or f"Summarize match {match_id}",
        top_k=3,
        match_id=match_id,
    )

    retrieved_context = "\n\n".join(
        item["summary_text"] for item in retrieved_matches
    )

    if not retrieved_context:
        retrieved_context = (
            "No indexed Pinecone context was found for this match."
        )

    # 3. Ask Groq to write the summary using verified data.
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from your .env file.")

    client = Groq(api_key=api_key)
    model_name = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    prompt = f"""
You are MatchPulse AI, a cricket match analyst.

Write a concise, readable historical match summary.
Use only the facts provided below.
Do not invent events, partnerships, turning points, or player statistics.
If information is unavailable, say so.
Distinguish match statistics from interpretation.

VERIFIED STATISTICS CALCULATED FROM DELIVERY DATA:
{verified_summary}

RETRIEVED PINECONE CONTEXT:
{retrieved_context}

USER REQUEST:
{question or "Summarize the match, including the innings scores, leading batters, and leading bowlers."}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You produce accurate cricket summaries. "
                    "Never contradict verified statistics."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        reasoning_effort="low",
    )

    generated_summary = response.choices[0].message.content

    return {
        "match_id": str(match_id),
        "model": model_name,
        "retrieved_context_count": len(retrieved_matches),
        "verified_statistics": summary_data,
        "summary": generated_summary,
    }


if __name__ == "__main__":
    result = generate_natural_language_summary(
        match_id="1001349",
        question="Summarize the match and highlight the leading performers.",
    )

    print("\n" + "=" * 60)
    print("MATCHPULSE AI - MATCH SUMMARY")
    print("=" * 60)
    print(result["summary"])
    print("\nRetrieved context records:", result["retrieved_context_count"])
