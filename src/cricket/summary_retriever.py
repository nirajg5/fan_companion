
from typing import Any

from src.cricket.summary_embeddings import (
    EMBEDDING_MODEL,
    NAMESPACE,
    get_pinecone_client,
    get_pinecone_index,
    generate_embedding,
)
from src.assistant.redis_cache import (
    cache_key,
    get_cached_json,
    set_cached_json,
)


def search_match_summaries(
    query: str,
    top_k: int = 3,
    match_id: str | None = None,
) -> list[dict]:
    """Search historical match summaries with Redis caching."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    normalized_query = query.strip()

    # 1. Build a cache key for this exact search.
    key = cache_key(
        "summary-search",
        {
            "query": normalized_query,
            "top_k": top_k,
            "match_id": str(match_id) if match_id is not None else None,
        },
    )

    # 2. Return cached results if available.
    cached_results = get_cached_json(key)

    if cached_results is not None:
        return cached_results

    # 3. Generate an embedding and query Pinecone on a cache miss.
    from src.cricket.summary_embeddings import (
        generate_embedding,
        get_pinecone_index,
    )

    embedding = generate_embedding(normalized_query)

    query_args = {
        "namespace": NAMESPACE,
        "vector": embedding,
        "top_k": top_k,
        "include_metadata": True,
    }

    if match_id is not None:
        query_args["filter"] = {"match_id": str(match_id)}

    response = get_pinecone_index().query(**query_args)

    # 4. Convert Pinecone matches into JSON-compatible results.
    results = []

    for match in response.matches:
        metadata = match.metadata or {}

        results.append(
            {
                "id": match.id,
                "score": match.score,
                "match_id": metadata.get("match_id"),
                "source_type": metadata.get("source_type"),
                "summary_text": metadata.get("summary_text", ""),
            }
        )

    # 5. Cache successful results for five minutes.
    set_cached_json(key, results, ttl_seconds=300)

    return results

if __name__ == "__main__":
    question = "Summarize the best batting performances and bowling figures."

    results = search_match_summaries(question, top_k=3)

    print(f"\nQuery: {question}")
    print(f"Results found: {len(results)}")

    for result in results:
        print("\n" + "=" * 60)
        print(f"Match ID: {result['match_id']}")
        print(f"Similarity score: {result['score']:.4f}")
        print(result["summary_text"])
