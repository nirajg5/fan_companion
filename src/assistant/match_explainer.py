import json
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from groq import APIConnectionError, APITimeoutError, APIStatusError, RateLimitError

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MAX_TOKENS = 700


def build_explanation_prompt(
    match_context: dict[str, Any],
    question: str = "Explain the current match situation.",
) -> str:
    """Build a grounded prompt from verified or explicitly labelled data."""
    context_json = json.dumps(
        match_context,
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    return f"""
You are an AI cricket analyst helping fans understand a match.

Fan question:
{question}

Match data:
{context_json}

Instructions:
- Answer the specific question in simple English.
- Use only facts supported by the supplied data.
- Do not invent scores, deliveries, players, teams, or statistics.
- Distinguish recorded facts from model estimates and simulations.
- If information is missing, say it is unavailable.
- Do not claim causation without supporting evidence.
- When comparing probabilities, express the difference in percentage
  points (pp), not percent.
- Follow interpretation rules supplied in the match data.
- If a list is requested, cover every supplied item.
- Prefer concise bullet points over long paragraphs.
- Never omit list items just to make individual explanations longer.
- Format probability levels with the % symbol, for example 34.00%.
- Format differences in probability with percentage points (pp),
  for example a decrease of 12.05 pp.
- Never append pp to an individual probability value.
""".strip()


def explain_match_situation(
    match_context: dict[str, Any],
    question: str = "Explain the current match situation.",
) -> str:
    """Generate a grounded cricket explanation using the Groq API."""
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file. "
            "Do not share your API key."
        )
    if not isinstance(match_context, dict) or not match_context:
        raise ValueError("Match context must be a non-empty dictionary.")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    try:
        max_tokens = int(
            os.getenv("GROQ_MAX_COMPLETION_TOKENS", str(DEFAULT_MAX_TOKENS))
        )
    except ValueError as exc:
        raise ValueError(
            "GROQ_MAX_COMPLETION_TOKENS must be an integer."
        ) from exc
    if max_tokens < 100:
        raise ValueError("GROQ_MAX_COMPLETION_TOKENS must be at least 100.")

    client = Groq(
        api_key=api_key,
        timeout=30.0,
        max_retries=1,
    )
    prompt = build_explanation_prompt(match_context, question.strip())

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful cricket analyst. Explain supplied "
                        "data accurately. Never fabricate match facts. Follow "
                        "team attribution and interpretation rules. Keep answers "
                        "concise and complete."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
    except APIConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Groq. Check internet/DNS access to "
            "api.groq.com and try again. This is a network connection issue, "
            "not evidence of an invalid API key."
        ) from exc
    except APITimeoutError as exc:
        raise RuntimeError(
            "Groq request timed out. Check your connection and try again."
        ) from exc
    except RateLimitError as exc:
        raise RuntimeError(
            "Groq rate limit or quota was reached. Check your Groq usage "
            "and retry later."
        ) from exc
    except APIStatusError as exc:
        raise RuntimeError(
            f"Groq returned HTTP {exc.status_code}. Check model availability "
            "and API account status."
        ) from exc

    choice = response.choices[0]
    explanation = choice.message.content
    if not explanation or not explanation.strip():
        raise RuntimeError("Groq returned an empty explanation.")

    if choice.finish_reason == "length":
        print(
            "Warning: Groq reached the output token limit "
            f"({max_tokens}). Consider increasing "
            "GROQ_MAX_COMPLETION_TOKENS if supported by the model."
        )
    return explanation.strip()


if __name__ == "__main__":
    sample_context = {
        "data_source": "simulation",
        "match_situation": {
            "score": "120/4",
            "overs": "15.0",
            "target": 180,
            "runs_required": 60,
            "balls_remaining": 30,
            "required_run_rate": 12.0,
        },
        "hypothetical_scenario": ["six", "four", "wicket"],
        "scenario_result": {
            "score_after": "130/5",
            "overs_after": "15.3",
            "runs_added": 10,
            "wickets_added": 1,
        },
        "win_probability": {
            "before_percent": 35.0,
            "after_percent": 40.0,
            "change_percentage_points": 5.0,
            "status": "illustrative_example_not_model_output",
        },
    }
    print(
        explain_match_situation(
            sample_context,
            "Explain this match situation and the hypothetical scenario.",
        )
    )
