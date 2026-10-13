import json
import os
import re
from typing import Annotated, Any, Literal, TypedDict
from langgraph.graph.message import add_messages

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, START, StateGraph
from src.assistant.checkpoint_store import get_checkpointer
from src.assistant.match_explainer import explain_match_situation
from src.cricket.predict_win_probability import predict_win_probability
from src.cricket.historical_turning_points import (
    analyze_historical_turning_points,
)
from src.cricket.summary_retriever import search_match_summaries
from src.cricket.historical_match import (
    load_match_deliveries,
    get_recorded_delivery_interval,
)
from src.cricket.historical_turning_points import (
    analyze_historical_turning_points,
)

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"


Intent = Literal[
    "catch_up",
    "turning_point",
    "scenario",
    "win_probability",
    "cricket_qa",
    "match_situation",
]

VALID_INTENTS = {
    "catch_up",
    "turning_point",
    "scenario",
    "win_probability",
    "cricket_qa",
    "match_situation",
}



class AssistantState(TypedDict):
    question: str
    match_context: dict[str, Any]
    intent: str
    answer: str
    messages: Annotated[list, add_messages]



# --------------------------------------------------
# 1. LLM client
# --------------------------------------------------

def get_groq_client() -> Groq:
    """Create a Groq client using the API key stored in .env."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file. "
            "Do not share your API key."
        )
    return Groq(api_key=api_key)


def get_model_name() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL)


# --------------------------------------------------
# 2. LLM-based intent classification
# --------------------------------------------------


def classify_intent(question: str) -> Intent:
    """Classify a cricket question and safely fall back on failure."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    prompt = f"""
Classify the cricket question into exactly one intent:
- scenario: hypothetical deliveries or what-if situations
- win_probability: chance of winning or probability changes
- turning_point: decisive moments or momentum changes
- match_situation: all other cricket questions

Return only a JSON object with one field named "intent".
Its value must be one of:
"scenario", "win_probability", "turning_point", "match_situation".

Do not answer the question.

Question: {question}
""".strip()

    try:
        response = get_groq_client().chat.completions.create(
            model=get_model_name(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify cricket questions. Return a valid JSON "
                        "object with the required intent field."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=300,
            reasoning_effort="low",
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or ""
        parsed = json.loads(raw)
        intent = str(parsed.get("intent", "")).strip().lower()

        if intent in VALID_INTENTS:
            return intent

        print(f"Invalid intent returned by model: {intent!r}")

    except Exception as exc:
        print(
            "Intent classification failed; using match_situation. "
            f"Details: {exc}"
        )

    return "match_situation"


# --------------------------------------------------
# 3. General match explanation
# --------------------------------------------------

def explain_node(state: AssistantState) -> dict[str, str]:
    """Explain the supplied match information in simple fan-friendly English."""

    return {
        "answer": explain_match_situation(
            match_context=state["match_context"],
            question=state["question"],
        )
    }


# --------------------------------------------------
# 4. Parse the delivery sequence for a what-if question
# --------------------------------------------------

def extract_scenario_events(question: str) -> list[str]:
    """
    Extract delivery outcomes. Remove phrases such as 'next three balls'
    first so that 'three' is not mistaken for three runs.
    """

    q = question.lower().replace("no ball", "noball")

    q = re.sub(
        r"\b(?:in\s+the\s+)?(?:next\s+)?(?:one|two|three|four|five|six|\d+)\s+"
        r"(?:balls?|deliveries)\b",
        " ",
        q,
    )

    aliases = {
        "six runs": "six",
        "maximum": "six",
        "six": "six",
        "four runs": "four",
        "boundary": "four",
        "four": "four",
        "three runs": "three",
        "two runs": "two",
        "one run": "single",
        "single": "single",
        "dot ball": "dot",
        "dot": "dot",
        "wicket": "wicket",
        "out": "wicket",
        "wide ball": "wide",
        "wide": "wide",
        "noball": "no_ball",
    }

    pattern = re.compile(
        r"\b(?:six runs|four runs|three runs|two runs|one run|"
        r"dot ball|wide ball|maximum|boundary|noball|"
        r"six|four|three|two|single|dot|wicket|out|wide)\b"
    )

    events: list[str] = []
    for match in pattern.finditer(q):
        event = aliases.get(match.group(0))
        if event:
            events.append(event)

    return events


# --------------------------------------------------
# 5. Calculate the what-if scenario using Python
# --------------------------------------------------


def calculate_scenario(
    context: dict[str, Any],
    events: list[str],
) -> dict[str, Any]:
    """
    Calculate a simplified hypothetical delivery sequence.

    Supports first-innings score simulation without a target and
    second-innings chase simulation when a valid target is available.
    """

    situation = context.get("match_situation", context)

    # Read the starting score.
    score = situation.get("score", "")
    score_match = re.fullmatch(
        r"\s*(\d+)\s*/\s*(\d+)\s*",
        str(score),
    )

    if "runs" in situation and "wickets" in situation:
        runs = int(situation["runs"])
        wickets = int(situation["wickets"])
    elif score_match:
        runs = int(score_match.group(1))
        wickets = int(score_match.group(2))
    else:
        raise ValueError(
            "The current score is missing. Provide runs and wickets, "
            "or a score such as '120/4'."
        )

    # Read the innings position.
    max_overs = int(situation.get("max_overs", 20))

    if "legal_balls" in situation:
        legal_balls = int(situation["legal_balls"])
    else:
        overs_text = str(situation.get("overs", ""))
        overs_match = re.fullmatch(
            r"\s*(\d+)(?:\.(\d+))?\s*",
            overs_text,
        )

        if not overs_match:
            raise ValueError(
                "Overs are missing. Provide legal_balls or an overs "
                "value such as '15.0'."
            )

        completed_overs = int(overs_match.group(1))
        balls_in_over = int(overs_match.group(2) or 0)

        if not 0 <= balls_in_over <= 5:
            raise ValueError(
                "The balls part of overs must be between 0 and 5."
            )

        legal_balls = completed_overs * 6 + balls_in_over

    # A target is optional for the first innings.
    raw_target = situation.get("target")
    target = int(raw_target) if raw_target is not None else None

    if runs < 0 or not 0 <= wickets <= 10:
        raise ValueError("The score or wicket count is invalid.")

    if max_overs <= 0:
        raise ValueError("The maximum overs must be positive.")

    max_balls = max_overs * 6

    if not 0 <= legal_balls <= max_balls:
        raise ValueError("The overs value is invalid.")

    if target is not None and target <= 0:
        raise ValueError("The target must be positive.")

    # Validate the starting state.
    if (
        wickets >= 10
        or legal_balls >= max_balls
        or (target is not None and runs >= target)
    ):
        raise ValueError(
            "The innings is already complete at the selected "
            "historical point. Move the replay slider earlier."
        )

    def make_state(
        current_runs: int,
        current_wickets: int,
        current_balls: int,
    ) -> dict[str, Any]:
        balls_remaining = max(0, max_balls - current_balls)

        state = {
            "runs": current_runs,
            "wickets": current_wickets,
            "score": f"{current_runs}/{current_wickets}",
            "legal_balls": current_balls,
            "overs": (
                f"{current_balls // 6}.{current_balls % 6}"
            ),
            "balls_remaining": balls_remaining,
            "target": target,
            "runs_required": (
                max(0, target - current_runs)
                if target is not None
                else None
            ),
        }

        return state

    before = make_state(runs, wickets, legal_balls)

    rules = {
        "six": {"runs": 6, "wicket": False, "legal": True},
        "four": {"runs": 4, "wicket": False, "legal": True},
        "three": {"runs": 3, "wicket": False, "legal": True},
        "two": {"runs": 2, "wicket": False, "legal": True},
        "single": {"runs": 1, "wicket": False, "legal": True},
        "dot": {"runs": 0, "wicket": False, "legal": True},
        "wicket": {"runs": 0, "wicket": True, "legal": True},
        "wide": {"runs": 1, "wicket": False, "legal": False},
        "no_ball": {"runs": 1, "wicket": False, "legal": False},
    }

    runs_added = 0
    wickets_added = 0
    legal_balls_used = 0
    completed_events: list[str] = []
    events_skipped = 0

    for index, event in enumerate(events):
        if event not in rules:
            continue

        # Stop when the innings reaches a terminal state.
        if (
            runs >= target if target is not None else False
        ) or wickets >= 10 or legal_balls >= max_balls:
            events_skipped = len(events) - index
            break

        rule = rules[event]

        runs += rule["runs"]
        runs_added += rule["runs"]

        if rule["wicket"]:
            wickets += 1
            wickets_added += 1

        if rule["legal"]:
            legal_balls += 1
            legal_balls_used += 1

        completed_events.append(event)

        # Do not count remaining deliveries after the chase is won.
        if target is not None and runs >= target:
            events_skipped = len(events) - index - 1
            break

        if wickets >= 10 or legal_balls >= max_balls:
            events_skipped = len(events) - index - 1
            break

    after = make_state(runs, wickets, legal_balls)
    after.update({
        "runs_added": runs_added,
        "wickets_added": wickets_added,
        "balls_used": legal_balls_used,
    })

    return {
        "events": completed_events,
        "before": before,
        "after": after,
        "runs_added": runs_added,
        "wickets_added": wickets_added,
        "events_skipped": events_skipped,
        "data_source": "hypothetical_simulation",
    }


def extract_over_range(
    question: str,
) -> tuple[int | None, int | None]:
    """Extract explicitly requested cricket over numbers."""

    text = question.lower()

    range_match = re.search(
        r"\b(?:from|between)\s+(?:over\s+)?(\d+)"
        r"\s*(?:to|through|and|-)\s*"
        r"(?:over\s+)?(\d+)\b",
        text,
    )

    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))

    start_match = re.search(
        r"\bfrom\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+over\b",
        text,
    )

    if start_match:
        return int(start_match.group(1)), None

    single_match = re.search(
        r"\b(?:in|during|summari[sz]e)\s+"
        r"(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+over\b",
        text,
    )

    if single_match:
        over_number = int(single_match.group(1))
        return over_number, over_number

    return None, None



def catch_up_node(state: AssistantState) -> dict[str, str]:
    """Summarize recorded deliveries for the requested over interval."""

    context = state["match_context"]
    question = state["question"].strip()

    try:
        match_id = str(context["match_id"])
        innings_number = int(context["innings"])

        replay_position = int(
            context.get(
                "replay_position",
                context.get("recorded_deliveries_included", 0),
            )
        )

        start_over, end_over = extract_over_range(question)

        available = get_recorded_delivery_interval(
            match_id=match_id,
            innings=innings_number,
            replay_position=replay_position,
            start_over=start_over,
            end_over=end_over,
        )

        if available.empty:
            if start_over is not None and end_over is not None:
                requested = (
                    f"over {start_over}"
                    if start_over == end_over
                    else f"overs {start_over}–{end_over}"
                )
            elif start_over is not None:
                requested = f"over {start_over} onward"
            else:
                requested = "the requested interval"

            return {
                "answer": (
                    f"No recorded deliveries are available for {requested} "
                    "at the current replay position. Move the replay slider "
                    "forward or select an earlier over."
                )
            }

        # Calculate verified statistics from the selected records.
        total_runs = int(available["total_runs"].sum())
        wickets = int(
            available["wicket"].fillna(False).astype(bool).sum()
        )
        fours = int(available["batter_runs"].eq(4).sum())
        sixes = int(available["batter_runs"].eq(6).sum())

        # Describe the exact interval supplied to the model.
        if start_over is not None and end_over is not None:
            interval_label = (
                f"over {start_over}"
                if start_over == end_over
                else f"overs {start_over} to {end_over}"
            )
        elif start_over is not None:
            interval_label = f"over {start_over} onward"
        else:
            interval_label = "all available overs"

        # Compact records to reduce prompt size.
        records = available[
            [
                "over",
                "delivery_in_over",
                "batter",
                "bowler",
                "batter_runs",
                "extra_runs",
                "total_runs",
                "wicket",
                "wicket_details_json",
            ]
        ].to_json(orient="records", force_ascii=False)

        # Build a verified fallback so the feature remains useful
        # even when the LLM response is empty.
        fallback_answer = (
            f"### Match catch-up: {interval_label}\n\n"
            f"- **Match:** {context.get('teams', 'Unknown')}\n"
            f"- **Innings:** {innings_number}\n"
            f"- **Recorded deliveries:** {len(available)}\n"
            f"- **Runs scored in this interval:** {total_runs}\n"
            f"- **Wickets recorded:** {wickets}\n"
            f"- **Fours:** {fours}\n"
            f"- **Sixes:** {sixes}\n\n"
            "This summary uses historical recorded deliveries available "
            "at the selected replay position."
        )

        prompt = f"""
Summarize this historical cricket interval for a fan.

Match: {context.get("teams", "Unknown")}
Innings: {innings_number}
Batting team: {context.get("batting_team", "Unknown")}
Data source: {context.get("data_source", "Historical records")}
Requested interval: {interval_label}

Verified statistics:
- Total runs: {total_runs}
- Wickets: {wickets}
- Fours: {fours}
- Sixes: {sixes}
- Recorded deliveries: {len(available)}

Recorded delivery data (JSON):
{records}

User question:
{question}

Requirements:
- Use only the supplied records and verified statistics.
- Explain notable recorded events concisely.
- Do not invent commentary, player actions, or dismissal reasons.
- Distinguish recorded facts from interpretation.
- Do not claim the data covers deliveries beyond the selected replay position.
- Return a concise, fan-friendly answer.
""".strip()

        response = get_groq_client().chat.completions.create(
            model=get_model_name(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are MatchPulse AI. Summarize historical "
                "cricket data accurately. Never invent events. "
                "Answer directly in concise plain text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
             temperature=0,
             reasoning_effort="low",
             max_tokens=1000,
        )

        choice = response.choices[0]
        message = choice.message

        # Diagnostic output: use this to investigate empty responses.
        print("\n===== GROQ CATCH-UP DEBUG =====")
        print("Model:", response.model)
        print("Finish reason:", choice.finish_reason)
        print("Message content:", repr(message.content))
        print("Refusal:", getattr(message, "refusal", None))
        print("Usage:", response.usage)
        print("===============================\n")

        answer = (message.content or "").strip()

        if not answer:
            print(
                "Groq returned an empty catch-up response. "
                "Using the verified statistics fallback."
            )
            return {"answer": fallback_answer}

        return {"answer": answer}

    except (ValueError, KeyError, FileNotFoundError) as exc:
        return {
            "answer": (
                "I could not retrieve the requested historical interval. "
                f"Details: {exc}"
            )
        }
    except Exception as exc:
        print(
            f"Catch-up generation failed: {type(exc).__name__}: {exc}"
        )
        return {
            "answer": (
                "The catch-up summary could not be generated. "
                f"Error type: {type(exc).__name__}."
            )
        }


def find_recorded_dismissals(
    innings_rows,
    question: str,
) -> list[dict[str, Any]]:
    """Find recorded dismissals matching a player's name."""

    question_lower = question.lower()
    results: list[dict[str, Any]] = []

    wicket_rows = innings_rows[
        innings_rows["wicket"].fillna(False).astype(bool)
    ]

    for _, row in wicket_rows.iterrows():
        raw_details = row.get("wicket_details_json")

        if not isinstance(raw_details, str) or not raw_details.strip():
            continue

        try:
            details = json.loads(raw_details)
        except (json.JSONDecodeError, TypeError):
            continue

        if not isinstance(details, list):
            continue

        for dismissal in details:
            if not isinstance(dismissal, dict):
                continue

            player_out = str(dismissal.get("player_out", "")).strip()
            if not player_out:
                continue

            # Match the player's full name as a phrase in the question.
            player_name_parts = player_out.lower().split()

            if not any(
              len(part) >= 4 and part in question_lower
             for part in player_name_parts
            ):
                continue

            fielders = dismissal.get("fielders", [])
            fielder_names = [
                str(fielder.get("name", "")).strip()
                for fielder in fielders
                if isinstance(fielder, dict) and fielder.get("name")
            ] if isinstance(fielders, list) else []

            results.append({
                "player_out": player_out,
                "over": int(row["over"]),
                "delivery_in_over": int(row["delivery_in_over"]),
                "bowler": str(row.get("bowler", "")),
                "dismissal_kind": str(dismissal.get("kind", "unknown")),
                "fielders": fielder_names,
            })

    return results



def retrieve_historical_context(
    question: str,
    match_id: str,
    top_k: int = 3,
) -> str:
    """Retrieve relevant historical match summaries for cricket Q&A."""

    try:
        results = search_match_summaries(
            query=question,
            top_k=top_k,
            match_id=str(match_id),
        )

        summaries = [
            str(item["summary_text"])
            for item in results
            if item.get("summary_text")
        ]

        return "\n\n".join(summaries)

    except Exception as exc:
        # RAG is supplementary; cricket Q&A should still work
        # using recorded deliveries and verified Python statistics.
        print(
            "Historical RAG retrieval failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return ""



def cricket_qa_node(state: AssistantState) -> dict[str, str]:
    """Answer cricket questions using LLM-guided retrieval and verified statistics."""

    context = state["match_context"]
    question = state["question"].strip()

    
    historical_context = retrieve_historical_context(
    question=question,
    match_id=str(context["match_id"]),
    )


    try:
        if not question:
            return {"answer": "Please enter a cricket question."}

        deliveries = load_match_deliveries(str(context["match_id"]))
        innings_number = int(context["innings"])

        innings_rows = deliveries[
            deliveries["innings"] == innings_number
        ].copy()

        innings_rows = innings_rows.sort_values(
            ["over", "delivery_in_over"]
        ).reset_index(drop=True)

        replay_position = int(
            context.get(
                "replay_position",
                context.get("recorded_deliveries_included", 0),
            )
        )

        if replay_position < 0:
            return {"answer": "The replay position cannot be negative."}

        innings_rows = innings_rows.iloc[
            :min(replay_position, len(innings_rows))
        ].copy()

        if innings_rows.empty:
            return {
                "answer": (
                    "No recorded deliveries are available at the "
                    "selected replay position."
                )
            }

        player_columns = [
            col for col in ("batter", "bowler", "non_striker")
            if col in innings_rows.columns
        ]

        available_players = sorted({
            str(value).strip()
            for col in player_columns
            for value in innings_rows[col].dropna().unique()
            if str(value).strip()
        })

        available_overs = sorted(
            int(value) for value in innings_rows["over"].dropna().unique()
        )

        # Ask the LLM to interpret the question and identify the retrieval needs.
        retrieval_prompt = f"""
You are MatchPulse AI's cricket retrieval planner.

Return valid JSON with exactly these fields:
{{
  "players": [],
  "overs": [],
  "needs_all_innings": false,
  "statistic": "general",
  "player_role": "unknown"
}}

Allowed statistic values:
- batting_runs
- bowling_runs_conceded
- wickets_taken
- dismissal
- general

Allowed player_role values:
- batter
- bowler
- unknown

Rules:
- Use exact names from the available player list.
- Include every player required to answer.
- Interpret "runs scored by a player" as batting_runs.
- Interpret "runs conceded/given away by a bowler" as bowling_runs_conceded.
- Interpret "wickets taken by a bowler" as wickets_taken.
- Identify the role of the player whose statistics are requested.
- Use overs only when the question explicitly restricts the answer to an over.
- Set needs_all_innings true for a full-innings summary or comparison.
- Do not answer the user's question.
- Do not invent players or overs.

Available players:
{json.dumps(available_players, ensure_ascii=False)}

Available overs:
{json.dumps(available_overs)}

Question:
{question}
""".strip()

        plan_response = get_groq_client().chat.completions.create(
            model=get_model_name(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify cricket retrieval needs accurately. "
                        "Return valid JSON only."
                    ),
                },
                {"role": "user", "content": retrieval_prompt},
            ],
            temperature=0,
            reasoning_effort="low",
            max_tokens=300,
            response_format={"type": "json_object"},
        )

        plan = json.loads(
            plan_response.choices[0].message.content or "{}"
        )

        requested_players = plan.get("players", [])
        requested_overs = plan.get("overs", [])
        needs_all_innings = plan.get("needs_all_innings", False)
        statistic = str(plan.get("statistic", "general")).strip().lower()
        player_role = str(plan.get("player_role", "unknown")).strip().lower()

        if not isinstance(requested_players, list):
            requested_players = []
        if not isinstance(requested_overs, list):
            requested_overs = []

        player_lookup = {
            name.casefold().strip(): name
            for name in available_players
        }

        selected_players = []
        for requested_name in requested_players:
            if isinstance(requested_name, str):
                key = requested_name.casefold().strip()
                if key in player_lookup:
                    selected_players.append(player_lookup[key])

        # Resolve complete player names mentioned in the question.
        normalized_question = " ".join(question.casefold().split())

        for name in available_players:
            normalized_name = " ".join(name.casefold().split())
            if normalized_name and normalized_name in normalized_question:
                if name not in selected_players:
                    selected_players.append(name)

        selected_overs = []
        for value in requested_overs:
            try:
                over_number = int(value)
            except (ValueError, TypeError):
                continue
            if over_number in available_overs:
                selected_overs.append(over_number)

        # Determine which player column is relevant for the requested statistic.
        role_column = {
            "batting_runs": "batter",
            "bowling_runs_conceded": "bowler",
            "wickets_taken": "bowler",
            "dismissal": "batter",
        }.get(statistic)

        relevant_rows = innings_rows.copy()

        if selected_players and not needs_all_innings:
            selected_names = {
                name.casefold().strip() for name in selected_players
            }

            if role_column and role_column in relevant_rows.columns:
                # For a role-specific statistic, filter only the correct role.
                relevant_rows = relevant_rows[
                    relevant_rows[role_column]
                    .astype(str).str.casefold().str.strip()
                    .isin(selected_names)
                ]
            else:
                # General questions may refer to any player role.
                mask = relevant_rows[player_columns].apply(
                    lambda column: column.astype(str)
                    .str.casefold().str.strip()
                    .isin(selected_names)
                ).any(axis=1)
                relevant_rows = relevant_rows[mask]

        if selected_overs and not needs_all_innings:
            relevant_rows = relevant_rows[
                relevant_rows["over"].isin(selected_overs)
            ]

        # If the requested role has no records, do not substitute another role.
        if relevant_rows.empty:
            if statistic in {"batting_runs", "bowling_runs_conceded", "wickets_taken"}:
                role_description = {
                    "batting_runs": "batting deliveries",
                    "bowling_runs_conceded": "bowling deliveries",
                    "wickets_taken": "bowling deliveries",
                }[statistic]

                player_name = selected_players[0] if selected_players else "The player"
                return {
                    "answer": (
                        f"No {role_description} for {player_name} were found "
                        f"in innings {innings_number} up to replay position "
                        f"{replay_position}. I cannot calculate that statistic "
                        "from the selected records."
                    )
                }

            return {
                "answer": (
                    "I couldn't find matching recorded deliveries at "
                    "the selected replay position."
                )
            }

        # Calculate role-specific statistics from the actual records.
        verified_facts = {}

        if statistic == "batting_runs" and "batter_runs" in relevant_rows.columns:
            verified_facts = {
                "statistic": "batting_runs",
                "player": selected_players,
                "runs_scored": int(relevant_rows["batter_runs"].sum()),
                "batting_deliveries": len(relevant_rows),
                "innings": innings_number,
            }

        elif statistic == "bowling_runs_conceded":
            # Parse extras to exclude byes, leg-byes and penalty runs.
            def conceded_runs(row):
                extras = {}
                raw_extras = row.get("extras_json")

                if isinstance(raw_extras, str) and raw_extras.strip():
                    try:
                        parsed = json.loads(raw_extras)
                        if isinstance(parsed, dict):
                            extras = parsed
                    except (json.JSONDecodeError, TypeError):
                        pass

                total = int(row.get("total_runs", 0) or 0)
                excluded = (
                    int(extras.get("byes", 0) or 0)
                    + int(extras.get("legbyes", 0) or 0)
                    + int(extras.get("penalty", 0) or 0)
                )
                return max(0, total - excluded)

            verified_facts = {
                "statistic": "bowling_runs_conceded",
                "player": selected_players,
                "runs_conceded": int(
                    relevant_rows.apply(conceded_runs, axis=1).sum()
                ),
                "bowling_deliveries": len(relevant_rows),
                "innings": innings_number,
            }

        elif statistic == "wickets_taken":
            # A dismissal is credited to the bowler for most dismissal types,
            # but not run out, retired hurt, obstructing the field, etc.
            credited_kinds = {
                "bowled", "caught", "caught and bowled", "lbw",
                "stumped", "hit wicket",
            }

            credited_wickets = 0
            for _, row in relevant_rows.iterrows():
                if not bool(row.get("wicket", False)):
                    continue

                raw_details = row.get("wicket_details_json")
                if not isinstance(raw_details, str):
                    continue

                try:
                    details = json.loads(raw_details)
                except (json.JSONDecodeError, TypeError):
                    continue

                if not isinstance(details, list):
                    continue

                for dismissal in details:
                    if not isinstance(dismissal, dict):
                        continue
                    kind = str(dismissal.get("kind", "")).casefold().strip()
                    if kind in credited_kinds:
                        credited_wickets += 1

            verified_facts = {
                "statistic": "wickets_taken",
                "player": selected_players,
                "wickets_taken": credited_wickets,
                "innings": innings_number,
            }

        useful_columns = [
            col for col in [
                "over", "delivery_in_over", "batter", "bowler",
                "non_striker", "batter_runs", "extra_runs",
                "total_runs", "extras_json", "wicket",
                "wicket_details_json",
            ]
            if col in relevant_rows.columns
        ]

        records_json = json.dumps(
            relevant_rows[useful_columns].to_dict(orient="records"),
            ensure_ascii=False,
            default=str,
        )

        answer_prompt = f"""
You are MatchPulse AI, a cricket analytics assistant.

Question:
{question}

Match: {context.get("teams", "Unknown")}
Innings: {innings_number}
Replay position: {replay_position}
Statistic classification: {statistic}
Player role: {player_role}
Verified statistics calculated from the records:
{json.dumps(verified_facts, ensure_ascii=False, default=str)}

Recorded deliveries:
{records_json}


Historical match summary context:
{historical_context or "No historical summaries were retrieved."}


Rules:
- Answer the exact question, using the verified statistics when provided.
- Do not confuse a batter's runs scored with a bowler's runs conceded.
- Batting runs come from the batter column and batter_runs values.
- Bowling runs conceded come from the bowler column and total runs minus
  byes, leg-byes and penalty runs.
- Do not credit run-outs to the bowler.
- If a player has no records in the requested role, state that clearly.
- Never invent events or statistics.
- Keep the answer concise and fan-friendly.
""".strip()

        response = get_groq_client().chat.completions.create(
            model=get_model_name(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are MatchPulse AI. Use verified statistics "
                        "when supplied. Never change the meaning of player "
                        "roles or invent cricket facts."
                    ),
                },
                {"role": "user", "content": answer_prompt},
            ],
            temperature=0,
            reasoning_effort="low",
            max_tokens=1000,
        )

        answer = (
            response.choices[0].message.content or ""
        ).strip()

        if not answer:
            return {
                "answer": (
                    "The model returned an empty answer. "
                    "Please try asking the question again."
                )
            }

        return {"answer": answer}

    except Exception as exc:
        print(f"Cricket Q&A failed: {type(exc).__name__}: {exc}")
        return {
            "answer": (
                "I couldn't answer the cricket question because "
                f"the request failed ({type(exc).__name__})."
            )
        }


# --------------------------------------------------
# 6. LLM explanation of the calculated scenario
# --------------------------------------------------


def scenario_node(state: AssistantState) -> dict[str, str]:
    events = extract_scenario_events(state["question"])

    if not events:
        return {
            "answer": (
                "I couldn't identify the delivery outcomes. "
                "Try specifying the sequence, such as "
                "'single, then four, then six.'"
            )
        }

    context = state["match_context"]
    situation = context.get("match_situation", context)

    try:
        result = calculate_scenario(context, events)
    except (ValueError, TypeError, KeyError) as exc:
        return {
            "answer": (
                f"I couldn't calculate that scenario: {exc}"
            )
        }

    before = result["before"]
    after = result["after"]
    target = situation.get("target")

    probability_result = None
    probability_note = ""

    # Win probability is only calculated for a valid chase target.
    if target is not None:
        try:
            before_prediction = predict_win_probability(
                runs=before["runs"],
                wickets=before["wickets"],
                legal_balls=before["legal_balls"],
                target=int(target),
                max_overs=int(situation.get("max_overs", 20)),
            )

            after_prediction = predict_win_probability(
                runs=after["runs"],
                wickets=after["wickets"],
                legal_balls=after["legal_balls"],
                target=int(target),
                max_overs=int(situation.get("max_overs", 20)),
            )

            before_probability = before_prediction["win_probability"]
            after_probability = after_prediction["win_probability"]

            probability_result = {
                "before": before_probability,
                "after": after_probability,
                "change_percentage_points": round(
                    after_probability - before_probability,
                    2,
                ),
                "method_before": before_prediction["method"],
                "method_after": after_prediction["method"],
            }

        except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
            probability_note = (
                "Win probability could not be calculated for this "
                f"scenario: {exc}"
            )

        except Exception as exc:
            probability_note = (
                "Win-probability calculation failed: "
                f"{type(exc).__name__}: {exc}"
            )

    else:
        probability_note = (
            "Win probability is unavailable because this is the "
            "first innings and the second-innings target has not "
            "yet been established. The current model estimates "
            "chasing-team win probability, not first-innings "
            "win probability."
        )

    # Construct the answer from calculated facts.
    overs_before = before["overs"]
    overs_after = after["overs"]

    lines = [
        "### What-if scenario",
        "",
        f"**Starting state:** {before['score']} after {overs_before} overs.",
        f"**Hypothetical deliveries applied:** "
        f"{', '.join(result['events']) or 'None'}.",
        f"**Resulting state:** {after['score']} after {overs_after} overs.",
        f"**Runs added:** {after['runs_added']}.",
        f"**Wickets lost in the scenario:** {after['wickets_added']}.",
        f"**Legal balls used:** {after['balls_used']}.",
    ]

    if target is not None:
        lines.extend([
            f"**Target:** {target}.",
            f"**Runs required after the scenario:** "
            f"{after['runs_required']}.",
            f"**Balls remaining after the scenario:** "
            f"{after['balls_remaining']}.",
        ])

    lines.append("")

    if probability_result is not None:
        lines.extend([
            "### Win probability",
            "",
            f"- Before: **{probability_result['before']:.2f}%**",
            f"- After: **{probability_result['after']:.2f}%**",
            f"- Change: **"
            f"{probability_result['change_percentage_points']:+.2f} "
            f"percentage points**",
            "",
            "These values estimate the chasing team's chance of winning. "
            "They are model estimates, not guarantees.",
        ])
    else:
        lines.extend([
            "### Win probability",
            "",
            probability_note,
        ])

    lines.extend([
        "",
        "**Important:** This is a hypothetical score simulation, "
        "not a prediction that these deliveries will occur.",
    ])

    return {"answer": "\n".join(lines)}


# --------------------------------------------------
# 7. Win probability and turning-point nodes
# --------------------------------------------------



def probability_node(state: AssistantState) -> dict[str, str]:
    context = dict(state["match_context"])
    situation = context.get("match_situation", context)

    target = situation.get("target")

    if target is None:
        return {
            "answer": (
                "A chasing-team win probability is not available at "
                "this replay position because the selected innings is "
                "the first innings. The target for the chase has not "
                "yet been established. Once the first innings is "
                "complete and you select the second innings, I can "
                "estimate the chasing team's win probability."
            )
        }

    try:
        prediction = predict_win_probability(
            runs=int(situation["runs"]),
            wickets=int(situation["wickets"]),
            legal_balls=int(situation["legal_balls"]),
            target=int(target),
            max_overs=int(situation.get("max_overs", 20)),
        )

        context["win_probability"] = prediction
        context["chasing_team"] = context.get(
            "chasing_team",
            context.get("batting_team", "Chasing team"),
        )

        context["probability_interpretation"] = (
            "This is the estimated probability that the chasing team "
            "wins. It is a model estimate, not a guarantee."
        )

    except Exception as exc:
        return {
            "answer": (
                "Win probability could not be calculated at this "
                f"replay position. Reason: {type(exc).__name__}: {exc}"
            )
        }

    return {
        "answer": explain_match_situation(
            match_context=context,
            question=state["question"],
        )
    }



def turning_point_node(state: AssistantState) -> dict[str, str]:
    context = dict(state["match_context"])

    try:
        match_id = context.get("match_id")
        if not match_id:
            raise ValueError(
                "A historical match_id is required to detect turning points."
            )

        threshold = float(context.get("turning_point_threshold", 10.0))

        result = analyze_historical_turning_points(
            match_id=str(match_id),
            threshold=threshold,
        )

        # Attach the actual detector output to the explanation context.
        context["turning_points"] = result["turning_points"]
        context["data_source"] = result["data_source"]
        context["target"] = result["target"]
        context["deliveries_analyzed"] = result["deliveries_analyzed"]
        context["teams"] = result["teams"]

        # Explicitly establish which team's probability is being measured.
        context["probability_interpretation"] = (
            "All win probabilities refer to the chasing team. "
            "A decrease means the chasing team's estimated chance fell; "
            "an increase means it rose. Changes are measured in "
            "percentage points (pp)."
        )

        if not context.get("chasing_team"):
            context["chasing_team"] = "Sri Lanka"

        if not context.get("defending_team"):
            context["defending_team"] = "Australia"

    except (ValueError, KeyError, FileNotFoundError) as exc:
        context["feature_status"] = (
            f"Historical turning-point analysis failed: {exc}"
        )
        context.pop("turning_points", None)

    return {
        "answer": explain_match_situation(
            match_context=context,
            question=state["question"],
        )
    }

# --------------------------------------------------
# 8. Build the LangGraph workflow
# --------------------------------------------------

def route_question(state: AssistantState) -> Intent:
    # Reuse the LLM classification already stored in state to avoid
    # making a second classification API call for the same question.
    intent = state.get("intent", "")
    if intent in VALID_INTENTS:
        return intent  # type: ignore[return-value]
    return classify_intent(state["question"])




def build_assistant():
    workflow = StateGraph(AssistantState)

    workflow.add_node("match_situation", explain_node)
    workflow.add_node("scenario", scenario_node)
    workflow.add_node("win_probability", probability_node)
    workflow.add_node("turning_point", turning_point_node)
    workflow.add_node("catch_up", catch_up_node)
    workflow.add_node("cricket_qa", cricket_qa_node)

    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "match_situation": "match_situation",
            "scenario": "scenario",
            "win_probability": "win_probability",
            "turning_point": "turning_point",
            "catch_up": "catch_up",
            "cricket_qa": "cricket_qa",
        },
    )

    workflow.add_edge("match_situation", END)
    workflow.add_edge("scenario", END)
    workflow.add_edge("win_probability", END)
    workflow.add_edge("turning_point", END)
    workflow.add_edge("catch_up", END)
    workflow.add_edge("cricket_qa", END)

    return workflow.compile(checkpointer=get_checkpointer())


assistant = build_assistant()


def resolve_follow_up_question(
    question: str,
    history: list[dict[str, str]],
) -> str:
    """Rewrite a follow-up question so it can be understood independently."""

    if not history:
        return question

    recent_history = history[-8:]

    conversation = "\n".join(
        f"{item['role']}: {item['content']}"
        for item in recent_history
    )

    prompt = f"""
Rewrite the latest cricket question as a standalone question.

Use the conversation to resolve references such as "that", "it",
"the next ball", and "what about a wicket".

Rules:
- Preserve the user's intended question.
- Use only facts supported by the conversation.
- Do not answer the question.
- Do not invent match details or delivery outcomes.
- Return only the rewritten question.

Previous conversation:
{conversation}

Latest question:
{question}
""".strip()

    try:
        response = get_groq_client().chat.completions.create(
            model=get_model_name(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite follow-up cricket questions. "
                        "Return only the standalone question."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=200,
        )

        rewritten = (
            response.choices[0].message.content or ""
        ).strip()

        return rewritten or question

    except Exception as exc:
        print(
            "Follow-up resolution failed; using original question: "
            f"{type(exc).__name__}: {exc}"
        )
        return question





def ask_cricket_assistant(
    question: str,
    match_context: dict[str, Any],
    thread_id: str | None = None,
) -> str:
    """Answer a cricket question while preserving conversation history."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(match_context, dict) or not match_context:
        raise ValueError("Match context must be a non-empty dictionary.")

    if thread_id is None:
        match_id = match_context.get("match_id")
        innings = match_context.get("innings")

        if match_id is None or innings is None:
            raise ValueError(
                "Match context must include match_id and innings."
            )

        thread_id = f"match-{match_id}-innings-{innings}"

    clean_question = question.strip()

    # Load the previous conversation for this thread.
    history = get_cricket_chat_history(thread_id)

    # Resolve references in follow-up questions.
    resolved_question = resolve_follow_up_question(
        clean_question,
        history,
    )

    # Select the appropriate analytics node.
    intent = classify_intent(resolved_question)

    result = assistant.invoke(
        {
            "question": resolved_question,
            "match_context": match_context,
            "intent": intent,
            "answer": "",
            "messages": [
                {"role": "user", "content": clean_question},
            ],
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    answer = result["answer"]

    assistant.update_state(
        {"configurable": {"thread_id": thread_id}},
        {
            "messages": [
                {"role": "assistant", "content": answer},
            ],
        },
    )

    return answer



def get_cricket_chat_history(thread_id: str) -> list[dict[str, str]]:
    """Return saved user and assistant messages for a conversation."""
    if not thread_id or not thread_id.strip():
        raise ValueError("Thread ID cannot be empty.")

    checkpoint = assistant.get_state(
        {"configurable": {"thread_id": thread_id}}
    )

    values = checkpoint.values or {}
    messages = values.get("messages", [])

    history = []

    for message in messages:
        if isinstance(message, dict):
            role = message.get("role") or message.get("type")
            content = message.get("content")
        else:
            role = getattr(message, "type", None)
            content = getattr(message, "content", None)

        if role in ("human", "user"):
            role = "user"
        elif role in ("ai", "assistant"):
            role = "assistant"
        else:
            continue

        if isinstance(content, str) and content.strip():
            history.append({"role": role, "content": content})

    return history


def clear_cricket_chat_history(thread_id: str) -> None:
    """Delete the saved conversation for a specific thread."""
    if not thread_id or not thread_id.strip():
        raise ValueError("Thread ID cannot be empty.")

    get_checkpointer().delete_thread(thread_id)


# --------------------------------------------------
# 9. Local test
# --------------------------------------------------

if __name__ == "__main__":
    context = {
    "data_source": "Cricsheet historical replay",
    "match_id": "1001349",
    "chasing_team": "Sri Lanka",
    "defending_team": "Australia",
    "target": 169,
    "max_overs": 20,
}

    question = "What were the key turning points in this match?"

    detected_intent = classify_intent(question)
    print("Detected intent:", detected_intent)
    print()
    print(ask_cricket_assistant(question, context))
