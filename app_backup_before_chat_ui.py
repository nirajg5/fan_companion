
import pandas as pd
import streamlit as st

from src.cricket.historical_match import load_deliveries
from src.cricket.match_state import calculate_match_state
from src.assistant.graph import (
    ask_cricket_assistant,
    get_cricket_chat_history,
    clear_cricket_chat_history,
)
from src.cricket.match_summary_generator import (
    generate_natural_language_summary,
)

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fan Companion | AI Cricket Assistant",
    page_icon="🏏",
    layout="wide",
)

st.title("🏏 Fan Companion")
st.caption(
    "Your AI cricket companion for historical match analysis, "
    "win probability, turning points, and what-if scenarios."
)

st.info(
    "Historical replay mode: this app uses recorded Cricsheet deliveries, "
    "not live match data. Move the replay slider to select a point in "
    "the innings before asking questions."
)


# =========================================================
# 2. LOAD HISTORICAL DATA
# =========================================================

@st.cache_data(show_spinner="Loading historical cricket data...")
def get_data():
    return load_deliveries()


@st.cache_data(show_spinner="Preparing historical match list...")
def get_match_list():
    df = get_data()

    matches = (
        df.groupby("match_id", as_index=False)
        .agg(
            teams=("teams", "first"),
            match_type=("match_type", "first"),
        )
    )

    matches["match_id"] = matches["match_id"].astype(str)

    return matches.sort_values("match_id").reset_index(drop=True)


try:
    df = get_data()
    matches = get_match_list()

except Exception as exc:
    st.error(f"Unable to load historical data: {exc}")
    st.stop()


if matches.empty:
    st.error("No historical matches were found in the dataset.")
    st.stop()


# =========================================================
# 3. SELECT MATCH
# =========================================================

with st.sidebar:
    st.header("🏏 Match context")

    selected_match_id = st.selectbox(
        "Select a historical match",
        options=matches["match_id"].tolist(),
        format_func=lambda match_id: (
            f"{match_id} — "
            f"{matches.loc[
                matches['match_id'] == match_id, 'teams'
            ].iloc[0]}"
        ),
        key="selected_match_id",
    )


match_rows = df[
    df["match_id"].astype(str) == str(selected_match_id)
].copy()

match_rows = match_rows.sort_values(
    ["innings", "over", "delivery_in_over"]
).reset_index(drop=True)


if match_rows.empty:
    st.error("No deliveries were found for this match.")
    st.stop()


# =========================================================
# 4. SELECT INNINGS
# =========================================================

innings_options = sorted(
    match_rows["innings"].unique().tolist()
)

with st.sidebar:
    selected_innings = st.selectbox(
        "Select innings",
        options=innings_options,
        format_func=lambda innings: (
            f"Innings {innings} — "
            f"{match_rows.loc[
                match_rows['innings'] == innings, 'batting_team'
            ].iloc[0]}"
        ),
        key="selected_innings",
    )


innings_rows = match_rows[
    match_rows["innings"] == selected_innings
].copy().reset_index(drop=True)

if innings_rows.empty:
    st.error("No deliveries were found for the selected innings.")
    st.stop()

batting_team = str(innings_rows["batting_team"].iloc[0])



def build_catch_up_summary(
    deliveries: pd.DataFrame,
    start_position: int,
    end_position: int,
) -> dict:
    """Summarize recorded deliveries between two replay positions."""

    if not 0 <= start_position <= end_position <= len(deliveries):
        raise ValueError("Invalid replay positions.")

    segment = deliveries.iloc[start_position:end_position]

    if segment.empty:
        return {
            "deliveries": 0,
            "runs": 0,
            "wickets": 0,
            "boundaries": 0,
            "summary": "No recorded deliveries in this interval.",
            "events": [],
        }

    runs = int(segment["total_runs"].sum())
    wickets = int(segment["wicket"].fillna(False).astype(bool).sum())

    boundaries = int(
        segment["batter_runs"].isin([4, 6]).sum()
    )

    events = []

    for _, row in segment.iterrows():
        batter_runs = int(row["batter_runs"])
        total_runs = int(row["total_runs"])
        is_wicket = bool(row["wicket"])

        if is_wicket:
            event = "Wicket"
        elif batter_runs == 6:
            event = "Six"
        elif batter_runs == 4:
            event = "Four"
        elif total_runs == 0:
            event = "Dot ball"
        else:
            event = f"{total_runs} run(s)"

        events.append({
            "over": int(row["over"]),
            "delivery_in_over": int(row["delivery_in_over"]),
            "event": event,
            "total_runs": total_runs,
            "batter": str(row.get("batter", "Unknown")),
            "bowler": str(row.get("bowler", "Unknown")),
        })

    summary = (
        f"Across {len(segment)} recorded deliveries, "
        f"the batting team scored {runs} runs and lost {wickets} wickets. "
        f"There were {boundaries} batter-hit boundaries "
        f"({int(segment['batter_runs'].eq(4).sum())} fours and "
        f"{int(segment['batter_runs'].eq(6).sum())} sixes)."
    )

    return {
        "deliveries": len(segment),
        "runs": runs,
        "wickets": wickets,
        "boundaries": boundaries,
        "summary": summary,
        "events": events,
    }

# =========================================================
# 5. HISTORICAL REPLAY SLIDER
# =========================================================

with st.sidebar:
    st.divider()
    st.subheader("⏪ Historical replay")

    delivery_position = st.slider(
        "Recorded deliveries to include",
        min_value=0,
        max_value=len(innings_rows),
        value=len(innings_rows),
        key=f"replay_{selected_match_id}_{selected_innings}",
        help=(
            "Zero means before the first delivery. The maximum means "
            "the end of the recorded innings. The chatbot analyzes "
            "the state after the selected number of recorded deliveries."
        ),
    )


selected_rows = innings_rows.iloc[:delivery_position].copy()
selected_deliveries = selected_rows.to_dict(orient="records")

st.sidebar.caption(
    f"Replay position: {delivery_position:,} / "
    f"{len(innings_rows):,} recorded deliveries"
)


# Smart Match Catch-Up
st.header("Smart Match Catch-Up")
st.caption(
    "Summarize a selected interval of recorded deliveries "
    "from the historical match."
)

catch_up_start = st.slider(
    "Summarize from recorded delivery",
    min_value=0,
    max_value=delivery_position,
    value=max(0, delivery_position - min(12, delivery_position)),
    key=f"catchup_start_{selected_match_id}_{selected_innings}",
)

catch_up = build_catch_up_summary(
    innings_rows,
    catch_up_start,
    delivery_position,
)

st.caption(
    f"Interval: deliveries {catch_up_start} to "
    f"{delivery_position} "
    f"({catch_up['deliveries']} recorded deliveries)"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Runs scored", catch_up["runs"])
col2.metric("Wickets lost", catch_up["wickets"])
col3.metric("Boundaries", catch_up["boundaries"])
col4.metric("Deliveries", catch_up["deliveries"])

st.write(catch_up["summary"])

with st.expander("View events in this interval"):
    if catch_up["events"]:
        st.dataframe(
            pd.DataFrame(catch_up["events"]),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No recorded deliveries in this interval.")



# =========================================================
# 6. DETERMINE THE CHASE TARGET
# =========================================================

target = None
defending_team = None

if selected_innings == 2:
    first_innings = match_rows[
        match_rows["innings"] == 1
    ]

    if not first_innings.empty:
        target = int(first_innings["total_runs"].sum()) + 1
        defending_team = str(
            first_innings["batting_team"].iloc[0]
        )


# =========================================================
# 7. CALCULATE STATE AT THE SELECTED HISTORICAL POINT
# =========================================================

try:
    match_state = calculate_match_state(
        deliveries=selected_deliveries,
        innings=int(selected_innings),
        target=target,
    )

except Exception as exc:
    st.error(f"Unable to calculate the selected match state: {exc}")
    st.stop()


# =========================================================
# 8. BUILD CONTEXT FOR LANGGRAPH
# =========================================================


match_context = {
    "data_source": "Cricsheet historical replay",
    "historical_replay": True,
    "match_id": str(selected_match_id),
    "teams": str(match_rows["teams"].iloc[0]),
    "match_type": str(match_rows["match_type"].iloc[0]),
    "batting_team": batting_team,
    "chasing_team": (
        batting_team if target is not None else None
    ),
    "defending_team": defending_team,
    "innings": int(selected_innings),
    "max_overs": 20,

    # Historical replay boundaries
    "replay_position": int(delivery_position),
    "recorded_deliveries": len(innings_rows),
    "recorded_deliveries_included": int(delivery_position),
    "total_recorded_deliveries": len(innings_rows),

    # Current match state at the selected replay position
    "match_situation": {
        "runs": match_state["runs"],
        "wickets": match_state["wickets"],
        "score": (
            f"{match_state['runs']}/{match_state['wickets']}"
        ),
        "overs": match_state["overs"],
        "legal_balls": match_state["legal_balls"],
        "run_rate": match_state["run_rate"],
        "target": target,
        "runs_required": match_state["runs_required"],
        "balls_remaining": match_state["balls_remaining"],
        "required_run_rate": match_state["required_run_rate"],
        "max_overs": 20,
    },
}



# =========================================================
# 9. CONVERSATION THREAD
# =========================================================

current_selection = (
    str(selected_match_id),
    int(selected_innings),
)

thread_id = (
    f"match-{selected_match_id}-innings-{selected_innings}"
)

# Track the selected conversation in the current Streamlit session.
# Changing the replay slider does not change the thread.
if "chat_selection" not in st.session_state:
    st.session_state["chat_selection"] = current_selection

elif st.session_state["chat_selection"] != current_selection:
    st.session_state["chat_selection"] = current_selection
    st.session_state.pop("pending_question", None)


# =========================================================
# 10. SIDEBAR MATCH SUMMARY AND CLEAR CONVERSATION
# =========================================================

with st.sidebar:
    st.divider()
    st.subheader("📊 Selected match state")

    st.metric(
        f"{batting_team} score",
        f"{match_state['runs']}/{match_state['wickets']}",
    )

    st.write(f"**Overs:** {match_state['overs']}")
    st.write(f"**Run rate:** {match_state['run_rate']}")

    if target is not None:
        st.write(f"**Target:** {target}")
        st.write(
            f"**Runs required:** {match_state['runs_required']}"
        )
        st.write(
            f"**Balls remaining:** {match_state['balls_remaining']}"
        )

        if match_state["required_run_rate"] is not None:
            st.write(
                f"**Required run rate:** "
                f"{match_state['required_run_rate']}"
            )

    st.caption(
        "The displayed state represents the selected historical "
        "replay position, not a live score."
    )

    if st.button(
        "Clear conversation",
        width="stretch",
    ):
        try:
            clear_cricket_chat_history(thread_id)
            st.session_state.pop("pending_question", None)
            st.session_state["chat_history"] = []
            st.session_state["chat_selection"] = current_selection
            st.rerun()

        except Exception as exc:
            st.error(
                "Could not clear the saved conversation. "
                f"Error type: {type(exc).__name__}"
            )


# =========================================================
# 11. CHATBOT INTERFACE
# =========================================================

st.subheader("💬 Chat with your cricket assistant")

st.markdown(
    f"**Selected match:** {match_context['teams']}  \n"
    f"**Replay state:** {batting_team} "
    f"{match_state['runs']}/{match_state['wickets']} "
    f"after {match_state['overs']} overs"
)

st.caption(
    "Ask about the selected historical state, win probability, "
    "turning points, or hypothetical deliveries."
)


# ---------------------------------------------------------
# SUGGESTED QUESTIONS
# ---------------------------------------------------------

with st.expander("Try asking"):
    suggestions = [
        "Explain the current match situation.",
        "What is the chasing team's win probability?",
        "What were the key turning points in this match?",
        "What if the next three balls are a single, a four, and a six?",
    ]

    for index, suggestion in enumerate(suggestions):
        if st.button(
            suggestion,
            key=f"suggested_question_{index}",
           width="stretch",
        ):
            st.session_state["pending_question"] = suggestion
            st.rerun()


# ---------------------------------------------------------
# RESTORE CHAT HISTORY FROM SQLITE
# ---------------------------------------------------------

try:
    chat_history = get_cricket_chat_history(thread_id)

except Exception as exc:
    st.error(
        "Unable to load saved conversation. "
        f"Error type: {type(exc).__name__}"
    )
    chat_history = []


st.session_state["chat_history"] = chat_history

for message in chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# PROCESS USER QUESTION
# ---------------------------------------------------------

pending_question = st.session_state.pop(
    "pending_question",
    None,
)

typed_question = st.chat_input(
    "Ask about this historical match..."
)

question = typed_question or pending_question

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing the selected match state..."):
            try:
                answer = ask_cricket_assistant(
                    question=question,
                    match_context=match_context,
                    thread_id=thread_id,
                )

                st.markdown(answer)

            except Exception as exc:
                answer = (
                    "I could not process that question. Check your "
                    "Groq API configuration and connection, then retry.\n\n"
                    f"Error type: `{type(exc).__name__}`"
                )

                st.error(answer)


# =========================================================
# 12. HISTORICAL DELIVERY LOG AT THE SELECTED POINT
# =========================================================

with st.expander("View historical ball-by-ball data at replay position"):
    columns = [
        column
        for column in [
            "over",
            "delivery_in_over",
            "batter",
            "bowler",
            "batter_runs",
            "extra_runs",
            "total_runs",
            "wicket",
        ]
        if column in selected_rows.columns
    ]

    if selected_rows.empty:
        st.info(
            "The replay is positioned before the first recorded delivery."
        )
    else:
        st.dataframe(
            selected_rows[columns].reset_index(drop=True),
           width="stretch",
            hide_index=True,
        )

    st.caption(
        f"Showing {len(selected_rows):,} of "
        f"{len(innings_rows):,} recorded deliveries."
    )


# =========================================================
# MATCH SUMMARY — PINECONE + GROQ
# =========================================================

st.divider()
st.header("AI Historical Match Summary")
st.caption(
    "Generate a natural-language summary using verified "
    "historical statistics and Pinecone-retrieved context."
)

if st.button(
    "Generate Match Summary",
    key="generate_match_summary",
    type="primary",
):
    try:
        with st.spinner("Generating match summary..."):
            result = generate_natural_language_summary(
                match_id=str(selected_match_id),
                question=(
                    "Summarize this cricket match. Include the innings "
                    "scores, leading batters, leading bowlers, and "
                    "the result if it is supported by the data."
                ),
            )

        st.subheader("Match Summary")
        st.write(result["summary"])

        st.subheader("Verified Statistics")
        for innings_data in result["verified_statistics"][
            "innings_summaries"
        ]:
            st.markdown(
                f"**Innings {innings_data['innings']}: "
                f"{innings_data['batting_team']} — "
                f"{innings_data['runs']}/"
                f"{innings_data['wickets']} "
                f"({innings_data['overs']} overs)**"
            )

        st.caption(
            f"Pinecone context records retrieved: "
            f"{result['retrieved_context_count']}"
        )

        with st.expander("View retrieved historical context"):
            from src.cricket.summary_retriever import (
                search_match_summaries,
            )

            context = search_match_summaries(
                query=f"Summary and key performances in match "
                f"{selected_match_id}",
                top_k=3,
                match_id=str(selected_match_id),
            )

            if context:
                for item in context:
                    st.markdown(
                        f"**Match:** {item['match_id']}  \n"
                        f"**Similarity:** {item['score']:.4f}"
                    )
                    st.write(item["summary_text"])
            else:
                st.info("No indexed context was found for this match.")

    except Exception as exc:
        st.error(f"Unable to generate match summary: {exc}")
