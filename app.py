import html

import re

import time

import uuid

import altair as alt

import pandas as pd

import streamlit as st

from src.cricket.historical_match import load_deliveries

from src.cricket.match_state import calculate_match_state

from src.assistant.graph import (

    ask_cricket_assistant,

    get_cricket_chat_history,

    clear_cricket_chat_history,

)

# True token streaming requires a streaming entry point in graph.py whose LLM

# calls themselves use stream=True. Import it if you have added that function;

# otherwise this UI uses the existing stable synchronous graph entry point.

try:

    from src.assistant.graph import ask_cricket_assistant_stream

except ImportError:

    ask_cricket_assistant_stream = None

# =========================================================

# 1. PAGE CONFIGURATION + THEME

# =========================================================

st.set_page_config(

    page_title="MatchPulse AI | Cricket Assistant",

    page_icon="🏏",

    layout="wide",

)

PITCH = "#17382E"

CHERRY = "#B4232C"

AMBER = "#F2B134"

st.markdown(

    """

<style>

@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');

:root {

  --pitch: #17382E;

  --pitch-2: #24574a;

  --amber: #F2B134;

  --cherry: #B4232C;

  --paper: #F4F6F2;

  --card: #FFFFFF;

  --ink: #17201C;

  --muted: #5B6A63;

  --line: #D6DDD7;

}

html, body, .stApp { font-family: 'Barlow', sans-serif; }

.stApp { background: var(--paper); color: var(--ink); }

.stApp p, .stApp li, .stApp label { color: var(--ink); }

.block-container { padding-top: 1.2rem; max-width: 1080px; }

header[data-testid="stHeader"] { background: transparent; }

h1, h2, h3, h4 { font-family: 'Barlow Condensed', sans-serif; letter-spacing: .2px; color: var(--ink); }

/* ---------- sidebar ---------- */

[data-testid="stSidebar"] { background: #E9EFE8; border-right: 1px solid var(--line); }

[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size: 1.35rem; }

/* ---------- buttons ---------- */

.stButton > button {

  border-radius: 10px; border: 1px solid var(--line); background: #fff; color: var(--ink);

  font-weight: 600; transition: border-color .15s, background .15s;

}

.stButton > button:hover { border-color: var(--pitch); color: var(--pitch); background: #fff; }

.stButton > button:focus-visible { outline: 3px solid var(--amber); outline-offset: 2px; }

.stButton > button[kind="primary"] { background: var(--pitch); color: #fff; border-color: var(--pitch); }

.stButton > button[kind="primary"]:hover { background: var(--pitch-2); color: #fff; }

/* ---------- scoreboard ---------- */

.board { background: var(--pitch); color: #F3F6F2; border-radius: 6px; padding: 14px 20px 12px; margin-bottom: 14px;

         border-bottom: 5px solid var(--amber); }

.board-top { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap;

             font-size: .88rem; color: #BFD0C7; }

.board-top .tag { background: rgba(242,177,52,.16); color: var(--amber); padding: 1px 10px; border-radius: 999px; font-weight: 600; }

.board-main { display: flex; align-items: baseline; gap: 22px; flex-wrap: wrap; margin: 4px 0 2px; }

.board-team { font-family: 'Barlow Condensed'; font-size: 1.6rem; font-weight: 600; color: #fff; }

.board-score { font-family: 'Barlow Condensed'; font-size: 3.6rem; font-weight: 700; color: var(--amber); line-height: 1; }

.board-score .slash { color: #8FA89C; margin: 0 2px; }

.board-overs { font-family: 'Barlow Condensed'; font-size: 1.7rem; color: #fff; }

.board-overs small { font-size: .95rem; color: #BFD0C7; margin-left: 4px; }

.board-stats { display: flex; gap: 26px; flex-wrap: wrap; margin-top: 6px; }

.board-stats div { font-size: .8rem; color: #BFD0C7; }

.board-stats b { display: block; font-family: 'Barlow Condensed'; font-size: 1.35rem; color: #fff; font-weight: 600; }

.board-bar { height: 4px; background: rgba(255,255,255,.14); border-radius: 4px; margin-top: 10px; overflow: hidden; }

.board-bar span { display: block; height: 100%; background: var(--amber); }

/* ---------- over strip ---------- */

.overstrip { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; background: var(--card);

             border: 1px solid var(--line); border-radius: 12px; padding: 10px 14px; margin-bottom: 14px; }

.overstrip .lbl { font-weight: 600; color: var(--muted); font-size: .9rem; }

.overstrip .last { color: var(--muted); font-size: .92rem; margin-left: auto; }

.ball { display: inline-flex; width: 34px; height: 34px; border-radius: 50%; align-items: center; justify-content: center;

        font-family: 'Barlow Condensed'; font-weight: 700; font-size: 1.1rem; margin-right: 6px;

        border: 2px solid var(--line); background: #fff; color: var(--ink); }

.ball.dot { color: #8A978F; }

.ball.four { background: #2E8B57; border-color: #2E8B57; color: #fff; }

.ball.six { background: #5B3FA0; border-color: #5B3FA0; color: #fff; }

.ball.w { background: var(--cherry); border-color: var(--cherry); color: #fff; }

.ball.x { border-style: dashed; border-color: #8A978F; }

/* ---------- step tracker ---------- */

.steps { display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 12px; }

.step { padding: 3px 14px; border-radius: 999px; border: 1px solid var(--line); background: #fff;

        font-size: .88rem; color: var(--muted); }

.step.done { border-color: var(--pitch); color: var(--pitch); }

.step.on { background: var(--pitch); border-color: var(--pitch); color: #fff; font-weight: 600; }

/* ---------- menu cards ---------- */

[class*="st-key-card"] { background: var(--card); border: 1px solid var(--line) !important; border-radius: 14px; }

[class*="st-key-card"]:hover { border-color: var(--pitch) !important; }

.card-icon { font-size: 1.7rem; margin-bottom: 2px; }

.card-title { font-family: 'Barlow Condensed'; font-size: 1.35rem; font-weight: 700; line-height: 1.15; margin-bottom: 4px; }

.card-desc { color: var(--muted); font-size: .94rem; margin-bottom: 6px; }

.card-list { margin: 0 0 8px 1.1rem; padding: 0; color: var(--ink); font-size: .9rem; }

/* ---------- chat ---------- */

[data-testid="stChatMessage"] { background: var(--card); border: 1px solid var(--line); border-radius: 14px; }

.pill { display: inline-block; background: #fff; border: 1px solid var(--pitch); color: var(--pitch);

        border-radius: 999px; padding: 2px 14px; font-weight: 600; }

.hint { color: var(--muted); font-size: .85rem; }

@media (max-width: 640px) {

  .board-score { font-size: 2.8rem; }

  .overstrip .last { margin-left: 0; }

}

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }

</style>

""",

    unsafe_allow_html=True,

)

st.markdown("# 🏏 MatchPulse AI")

# =========================================================

# 2. LOAD HISTORICAL DATA

# =========================================================

@st.cache_data(show_spinner="Loading historical cricket data...")

def get_data():

    return load_deliveries()

@st.cache_data(show_spinner="Preparing historical match list...")

def get_match_list():

    data = get_data()

    result = (

        data.groupby("match_id", as_index=False)

        .agg(teams=("teams", "first"), match_type=("match_type", "first"))

    )

    result["match_id"] = result["match_id"].astype(str)

    return result.sort_values("match_id").reset_index(drop=True)

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

# 3. HELPERS

# =========================================================

def to_int(value, default=0):

    try:

        if value is None or pd.isna(value):

            return default

        return int(value)

    except (TypeError, ValueError):

        return default

def is_wicket(value) -> bool:

    if value is None:

        return False

    if isinstance(value, float) and pd.isna(value):

        return False

    return str(value).strip().lower() not in ("", "false", "0", "none", "nan")

def esc(value) -> str:

    return html.escape(str(value))

def ball_chip(delivery: dict) -> str:

    batter_runs = to_int(delivery.get("batter_runs"))

    extra_runs = to_int(delivery.get("extra_runs"))

    total = to_int(delivery.get("total_runs"))

    if is_wicket(delivery.get("wicket")):

        text, cls = "W", "w"

    elif batter_runs == 6:

        text, cls = "6", "six"

    elif batter_runs == 4:

        text, cls = "4", "four"

    elif extra_runs > 0:

        text, cls = str(total), "x"

    elif total == 0:

        text, cls = "•", "dot"

    else:

        text, cls = str(total), "run"

    return f'<span class="ball {cls}">{text}</span>'

def describe_delivery(delivery: dict) -> str:

    bowler = delivery.get("bowler")

    batter = delivery.get("batter")

    who = f"{bowler} to {batter}" if bowler and batter else "Last ball"

    if is_wicket(delivery.get("wicket")):

        what = "wicket"

    else:

        total = to_int(delivery.get("total_runs"))

        what = "no run" if total == 0 else f"{total} run{'s' if total != 1 else ''}"

    return f"{who}: {what}"

# =========================================================

# 4. SIDEBAR — MATCH + REPLAY CONTROLS

# =========================================================

with st.sidebar:

    st.header("🏏 Match context")

    selected_match_id = st.selectbox(

        "Historical match",

        options=matches["match_id"].tolist(),

        format_func=lambda match_id: (

            f"{match_id} — "

            f"{matches.loc[matches['match_id'] == match_id, 'teams'].iloc[0]}"

        ),

        key="selected_match_id",

    )

match_rows = df[df["match_id"].astype(str) == str(selected_match_id)].copy()

match_rows = match_rows.sort_values(

    ["innings", "over", "delivery_in_over"]

).reset_index(drop=True)

if match_rows.empty:

    st.error("No deliveries were found for this match.")

    st.stop()

innings_options = sorted(match_rows["innings"].unique().tolist())

with st.sidebar:

    selected_innings = st.selectbox(

        "Innings",

        options=innings_options,

        format_func=lambda innings: (

            f"Innings {innings} — "

            f"{match_rows.loc[match_rows['innings'] == innings, 'batting_team'].iloc[0]}"

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

total_deliveries = len(innings_rows)

# Pre-compute jump points for the replay controls.

wicket_positions = (

    [i + 1 for i, v in enumerate(innings_rows["wicket"]) if is_wicket(v)]

    if "wicket" in innings_rows.columns

    else []

)

over_values = innings_rows["over"].tolist()

over_end_positions = [

    i + 1 for i in range(total_deliveries - 1) if over_values[i] != over_values[i + 1]

] + [total_deliveries]

slider_key = f"replay_{selected_match_id}_{selected_innings}"

st.session_state.setdefault(slider_key, total_deliveries)

def set_position(value: int) -> None:

    st.session_state[slider_key] = max(0, min(total_deliveries, int(value)))

def current_position() -> int:

    return to_int(st.session_state.get(slider_key), total_deliveries)

def step_ball(delta: int) -> None:

    set_position(current_position() + delta)

def next_over() -> None:

    cur = current_position()

    set_position(next((p for p in over_end_positions if p > cur), total_deliveries))

def prev_over() -> None:

    cur = current_position()

    set_position(next((p for p in reversed(over_end_positions) if p < cur), 0))

def next_wicket() -> None:

    cur = current_position()

    set_position(next((p for p in wicket_positions if p > cur), total_deliveries))

def prev_wicket() -> None:

    cur = current_position()

    set_position(next((p for p in reversed(wicket_positions) if p < cur), 0))

with st.sidebar:

    st.divider()

    st.subheader("⏪ Replay")

    delivery_position = st.slider(

        "Replay position (recorded deliveries)",

        min_value=0,

        max_value=total_deliveries,

        key=slider_key,

        help=(

            "Zero means before the first delivery. The maximum means the end "

            "of the recorded innings. The assistant analyzes the state after "

            "the selected number of recorded deliveries."

        ),

    )

    c1, c2, c3, c4 = st.columns(4)

    c1.button("⏮", key="rp_start", on_click=set_position, args=(0,), help="Before first ball", width="stretch")

    c2.button("◀", key="rp_back", on_click=step_ball, args=(-1,), help="Back one ball", width="stretch")

    c3.button("▶", key="rp_fwd", on_click=step_ball, args=(1,), help="Forward one ball", width="stretch")

    c4.button("⏭", key="rp_end", on_click=set_position, args=(total_deliveries,), help="End of innings", width="stretch")

    o1, o2 = st.columns(2)

    o1.button("◀ Over", key="rp_pover", on_click=prev_over, width="stretch")

    o2.button("Over ▶", key="rp_nover", on_click=next_over, width="stretch")

    w1, w2 = st.columns(2)

    w1.button("◀ Wicket", key="rp_pwkt", on_click=prev_wicket, disabled=not wicket_positions, width="stretch")

    w2.button("Wicket ▶", key="rp_nwkt", on_click=next_wicket, disabled=not wicket_positions, width="stretch")

    st.caption(f"Ball {delivery_position:,} of {total_deliveries:,} recorded deliveries")

selected_rows = innings_rows.iloc[:delivery_position].copy()

selected_deliveries = selected_rows.to_dict(orient="records")

# =========================================================

# 5. CALCULATE MATCH STATE AND CONTEXT

# =========================================================

target = None

defending_team = None

if selected_innings == 2:

    first_innings = match_rows[match_rows["innings"] == 1]

    if not first_innings.empty:

        target = int(first_innings["total_runs"].sum()) + 1

        defending_team = str(first_innings["batting_team"].iloc[0])

try:

    match_state = calculate_match_state(

        deliveries=selected_deliveries,

        innings=int(selected_innings),

        target=target,

    )

except Exception as exc:

    st.error(f"Unable to calculate the selected match state: {exc}")

    st.stop()

match_context = {

    "data_source": "Cricsheet historical replay",

    "historical_replay": True,

    "match_id": str(selected_match_id),

    "teams": str(match_rows["teams"].iloc[0]),

    "match_type": str(match_rows["match_type"].iloc[0]),

    "batting_team": batting_team,

    "chasing_team": batting_team if target is not None else None,

    "defending_team": defending_team,

    "innings": int(selected_innings),

    "max_overs": 20,

    "replay_position": int(delivery_position),

    "recorded_deliveries": len(innings_rows),

    "recorded_deliveries_included": int(delivery_position),

    "total_recorded_deliveries": len(innings_rows),

    "match_situation": {

        "runs": match_state["runs"],

        "wickets": match_state["wickets"],

        "score": f"{match_state['runs']}/{match_state['wickets']}",

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

current_selection = (str(selected_match_id), int(selected_innings))

thread_id = f"match-{selected_match_id}-innings-{selected_innings}"

# =========================================================

# 6. MENU CONFIGURATION

# =========================================================

MAIN_MENU = {

    "A. Match Understanding and Analysis": [

        "AI Match Assistant",

        "Match Situation Explanation",

        "Match Recap Generator",

    ],

    "B. Live Match Companion (Historical Replay)": [

        "Live Match State",

        "Live Catch-up",

        "Ball-by-ball Explanations",

        "AI Event Narration",

    ],

    "C. Match Intelligence and Predictive Analysis": [

        "What-if Scenario Simulation",

        "Turning-point Detection",

        "Win Probability",

    ],

}

CATEGORY_META = {

    "A. Match Understanding and Analysis": {

        "icon": "🧠",

        "blurb": "Understand what is happening and why, in plain language.",

    },

    "B. Live Match Companion (Historical Replay)": {

        "icon": "📡",

        "blurb": "Follow the match as it unfolds, ball by ball.",

    },

    "C. Match Intelligence and Predictive Analysis": {

        "icon": "📈",

        "blurb": "Test scenarios and find the moments that decided the game.",

    },

}

FEATURE_PROMPTS = {

    "AI Match Assistant":

        "Act as my cricket match assistant. Help me understand this selected match.",

    "Match Situation Explanation":

        "Explain the current match situation using the selected historical replay state.",

    "Match Recap Generator":

        "Generate a concise match recap using verified historical data.",

    "Live Match State":

        "Explain the selected replay state. This is historical data, not a live feed.",

    "Live Catch-up":

        "Catch me up on key events in this selected historical match.",

    "Ball-by-ball Explanations":

        "Explain the recorded deliveries around the selected replay position.",

    "AI Event Narration":

        "Narrate important recorded events supported by the dataset. Do not invent events.",

    "What-if Scenario Simulation":

        "Help me simulate a hypothetical delivery sequence and explain its impact. Ask me for details if the scenario is not specified.",

    "Turning-point Detection":

        "Identify turning points supported by the selected historical match data.",

    "Win Probability":

        "Estimate win probability if the available model supports this match state, and explain limitations.",

}

FEATURE_META = {

    "AI Match Assistant": {

        "icon": "🤝", "desc": "Ask anything about this match, like a knowledgeable friend.",

        "chips": ["Who is on top right now?", "What should I watch for next?"],

    },

    "Match Situation Explanation": {

        "icon": "🧭", "desc": "A clear read of the score, pressure and momentum at this point.",

        "chips": ["Is the batting side ahead of the game?", "What does the run rate tell us?"],

    },

    "Match Recap Generator": {

        "icon": "📝", "desc": "A short recap built only from the recorded data.",

        "chips": ["Make it shorter", "Recap in three bullet points"],

    },

    "Live Match State": {

        "icon": "📊", "desc": "The state of play at the selected replay point.",

        "chips": ["Summarise this over", "How many runs in the last 3 overs?"],

    },

    "Live Catch-up": {

        "icon": "⏩", "desc": "Missed the start? Get the key events so far.",

        "chips": ["Only the wickets", "Only the boundaries"],

    },

    "Ball-by-ball Explanations": {

        "icon": "🔴", "desc": "What happened on the deliveries around this point, and why it mattered.",

        "chips": ["Explain the last ball", "Explain the last over"],

    },

    "AI Event Narration": {

        "icon": "🎙️", "desc": "Commentary-style narration of the recorded events.",

        "chips": ["Narrate the last over", "Narrate the most recent wicket"],

    },

    "What-if Scenario Simulation": {

        "icon": "🧪", "desc": "Change the next few deliveries and see how the game shifts.",

        "chips": ["What if the next ball is a six?", "What if a wicket falls next ball?"],

    },

    "Turning-point Detection": {

        "icon": "🔀", "desc": "Find the moments where the match swung.",

        "chips": ["Biggest turning point so far", "Why did it matter?"],

    },

    "Win Probability": {

        "icon": "🎯", "desc": "Estimated chances at this point, with the limits explained.",

        "chips": ["What would change this estimate?", "Explain how you got this"],

    },

}

# =========================================================

# 7. STATE MANAGEMENT

# =========================================================

def normalize_command(value: str) -> str:

    return re.sub(r"\s+", " ", value.strip().lower())

def add_message(role: str, content: str) -> None:

    st.session_state["mp_messages"].append({"role": role, "content": content})

def start_new_guided_session() -> None:

    """Start a clean UI flow with a fresh LangGraph thread id."""

    st.session_state["mp_session_id"] = uuid.uuid4().hex[:12]

    st.session_state["mp_stage"] = "welcome"

    st.session_state["mp_category"] = None

    st.session_state["mp_feature"] = None

    st.session_state["mp_status"] = "active"

    st.session_state["mp_messages"] = []

    st.session_state["mp_streaming"] = False

    st.session_state["mp_thread_id"] = (

        f"{thread_id}-session-{st.session_state['mp_session_id']}"

    )

    add_message(

        "assistant",

        "Welcome to MatchPulse AI! 👋 Send **hi** or **hello** to get started. "

        "I will then show you three categories. After you choose a category "

        "and feature, the conversation will stay focused on that feature.",

    )

if "mp_selection" not in st.session_state:

    st.session_state["mp_selection"] = current_selection

elif st.session_state["mp_selection"] != current_selection:

    st.session_state["mp_selection"] = current_selection

    # A different match/innings gets a new UI conversation, without altering

    # any of the user's other persisted LangGraph threads.

    st.session_state.pop("mp_session_id", None)

if "mp_session_id" not in st.session_state:

    start_new_guided_session()

elif "mp_messages" not in st.session_state:

    st.session_state["mp_messages"] = []

if "mp_stage" not in st.session_state:

    st.session_state["mp_stage"] = "main_menu"

if "mp_status" not in st.session_state:

    st.session_state["mp_status"] = "active"

if "mp_streaming" not in st.session_state:

    st.session_state["mp_streaming"] = False

with st.sidebar:

    st.divider()

    if st.button("Clear conversation", key="mp_clear", width="stretch"):

        try:

            clear_cricket_chat_history(st.session_state["mp_thread_id"])

        except Exception:

            # The UI session can still be cleared if the persistence layer fails.

            pass

        start_new_guided_session()

        st.rerun()

    st.caption("Historical replay from Cricsheet recordings. Not a live score.")

def go_main_menu() -> None:

    st.session_state["mp_stage"] = "main_menu"

    st.session_state["mp_category"] = None

    st.session_state["mp_feature"] = None

# =========================================================

# 8. SCOREBOARD HEADER

# =========================================================

legal_balls = to_int(match_state.get("legal_balls"))

progress = min(100, round(legal_balls / (match_context["max_overs"] * 6) * 100))

if target:

    progress = min(100, round(match_state["runs"] / target * 100))

stats = [("Run rate", match_state["run_rate"])]

if target is not None:

    stats.append(("Target", target))

    stats.append(("Need", f"{match_state['runs_required']} off {match_state['balls_remaining']}"))

    if match_state["required_run_rate"] is not None:

        stats.append(("Required rate", match_state["required_run_rate"]))

stats_html = "".join(f"<div>{esc(label)}<b>{esc(value)}</b></div>" for label, value in stats)

st.markdown(

    f"""<div class="board">

<div class="board-top"><span class="tag">Historical replay</span>

<span>{esc(match_context['teams'])} · {esc(match_context['match_type'])} · Innings {int(selected_innings)}</span></div>

<div class="board-main">

<span class="board-team">{esc(batting_team)}</span>

<span class="board-score">{match_state['runs']}<span class="slash">/</span>{match_state['wickets']}</span>

<span class="board-overs">{esc(match_state['overs'])}<small>overs</small></span>

</div>

<div class="board-stats">{stats_html}</div>

<div class="board-bar" title="{'Chase progress' if target else 'Innings progress'}"><span style="width:{progress}%"></span></div>

</div>""",

    unsafe_allow_html=True,

)

# Latest over strip

if selected_deliveries:

    last_delivery = selected_deliveries[-1]

    latest_over = [d for d in selected_deliveries if d.get("over") == last_delivery.get("over")]

    chips_html = "".join(ball_chip(d) for d in latest_over)

    st.markdown(

        f"""<div class="overstrip"><span class="lbl">Latest over</span><span>{chips_html}</span>

<span class="last">{esc(describe_delivery(last_delivery))}</span></div>""",

        unsafe_allow_html=True,

    )

else:

    st.markdown(

        '<div class="overstrip"><span class="lbl">Before the first ball. Use the replay controls in the sidebar.</span></div>',

        unsafe_allow_html=True,

    )

# Runs-per-over scrubber chart

if "over" in selected_rows.columns and "total_runs" in innings_rows.columns:

    current_over = to_int(selected_rows["over"].iloc[-1], -1) if not selected_rows.empty else -1

    def per_over(frame: pd.DataFrame) -> pd.DataFrame:

        if frame.empty:

            return pd.DataFrame(columns=["over", "runs", "wickets"])

        wk = frame["wicket"].map(is_wicket) if "wicket" in frame.columns else pd.Series(False, index=frame.index)

        return (

            frame.assign(_w=wk)

            .groupby("over", as_index=False)

            .agg(runs=("total_runs", "sum"), wickets=("_w", "sum"))

        )

    played = per_over(selected_rows).assign(state="Played")

    ahead = per_over(innings_rows[innings_rows["over"] > current_over]).assign(state="Ahead")

    chart_data = pd.concat([played, ahead], ignore_index=True)

    chart_data["over_label"] = chart_data["over"].astype(int) + 1

    base = alt.Chart(chart_data).encode(x=alt.X("over_label:O", title="Over", axis=alt.Axis(labelAngle=0)))

    bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(

        y=alt.Y("runs:Q", title="Runs"),

        color=alt.Color(

            "state:N",

            scale=alt.Scale(domain=["Played", "Ahead"], range=[PITCH, "#D6DDD7"]),

            legend=None,

        ),

        tooltip=[alt.Tooltip("over_label:O", title="Over"), alt.Tooltip("runs:Q", title="Runs"),

                 alt.Tooltip("wickets:Q", title="Wickets")],

    )

    wicket_marks = (

        base.transform_filter("datum.wickets > 0 && datum.state == 'Played'")

        .mark_circle(size=110, color=CHERRY, opacity=1)

        .encode(y=alt.Y("runs:Q"), tooltip=[alt.Tooltip("over_label:O", title="Over"),

                                            alt.Tooltip("wickets:Q", title="Wickets")])

    )

    with st.expander("📈 Innings shape: runs per over (red dots are wickets)", expanded=False):

        st.altair_chart((bars + wicket_marks).properties(height=180), width="stretch")

        st.caption("Dark bars are overs already played at the replay position; pale bars are still ahead.")

# =========================================================

# 9. GUIDED CHAT — THREE STAGES

# =========================================================

stage = st.session_state["mp_stage"]

if st.session_state["mp_status"] != "closed":

    step_state = {

        "welcome": ("", "", ""),

        "main_menu": ("on", "", ""),

        "submenu": ("done", "on", ""),

        "conversation": ("done", "done", "on"),

    }.get(stage, ("", "", ""))

    st.markdown(

        '<div class="steps">'

        f'<span class="step {step_state[0]}">1 · Category</span>'

        f'<span class="step {step_state[1]}">2 · Feature</span>'

        f'<span class="step {step_state[2]}">3 · Conversation</span>'

        "</div>",

        unsafe_allow_html=True,

    )

for message in st.session_state["mp_messages"]:

    avatar = "🏏" if message["role"] == "assistant" else "🧑"

    with st.chat_message(message["role"], avatar=avatar):

        st.markdown(message["content"])

if st.session_state["mp_status"] == "closed":

    st.info("This conversation is closed. Start a new conversation to begin again.")

    if st.button("Start a new conversation", key="mp_start_new", type="primary"):

        start_new_guided_session()

        st.rerun()

    st.stop()

# ---- Stage 1: category cards -------------------------------------------

if stage == "main_menu":

    cols = st.columns(3)

    for idx, category in enumerate(MAIN_MENU):

        meta = CATEGORY_META[category]

        with cols[idx]:

            with st.container(border=True, key=f"card_cat_{idx}"):

                features_html = "".join(f"<li>{esc(f)}</li>" for f in MAIN_MENU[category])

                st.markdown(

                    f'<div class="card-icon">{meta["icon"]}</div>'

                    f'<div class="card-title">{esc(category)}</div>'

                    f'<div class="card-desc">{esc(meta["blurb"])}</div>'

                    f'<ul class="card-list">{features_html}</ul>',

                    unsafe_allow_html=True,

                )

                if st.button("Open", key=f"mp_category_{idx}", type="primary", width="stretch"):

                    st.session_state["mp_category"] = category

                    st.session_state["mp_stage"] = "submenu"

                    add_message("assistant", f"You selected **{category}**. Now choose a feature to continue.")

                    st.rerun()

# ---- Stage 2: feature cards --------------------------------------------

elif stage == "submenu":

    category = st.session_state["mp_category"]

    top_l, top_r = st.columns([4, 1])

    top_l.markdown(f"#### {CATEGORY_META[category]['icon']} {category}")

    top_r.button("← Categories", key="mp_back_main", on_click=go_main_menu, width="stretch")

    cols = st.columns(2)

    for idx, feature in enumerate(MAIN_MENU[category]):

        meta = FEATURE_META[feature]

        with cols[idx % 2]:

            with st.container(border=True, key=f"card_feat_{idx}"):

                st.markdown(

                    f'<div class="card-icon">{meta["icon"]}</div>'

                    f'<div class="card-title">{esc(feature)}</div>'

                    f'<div class="card-desc">{esc(meta["desc"])}</div>',

                    unsafe_allow_html=True,

                )

                if st.button("Start", key=f"mp_feature_{idx}", type="primary", width="stretch"):

                    st.session_state["mp_feature"] = feature

                    st.session_state["mp_stage"] = "conversation"

                    feature_slug = re.sub(r"[^a-z0-9]+", "-", feature.lower()).strip("-")

                    st.session_state["mp_thread_id"] = (

                        f"{thread_id}-session-{st.session_state['mp_session_id']}-feature-{feature_slug}"

                    )

                    add_message(

                        "assistant",

                        f"Let's work on **{feature}**. Ask follow-up questions or add details. "

                        "Type `menu` to choose another feature or `exit` to end.",

                    )

                    # The feature prompt is sent once, without showing it as a user message.

                    st.session_state["mp_pending_prompt"] = FEATURE_PROMPTS[feature]

                    st.session_state["mp_pending_visible"] = False

                    st.rerun()

# ---- Stage 3: conversation header + quick prompts ----------------------

elif stage == "conversation":

    feature = st.session_state["mp_feature"]

    head_l, head_r = st.columns([4, 1])

    head_l.markdown(

        f'<span class="pill">{FEATURE_META[feature]["icon"]} {esc(feature)}</span>',

        unsafe_allow_html=True,

    )

    head_r.button("← Change feature", key="mp_change_feature", on_click=go_main_menu, width="stretch")

    quick = list(FEATURE_META[feature]["chips"]) + ["Explain the last delivery"]

    quick = list(dict.fromkeys(quick))[:3]

    st.markdown('<span class="hint">Quick questions</span>', unsafe_allow_html=True)

    qcols = st.columns(len(quick))

    for idx, text in enumerate(quick):

        if qcols[idx].button(text, key=f"mp_quick_{idx}", width="stretch"):

            st.session_state["mp_pending_prompt"] = text

            st.session_state["mp_pending_visible"] = True

            st.rerun()

st.markdown(

    '<span class="hint">Type <code>menu</code> to return to the categories, or <code>exit</code> to close the conversation.</span>',

    unsafe_allow_html=True,

)

# =========================================================

# 10. PROCESS INPUT

# =========================================================

pending_prompt = st.session_state.pop("mp_pending_prompt", None)

pending_visible = st.session_state.pop("mp_pending_visible", False)

typed_question = st.chat_input(

    "Say hi to open the menu..." if stage == "welcome"

    else "Choose a category and feature first..." if stage != "conversation"

    else "Ask a follow-up question, or type menu / exit...",

    key="mp_chat_input",

    disabled=stage not in ("welcome", "conversation"),

)

question = typed_question or pending_prompt

if question and st.session_state["mp_status"] == "active":

    normalized = normalize_command(question)

    if typed_question or pending_visible:

        add_message("user", question)

    if stage == "welcome":

        greetings = {"hi", "hii", "hiii", "hello", "hey", "heyy", "start", "start chat"}

        if normalized in greetings:

            st.session_state["mp_stage"] = "main_menu"

            add_message(

                "assistant",

                "Hi! 👋 What would you like help with? Choose one of these three categories below. "

                "After that, select a feature and we will keep the conversation focused on it.",

            )

        else:

            add_message(

                "assistant",

                "Please send **hi** or **hello** to open the three-category menu.",

            )

        st.rerun()

    elif normalized == "menu":

        go_main_menu()

        add_message("assistant", "Returning to the main menu. Choose one of the three categories.")

        st.rerun()

    elif normalized == "exit":

        add_message("assistant", "Thanks for using MatchPulse AI. Goodbye! 👋")

        st.session_state["mp_status"] = "closed"

        st.session_state["mp_stage"] = "closed"

        st.rerun()

    else:

        # Show the user's bubble immediately for typed or quick questions.

        if typed_question or pending_visible:

            with st.chat_message("user", avatar="🧑"):

                st.markdown(question)

        # Keep every follow-up grounded in the feature the user selected.
        # The UI still displays only the user's original message.
        feature = st.session_state["mp_feature"]
        assistant_question = (
            f"Selected MatchPulse AI feature: {feature}\n"
            f"Feature goal: {FEATURE_PROMPTS[feature]}\n"
            "Keep the answer focused on this selected feature. Do not switch "
            "to another feature unless the user explicitly asks for the menu.\n"
            f"User's message: {question}"
        )

        start_time = time.perf_counter()

        st.session_state["mp_streaming"] = True

        try:

            with st.chat_message("assistant", avatar="🏏"):

                placeholder = st.empty()

                with st.spinner("Analyzing the selected match..."):

                    if ask_cricket_assistant_stream is not None:

                        # Expected streaming function contract: yields text chunks.

                        chunks = ask_cricket_assistant_stream(

                            question=assistant_question,

                            match_context=match_context,

                            thread_id=st.session_state["mp_thread_id"],

                        )

                        answer_parts = []

                        for chunk in chunks:

                            if chunk:

                                answer_parts.append(str(chunk))

                                placeholder.markdown("".join(answer_parts) + " ▌")

                        answer = "".join(answer_parts)

                        placeholder.markdown(answer or "The assistant returned an empty response.")

                    else:

                        answer = ask_cricket_assistant(

                            question=assistant_question,

                            match_context=match_context,

                            thread_id=st.session_state["mp_thread_id"],

                        )

                        placeholder.markdown(answer)

            elapsed = time.perf_counter() - start_time

            add_message("assistant", answer)

            st.session_state["mp_last_latency_seconds"] = round(elapsed, 2)

        except Exception as exc:

            elapsed = time.perf_counter() - start_time

            error_message = (

                "I couldn't complete that response. Please retry; your selected "

                f"feature and match context are unchanged. Error: {type(exc).__name__}."

            )

            add_message("assistant", error_message)

            st.session_state["mp_last_latency_seconds"] = round(elapsed, 2)

        finally:

            st.session_state["mp_streaming"] = False

        st.rerun()

if "mp_last_latency_seconds" in st.session_state:

    st.caption(f"Last response: {st.session_state['mp_last_latency_seconds']:.2f} s")

# =========================================================
