
import operator
import tempfile
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


class TestState(TypedDict):
    message: str
    history: Annotated[list[str], operator.add]


def record_message(state: TestState) -> dict:
    return {"history": [state["message"]]}


def build_test_graph(checkpointer):
    workflow = StateGraph(TestState)
    workflow.add_node("record", record_message)
    workflow.add_edge(START, "record")
    workflow.add_edge("record", END)
    return workflow.compile(checkpointer=checkpointer)


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        database_path = Path(temp_dir) / "checkpoint_test.sqlite"
        config = {"configurable": {"thread_id": "match-1001349-innings-2"}}

        # First application session: save a conversation checkpoint.
        with SqliteSaver.from_conn_string(str(database_path)) as saver:
            graph = build_test_graph(saver)
            first = graph.invoke(
                {"message": "Explain the current score.", "history": []},
                config=config,
            )
            assert first["history"] == ["Explain the current score."]

        # Second application session: reopen SQLite and restore the thread.
        with SqliteSaver.from_conn_string(str(database_path)) as saver:
            graph = build_test_graph(saver)
            second = graph.invoke(
                {"message": "What is the win probability?"},
                config=config,
            )

            expected = [
                "Explain the current score.",
                "What is the win probability?",
            ]
            assert second["history"] == expected, second

        print("PASS: SQLite checkpoint persisted across saver restarts.")
        print("Restored conversation:")
        for message in second["history"]:
            print(f"- {message}")


if __name__ == "__main__":
    main()
