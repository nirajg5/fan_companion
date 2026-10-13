
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from src.assistant.graph import classify_intent


@pytest.fixture
def mock_groq(monkeypatch):
    mock_client = Mock()

    monkeypatch.setattr(
        "src.assistant.graph.get_groq_client",
        lambda: mock_client,
    )

    def set_intent(intent):
        mock_client.chat.completions.create.return_value = (
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps({"intent": intent})
                        )
                    )
                ]
            )
        )

    mock_groq_client = mock_client
    mock_groq_client.set_intent = set_intent
    return mock_groq_client


def test_scenario_question(mock_groq):
    mock_groq.set_intent("scenario")
    assert classify_intent("What if the next ball is a six?") == "scenario"


def test_win_probability_question(mock_groq):
    mock_groq.set_intent("win_probability")
    assert classify_intent("What is the win probability?") == "win_probability"


def test_turning_point_question(mock_groq):
    mock_groq.set_intent("turning_point")
    assert classify_intent("What was the turning point?") == "turning_point"


def test_general_match_question(mock_groq):
    mock_groq.set_intent("match_situation")
    assert classify_intent("Explain the current match situation") == "match_situation"


def test_invalid_intent_falls_back(mock_groq):
    mock_groq.set_intent("invalid_intent")
    assert classify_intent("Explain the match") == "match_situation"


def test_empty_question_raises_error():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        classify_intent("   ")


def test_groq_failure_falls_back_to_match_situation(monkeypatch):
    def raise_error():
        raise RuntimeError("Groq unavailable")

    monkeypatch.setattr(
        "src.assistant.graph.get_groq_client",
        raise_error,
    )

    assert classify_intent("Explain the match") == "match_situation"


def test_invalid_json_falls_back_to_match_situation(mock_groq):
    mock_groq.chat.completions.create.return_value = (
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="not valid json"
                    )
                )
            ]
        )
    )

    assert classify_intent("Explain the match") == "match_situation"
