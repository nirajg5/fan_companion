
from unittest.mock import patch

from src.assistant.graph import resolve_follow_up_question


def test_follow_up_question_resolution():
    history = [
        {
            "role": "user",
            "content": "Explain the current match situation at 120/4.",
        },
        {
            "role": "assistant",
            "content": "The batting team has scored 120 runs and lost 4 wickets.",
        },
    ]

    expected = "What happens if the next delivery is a six?"

    with patch(
        "src.assistant.graph.get_groq_client"
    ) as mock_client:
        mock_client.return_value.chat.completions.create.return_value.choices = [
            type(
                "Choice",
                (),
                {
                    "message": type(
                        "Message",
                        (),
                        {"content": expected},
                    )()
                },
            )()
        ]

        actual = resolve_follow_up_question(
            "What if the next ball is a six?",
            history,
        )

    assert actual == expected
