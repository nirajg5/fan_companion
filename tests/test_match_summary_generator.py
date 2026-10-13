
from unittest.mock import patch

from src.cricket.match_summary_generator import (
    generate_natural_language_summary,
)


def test_generate_natural_language_summary():
    with (
        patch(
            "src.cricket.match_summary_generator.generate_match_summary_data"
        ) as mock_data,
        patch(
            "src.cricket.match_summary_generator.build_summary_text"
        ) as mock_text,
        patch(
            "src.cricket.match_summary_generator.search_match_summaries"
        ) as mock_search,
        patch(
            "src.cricket.match_summary_generator.Groq"
        ) as mock_groq,
    ):
        mock_data.return_value = {
            "match_id": "1001349",
            "innings_summaries": [],
        }
        mock_text.return_value = "Australia 168/6; Sri Lanka 172/5."
        mock_search.return_value = [
            {"summary_text": "Historical match context"}
        ]

        mock_groq.return_value.chat.completions.create.return_value = (
            type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {"content": "Sri Lanka won the match."},
                                )()
                            },
                        )()
                    ]
                },
            )()
        )

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            result = generate_natural_language_summary("1001349")

        assert result["match_id"] == "1001349"
        assert result["retrieved_context_count"] == 1
        assert result["summary"] == "Sri Lanka won the match."
