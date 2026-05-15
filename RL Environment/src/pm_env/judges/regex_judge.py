import re
from collections.abc import Sequence
from typing import Final, override

from pm_env.judges.judge import Judge
from pm_env.schemas.scoring import Metadata, Score, Scoring
from pm_env.schemas.transcript import Transcript


class RegexJudge(Judge):
    """Results from the transcript must match the expected patterns via `re.search`.

    If `expected` is a dictionary, the keys are used to retrieve answers from
    the transcript that must match the expected patterns.

    If the expected patterns are a sequence, they get directly matched against a
    concatenated string of all messages the model wrote.
    """

    def __init__(
        self, expected: dict[str, re.Pattern[str]] | Sequence[re.Pattern[str]]
    ) -> None:
        self.expected: Final = expected

    @override
    def evaluate(self, transcript: Transcript) -> Scoring:
        if isinstance(self.expected, dict):
            return self._evaluate_answers(transcript)

        return self._evaluate_full_transcript(transcript)

    def _evaluate_answers(self, transcript: Transcript) -> Scoring:
        assert isinstance(self.expected, dict)

        score: Score = 1.0
        metadata: Metadata = {}

        for key, expected in self.expected.items():
            failure_reason = _check_value(transcript, key, expected)

            if failure_reason:
                score = 0
                metadata[key] = failure_reason

        return Scoring(score, metadata, score == 1.0)

    def _evaluate_full_transcript(self, transcript: Transcript) -> Scoring:
        assert isinstance(self.expected, Sequence)

        full_transcript = "\n\n".join(
            str(m.content) for m in transcript.messages if m.content
        )

        score: Score = 1.0
        metadata: Metadata = {}

        for expected in self.expected:
            if not expected.search(full_transcript):
                score = 0
                metadata[expected.pattern] = "Transcript contains no match."

        return Scoring(score, metadata, score == 1.0)


def _check_value(
    transcript: Transcript, key: str, expected: re.Pattern[str]
) -> str | None:
    """Checks if an answer in the `transcript` matches the `expected` pattern.

    Returns None if it matches, or an error message if it does not.
    """
    value = transcript.answers.get(key)

    if value is None:
        return "Answer not found in transcript."
    else:
        if expected.search(value):
            return None
        else:
            return f"{expected.pattern!r} does not match {value!r}."
