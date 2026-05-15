import os
from textwrap import dedent
from typing import Final, TypedDict, override

import anthropic

from pm_env.judges.judge import Judge
from pm_env.schemas.scoring import Metadata, Scoring
from pm_env.schemas.transcript import Transcript


class RubricCriterion(TypedDict):
    """A single criterion in the rubric."""

    criterion: str
    weight: str | float


class RubricJudge(Judge):
    """Evaluates transcript answers against a rubric using an LLM.

    Each criterion in the rubric is evaluated by an LLM, which determines
    whether the criterion is met (binary yes/no). The final score is the
    sum of weights for met criteria.

    The task continues if the final score is greater than the continue_threshold.

    Args:
        rubric: List of criteria, each with a "criterion" (str) and "weight" (float/str)
        model: Model name for Anthropic (e.g., "claude-3-5-sonnet-20241022")
        api_key: API key for the model. If empty, will use ANTHROPIC_API_KEY environment variable
        answer_key: Optional key in transcript.answers to evaluate. If None,
                   all answers are concatenated and evaluated together.
        continue_threshold: Score threshold for continuing the task. Defaults to 0.0.
    """

    def __init__(
        self,
        rubric: list[RubricCriterion],
        model: str,
        api_key: str | None = None,
        answer_key: str | None = None,
        continue_threshold: float = 0.0,
    ) -> None:
        self.rubric: Final = rubric
        self.model: Final = model or "claude-opus-4-5-20251101"
        # Use environment variable if api_key is empty
        self.api_key: Final = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.answer_key: Final = answer_key
        self.continue_threshold: Final = continue_threshold
        self.client: Final = anthropic.Anthropic(api_key=self.api_key)

    @override
    def evaluate(self, transcript: Transcript) -> Scoring:
        # Extract the answer to evaluate
        if self.answer_key:
            answer = transcript.answers.get(self.answer_key, "")
            if not answer:
                return Scoring(
                    score=0.0,
                    metadata={"error": f"Answer key '{self.answer_key}' not found"},
                    continue_task=False,
                )
        else:
            answer = "\n\n".join(
                f"{key}: {value}" for key, value in transcript.answers.items()
            )
            if not answer:
                return Scoring(
                    score=0.0,
                    metadata={"error": "No answers found in transcript"},
                    continue_task=False,
                )

        # Evaluate each criterion
        total_score: float = 0.0
        metadata: Metadata = {}

        for i, item in enumerate(self.rubric):
            criterion = item["criterion"]
            weight = float(item["weight"])

            # Ask LLM if criterion is met
            is_met, reasoning = self._evaluate_criterion(answer, criterion)

            criterion_key = f"criterion_{i}_{criterion[:30]}"
            if is_met:
                total_score += weight
                metadata[criterion_key] = f"✓: {reasoning}"
            else:
                metadata[criterion_key] = f"✗: {reasoning}"

        # Store final score
        metadata["final_score"] = str(total_score)
        metadata["pass_threshold"] = f"> {self.continue_threshold}"

        return Scoring(
            score=total_score,
            metadata=metadata,
            continue_task=total_score > self.continue_threshold,
        )

    def _evaluate_criterion(self, answer: str, criterion: str) -> tuple[bool, str]:
        """Evaluate a single criterion using the LLM.

        Returns:
            Tuple of (is_met, reasoning) where is_met is True if the criterion
            is met, and reasoning is the LLM's explanation.
        """
        prompt = dedent(
            f"""
            You are evaluating whether an answer meets a specific criterion.

            Answer to evaluate:
            {answer}

            Criterion:
            {criterion}

            Does the answer meet this criterion? Respond with ONLY "YES" or "NO", followed by a brief explanation on a new line.

            Format:
            YES
            [brief explanation]

            or

            NO
            [brief explanation]"""
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )

            content = response.content[
                0
            ].text  # pyright: ignore[reportAttributeAccessIssue]
            if not content:
                return False, "LLM returned empty response"

            # Parse response
            lines = content.strip().split("\n", 1)
            decision = lines[0].strip().upper()
            reasoning = (
                lines[1].strip() if len(lines) > 1 else "No explanation provided"
            )

            is_met = decision.startswith("YES")
            return is_met, reasoning

        except Exception as e:
            error_msg = f"Error evaluating criterion: {str(e)}"
            return False, error_msg
