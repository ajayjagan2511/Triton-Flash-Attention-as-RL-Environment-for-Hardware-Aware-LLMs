from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pm_env.schemas.scoring import Scoring
from pm_env.schemas.transcript import Transcript

if TYPE_CHECKING:
    from pm_env.judges._composite_judge import AndJudge, OrJudge


class Judge(ABC):
    @abstractmethod
    def evaluate(self, transcript: Transcript) -> Scoring: ...

    def __and__(self, other: "Judge") -> "AndJudge":
        from pm_env.judges._composite_judge import AndJudge

        if isinstance(self, AndJudge):
            return AndJudge([*self.judges, other])
        return AndJudge([self, other])

    def __or__(self, other: "Judge") -> "OrJudge":
        from pm_env.judges._composite_judge import OrJudge

        if isinstance(self, OrJudge):
            return OrJudge([*self.judges, other])
        return OrJudge([self, other])
