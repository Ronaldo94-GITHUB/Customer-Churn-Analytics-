from dataclasses import dataclass
from enum import Enum


class ChurnRiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class ChurnRiskAssessment:
    probability: float
    level: ChurnRiskLevel


class ChurnRiskSegmenter:
    """Convert churn probability into an operational risk segment."""

    def __init__(
        self,
        medium_threshold: float = 0.30,
        high_threshold: float = 0.60,
    ) -> None:
        if not 0.0 <= medium_threshold < high_threshold <= 1.0:
            raise ValueError(
                "Risk thresholds must satisfy "
                "0 <= medium < high <= 1."
            )

        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold

    def segment(
        self,
        probability: float,
    ) -> ChurnRiskAssessment:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "Churn probability must be between 0 and 1."
            )

        if probability >= self.high_threshold:
            level = ChurnRiskLevel.HIGH
        elif probability >= self.medium_threshold:
            level = ChurnRiskLevel.MEDIUM
        else:
            level = ChurnRiskLevel.LOW

        return ChurnRiskAssessment(
            probability=float(probability),
            level=level,
        )