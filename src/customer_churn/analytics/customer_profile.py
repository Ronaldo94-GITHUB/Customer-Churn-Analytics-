from dataclasses import dataclass

from customer_churn.analytics.retention import (
    ChurnRetentionAdvisor,
    RetentionPriority,
)
from customer_churn.analytics.risk import (
    ChurnRiskLevel,
    ChurnRiskSegmenter,
)


@dataclass(frozen=True)
class CustomerRetentionProfile:
    customer_id: str
    churn_probability: float
    risk_level: ChurnRiskLevel
    retention_priority: RetentionPriority


class CustomerRetentionProfiler:
    """Build an operational retention profile for a customer."""

    def __init__(
        self,
        segmenter: ChurnRiskSegmenter | None = None,
        advisor: ChurnRetentionAdvisor | None = None,
    ) -> None:
        self.segmenter = segmenter or ChurnRiskSegmenter()
        self.advisor = advisor or ChurnRetentionAdvisor()

    def build(
        self,
        customer_id: str,
        churn_probability: float,
    ) -> CustomerRetentionProfile:
        if not customer_id.strip():
            raise ValueError(
                "Customer ID must not be empty."
            )

        risk = self.segmenter.segment(
            churn_probability
        )
        retention = self.advisor.assess(risk)

        return CustomerRetentionProfile(
            customer_id=customer_id,
            churn_probability=risk.probability,
            risk_level=risk.level,
            retention_priority=retention.priority,
        )