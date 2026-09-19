from dataclasses import dataclass

import pandas as pd

from customer_churn.analytics.customer_profile import (
    CustomerRetentionProfiler,
)


@dataclass(frozen=True)
class ChurnCustomerIntelligence:
    customer_id: str
    churn_probability: float
    risk_level: str
    retention_priority: str


class ChurnIntelligenceBuilder:
    """Build customer-level churn intelligence for BI and operations."""

    def __init__(
        self,
        profiler: CustomerRetentionProfiler | None = None,
    ) -> None:
        self.profiler = profiler or CustomerRetentionProfiler()

    def build(
        self,
        customer_ids,
        probabilities,
    ) -> pd.DataFrame:
        if len(customer_ids) != len(probabilities):
            raise ValueError(
                "Customer IDs and probabilities must have the same length."
            )

        records = []

        for customer_id, probability in zip(
            customer_ids,
            probabilities,
            strict=True,
        ):
            profile = self.profiler.build(
                str(customer_id),
                float(probability),
            )

            records.append(
                {
                    "customer_id": profile.customer_id,
                    "churn_probability": profile.churn_probability,
                    "risk_level": profile.risk_level.value,
                    "retention_priority": (
                        profile.retention_priority.value
                    ),
                }
            )

        return pd.DataFrame.from_records(
            records,
            columns=[
                "customer_id",
                "churn_probability",
                "risk_level",
                "retention_priority",
            ],
        )