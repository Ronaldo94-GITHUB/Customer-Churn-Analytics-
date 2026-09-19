from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChurnOverview:
    customers: int
    churned_customers: int
    retained_customers: int
    churn_rate: float
    average_tenure: float
    average_monthly_charges: float


class ChurnEDA:
    """Exploratory analytics for customer churn."""

    def overview(self, dataframe: pd.DataFrame) -> ChurnOverview:
        customers = len(dataframe)

        if customers == 0:
            return ChurnOverview(
                customers=0,
                churned_customers=0,
                retained_customers=0,
                churn_rate=0.0,
                average_tenure=0.0,
                average_monthly_charges=0.0,
            )

        churned = int(dataframe["Churn"].eq("Yes").sum())
        retained = int(dataframe["Churn"].eq("No").sum())

        return ChurnOverview(
            customers=customers,
            churned_customers=churned,
            retained_customers=retained,
            churn_rate=churned / customers,
            average_tenure=float(dataframe["tenure"].mean()),
            average_monthly_charges=float(
                dataframe["MonthlyCharges"].mean()
            ),
        )

    def churn_by_category(
        self,
        dataframe: pd.DataFrame,
        column: str,
    ) -> pd.DataFrame:
        grouped = (
            dataframe.groupby(column, dropna=False)["Churn"]
            .agg(
                customers="size",
                churned=lambda values: int(values.eq("Yes").sum()),
            )
            .reset_index()
        )

        grouped["churn_rate"] = (
            grouped["churned"] / grouped["customers"]
        )

        return grouped.sort_values(
            "churn_rate",
            ascending=False,
        ).reset_index(drop=True)

    def numerical_by_churn(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compare numerical customer characteristics by churn outcome."""

        columns = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ]

        return (
            dataframe.groupby("Churn")[columns]
            .agg(["mean", "median"])
            .round(2)
        )

    def tenure_segments(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate churn across customer tenure segments."""

        analyzed = dataframe.copy()

        analyzed["tenure_segment"] = pd.cut(
            analyzed["tenure"],
            bins=[-1, 12, 24, 48, 60, float("inf")],
            labels=[
                "0-12 months",
                "13-24 months",
                "25-48 months",
                "49-60 months",
                "61+ months",
            ],
        )

        return self.churn_by_category(
            analyzed,
            "tenure_segment",
        )