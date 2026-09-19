from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


@dataclass(frozen=True)
class NumericalAssociationResult:
    feature: str
    u_statistic: float
    p_value: float
    rank_biserial: float
    group_no_size: int
    group_yes_size: int
    group_no_median: float
    group_yes_median: float

@dataclass(frozen=True)
class CategoricalAssociationResult:
    feature: str
    chi2: float
    p_value: float
    degrees_of_freedom: int
    cramers_v: float
    sample_size: int


class ChurnStatisticalAnalyzer:
    """Statistical association analysis for customer churn."""

    def categorical_association(
        self,
        data: pd.DataFrame,
        feature: str,
        target: str = "Churn",
    ) -> CategoricalAssociationResult:
        if feature not in data.columns:
            raise KeyError(f"Feature not found: {feature}")

        if target not in data.columns:
            raise KeyError(f"Target not found: {target}")

        contingency = pd.crosstab(
            data[feature],
            data[target],
        )

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            raise ValueError(
                "Chi-square requires at least two categories "
                "for both feature and target."
            )

        chi2, p_value, dof, _ = chi2_contingency(
            contingency,
        )

        n = int(contingency.to_numpy().sum())
        rows, columns = contingency.shape

        denominator = min(rows - 1, columns - 1)

        if denominator <= 0 or n == 0:
            cramers_v = 0.0
        else:
            cramers_v = float(
                np.sqrt(
                    (chi2 / n) / denominator
                )
            )

        return CategoricalAssociationResult(
            feature=feature,
            chi2=float(chi2),
            p_value=float(p_value),
            degrees_of_freedom=int(dof),
            cramers_v=cramers_v,
            sample_size=n,
        )
    def numerical_association(
        self,
        data: pd.DataFrame,
        feature: str,
        target: str = "Churn",
    ) -> NumericalAssociationResult:
        if feature not in data.columns:
            raise KeyError(f"Feature not found: {feature}")

        if target not in data.columns:
            raise KeyError(f"Target not found: {target}")

        group_no = (
            pd.to_numeric(
                data.loc[data[target] == "No", feature],
                errors="coerce",
            )
            .dropna()
            .to_numpy()
        )

        group_yes = (
            pd.to_numeric(
                data.loc[data[target] == "Yes", feature],
                errors="coerce",
            )
            .dropna()
            .to_numpy()
        )

        if len(group_no) == 0 or len(group_yes) == 0:
            raise ValueError(
                "Both churn groups must contain numerical observations."
            )

        result = mannwhitneyu(
            group_yes,
            group_no,
            alternative="two-sided",
        )

        n_yes = len(group_yes)
        n_no = len(group_no)

        rank_biserial = (
            (2.0 * float(result.statistic))
            / (n_yes * n_no)
            - 1.0
        )

        return NumericalAssociationResult(
            feature=feature,
            u_statistic=float(result.statistic),
            p_value=float(result.pvalue),
            rank_biserial=float(rank_biserial),
            group_no_size=n_no,
            group_yes_size=n_yes,
            group_no_median=float(np.median(group_no)),
            group_yes_median=float(np.median(group_yes)),
        )
