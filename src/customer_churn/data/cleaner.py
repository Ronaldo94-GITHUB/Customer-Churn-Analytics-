import pandas as pd


class ChurnDataCleaningError(Exception):
    """Raised when churn data cannot be safely cleaned."""


class ChurnDataCleaner:
    """Clean raw customer churn data while preserving customer records."""

    def clean(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        cleaned = dataframe.copy()

        cleaned["TotalCharges"] = pd.to_numeric(
            cleaned["TotalCharges"],
            errors="coerce",
        )

        invalid_total_charges = cleaned["TotalCharges"].isna()

        safe_zero_fill = (
            invalid_total_charges
            & cleaned["tenure"].eq(0)
        )

        cleaned.loc[
            safe_zero_fill,
            "TotalCharges",
        ] = 0.0

        unresolved = cleaned["TotalCharges"].isna()

        if unresolved.any():
            count = int(unresolved.sum())

            raise ChurnDataCleaningError(
                f"{count} TotalCharges values could not be resolved."
            )

        return cleaned