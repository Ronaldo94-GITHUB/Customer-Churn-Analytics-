from dataclasses import dataclass


@dataclass(frozen=True)
class ChurnDatasetContract:
    target: str = "Churn"
    customer_id: str = "customerID"

    required_columns: tuple[str, ...] = (
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    )

    numeric_columns: tuple[str, ...] = (
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    )

    categorical_columns: tuple[str, ...] = (
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    )


DEFAULT_CHURN_CONTRACT = ChurnDatasetContract()