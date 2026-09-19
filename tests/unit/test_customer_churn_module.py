from customer_churn import CustomerChurnModule
from customer_churn.contracts import ModuleStatus
from customer_churn.data.contract import (
    DEFAULT_CHURN_CONTRACT,
)


def test_customer_churn_metadata():
    module = CustomerChurnModule()

    assert module.metadata.key == "customer_churn"
    assert module.metadata.name == "Customer Churn"
    assert module.metadata.version == "1.0.0"
    assert module.metadata.status is ModuleStatus.ACTIVE


def test_customer_churn_health():
    module = CustomerChurnModule()

    assert module.health_check() is True


def test_churn_contract_target():
    assert DEFAULT_CHURN_CONTRACT.target == "Churn"
    assert DEFAULT_CHURN_CONTRACT.customer_id == "customerID"


def test_churn_contract_required_columns():
    required = DEFAULT_CHURN_CONTRACT.required_columns

    assert "customerID" in required
    assert "tenure" in required
    assert "MonthlyCharges" in required
    assert "TotalCharges" in required
    assert "Contract" in required
    assert "Churn" in required


def test_churn_contract_feature_groups():
    contract = DEFAULT_CHURN_CONTRACT

    assert "tenure" in contract.numeric_columns
    assert "MonthlyCharges" in contract.numeric_columns
    assert "Contract" in contract.categorical_columns
    assert "PaymentMethod" in contract.categorical_columns