import pandas as pd

from customer_churn.features.builder import (
    ChurnFeatureBuilder,
)
from customer_churn.features.splitter import (
    ChurnDatasetSplitter,
)


def _dataset():
    rows = 100

    dataframe = pd.DataFrame(
        {
            "customerID": [f"C{i:03d}" for i in range(rows)],
            "tenure": list(range(rows)),
            "MonthlyCharges": [50.0] * rows,
            "Churn": ["Yes"] * 25 + ["No"] * 75,
        }
    )

    return ChurnFeatureBuilder().build(dataframe)


def test_split_preserves_all_rows():
    result = ChurnDatasetSplitter().split(_dataset())

    assert len(result.x_train) + len(result.x_test) == 100


def test_split_uses_expected_test_size():
    result = ChurnDatasetSplitter().split(_dataset())

    assert len(result.x_train) == 80
    assert len(result.x_test) == 20


def test_split_is_stratified():
    result = ChurnDatasetSplitter().split(_dataset())

    original_rate = _dataset().target.mean()

    assert abs(result.y_train.mean() - original_rate) < 0.02
    assert abs(result.y_test.mean() - original_rate) < 0.02


def test_customer_ids_remain_aligned():
    result = ChurnDatasetSplitter().split(_dataset())

    assert result.x_train.index.equals(
        result.customer_ids_train.index
    )

    assert result.x_test.index.equals(
        result.customer_ids_test.index
    )


def test_split_is_reproducible():
    dataset = _dataset()

    first = ChurnDatasetSplitter().split(dataset)
    second = ChurnDatasetSplitter().split(dataset)

    assert first.x_train.index.tolist() == second.x_train.index.tolist()
    assert first.x_test.index.tolist() == second.x_test.index.tolist()