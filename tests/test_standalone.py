from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from customer_churn.data.cleaner import ChurnDataCleaner
from customer_churn.data.loader import ChurnDatasetLoader

ROOT = Path(__file__).resolve().parents[1]


def test_copied_dataset_matches_cleaning():
    raw = ChurnDatasetLoader().load(ROOT / "data/raw/telco_customer_churn.csv")
    expected = pd.read_csv(ROOT / "data/processed/telco_customer_churn_clean.csv")
    actual = ChurnDataCleaner().clean(raw)
    assert len(raw) == 7043
    pd.testing.assert_frame_equal(actual, expected)


def test_dashboard_and_retention_filter():
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
    assert not app.exception
    assert app.metric[0].value == "7,043"
    assert len(app.dataframe[-1].value) == 199
    app.multiselect[0].set_value(["LOW"]).run(timeout=60)
    assert not app.exception
    assert set(app.dataframe[-1].value["risk_level"]) == {"LOW"}
    assert len(app.dataframe[-1].value) == 874
