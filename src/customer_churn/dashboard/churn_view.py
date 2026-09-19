from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data/processed/telco_customer_churn_clean.csv"
BI_PATH = ROOT / "data/processed/churn_customer_intelligence_bi.csv"
STATS_PATH = ROOT / "reports/churn_statistical_analysis.csv"


@st.cache_data
def load_churn_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(DATA_PATH)
    bi = pd.read_csv(BI_PATH)
    statistics = pd.read_csv(STATS_PATH)

    return data, bi, statistics


def render_churn_dashboard() -> None:
    data, bi, statistics = load_churn_data()

    st.caption("Resultados de referência importados da plataforma original. "
               "Risco e retenção referem-se aos 1.409 clientes da amostra de teste.")
    customers = len(data)
    churned = int((data["Churn"] == "Yes").sum())
    churn_rate = churned / customers * 100

    high_risk = int((bi["risk_level"] == "HIGH").sum())
    urgent = int((bi["retention_priority"] == "URGENT").sum())

    st.markdown(
        """
        <div class="section-title">Customer Churn Intelligence</div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Data Science, Machine Learning, Explainability, "
        "Risk Segmentation and Retention Intelligence"
    )

    columns = st.columns(5)

    metrics = (
        ("Customers", f"{customers:,}"),
        ("Churned", f"{churned:,}"),
        ("Churn Rate", f"{churn_rate:.2f}%"),
        ("High Risk", f"{high_risk:,}"),
        ("Urgent", f"{urgent:,}"),
    )

    for column, (label, value) in zip(columns, metrics, strict=True):
        with column:
            st.metric(label, value)

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True,
    )

    model_metrics = pd.read_csv(
        ROOT / "dashboards/powerbi/data/churn_model_comparison.csv"
    ).rename(columns={
        "model": "Model", "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
    })

    st.dataframe(
        model_metrics,
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Model Comparison")

    model_chart = model_metrics.set_index("Model")[
        ["Precision", "Recall", "F1", "ROC-AUC"]
    ]

    st.bar_chart(model_chart)

    st.caption(
        "Baseline evaluation on the same stratified held-out test split "
        "(1,409 customers) using threshold 0.50."
    )
    st.markdown(
        '<div class="section-title">SHAP Explainability</div>',
        unsafe_allow_html=True,
    )

    shap_importance = pd.read_csv(
        ROOT / "dashboards/powerbi/data/churn_shap_importance.csv"
    ).rename(columns={
        "feature": "Feature", "mean_abs_shap": "Mean Absolute SHAP",
    })

    shap_columns = st.columns([2, 3])

    with shap_columns[0]:
        st.dataframe(
            shap_importance,
            width="stretch",
            hide_index=True,
        )

    with shap_columns[1]:
        st.bar_chart(
            shap_importance.set_index("Feature")[
                "Mean Absolute SHAP"
            ]
        )

    st.caption(
        "Mean absolute SHAP measures the magnitude of each feature's "
        "contribution to XGBoost predictions. It does not imply causality."
    )
    st.markdown(
        '<div class="section-title">Risk Intelligence</div>',
        unsafe_allow_html=True,
    )

    risk_summary = (
        bi.groupby(
            ["risk_level", "retention_priority"],
            as_index=False,
        )
        .agg(
            customers=("customer_id", "count"),
            avg_probability=("churn_probability", "mean"),
        )
    )

    risk_summary["avg_probability"] = (
        risk_summary["avg_probability"] * 100
    ).round(2)

    risk_summary = risk_summary.rename(
        columns={
            "risk_level": "Risk Level",
            "retention_priority": "Retention Priority",
            "customers": "Customers",
            "avg_probability": "Avg Probability (%)",
        }
    )

    st.dataframe(
        risk_summary,
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        '<div class="section-title">Business Analytics</div>',
        unsafe_allow_html=True,
    )

    chart_columns = st.columns(2)

    contract_summary = (
        data.groupby("Contract", as_index=False)
        .agg(
            Customers=("customerID", "count"),
            Churned=("Churn", lambda values: (values == "Yes").sum()),
        )
    )
    contract_summary["Churn Rate (%)"] = (
        100 * contract_summary["Churned"] / contract_summary["Customers"]
    ).round(2)
    contract_summary = contract_summary.sort_values(
        "Churn Rate (%)",
        ascending=False,
    )

    with chart_columns[0]:
        st.markdown("#### Churn Rate by Contract")
        st.bar_chart(
            contract_summary.set_index("Contract")["Churn Rate (%)"]
        )

    risk_order = ["LOW", "MEDIUM", "HIGH"]

    risk_distribution = (
        bi["risk_level"]
        .value_counts()
        .reindex(risk_order)
        .fillna(0)
        .astype(int)
    )

    with chart_columns[1]:
        st.markdown("#### Customer Risk Distribution")
        st.bar_chart(risk_distribution)

    st.caption(
        "Risk levels are operational segmentation thresholds. "
        "They are not causal classifications."
    )
    st.markdown(
        '<div class="section-title">Statistical Analysis</div>',
        unsafe_allow_html=True,
    )

    top_effects = (
        statistics.sort_values(
            "absolute_effect_size",
            ascending=False,
        )
        .head(10)
        .loc[
            :,
            [
                "feature",
                "test",
                "p_value",
                "effect_size",
                "effect_metric",
            ],
        ]
    )

    top_effects = top_effects.rename(
        columns={
            "feature": "Feature",
            "test": "Test",
            "p_value": "P-value",
            "effect_size": "Effect Size",
            "effect_metric": "Effect Metric",
        }
    )

    top_effects["P-value"] = top_effects["P-value"].map(
        lambda value: f"{value:.3e}"
    )
    top_effects["Effect Size"] = top_effects["Effect Size"].round(4)

    st.dataframe(
        top_effects,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Fila de retenção")
    levels = st.multiselect("Filtrar risco", ["LOW", "MEDIUM", "HIGH"], default=["HIGH"])
    selected = bi[bi["risk_level"].isin(levels)].sort_values(
        "churn_probability", ascending=False
    )
    st.dataframe(selected, hide_index=True)
    st.download_button("Baixar clientes filtrados", selected.to_csv(index=False),
                       "clientes_retencao.csv", "text/csv")

    st.markdown(
        '<div class="section-title">Data Science Stack</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="stack-box">
        <div class="stack-title">
        Customer Churn · Data Science & ML Intelligence
        </div>
        <b>Data:</b> Pandas · SQL · SQLite<br>
        <b>Statistics:</b> Chi-Square · Cramér's V · Mann-Whitney U · Rank-Biserial<br>
        <b>Machine Learning:</b> Logistic Regression · Random Forest · XGBoost<br>
        <b>Explainability:</b> SHAP<br>
        <b>Intelligence:</b> Risk Segmentation · Retention Prioritization<br>
        <b>BI:</b> Streamlit · Power BI Ready Dataset
        </div>
        """,
        unsafe_allow_html=True,
    )
