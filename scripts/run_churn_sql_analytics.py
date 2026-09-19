from pathlib import Path

import pandas as pd

from customer_churn.analytics.sql_analytics import (
    ChurnSQLAnalytics,
)

DATA_PATH = Path("data/processed/telco_customer_churn_clean.csv")
BI_PATH = Path("data/processed/churn_customer_intelligence_bi.csv")
REPORT_DIR = Path("reports")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH)

    analytics = ChurnSQLAnalytics()

    try:
        analytics.load_dataframe(data)

        by_contract = analytics.query(
            """
            SELECT
                Contract,
                COUNT(*) AS customers,
                SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
                ROUND(
                    100.0 * SUM(
                        CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END
                    ) / COUNT(*),
                    2
                ) AS churn_rate_pct
            FROM customer_churn
            GROUP BY Contract
            ORDER BY churn_rate_pct DESC
            """
        )

        by_internet = analytics.query(
            """
            SELECT
                InternetService,
                COUNT(*) AS customers,
                SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
                ROUND(
                    100.0 * SUM(
                        CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END
                    ) / COUNT(*),
                    2
                ) AS churn_rate_pct
            FROM customer_churn
            GROUP BY InternetService
            ORDER BY churn_rate_pct DESC
            """
        )

        by_payment = analytics.query(
            """
            SELECT
                PaymentMethod,
                COUNT(*) AS customers,
                SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
                ROUND(
                    100.0 * SUM(
                        CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END
                    ) / COUNT(*),
                    2
                ) AS churn_rate_pct
            FROM customer_churn
            GROUP BY PaymentMethod
            ORDER BY churn_rate_pct DESC
            """
        )

        by_tenure = analytics.query(
            """
            SELECT
                CASE
                    WHEN tenure <= 12 THEN '0-12'
                    WHEN tenure <= 24 THEN '13-24'
                    WHEN tenure <= 48 THEN '25-48'
                    WHEN tenure <= 60 THEN '49-60'
                    ELSE '61+'
                END AS tenure_segment,
                COUNT(*) AS customers,
                SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
                ROUND(
                    100.0 * SUM(
                        CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END
                    ) / COUNT(*),
                    2
                ) AS churn_rate_pct
            FROM customer_churn
            GROUP BY tenure_segment
            ORDER BY churn_rate_pct DESC
            """
        )

        by_contract.to_csv(
            REPORT_DIR / "churn_sql_by_contract.csv",
            index=False,
        )
        by_internet.to_csv(
            REPORT_DIR / "churn_sql_by_internet_service.csv",
            index=False,
        )
        by_payment.to_csv(
            REPORT_DIR / "churn_sql_by_payment_method.csv",
            index=False,
        )
        by_tenure.to_csv(
            REPORT_DIR / "churn_sql_by_tenure.csv",
            index=False,
        )

    finally:
        analytics.close()

    bi = pd.read_csv(BI_PATH)

    intelligence = ChurnSQLAnalytics()

    try:
        intelligence.load_dataframe(
            bi,
            table_name="churn_intelligence",
        )

        by_risk = intelligence.query(
            """
            SELECT
                risk_level,
                retention_priority,
                COUNT(*) AS customers,
                ROUND(AVG(churn_probability), 4) AS avg_probability,
                ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
                ROUND(
                    100.0 * SUM(
                        CASE
                            WHEN actual_churn = 'Yes' THEN 1
                            ELSE 0
                        END
                    ) / COUNT(*),
                    2
                ) AS observed_churn_pct
            FROM churn_intelligence
            GROUP BY risk_level, retention_priority
            ORDER BY avg_probability DESC
            """
        )

        by_risk.to_csv(
            REPORT_DIR / "churn_sql_by_risk.csv",
            index=False,
        )

    finally:
        intelligence.close()

    print("CUSTOMERS=", len(data))
    print("BI_CUSTOMERS=", len(bi))
    print("SQL_REPORTS=5")
    print("CONTRACT_ROWS=", len(by_contract))
    print("INTERNET_ROWS=", len(by_internet))
    print("PAYMENT_ROWS=", len(by_payment))
    print("TENURE_ROWS=", len(by_tenure))
    print("RISK_ROWS=", len(by_risk))

    print()
    print("RISK_ANALYTICS")
    print(by_risk.to_string(index=False))

    print()
    print("CHURN_SQL_BUSINESS_ANALYTICS=PASS")


if __name__ == "__main__":
    main()
