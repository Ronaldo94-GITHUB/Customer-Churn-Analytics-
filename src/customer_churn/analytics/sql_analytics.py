from __future__ import annotations

import sqlite3

import pandas as pd


class ChurnSQLAnalytics:
    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:")

    def load_dataframe(
        self,
        data: pd.DataFrame,
        table_name: str = "customer_churn",
    ) -> None:
        if data.empty:
            raise ValueError("DataFrame cannot be empty.")

        data.to_sql(
            table_name,
            self.connection,
            if_exists="replace",
            index=False,
        )

    def query(self, sql: str) -> pd.DataFrame:
        if not sql.strip():
            raise ValueError("SQL query cannot be empty.")

        return pd.read_sql_query(sql, self.connection)

    def close(self) -> None:
        self.connection.close()