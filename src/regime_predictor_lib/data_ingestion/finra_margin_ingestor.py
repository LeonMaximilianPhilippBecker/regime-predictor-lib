import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FinraMarginIngestor:
    def __init__(self, csv_filepath: Path | str):
        self.csv_filepath = Path(csv_filepath)
        logger.info(f"FinraMarginIngestor initialized for file: {self.csv_filepath}")

    def _clean_numeric_value(self, value):
        if pd.isna(value) or value == "":
            return np.nan
        if isinstance(value, (int, float)):
            return int(value)
        try:
            return int(str(value).replace(",", "").strip())
        except ValueError:
            logger.warning(f"Could not convert '{value}' to numeric integer.")
            return np.nan

    def load_and_process_data(self) -> pd.DataFrame | None:
        if not self.csv_filepath.exists():
            logger.error(f"FINRA margin stats CSV file not found: {self.csv_filepath}")
            return None

        try:
            df = pd.read_csv(self.csv_filepath)
            logger.info(f"Successfully loaded FINRA margin stats data from {self.csv_filepath}")
        except Exception as e:
            logger.error(f"Error loading FINRA margin stats CSV {self.csv_filepath}: {e}")
            return None

        df.rename(
            columns={
                "Year-Month": "year_month_orig",
                "Debit Balances in Customers' Securities Margin "
                "Accounts": "debit_balances_margin_accounts",
                "Free Credit Balances in Customers' Cash Accounts": "free_credit_cash_accounts",
                "Free Credit Balances in Customers' Securities Margin "
                "Accounts": "free_credit_margin_accounts",
            },
            inplace=True,
        )

        try:
            df["reference_date"] = pd.to_datetime(
                df["year_month_orig"], format="%Y-%m"
            ) + pd.offsets.MonthEnd(0)
        except Exception as e:
            logger.error(f"Error converting 'Year-Month' column to reference_date: {e}")
            return None

        df["release_date"] = df["reference_date"] + pd.offsets.MonthEnd(1)

        numeric_cols = [
            "debit_balances_margin_accounts",
            "free_credit_cash_accounts",
            "free_credit_margin_accounts",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(self._clean_numeric_value)
            else:
                logger.warning(f"Numeric column '{col}' not found in DataFrame.")
                df[col] = np.nan

        final_cols = [
            "reference_date",
            "release_date",
            "debit_balances_margin_accounts",
            "free_credit_cash_accounts",
            "free_credit_margin_accounts",
        ]
        for col in final_cols:
            if col not in df.columns:
                df[col] = np.nan

        df = df[final_cols]

        df["reference_date"] = df["reference_date"].dt.strftime("%Y-%m-%d")
        df["release_date"] = df["release_date"].dt.strftime("%Y-%m-%d")

        df.dropna(subset=["reference_date", "release_date"], inplace=True)

        logger.info(f"Processed FINRA margin stats data. Shape: {df.shape}")
        return df
