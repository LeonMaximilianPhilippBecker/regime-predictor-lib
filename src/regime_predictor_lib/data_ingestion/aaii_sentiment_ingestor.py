import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AaiiSentimentIngestor:
    def __init__(self, csv_filepath: Path | str):
        self.csv_filepath = Path(csv_filepath)
        logger.info(f"AaiiSentimentIngestor initialized for file: {self.csv_filepath}")

    def _clean_percentage_value(self, value):
        if pd.isna(value) or value == "":
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).replace("%", "").strip())
        except ValueError:
            logger.warning(f"Could not convert '{value}' to percentage float.")
            return np.nan

    def load_and_process_data(self) -> pd.DataFrame | None:
        if not self.csv_filepath.exists():
            logger.error(f"AAII sentiment CSV file not found: {self.csv_filepath}")
            return None

        try:
            df = pd.read_csv(self.csv_filepath)
            logger.info(f"Successfully loaded AAII sentiment data from {self.csv_filepath}")
        except Exception as e:
            logger.error(f"Error loading AAII sentiment CSV {self.csv_filepath}: {e}")
            return None

        df.rename(
            columns={
                "Date": "reference_date_orig",
                "Bullish": "bullish_pct",
                "Neutral": "neutral_pct",
                "Bearish": "bearish_pct",
                "Total": "total_pct",
                "Bullish 8 Week MA": "bullish_8wk_ma_pct",
                "Bull-Bear Spread": "bull_bear_spread_pct",
            },
            inplace=True,
        )

        try:
            df["reference_date"] = pd.to_datetime(df["reference_date_orig"])
        except Exception as e:
            logger.error(f"Error converting 'Date' column to datetime: {e}")
            return None

        df["release_date"] = df["reference_date"] + pd.offsets.MonthEnd(1)

        pct_cols = [
            "bullish_pct",
            "neutral_pct",
            "bearish_pct",
            "total_pct",
            "bullish_8wk_ma_pct",
            "bull_bear_spread_pct",
        ]
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].apply(self._clean_percentage_value)
            else:
                logger.warning(f"Percentage column '{col}' not found in DataFrame.")
                df[col] = np.nan

        final_cols = [
            "reference_date",
            "release_date",
            "bullish_pct",
            "neutral_pct",
            "bearish_pct",
            "total_pct",
            "bullish_8wk_ma_pct",
            "bull_bear_spread_pct",
        ]
        for col in final_cols:
            if col not in df.columns:
                df[col] = np.nan

        df = df[final_cols]

        df["reference_date"] = df["reference_date"].dt.strftime("%Y-%m-%d")
        df["release_date"] = df["release_date"].dt.strftime("%Y-%m-%d")

        df.dropna(subset=["reference_date", "release_date"], inplace=True)

        logger.info(f"Processed AAII sentiment data. Shape: {df.shape}")
        return df
