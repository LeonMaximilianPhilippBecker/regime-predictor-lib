import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class CreditSpreadCalculator:
    def __init__(
        self,
        high_yield_csv_path: Path | str,
        investment_grade_csv_path: Path | str,
        high_yield_series_id: str,
        investment_grade_series_id: str,
    ):
        self.high_yield_csv_path = Path(high_yield_csv_path)
        self.investment_grade_csv_path = Path(investment_grade_csv_path)
        self.high_yield_series_id = high_yield_series_id
        self.investment_grade_series_id = investment_grade_series_id
        logger.info(
            "CreditSpreadCalculator initialized with: \n"
            f"  High Yield CSV: {self.high_yield_csv_path}\n"
            f"  Inv. Grade CSV: {self.investment_grade_csv_path}"
        )

    def _load_and_clean_vintage_series(
        self, csv_path: Path, series_name: str
    ) -> pd.DataFrame | None:
        if not csv_path.exists():
            logger.error(f"Vintage data CSV not found: {csv_path}")
            return None
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Error reading CSV {csv_path}: {e}")
            return None

        if df.empty:
            logger.warning(f"Vintage data CSV is empty: {csv_path}")
            return pd.DataFrame(columns=[f"value_{series_name}", f"release_date_{series_name}"])

        required_cols = ["reference_date", "release_date", "value"]
        if not all(col in df.columns for col in required_cols):
            logger.error(
                f"CSV {csv_path} missing one or more required columns: {required_cols}. "
                f"Found: {df.columns.tolist()}"
            )
            return None

        df["reference_date"] = pd.to_datetime(df["reference_date"], errors="coerce")
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        df.dropna(subset=["reference_date", "release_date", "value"], inplace=True)

        df.sort_values(by=["reference_date", "release_date"], ascending=[True, True], inplace=True)

        cleaned_df = df.groupby("reference_date").first().reset_index()

        cleaned_df = cleaned_df[["reference_date", "value", "release_date"]]
        cleaned_df.rename(
            columns={
                "value": f"value_{series_name}",
                "release_date": f"release_date_{series_name}",
            },
            inplace=True,
        )
        cleaned_df.set_index("reference_date", inplace=True)
        logger.info(f"Cleaned data for {series_name} from {csv_path}. Shape: {cleaned_df.shape}")
        return cleaned_df

    def calculate_spreads(self) -> pd.DataFrame | None:
        logger.info("Starting credit spread calculation...")

        hy_df = self._load_and_clean_vintage_series(self.high_yield_csv_path, "hy")
        ig_df = self._load_and_clean_vintage_series(self.investment_grade_csv_path, "ig")

        if hy_df is None or ig_df is None:
            logger.error(
                "Failed to load or clean one or both vintage series. Cannot calculate spreads."
            )
            return None

        if hy_df.empty and ig_df.empty:
            logger.warning("Both cleaned high-yield and investment-grade DataFrames are empty.")
            return pd.DataFrame()
        if hy_df.empty:
            logger.warning("Cleaned high-yield DataFrame is empty.")
        if ig_df.empty:
            logger.warning("Cleaned investment-grade DataFrame is empty.")

        merged_df = pd.merge(hy_df, ig_df, left_index=True, right_index=True, how="outer")

        if merged_df.empty:
            logger.warning("Merged DataFrame is empty after joining cleaned HY and IG series.")
            return pd.DataFrame()

        merged_df["spread_value"] = merged_df["value_hy"] - merged_df["value_ig"]

        merged_df["release_date"] = merged_df[["release_date_hy", "release_date_ig"]].max(axis=1)

        merged_df.dropna(subset=["spread_value", "release_date"], inplace=True)

        if merged_df.empty:
            logger.warning("No valid spread data after calculation and NaN handling.")
            return pd.DataFrame()

        result_df = merged_df[["release_date", "spread_value"]].reset_index()
        result_df.rename(columns={"reference_date": "reference_date"}, inplace=True)

        result_df["high_yield_series_id"] = self.high_yield_series_id
        result_df["investment_grade_series_id"] = self.investment_grade_series_id

        result_df["reference_date"] = result_df["reference_date"].dt.strftime("%Y-%m-%d")
        result_df["release_date"] = result_df["release_date"].dt.strftime("%Y-%m-%d")

        final_columns = [
            "reference_date",
            "release_date",
            "spread_value",
            "high_yield_series_id",
            "investment_grade_series_id",
        ]
        result_df = result_df[final_columns]

        logger.info(f"Credit spreads calculated. Shape: {result_df.shape}")
        return result_df
