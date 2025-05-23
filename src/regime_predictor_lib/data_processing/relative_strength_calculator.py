import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RelativeStrengthCalculator:
    def __init__(
        self,
        data_a: pd.DataFrame,
        data_b: pd.DataFrame,
        config: dict | None = None,
    ):
        self.config = config or {
            "log_diff_deltas": [1, 5, 20],
            "return_period_for_spread": 1,
            "z_score_rolling_windows": [20, 60],
        }

        if data_a is None or data_a.empty or data_b is None or data_b.empty:
            logger.warning("Input data_a or data_b is None or empty. Cannot proceed.")
            self.merged_df = pd.DataFrame()
            return

        self.data_a = self._prepare_data(data_a.copy(), "a")
        self.data_b = self._prepare_data(data_b.copy(), "b")

        if self.data_a.empty or self.data_b.empty:
            logger.warning("Prepared data_a or data_b is empty after processing. Cannot proceed.")
            self.merged_df = pd.DataFrame()
            return

        self.merged_df = pd.merge(
            self.data_a, self.data_b, on="date", how="inner", suffixes=("_a", "_b")
        )
        if self.merged_df.empty:
            logger.warning("DataFrames for series A and B have no overlapping dates.")
            return

        self.merged_df.sort_index(inplace=True)
        logger.info(
            f"RelativeStrengthCalculator initialized. Merged data shape: {self.merged_df.shape}"
        )

    def _prepare_data(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        if "date" not in df.columns:
            logger.error(f"DataFrame {name} missing 'date' column.")
            return pd.DataFrame()
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            logger.error(f"Could not convert 'date' column to datetime for df {name}: {e}")
            return pd.DataFrame()

        if "close" not in df.columns:
            logger.error(f"DataFrame {name} missing 'close' column.")
            return pd.DataFrame()

        df = df[["date", "close"]].set_index("date")
        df.columns = [f"close_{name}"]
        df = df[~df.index.duplicated(keep="first")]
        return df

    def calculate_metrics(self) -> pd.DataFrame:
        if self.merged_df.empty:
            logger.warning("Merged data is empty, cannot calculate metrics.")
            return pd.DataFrame()

        results_df = pd.DataFrame(index=self.merged_df.index)

        results_df["rs_ratio"] = self.merged_df["close_a"] / self.merged_df["close_b"]

        for delta in self.config["log_diff_deltas"]:
            col_name = f"log_diff_{delta}d"
            log_rs_ratio = np.log(results_df["rs_ratio"])
            results_df[col_name] = log_rs_ratio - log_rs_ratio.shift(delta)

        ret_period = self.config["return_period_for_spread"]
        ret_a = self.merged_df["close_a"].pct_change(periods=ret_period)
        ret_b = self.merged_df["close_b"].pct_change(periods=ret_period)
        spread_col_name = f"return_spread_{ret_period}d"
        results_df[spread_col_name] = ret_a - ret_b

        for window in self.config["z_score_rolling_windows"]:
            z_col_name = f"z_score_spread_{ret_period}d_window{window}d"
            spread_series = results_df[spread_col_name]
            mean_spread = spread_series.rolling(window=window, min_periods=window // 2).mean()
            std_spread = spread_series.rolling(window=window, min_periods=window // 2).std()
            results_df[z_col_name] = (spread_series - mean_spread) / std_spread
            results_df[z_col_name].replace([np.inf, -np.inf], np.nan, inplace=True)

        results_df.reset_index(inplace=True)
        results_df["date"] = pd.to_datetime(results_df["date"]).dt.strftime("%Y-%m-%d")

        logger.info(f"Calculated relative strength metrics. Shape: {results_df.shape}")
        return results_df
