import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


class GexCalculator:
    def __init__(
        self,
        percentile_window_days: int = 252,
        z_score_window_days: int = 63,
    ):
        self.percentile_window_days = percentile_window_days
        self.z_score_window_days = z_score_window_days
        self.min_periods_pct = percentile_window_days // 2
        self.min_periods_z = z_score_window_days // 3
        logger.info(
            "GexCalculator initialized with percentile "
            f"window: {self.percentile_window_days} days, "
            f"Z-score window: {self.z_score_window_days} days."
        )

    def calculate_signals(self, gex_df: pd.DataFrame) -> pd.DataFrame | None:
        if gex_df is None or gex_df.empty:
            logger.warning("Input GEX DataFrame is empty or None. Cannot calculate signals.")
            return None

        if "gex_value" not in gex_df.columns:
            logger.error("'gex_value' column not found in input DataFrame.")
            return None

        if not isinstance(gex_df.index, pd.DatetimeIndex):
            logger.error("Input DataFrame must have a DatetimeIndex.")
            return None

        signals_df = pd.DataFrame(index=gex_df.index)
        signals_df["gex_value"] = gex_df["gex_value"]

        signals_df["gex_percentile_rank_1y"] = (
            gex_df["gex_value"]
            .rolling(window=self.percentile_window_days, min_periods=self.min_periods_pct)
            .apply(lambda x: rankdata(x)[-1] / len(x) if len(x) > 0 else np.nan, raw=False)
        )

        rolling_mean_gex = (
            gex_df["gex_value"]
            .rolling(window=self.z_score_window_days, min_periods=self.min_periods_z)
            .mean()
        )
        rolling_std_gex = (
            gex_df["gex_value"]
            .rolling(window=self.z_score_window_days, min_periods=self.min_periods_z)
            .std()
        )

        signals_df["gex_z_score_3m"] = (gex_df["gex_value"] - rolling_mean_gex) / rolling_std_gex
        signals_df["gex_z_score_3m"].replace([np.inf, -np.inf], np.nan, inplace=True)

        signals_df.reset_index(inplace=True)
        signals_df["date"] = signals_df["date"].dt.strftime("%Y-%m-%d")

        signals_df.dropna(
            subset=["gex_percentile_rank_1y", "gex_z_score_3m"],
            how="all",
            inplace=True,
        )

        if signals_df.empty:
            logger.warning("No GEX signals calculated after processing and NaN drop.")
            return None

        logger.info(f"Calculated GEX signals. Shape: {signals_df.shape}")
        return signals_df[["date", "gex_value", "gex_percentile_rank_1y", "gex_z_score_3m"]]
