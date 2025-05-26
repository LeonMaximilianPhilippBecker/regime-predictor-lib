import logging

import pandas as pd

from regime_predictor_lib.utils.financial_calculations import (
    calculate_percentile_rank,
    calculate_roc,
    calculate_sma,
)

logger = logging.getLogger(__name__)


class SentimentConfidenceCalculator:
    def __init__(
        self,
        roc_periods_months: list[int] | None = None,
        percentile_windows_months: list[int] | None = None,
        sma_windows_months: list[int] | None = None,
    ):
        self.roc_periods = roc_periods_months or [1, 3, 6]
        self.percentile_windows = percentile_windows_months or [12, 24]  # 1Y, 2Y
        self.sma_windows = sma_windows_months or [3, 6]
        logger.info("SentimentConfidenceCalculator initialized.")

    def calculate_signals(self, raw_df: pd.DataFrame) -> pd.DataFrame | None:
        if raw_df is None or raw_df.empty or "date" not in raw_df.columns:
            logger.warning("Raw SMCI/DMCI DataFrame is empty or missing 'date' column.")
            return None

        df = raw_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        signals_df = pd.DataFrame(index=df.index)

        value_cols = ["smci_value", "dmci_value", "smci_pct_dmci_ratio"]
        for col in value_cols:
            if col in df.columns:
                signals_df[col] = df[col]

                for period in self.roc_periods:
                    signals_df[f"{col}_roc_{period}m"] = calculate_roc(df[col], period)

                for window in self.sma_windows:
                    signals_df[f"{col}_sma_{window}m"] = calculate_sma(
                        df[col], window, min_periods=window
                    )

                for window in self.percentile_windows:
                    min_p_perc = max(1, window // 2)
                    signals_df[f"{col}_percentile_{window // 12}y"] = calculate_percentile_rank(
                        df[col], window, min_periods=min_p_perc
                    )
            else:
                logger.warning(
                    f"Column {col} not found in input DataFrame for SMCI/DMCI calculator."
                )

        signals_df.reset_index(inplace=True)
        signals_df["date"] = signals_df["date"].dt.strftime("%Y-%m-%d")
        logger.info(f"Calculated SMCI/DMCI signals. Shape: {signals_df.shape}")
        return signals_df
