import numpy as np
import pandas as pd
from scipy.stats import rankdata


def calculate_roc(series: pd.Series, window: int) -> pd.Series:
    if series.empty or len(series) < window + 1:
        return pd.Series(index=series.index, dtype=float)
    return series.pct_change(periods=window)


def calculate_percentile_rank(
    series: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    if min_periods is None:
        min_periods = window // 2
    if series.empty or len(series) < min_periods:
        return pd.Series(index=series.index, dtype=float)
    return series.rolling(window=window, min_periods=min_periods).apply(
        lambda x: rankdata(x)[-1] / len(x) if len(x) > 0 and not np.isnan(x[-1]) else np.nan,
        raw=True,
    )


def calculate_sma(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    if series.empty or len(series) < min_periods:
        return pd.Series(index=series.index, dtype=float)
    return series.rolling(window=window, min_periods=min_periods).mean()


def calculate_value_vs_sma_signal(value_series: pd.Series, sma_series: pd.Series) -> pd.Series:
    signal = pd.Series(np.nan, index=value_series.index)
    if not value_series.empty and not sma_series.empty:
        signal = np.sign(value_series - sma_series)
    return signal.astype("Int64")


def calculate_sma_crossover_signal(sma_short: pd.Series, sma_long: pd.Series) -> pd.Series:
    signal = pd.Series(np.nan, index=sma_short.index)
    if not sma_short.empty and not sma_long.empty:
        signal = np.sign(sma_short - sma_long)
    return signal.astype("Int64")
