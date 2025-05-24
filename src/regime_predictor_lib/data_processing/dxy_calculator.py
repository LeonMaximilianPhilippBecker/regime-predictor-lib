import logging

import numpy as np
import pandas as pd
from scipy.stats import linregress, rankdata

from ..utils.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class DxyCalculator:
    def __init__(self, db_manager: DatabaseManager, dxy_series_id: str = "DTWEXBGS"):
        self.db_manager = db_manager
        self.dxy_series_id = dxy_series_id
        logger.info(f"DxyCalculator initialized for series_id: {dxy_series_id}")

    def _fetch_raw_dxy_data(self) -> pd.DataFrame:
        query = f"""
            SELECT reference_date, value
            FROM dxy_raw_fred
            WHERE series_id = '{self.dxy_series_id}'
            ORDER BY reference_date ASC;
        """
        session = self.db_manager.get_session()
        try:
            df = pd.read_sql_query(query, session.bind, parse_dates=["reference_date"])
            df.rename(columns={"reference_date": "date", "value": "dxy_value"}, inplace=True)
            df.set_index("date", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching DXY raw data for {self.dxy_series_id}: {e}")
            return pd.DataFrame()
        finally:
            session.close()

    def calculate_signals(self) -> pd.DataFrame | None:
        dxy_df = self._fetch_raw_dxy_data()
        if dxy_df.empty:
            logger.warning(
                f"No raw DXY data found for {self.dxy_series_id}. Cannot calculate signals."
            )
            return None

        signals_df = pd.DataFrame(index=dxy_df.index)
        signals_df["dxy_value"] = dxy_df["dxy_value"]
        signals_df["series_id"] = self.dxy_series_id

        # 21-day % change (momentum)
        signals_df["pct_change_21d"] = dxy_df["dxy_value"].pct_change(periods=21)

        # Z-score of 3M return (vol-adjusted momentum)
        return_3m = dxy_df["dxy_value"].pct_change(periods=63)
        rolling_mean_3m_ret = return_3m.rolling(window=63, min_periods=21).mean()
        rolling_std_3m_ret = return_3m.rolling(window=63, min_periods=21).std()
        signals_df["z_score_return_3m"] = (return_3m - rolling_mean_3m_ret) / rolling_std_3m_ret

        # 1-year percentile rank
        signals_df["percentile_rank_1y"] = (
            dxy_df["dxy_value"]
            .rolling(window=252, min_periods=126)
            .apply(lambda x: rankdata(x)[-1] / len(x) if len(x) > 0 else np.nan, raw=False)
        )

        daily_returns = dxy_df["dxy_value"].pct_change()
        signals_df["volatility_30d"] = daily_returns.rolling(
            window=30, min_periods=15
        ).std() * np.sqrt(252)

        def get_slope(y_series):
            if len(y_series) < 2 or y_series.isnull().all():
                return np.nan
            y_series_clean = y_series.dropna()
            if len(y_series_clean) < 2:
                return np.nan
            x_series = np.arange(len(y_series_clean))
            slope, _, _, _, _ = linregress(x_series, y_series_clean)
            return slope

        signals_df["slope_30d_regression"] = (
            dxy_df["dxy_value"].rolling(window=30, min_periods=15).apply(get_slope, raw=False)
        )

        signals_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        signals_df.dropna(
            how="all",
            subset=[
                "pct_change_21d",
                "z_score_return_3m",
                "percentile_rank_1y",
                "volatility_30d",
                "slope_30d_regression",
            ],
            inplace=True,
        )

        signals_df.reset_index(inplace=True)
        signals_df["date"] = signals_df["date"].dt.strftime("%Y-%m-%d")

        logger.info(f"Calculated DXY signals for {self.dxy_series_id}. Shape: {signals_df.shape}")
        return signals_df
