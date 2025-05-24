import logging

import numpy as np
import pandas as pd

from ..utils.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class OilPriceCalculator:
    def __init__(self, db_manager: DatabaseManager, oil_series_id: str = "DCOILWTICO"):
        self.db_manager = db_manager
        self.oil_series_id = oil_series_id
        logger.info(f"OilPriceCalculator initialized for series_id: {oil_series_id}")

    def _fetch_raw_oil_data(self) -> pd.DataFrame:
        query = f"""
            SELECT reference_date, value
            FROM oil_raw_fred
            WHERE series_id = '{self.oil_series_id}'
            ORDER BY reference_date ASC;
        """
        session = self.db_manager.get_session()
        try:
            df = pd.read_sql_query(query, session.bind, parse_dates=["reference_date"])
            df.rename(columns={"reference_date": "date", "value": "price"}, inplace=True)
            df.set_index("date", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching Oil raw data for {self.oil_series_id}: {e}")
            return pd.DataFrame()
        finally:
            session.close()

    def calculate_signals(self) -> pd.DataFrame | None:
        oil_df = self._fetch_raw_oil_data()
        if oil_df.empty:
            logger.warning(
                f"No raw Oil data found for {self.oil_series_id}. Cannot calculate signals."
            )
            return None

        signals_df = pd.DataFrame(index=oil_df.index)
        signals_df["symbol"] = self.oil_series_id
        signals_df["price"] = oil_df["price"]

        # 1M % change (approx 21 trading days)
        signals_df["pct_change_1m"] = oil_df["price"].pct_change(periods=21)

        # 6M % change (momentum, approx 126 trading days)
        signals_df["pct_change_6m"] = oil_df["price"].pct_change(periods=126)

        # Z-score of 3M returns (approx 63 trading days)
        return_3m = oil_df["price"].pct_change(periods=63)
        rolling_mean_3m_ret = return_3m.rolling(window=63, min_periods=21).mean()
        rolling_std_3m_ret = return_3m.rolling(window=63, min_periods=21).std()
        signals_df["z_score_return_3m"] = (return_3m - rolling_mean_3m_ret) / rolling_std_3m_ret

        # 30d rolling volatility (annualized)
        daily_returns = oil_df["price"].pct_change()
        signals_df["volatility_30d"] = daily_returns.rolling(
            window=30, min_periods=15
        ).std() * np.sqrt(252)

        signals_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        signals_df.dropna(
            how="all",
            subset=["pct_change_1m", "pct_change_6m", "z_score_return_3m", "volatility_30d"],
            inplace=True,
        )

        signals_df.reset_index(inplace=True)
        signals_df["date"] = signals_df["date"].dt.strftime("%Y-%m-%d")

        logger.info(
            f"Calculated Oil Price signals for {self.oil_series_id}. Shape: {signals_df.shape}"
        )
        return signals_df
