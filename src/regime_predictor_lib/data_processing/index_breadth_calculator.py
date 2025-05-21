import logging
from pathlib import Path

import pandas as pd
import sqlalchemy
from tqdm import tqdm

logger = logging.getLogger(__name__)


class IndexBreadthCalculator:
    def __init__(
        self,
        raw_data_dir: Path,
        processed_data_dir: Path,
        db_path: Path,
        constituents_filename: str = "sp500_daily_constituents.csv",
        extended_prices_filename: str = "sp500_ticker_prices_extended.csv",
        db_table_name: str = "index_breadth_indicators",
    ):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.db_path = db_path
        self.input_csv_daily_constituents = self.raw_data_dir / constituents_filename
        self.input_csv_extended_prices = self.processed_data_dir / extended_prices_filename
        self.db_table_name = db_table_name
        self.db_engine = sqlalchemy.create_engine(f"sqlite:///{self.db_path}")

        logger.info(
            "IndexBreadthCalculator initialized. "
            f"Constituents: {self.input_csv_daily_constituents}, "
            f"Prices: {self.input_csv_extended_prices}, DB: {self.db_path}, "
            f"Table: {self.db_table_name}"
        )

    def _calculate_smas_and_ad(self, price_df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Calculating SMAs and advance/decline data for price_df...")
        price_df = price_df.sort_values(by=["ticker", "date"])

        price_df["sma50"] = price_df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(window=50, min_periods=50).mean()
        )
        price_df["sma200"] = price_df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(window=200, min_periods=200).mean()
        )

        price_df["prev_close"] = price_df.groupby("ticker")["close"].shift(1)
        price_df["price_change"] = price_df["close"] - price_df["prev_close"]
        price_df["advancing"] = price_df["price_change"] > 0
        price_df["declining"] = price_df["price_change"] < 0
        return price_df

    def calculate_and_store_breadth(self, if_exists_db: str = "replace"):
        if not self.input_csv_daily_constituents.exists():
            msg = f"Constituents file not found: {self.input_csv_daily_constituents}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        if not self.input_csv_extended_prices.exists():
            msg = f"Extended prices file not found: {self.input_csv_extended_prices}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Reading daily constituents from: {self.input_csv_daily_constituents}")
        try:
            daily_constituents_df_orig = pd.read_csv(
                self.input_csv_daily_constituents, parse_dates=["date"]
            )
            daily_constituents_df_orig["tickers"] = daily_constituents_df_orig["tickers"].apply(
                lambda x: (x.split(",") if isinstance(x, str) else ([] if pd.isna(x) else x))
            )
        except Exception as e:
            logger.error(f"Error reading {self.input_csv_daily_constituents}: {e}")
            raise

        logger.info(f"Reading extended prices from: {self.input_csv_extended_prices}")
        try:
            extended_prices_df = pd.read_csv(self.input_csv_extended_prices, parse_dates=["date"])
        except Exception as e:
            logger.error(f"Error reading {self.input_csv_extended_prices}: {e}")
            raise

        if daily_constituents_df_orig.empty:
            logger.warning("Original daily constituents DataFrame is empty. Cannot proceed.")
            return
        if extended_prices_df.empty:
            logger.warning("Extended prices DataFrame is empty. Cannot proceed.")
            return

        logger.info("Processing daily constituents data to fill gaps...")
        min_const_date = daily_constituents_df_orig["date"].min()
        max_overall_date = max(
            daily_constituents_df_orig["date"].max(),
            extended_prices_df["date"].max(),
        )

        all_business_days = pd.bdate_range(start=min_const_date, end=max_overall_date, name="date")

        daily_constituents_prepared_df = daily_constituents_df_orig.set_index("date").sort_index()

        daily_constituents_resampled_df = daily_constituents_prepared_df.reindex(all_business_days)
        daily_constituents_resampled_df["tickers"] = daily_constituents_resampled_df[
            "tickers"
        ].ffill()

        daily_constituents_filled_df = daily_constituents_resampled_df.dropna(
            subset=["tickers"]
        ).reset_index()

        if daily_constituents_filled_df.empty:
            logger.warning(
                "After attempting to fill gaps, the constituents DataFrame is empty."
                " Cannot proceed."
            )
            return

        logger.info(
            f"Constituents data expanded from {len(daily_constituents_df_orig)} "
            f"to {len(daily_constituents_filled_df)} entries "
            f"covering date range {daily_constituents_filled_df['date'].min().date()} "
            f"to {daily_constituents_filled_df['date'].max().date()}."
        )
        daily_constituents_loop_df = daily_constituents_filled_df

        price_analysis_df = self._calculate_smas_and_ad(extended_prices_df.copy())
        price_analysis_df.set_index(["date", "ticker"], inplace=True)

        breadth_results = []
        ad_line_cumulative = 0

        logger.info("Calculating daily breadth indicators...")
        for _, row in tqdm(
            daily_constituents_loop_df.iterrows(),
            total=daily_constituents_loop_df.shape[0],
        ):
            current_date = row["date"]
            constituent_tickers = row["tickers"]

            if not isinstance(constituent_tickers, list) or not constituent_tickers:
                logger.warning(
                    f"No constituent tickers or invalid format for {current_date}. "
                    f"Skipping. Tickers: {constituent_tickers}"
                )
                breadth_results.append(
                    {
                        "date": current_date,
                        "pct_above_sma50": None,
                        "pct_above_sma200": None,
                        "ad_line": ad_line_cumulative,
                        "ad_ratio": None,
                    }
                )
                continue

            if current_date not in price_analysis_df.index.get_level_values("date"):
                logger.warning(
                    f"No price data available in price_analysis_df for date {current_date}. "
                    "This might be a non-trading day not filtered by bdate_range "
                    "or missing price data. "
                    "Skipping breadth calculation for this date."
                )
                breadth_results.append(
                    {
                        "date": current_date,
                        "pct_above_sma50": None,
                        "pct_above_sma200": None,
                        "ad_line": ad_line_cumulative,
                        "ad_ratio": None,
                    }
                )
                continue

            daily_data = pd.DataFrame()
            try:
                daily_prices_for_date = price_analysis_df.loc[current_date]
                unique_constituent_tickers = list(set(constituent_tickers))
                daily_data = daily_prices_for_date[
                    daily_prices_for_date.index.isin(unique_constituent_tickers)
                ]
            except KeyError:
                logger.warning(
                    f"KeyError while accessing price data for {current_date}. "
                    f"Some constituent tickers might be missing. "
                    f"Attempting to proceed with available data."
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error looking up price data for {current_date} "
                    f"and tickers {constituent_tickers}: {e}"
                )

            if daily_data.empty:
                logger.warning(
                    f"No price data found for any constituents on {current_date} (after filtering)"
                    "Breadth indicators will be NaN or carry over AD-Line."
                )
                breadth_results.append(
                    {
                        "date": current_date,
                        "pct_above_sma50": None,
                        "pct_above_sma200": None,
                        "ad_line": ad_line_cumulative,
                        "ad_ratio": None,
                    }
                )
                continue

            above_sma50 = daily_data[daily_data["close"] > daily_data["sma50"]].shape[0]
            above_sma200 = daily_data[daily_data["close"] > daily_data["sma200"]].shape[0]

            valid_sma50_tickers = daily_data["sma50"].notna().sum()
            valid_sma200_tickers = daily_data["sma200"].notna().sum()

            pct_above_sma50 = (
                (above_sma50 / valid_sma50_tickers) * 100 if valid_sma50_tickers > 0 else None
            )
            pct_above_sma200 = (
                (above_sma200 / valid_sma200_tickers) * 100 if valid_sma200_tickers > 0 else None
            )

            advancing_issues = daily_data["advancing"].sum()
            declining_issues = daily_data["declining"].sum()

            net_advancers = advancing_issues - declining_issues
            ad_line_cumulative += net_advancers

            ad_ratio = None
            if declining_issues > 0:
                ad_ratio = advancing_issues / declining_issues
            elif advancing_issues > 0:
                ad_ratio = float("inf")

            breadth_results.append(
                {
                    "date": current_date,
                    "pct_above_sma50": pct_above_sma50,
                    "pct_above_sma200": pct_above_sma200,
                    "ad_line": ad_line_cumulative,
                    "ad_ratio": ad_ratio,
                }
            )

        if not breadth_results:
            logger.warning("No breadth indicators were calculated. DB will not be updated.")
            return

        results_df = pd.DataFrame(breadth_results)
        results_df["date"] = pd.to_datetime(results_df["date"])

        logger.info(
            f"Persisting {len(results_df)} breadth "
            f"indicators to table '{self.db_table_name}' in database: {self.db_path}"
        )
        try:
            results_df.to_sql(
                self.db_table_name,
                self.db_engine,
                if_exists=if_exists_db,
                index=False,
                dtype={
                    "date": sqlalchemy.types.Date,
                    "ad_line": sqlalchemy.types.Float,
                },
            )
            logger.info(f"Successfully saved data to table {self.db_table_name}")
        except Exception as e:
            logger.error(f"Error writing to database: {e}")
            raise
