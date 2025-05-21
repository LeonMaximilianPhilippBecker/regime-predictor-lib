import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from regime_predictor_lib.data_ingestion.api_clients import BaseAPIClient

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
]


class SP500HistoricalDataProcessor:
    def __init__(
        self,
        raw_data_dir: Path,
        processed_data_dir: Path,
        api_clients: List[BaseAPIClient],
        input_start_end_filename: str = "sp500_ticker_start_end.csv",
        output_extended_prices_filename: str = "sp500_ticker_prices_extended.csv",
    ):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.input_csv_start_end = self.raw_data_dir / input_start_end_filename
        self.output_csv_extended_prices = self.processed_data_dir / output_extended_prices_filename

        if not api_clients:
            raise ValueError("At least one API client must be provided.")
        self.api_clients = api_clients

        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        client_names = [client.client_name for client in self.api_clients]
        logger.info(
            f"SP500HistoricalDataProcessor initialized. Input: {self.input_csv_start_end}, "
            f"Output: {self.output_csv_extended_prices}, API Clients: {', '.join(client_names)}"
        )

    def _get_processed_tickers(self) -> set:
        processed_tickers = set()
        if self.output_csv_extended_prices.exists():
            try:
                chunk_iter = pd.read_csv(
                    self.output_csv_extended_prices,
                    usecols=["ticker"],
                    chunksize=100000,
                )
                for chunk in chunk_iter:
                    processed_tickers.update(chunk["ticker"].unique())
                logger.info(
                    f"Found {len(processed_tickers)} previously processed "
                    f"tickers in {self.output_csv_extended_prices}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not read processed tickers "
                    f"from {self.output_csv_extended_prices}: {e}. Will re-fetch all."
                )
        return processed_tickers

    def _fetch_data_with_fallback(
        self, ticker: str, start_date_str: str, end_date_str: str
    ) -> pd.DataFrame | None:
        for client in self.api_clients:
            logger.debug(
                f"Attempting to fetch {ticker} using {client.client_name} "
                f"from {start_date_str} to {end_date_str}"
            )
            price_df = client.fetch_ohlcv_data(ticker, start_date_str, end_date_str)
            if price_df is not None and not price_df.empty:
                logger.info(f"Successfully fetched data for {ticker} using {client.client_name}")
                return price_df
            logger.warning(
                f"Failed to fetch data for {ticker} using "
                f"{client.client_name}. Trying next client if available."
            )
        logger.error(f"All API clients failed to fetch data for {ticker}.")
        return None

    def generate_extended_price_data(self, force_regenerate_all: bool = False) -> Path:
        if not self.input_csv_start_end.exists():
            logger.error(f"Input file not found: {self.input_csv_start_end}")
            raise FileNotFoundError(f"Input file not found: {self.input_csv_start_end}")

        logger.info(f"Reading ticker start/end dates from: {self.input_csv_start_end}")
        try:
            start_end_df = pd.read_csv(
                self.input_csv_start_end, parse_dates=["start_date", "end_date"]
            )
        except Exception as e:
            logger.error(f"Error reading {self.input_csv_start_end}: {e}")
            raise

        processed_tickers = set()
        file_exists = self.output_csv_extended_prices.exists()
        write_mode = "a"
        include_header = not file_exists

        if force_regenerate_all:
            logger.info(
                "`force_regenerate_all` is True. Re-fetching all data and overwriting output CSV."
            )
            write_mode = "w"
            include_header = True
            if file_exists:
                try:
                    self.output_csv_extended_prices.unlink()
                    logger.info(f"Removed existing file: {self.output_csv_extended_prices}")
                except OSError as e:
                    logger.error(
                        f"Error removing existing file {self.output_csv_extended_prices}: {e}"
                    )
        elif file_exists:
            processed_tickers = self._get_processed_tickers()

        logger.info("Processing tickers for extended price data...")

        with open(
            self.output_csv_extended_prices,
            mode=write_mode,
            newline="",
            encoding="utf-8",
        ) as f:
            if include_header:
                pd.DataFrame(columns=CSV_COLUMNS).to_csv(f, index=False, header=True)
                logger.info(f"Writing header to new file: {self.output_csv_extended_prices}")

            for _, row in tqdm(
                start_end_df.iterrows(),
                total=start_end_df.shape[0],
                desc="Processing Tickers",
            ):
                ticker = row["ticker"]
                start_date_dt = row["start_date"]
                end_date_dt = row["end_date"]

                if not force_regenerate_all and ticker in processed_tickers:
                    logger.debug(
                        f"Skipping {ticker}, already processed and not forcing regeneration."
                    )
                    continue

                if pd.isna(start_date_dt):
                    logger.warning(f"Skipping {ticker} due to missing start_date.")
                    continue

                extended_start_date_dt = start_date_dt - timedelta(days=250)
                fetch_start_date_str = extended_start_date_dt.strftime("%Y-%m-%d")

                if pd.isna(end_date_dt):
                    fetch_end_date_str = datetime.today().strftime("%Y-%m-%d")
                else:
                    fetch_end_date_str = end_date_dt.strftime("%Y-%m-%d")

                time.sleep(1)

                price_df = self._fetch_data_with_fallback(
                    ticker, fetch_start_date_str, fetch_end_date_str
                )

                if price_df is not None and not price_df.empty:
                    price_df_ordered = price_df.reindex(columns=CSV_COLUMNS)

                    if price_df_ordered.isnull().all().all():
                        logger.error(
                            f"[{ticker}] ENTIRE price_df_ordered IS NaN after reindex."
                            f" Original price_df might have been all NaNs or there's"
                            f" a severe column name mismatch."
                        )
                    elif (
                        price_df_ordered.drop(columns=["ticker", "date"], errors="ignore")
                        .isnull()
                        .all(axis=1)
                        .any()
                    ):
                        logger.warning(
                            f"[{ticker}] price_df_ordered contains"
                            " rows where all numeric columns are NaN."
                        )
                    elif price_df_ordered.isnull().values.any():
                        logger.warning(
                            f"[{ticker}] price_df_ordered CONTAINS SOME NaNs "
                            f"after reindex (but not all-NaN rows for numeric cols). "
                            f"Head:\n{price_df_ordered.head().to_string()}"
                        )

                    subset_cols_for_dropna = [
                        "open",
                        "high",
                        "low",
                        "close",
                        "adjusted_close",
                        "volume",
                    ]
                    original_row_count = len(price_df_ordered)
                    cleaned_price_df_ordered = price_df_ordered.dropna(
                        subset=subset_cols_for_dropna, how="all"
                    )

                    if len(cleaned_price_df_ordered) < original_row_count:
                        logger.warning(
                            f"[{ticker}] "
                            f"Dropped {original_row_count - len(cleaned_price_df_ordered)} "
                            "rows with all NaNs in key financial columns."
                        )

                    if not cleaned_price_df_ordered.empty:
                        cleaned_price_df_ordered.to_csv(
                            f, header=False, index=False, date_format="%Y-%m-%d"
                        )
                        f.flush()
                        logger.debug(f"Appended data for {ticker} to CSV.")
                    else:
                        logger.warning(
                            f"[{ticker}] No valid data rows left "
                            f"for {ticker} after dropping NaNs. Not writing to CSV."
                        )

                else:
                    logger.warning(f"No data fetched for {ticker}. Not writing to CSV.")

        logger.info(
            f"Extended price data processing complete. Output at: {self.output_csv_extended_prices}"
        )
        return self.output_csv_extended_prices
