import logging
import os
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

logger = logging.getLogger(__name__)
load_dotenv()

FRED_API_KEY_ENV = os.getenv("FRED_API_KEY")


class FredApiClient:
    def __init__(self, api_key: str | None = None):
        self.client_name = "FredApiClient"
        self.api_key = api_key or FRED_API_KEY_ENV
        if not self.api_key:
            msg = (
                f"[{self.client_name}] FRED API key not provided or found "
                "in environment variable FRED_API_KEY. Client will not function."
            )
            logger.error(msg)
            raise ValueError(msg)
        try:
            self.fred = Fred(api_key=self.api_key)
        except Exception as e:
            logger.error(f"[{self.client_name}] Failed to initialize Fred connection: {e}")
            raise
        logger.info(f"{self.client_name} initialized.")

    def fetch_series_all_releases(
        self,
        series_id: str,
        retry_attempts: int = 3,
        retry_delay: int = 5,
        realtime_chunk_years: int = 5,
        initial_realtime_start_date_str: str = "1970-01-01",
    ) -> pd.DataFrame | None:
        all_vintage_data_dfs = []

        current_realtime_start_dt = pd.to_datetime(initial_realtime_start_date_str)
        final_overall_realtime_end_dt = pd.to_datetime(datetime.now().date())

        logger.info(
            f"[{self.client_name}] Starting chunked fetch for all releases of series {series_id}. "
            f"Targeting releases up to {final_overall_realtime_end_dt.strftime('%Y-%m-%d')}."
        )

        iteration_count = 0
        max_iterations = 50

        while (
            current_realtime_start_dt <= final_overall_realtime_end_dt
            and iteration_count < max_iterations
        ):
            iteration_count += 1
            realtime_chunk_end_dt = (
                current_realtime_start_dt
                + pd.DateOffset(years=realtime_chunk_years)
                - pd.Timedelta(days=1)
            )

            if realtime_chunk_end_dt > final_overall_realtime_end_dt:
                realtime_chunk_end_dt = final_overall_realtime_end_dt

            current_realtime_start_str = current_realtime_start_dt.strftime("%Y-%m-%d")
            realtime_chunk_end_str = realtime_chunk_end_dt.strftime("%Y-%m-%d")

            logger.info(
                f"[{self.client_name}] Chunk {iteration_count}: Fetching releases for {series_id} "
                f"in realtime period: {current_realtime_start_str} to {realtime_chunk_end_str}"
            )

            attempt = 0
            chunk_data_df = None
            while attempt < retry_attempts:
                try:
                    data_series_chunk = self.fred.get_series_all_releases(
                        series_id,
                        realtime_start=current_realtime_start_str,
                        realtime_end=realtime_chunk_end_str,
                    )

                    if not data_series_chunk.empty:
                        chunk_data_df = data_series_chunk.reset_index()
                        all_vintage_data_dfs.append(chunk_data_df)
                        logger.debug(
                            f"[{self.client_name}] Chunk {iteration_count}: "
                            f"Fetched {len(chunk_data_df)} records."
                        )
                    else:
                        logger.debug(
                            f"[{self.client_name}] Chunk {iteration_count}: "
                            "No data in this realtime period."
                        )
                    break

                except Exception as e:
                    logger.error(
                        f"[{self.client_name}] Chunk {iteration_count}: Error "
                        f"fetching data for series {series_id} "
                        f"(Attempt {attempt + 1}/{retry_attempts}) "
                        f"for realtime chunk {current_realtime_start_str} "
                        f"to {realtime_chunk_end_str}: {e}",
                    )
                    attempt += 1
                    if attempt < retry_attempts:
                        logger.info(f"Retrying chunk in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(
                            f"Failed to fetch chunk {iteration_count} for {series_id} "
                            f"after {retry_attempts} attempts."
                        )
                        break

            if realtime_chunk_end_dt >= final_overall_realtime_end_dt:
                logger.info(
                    f"[{self.client_name}] Reached final overall realtime end "
                    f"date {final_overall_realtime_end_dt.strftime('%Y-%m-%d')}. "
                    "Ending chunk loop."
                )
                break

            current_realtime_start_dt = realtime_chunk_end_dt + pd.Timedelta(days=1)

            if current_realtime_start_dt > final_overall_realtime_end_dt:
                logger.info(
                    f"[{self.client_name}] Next chunk "
                    f"start {current_realtime_start_dt.strftime('%Y-%m-%d')} "
                    f"is after final "
                    f"end {final_overall_realtime_end_dt.strftime('%Y-%m-%d')}. "
                    "Ending chunk loop."
                )
                break

            time.sleep(0.5)

        if iteration_count >= max_iterations:
            logger.warning(
                f"[{self.client_name}] Reached max iterations ({max_iterations}) "
                f"for series {series_id}. Loop terminated."
            )

        if not all_vintage_data_dfs:
            logger.warning(
                f"[{self.client_name}] No data returned by FRED "
                f"for series {series_id} after all chunks."
            )
            return None

        combined_df = pd.concat(all_vintage_data_dfs, ignore_index=True)

        combined_df.rename(
            columns={
                "realtime_start": "release_date",
                "date": "reference_date",
            },
            inplace=True,
        )
        if "value" not in combined_df.columns and 0 in combined_df.columns:
            combined_df.rename(columns={0: "value"}, inplace=True)

        combined_df["release_date"] = pd.to_datetime(
            combined_df["release_date"], errors="coerce"
        ).dt.date
        combined_df["reference_date"] = pd.to_datetime(
            combined_df["reference_date"], errors="coerce"
        ).dt.date
        combined_df["value"] = pd.to_numeric(combined_df["value"], errors="coerce")

        initial_count = len(combined_df)
        combined_df.dropna(subset=["release_date", "reference_date"], inplace=True)
        if len(combined_df) < initial_count:
            logger.debug(
                f"Dropped {initial_count - len(combined_df)} rows "
                "due to missing release_date or reference_date."
            )

        combined_df["series_id"] = series_id

        if combined_df.empty:
            logger.warning(
                f"[{self.client_name}] Combined DataFrame is empty "
                f"for {series_id} before deduplication."
            )
            return None

        combined_df.drop_duplicates(
            subset=["reference_date", "release_date", "value", "series_id"],
            keep="first",
            inplace=True,
        )

        combined_df.sort_values(by=["series_id", "reference_date", "release_date"], inplace=True)

        if not combined_df.empty:
            logger.info(
                f"[{self.client_name}] Max reference_date "
                f"fetched: {combined_df['reference_date'].max()}"
            )
            logger.info(
                f"[{self.client_name}] Max release_date "
                f"fetched: {combined_df['release_date'].max()}"
            )

        logger.info(
            f"[{self.client_name}] Successfully fetched and combined {len(combined_df)} "
            f"unique vintage records for series {series_id} from all chunks."
        )
        return combined_df[["reference_date", "release_date", "value", "series_id"]]
