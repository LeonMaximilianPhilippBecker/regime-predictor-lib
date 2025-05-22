import logging
import time

import pandas as pd

from ..utils.database_manager import DatabaseManager
from .api_clients.fred_client import FredApiClient

logger = logging.getLogger(__name__)

FRED_SERIES_CONFIG = {
    "PAYEMS": {  # Non-Farm Payrolls
        "table_name": "non_farm_payrolls",
        "conflict_columns": ["reference_date", "release_date", "series_id"],
        "expected_db_cols": ["reference_date", "release_date", "value", "series_id"],
    },
    "ICSA": {  # Initial Jobless Claims
        "table_name": "initial_jobless_claims",
        "conflict_columns": ["reference_date", "release_date", "series_id"],
        "expected_db_cols": ["reference_date", "release_date", "value", "series_id"],
    },
    "CPIAUCSL": {  # CPI
        "table_name": "cpi",
        "conflict_columns": ["reference_date", "release_date", "series_id"],
        "expected_db_cols": ["reference_date", "release_date", "value", "series_id"],
    },
    "RGDPQOQFOR": {  # GDP Growth Forecasts (SPF)
        "table_name": "gdp_forecasts",
        "conflict_columns": ["reference_date", "release_date", "series_id"],
        "preprocess_func": lambda df: df.assign(
            forecast_period=df["reference_date"].dt.to_period("Q").astype(str)
        ),
        "expected_db_cols": [
            "reference_date",
            "forecast_period",
            "release_date",
            "value",
            "series_id",
        ],
    },
    "RSAFS": {  # Retail Sales
        "table_name": "retail_sales",
        "conflict_columns": ["reference_date", "release_date", "series_id"],
        "expected_db_cols": ["reference_date", "release_date", "value", "series_id"],
    },
    "M2SL": {  # M2 Money Supply
        "table_name": "m2_money_supply",
        "conflict_columns": ["reference_date", "release_date", "series_id"],
        "expected_db_cols": ["reference_date", "release_date", "value", "series_id"],
    },
    "HOUST": {  # Housing Starts
        "table_name": "housing_starts",
        "conflict_columns": ["reference_date", "release_date", "series_id"],
        "expected_db_cols": ["reference_date", "release_date", "value", "series_id"],
    },
    "CSUSHPINSA": {  # Housing Prices (Case-Shiller)
        "table_name": "housing_prices",
        "conflict_columns": ["reference_date", "release_date", "series_id"],
        "expected_db_cols": ["reference_date", "release_date", "value", "series_id"],
    },
    "UMCSENT": {  # Conference Board Consumer Confidence Index
        "table_name": "consumer_confidence",
        "conflict_columns": ["reference_date", "release_date", "series_id"],
        "expected_db_cols": ["reference_date", "release_date", "value", "series_id"],
    },
}


class FredEconomicIndicatorIngestor:
    def __init__(
        self,
        fred_client: FredApiClient,
        db_manager: DatabaseManager,
        series_config: dict | None = None,
    ):
        self.fred_client = fred_client
        self.db_manager = db_manager
        self.series_config = series_config or FRED_SERIES_CONFIG
        logger.info("FredEconomicIndicatorIngestor initialized.")

    def _prepare_dataframe_for_db(
        self, df: pd.DataFrame, series_id: str, config_entry: dict
    ) -> pd.DataFrame | None:
        processed_df = df.copy()

        if "preprocess_func" in config_entry and callable(config_entry["preprocess_func"]):
            try:
                processed_df = config_entry["preprocess_func"](processed_df)
                logger.debug(f"Applied preprocessing for series {series_id}")
            except Exception as e:
                logger.error(f"Error during preprocessing for {series_id}: {e}", exc_info=True)
                return None

        expected_cols = config_entry.get("expected_db_cols")
        if not expected_cols:
            logger.warning(
                f"No 'expected_db_cols' defined for series {series_id}. "
                "Using all columns from fetched data."
            )
            if "series_id" not in processed_df.columns:
                processed_df["series_id"] = series_id
            return processed_df

        if "series_id" not in processed_df.columns and "series_id" in expected_cols:
            processed_df["series_id"] = series_id

        missing_cols = [col for col in expected_cols if col not in processed_df.columns]
        if missing_cols:
            logger.error(
                f"DataFrame for series {series_id} is missing expected columns for table "
                f"'{config_entry['table_name']}': {missing_cols}. "
                f"Available columns: {processed_df.columns.tolist()}"
            )
            return None

        return processed_df[expected_cols]

    def ingest_series(self, series_id: str, config_entry: dict) -> bool:
        logger.info(f"Ingesting FRED series: {series_id}")
        table_name = config_entry["table_name"]
        conflict_cols = config_entry["conflict_columns"]

        raw_df = self.fred_client.fetch_series_all_releases(series_id)

        if raw_df is None or raw_df.empty:
            logger.warning(f"No data fetched for series {series_id}. Skipping database update.")
            return False

        df_to_upsert = self._prepare_dataframe_for_db(raw_df, series_id, config_entry)

        if df_to_upsert is None or df_to_upsert.empty:
            logger.warning(
                f"DataFrame for {series_id} is empty or None after preparation. "
                "Skipping database upsert."
            )
            return False

        try:
            self.db_manager.upsert_dataframe(
                df=df_to_upsert,
                table_name=table_name,
                conflict_columns=conflict_cols,
            )
            logger.info(
                f"Successfully upserted {len(df_to_upsert)} records for {series_id} "
                f"into table {table_name}."
            )
            return True
        except Exception as e:
            logger.error(
                f"Error upserting data for {series_id} into {table_name}: {e}",
                exc_info=True,
            )
            return False

    def ingest_all_configured_series(
        self, series_ids_to_process: list[str] | None = None, inter_series_delay: float = 0.5
    ) -> tuple[int, int]:
        logger.info("Starting ingestion of FRED economic indicators.")

        series_to_iterate = {}
        if series_ids_to_process is None:
            logger.info("Processing all series defined in FRED_SERIES_CONFIG.")
            series_to_iterate = self.series_config
        else:
            logger.info(f"Processing specific series_ids: {series_ids_to_process}")
            for series_id_key in series_ids_to_process:
                if series_id_key in self.series_config:
                    series_to_iterate[series_id_key] = self.series_config[series_id_key]
                else:
                    logger.warning(
                        f"Series ID '{series_id_key}' provided in series_ids_to_process "
                        f"but not found in FRED_SERIES_CONFIG. Skipping."
                    )
            if not series_to_iterate:
                logger.warning("No valid series found to process based on series_ids_to_process.")
                return 0, 0

        total_series_to_process = len(series_to_iterate)
        if total_series_to_process == 0:
            logger.info("No series selected for processing.")
            return 0, 0

        success_count = 0
        failure_count = 0

        for i, (series_id, config_entry) in enumerate(series_to_iterate.items()):
            logger.info(
                f"Processing series {i + 1}/{total_series_to_process}: {series_id} "
                f"-> table '{config_entry['table_name']}'"
            )

            if self.ingest_series(series_id, config_entry):
                success_count += 1
            else:
                failure_count += 1

            if i < total_series_to_process - 1 and inter_series_delay > 0:
                logger.debug(f"Waiting {inter_series_delay}s before next series request...")
                time.sleep(inter_series_delay)

        logger.info(
            "FRED economic indicator ingestion process finished. "
            f"Successfully ingested: {success_count}, Failed: {failure_count}."
        )
        return success_count, failure_count
