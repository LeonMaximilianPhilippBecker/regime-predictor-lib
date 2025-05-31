import logging
from pathlib import Path

import pandas as pd
import sqlalchemy

from regime_predictor_lib.utils.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class StationarityBasedFeatureSelector:
    def __init__(
        self,
        db_manager: DatabaseManager,
        stationarity_summary_csv_path: Path,
        master_features_table_name: str = "master_features",
    ):
        self.db_manager = db_manager
        self.stationarity_summary_csv_path = stationarity_summary_csv_path
        self.master_features_table_name = master_features_table_name
        logger.info(
            "StationarityBasedFeatureSelector initialized. "
            f"Stationarity summary: {self.stationarity_summary_csv_path}"
        )

    def _load_master_features_table(self) -> pd.DataFrame | None:
        logger.info(f"Loading '{self.master_features_table_name}' table...")
        try:
            query = f"SELECT * FROM {self.master_features_table_name} ORDER BY date ASC"
            df = pd.read_sql_query(
                sql=sqlalchemy.text(query),
                con=self.db_manager.engine,
                parse_dates=["date"],
            )
            logger.info(f"Loaded '{self.master_features_table_name}' table with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading '{self.master_features_table_name}' table: {e}", exc_info=True)
            return None

    def select_features_and_update_db(self):
        if not self.stationarity_summary_csv_path.exists():
            logger.error(f"Stationarity summary CSV not found: {self.stationarity_summary_csv_path}")
            return

        logger.info(f"Reading stationarity summary from: {self.stationarity_summary_csv_path}")
        try:
            stationarity_df = pd.read_csv(self.stationarity_summary_csv_path)
        except Exception as e:
            logger.error(f"Error reading stationarity summary CSV: {e}", exc_info=True)
            return

        non_stationary_cols = stationarity_df[stationarity_df["overall_stationarity_0.05"] == "Non-Stationary"][
            "column_name"
        ].tolist()

        if not non_stationary_cols:
            logger.info("No columns marked as 'Non-Stationary' in the summary. No changes to make.")
            return

        logger.info(f"Identified {len(non_stationary_cols)} non-stationary columns to potentially drop.")

        master_df = self._load_master_features_table()
        if master_df is None or master_df.empty:
            logger.error("Master features table could not be loaded or is empty. Aborting.")
            return

        initial_columns_count = len(master_df.columns)
        columns_to_drop_existing = [col for col in non_stationary_cols if col in master_df.columns]

        if not columns_to_drop_existing:
            logger.info(
                "None of the non-stationary columns from summary exist in the current master_features table."
            )
            return

        logger.info(
            f"Dropping {len(columns_to_drop_existing)} non-stationary columns: {columns_to_drop_existing}"
        )
        master_df_updated = master_df.drop(columns=columns_to_drop_existing, errors="ignore")

        logger.info(
            f"Columns before dropping: {initial_columns_count}, "
            f"Columns after dropping: {len(master_df_updated.columns)}"
        )

        try:
            logger.info(f"Replacing data in '{self.master_features_table_name}' table in the database...")
            if "date" in master_df_updated.columns and pd.api.types.is_datetime64_any_dtype(
                master_df_updated["date"]
            ):
                master_df_updated["date"] = master_df_updated["date"].dt.strftime("%Y-%m-%d")

            master_df_updated.to_sql(
                self.master_features_table_name,
                self.db_manager.engine,
                if_exists="replace",
                index=False,
                chunksize=1000,
            )
            logger.info(f"Successfully updated '{self.master_features_table_name}' table in the database.")
        except Exception as e:
            logger.error(
                f"Error updating '{self.master_features_table_name}' table in database: {e}",
                exc_info=True,
            )
