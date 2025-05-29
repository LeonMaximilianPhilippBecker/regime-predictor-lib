import logging
import sys
from pathlib import Path

import pandas as pd

from regime_predictor_lib.feature_engineering.master_feature_constructor import MasterFeatureConstructor
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH / "src"))
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DB_DIR = PROJECT_ROOT_PATH / "data" / "db" / "volume"
DB_PATH = DB_DIR / "quant.db"
SCHEMA_MASTER_FEATURES_PATH = PROJECT_ROOT_PATH / "data" / "db" / "schema_master_features.sql"

REGIME_CSV_PATH = (
    PROJECT_ROOT_PATH
    / "data"
    / "processed"
    / "regime_identification"
    / "smoothed"
    / "summaries"
    / "sp500_ret126d_logvol21d_3states_smoothed200_full_data_with_states.csv"
)
MASTER_TABLE_NAME = "master_features"

MASTER_START_DATE = "1986-01-01"
MASTER_END_DATE = None

if __name__ == "__main__":
    logger.info("Starting Master Feature Table Creation Script...")

    if not DB_PATH.parent.exists():
        logger.info(f"Creating database directory: {DB_PATH.parent}")
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not REGIME_CSV_PATH.exists():
        logger.error(f"Regime data CSV not found at: {REGIME_CSV_PATH}. Please ensure it exists.")
        sys.exit(1)

    if not SCHEMA_MASTER_FEATURES_PATH.exists():
        logger.error(f"Master features schema file not found at: {SCHEMA_MASTER_FEATURES_PATH}.")
        logger.error("Please create this file with the CREATE TABLE statement for 'master_features'.")
        sys.exit(1)

    db_manager = DatabaseManager(db_path=DB_PATH)

    try:
        logger.info(f"Applying schema for master_features table from: {SCHEMA_MASTER_FEATURES_PATH}")
        db_manager.create_tables_from_schema_file(SCHEMA_MASTER_FEATURES_PATH)
    except Exception as e:
        logger.error(f"Error applying master_features schema: {e}", exc_info=True)
        sys.exit(1)

    constructor = MasterFeatureConstructor(db_manager=db_manager, regime_csv_path=REGIME_CSV_PATH)

    try:
        master_df = constructor.construct_master_table(
            start_date_str=MASTER_START_DATE, end_date_str=MASTER_END_DATE
        )

        if master_df is not None and not master_df.empty:
            logger.info(f"Master DataFrame constructed with shape: {master_df.shape}")

            if db_manager.table_exists(MASTER_TABLE_NAME):
                logger.info(f"Upserting data to '{MASTER_TABLE_NAME}'...")

                df_for_sql = master_df.copy()
                for col in df_for_sql.columns:
                    if df_for_sql[col].dtype.name.startswith("Int"):
                        df_for_sql[col] = df_for_sql[col].astype(float)
                    elif df_for_sql[col].dtype == object:
                        try:
                            if pd.NA in df_for_sql[col].unique():
                                df_for_sql[col] = df_for_sql[col].replace({pd.NA: None})
                        except TypeError:
                            pass

                if "regime_t" in df_for_sql.columns:
                    df_for_sql["regime_t"] = (
                        pd.to_numeric(df_for_sql["regime_t"], errors="coerce").astype("Int64").astype(float)
                    )

                df_for_sql.dropna(subset=["regime_t"], inplace=True)

                try:
                    db_manager.upsert_dataframe(
                        df=df_for_sql,
                        table_name=MASTER_TABLE_NAME,
                        conflict_columns=["date"],
                    )
                    logger.info(f"Successfully upserted data into '{MASTER_TABLE_NAME}'.")
                except Exception as e:
                    logger.error(f"Error upserting data into {MASTER_TABLE_NAME}: {e}", exc_info=True)
                    logger.error("Check data types in DataFrame vs. DB schema, especially for NaNs/pd.NA.")
            else:
                logger.error(
                    f"Table '{MASTER_TABLE_NAME}' does not exist after schema application. Cannot upsert."
                )

        else:
            logger.warning("Master DataFrame is empty. Nothing to save or upsert.")

    except Exception as e:
        logger.error(
            f"An error occurred during master feature table construction: {e}",
            exc_info=True,
        )

    logger.info("Master Feature Table Creation Script finished.")
