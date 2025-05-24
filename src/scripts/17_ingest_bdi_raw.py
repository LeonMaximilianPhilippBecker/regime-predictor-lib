import logging
import sys
from pathlib import Path

from regime_predictor_lib.data_ingestion.bdi_ingestor import BdiIngestor
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DB_DIR = PROJECT_ROOT / "data" / "db" / "volume"
    DB_PATH = DB_DIR / "quant.db"
    SCHEMA_PATH = PROJECT_ROOT / "data" / "db" / "schema.sql"
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "bdi"
    BDI_DATA_SOURCE = RAW_DATA_DIR

    DB_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    bdi_files_exist = any(RAW_DATA_DIR.glob("*.csv"))  # A simple check
    if not bdi_files_exist:
        logger.warning(
            f"No CSV files found in {RAW_DATA_DIR}. "
            f"Please place your BDI CSV files (e.g., bdi_part1.csv, etc.) in this directory. "
            "The script will attempt to run, but BDI ingestion might fail or be empty."
        )
    db_manager = DatabaseManager(db_path=DB_PATH)
    try:
        logger.info(f"Creating/verifying database tables from schema: {SCHEMA_PATH}")
        db_manager.create_tables_from_schema_file(SCHEMA_PATH)
    except FileNotFoundError as e:
        logger.error(f"Schema file not found: {e}. Cannot proceed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error initializing database tables: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Starting BDI raw data ingestion process from source: {BDI_DATA_SOURCE}")
    ingestor = BdiIngestor(csv_source=BDI_DATA_SOURCE)
    bdi_df = ingestor.load_and_process_data()

    if bdi_df is not None and not bdi_df.empty:
        try:
            db_manager.upsert_dataframe(
                df=bdi_df,
                table_name="bdi_raw_csv",
                conflict_columns=["date"],
            )
            logger.info(f"Successfully upserted {len(bdi_df)} records into 'bdi_raw_csv' table.")
        except Exception as e:
            logger.error(
                f"Error upserting BDI raw data into database: {e}",
                exc_info=True,
            )
    elif bdi_df is None:
        logger.error("BDI raw data loading failed. Check CSV file(s) path and format.")
    else:
        logger.warning("No BDI raw data processed. DataFrame is empty.")

    logger.info("Script 17_ingest_bdi_raw.py finished.")
