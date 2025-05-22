import logging
import sys
from pathlib import Path

from regime_predictor_lib.data_ingestion.aaii_sentiment_ingestor import AaiiSentimentIngestor
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    DB_DIR = PROJECT_ROOT_PATH / "data" / "db" / "volume"
    DB_PATH = DB_DIR / "quant.db"
    SCHEMA_PATH = PROJECT_ROOT_PATH / "data" / "db" / "schema.sql"
    AAII_CSV_PATH = PROJECT_ROOT_PATH / "data" / "raw" / "aaii_sentiment_data.csv"

    DB_DIR.mkdir(parents=True, exist_ok=True)

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

    logger.info("Starting AAII sentiment data ingestion process.")
    ingestor = AaiiSentimentIngestor(csv_filepath=AAII_CSV_PATH)
    aaii_df = ingestor.load_and_process_data()

    if aaii_df is not None and not aaii_df.empty:
        try:
            db_manager.upsert_dataframe(
                df=aaii_df,
                table_name="aaii_sentiment",
                conflict_columns=["reference_date", "release_date"],
            )
            logger.info(
                f"Successfully upserted {len(aaii_df)} records into 'aaii_sentiment' table."
            )
        except Exception as e:
            logger.error(
                f"Error upserting AAII sentiment data into database: {e}",
                exc_info=True,
            )
    elif aaii_df is None:
        logger.error("AAII sentiment data loading failed. Check CSV file path and format.")
    else:
        logger.warning("No AAII sentiment data processed. DataFrame is empty.")

    logger.info("AAII sentiment data ingestion script finished.")
