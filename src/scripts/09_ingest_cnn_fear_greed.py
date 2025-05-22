import logging
import sys
from pathlib import Path

from regime_predictor_lib.data_ingestion.cnn_fear_greed_ingestor import CnnFearGreedIngestor
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    DB_DIR = PROJECT_ROOT_PATH / "data" / "db" / "volume"
    DB_PATH = DB_DIR / "quant.db"
    SCHEMA_PATH = PROJECT_ROOT_PATH / "data" / "db" / "schema.sql"
    CNN_FG_JSON_PATH = PROJECT_ROOT_PATH / "data" / "raw" / "cnn_fear_greed_data.json"

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

    logger.info("Starting CNN Fear & Greed Index data ingestion process.")
    ingestor = CnnFearGreedIngestor(json_filepath=CNN_FG_JSON_PATH)
    cnn_fg_df = ingestor.load_and_process_data()

    if cnn_fg_df is not None and not cnn_fg_df.empty:
        try:
            db_manager.upsert_dataframe(
                df=cnn_fg_df,
                table_name="cnn_fear_greed_index",
                conflict_columns=["reference_date"],
            )
            logger.info(
                f"Successfully upserted {len(cnn_fg_df)} records into 'cnn_fear_greed_index' table."
            )
        except Exception as e:
            logger.error(
                f"Error upserting CNN Fear & Greed data into database: {e}",
                exc_info=True,
            )
    elif cnn_fg_df is None:
        logger.error("CNN Fear & Greed data loading failed. Check JSON file path and format.")
    else:
        logger.warning("No CNN Fear & Greed data processed. DataFrame is empty.")

    logger.info("CNN Fear & Greed Index data ingestion script finished.")
