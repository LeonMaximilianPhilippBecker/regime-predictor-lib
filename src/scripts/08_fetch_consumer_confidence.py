import logging
import sys
from pathlib import Path

from regime_predictor_lib.data_ingestion.api_clients.fred_client import FredApiClient
from regime_predictor_lib.data_ingestion.fred_economic_indicator_ingestor import (
    FredEconomicIndicatorIngestor,
)
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONSUMER_CONFIDENCE_SERIES_ID = "UMCSENT"

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DB_DIR = PROJECT_ROOT / "data" / "db" / "volume"
    DB_PATH = DB_DIR / "quant.db"
    SCHEMA_PATH = PROJECT_ROOT / "data" / "db" / "schema.sql"

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

    try:
        fred_client = FredApiClient()
    except ValueError as e:
        logger.error(f"Failed to initialize FredApiClient: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(
            f"An unexpected error occurred initializing FredApiClient: {e}",
            exc_info=True,
        )
        sys.exit(1)

    ingestor = FredEconomicIndicatorIngestor(fred_client=fred_client, db_manager=db_manager)

    logger.info(
        f"Starting Consumer Confidence (Series ID: {CONSUMER_CONFIDENCE_SERIES_ID}) "
        "data fetching and storage process."
    )
    try:
        success_count, failure_count = ingestor.ingest_all_configured_series(
            series_ids_to_process=[CONSUMER_CONFIDENCE_SERIES_ID],
            inter_series_delay=0,
        )
        if success_count > 0:
            logger.info(
                f"Consumer Confidence data (Series ID: {CONSUMER_CONFIDENCE_SERIES_ID}) "
                f"ingestion complete. Succeeded: {success_count}."
            )
        elif failure_count > 0:
            logger.warning(
                f"Consumer Confidence data (Series ID: {CONSUMER_CONFIDENCE_SERIES_ID}) "
                f"ingestion failed. Failed: {failure_count}."
            )
        else:
            logger.info(
                f"No data processed for Consumer Confidence (Series "
                f"ID: {CONSUMER_CONFIDENCE_SERIES_ID}). This might be because the series ID"
                f" was not found in the configuration or no data was returned from FRED."
            )

    except Exception as e:
        logger.error(
            "An uncaught error occurred during the Consumer "
            "Confidence indicator ingestion process: "
            f"{e}",
            exc_info=True,
        )

    logger.info("Script 08_fetch_consumer_confidence.py finished.")
