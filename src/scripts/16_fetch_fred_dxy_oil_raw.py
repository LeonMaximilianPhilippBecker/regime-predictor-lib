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

FRED_SERIES_FOR_RAW_FETCH = [
    "DTWEXBGS",  # DXY Broad
    "DTWEXM",  # DXY Major
    "DCOILWTICO",  # WTI
    "DCOILBRENTEU",  # Brent
]

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

    logger.info(f"Starting DXY and Oil raw data fetching using series: {FRED_SERIES_FOR_RAW_FETCH}")
    try:
        success_count, failure_count = ingestor.ingest_all_configured_series(
            series_ids_to_process=FRED_SERIES_FOR_RAW_FETCH, inter_series_delay=0.5
        )
        logger.info(
            "DXY and Oil raw data ingestion complete. "
            f"Succeeded: {success_count}, Failed: {failure_count}."
        )
    except Exception as e:
        logger.error(
            "An uncaught error occurred during the DXY/Oil raw data ingestion process: " f"{e}",
            exc_info=True,
        )

    logger.info("Script 16_fetch_fred_dxy_oil_raw.py finished.")
