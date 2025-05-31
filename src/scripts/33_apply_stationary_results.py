import logging
import sys
from pathlib import Path

from regime_predictor_lib.feature_engineering import StationarityBasedFeatureSelector
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

DB_PATH = PROJECT_ROOT_PATH / "data" / "db" / "volume" / "quant.db"
STATIONARITY_SUMMARY_CSV_PATH = (
    PROJECT_ROOT_PATH
    / "data"
    / "reports"
    / "stationarity_analysis"
    / "master_features_stationarity_summary.csv"
)
MASTER_FEATURES_TABLE_NAME = "master_features"

if __name__ == "__main__":
    logger.info("Starting script to apply stationarity results to master_features table...")

    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}. Cannot proceed.")
        sys.exit(1)

    if not STATIONARITY_SUMMARY_CSV_PATH.exists():
        logger.error(
            f"Stationarity summary CSV not found at {STATIONARITY_SUMMARY_CSV_PATH}. "
            "Run script '32_perform_stationarity_tests.py' first."
        )
        sys.exit(1)

    db_manager = DatabaseManager(db_path=DB_PATH)
    selector = StationarityBasedFeatureSelector(
        db_manager=db_manager,
        stationarity_summary_csv_path=STATIONARITY_SUMMARY_CSV_PATH,
        master_features_table_name=MASTER_FEATURES_TABLE_NAME,
    )

    try:
        selector.select_features_and_update_db()
        logger.info("Stationarity results application script finished successfully.")
    except Exception as e:
        logger.error(
            f"An error occurred during the application of stationarity results: {e}",
            exc_info=True,
        )
