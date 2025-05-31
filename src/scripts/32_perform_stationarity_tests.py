import logging
import sys
from pathlib import Path

from regime_predictor_lib.feature_engineering import StationarityAnalyzer
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
REPORTS_DIR = PROJECT_ROOT_PATH / "data" / "reports"
STATIONARITY_REPORTS_DIR = REPORTS_DIR / "stationarity_analysis"
OUTPUT_CSV_PATH = STATIONARITY_REPORTS_DIR / "master_features_stationarity_summary.csv"


if __name__ == "__main__":
    logger.info("Starting Stationarity Analysis Script for Master Features...")

    STATIONARITY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}. Cannot proceed.")
        sys.exit(1)

    db_manager = DatabaseManager(db_path=DB_PATH)
    stationarity_analyzer = StationarityAnalyzer(db_manager=db_manager, output_csv_path=OUTPUT_CSV_PATH)

    try:
        stationarity_analyzer.run_analysis_and_save()
        logger.info("Stationarity analysis script finished successfully.")
        logger.info(f"Results saved to: {OUTPUT_CSV_PATH}")
    except Exception as e:
        logger.error(
            f"An error occurred during the stationarity analysis: {e}",
            exc_info=True,
        )
