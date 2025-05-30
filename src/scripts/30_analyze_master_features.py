import logging
import sys
from pathlib import Path

from regime_predictor_lib.analytics.column_analyzer import ColumnAnalyzer
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
OUTPUT_YAML_PATH = PROJECT_ROOT_PATH / "data" / "processed" / "master_features_column_analysis.yaml"


if __name__ == "__main__":
    logger.info("Starting Master Features Column Analysis Script...")

    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}. Cannot proceed.")
        sys.exit(1)

    db_manager = DatabaseManager(db_path=DB_PATH)
    analyzer = ColumnAnalyzer(db_manager=db_manager, output_yaml_path=OUTPUT_YAML_PATH)

    try:
        analyzer.run_analysis_and_save()
        logger.info("Master features column analysis finished successfully.")
    except Exception as e:
        logger.error(
            f"An error occurred during the master features column analysis: {e}",
            exc_info=True,
        )
