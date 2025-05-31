import logging
import sys
from pathlib import Path

from regime_predictor_lib.feature_engineering import FeatureReducer
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
OUTPUT_DIR_FEATURE_LISTS = (
    PROJECT_ROOT_PATH / "data" / "processed" / "feature_selection" / "thematic_feature_lists"
)

MUTUAL_INFO_TARGET_COL = "regime_t"
CORRELATION_THRESHOLD = 0.9
VIF_THRESHOLD = 7.5

if __name__ == "__main__":
    logger.info("Starting Thematic Feature Reduction Script...")

    OUTPUT_DIR_FEATURE_LISTS.mkdir(parents=True, exist_ok=True)

    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}. Cannot proceed.")
        sys.exit(1)

    db_manager = DatabaseManager(db_path=DB_PATH)

    reducer = FeatureReducer(
        db_manager=db_manager,
        output_dir=OUTPUT_DIR_FEATURE_LISTS,
        mutual_info_target_col=MUTUAL_INFO_TARGET_COL,
        correlation_threshold=CORRELATION_THRESHOLD,
        vif_threshold=VIF_THRESHOLD,
    )

    try:
        reducer.process_all_thematic_tables()
        logger.info("Thematic feature reduction script finished successfully.")
        logger.info(f"Selected feature lists saved to: {OUTPUT_DIR_FEATURE_LISTS}")
    except Exception as e:
        logger.error(
            f"An error occurred during thematic feature reduction: {e}",
            exc_info=True,
        )
