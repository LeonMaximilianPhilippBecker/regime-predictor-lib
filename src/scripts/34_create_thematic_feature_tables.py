import logging
import sys
from pathlib import Path

from regime_predictor_lib.feature_engineering import ThematicFeatureTableCreator
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
MASTER_FEATURES_TABLE_NAME = "master_features"
COLUMN_ANALYSIS_YAML_PATH = PROJECT_ROOT_PATH / "data" / "processed" / "master_features_column_analysis.yaml"

if __name__ == "__main__":
    logger.info("Starting script to create thematic feature tables...")

    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}. Cannot proceed.")
        sys.exit(1)

    if not COLUMN_ANALYSIS_YAML_PATH.exists():
        logger.error(
            f"Column analysis YAML file not found at {COLUMN_ANALYSIS_YAML_PATH}. "
            "Run script '30_analyze_master_features.py' first."
        )
        sys.exit(1)

    db_manager = DatabaseManager(db_path=DB_PATH)
    creator = ThematicFeatureTableCreator(
        db_manager=db_manager,
        master_features_table_name=MASTER_FEATURES_TABLE_NAME,
        column_analysis_yaml_path=COLUMN_ANALYSIS_YAML_PATH,
    )

    try:
        creator.generate_tables()
        logger.info("Thematic feature table creation script finished successfully.")
    except Exception as e:
        logger.error(
            f"An error occurred during the thematic table creation: {e}",
            exc_info=True,
        )
