import logging
import sys
from pathlib import Path

from regime_predictor_lib.data_cleaning.master_features_imputer import MasterFeaturesImputer
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_PATH / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH / "src"))
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

script_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


DB_PATH = PROJECT_ROOT_PATH / "data" / "db" / "volume" / "quant.db"
ANALYSIS_YAML_PATH = PROJECT_ROOT_PATH / "data" / "processed" / "master_features_column_analysis.yaml"
IMPUTATION_LOG_DETAILS_PATH = PROJECT_ROOT_PATH / "logs" / "master_features_imputation_details.log"


if __name__ == "__main__":
    script_logger.info("Starting Master Features Imputation Script...")

    IMPUTATION_LOG_DETAILS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not DB_PATH.exists():
        script_logger.error(f"Database not found at {DB_PATH}. Cannot proceed.")
        sys.exit(1)

    if not ANALYSIS_YAML_PATH.exists():
        script_logger.error(
            f"Column analysis YAML file not found at {ANALYSIS_YAML_PATH}. "
            "Run script '30_analyze_master_features.py' first."
        )
        sys.exit(1)

    db_manager = DatabaseManager(db_path=DB_PATH)
    imputer = MasterFeaturesImputer(
        db_manager=db_manager,
        analysis_yaml_path=ANALYSIS_YAML_PATH,
        imputation_log_file_path=IMPUTATION_LOG_DETAILS_PATH,
    )

    try:
        imputer.run_imputation_workflow()
        script_logger.info("Master features imputation script finished successfully.")
        script_logger.info(f"Detailed imputation log saved to: {IMPUTATION_LOG_DETAILS_PATH}")
    except Exception as e:
        script_logger.error(
            f"An error occurred during the master features imputation: {e}",
            exc_info=True,
        )
