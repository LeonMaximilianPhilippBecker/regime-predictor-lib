import logging
import sys
from pathlib import Path

from regime_predictor_lib.regime_identification import HMMRegimeAnalyzer
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_DIR = PROJECT_ROOT_PATH / "data" / "db" / "volume"
DB_PATH = DB_DIR / "quant.db"
SCHEMA_PATH = PROJECT_ROOT_PATH / "data" / "db" / "schema.sql"

MODELS_OUTPUT_DIR = PROJECT_ROOT_PATH / "data" / "models" / "hmm"
RESULTS_OUTPUT_DIR = PROJECT_ROOT_PATH / "data" / "processed" / "regime_identification"

ANALYSIS_NAME = "sp500_ret252d_logvol126d_3states"
FEATURES_TO_USE = ["ret_126d", "log_vol_21d"]
N_HMM_STATES = 3

if __name__ == "__main__":
    logger.info(f"Starting HMM regime identification script: {ANALYSIS_NAME}")

    DB_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_OUTPUT_DIR / "summaries").mkdir(parents=True, exist_ok=True)
    (RESULTS_OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

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

    analyzer = HMMRegimeAnalyzer(
        db_manager=db_manager,
        model_output_dir=MODELS_OUTPUT_DIR,
        results_output_dir=RESULTS_OUTPUT_DIR,
        n_hmm_states=N_HMM_STATES,
    )

    try:
        analyzer.analyze_and_save(feature_columns=FEATURES_TO_USE, analysis_name=ANALYSIS_NAME)
        logger.info(f"HMM analysis '{ANALYSIS_NAME}' completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during HMM analysis '{ANALYSIS_NAME}': {e}", exc_info=True)

    logger.info(f"Script {Path(__file__).name} finished.")
