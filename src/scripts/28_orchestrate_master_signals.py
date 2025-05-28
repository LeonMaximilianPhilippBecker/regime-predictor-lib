import logging
import sys
from pathlib import Path

from regime_predictor_lib.data_processing.master_signal_orchestrator import MasterSignalOrchestrator
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

DB_DIR = PROJECT_ROOT_PATH / "data" / "db" / "volume"
DB_PATH = DB_DIR / "quant.db"
SCHEMA_UPDATES_SQL_PATH = PROJECT_ROOT_PATH / "data" / "db" / "schema_updates_for_signals.sql"

if __name__ == "__main__":
    logger.info("Starting Master Signal Processing Orchestration Script...")

    if not DB_PATH.parent.exists():
        logger.info(f"Creating database directory: {DB_PATH.parent}")
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    db_manager = DatabaseManager(db_path=DB_PATH)
    orchestrator = MasterSignalOrchestrator(db_manager=db_manager)

    try:
        orchestrator.run_all_processing(schema_updates_sql_file=SCHEMA_UPDATES_SQL_PATH)
        logger.info("Master signal processing orchestration finished successfully.")
    except Exception as e:
        logger.error(
            f"An error occurred during master signal processing orchestration: {e}", exc_info=True
        )
