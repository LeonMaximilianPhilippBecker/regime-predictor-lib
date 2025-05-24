import logging
import sys
from pathlib import Path

from regime_predictor_lib.data_processing.baltic_dry_index_calculator import (
    BalticDryIndexCalculator,
)
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OIL_SERIES_FOR_BDI_RATIO = "DCOILWTICO"

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DB_DIR = PROJECT_ROOT / "data" / "db" / "volume"
    DB_PATH = DB_DIR / "quant.db"

    db_manager = DatabaseManager(db_path=DB_PATH)
    calculator = BalticDryIndexCalculator(
        db_manager=db_manager, oil_series_for_ratio=OIL_SERIES_FOR_BDI_RATIO
    )

    logger.info(f"Starting BDI signal processing. Oil for ratio: {OIL_SERIES_FOR_BDI_RATIO}")
    try:
        signals_df = calculator.calculate_signals()
        if signals_df is not None and not signals_df.empty:
            db_manager.upsert_dataframe(
                df=signals_df,
                table_name="bdi_signals",
                conflict_columns=["date"],
            )
            logger.info("Successfully processed and stored BDI signals.")
        else:
            logger.warning("No BDI signals calculated.")
    except Exception as e:
        logger.error(
            f"An error occurred during BDI signal processing: {e}",
            exc_info=True,
        )
    logger.info("Script 21_process_bdi_signals.py finished.")
