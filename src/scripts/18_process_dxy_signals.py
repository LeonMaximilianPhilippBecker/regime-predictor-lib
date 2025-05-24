import logging
import sys
from pathlib import Path

from regime_predictor_lib.data_processing.dxy_calculator import DxyCalculator
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DXY_SERIES_TO_PROCESS = "DTWEXBGS"

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DB_DIR = PROJECT_ROOT / "data" / "db" / "volume"
    DB_PATH = DB_DIR / "quant.db"

    db_manager = DatabaseManager(db_path=DB_PATH)
    calculator = DxyCalculator(db_manager=db_manager, dxy_series_id=DXY_SERIES_TO_PROCESS)

    logger.info(f"Starting DXY signal processing for series: {DXY_SERIES_TO_PROCESS}")
    try:
        signals_df = calculator.calculate_signals()
        if signals_df is not None and not signals_df.empty:
            db_manager.upsert_dataframe(
                df=signals_df,
                table_name="dxy_signals",
                conflict_columns=[
                    "date",
                    "series_id",
                ],
            )
            logger.info(
                f"Successfully processed and stored DXY signals for {DXY_SERIES_TO_PROCESS}."
            )
        else:
            logger.warning(f"No DXY signals calculated for {DXY_SERIES_TO_PROCESS}.")
    except Exception as e:
        logger.error(
            f"An error occurred during DXY signal processing for {DXY_SERIES_TO_PROCESS}: {e}",
            exc_info=True,
        )
    logger.info("Script 18_process_dxy_signals.py finished.")
