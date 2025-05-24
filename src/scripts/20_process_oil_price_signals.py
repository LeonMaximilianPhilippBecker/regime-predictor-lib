import logging
import sys
from pathlib import Path

from regime_predictor_lib.data_processing.oil_price_calculator import OilPriceCalculator
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OIL_SERIES_TO_PROCESS = ["DCOILWTICO", "DCOILBRENTEU"]

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DB_DIR = PROJECT_ROOT / "data" / "db" / "volume"
    DB_PATH = DB_DIR / "quant.db"

    db_manager = DatabaseManager(db_path=DB_PATH)

    for series_id in OIL_SERIES_TO_PROCESS:
        logger.info(f"Starting Oil Price signal processing for series: {series_id}")
        calculator = OilPriceCalculator(db_manager=db_manager, oil_series_id=series_id)
        try:
            signals_df = calculator.calculate_signals()
            if signals_df is not None and not signals_df.empty:
                db_manager.upsert_dataframe(
                    df=signals_df,
                    table_name="oil_price_signals",
                    conflict_columns=["date", "symbol"],
                )
                logger.info(f"Successfully processed and stored Oil Price signals for {series_id}.")
            else:
                logger.warning(f"No Oil Price signals calculated for {series_id}.")
        except Exception as e:
            logger.error(
                f"An error occurred during Oil Price signal processing for {series_id}: {e}",
                exc_info=True,
            )
    logger.info("Script 20_process_oil_price_signals.py finished.")
