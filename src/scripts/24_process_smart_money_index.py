import logging
import sys
from pathlib import Path

from regime_predictor_lib.data_ingestion.api_clients import YFinanceClient
from regime_predictor_lib.data_processing.smart_money_index_calculator import (
    SmartMoneyIndexCalculator,
)
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH / "src"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DB_DIR = PROJECT_ROOT_PATH / "data" / "db" / "volume"
DB_PATH = DB_DIR / "quant.db"
SCHEMA_PATH = PROJECT_ROOT_PATH / "data" / "db" / "schema.sql"

SPY_TICKER = "SPY"
SMI_START_DATE = "1993-01-29"
INITIAL_SMI_VALUE = 0.0

if __name__ == "__main__":
    DB_DIR.mkdir(parents=True, exist_ok=True)

    db_manager = DatabaseManager(db_path=DB_PATH)
    try:
        db_manager.create_tables_from_schema_file(SCHEMA_PATH)
    except FileNotFoundError as e:
        logger.error(f"Schema file not found: {e}. Cannot proceed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error initializing database tables: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Starting Smart Money Index (SMI) for {SPY_TICKER} calculation and processing.")

    yf_client = YFinanceClient()

    smi_calculator = SmartMoneyIndexCalculator(
        yf_client=yf_client,
        roc_periods_days=[21, 63, 126],
        percentile_windows_days=[252, 504],
        sma_windows_days=[20, 50, 200],
    )

    smi_signals_df = smi_calculator.calculate_smi_and_signals(
        symbol=SPY_TICKER, start_date=SMI_START_DATE, initial_smi_value=INITIAL_SMI_VALUE
    )

    if smi_signals_df is not None and not smi_signals_df.empty:
        try:
            db_manager.upsert_dataframe(
                df=smi_signals_df,
                table_name="smart_money_index",
                conflict_columns=["date"],
            )
            logger.info(
                f"Successfully upserted {len(smi_signals_df)} records for SMI into "
                "'smart_money_index' table."
            )
        except Exception as e:
            logger.error(
                f"Error upserting SMI data into database: {e}",
                exc_info=True,
            )
    else:
        logger.warning("No SMI signals calculated. DataFrame is empty.")

    logger.info("Script 24_process_smart_money_index.py finished.")
