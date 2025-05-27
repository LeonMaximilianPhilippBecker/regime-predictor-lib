import logging
import sys
from pathlib import Path

import pandas as pd

from regime_predictor_lib.data_ingestion.api_clients import YFinanceClient
from regime_predictor_lib.data_processing.sp500_derived_indicator_calculator import (
    SP500DerivedIndicatorCalculator,
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

SP500_TICKER = "^GSPC"
HISTORICAL_START_DATE = "1985-01-01"

if __name__ == "__main__":
    DB_DIR.mkdir(parents=True, exist_ok=True)

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

    logger.info(f"Starting S&P 500 ({SP500_TICKER}) derived indicators calculation and processing.")

    yf_client = YFinanceClient()
    calculator = SP500DerivedIndicatorCalculator(yf_client=yf_client)

    current_end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    derived_indicators_df = calculator.calculate_indicators(
        start_date=HISTORICAL_START_DATE, end_date=current_end_date
    )

    if derived_indicators_df is not None and not derived_indicators_df.empty:
        try:
            db_manager.upsert_dataframe(
                df=derived_indicators_df,
                table_name="sp500_derived_indicators",
                conflict_columns=["date"],
            )
            logger.info(
                f"Successfully upserted {len(derived_indicators_df)} "
                "records for S&P 500 derived indicators "
                "into 'sp500_derived_indicators' table."
            )
        except Exception as e:
            logger.error(
                f"Error upserting S&P 500 derived indicators data into database: {e}",
                exc_info=True,
            )
    else:
        logger.warning("No S&P 500 derived indicators calculated. DataFrame is empty.")

    logger.info("Script 26_process_sp500_derived_indicators.py finished.")
