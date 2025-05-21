import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from regime_predictor_lib.data_ingestion.api_clients import YFinanceClient
from regime_predictor_lib.data_processing.technical_indicator_calculator import (
    TechnicalIndicatorCalculator,
)
from regime_predictor_lib.utils.database_manager import DatabaseManager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INDEX_TICKER = "^GSPC"
HISTORICAL_START_DATE = "1984-01-01"

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DB_DIR = PROJECT_ROOT / "data" / "db" / "volume"
    DB_PATH = DB_DIR / "quant.db"
    SCHEMA_PATH = PROJECT_ROOT / "data" / "db" / "schema.sql"

    DB_DIR.mkdir(parents=True, exist_ok=True)

    db_manager = DatabaseManager(db_path=DB_PATH)
    try:
        db_manager.create_tables_from_schema_file(SCHEMA_PATH)
    except FileNotFoundError as e:
        logger.error(f"Schema file not found: {e}. Cannot proceed to create tables.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error initializing database tables: {e}")
        sys.exit(1)

    calculator = TechnicalIndicatorCalculator()
    yf_client = YFinanceClient()

    end_date_dt = datetime.now() - timedelta(days=1)
    end_date_str = end_date_dt.strftime("%Y-%m-%d")

    logger.info(
        f"Fetching OHLCV data for index {INDEX_TICKER} "
        f"from {HISTORICAL_START_DATE} to {end_date_str}."
    )

    index_ohlcv_df_raw = yf_client.fetch_ohlcv_data(
        symbol=INDEX_TICKER, start_date=HISTORICAL_START_DATE, end_date=end_date_str
    )

    if index_ohlcv_df_raw is None or index_ohlcv_df_raw.empty:
        logger.error(f"Could not fetch OHLCV data for index {INDEX_TICKER}. Aborting.")
        sys.exit(1)

    try:
        index_ohlcv_df_raw["date"] = pd.to_datetime(index_ohlcv_df_raw["date"])
        index_ohlcv_df = index_ohlcv_df_raw.set_index("date")
    except KeyError:
        logger.error("Fetched OHLCV data does not have a 'date' column. Aborting.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error processing date column in fetched OHLCV data: {e}. Aborting.")
        sys.exit(1)

    if len(index_ohlcv_df) < 250:
        logger.warning(
            f"Insufficient data for {INDEX_TICKER} (rows: {len(index_ohlcv_df)} "
            f"after fetching from {HISTORICAL_START_DATE}). "
            "Some indicators might be NaN or missing."
        )

    logger.info(f"Calculating technical indicators for {INDEX_TICKER}...")
    try:
        indicators_df = calculator.calculate_all_indicators(
            ohlcv_df=index_ohlcv_df.copy(),
            symbol=INDEX_TICKER,
        )

        if indicators_df is not None and not indicators_df.empty:
            logger.info(
                f"Technical indicators calculated for {INDEX_TICKER}. Shape: {indicators_df.shape}."
                f"Upserting to database..."
            )
            db_manager.upsert_dataframe(
                df=indicators_df,
                table_name="technical_indicators",
                conflict_columns=["symbol", "date"],
            )
            logger.info(
                f"Successfully upserted technical indicators for {INDEX_TICKER} to the database."
            )
        else:
            logger.warning(f"No indicators calculated for {INDEX_TICKER}. Database not updated.")

    except Exception as e:
        logger.error(
            f"An error occurred during technical indicator processing for {INDEX_TICKER}: {e}",
            exc_info=True,
        )

    logger.info("Technical indicators calculation and storage process finished.")
