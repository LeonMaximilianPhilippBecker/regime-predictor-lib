import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from regime_predictor_lib.data_ingestion.api_clients import YFinanceClient
from regime_predictor_lib.data_processing.volatility_calculator import VolatilityCalculator
from regime_predictor_lib.utils.database_manager import DatabaseManager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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

    yf_client = YFinanceClient()
    volatility_calculator = VolatilityCalculator(yf_client=yf_client)

    end_date_dt = datetime.now() - timedelta(days=1)
    start_date_dt = datetime(1990, 1, 1)  # Or a more relevant start date

    start_date_str = start_date_dt.strftime("%Y-%m-%d")
    end_date_str = end_date_dt.strftime("%Y-%m-%d")

    logger.info(
        f"Attempting to fetch and calculate volatility indicators "
        f"from {start_date_str} to {end_date_str}."
    )

    try:
        volatility_df = volatility_calculator.fetch_and_calculate_volatility_metrics(
            start_date=start_date_str, end_date=end_date_str
        )

        if volatility_df is not None and not volatility_df.empty:
            logger.info(
                f"Volatility data processed. Shape: {volatility_df.shape}. "
                f"Upserting to database..."
            )
            conflict_cols = ["date"]
            db_manager.upsert_dataframe(
                df=volatility_df,
                table_name="volatility_indicators",
                conflict_columns=conflict_cols,
            )
            logger.info("Volatility indicators successfully upserted to the database.")
        else:
            logger.warning("No volatility data was processed or returned. Database not updated.")

    except Exception as e:
        logger.error(
            f"An error occurred during volatility indicator processing: {e}",
            exc_info=True,
        )

    logger.info("Volatility indicator fetching process finished.")
