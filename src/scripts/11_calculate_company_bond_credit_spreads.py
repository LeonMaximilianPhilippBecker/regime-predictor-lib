import logging
import sys
from pathlib import Path

from regime_predictor_lib.data_processing.credit_spread_calculator import CreditSpreadCalculator
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HIGH_YIELD_SERIES_ID = "BAMLH0A0HYM2"
INVESTMENT_GRADE_SERIES_ID = "BAMLC0A4CBBB"

HIGH_YIELD_CSV_FILENAME = f"fred_{HIGH_YIELD_SERIES_ID}_raw_vintage.csv"
INVESTMENT_GRADE_CSV_FILENAME = f"fred_{INVESTMENT_GRADE_SERIES_ID}_raw_vintage.csv"

if __name__ == "__main__":
    RAW_DATA_DIR = PROJECT_ROOT_PATH / "data" / "raw"
    DB_DIR = PROJECT_ROOT_PATH / "data" / "db" / "volume"
    DB_PATH = DB_DIR / "quant.db"
    SCHEMA_PATH = PROJECT_ROOT_PATH / "data" / "db" / "schema.sql"

    hy_csv_path = RAW_DATA_DIR / HIGH_YIELD_CSV_FILENAME
    ig_csv_path = RAW_DATA_DIR / INVESTMENT_GRADE_CSV_FILENAME

    if not hy_csv_path.exists() or not ig_csv_path.exists():
        logger.error(
            f"One or both raw vintage CSV files not found: "
            f"{hy_csv_path.name}, {ig_csv_path.name}. "
            f"Please run script '10_prepare_raw_fred_yield_csvs.py' first."
        )
        sys.exit(1)

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

    logger.info("Starting credit spread calculation and storage process.")

    calculator = CreditSpreadCalculator(
        high_yield_csv_path=hy_csv_path,
        investment_grade_csv_path=ig_csv_path,
        high_yield_series_id=HIGH_YIELD_SERIES_ID,
        investment_grade_series_id=INVESTMENT_GRADE_SERIES_ID,
    )

    spread_df = calculator.calculate_spreads()

    if spread_df is not None and not spread_df.empty:
        try:
            db_manager.upsert_dataframe(
                df=spread_df,
                table_name="investment_grade_junk_bond_yield_spread",
                conflict_columns=[
                    "reference_date",
                    "release_date",
                    "high_yield_series_id",
                    "investment_grade_series_id",
                ],
            )
            logger.info(
                f"Successfully upserted {len(spread_df)} records into 'credit_spreads' table."
            )
        except Exception as e:
            logger.error(
                f"Error upserting credit spread data into database: {e}",
                exc_info=True,
            )
    elif spread_df is not None and spread_df.empty:
        logger.warning(
            "No credit spread data calculated (DataFrame is empty). Database not updated."
        )
    else:
        logger.error("Credit spread calculation failed. Databasedon not updated.")

    logger.info("Credit spread calculation and storage script finished.")
