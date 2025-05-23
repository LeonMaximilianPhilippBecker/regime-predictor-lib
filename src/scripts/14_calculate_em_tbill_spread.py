import logging
import sys
from pathlib import Path

from regime_predictor_lib.data_processing.credit_spread_calculator import CreditSpreadCalculator
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EM_YIELD_SERIES_ID = "BAMLEMCBPIEY"
TBILL_YIELD_SERIES_ID = "DTB3"

EM_YIELD_CSV_FILENAME = f"fred_{EM_YIELD_SERIES_ID}_raw_vintage.csv"
TBILL_YIELD_CSV_FILENAME = f"fred_{TBILL_YIELD_SERIES_ID}_raw_vintage.csv"

TARGET_TABLE_NAME = "em_corporate_vs_tbill_spread"

if __name__ == "__main__":
    RAW_DATA_DIR = PROJECT_ROOT_PATH / "data" / "raw"
    DB_DIR = PROJECT_ROOT_PATH / "data" / "db" / "volume"
    DB_PATH = DB_DIR / "quant.db"
    SCHEMA_PATH = PROJECT_ROOT_PATH / "data" / "db" / "schema.sql"

    em_csv_path = RAW_DATA_DIR / EM_YIELD_CSV_FILENAME
    tbill_csv_path = RAW_DATA_DIR / TBILL_YIELD_CSV_FILENAME

    if not em_csv_path.exists() or not tbill_csv_path.exists():
        logger.error(
            f"One or both raw vintage CSV files not found: "
            f"{em_csv_path.name}, {tbill_csv_path.name}. "
            f"Please run script '13_prepare_raw_fred_em_tbill_yield_csvs.py' first."
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

    logger.info(
        "Starting EM Corporate vs T-Bill spread calculation "
        f"and storage process for table '{TARGET_TABLE_NAME}'."
    )

    calculator = CreditSpreadCalculator(
        high_yield_csv_path=em_csv_path,
        investment_grade_csv_path=tbill_csv_path,
        high_yield_series_id=EM_YIELD_SERIES_ID,
        investment_grade_series_id=TBILL_YIELD_SERIES_ID,
    )

    spread_df = calculator.calculate_spreads()

    if spread_df is not None and not spread_df.empty:
        spread_df.rename(
            columns={
                "high_yield_series_id": "em_yield_series_id",
                "investment_grade_series_id": "tbill_yield_series_id",
            },
            inplace=True,
        )

        try:
            db_manager.upsert_dataframe(
                df=spread_df,
                table_name=TARGET_TABLE_NAME,
                conflict_columns=[
                    "reference_date",
                    "em_yield_series_id",
                    "tbill_yield_series_id",
                ],
            )
            logger.info(
                f"Successfully upserted {len(spread_df)} records into '{TARGET_TABLE_NAME}' table."
            )
        except Exception as e:
            logger.error(
                f"Error upserting EM vs T-Bill spread data into database: {e}",
                exc_info=True,
            )
    elif spread_df is not None and spread_df.empty:
        logger.warning(
            "No EM vs T-Bill spread data calculated (DataFrame is empty). Database not updated."
        )
    else:
        logger.error("EM vs T-Bill spread calculation failed. Database not updated.")

    logger.info("EM Corporate vs T-Bill spread calculation and storage script finished.")
