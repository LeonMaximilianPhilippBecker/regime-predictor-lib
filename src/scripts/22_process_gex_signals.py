import logging
import sys
from pathlib import Path

from regime_predictor_lib.data_ingestion.gex_csv_ingestor import GexCsvIngestor
from regime_predictor_lib.data_processing.gex_calculator import GexCalculator
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GEX_CSV_FILENAME = "spx_gamma_exposure.csv"

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DB_DIR = PROJECT_ROOT / "data" / "db" / "volume"
    DB_PATH = DB_DIR / "quant.db"
    SCHEMA_PATH = PROJECT_ROOT / "data" / "db" / "schema.sql"
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    GEX_CSV_PATH = RAW_DATA_DIR / GEX_CSV_FILENAME

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

    if not GEX_CSV_PATH.exists():
        logger.error(
            f"GEX data file not found at {GEX_CSV_PATH}. "
            "Please ensure the file exists and is correctly named."
        )
        sys.exit(1)

    logger.info(f"Ingesting GEX data from: {GEX_CSV_PATH}")
    gex_ingestor = GexCsvIngestor(csv_filepath=GEX_CSV_PATH)
    raw_gex_df = gex_ingestor.load_data()

    if raw_gex_df is None or raw_gex_df.empty:
        logger.warning("No raw GEX data loaded. Exiting.")
        sys.exit(0)

    logger.info("Calculating GEX signals...")
    gex_calculator = GexCalculator()
    gex_signals_df = gex_calculator.calculate_signals(raw_gex_df)

    if gex_signals_df is not None and not gex_signals_df.empty:
        logger.info(
            f"GEX signals calculated. Shape: {gex_signals_df.shape}. " f"Upserting to database..."
        )
        try:
            db_manager.upsert_dataframe(
                df=gex_signals_df,
                table_name="gex_signals",
                conflict_columns=["date"],
            )
            logger.info("GEX signals successfully upserted to the database.")
        except Exception as e:
            logger.error(f"Error upserting GEX signals into database: {e}", exc_info=True)
    else:
        logger.warning("No GEX signals were calculated or data was empty. Database not updated.")

    logger.info("Script 22_process_gex_signals.py finished.")
