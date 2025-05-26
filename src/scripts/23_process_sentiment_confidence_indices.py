import logging
import sys
from pathlib import Path

from regime_predictor_lib.data_ingestion.smci_dmci_json_ingestor import SmciDmciJsonIngestor
from regime_predictor_lib.data_processing.sentiment_confidence_calculator import (
    SentimentConfidenceCalculator,
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
SMCI_DMCI_JSON_PATH = PROJECT_ROOT_PATH / "data" / "raw" / "smcidmci.json"

if __name__ == "__main__":
    DB_DIR.mkdir(parents=True, exist_ok=True)

    if not SMCI_DMCI_JSON_PATH.exists():
        logger.error(f"SMCI/DMCI JSON file not found at: {SMCI_DMCI_JSON_PATH}")
        logger.error(
            "Please ensure the file 'smcdmca.json' "
            "exists in data/raw/ with the provided JSON content."
        )
        sys.exit(1)

    db_manager = DatabaseManager(db_path=DB_PATH)
    try:
        db_manager.create_tables_from_schema_file(SCHEMA_PATH)
    except FileNotFoundError as e:
        logger.error(f"Schema file not found: {e}. Cannot proceed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error initializing database tables: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Starting SMCI/DMCI data ingestion and processing.")

    ingestor = SmciDmciJsonIngestor(json_filepath=SMCI_DMCI_JSON_PATH)
    raw_sentiment_df = ingestor.load_and_process_data()

    if raw_sentiment_df is None or raw_sentiment_df.empty:
        logger.warning("No raw sentiment data loaded. Exiting.")
        sys.exit(0)

    calculator = SentimentConfidenceCalculator(
        roc_periods_months=[1, 3, 6], percentile_windows_months=[12, 24], sma_windows_months=[3, 6]
    )
    sentiment_signals_df = calculator.calculate_signals(raw_sentiment_df)

    if sentiment_signals_df is not None and not sentiment_signals_df.empty:
        try:
            db_manager.upsert_dataframe(
                df=sentiment_signals_df,
                table_name="sentiment_confidence_indices",
                conflict_columns=["date"],
            )
            logger.info(
                f"Successfully upserted {len(sentiment_signals_df)} records into "
                "'sentiment_confidence_indices' table."
            )
        except Exception as e:
            logger.error(
                f"Error upserting sentiment confidence data into database: {e}",
                exc_info=True,
            )
    else:
        logger.warning("No sentiment confidence signals calculated. DataFrame is empty.")

    logger.info("Script 23_process_sentiment_confidence_indices.py finished.")
