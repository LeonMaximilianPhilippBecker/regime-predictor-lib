import logging
import sys
from pathlib import Path

import pandas as pd

from regime_predictor_lib.data_ingestion.api_clients import YFinanceClient
from regime_predictor_lib.data_processing.intermarket_analyzer import IntermarketAnalyzer
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

SIGNALS_START_DATE = "2000-01-01"


INTERMARKET_RATIOS_CONFIG = [
    {
        "numerator_ticker": "SPY",
        "denominator_ticker": "TLT",
        "ratio_name": "SPY_TLT_RATIO",
        "start_date_signals": "2003-01-01",
    },
    {
        "numerator_ticker": "GLD",
        "denominator_ticker": "SLV",
        "ratio_name": "GOLD_SILVER_RATIO",
        "start_date_signals": "2007-01-01",
    },
    {
        "numerator_ticker": "CPER",
        "denominator_ticker": "GLD",
        "ratio_name": "COPPER_GOLD_RATIO",
        "start_date_signals": "2012-01-01",
    },
]

STOCK_BOND_RETURN_DIFF_CONFIG = {
    "stock_ticker": "SPY",
    "bond_ticker": "TLT",
    "start_date_signals": "2003-01-01",
}


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

    yf_client = YFinanceClient()
    analyzer = IntermarketAnalyzer(yf_client=yf_client)
    current_end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    logger.info("Processing Intermarket Stock-Bond Return Difference...")
    sb_config = STOCK_BOND_RETURN_DIFF_CONFIG
    stock_bond_df = analyzer.calculate_stock_bond_return_difference(
        stock_ticker=sb_config["stock_ticker"],
        bond_ticker=sb_config["bond_ticker"],
        start_date_signals=sb_config["start_date_signals"],
        end_date=current_end_date,
    )

    if stock_bond_df is not None and not stock_bond_df.empty:
        try:
            db_manager.upsert_dataframe(
                df=stock_bond_df,
                table_name="intermarket_stock_bond_returns_diff",
                conflict_columns=["date"],
            )
            logger.info("Successfully upserted Stock-Bond Return Difference data to the database.")
        except Exception as e:
            logger.error(f"Error upserting Stock-Bond Return Difference data: {e}", exc_info=True)
    else:
        logger.warning("No Stock-Bond Return Difference data calculated.")

    logger.info("Processing Intermarket Ratios...")
    for config in INTERMARKET_RATIOS_CONFIG:
        logger.info(
            f"Calculating ratio: {config['ratio_name']} "
            f"({config['numerator_ticker']}/{config['denominator_ticker']})"
        )
        ratio_df = analyzer.calculate_generic_ratio(
            numerator_ticker=config["numerator_ticker"],
            denominator_ticker=config["denominator_ticker"],
            ratio_name=config["ratio_name"],
            start_date_signals=config["start_date_signals"],
            end_date=current_end_date,
        )

        if ratio_df is not None and not ratio_df.empty:
            try:
                db_manager.upsert_dataframe(
                    df=ratio_df,
                    table_name="intermarket_ratios",
                    conflict_columns=["date", "ratio_name"],
                )
                logger.info(f"Successfully upserted {config['ratio_name']} data to the database.")
            except Exception as e:
                logger.error(f"Error upserting {config['ratio_name']} data: {e}", exc_info=True)
        else:
            logger.warning(f"No data calculated for ratio: {config['ratio_name']}.")

    logger.info("Script 25_process_intermarket_relationships.py finished.")
