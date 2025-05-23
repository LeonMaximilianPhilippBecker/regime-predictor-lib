import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from regime_predictor_lib.data_ingestion.api_clients import YFinanceClient
from regime_predictor_lib.data_processing.relative_strength_calculator import (
    RelativeStrengthCalculator,
)
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


COMPARISON_PAIRS = [
    ("^DJT", "^GSPC", "DJT_vs_GSPC"),  # Dow Jones Transport vs S&P 500
    ("^RUT", "^GSPC", "RUT_vs_GSPC"),  # Russell 2000 vs S&P 500
    ("QQQ", "^DJU", "QQQ_vs_DJU"),  # Nasdaq 100 vs Dow Jones Utilities
    ("XLV", "^GSPC", "XLV_vs_GSPC"),  # S&P 500 Health Sector vs S&P 500
]

HISTORICAL_START_DATE = "1987-01-01"

CALCULATOR_CONFIG = {
    "log_diff_deltas": [1, 5, 20],
    "return_period_for_spread": 1,
    "z_score_rolling_windows": [20, 60],
}

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
        logger.error(f"Error initializing database tables: {e}", exc_info=True)
        sys.exit(1)

    yf_client = YFinanceClient()
    end_date_str = (datetime.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    for ticker_a, ticker_b, pair_name in COMPARISON_PAIRS:
        logger.info(
            f"Processing pair: {pair_name} ({ticker_a} vs {ticker_b}) "
            f"from {HISTORICAL_START_DATE} to {end_date_str}"
        )

        data_a_raw = yf_client.fetch_ohlcv_data(
            symbol=ticker_a, start_date=HISTORICAL_START_DATE, end_date=end_date_str
        )
        if data_a_raw is None or data_a_raw.empty:
            logger.warning(
                f"Could not fetch data for {ticker_a} (Index A of pair {pair_name}). Skipping pair."
            )
            continue

        data_b_raw = yf_client.fetch_ohlcv_data(
            symbol=ticker_b, start_date=HISTORICAL_START_DATE, end_date=end_date_str
        )
        if data_b_raw is None or data_b_raw.empty:
            logger.warning(
                f"Could not fetch data for {ticker_b} (Index B of pair {pair_name}). Skipping pair."
            )
            continue

        try:
            calculator = RelativeStrengthCalculator(
                data_a=data_a_raw, data_b=data_b_raw, config=CALCULATOR_CONFIG
            )
            metrics_df = calculator.calculate_metrics()

            if metrics_df is None or metrics_df.empty:
                logger.warning(f"No metrics calculated for pair {pair_name}. Skipping storage.")
                continue

            metrics_df["comparison_pair"] = pair_name
            metrics_df["index_a_ticker"] = ticker_a
            metrics_df["index_b_ticker"] = ticker_b

            cols_ordered = [
                "date",
                "comparison_pair",
                "index_a_ticker",
                "index_b_ticker",
                "rs_ratio",
            ]
            for d in CALCULATOR_CONFIG["log_diff_deltas"]:
                cols_ordered.append(f"log_diff_{d}d")

            ret_p = CALCULATOR_CONFIG["return_period_for_spread"]
            cols_ordered.append(f"return_spread_{ret_p}d")

            for w in CALCULATOR_CONFIG["z_score_rolling_windows"]:
                cols_ordered.append(f"z_score_spread_{ret_p}d_window{w}d")

            for col in cols_ordered:
                if col not in metrics_df.columns:
                    metrics_df[col] = pd.NA

            final_df_to_store = metrics_df[cols_ordered]

            logger.info(
                f"Metrics calculated for {pair_name}. Shape: {final_df_to_store.shape}. "
                f"Upserting to database..."
            )
            db_manager.upsert_dataframe(
                df=final_df_to_store,
                table_name="relative_strength_metrics",
                conflict_columns=["date", "comparison_pair"],
            )
            logger.info(
                f"Successfully upserted relative strength metrics for {pair_name} to the database."
            )

        except Exception as e:
            logger.error(
                f"An error occurred during metric calculation or storage for pair {pair_name}: {e}",
                exc_info=True,
            )

    logger.info("Relative strength metrics calculation and storage process finished.")
