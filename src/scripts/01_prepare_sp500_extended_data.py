import logging
import sys
from pathlib import Path

import pandas as pd

from regime_predictor_lib.data_ingestion.api_clients import AlphaVantageClient, YFinanceClient
from regime_predictor_lib.data_processing.sp500_historical_data_processor import (
    SP500HistoricalDataProcessor,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    input_csv_start_end = RAW_DATA_DIR / "sp500_ticker_start_end.csv"
    if not input_csv_start_end.exists():
        logger.info(f"Creating dummy {input_csv_start_end.name} for testing.")
        dummy_data = {
            "ticker": [
                "AAPL",
                "MSFT",
                "AAL",
                "NONEXISTENTTICKER",
            ],
            "start_date": ["2020-01-01", "2020-01-01", "2023-01-01", "2023-01-01"],
            "end_date": [pd.NaT, pd.NaT, "2023-06-30", "2023-02-01"],
        }
        df_dummy = pd.DataFrame(dummy_data)
        df_dummy["start_date"] = pd.to_datetime(df_dummy["start_date"])
        df_dummy["end_date"] = pd.to_datetime(df_dummy["end_date"])
        df_dummy.to_csv(input_csv_start_end, index=False)
        logger.info(
            f"Dummy {input_csv_start_end.name} created. Please replace with your actual data."
        )

    yfinance_client = YFinanceClient()
    alphavantage_client = AlphaVantageClient()

    api_clients_to_use = [
        yfinance_client,
        alphavantage_client,
    ]
    processor = SP500HistoricalDataProcessor(
        raw_data_dir=RAW_DATA_DIR,
        processed_data_dir=PROCESSED_DATA_DIR,
        api_clients=api_clients_to_use,
    )
    try:
        output_file = processor.generate_extended_price_data(force_regenerate_all=False)
        logger.info(f"Extended price data generation process finished. Output: {output_file}")
    except FileNotFoundError:
        logger.error("Input file for historical data processor not found. Aborting.")
    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
    except Exception as e:
        logger.error(f"An error occurred during historical data processing: {e}", exc_info=True)
