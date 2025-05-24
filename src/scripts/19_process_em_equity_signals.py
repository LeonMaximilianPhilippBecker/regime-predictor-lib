import logging
import sys
from pathlib import Path

from regime_predictor_lib.data_ingestion.api_clients import YFinanceClient
from regime_predictor_lib.data_processing.emerging_market_equity_calculator import (
    EmergingMarketEquityCalculator,
)
from regime_predictor_lib.utils.database_manager import DatabaseManager

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EM_SYMBOL = "EEM"  # MSCI Emerging Markets ETF
MARKET_SYMBOL = "SPY"  # S&P 500 ETF for comparison
START_DATE = "2003-01-01"

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DB_DIR = PROJECT_ROOT / "data" / "db" / "volume"
    DB_PATH = DB_DIR / "quant.db"

    db_manager = DatabaseManager(db_path=DB_PATH)
    yf_client = YFinanceClient()

    calculator = EmergingMarketEquityCalculator(
        yf_client=yf_client,
        db_manager=db_manager,
        em_symbol=EM_SYMBOL,
        market_symbol=MARKET_SYMBOL,
    )

    logger.info(
        f"Starting Emerging Market Equity signal processing for {EM_SYMBOL} vs {MARKET_SYMBOL}"
    )
    try:
        signals_df = calculator.calculate_signals(start_date=START_DATE)
        if signals_df is not None and not signals_df.empty:
            db_manager.upsert_dataframe(
                df=signals_df,
                table_name="em_equity_signals",
                conflict_columns=["date", "symbol"],
            )
            logger.info(f"Successfully processed and stored EM Equity signals for {EM_SYMBOL}.")
        else:
            logger.warning(f"No EM Equity signals calculated for {EM_SYMBOL}.")
    except Exception as e:
        logger.error(
            f"An error occurred during EM Equity signal processing for {EM_SYMBOL}: {e}",
            exc_info=True,
        )
    logger.info("Script 19_process_em_equity_signals.py finished.")
