import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


PROJECT_ROOT = get_project_root()
DB_FILE = PROJECT_ROOT / "data" / "db" / "volume" / "quant.db"
CSV_FILE_PATH = PROJECT_ROOT / "data" / "raw" / "cboe_equity_pc_ratios.csv"
CBOE_DAILY_STATS_URL = "https://www.cboe.com/us/options/market_statistics/daily/?dt={date_str}"

engine = None
SessionLocal = None


def init_db():
    global engine, SessionLocal
    if engine is None:
        engine = create_engine(f"sqlite:///{DB_FILE}")
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def clean_numeric_value(value_str):
    if value_str is None:
        return None
    if isinstance(value_str, (int, float)):
        return value_str
    s = str(value_str).strip()
    if not s or s == "-":
        return None
    s = s.replace(".", "")
    s = s.replace(",", "")
    try:
        return float(s) if "." in str(value_str) else int(s)
    except ValueError:
        logger.warning(f"Could not convert '{value_str}' to numeric value.")
        return None


def insert_pcr_data(session, date_val, call_vol, put_vol, total_vol, pc_ratio, source):
    if isinstance(date_val, (datetime, pd.Timestamp)):
        date_str = date_val.strftime("%Y-%m-%d")
    else:
        date_str = date_val

    call_vol_clean = clean_numeric_value(call_vol)
    put_vol_clean = clean_numeric_value(put_vol)
    total_vol_clean = clean_numeric_value(total_vol)
    pc_ratio_clean = clean_numeric_value(pc_ratio)

    if pc_ratio_clean is None and call_vol_clean is not None and put_vol_clean is not None:
        if call_vol_clean > 0:
            pc_ratio_clean = put_vol_clean / call_vol_clean
        else:
            pc_ratio_clean = None

    if date_str is None or pc_ratio_clean is None:
        logger.warning(f"Skipping insertion for date {date_str} due to missing critical data.")
        return

    try:
        existing_record = session.execute(
            text("SELECT id FROM put_call_ratios WHERE date = :date"),
            {"date": date_str},
        ).fetchone()

        if existing_record:
            session.execute(
                text(
                    """
                    UPDATE put_call_ratios
                    SET equity_call_volume = :call_vol,
                        equity_put_volume = :put_vol,
                        equity_total_volume = :total_vol,
                        equity_pc_ratio = :pc_ratio,
                        source = :source,
                        created_at = CURRENT_TIMESTAMP
                    WHERE date = :date
                """
                ),
                {
                    "date": date_str,
                    "call_vol": call_vol_clean,
                    "put_vol": put_vol_clean,
                    "total_vol": total_vol_clean,
                    "pc_ratio": pc_ratio_clean,
                    "source": source,
                },
            )
            logger.debug(f"Updated PCR data for {date_str} from {source}")
        else:
            session.execute(
                text(
                    """
                    INSERT INTO put_call_ratios
                    (date, equity_call_volume, equity_put_volume,
                    equity_total_volume, equity_pc_ratio, source)
                    VALUES (:date, :call_vol, :put_vol, :total_vol, :pc_ratio, :source)
                """
                ),
                {
                    "date": date_str,
                    "call_vol": call_vol_clean,
                    "put_vol": put_vol_clean,
                    "total_vol": total_vol_clean,
                    "pc_ratio": pc_ratio_clean,
                    "source": source,
                },
            )
            logger.debug(f"Inserted PCR data for {date_str} from {source}")
        session.commit()
    except IntegrityError:
        session.rollback()
        logger.error(f"IntegrityError for date {date_str}. This shouldn't happen.")
    except Exception as e:
        session.rollback()
        logger.error(f"Error inserting/updating data for {date_str}: {e}")


def process_cboe_csv(csv_filepath: Path):
    logger.info(f"Processing CSV file: {csv_filepath}")
    if not csv_filepath.exists():
        logger.error(f"CSV file not found at {csv_filepath}")
        return

    try:
        with open(csv_filepath, "r", encoding="utf-8", errors="surrogateescape") as f:
            lines = f.readlines()

        header_row_index = -1
        for i, line in enumerate(lines):
            if "DATE,CALL,PUT,TOTAL,P/C Ratio" in line:
                header_row_index = i
                break

        if header_row_index == -1:
            logger.error("Could not find the header row in the CSV.")
            return

        df = pd.read_csv(
            csv_filepath,
            skiprows=header_row_index,
            encoding="utf-8",
            encoding_errors="surrogateescape",
        )

        expected_cols = ["DATE", "CALL", "PUT", "TOTAL", "P/C Ratio"]
        if not all(col in df.columns for col in expected_cols):
            logger.error(f"CSV missing expected columns. Found: {df.columns.tolist()}")
            return

        df.rename(
            columns={
                "DATE": "date",
                "CALL": "equity_call_volume",
                "PUT": "equity_put_volume",
                "TOTAL": "equity_total_volume",
                "P/C Ratio": "equity_pc_ratio",
            },
            inplace=True,
        )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)

        numeric_cols = [
            "equity_call_volume",
            "equity_put_volume",
            "equity_total_volume",
            "equity_pc_ratio",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )

        session = init_db()
        for _, row in df.iterrows():
            if pd.isna(row["date"]):
                continue
            insert_pcr_data(
                session,
                row["date"],
                row["equity_call_volume"],
                row["equity_put_volume"],
                row["equity_total_volume"],
                row["equity_pc_ratio"],
                "cboe_csv_equitypc",
            )
        session.close()
        logger.info("Finished processing CSV file.")

    except Exception as e:
        logger.error(f"Error processing CSV file {csv_filepath}: {e}", exc_info=True)


def parse_cboe_html_for_equity_options(html_content: str):
    soup = BeautifulSoup(html_content, "lxml")

    equity_options_table = None
    all_tables = soup.find_all("table")
    for table in all_tables:
        header = table.find("th", colspan="4")
        if header and "EQUITY OPTIONS" in header.get_text(strip=True):
            equity_options_table = table
            break

    if not equity_options_table:
        logger.warning("Could not find 'EQUITY OPTIONS' table on the page.")
        return None

    volume_row = None
    for row in equity_options_table.find_all("tr"):
        cells = row.find_all("td")
        if cells and "VOLUME" in cells[0].get_text(strip=True):
            volume_row = cells
            break

    if not volume_row or len(volume_row) < 4:
        logger.warning("Could not find 'VOLUME' row or not enough cells in 'EQUITY OPTIONS' table.")
        return None

    try:
        call_volume_str = volume_row[1].get_text(strip=True)
        put_volume_str = volume_row[2].get_text(strip=True)
        total_volume_str = volume_row[3].get_text(strip=True)

        call_volume = clean_numeric_value(call_volume_str)
        put_volume = clean_numeric_value(put_volume_str)
        total_volume = clean_numeric_value(total_volume_str)

        pc_ratio = None
        if call_volume is not None and put_volume is not None and call_volume > 0:
            pc_ratio = put_volume / call_volume
        elif call_volume == 0 and put_volume > 0:
            pc_ratio = float("inf")

        return call_volume, put_volume, total_volume, pc_ratio
    except Exception as e:
        logger.error(f"Error parsing volume data from HTML: {e}")
        return None


def scrape_cboe_website(start_date: datetime, end_date: datetime):
    logger.info(f"Starting CBOE website scraping from {start_date.date()} to {end_date.date()}")

    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = uc.Chrome(options=options)

    current_date = start_date
    session = init_db()

    while current_date <= end_date:
        date_str_url = current_date.strftime("%Y-%m-%d")
        if current_date.weekday() >= 5:
            logger.info(f"Skipping weekend: {date_str_url}")
            current_date += timedelta(days=1)
            continue

        url = CBOE_DAILY_STATS_URL.format(date_str=date_str_url)
        logger.info(f"Fetching data for {date_str_url} from {url}")

        try:
            driver.get(url)
            time.sleep(2)

            html_content = driver.page_source
            parsed_data = parse_cboe_html_for_equity_options(html_content)

            if parsed_data:
                call_vol, put_vol, total_vol, pc_ratio_calc = parsed_data
                insert_pcr_data(
                    session,
                    current_date,
                    call_vol,
                    put_vol,
                    total_vol,
                    pc_ratio_calc,
                    "cboe_web_equity_options",
                )
                logger.info(f"Successfully processed and stored data for {date_str_url}")
            else:
                logger.warning(
                    f"No data parsed {date_str_url}. " "Might be holiday or no data available."
                )

        except Exception as e:
            logger.error(f"Error scraping data for {date_str_url}: {e}", exc_info=True)

        current_date += timedelta(days=1)
        time.sleep(1)

    driver.quit()
    session.close()
    logger.info("Finished CBOE website scraping.")


if __name__ == "__main__":
    CSV_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Database file will be at: {DB_FILE}")
    logger.info(f"CSV file expected at: {CSV_FILE_PATH}")

    process_cboe_csv(CSV_FILE_PATH)

    scrape_start_date = datetime(2019, 10, 11)
    scrape_end_date = datetime(2025, 5, 20)
    scrape_cboe_website(scrape_start_date, scrape_end_date)

    logger.info("All PCR data ingestion tasks complete.")
