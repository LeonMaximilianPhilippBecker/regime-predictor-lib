import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class GexCsvIngestor:
    def __init__(self, csv_filepath: Path | str):
        self.csv_filepath = Path(csv_filepath)
        logger.info(f"GexCsvIngestor initialized for file: {self.csv_filepath}")

    def load_data(self) -> pd.DataFrame | None:
        if not self.csv_filepath.exists():
            logger.error(f"GEX CSV file not found: {self.csv_filepath}")
            return None

        try:
            df = pd.read_csv(self.csv_filepath, usecols=["date", "gex"])
            logger.info(f"Successfully loaded GEX data from {self.csv_filepath}")
        except ValueError as e:
            logger.error(
                f"Error loading GEX CSV {self.csv_filepath}. "
                f"Ensure 'date' and 'gex' columns exist: {e}"
            )
            return None
        except Exception as e:
            logger.error(f"Error loading GEX CSV {self.csv_filepath}: {e}")
            return None

        if df.empty:
            logger.warning(f"GEX data CSV is empty: {self.csv_filepath}")
            return pd.DataFrame(columns=["date", "gex_value"])

        df.rename(columns={"gex": "gex_value"}, inplace=True)

        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            logger.error(f"Error converting 'date' column to datetime: {e}")
            return None

        df["gex_value"] = pd.to_numeric(df["gex_value"], errors="coerce")
        df.dropna(subset=["date", "gex_value"], inplace=True)

        df.sort_values(by="date", inplace=True)
        df.set_index("date", inplace=True)

        logger.info(f"Processed GEX data. Shape: {df.shape}")
        return df
