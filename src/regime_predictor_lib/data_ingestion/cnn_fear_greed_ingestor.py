import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CnnFearGreedIngestor:
    def __init__(self, json_filepath: Path | str):
        self.json_filepath = Path(json_filepath)
        logger.info(f"CnnFearGreedIngestor initialized for file: {self.json_filepath}")

    def load_and_process_data(self) -> pd.DataFrame | None:
        if not self.json_filepath.exists():
            logger.error(f"CNN Fear & Greed JSON file not found: {self.json_filepath}")
            return None

        try:
            with open(self.json_filepath, "r") as f:
                data = json.load(f)

            if not isinstance(data, list):
                logger.error(f"JSON data in {self.json_filepath} is not a list as expected.")
                return None

            df = pd.DataFrame(data)
            logger.info(f"Successfully loaded CNN Fear & Greed data from {self.json_filepath}")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.json_filepath}: {e}")
            return None
        except Exception as e:
            logger.error(
                f"Error loading or processing CNN Fear & Greed JSON {self.json_filepath}: {e}"
            )
            return None

        if "date" not in df.columns or "value" not in df.columns:
            logger.error("JSON data must contain 'date' and 'value' keys for each entry.")
            return None

        df.rename(columns={"date": "reference_date_orig", "value": "value_orig"}, inplace=True)

        try:
            df["reference_date"] = pd.to_datetime(df["reference_date_orig"])
        except Exception as e:
            logger.error(f"Error converting 'date' column to datetime: {e}")
            return None

        df["release_date"] = df["reference_date"]

        try:
            df["value"] = pd.to_numeric(df["value_orig"], errors="coerce").astype("Int64")
        except Exception as e:
            logger.error(f"Error converting 'value' column to integer: {e}")
            df["value"] = np.nan

        final_cols = [
            "reference_date",
            "release_date",
            "value",
        ]
        for col in final_cols:
            if col not in df.columns:
                df[col] = np.nan

        df = df[final_cols]

        df["reference_date"] = df["reference_date"].dt.strftime("%Y-%m-%d")
        df["release_date"] = df["release_date"].dt.strftime("%Y-%m-%d")

        df.dropna(subset=["reference_date", "release_date", "value"], inplace=True)

        logger.info(f"Processed CNN Fear & Greed data. Shape: {df.shape}")
        return df
