import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class SmciDmciJsonIngestor:
    def __init__(self, json_filepath: Path | str):
        self.json_filepath = Path(json_filepath)
        logger.info(f"SmciDmciJsonIngestor initialized for file: {self.json_filepath}")

    def _parse_series_data(self, series_data: list, value_col_name: str) -> pd.DataFrame:
        if not series_data or not isinstance(series_data, list):
            logger.warning(f"Empty or invalid data provided for {value_col_name}.")
            return pd.DataFrame(columns=["date", value_col_name]).set_index("date")

        df = pd.DataFrame(series_data, columns=["date_orig", value_col_name])
        try:
            df["date"] = pd.to_datetime(df["date_orig"])
            df.set_index("date", inplace=True)
            df.drop(columns=["date_orig"], inplace=True)
            # Ensure value column is numeric
            df[value_col_name] = pd.to_numeric(df[value_col_name], errors="coerce")
        except Exception as e:
            logger.error(f"Error processing data for {value_col_name}: {e}")
            return pd.DataFrame(columns=[value_col_name]).set_index(pd.to_datetime([]))
        return df[[value_col_name]]

    def load_and_process_data(self) -> pd.DataFrame | None:
        if not self.json_filepath.exists():
            logger.error(f"SMCI/DMCI JSON file not found: {self.json_filepath}")
            return None

        try:
            with open(self.json_filepath, "r") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.json_filepath}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading SMCI/DMCI JSON {self.json_filepath}: {e}")
            return None

        if not isinstance(raw_data, list) or len(raw_data) != 3:
            logger.error(
                f"JSON data in {self.json_filepath} is not a list of 3 arrays as expected."
            )
            return None

        df_smci_pct_dmci = self._parse_series_data(raw_data[0], "smci_pct_dmci_ratio")
        df_smci = self._parse_series_data(raw_data[1], "smci_value")
        df_dmci = self._parse_series_data(raw_data[2], "dmci_value")

        # Merge the DataFrames
        merged_df = df_smci_pct_dmci.join(df_smci, how="outer").join(df_dmci, how="outer")

        if merged_df.empty:
            logger.warning("Merged SMCI/DMCI DataFrame is empty.")
            return None

        # Data is monthly, ensure dates are consistent (e.g., month end)
        # Assuming dates in JSON are already start/end of month.
        # If they need to be normalized, do it here.
        # Example: merged_df.index = merged_df.index + pd.offsets.MonthEnd(0)

        merged_df.sort_index(inplace=True)
        logger.info(f"Successfully loaded and merged SMCI/DMCI data. Shape: {merged_df.shape}")
        return merged_df.reset_index()  # Return with date as column for calculator
