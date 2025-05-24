# src/regime_predictor_lib/data_ingestion/bdi_ingestor.py
import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BdiIngestor:
    def __init__(self, csv_source: Union[List[Union[Path, str]], Path, str]):
        self.csv_files: List[Path] = []
        if isinstance(csv_source, list):
            self.csv_files = [Path(p) for p in csv_source]
        elif isinstance(csv_source, (Path, str)):
            source_path = Path(csv_source)
            if source_path.is_dir():
                self.csv_files = sorted(
                    list(source_path.glob("*.csv"))
                )  # Sort to maintain order if important
                if not self.csv_files:
                    logger.warning(f"No CSV files found in directory: {source_path}")
            elif source_path.is_file():
                self.csv_files = [source_path]
            else:
                logger.error(
                    f"CSV source path does not exist or is not a file/directory: {source_path}"
                )
        else:
            raise TypeError(
                "csv_source must be a list of paths, a single path, or a directory path."
            )

        if self.csv_files:
            logger.info(
                f"BdiIngestor initialized. Will process files: {[str(f) for f in self.csv_files]}"
            )
        else:
            logger.warning("BdiIngestor initialized, but no CSV files specified or found.")

    def _clean_price_value(self, value_str):
        if pd.isna(value_str):
            return np.nan
        s = str(value_str).strip().replace(",", "")
        try:
            return float(s)
        except ValueError:
            logger.warning(f"Could not convert BDI price '{value_str}' to float.")
            return np.nan

    def load_and_process_data(self) -> pd.DataFrame | None:
        if not self.csv_files:
            logger.error("No BDI CSV files to process.")
            return None

        all_dfs: List[pd.DataFrame] = []
        for i, csv_filepath in enumerate(self.csv_files):
            if not csv_filepath.exists():
                logger.error(f"BDI CSV file not found: {csv_filepath}")
                continue

            logger.info(f"Loading BDI data from {csv_filepath} ({i+1}/{len(self.csv_files)})")
            try:
                df_part = pd.read_csv(csv_filepath)
            except Exception as e:
                logger.error(f"Error loading BDI CSV {csv_filepath}: {e}")
                continue

            if "Date" not in df_part.columns or "Price" not in df_part.columns:
                logger.error(
                    f"CSV file {csv_filepath} is missing 'Date' or 'Price' column. Skipping."
                )
                continue

            df_part = df_part[["Date", "Price"]].copy()
            df_part.rename(columns={"Date": "date", "Price": "value"}, inplace=True)

            try:
                df_part["date"] = pd.to_datetime(
                    df_part["date"], format="%m/%d/%Y", errors="coerce"
                )
            except Exception as e:
                logger.error(f"Error converting 'date' column to datetime in {csv_filepath}: {e}")
                df_part.dropna(subset=["date"], inplace=True)
                if df_part.empty:
                    continue

            df_part["value"] = df_part["value"].apply(self._clean_price_value)
            df_part.dropna(subset=["value"], inplace=True)

            if not df_part.empty:
                all_dfs.append(df_part)
            else:
                logger.warning(f"No valid data after processing {csv_filepath}")

        if not all_dfs:
            logger.error("No data loaded from any BDI CSV files.")
            return None

        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(
            f"Combined BDI data from {len(all_dfs)} files. Initial "
            f"combined shape: {combined_df.shape}"
        )

        combined_df.dropna(subset=["date", "value"], inplace=True)

        combined_df.sort_values(by="date", inplace=True)
        combined_df.drop_duplicates(subset=["date"], keep="first", inplace=True)

        combined_df["date"] = combined_df["date"].dt.strftime("%Y-%m-%d")

        final_cols = ["date", "value"]
        combined_df = combined_df[final_cols]

        logger.info(f"Processed final BDI data. Shape: {combined_df.shape}")
        return combined_df
