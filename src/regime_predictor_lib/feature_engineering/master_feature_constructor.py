import logging
from pathlib import Path

import pandas as pd
import sqlalchemy

from regime_predictor_lib.utils.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

DEFAULT_TABLE_FILTERS = {
    "technical_indicators": {"symbol": "^GSPC"},
    "dxy_signals": {"series_id": "DTWEXBGS"},
    "em_equity_signals": {"symbol": "EEM"},
    "oil_price_signals": {"symbol": "DCOILWTICO"},
    "non_farm_payrolls_signals": {"series_id": "PAYEMS"},
    "initial_jobless_claims_signals": {"series_id": "ICSA"},
    "cpi_signals": {"series_id": "CPIAUCSL"},
    "retail_sales_signals": {"series_id": "RSAFS"},
    "m2_money_supply_signals": {"series_id": "M2SL"},
    "housing_starts_signals": {"series_id": "HOUST"},
    "housing_prices_signals": {"series_id": "CSUSHPINSA"},
    "consumer_confidence_signals": {"series_id": "UMCSENT"},
    "investment_grade_junk_bond_yield_spread_signals": {
        "high_yield_series_id": "BAMLH0A0HYM2",
        "investment_grade_series_id": "BAMLC0A4CBBB",
    },
    "em_corporate_vs_tbill_spread": {
        "em_yield_series_id": "BAMLEMCBPIEY",
        "tbill_yield_series_id": "DTB3",
    },
    "corporate_bond_oas": {"series_id": "BAMLC0A0CM"},
}

EXCLUDE_COLS_UNIVERSAL = [
    "id",
    "created_at",
    "date",
    "reference_date",
    "release_date",
    "date_orig",
    "reference_date_orig",
    "year_month_orig",
    "symbol",
    "series_id",
    "comparison_pair",
    "index_a_ticker",
    "index_b_ticker",
    "ratio_name",
    "numerator_ticker",
    "denominator_ticker",
    "high_yield_series_id",
    "investment_grade_series_id",
    "em_yield_series_id",
    "tbill_yield_series_id",
    "oil_symbol_for_ratio",
    "forecast_period",
    "source",
    "spread_type",
    "sp500_open",
    "sp500_high",
    "sp500_low",
    "sp500_close",
    "sp500_adjusted_close",
    "sp500_volume",
    "equity_call_volume",
    "equity_put_volume",
    "equity_total_volume",
]


class MasterFeatureConstructor:
    def __init__(
        self,
        db_manager: DatabaseManager,
        regime_csv_path: str | Path,
        table_filters: dict | None = None,
    ):
        self.db_manager = db_manager
        self.regime_csv_path = Path(regime_csv_path)
        self.table_filters = table_filters or DEFAULT_TABLE_FILTERS
        logger.info(f"MasterFeatureConstructor initialized. Regime data: {self.regime_csv_path}")

    def _get_db_table_columns(self, table_name: str) -> list[str]:
        try:
            with self.db_manager.engine.connect() as connection:
                result = connection.execute(sqlalchemy.text(f"PRAGMA table_info({table_name});"))
                return [row[1] for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Could not get columns for table {table_name}: {e}")
            return []

    def _load_regime_data(self) -> pd.DataFrame | None:
        if not self.regime_csv_path.exists():
            logger.error(f"Regime CSV file not found: {self.regime_csv_path}")
            return None
        try:
            df = pd.read_csv(self.regime_csv_path, parse_dates=["date"])
            df = df[["date", "hmm_state_smoothed"]].copy()
            df.rename(columns={"hmm_state_smoothed": "regime_t"}, inplace=True)
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            df["regime_t_plus_6m"] = df["regime_t"].shift(-126)

            logger.info(f"Loaded and processed regime data. Shape: {df.shape}")
            return df[["regime_t", "regime_t_plus_6m"]]
        except Exception as e:
            logger.error(f"Error loading regime data: {e}", exc_info=True)
            return None

    def _load_and_prefix_data(
        self,
        table_name: str,
        prefix: str,
        date_col_in_source: str,
        filters: dict | None = None,
        is_vintage: bool = False,
        vintage_value_cols: list | None = None,
        extra_vintage_key_cols: list | None = None,
    ) -> pd.DataFrame | None:
        logger.info(f"Loading data from {table_name} with prefix '{prefix}'...")
        try:
            all_db_cols = self._get_db_table_columns(table_name)
            if not all_db_cols:
                logger.warning(f"No columns found for table {table_name}, skipping.")
                return None

            potential_signal_cols = [col for col in all_db_cols if col not in EXCLUDE_COLS_UNIVERSAL]

            if is_vintage:
                if not vintage_value_cols:
                    logger.warning(
                        f"Vintage table {table_name} but no 'vintage_value_cols' provided. Using auto-selected."
                    )
                    actual_signal_cols_from_db = [
                        col
                        for col in potential_signal_cols
                        if col not in ["reference_date", "release_date"] + (extra_vintage_key_cols or [])
                    ]

                else:
                    actual_signal_cols_from_db = [col for col in vintage_value_cols if col in all_db_cols]

                cols_to_fetch_db = ["reference_date", "release_date"] + actual_signal_cols_from_db
                if extra_vintage_key_cols:
                    cols_to_fetch_db.extend(extra_vintage_key_cols)
            else:
                actual_signal_cols_from_db = [col for col in potential_signal_cols if col != date_col_in_source]
                cols_to_fetch_db = [date_col_in_source] + actual_signal_cols_from_db

            cols_to_fetch_db = list(dict.fromkeys(cols_to_fetch_db))

            select_clause = ", ".join(f'"{c}"' for c in cols_to_fetch_db)
            query = f"SELECT {select_clause} FROM {table_name}"

            filter_clauses_sql = []
            if filters:
                for k, v in filters.items():
                    if k in all_db_cols:
                        filter_clauses_sql.append(f"\"{k}\" = '{v}'")
                    else:
                        logger.warning(
                            f"Filter key '{k}' not found in table '{table_name}'. Skipping this filter."
                        )

            if filter_clauses_sql:
                query += " WHERE " + " AND ".join(filter_clauses_sql)

            order_by_date_col = "release_date" if is_vintage else date_col_in_source
            secondary_order_col = ", reference_date DESC" if is_vintage else ""
            query += f' ORDER BY "{order_by_date_col}" ASC{secondary_order_col};'

            df_raw = pd.read_sql_query(
                sqlalchemy.text(query),
                self.db_manager.engine,
                parse_dates=[date_col_in_source] if not is_vintage else ["reference_date", "release_date"],
            )

            if df_raw.empty:
                logger.warning(f"No data loaded for {table_name} with query: {query[:300]}...")
                return None

            df = df_raw.copy()

            if is_vintage:
                df.sort_values(["release_date", "reference_date"], ascending=[True, False], inplace=True)

                key_subset_for_duplicates = ["release_date"]
                if extra_vintage_key_cols:
                    key_subset_for_duplicates.extend(extra_vintage_key_cols)

                df.drop_duplicates(subset=key_subset_for_duplicates, keep="first", inplace=True)

                df.set_index("release_date", inplace=True)
                df.sort_index(inplace=True)

                df.rename(columns={"reference_date": f"{prefix}ref_date"}, inplace=True)

                cols_to_return = [f"{prefix}ref_date"]
                for s_col in actual_signal_cols_from_db:
                    prefixed_name = f"{prefix}{s_col}"
                    df.rename(columns={s_col: prefixed_name}, inplace=True)
                    cols_to_return.append(prefixed_name)

                if extra_vintage_key_cols:
                    df.drop(
                        columns=[
                            k
                            for k in extra_vintage_key_cols
                            if k not in actual_signal_cols_from_db and k in df.columns
                        ],
                        errors="ignore",
                        inplace=True,
                    )

                return df[list(dict.fromkeys(cols_to_return))]
            else:  # Daily data
                df.set_index(date_col_in_source, inplace=True)
                df.sort_index(inplace=True)
                df = df[~df.index.duplicated(keep="first")]

                rename_dict = {col: f"{prefix}{col}" for col in actual_signal_cols_from_db}
                df.rename(columns=rename_dict, inplace=True)

                final_cols_to_keep = [
                    rename_dict[col] for col in actual_signal_cols_from_db if col in rename_dict
                ]
                return df[final_cols_to_keep]

        except Exception as e:
            logger.error(f"Error loading data from {table_name}: {e}", exc_info=True)
            return None

    def construct_master_table(
        self, start_date_str: str = "1986-01-01", end_date_str: str | None = None
    ) -> pd.DataFrame | None:
        logger.info("Starting master feature table construction.")

        regime_df = self._load_regime_data()
        if regime_df is None or regime_df.empty:
            logger.error("Cannot proceed without regime data.")
            return None

        min_regime_date = regime_df.index.min()
        max_regime_date = regime_df.index.max()

        effective_start_date = max(pd.to_datetime(start_date_str), min_regime_date)
        effective_end_date = pd.to_datetime(end_date_str) if end_date_str else max_regime_date

        master_date_index = pd.bdate_range(start=effective_start_date, end=effective_end_date, name="date")
        master_df = pd.DataFrame(index=master_date_index)
        master_df = master_df.join(regime_df, how="left")
        master_df["regime_t"] = pd.to_numeric(master_df["regime_t"], errors="coerce").astype("Int64")
        master_df["regime_t_plus_6m"] = pd.to_numeric(master_df["regime_t_plus_6m"], errors="coerce")

        table_configs = [
            {"table": "technical_indicators", "prefix": "ti_gspc_", "date_col": "date"},
            {"table": "volatility_indicators", "prefix": "vol_", "date_col": "date"},
            {"table": "put_call_ratios", "prefix": "pcr_", "date_col": "date"},
            {"table": "index_breadth_indicators", "prefix": "breadth_", "date_col": "date"},
            {"table": "cnn_fear_greed_index", "prefix": "fg_", "date_col": "reference_date"},
            {"table": "dxy_signals", "prefix": "dxy_", "date_col": "date"},
            {"table": "em_equity_signals", "prefix": "em_", "date_col": "date"},
            {"table": "oil_price_signals", "prefix": "oil_", "date_col": "date"},
            {"table": "bdi_signals", "prefix": "bdi_", "date_col": "date"},
            {"table": "gex_signals", "prefix": "gex_", "date_col": "date"},
            {"table": "sentiment_confidence_indices", "prefix": "sentconf_", "date_col": "date"},
            {"table": "smart_money_index", "prefix": "smi_", "date_col": "date"},
            {
                "table": "intermarket_stock_bond_returns_diff",
                "prefix": "stk_bond_diff_",
                "date_col": "date",
            },
            {"table": "sp500_derived_indicators", "prefix": "sp500_", "date_col": "date"},
            {"table": "corporate_bond_oas", "prefix": "corp_oas_", "date_col": "reference_date"},
            {
                "table": "non_farm_payrolls_signals",
                "prefix": "nfp_",
                "date_col": "reference_date",
                "is_vintage": True,
                "vintage_value_cols": self._get_db_table_columns("non_farm_payrolls_signals"),
            },
            {
                "table": "initial_jobless_claims_signals",
                "prefix": "icj_",
                "date_col": "reference_date",
                "is_vintage": True,
                "vintage_value_cols": self._get_db_table_columns("initial_jobless_claims_signals"),
            },
            {
                "table": "cpi_signals",
                "prefix": "cpi_",
                "date_col": "reference_date",
                "is_vintage": True,
                "vintage_value_cols": self._get_db_table_columns("cpi_signals"),
            },
            {
                "table": "retail_sales_signals",
                "prefix": "retail_",
                "date_col": "reference_date",
                "is_vintage": True,
                "vintage_value_cols": self._get_db_table_columns("retail_sales_signals"),
            },
            {
                "table": "m2_money_supply_signals",
                "prefix": "m2_",
                "date_col": "reference_date",
                "is_vintage": True,
                "vintage_value_cols": self._get_db_table_columns("m2_money_supply_signals"),
            },
            {
                "table": "housing_starts_signals",
                "prefix": "houst_",
                "date_col": "reference_date",
                "is_vintage": True,
                "vintage_value_cols": self._get_db_table_columns("housing_starts_signals"),
            },
            {
                "table": "housing_prices_signals",
                "prefix": "hpi_",
                "date_col": "reference_date",
                "is_vintage": True,
                "vintage_value_cols": self._get_db_table_columns("housing_prices_signals"),
            },
            {
                "table": "aaii_sentiment_signals",
                "prefix": "aaii_",
                "date_col": "reference_date",
                "is_vintage": True,
                "vintage_value_cols": self._get_db_table_columns("aaii_sentiment_signals"),
            },
            {
                "table": "finra_margin_statistics_signals",
                "prefix": "finra_",
                "date_col": "reference_date",
                "is_vintage": True,
                "vintage_value_cols": self._get_db_table_columns("finra_margin_statistics_signals"),
            },
            {
                "table": "consumer_confidence_signals",
                "prefix": "conf_",
                "date_col": "reference_date",
                "is_vintage": True,
                "vintage_value_cols": self._get_db_table_columns("consumer_confidence_signals"),
            },
            {
                "table": "investment_grade_junk_bond_yield_spread_signals",
                "prefix": "junk_spread_",
                "date_col": "reference_date",
                "is_vintage": True,
                "vintage_value_cols": self._get_db_table_columns(
                    "investment_grade_junk_bond_yield_spread_signals"
                ),
                "extra_vintage_key_cols": ["high_yield_series_id", "investment_grade_series_id"],
            },
            {
                "table": "em_corporate_vs_tbill_spread",
                "prefix": "em_tbill_spread_",
                "date_col": "reference_date",
                "is_vintage": True,
                "vintage_value_cols": ["spread_value"],
                "extra_vintage_key_cols": ["em_yield_series_id", "tbill_yield_series_id"],
            },
        ]

        for cfg in table_configs:
            if cfg.get("is_vintage"):
                cfg["vintage_value_cols"] = [
                    col
                    for col in cfg["vintage_value_cols"]
                    if col
                    not in EXCLUDE_COLS_UNIVERSAL
                    + ["release_date", "reference_date"]
                    + (cfg.get("extra_vintage_key_cols", []))
                ]

        all_data_parts = {}

        for config in table_configs:
            table_name = config["table"]
            filters_for_table = self.table_filters.get(table_name)

            df_signals = self._load_and_prefix_data(
                table_name=table_name,
                prefix=config["prefix"],
                date_col_in_source=config["date_col"],
                filters=filters_for_table,
                is_vintage=config.get("is_vintage", False),
                vintage_value_cols=config.get("vintage_value_cols"),
                extra_vintage_key_cols=config.get("extra_vintage_key_cols"),
            )
            if df_signals is not None and not df_signals.empty:
                all_data_parts[config["prefix"]] = {
                    "data": df_signals,
                    "is_vintage": config.get("is_vintage", False),
                }
            else:
                logger.warning(f"No data processed for table {table_name} with prefix {config['prefix']}")

        for _, item in all_data_parts.items():
            if not item["is_vintage"]:
                master_df = master_df.join(item["data"], how="left")

        master_df_reset = master_df.reset_index()

        for prefix_key, item in all_data_parts.items():
            if item["is_vintage"]:
                vintage_df = item["data"]
                if not vintage_df.index.name == "release_date":
                    logger.error(f"Vintage df for prefix {prefix_key} not indexed by release_date. Skipping.")
                    continue

                cols_from_vintage_to_merge = vintage_df.columns.tolist()

                if not cols_from_vintage_to_merge:
                    logger.debug(f"No new signal columns to merge from vintage source {prefix_key}")
                    continue

                master_df_reset.sort_values("date", inplace=True)

                merged_vintage = pd.merge_asof(
                    master_df_reset[["date"]],
                    vintage_df.reset_index()[["release_date"] + cols_from_vintage_to_merge],
                    left_on="date",
                    right_on="release_date",
                    direction="backward",
                )

                merged_vintage.set_index(master_df_reset["date"], inplace=True)

                cols_to_add = [
                    col
                    for col in merged_vintage.columns
                    if col not in master_df.columns and col not in ["date", "release_date"]
                ]

                if cols_to_add:
                    master_df = master_df.join(merged_vintage[cols_to_add], how="left")

        master_df.reset_index(inplace=True)

        pivoted_tables_config = [
            {
                "table": "intermarket_ratios",
                "prefix_base": "",
                "value_col_prefix": lambda cn, rn: f"{rn.lower().replace(' ', '_')}_{cn}",
            },
            {
                "table": "treasury_yield_spreads",
                "prefix_base": "tsy_spread_",
                "value_col_prefix": lambda cn, rn: f"tsy_spread_{rn.lower().replace('-', '')}_{cn}",
            },
            {
                "table": "relative_strength_metrics",
                "prefix_base": "",
                "value_col_prefix": lambda cn, rn: f"{rn.lower().replace(' ', '_')}_{cn}",
            },
        ]

        for piv_config in pivoted_tables_config:
            table_name = piv_config["table"]
            try:
                df_to_pivot = pd.read_sql_table(
                    table_name,
                    self.db_manager.engine,
                    parse_dates=["date" if table_name != "treasury_yield_spreads" else "reference_date"],
                )
                if not df_to_pivot.empty:
                    if table_name == "treasury_yield_spreads":
                        df_to_pivot.rename(
                            columns={
                                "reference_date": "date",
                                "spread_type": "pivot_key_col",
                                "value": "pivot_value_col",
                            },
                            inplace=True,
                        )
                    elif table_name == "intermarket_ratios":
                        df_to_pivot.rename(columns={"ratio_name": "pivot_key_col"}, inplace=True)
                        value_cols_for_pivot = [
                            c
                            for c in df_to_pivot.columns
                            if c
                            not in EXCLUDE_COLS_UNIVERSAL
                            + ["id", "pivot_key_col", "numerator_ticker", "denominator_ticker"]
                        ]
                    elif table_name == "relative_strength_metrics":
                        df_to_pivot.rename(columns={"comparison_pair": "pivot_key_col"}, inplace=True)
                        value_cols_for_pivot = [
                            c
                            for c in df_to_pivot.columns
                            if c
                            not in EXCLUDE_COLS_UNIVERSAL
                            + ["id", "pivot_key_col", "index_a_ticker", "index_b_ticker"]
                        ]

                    if table_name != "treasury_yield_spreads":
                        pivoted_df = df_to_pivot.pivot_table(
                            index="date", columns="pivot_key_col", values=value_cols_for_pivot
                        )
                        pivoted_df.columns = [
                            piv_config["value_col_prefix"](col_val, col_key)
                            for col_val, col_key in pivoted_df.columns
                        ]
                    else:  # Treasury spreads
                        pivoted_df = df_to_pivot.pivot_table(
                            index="date", columns="pivot_key_col", values="pivot_value_col"
                        )
                        pivoted_df.columns = [
                            piv_config["value_col_prefix"]("value", col_key) for col_key in pivoted_df.columns
                        ]

                    pivoted_df.sort_index(inplace=True)
                    master_df = pd.merge(master_df, pivoted_df.reset_index(), on="date", how="left")
                    logger.info(f"Merged pivoted {table_name}. Master df shape: {master_df.shape}")
            except Exception as e:
                logger.error(f"Error processing pivoted table {table_name}: {e}", exc_info=True)

        master_df["date"] = pd.to_datetime(master_df["date"]).dt.strftime("%Y-%m-%d")

        cols = master_df.columns.tolist()
        if "date" in cols:
            cols.insert(0, cols.pop(cols.index("date")))
            master_df = master_df.loc[:, cols]

        for col in master_df.columns:
            if master_df[col].dtype == "datetime64[ns]":
                master_df[col] = master_df[col].dt.strftime("%Y-%m-%d")
            elif pd.api.types.is_integer_dtype(master_df[col]) and master_df[col].isnull().any():
                master_df[col] = master_df[col].astype(float)
            elif pd.api.types.is_bool_dtype(master_df[col]):
                master_df[col] = master_df[col].astype(int)

        logger.info(f"Master feature table constructed. Final shape: {master_df.shape}")
        logger.info(f"Final columns: {master_df.columns.tolist()}")
        return master_df
