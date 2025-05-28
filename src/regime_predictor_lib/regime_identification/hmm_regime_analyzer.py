import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sqlalchemy
from hmmlearn.hmm import GaussianHMM
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

from regime_predictor_lib.utils.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class HMMRegimeAnalyzer:
    def __init__(
        self,
        db_manager: DatabaseManager,
        model_output_dir: Path | str,
        results_output_dir: Path | str,
        n_hmm_states: int = 3,
        hmm_random_state: int = 42,
        hmm_n_iter: int = 1000,
        hmm_tol: float = 1e-3,
        smoothing_window_size: int | None = None,
    ):
        self.db_manager = db_manager
        self.model_output_dir = Path(model_output_dir)
        self.results_output_dir = Path(results_output_dir)
        self.summary_dir = self.results_output_dir / "summaries"
        self.plots_dir = self.results_output_dir / "plots"

        self.n_hmm_states = n_hmm_states
        self.hmm_random_state = hmm_random_state
        self.hmm_n_iter = hmm_n_iter
        self.hmm_tol = hmm_tol
        self.smoothing_window_size = smoothing_window_size  # Store it

        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("notebook")
        plt.rcParams["figure.dpi"] = 100

        logger.info(
            f"HMMRegimeAnalyzer initialized. HMM states: {self.n_hmm_states}, "
            f"Model dir: {self.model_output_dir}, Results dir: {self.results_output_dir}, "
            "Smoothing window: "
            f"{self.smoothing_window_size if self.smoothing_window_size else 'None'}"
        )

    def _load_sp500_data(self, feature_columns: list[str]) -> pd.DataFrame | None:
        try:
            with self.db_manager.engine.connect() as connection:
                required_plot_cols = ["sp500_adjusted_close"]
                all_cols_to_fetch = list(set(feature_columns + required_plot_cols))
                select_cols_str = ", ".join([f'"{col}"' for col in all_cols_to_fetch])

                query = f"""
                    SELECT date, {select_cols_str}
                    FROM sp500_derived_indicators
                    ORDER BY date ASC;
                """
                df = pd.read_sql_query(sqlalchemy.text(query), connection, parse_dates=["date"])
            logger.info(
                f"Successfully loaded {len(df)} records for features: {feature_columns} "
                f"and S&P500 price from 'sp500_derived_indicators'."
            )
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error loading S&P 500 data: {e}", exc_info=True)
            return None

    def _apply_rolling_mode_filter(self, state_series: pd.Series, window_size: int) -> pd.Series:
        logger.info(f"Applying rolling mode filter with window size: {window_size}")

        if not isinstance(state_series, pd.Series):
            raise TypeError("Input state_series must be a pandas Series.")
        if window_size <= 0:
            logger.warning("Smoothing window size must be positive. Skipping smoothing.")
            return state_series.copy()

        def get_window_mode(window_data_series: pd.Series) -> int | float:
            window_data_no_na = window_data_series.dropna()
            if window_data_no_na.empty:
                return np.nan
            return window_data_no_na.mode()[0]

        smoothed_states = state_series.rolling(
            window=window_size, center=True, min_periods=1
        ).apply(get_window_mode, raw=False)

        smoothed_states = smoothed_states.bfill().ffill()

        if not smoothed_states.isnull().any():
            smoothed_states = smoothed_states.astype(int)
        else:
            logger.warning(
                "Smoothed states still contain NaNs after bfill/ffill. This is unexpected."
            )

        return smoothed_states

    def _plot_price_with_regimes(
        self,
        price_series: pd.Series,
        regime_series: pd.Series,
        num_regimes: int,
        title_suffix: str,
        output_filename: Path,
        main_price_label: str = "S&P 500 Price",
        yscale: str = "linear",
    ):
        fig, ax = plt.subplots(figsize=(17, 7))

        if num_regimes <= 10:
            colors = sns.color_palette("tab10", num_regimes)
        elif num_regimes <= 20:
            colors = sns.color_palette("tab20", num_regimes)
        else:
            colors = sns.color_palette("hsv", num_regimes)

        ax.plot(
            price_series.index,
            price_series.values,
            color="lightgray",
            lw=1,
            alpha=0.5,
            zorder=0,
        )
        aligned_regimes = regime_series.reindex(price_series.index).ffill().bfill()

        for i in range(len(price_series) - 1):
            current_regime_val = aligned_regimes.iloc[i]
            if pd.isna(current_regime_val) or not np.isfinite(current_regime_val):
                segment_color = "gray"
            else:
                try:
                    segment_color = colors[int(current_regime_val)]
                except IndexError:
                    logger.warning(
                        f"Regime value {current_regime_val} out of bounds for colors. Using gray."
                    )
                    segment_color = "gray"

            ax.plot(
                price_series.index[i : i + 2],
                price_series.iloc[i : i + 2],
                color=segment_color,
                lw=2,
            )

        legend_elements = [
            Line2D([0], [0], color=colors[i], lw=2, label=f"Regime {i}") for i in range(num_regimes)
        ]
        if main_price_label:
            legend_elements.append(
                Line2D([0], [0], color="lightgray", lw=1, label=main_price_label)
            )

        ax.legend(handles=legend_elements, title="Legend", loc="upper left")
        ax.set_yscale(yscale)
        plot_title = f"S&P 500 Price ({yscale.capitalize()} Scale) with {title_suffix} Regimes"
        ax.set_title(plot_title)
        ax.set_xlabel("Date")
        ax.set_ylabel(f"S&P 500 Adjusted Close ({yscale.capitalize()} Scale)")
        ax.grid(True, which="both", ls="--", alpha=0.7 if yscale == "log" else 0.5)
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close(fig)
        logger.info(f"Saved regime plot to {output_filename}")

    def analyze_and_save(
        self,
        feature_columns: list[str],
        analysis_name: str = "hmm_analysis",
        sp500_price_col: str = "sp500_adjusted_close",
    ):
        logger.info(f"Starting HMM analysis: {analysis_name} with features: {feature_columns}")

        data_df_full = self._load_sp500_data(feature_columns + [sp500_price_col])
        if data_df_full is None or data_df_full.empty:
            logger.error("Failed to load data for HMM analysis.")
            return

        features_df_raw = data_df_full[feature_columns].copy()
        features_df_raw.dropna(inplace=True)
        logger.info(f"Shape of feature data after dropna: {features_df_raw.shape}")

        if features_df_raw.empty or len(features_df_raw) < self.n_hmm_states * 10:
            logger.warning(
                f"Skipping HMM analysis due to insufficient data (rows: {len(features_df_raw)})."
            )
            return

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df_raw)
        features_scaled_df = pd.DataFrame(
            features_scaled, columns=feature_columns, index=features_df_raw.index
        )

        # Train HMM
        hmm_model = GaussianHMM(
            n_components=self.n_hmm_states,
            covariance_type="diag",
            n_iter=self.hmm_n_iter,
            random_state=self.hmm_random_state,
            tol=self.hmm_tol,
        )
        try:
            hmm_model.fit(features_scaled_df.values)
            predicted_states_raw = hmm_model.predict(features_scaled_df.values)
        except Exception as e:
            logger.error(f"Error fitting HMM model: {e}", exc_info=True)
            return

        hmm_states_series_raw = pd.Series(
            predicted_states_raw, index=features_scaled_df.index, name="hmm_state_raw"
        )

        analysis_name_for_files = analysis_name
        plot_title_suffix = f"HMM (N={self.n_hmm_states}) on {analysis_name}"

        if self.smoothing_window_size is not None and self.smoothing_window_size > 1:
            hmm_states_series_for_analysis = self._apply_rolling_mode_filter(
                hmm_states_series_raw.copy(), self.smoothing_window_size
            ).rename("hmm_state_smoothed")
            analysis_name_for_files = f"{analysis_name}_smoothed{self.smoothing_window_size}"
            plot_title_suffix = (
                f"HMM (N={self.n_hmm_states}, "
                f"Smoothed Win={self.smoothing_window_size}) on {analysis_name}"
            )
            logger.info(
                f"Using smoothed states (window={self.smoothing_window_size}) "
                f"for performance analysis and plots."
            )
        else:
            hmm_states_series_for_analysis = hmm_states_series_raw.rename("hmm_state")
            logger.info("Using raw HMM states for analysis and plots (no smoothing applied).")

        model_filename = self.model_output_dir / f"{analysis_name}_model.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump({"model": hmm_model, "scaler": scaler, "features": feature_columns}, f)
        logger.info(f"Saved HMM model and scaler to {model_filename}")

        transition_matrix_df = pd.DataFrame(
            hmm_model.transmat_,
            columns=[f"To State {i}" for i in range(self.n_hmm_states)],
            index=[f"From State {i}" for i in range(self.n_hmm_states)],
        ).round(4)
        transition_matrix_csv_filename = (
            self.summary_dir / f"{analysis_name}_transition_matrix.csv"
        )  # Original name
        transition_matrix_df.to_csv(transition_matrix_csv_filename)
        logger.info(
            f"Saved HMM transition matrix (original model) to {transition_matrix_csv_filename}"
        )

        regime_performance_df = features_df_raw.copy()
        regime_performance_df[sp500_price_col] = data_df_full.loc[
            features_df_raw.index, sp500_price_col
        ]
        regime_performance_df["sp500_daily_return"] = regime_performance_df[
            sp500_price_col
        ].pct_change()
        regime_performance_df["hmm_state"] = hmm_states_series_for_analysis

        full_results_df = data_df_full.loc[hmm_states_series_raw.index, [sp500_price_col]].copy()
        full_results_df = full_results_df.join(features_df_raw)
        full_results_df["hmm_state_raw"] = hmm_states_series_raw
        if self.smoothing_window_size is not None and self.smoothing_window_size > 1:
            full_results_df["hmm_state_smoothed"] = hmm_states_series_for_analysis
        else:
            full_results_df["hmm_state_smoothed"] = hmm_states_series_raw

        full_results_csv_filename = (
            self.summary_dir / f"{analysis_name_for_files}_full_data_with_states.csv"
        )
        full_results_df.to_csv(full_results_csv_filename)
        logger.info(f"Saved full data with HMM states to {full_results_csv_filename}")

        hmm_perf_summary = (
            regime_performance_df.groupby("hmm_state")["sp500_daily_return"]
            .agg(count="count", mean_daily_return="mean", std_daily_return="std")
            .reset_index()
        )
        hmm_perf_summary["annualized_return"] = hmm_perf_summary["mean_daily_return"] * 252
        hmm_perf_summary["annualized_volatility"] = hmm_perf_summary["std_daily_return"] * np.sqrt(
            252
        )

        unscaled_means_hmm = scaler.inverse_transform(hmm_model.means_)
        hmm_state_characteristics_list = []
        for i in range(self.n_hmm_states):
            state_data = {"HMM Model State Label": i}
            for j, feature_name in enumerate(feature_columns):
                state_data[f"Mean Original {feature_name}"] = unscaled_means_hmm[i, j]
                variance_scaled = (
                    hmm_model.covars_[i][j]
                    if hmm_model.covars_[i].ndim == 1
                    else hmm_model.covars_[i][j, j]
                )
                state_data[f"Std Original {feature_name}"] = (
                    np.sqrt(scaler.var_[j] * variance_scaled)
                    if hasattr(scaler, "var_") and scaler.var_ is not None
                    else np.sqrt(variance_scaled)
                )
                state_data[f"Mean Scaled {feature_name}"] = hmm_model.means_[i, j]
                state_data[f"Variance Scaled {feature_name}"] = variance_scaled

            perf_row = hmm_perf_summary[hmm_perf_summary["hmm_state"] == i]
            if not perf_row.empty:
                perf_data_for_state = (
                    perf_row.iloc[0]
                    .rename(lambda x: f"Perf_{x}" if x != "hmm_state" else "Perf_State_Label")
                    .to_dict()
                )
                state_data.update(perf_data_for_state)
            hmm_state_characteristics_list.append(state_data)

        hmm_characteristics_df = pd.DataFrame(hmm_state_characteristics_list)
        summary_csv_filename = self.summary_dir / f"{analysis_name_for_files}_summary_stats.csv"
        hmm_characteristics_df.to_csv(summary_csv_filename, index=False)
        logger.info(
            "Saved HMM summary "
            f"(model characteristics & analysis state performance) to {summary_csv_filename}"
        )

        sp500_price_for_plot = data_df_full.loc[
            hmm_states_series_for_analysis.index, sp500_price_col
        ].copy()

        if not sp500_price_for_plot.empty:
            plot_lin_filename = (
                self.plots_dir / f"{analysis_name_for_files}_price_regimes_linear.png"
            )
            self._plot_price_with_regimes(
                sp500_price_for_plot,
                hmm_states_series_for_analysis,
                self.n_hmm_states,
                plot_title_suffix,
                plot_lin_filename,
                yscale="linear",
            )
            plot_log_filename = self.plots_dir / f"{analysis_name_for_files}_price_regimes_log.png"
            self._plot_price_with_regimes(
                sp500_price_for_plot,
                hmm_states_series_for_analysis,
                self.n_hmm_states,
                plot_title_suffix,
                plot_log_filename,
                yscale="log",
            )

        fig_timeline, ax_timeline = plt.subplots(figsize=(15, 4))
        ax_timeline.plot(
            hmm_states_series_for_analysis.index,
            hmm_states_series_for_analysis.values,
            drawstyle="steps-post",
            alpha=0.7,
            lw=1.5,
        )
        ax_timeline.set_yticks(range(self.n_hmm_states))
        ax_timeline.set_ylabel("HMM Regime")
        ax_timeline.set_xlabel("Date")
        ax_timeline.set_title(plot_title_suffix.replace("on ", "Timeline for "))
        plt.tight_layout()
        timeline_plot_filename = self.plots_dir / f"{analysis_name_for_files}_regime_timeline.png"
        plt.savefig(timeline_plot_filename)
        plt.close(fig_timeline)
        logger.info(f"Saved HMM regime timeline plot to {timeline_plot_filename}")

        logger.info(
            f"HMM analysis '{analysis_name}' "
            f"(output prefix: '{analysis_name_for_files}') completed and results saved."
        )
