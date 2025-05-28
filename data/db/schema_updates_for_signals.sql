ALTER TABLE technical_indicators ADD COLUMN price_vs_sma50_pct_diff REAL;
ALTER TABLE technical_indicators ADD COLUMN price_vs_sma200_pct_diff REAL;
ALTER TABLE technical_indicators ADD COLUMN sma20_vs_sma50_signal INTEGER; -- -1, 0, 1
ALTER TABLE technical_indicators ADD COLUMN sma50_vs_sma200_signal INTEGER; -- -1, 0, 1
ALTER TABLE technical_indicators ADD COLUMN rsi_14_zscore_60d REAL;
ALTER TABLE technical_indicators ADD COLUMN rsi_14_percentile_252d REAL;
ALTER TABLE technical_indicators ADD COLUMN macd_hist_roc_5d REAL;
ALTER TABLE technical_indicators ADD COLUMN adx_level_signal INTEGER; -- e.g., 0, 1, 2
ALTER TABLE technical_indicators ADD COLUMN stoch_k_oversold_signal INTEGER; -- 0 or 1
ALTER TABLE technical_indicators ADD COLUMN stoch_k_overbought_signal INTEGER; -- 0 or 1

ALTER TABLE volatility_indicators ADD COLUMN vix_roc_21d REAL;
ALTER TABLE volatility_indicators ADD COLUMN vix_roc_63d REAL;
ALTER TABLE volatility_indicators ADD COLUMN vix_zscore_63d REAL;
ALTER TABLE volatility_indicators ADD COLUMN vix_percentile_252d REAL;
ALTER TABLE volatility_indicators ADD COLUMN vix_vs_sma20_signal INTEGER;
ALTER TABLE volatility_indicators ADD COLUMN vvix_roc_21d REAL;
ALTER TABLE volatility_indicators ADD COLUMN vvix_zscore_63d REAL;
ALTER TABLE volatility_indicators ADD COLUMN vvix_percentile_252d REAL;
ALTER TABLE volatility_indicators ADD COLUMN vix_vvix_ratio REAL;
ALTER TABLE volatility_indicators ADD COLUMN skew_index_zscore_63d REAL;
ALTER TABLE volatility_indicators ADD COLUMN skew_index_percentile_252d REAL;
ALTER TABLE volatility_indicators ADD COLUMN atr_pct_of_price REAL;

ALTER TABLE put_call_ratios ADD COLUMN equity_pc_ratio_sma_5d REAL;
ALTER TABLE put_call_ratios ADD COLUMN equity_pc_ratio_sma_21d REAL;
ALTER TABLE put_call_ratios ADD COLUMN equity_pc_ratio_vs_sma21d_diff REAL;
ALTER TABLE put_call_ratios ADD COLUMN equity_pc_ratio_roc_5d REAL;
ALTER TABLE put_call_ratios ADD COLUMN equity_pc_ratio_roc_21d REAL;
ALTER TABLE put_call_ratios ADD COLUMN equity_pc_ratio_zscore_63d REAL;
ALTER TABLE put_call_ratios ADD COLUMN equity_pc_ratio_percentile_252d REAL;

ALTER TABLE index_breadth_indicators ADD COLUMN pct_above_sma50_roc_21d REAL;
ALTER TABLE index_breadth_indicators ADD COLUMN pct_above_sma200_roc_21d REAL;
ALTER TABLE index_breadth_indicators ADD COLUMN ad_line_roc_21d REAL;
ALTER TABLE index_breadth_indicators ADD COLUMN ad_line_sma_21d_diff REAL;
ALTER TABLE index_breadth_indicators ADD COLUMN ad_ratio_sma_5d REAL;

ALTER TABLE cnn_fear_greed_index ADD COLUMN fg_value_sma_5d REAL;
ALTER TABLE cnn_fear_greed_index ADD COLUMN fg_value_sma_21d REAL;
ALTER TABLE cnn_fear_greed_index ADD COLUMN fg_value_vs_sma21d_diff REAL;
ALTER TABLE cnn_fear_greed_index ADD COLUMN fg_value_roc_5d REAL;
ALTER TABLE cnn_fear_greed_index ADD COLUMN is_extreme_fear_signal INTEGER;
ALTER TABLE cnn_fear_greed_index ADD COLUMN is_extreme_greed_signal INTEGER;
ALTER TABLE cnn_fear_greed_index ADD COLUMN fg_value_percentile_252d REAL;


CREATE TABLE IF NOT EXISTS non_farm_payrolls_signals (
    reference_date DATE NOT NULL,
    series_id TEXT NOT NULL,
    value_pit REAL,
    release_date DATE,
    yoy_change REAL,
    mom_change REAL,
    change_3m_annualized REAL,
    change_6m_annualized REAL,
    yoy_change_zscore_24m REAL,
    yoy_change_percentile_36m REAL,
    yoy_change_vs_sma12m_diff REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (reference_date, series_id)
);

CREATE TABLE IF NOT EXISTS initial_jobless_claims_signals (
    reference_date DATE NOT NULL,
    series_id TEXT NOT NULL,
    value_pit REAL,
    release_date DATE,
    value_pit_4wk_ma REAL,
    change_1wk REAL,
    change_4wk_ma_yoy REAL,
    change_4wk_ma_zscore_52wk REAL,
    change_4wk_ma_percentile_104wk REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (reference_date, series_id)
);

CREATE TABLE IF NOT EXISTS cpi_signals (
    reference_date DATE NOT NULL,
    series_id TEXT NOT NULL,
    value_pit REAL,
    release_date DATE,
    yoy_change REAL,
    mom_change REAL,
    change_3m_annualized REAL,
    change_6m_annualized REAL,
    yoy_change_zscore_24m REAL,
    yoy_change_percentile_36m REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (reference_date, series_id)
);

CREATE TABLE IF NOT EXISTS retail_sales_signals (
    reference_date DATE NOT NULL,
    series_id TEXT NOT NULL,
    value_pit REAL,
    release_date DATE,
    yoy_change REAL,
    mom_change REAL,
    change_3m_annualized REAL,
    yoy_change_zscore_24m REAL,
    yoy_change_percentile_36m REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (reference_date, series_id)
);

CREATE TABLE IF NOT EXISTS m2_money_supply_signals (
    reference_date DATE NOT NULL,
    series_id TEXT NOT NULL,
    value_pit REAL,
    release_date DATE,
    yoy_change REAL,
    mom_change REAL,
    change_3m_annualized REAL,
    yoy_change_zscore_24m REAL,
    yoy_change_percentile_60m REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (reference_date, series_id)
);

CREATE TABLE IF NOT EXISTS housing_starts_signals (
    reference_date DATE NOT NULL,
    series_id TEXT NOT NULL,
    value_pit REAL,
    release_date DATE,
    yoy_change REAL,
    mom_change REAL,
    change_3m_sma REAL,
    yoy_change_zscore_24m REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (reference_date, series_id)
);

CREATE TABLE IF NOT EXISTS housing_prices_signals (
    reference_date DATE NOT NULL,
    series_id TEXT NOT NULL,
    value_pit REAL,
    release_date DATE,
    yoy_change REAL,
    mom_change REAL,
    yoy_change_zscore_36m REAL,
    yoy_change_percentile_60m REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (reference_date, series_id)
);

CREATE TABLE IF NOT EXISTS aaii_sentiment_signals (
    reference_date DATE PRIMARY KEY,
    release_date DATE,
    bullish_pct_pit REAL,
    neutral_pct_pit REAL,
    bearish_pct_pit REAL,
    bull_bear_spread_pct_pit REAL,
    bullish_8wk_ma_pct_pit REAL,
    bull_bear_spread_pct_sma_4wk REAL,
    bull_bear_spread_pct_zscore_52wk REAL,
    bullish_pct_percentile_52wk REAL,
    bearish_pct_percentile_52wk REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS finra_margin_statistics_signals (
    reference_date DATE PRIMARY KEY,
    release_date DATE,
    debit_balances_pit REAL,
    free_credit_cash_pit REAL,
    free_credit_margin_pit REAL,
    debit_balances_yoy_change REAL,
    debit_balances_roc_3m REAL,
    debit_balances_roc_6m REAL,
    normalized_margin_debt REAL,
    margin_debt_vs_free_credit_ratio REAL,
    debit_balances_yoy_change_zscore_24m REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS consumer_confidence_signals (
    reference_date DATE NOT NULL,
    series_id TEXT NOT NULL,
    value_pit REAL,
    release_date DATE,
    value_mom_diff REAL,
    value_yoy_diff REAL,
    value_sma_3m REAL,
    value_sma_6m REAL,
    value_zscore_24m REAL,
    value_percentile_36m REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (reference_date, series_id)
);

CREATE TABLE IF NOT EXISTS investment_grade_junk_bond_yield_spread_signals (
    reference_date DATE NOT NULL,
    high_yield_series_id TEXT NOT NULL,
    investment_grade_series_id TEXT NOT NULL,
    release_date DATE,
    spread_value_pit REAL,
    spread_value_sma_21d REAL,
    spread_value_sma_63d REAL,
    spread_value_vs_sma63d_diff REAL,
    spread_value_roc_21d REAL,
    spread_value_roc_63d REAL,
    spread_value_zscore_252d REAL,
    spread_value_percentile_252d REAL,
    spread_widening_signal_5d INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (reference_date, high_yield_series_id, investment_grade_series_id)
);
