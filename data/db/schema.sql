CREATE TABLE IF NOT EXISTS technical_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,

    sma_20 REAL,
    sma_50 REAL,
    sma_100 REAL,
    sma_200 REAL,

    ema_20 REAL,
    ema_50 REAL,
    ema_100 REAL,
    ema_200 REAL,

    macd REAL,
    macd_histogram REAL,

    rsi_14 REAL,
    roc REAL,
    adx REAL,

    stochastic_k REAL,
    stochastic_d REAL,
    williams_r REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

CREATE TABLE IF NOT EXISTS volatility_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,

    vix REAL,
    vix_sma_20 REAL,
    vix_sma_50 REAL,

    vvix REAL,
    atr REAL,
    skew_index REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

CREATE TABLE IF NOT EXISTS put_call_ratios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL UNIQUE,
    equity_call_volume INTEGER,
    equity_put_volume INTEGER,
    equity_total_volume INTEGER,
    equity_pc_ratio REAL,
    source TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS index_breadth_indicators (
    date DATE PRIMARY KEY,
    pct_above_sma50 REAL,
    pct_above_sma200 REAL,
    ad_line REAL,
    ad_ratio REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS non_farm_payrolls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    value REAL,
    series_id TEXT DEFAULT 'PAYEMS',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, release_date, series_id)
);

CREATE TABLE IF NOT EXISTS initial_jobless_claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    value REAL,
    series_id TEXT DEFAULT 'ICSA',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, release_date, series_id)
);

CREATE TABLE IF NOT EXISTS cpi (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    value REAL,
    series_id TEXT DEFAULT 'CPIAUCSL',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, release_date, series_id)
);

CREATE TABLE IF NOT EXISTS gdp_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    forecast_period TEXT NOT NULL,
    release_date DATE NOT NULL,
    value REAL,
    series_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, release_date, series_id)
);

CREATE TABLE IF NOT EXISTS retail_sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    value REAL,
    series_id TEXT DEFAULT 'RSAFS',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, release_date, series_id)
);

CREATE TABLE IF NOT EXISTS m2_money_supply (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    value REAL,
    series_id TEXT DEFAULT 'M2SL',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, release_date, series_id)
);

CREATE TABLE IF NOT EXISTS housing_starts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    value REAL,
    series_id TEXT DEFAULT 'HOUST',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, release_date, series_id)
);

CREATE TABLE IF NOT EXISTS housing_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    value REAL,
    series_id TEXT DEFAULT 'CSUSHPINSA',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, release_date, series_id)
);

CREATE TABLE IF NOT EXISTS aaii_sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    bullish_pct REAL,
    neutral_pct REAL,
    bearish_pct REAL,
    total_pct REAL,
    bullish_8wk_ma_pct REAL,
    bull_bear_spread_pct REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, release_date)
);

CREATE TABLE IF NOT EXISTS finra_margin_statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    debit_balances_margin_accounts BIGINT,
    free_credit_cash_accounts BIGINT,
    free_credit_margin_accounts BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, release_date)
);

CREATE TABLE IF NOT EXISTS consumer_confidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    value REAL,
    series_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, release_date, series_id)
);

CREATE TABLE IF NOT EXISTS cnn_fear_greed_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    value INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date)
);

CREATE TABLE IF NOT EXISTS investment_grade_junk_bond_yield_spread (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    spread_value REAL,
    high_yield_series_id TEXT,
    investment_grade_series_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, release_date, high_yield_series_id, investment_grade_series_id)
);

CREATE TABLE IF NOT EXISTS corporate_bond_oas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    value REAL,
    series_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, series_id)
);

CREATE TABLE IF NOT EXISTS treasury_yield_spreads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    value REAL,
    spread_type TEXT NOT NULL,
    series_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, series_id)
);

CREATE TABLE IF NOT EXISTS em_corporate_vs_tbill_spread (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    spread_value REAL,
    em_yield_series_id TEXT,
    tbill_yield_series_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, em_yield_series_id, tbill_yield_series_id)
);


CREATE TABLE IF NOT EXISTS relative_strength_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    comparison_pair TEXT NOT NULL,
    index_a_ticker TEXT NOT NULL,
    index_b_ticker TEXT NOT NULL,
    rs_ratio REAL,

    log_diff_1d REAL,              -- log(rs_ratio_t) - log(rs_ratio_t-1)
    log_diff_5d REAL,              -- log(rs_ratio_t) - log(rs_ratio_t-5)
    log_diff_20d REAL,             -- log(rs_ratio_t) - log(rs_ratio_t-20)

    return_spread_1d REAL,

    z_score_spread_1d_window20d REAL,
    z_score_spread_1d_window60d REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, comparison_pair)
);

CREATE TABLE IF NOT EXISTS dxy_raw_fred (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    value REAL,
    series_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, series_id)
);

CREATE TABLE IF NOT EXISTS dxy_signals (
    date DATE NOT NULL,
    series_id TEXT,
    dxy_value REAL,
    pct_change_21d REAL,
    z_score_return_3m REAL,
    percentile_rank_1y REAL,
    volatility_30d REAL,
    slope_30d_regression REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, series_id)
);

CREATE TABLE IF NOT EXISTS em_equity_ohlcv_raw (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    symbol TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adjusted_close REAL,
    volume INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, symbol)
);

CREATE TABLE IF NOT EXISTS em_equity_signals (
    date DATE NOT NULL,
    symbol TEXT NOT NULL,
    close_price REAL,
    pct_return_21d REAL,
    z_score_return_3m REAL,
    above_sma_200 INTEGER,
    relative_performance_spy REAL,
    beta_to_spy_90d REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, symbol)
);

CREATE TABLE IF NOT EXISTS oil_raw_fred (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_date DATE NOT NULL,
    release_date DATE NOT NULL,
    value REAL,
    series_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_date, series_id)
);

CREATE TABLE IF NOT EXISTS oil_price_signals (
    date DATE NOT NULL,
    symbol TEXT NOT NULL,
    price REAL,
    pct_change_1m REAL,
    pct_change_6m REAL,
    z_score_return_3m REAL,
    volatility_30d REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, symbol)
);

CREATE TABLE IF NOT EXISTS bdi_signals (
    date DATE PRIMARY KEY,
    bdi_value REAL,
    pct_change_1m REAL,
    z_score_return_3m REAL,
    ema_30_minus_ema_90 REAL,
    bdi_oil_ratio REAL,
    oil_symbol_for_ratio TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bdi_raw_csv (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL UNIQUE,
    value REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS gex_signals (
    date DATE PRIMARY KEY,
    gex_value REAL,
    gex_percentile_rank_1y REAL,
    gex_z_score_3m REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sentiment_confidence_indices (
    date DATE PRIMARY KEY,
    smci_value REAL,
    dmci_value REAL,
    smci_pct_dmci_ratio REAL,
    smci_value_roc_1m REAL,
    smci_value_roc_3m REAL,
    smci_value_roc_6m REAL,
    dmci_value_roc_1m REAL,
    dmci_value_roc_3m REAL,
    dmci_value_roc_6m REAL,
    smci_pct_dmci_ratio_roc_1m REAL,
    smci_pct_dmci_ratio_roc_3m REAL,
    smci_pct_dmci_ratio_roc_6m REAL,
    smci_value_sma_3m REAL,
    smci_value_sma_6m REAL,
    dmci_value_sma_3m REAL,
    dmci_value_sma_6m REAL,
    smci_pct_dmci_ratio_sma_3m REAL,
    smci_pct_dmci_ratio_sma_6m REAL,
    smci_value_percentile_1y REAL,
    smci_value_percentile_2y REAL,
    dmci_value_percentile_1y REAL,
    dmci_value_percentile_2y REAL,
    smci_pct_dmci_ratio_percentile_1y REAL,
    smci_pct_dmci_ratio_percentile_2y REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS smart_money_index (
    date DATE PRIMARY KEY,
    smi_value REAL,
    spy_open REAL,
    spy_close REAL,
    smi_roc_21d REAL,
    smi_roc_63d REAL,
    smi_roc_126d REAL,
    smi_sma_20d REAL,
    smi_sma_50d REAL,
    smi_sma_200d REAL,
    smi_vs_sma20_signal INTEGER,
    smi_sma20_vs_sma50_signal INTEGER,
    smi_percentile_252d REAL,
    smi_percentile_504d REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
