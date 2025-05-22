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
