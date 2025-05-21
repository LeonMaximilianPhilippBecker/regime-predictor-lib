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
