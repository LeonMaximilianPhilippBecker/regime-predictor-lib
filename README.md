# Market Regime Predictor

A Python framework for predicting market regime shifts using a multi-resolution temporal ensemble of machine learning models. This project moves beyond simple price prediction to forecast the underlying character of the market (e.g., bull, bear, volatile) by synthesizing signals from hundreds of financial and economic indicators.

## 🏛️ Philosophy & Methodological Rigor

This project is heavily inspired by the methodologies outlined in Dr. Marcos Lopez de Prado's book, *Advances in Financial Machine Learning*. The core philosophy is a defense-first approach against overfitting and data leakage, which are the primary failure points of quantitative strategies.

Key precautions implemented throughout the pipeline include:

**Purged K-Fold Cross-Validation:** Standard CV fails in finance due to data autocorrelation. This pipeline uses a Purged K-Fold splitter, which:
* **Purges:** Removes training data points whose labels overlap with the information available during the test set period.
* **Embargoes:** Applies a gap after each test set to prevent future information from leaking into subsequent training folds.

**Point-in-Time (PIT) Data:** All macroeconomic data is handled as "vintage" data. The system correctly uses the data as it would have been known at the time of the decision (based on `release_date`), not based on its revised `reference_date`. This is critical for preventing lookahead bias.

**Probabilistic Labeling:** Instead of relying on a single, fragile "ground truth" from one HMM run, the target variable is a stable probability distribution generated from an ensemble of HMMs. This reduces the model risk associated with the labeling process itself.

**Out-of-Sample Feature Importance:** Feature importance is not evaluated using brittle, in-sample metrics (like Gini importance). Instead, Mean Decrease Accuracy (MDA) is calculated on out-of-sample validation sets to provide a more robust measure of a feature's predictive power.

**Hyperparameter Tuning with Time-Series CV:** Hyperparameter optimization is performed using Optuna, but the objective function is evaluated using the same robust Purged K-Fold CV to find parameters that generalize well over time.

## 🏗️ System Architecture

The project is built as a sequential, multi-stage pipeline designed for robustness and reproducibility.

### 1. Data Ingestion & Warehousing
Fetches a diverse dataset from APIs (FRED, yfinance) and local files, centralizing everything into a SQLite database managed by SQLAlchemy and versioned with DVC.

### 2. Unsupervised Regime Identification (HMM)
Trains an ensemble of Hidden Markov Models on S&P 500 returns and volatility to create stable, probabilistic target labels for the supervised models.

### 3. Feature Engineering & Signal Generation
Transforms raw data into a rich feature set of over 200 point-in-time correct signals, including technicals, credit spreads, sentiment scores, and intermarket relationships.

### 4. Thematic Table Construction
Groups related features into "themes" (e.g., `theme_volatility_and_market_stress`) to allow for the training of specialist models.

### 5. Feature Reduction & Selection
This multi-stage process reduces multicollinearity within each theme:
* **Stationarity Analysis:** Removes non-stationary features.
* **Correlation Pruning:** Removes highly correlated features based on Spearman correlation, keeping the one with the highest mutual information to the target.
* **VIF Reduction:** Iteratively removes features with a high Variance Inflation Factor (VIF).

### 6. Thematic Model Training (Inner Models)
Trains XGBoost, LightGBM, and CatBoost models on each refined thematic feature set.
* **Robust Validation:** All models are evaluated using Purged K-Fold CV to prevent data leakage and ensure out-of-sample performance is properly estimated.
* **HPO:** Hyperparameters are systematically tuned using Optuna integrated with the time-series cross-validation process.

### 7. Ensemble Modeling (Outer Model)
Synthesizes the predictions from all specialist models into a single forecast.
* **Ensemble Feature Generation:** Generates out-of-sample predictions from each tuned thematic model during the cross-validation process. These OOS predictions become the feature set for the final meta-model.
* **Final Model Training:** A final meta-model (e.g., Logistic Regression) is trained on these ensemble features.

## 📊 Features & Data Sources

The system is built on a comprehensive feature set of over 200 indicators, including point-in-time vintage macroeconomic data to avoid lookahead bias.

* **Technical Trend & Momentum:** SMAs, EMAs, MACD, RSI, ADX.
* **Volatility & Market Stress:** VIX, VVIX, CBOE Skew Index, ATR.
* **Market Internals:** Advance-Decline Line, % of Stocks above 50/200 SMA, Equity Put/Call Ratio.
* **Intermarket Relationships:** S&P 500 vs. Bonds (TLT) ratio, Gold/Silver ratio, Copper/Gold ratio.
* **Credit & Bond Market Tells:** Investment Grade vs. Junk Bond Yield Spread, Corporate Bond OAS, 10Y-2Y Treasury Spread.
* **Sentiment & Behavior Gauges:** CNN Fear & Greed, AAII Sentiment, Consumer Confidence, FINRA Margin Debt.
* **Macroeconomic Data (Vintage):** Non-Farm Payrolls, CPI, Initial Jobless Claims, Retail Sales, M2 Money Supply.
* **Market Structure & Fund Flows:** Smart Money Index (SMI).
* **Sector & Micro-Market Tells:** Transports vs. S&P 500, Small Caps vs. Large Caps.
* **Global Markets & Currency:** Dollar Index (DXY), Emerging Markets (EEM) vs. SPY, Oil Prices, Baltic Dry Index (BDI).
* **Advanced Derivative Metrics:** Gamma Exposure (GEX).

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Dependency Management:** Poetry
* **Data Handling:** Pandas, SQLAlchemy, NumPy
* **ML Models:** Scikit-learn, XGBoost, LightGBM, CatBoost, HMMlearn
* **Hyperparameter Tuning:** Optuna
* **Data Versioning:** DVC
* **CI/CD & Tooling:** GitHub Actions, Pre-commit, Flake8, Black, isort, MyPy

## ⚙️ Setup and Installation

**Prerequisites:** Python 3.10+, Poetry, Git, DVC.

1. **Clone:** `git clone <your-repo-url>`
2. **API Keys:** Copy `.env.example` to `.env` and fill in your API keys.
3. **Install:** `poetry install`
4. **Hooks:** `poetry run pre-commit install`
5. **Data:** `dvc pull`

## 🚀 Running the Pipeline

The project is executed via the numbered scripts in `src/scripts/`. Run them in order to ensure data dependencies are met.

* **Configuration:** Key parameters for model training, HPO, and pipeline definitions are located in the `config/` directory.
* **Execution:** Use `poetry run python src/scripts/<script_name>.py` to run a specific step of the pipeline.
