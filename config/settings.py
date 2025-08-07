# config/settings.py
"""
Configuration settings for FX Options Trading System
Complete version with all enhancements
"""
import numpy as np
from datetime import datetime
from pathlib import Path

# =======================
# PATH CONFIGURATION
# =======================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
PLOTS_DIR = BASE_DIR / 'plots'
RESULTS_DIR = BASE_DIR / 'results'
SRC_DIR = BASE_DIR / 'src'

# Create directories if they don't exist
PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Data file paths
FX_DATA_PATH = DATA_DIR / 'FX.parquet'
DISCOUNT_CURVES_PATH = DATA_DIR / 'discount_curves.parquet'
FRED_DATA_DIR = DATA_DIR / 'FRED'

# =======================
# CURRENCY CONFIGURATION
# =======================
CURRENCY_PAIRS = ['USDJPY', 'GBPNZD', 'USDCAD']

# Tenors for options
TENORS = ['1W', '2W', '3W', '1M', '2M', '3M', '4M', '6M', '9M', '1Y']

# Tenor to days mapping
TENOR_DAYS = {
    '1W': 7, '2W': 14, '3W': 21, 
    '1M': 30, '2M': 60, '3M': 90,
    '4M': 120, '6M': 180, '9M': 270, 
    '1Y': 365, '12M': 365
}

# Tenor to years mapping
TENOR_YEARS = {
    '1W': 1/52, '2W': 2/52, '3W': 3/52,
    '1M': 1/12, '2M': 2/12, '3M': 3/12,
    '4M': 4/12, '6M': 6/12, '9M': 9/12,
    '1Y': 1.0, '12M': 1.0
}

# =======================
# TRADING COSTS
# =======================
BID_ASK_SPREAD = {
    'spot': 2,       # 2 basis points
    'forward': 3,    # 3 basis points
    'atm_vol': 20,   # 20 basis points of vol
    'rr_25': 15,     # 15 basis points
    'rr_10': 20,     # 20 basis points
    'bf_25': 10,     # 10 basis points
    'bf_10': 15      # 15 basis points
}

# Transaction costs (as fraction)
TRANSACTION_COST = 0.0002  # 2 bps

# =======================
# RISK PARAMETERS
# =======================
# Position limits
MAX_POSITION_SIZE = 1000000     # Max position per trade in USD
INITIAL_CAPITAL = 10000000      # $10M initial capital
MAX_DRAWDOWN = 0.10             # 10% max drawdown threshold
LEVERAGE_LIMIT = 5              # Maximum leverage

# Greeks limits
DELTA_LIMIT = 1000000          # Max delta exposure
GAMMA_LIMIT = 50000            # Max gamma exposure  
VEGA_LIMIT = 100000            # Max vega exposure
THETA_LIMIT = 50000            # Max theta exposure

# Risk metrics
VAR_CONFIDENCE = 0.95          # 95% VaR
CVAR_CONFIDENCE = 0.95         # 95% CVaR

# =======================
# MODEL PARAMETERS
# =======================
# Pricing models
CONFIDENCE_INTERVAL = 0.95     # For value calculations
MIN_EDGE_THRESHOLD = 0.001     # Minimum edge to trade (10 bps)

# Discount curve building
CURVE_METHOD = 'nelson_siegel_svensson'  # 'pchip', 'cubic', 'linear', 'nelson_siegel', 'nelson_siegel_svensson'
CURVE_REFIT_FREQ = 20          # Refit curve every N days

# =======================
# GARCH PARAMETERS
# =======================
# GARCH model settings
GARCH_TYPE = 'GARCH'           # 'GARCH', 'EGARCH', 'GJR-GARCH', 'HARCH'
GARCH_P = 1                    # GARCH lag order
GARCH_Q = 1                    # ARCH lag order
GARCH_REFIT_WINDOW = 252       # Refit GARCH every N observations
GARCH_MIN_OBS = 100           # Minimum observations for GARCH

# Multivariate GARCH
USE_MULTIVARIATE_GARCH = True
DCC_WINDOW = 252              # Window for DCC estimation

# =======================
# HMM PARAMETERS
# =======================
HMM_STATES = 2                # Number of volatility regimes
HMM_WINDOW = 252              # Lookback window for regime detection
HMM_MIN_OBS = 100            # Minimum observations for HMM

# =======================
# HEDGING PARAMETERS
# =======================
ENABLE_DELTA_HEDGING = True   # Enable automatic delta hedging
HEDGE_THRESHOLD = 0.01        # 1% of notional triggers hedge
USE_FORWARDS_FOR_HEDGING = False  # Use spot (False) or forwards (True)
HEDGE_REBALANCE_FREQ = 'D'   # Daily hedging

# =======================
# BACKTESTING PARAMETERS
# =======================
# Data split
TRAIN_RATIO = 0.6             # 60% for training
VALIDATION_RATIO = 0.2        # 20% for validation
TEST_RATIO = 0.2             # 20% for testing

# Rebalancing
REBALANCE_FREQUENCY = 'W'     # Weekly rebalancing ('D', 'W', 'M')
SIGNAL_LAG = 1               # Lag between signal and execution (days)

# =======================
# STRATEGY PARAMETERS
# =======================
# Volatility Arbitrage
VOL_ARB_LOOKBACK = 20        # Days for realized vol calculation
VOL_ARB_ZSCORE_THRESHOLD = 2.0  # Z-score threshold for signals
VOL_ARB_MIN_EDGE = 0.005     # Minimum vol edge (50 bps)

# Carry Strategy
CARRY_MIN_RATIO = 0.5        # Minimum carry/vol ratio
CARRY_LOOKBACK = 60          # Days for carry calculation

# Position sizing
USE_KELLY_SIZING = True       # Use Kelly criterion
KELLY_FRACTION = 0.25        # Max Kelly fraction
MIN_POSITION_SIZE = 10000    # Minimum position size
MAX_POSITIONS = 20           # Maximum concurrent positions

# =======================
# LIGHTGBM PARAMETERS
# =======================
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 4,
    'min_data_in_leaf': 20,
    'max_depth': -1
}

LGBM_NUM_ROUNDS = 1000
LGBM_EARLY_STOPPING = 50

# For classification (signals)
LGBM_SIGNAL_PARAMS = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 4
}

# =======================
# VISUALIZATION SETTINGS
# =======================
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (12, 8)
DPI = 100
SAVE_PLOTS = True
INTERACTIVE_PLOTS = True     # Use plotly for interactive plots

# Colors for plots
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8'
}

# =======================
# PERFORMANCE TARGETS
# =======================
TARGET_SHARPE = 1.5          # Target Sharpe ratio
TARGET_RETURN = 0.15         # Target annual return (15%)
TARGET_WIN_RATE = 0.55       # Target win rate (55%)

# =======================
# LOGGING SETTINGS
# =======================
LOG_LEVEL = 'INFO'           # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_TO_FILE = True
LOG_FILE = RESULTS_DIR / f'trading_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# =======================
# SYSTEM SETTINGS
# =======================
RANDOM_SEED = 42            # For reproducibility
N_JOBS = -1                 # Number of parallel jobs (-1 = all cores)
CHUNK_SIZE = 10000          # Chunk size for large data processing

# =======================
# VALIDATION SETTINGS
# =======================
# Check data quality
CHECK_DATA_QUALITY = True
MAX_MISSING_RATIO = 0.1     # Maximum 10% missing data
MIN_DATA_POINTS = 500       # Minimum data points required

# =======================
# EXPORT SETTINGS
# =======================
EXPORT_FORMAT = 'parquet'    # 'csv', 'parquet', 'excel'
COMPRESS_EXPORTS = True
EXPORT_SAMPLE_SIZE = 1000    # Sample size for exports

print(f"Settings loaded: {datetime.now()}")
print(f"Base directory: {BASE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Currency pairs: {CURRENCY_PAIRS}")
print(f"Delta hedging: {ENABLE_DELTA_HEDGING}")
print(f"GARCH type: {GARCH_TYPE}")
print(f"HMM states: {HMM_STATES}")