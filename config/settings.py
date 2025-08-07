import numpy as np
from datetime import datetime
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
PLOTS_DIR = BASE_DIR / 'plots'
RESULTS_DIR = BASE_DIR / 'results'

# Create directories if they don't exist
PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Data settings
FX_DATA_PATH = DATA_DIR / 'FX.parquet'
DISCOUNT_CURVES_PATH = DATA_DIR / 'discount_curves.parquet'

# Currency pairs
CURRENCY_PAIRS = ['USDJPY', 'GBPNZD', 'USDCAD']

# Tenors for options
TENORS = ['1W', '2W', '3W', '1M', '2M', '3M', '4M', '6M', '9M', '1Y']
TENOR_DAYS = {
    '1W': 7, '2W': 14, '3W': 21, '1M': 30, '2M': 60, '3M': 90,
    '4M': 120, '6M': 180, '9M': 270, '1Y': 365
}

# Trading costs (in basis points)
BID_ASK_SPREAD = {
    'spot': 2,      # 2 bps
    'atm_vol': 20,  # 20 bps of vol
    'rr_25': 15,    # 15 bps
    'bf_25': 10,    # 10 bps
}

# Risk parameters
MAX_POSITION_SIZE = 1000000  # Max position per trade in USD
INITIAL_CAPITAL = 10000000   # $10M initial capital
MAX_DRAWDOWN = 0.10          # 10% max drawdown threshold
LEVERAGE_LIMIT = 5            # Maximum leverage

# Model parameters
CONFIDENCE_INTERVAL = 0.95   # For value calculations
MIN_EDGE_THRESHOLD = 0.001   # Minimum edge to trade (10 bps)

# Backtesting parameters
TRAIN_RATIO = 0.6            # 60% for training
VALIDATION_RATIO = 0.2       # 20% for validation
TEST_RATIO = 0.2            # 20% for testing
REBALANCE_FREQUENCY = 'W'   # Weekly rebalancing

# Greeks limits
DELTA_LIMIT = 1000000       # Max delta exposure
GAMMA_LIMIT = 50000         # Max gamma exposure
VEGA_LIMIT = 100000         # Max vega exposure

