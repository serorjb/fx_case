"""
Configuration file for FX Options Trading System
All parameters and settings in one place
"""

from pathlib import Path
import numpy as np


class Config:
    """System configuration"""

    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    RESULTS_DIR = BASE_DIR / 'results'
    PLOTS_DIR = BASE_DIR / 'plots'

    # Data files
    FX_DATA_FILE = DATA_DIR / 'FX.parquet'
    DISCOUNT_CURVES_FILE = DATA_DIR / 'discount_curves.parquet'

    # Currency pairs and tenors
    CURRENCY_PAIRS = ['USDJPY', 'GBPNZD', 'USDCAD']
    TENORS = ['1W', '2W', '3W', '1M', '2M', '3M', '4M', '6M', '9M', '1Y']

    # Strike deltas for option chain
    DELTAS = [10, 20, 30, 40, 50, 60, 70, 80, 90]  # 50 is ATM

    # Tenor mapping
    TENOR_DAYS = {
        '1W': 7, '2W': 14, '3W': 21,
        '1M': 30, '2M': 60, '3M': 90,
        '4M': 120, '6M': 180, '9M': 270,
        '1Y': 365
    }

    TENOR_YEARS = {
        tenor: days / 365 for tenor, days in TENOR_DAYS.items()
    }

    # Trading parameters
    INITIAL_CAPITAL = 10_000_000  # $10M
    MAX_LEVERAGE = 5  # Maximum 5x leverage
    MAX_POSITIONS = 500  # Maximum concurrent positions
    POSITION_SIZE_PCT = 0.02  # 2% of capital per position
    MAX_POSITION_SIZE_PCT = 0.05  # Max 5% per position

    # Risk limits
    MAX_DELTA_PCT = 0.03  # Maximum 3% delta exposure
    MAX_DRAWDOWN = 0.15  # 15% max drawdown
    VAR_CONFIDENCE = 0.95  # 95% VaR

    # Transaction costs (in basis points)
    SPOT_SPREAD = 2  # 2 bps for spot
    OPTION_SPREAD = 20  # 20 bps for options
    COMMISSION = 5  # 5 bps commission

    # Interest rates for margin
    MARGIN_RATE = 0.05  # 5% annual rate on borrowed funds
    CASH_RATE = 0.02  # 2% annual rate on cash balance

    # Model parameters
    SIGNAL_THRESHOLD = 2.0  # Z-score threshold for signals
    LOOKBACK_DAYS = 20  # Days for moving averages
    VOL_WINDOW = 20  # Window for realized volatility

    # IV Filter strategy parameters
    IV_MA_WINDOW = 20  # Moving average window for IV filter
    IV_FILTER_BASE_MODEL = 'GK'  # Base model for IV filter strategy

    # ML model parameters
    ML_FEATURES = [
        'moneyness', 'time_to_maturity', 'spot_return',
        'realized_vol', 'implied_vol', 'vol_premium',
        'risk_reversal', 'butterfly', 'term_structure',
        'rate_differential', 'gk_price', 'gvv_price', 'sabr_price'
    ]
    ML_TRAINING_WINDOW = 5 * 252  # 5 years rolling window
    ML_RETRAIN_FREQ = 21  # Retrain every 21 days

    # SABR model parameters
    SABR_BETA = 0.5  # Beta parameter for SABR (0.5 for FX)

    # Hedging parameters
    HEDGE_FREQUENCY = 'daily'  # Hedge daily
    DELTA_HEDGE_THRESHOLD = 0.01  # Hedge if delta > 1% of capital

    # Backtesting parameters
    REBALANCE_FREQUENCY = 'daily'
    SIGNAL_LAG = 1  # 1 day lag between signal and execution

    # Visualization
    FIGURE_SIZE = (12, 8)
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'

    def __init__(self):
        """Create necessary directories"""
        self.DATA_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)
        self.PLOTS_DIR.mkdir(exist_ok=True)

    def get_transaction_cost(self, trade_type: str, notional: float) -> float:
        """Calculate transaction cost for a trade"""
        if trade_type == 'spot':
            spread_cost = notional * self.SPOT_SPREAD / 10000
        else:  # option
            spread_cost = notional * self.OPTION_SPREAD / 10000

        commission = notional * self.COMMISSION / 10000

        return spread_cost + commission

    def get_interest_rate(self, is_borrowing: bool) -> float:
        """Get applicable interest rate"""
        return self.MARGIN_RATE if is_borrowing else self.CASH_RATE