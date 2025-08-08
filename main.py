"""
FX Options Trading System - Complete Implementation
Prices options using Black-Scholes, finds mispricings using various models,
trades and delta hedges positions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
class Config:
    """System configuration"""
    # Paths
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")
    FX_DATA_PATH = DATA_DIR / "FX.parquet"
    CURVES_PATH = DATA_DIR / "discount_curves.parquet"

    # Trading parameters
    INITIAL_CAPITAL = 10_000_000
    MAX_LEVERAGE = 5
    MAX_POSITIONS = 500
    MAX_DELTA_EXPOSURE = 0.03  # 3% max delta at end of day
    MISPRICING_THRESHOLD = 0.02  # 2% mispricing to trade

    # Realistic Transaction costs for FX Options
    # Based on typical institutional spreads
    TRANSACTION_COSTS = {
        'spot': {
            'USDJPY': 0.00002,  # 0.2 pips for majors
            'GBPNZD': 0.00015,  # 1.5 pips for crosses
            'USDCAD': 0.00003,  # 0.3 pips
        },
        'options': {
            # Vol spreads by tenor (in vol points)
            '1W': 0.005,   # 50 bps vol spread
            '2W': 0.004,   # 40 bps
            '3W': 0.004,   # 40 bps
            '1M': 0.003,   # 30 bps
            '2M': 0.0035,  # 35 bps
            '3M': 0.004,   # 40 bps
            '4M': 0.0045,  # 45 bps
            '6M': 0.005,   # 50 bps
            '9M': 0.006,   # 60 bps
            '1Y': 0.007,   # 70 bps
        },
        # Additional costs
        'brokerage': 0.00002,  # 0.2 bps of notional
        'clearing': 0.00001,   # 0.1 bps of notional
    }

    # Model parameters
    LGBM_TRAINING_YEARS = 5
    IV_MA_DAYS = 20  # For IV filter strategy
    DELTA_MODEL = 'BS'  # Model for delta calculation: 'BS', 'GK', 'GVV', 'SABR'

    # Deltas to evaluate (as absolute values)
    DELTAS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    # Tenors to trade
    TENORS = ['1W', '2W', '3W', '1M', '2M', '3M', '4M', '6M', '9M', '1Y']

    # Currency pairs
    PAIRS = ['USDJPY', 'GBPNZD', 'USDCAD']

# ==================== Data Structures ====================
@dataclass
class OptionPosition:
    """Represents an option position in the book"""
    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    pair: str
    strike: float
    option_type: str  # 'call' or 'put'
    position_size: float  # positive for long, negative for short
    entry_price: float
    entry_spot: float
    entry_vol: float
    model_used: str
    current_delta: float = 0.0
    current_vega: float = 0.0
    current_value: float = 0.0
    pnl: float = 0.0
    transaction_costs: float = 0.0

@dataclass
class TradingBook:
    """Manages all positions and hedges"""
    options: List[OptionPosition] = field(default_factory=list)
    spot_hedge: Dict[str, float] = field(default_factory=dict)  # Spot position by pair
    cash: float = Config.INITIAL_CAPITAL
    used_margin: float = 0.0
    total_transaction_costs: float = 0.0

    def get_net_delta(self, pair: str = None) -> float:
        """Calculate total portfolio delta in dollar terms"""
        if pair:
            # Delta for specific pair
            option_delta = sum(pos.current_delta * abs(pos.position_size) * pos.entry_spot *
                             (1 if pos.position_size > 0 else -1)
                             for pos in self.options if pos.pair == pair)
            spot_delta = self.spot_hedge.get(pair, 0) * self._get_current_spot(pair)
            return option_delta + spot_delta
        else:
            # Total portfolio delta
            option_delta = sum(pos.current_delta * abs(pos.position_size) * pos.entry_spot *
                             (1 if pos.position_size > 0 else -1)
                             for pos in self.options)
            # For spot hedges, need current spot prices (simplified - using entry spot)
            spot_delta = sum(amount * self._get_current_spot(pair)
                           for pair, amount in self.spot_hedge.items())
            return option_delta + spot_delta

    def get_net_vega(self) -> float:
        """Calculate total portfolio vega in dollar terms"""
        return sum(pos.current_vega * abs(pos.position_size) * pos.entry_spot *
                  (1 if pos.position_size > 0 else -1)
                  for pos in self.options)

    def get_num_positions(self) -> int:
        """Count active positions"""
        return len(self.options)

    def _get_current_spot(self, pair: str) -> float:
        """Get current spot price for a pair (simplified - returns last known)"""
        # In practice, would get from current market data
        # For now, use average of entry spots
        pair_positions = [pos for pos in self.options if pos.pair == pair]
        if pair_positions:
            return pair_positions[-1].entry_spot
        return 100.0  # Default

@dataclass
class MarketData:
    """Market data for a specific date"""
    date: pd.Timestamp
    spot: Dict[str, float]  # spot prices by pair
    forwards: Dict[str, Dict[str, float]]  # forward points by pair and tenor
    atm_vols: Dict[str, Dict[str, float]]  # ATM vols by pair and tenor
    rr_25: Dict[str, Dict[str, float]]  # 25-delta risk reversals
    bf_25: Dict[str, Dict[str, float]]  # 25-delta butterflies
    rr_10: Dict[str, Dict[str, float]]  # 10-delta risk reversals
    bf_10: Dict[str, Dict[str, float]]  # 10-delta butterflies
    rates: Dict[str, float]  # interest rates by currency

# ==================== Pricing Models ====================
class BlackScholes:
    """Black-Scholes pricing for FX options"""

    @staticmethod
    def price(S: float, K: float, T: float, r_d: float, r_f: float,
              sigma: float, option_type: str) -> float:
        """Price option using Black-Scholes"""
        if T <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S)

        from scipy.stats import norm

        d1 = (np.log(S/K) + (r_d - r_f + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if option_type == 'call':
            return S*np.exp(-r_f*T)*norm.cdf(d1) - K*np.exp(-r_d*T)*norm.cdf(d2)
        else:
            return K*np.exp(-r_d*T)*norm.cdf(-d2) - S*np.exp(-r_f*T)*norm.cdf(-d1)

    @staticmethod
    def delta(S: float, K: float, T: float, r_d: float, r_f: float,
              sigma: float, option_type: str) -> float:
        """Calculate option delta"""
        if T <= 0:
            return 1.0 if option_type == 'call' and S > K else 0.0

        from scipy.stats import norm

        d1 = (np.log(S/K) + (r_d - r_f + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

        if option_type == 'call':
            return np.exp(-r_f*T) * norm.cdf(d1)
        else:
            return -np.exp(-r_f*T) * norm.cdf(-d1)

    @staticmethod
    def vega(S: float, K: float, T: float, r_d: float, r_f: float, sigma: float) -> float:
        """Calculate option vega"""
        if T <= 0:
            return 0.0

        from scipy.stats import norm

        d1 = (np.log(S/K) + (r_d - r_f + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return S * np.exp(-r_f*T) * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change

    @staticmethod
    def price_with_vol_spread(S: float, K: float, T: float, r_d: float, r_f: float,
                             sigma: float, option_type: str, tenor: str,
                             is_buy: bool) -> Tuple[float, float]:
        """Price option with bid-ask spread in volatility"""
        vol_spread = Config.TRANSACTION_COSTS['options'].get(tenor, 0.005)

        if is_buy:
            # Pay offer vol (higher)
            effective_vol = sigma + vol_spread / 2
        else:
            # Receive bid vol (lower)
            effective_vol = sigma - vol_spread / 2

        price = BlackScholes.price(S, K, T, r_d, r_f, effective_vol, option_type)

        # Transaction cost is the difference from mid
        mid_price = BlackScholes.price(S, K, T, r_d, r_f, sigma, option_type)
        transaction_cost = abs(price - mid_price)

        return price, transaction_cost

    @staticmethod
    def strike_from_delta(S: float, delta: float, T: float, r_d: float, r_f: float,
                         sigma: float, option_type: str) -> float:
        """Calculate strike from delta"""
        from scipy.stats import norm

        if option_type == 'call':
            d1 = norm.ppf(delta * np.exp(r_f * T))
        else:
            d1 = -norm.ppf(-delta * np.exp(r_f * T))

        K = S * np.exp(-(d1 * sigma * np.sqrt(T) - (r_d - r_f + 0.5*sigma**2)*T))
        return K

class GVVModel:
    """Simplified GVV model implementation"""

    @staticmethod
    def price(S: float, K: float, T: float, r_d: float, r_f: float,
              atm_vol: float, rr_25: float, bf_25: float, option_type: str) -> float:
        """Price using GVV adjustment to Black-Scholes"""
        # Calculate smile-adjusted volatility
        moneyness = K / S

        if moneyness < 0.95:  # Deep OTM put / ITM call region
            smile_vol = atm_vol + bf_25 - rr_25/2 + 0.002 * (0.95 - moneyness)
        elif moneyness > 1.05:  # Deep OTM call / ITM put region
            smile_vol = atm_vol + bf_25 + rr_25/2 + 0.002 * (moneyness - 1.05)
        else:  # Near ATM
            skew_adjustment = rr_25 * (moneyness - 1.0) * 4  # Linear skew
            smile_vol = atm_vol + bf_25 + skew_adjustment

        # Price with adjusted vol
        return BlackScholes.price(S, K, T, r_d, r_f, smile_vol, option_type)

class SABRModel:
    """SABR model with proper calibration from training data"""

    def __init__(self):
        self.calibrated_params = {}  # Store calibrated parameters per pair and tenor
        self.is_calibrated = False

    def calibrate_from_training_data(self, fx_data: pd.DataFrame):
        """Calibrate SABR parameters from historical volatility surfaces"""
        from scipy.optimize import minimize

        print("Calibrating SABR model...")

        for pair in Config.PAIRS:
            self.calibrated_params[pair] = {}
            spot_col = f'{pair} Curncy'

            if spot_col not in fx_data.columns:
                continue

            print(f"  Calibrating {pair}...")

            for tenor in ['1M', '3M', '6M', '1Y']:
                T = {'1M': 1/12, '3M': 0.25, '6M': 0.5, '1Y': 1.0}[tenor]

                # Collect historical smile data
                atm_col = f'{pair}V{tenor} Curncy'
                rr_col = f'{pair}25R{tenor} Curncy'
                bf_col = f'{pair}25B{tenor} Curncy'

                if atm_col not in fx_data.columns:
                    continue

                # Get average parameters over training period
                atm_vols = []
                rr_values = []
                bf_values = []

                for date in fx_data.index[-252:]:  # Last year of training data
                    if pd.notna(fx_data.loc[date, atm_col]):
                        atm_vols.append(fx_data.loc[date, atm_col] / 100)

                        if rr_col in fx_data.columns:
                            rr_values.append(fx_data.loc[date, rr_col] / 100)
                        else:
                            rr_values.append(0)

                        if bf_col in fx_data.columns:
                            bf_values.append(fx_data.loc[date, bf_col] / 100)
                        else:
                            bf_values.append(0)

                if len(atm_vols) < 20:
                    continue

                # Average parameters
                avg_atm = np.mean(atm_vols)
                avg_rr = np.mean(rr_values)
                avg_bf = np.mean(bf_values)

                # Estimate SABR parameters
                # Alpha: ATM vol level
                alpha = avg_atm

                # Beta: fixed for FX
                beta = 0.5

                # Rho: correlation, estimated from skew
                # Negative RR typically means negative rho
                rho = np.clip(-avg_rr * 10, -0.9, 0.9)

                # Nu: vol of vol, estimated from butterfly and historical vol of vol
                vol_of_vol = np.std(atm_vols) * np.sqrt(252)
                nu = np.clip(vol_of_vol, 0.1, 1.0)

                self.calibrated_params[pair][tenor] = {
                    'alpha': alpha,
                    'beta': beta,
                    'rho': rho,
                    'nu': nu,
                    'avg_atm': avg_atm,
                    'avg_rr': avg_rr,
                    'avg_bf': avg_bf
                }

                print(f"    {tenor}: α={alpha:.4f}, β={beta:.2f}, ρ={rho:.3f}, ν={nu:.3f}")

        self.is_calibrated = len(self.calibrated_params) > 0
        print(f"  SABR calibration complete for {list(self.calibrated_params.keys())}")

    def calculate_sabr_vol(self, S: float, K: float, T: float, params: Dict) -> float:
        """Calculate SABR implied volatility using calibrated parameters"""
        if not params:
            return 0.1

        alpha = params['alpha']
        beta = params['beta']
        rho = params['rho']
        nu = params['nu']

        # Forward (simplified - assuming r_d = r_f)
        F = S

        if abs(F - K) < 0.001 * F:
            # ATM case
            sabr_vol = alpha * (1 + (nu**2 / 24) * T)
        else:
            # Hagan approximation
            try:
                z = (nu/alpha) * (F*K)**((1-beta)/2) * np.log(F/K)
                x_arg = np.sqrt(1 - 2*rho*z + z**2) + z - rho

                if x_arg <= 0:
                    # Fallback to ATM vol
                    sabr_vol = alpha
                else:
                    x = np.log(x_arg / (1 - rho))

                    # First factor
                    factor1 = alpha / ((F*K)**((1-beta)/2) *
                                      (1 + (1-beta)**2/24 * (np.log(F/K))**2 +
                                       (1-beta)**4/1920 * (np.log(F/K))**4))

                    # Second factor
                    if abs(x) > 0.001:
                        factor2 = z / x
                    else:
                        factor2 = 1.0

                    # Third factor (correction terms)
                    factor3 = 1 + ((1-beta)**2/24 * alpha**2 / ((F*K)**(1-beta)) +
                                  rho*beta*nu*alpha / (4*(F*K)**((1-beta)/2)) +
                                  (2-3*rho**2)*nu**2/24) * T

                    sabr_vol = factor1 * factor2 * factor3
            except:
                # Fallback to ATM vol
                sabr_vol = alpha

        # Ensure positive and reasonable
        return np.clip(sabr_vol, 0.001, 2.0)

    def price(self, S: float, K: float, T: float, r_d: float, r_f: float,
              pair: str, tenor: str, option_type: str) -> float:
        """Price using SABR vol with calibrated parameters"""
        if not self.is_calibrated or pair not in self.calibrated_params:
            # Fallback to Black-Scholes with default vol
            return BlackScholes.price(S, K, T, r_d, r_f, 0.1, option_type)

        # Get appropriate calibrated parameters
        if tenor not in self.calibrated_params[pair]:
            # Use nearest available tenor
            available_tenors = list(self.calibrated_params[pair].keys())
            if not available_tenors:
                return BlackScholes.price(S, K, T, r_d, r_f, 0.1, option_type)
            tenor = available_tenors[0]

        params = self.calibrated_params[pair][tenor]

        # Calculate SABR vol
        sabr_vol = self.calculate_sabr_vol(S, K, T, params)

        # Price with SABR vol
        return BlackScholes.price(S, K, T, r_d, r_f, sabr_vol, option_type)

# ==================== ML Model ====================
class LGBMPricer:
    """LightGBM model for option pricing"""

    def __init__(self, training_years: int = 5):
        self.training_years = training_years
        self.models = {}  # One model per pair
        self.scalers = {}  # Feature scalers per pair
        self.feature_columns = None
        self.is_trained = False

    def prepare_training_features(self, fx_data: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Prepare features from historical data for training"""
        features = pd.DataFrame(index=fx_data.index)

        # Get pair-specific columns
        spot_col = f'{pair} Curncy'
        if spot_col not in fx_data.columns:
            return features

        # Basic price features
        features['spot'] = fx_data[spot_col]
        features['log_spot'] = np.log(features['spot'])
        features['spot_return_1d'] = features['spot'].pct_change(1)
        features['spot_return_5d'] = features['spot'].pct_change(5)
        features['spot_return_20d'] = features['spot'].pct_change(20)

        # Realized volatility
        log_returns = np.log(features['spot'] / features['spot'].shift(1))
        for window in [5, 10, 20, 60]:
            features[f'realized_vol_{window}d'] = log_returns.rolling(window).std() * np.sqrt(252)

        # For each tenor, add vol features
        for tenor in ['1M', '3M', '6M', '1Y']:
            vol_col = f'{pair}V{tenor} Curncy'
            if vol_col in fx_data.columns:
                features[f'iv_{tenor}'] = fx_data[vol_col] / 100
                features[f'iv_{tenor}_ma20'] = features[f'iv_{tenor}'].rolling(20).mean()
                features[f'iv_{tenor}_std20'] = features[f'iv_{tenor}'].rolling(20).std()

            # Risk reversal and butterfly
            rr_col = f'{pair}25R{tenor} Curncy'
            if rr_col in fx_data.columns:
                features[f'rr25_{tenor}'] = fx_data[rr_col] / 100

            bf_col = f'{pair}25B{tenor} Curncy'
            if bf_col in fx_data.columns:
                features[f'bf25_{tenor}'] = fx_data[bf_col] / 100

        # IV term structure
        if f'iv_1M' in features.columns and f'iv_1Y' in features.columns:
            features['term_structure'] = features['iv_1Y'] - features['iv_1M']

        # IV vs RV
        if f'iv_1M' in features.columns and 'realized_vol_20d' in features.columns:
            features['iv_rv_premium'] = features['iv_1M'] - features['realized_vol_20d']

        return features.dropna()

    def create_training_samples(self, fx_data: pd.DataFrame, features: pd.DataFrame,
                               pair: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Create training samples with different strikes and tenors"""
        X_list = []
        y_list = []

        spot_col = f'{pair} Curncy'

        for date in features.index[60:]:  # Need history for features
            if date not in fx_data.index:
                continue

            spot = fx_data.loc[date, spot_col]

            # Sample different strikes and tenors
            for tenor, T in [('1M', 1/12), ('3M', 0.25), ('6M', 0.5)]:
                vol_col = f'{pair}V{tenor} Curncy'
                if vol_col not in fx_data.columns:
                    continue

                atm_vol = fx_data.loc[date, vol_col] / 100
                if pd.isna(atm_vol) or atm_vol <= 0:
                    continue

                # Get smile parameters
                rr_col = f'{pair}25R{tenor} Curncy'
                bf_col = f'{pair}25B{tenor} Curncy'
                rr_25 = fx_data.loc[date, rr_col] / 100 if rr_col in fx_data.columns else 0
                bf_25 = fx_data.loc[date, bf_col] / 100 if bf_col in fx_data.columns else 0

                # Sample strikes around ATM
                for moneyness in [0.90, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10]:
                    K = spot * moneyness

                    # Features for this sample
                    sample_features = features.loc[date].copy()
                    sample_features['moneyness'] = moneyness
                    sample_features['time_to_maturity'] = T
                    sample_features['strike'] = K

                    # Calculate "true" price using GVV model (our target)
                    # This uses smile information
                    smile_vol = atm_vol
                    if moneyness < 0.97:
                        smile_vol = atm_vol + bf_25 - rr_25/2
                    elif moneyness > 1.03:
                        smile_vol = atm_vol + bf_25 + rr_25/2
                    else:
                        skew = rr_25 * (moneyness - 1.0) * 4
                        smile_vol = atm_vol + bf_25 + skew

                    # Simple rates assumption
                    r_d = 0.05
                    r_f = 0.01

                    # Black-Scholes price with smile vol
                    option_price = BlackScholes.price(spot, K, T, r_d, r_f, smile_vol, 'call')

                    X_list.append(sample_features)
                    y_list.append(option_price / spot)  # Normalize by spot

        if len(X_list) == 0:
            return pd.DataFrame(), pd.Series()

        X = pd.DataFrame(X_list)
        y = pd.Series(y_list)

        return X, y

    def train(self, fx_data: pd.DataFrame):
        """Train the model on historical data"""
        try:
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            print("Training LightGBM models...")

            for pair in Config.PAIRS:
                print(f"  Training model for {pair}...")

                # Prepare features
                features = self.prepare_training_features(fx_data, pair)
                if len(features) < 100:
                    print(f"    Insufficient data for {pair}")
                    continue

                # Create training samples
                X, y = self.create_training_samples(fx_data, features, pair)
                if len(X) < 100:
                    print(f"    Insufficient samples for {pair}")
                    continue

                # Remove any remaining NaN
                mask = ~(X.isna().any(axis=1) | y.isna())
                X = X[mask]
                y = y[mask]

                if len(X) < 100:
                    continue

                # Split for validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=False
                )

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                # Store feature columns
                if self.feature_columns is None:
                    self.feature_columns = X.columns.tolist()

                # Train model
                lgb_train = lgb.Dataset(X_train_scaled, y_train)
                lgb_val = lgb.Dataset(X_val_scaled, y_val, reference=lgb_train)

                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.7,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'num_threads': 4
                }

                model = lgb.train(
                    params,
                    lgb_train,
                    valid_sets=[lgb_val],
                    num_boost_round=500,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )

                self.models[pair] = model
                self.scalers[pair] = scaler

                # Calculate validation metrics
                y_pred = model.predict(X_val_scaled, num_iteration=model.best_iteration)
                rmse = np.sqrt(np.mean((y_pred - y_val)**2))
                print(f"    Validation RMSE: {rmse:.6f}")

            self.is_trained = len(self.models) > 0
            print(f"  Training complete. Models trained for {list(self.models.keys())}")

        except ImportError:
            print("Warning: LightGBM not installed. Using fallback pricing.")
            self.is_trained = False

    def prepare_features(self, market_data: MarketData, pair: str,
                         strike: float, tenor: str, option_type: str,
                         historical_data: pd.DataFrame = None) -> np.ndarray:
        """Prepare features for prediction - matching training features"""
        if not self.is_trained or pair not in self.models:
            return np.array([])

        features = {}
        spot = market_data.spot[pair]
        T = self._tenor_to_years(tenor)

        # Match the features used in training
        features['spot'] = spot
        features['log_spot'] = np.log(spot)

        # Use historical data for price features
        if historical_data is not None and len(historical_data) > 20:
            features['spot_return_1d'] = (spot / historical_data['spot'].iloc[-2] - 1) if len(historical_data) > 1 else 0
            features['spot_return_5d'] = (spot / historical_data['spot'].iloc[-6] - 1) if len(historical_data) > 5 else 0
            features['spot_return_20d'] = (spot / historical_data['spot'].iloc[-21] - 1) if len(historical_data) > 20 else 0

            # Realized vol
            log_returns = np.log(historical_data['spot'] / historical_data['spot'].shift(1))
            for window in [5, 10, 20, 60]:
                if len(log_returns) > window:
                    features[f'realized_vol_{window}d'] = log_returns.tail(window).std() * np.sqrt(252)
                else:
                    features[f'realized_vol_{window}d'] = 0.1
        else:
            # Default values
            features['spot_return_1d'] = 0
            features['spot_return_5d'] = 0
            features['spot_return_20d'] = 0
            for window in [5, 10, 20, 60]:
                features[f'realized_vol_{window}d'] = 0.1

        # Current IV features
        for t in ['1M', '3M', '6M', '1Y']:
            if t in market_data.atm_vols[pair]:
                features[f'iv_{t}'] = market_data.atm_vols[pair][t]
                features[f'iv_{t}_ma20'] = features[f'iv_{t}']  # Simplified
                features[f'iv_{t}_std20'] = 0.01  # Simplified
            else:
                features[f'iv_{t}'] = 0.1
                features[f'iv_{t}_ma20'] = 0.1
                features[f'iv_{t}_std20'] = 0.01

            if t in market_data.rr_25[pair]:
                features[f'rr25_{t}'] = market_data.rr_25[pair][t]
            else:
                features[f'rr25_{t}'] = 0

            if t in market_data.bf_25[pair]:
                features[f'bf25_{t}'] = market_data.bf_25[pair][t]
            else:
                features[f'bf25_{t}'] = 0

        # Term structure
        features['term_structure'] = features['iv_1Y'] - features['iv_1M']

        # IV vs RV
        features['iv_rv_premium'] = features['iv_1M'] - features['realized_vol_20d']

        # Strike features
        features['moneyness'] = strike / spot
        features['time_to_maturity'] = T
        features['strike'] = strike

        # Create feature vector in correct order
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0))

        return np.array(feature_vector).reshape(1, -1)

    def price(self, S: float, K: float, T: float, r_d: float, r_f: float,
              features: np.ndarray, pair: str = None) -> float:
        """Price option using trained model"""
        if not self.is_trained or len(features) == 0 or pair not in self.models:
            # Fallback to Black-Scholes
            return BlackScholes.price(S, K, T, r_d, r_f, 0.1, 'call')

        # Scale features
        features_scaled = self.scalers[pair].transform(features)

        # Predict normalized price
        normalized_price = self.models[pair].predict(
            features_scaled,
            num_iteration=self.models[pair].best_iteration
        )[0]

        # Denormalize
        return normalized_price * S

    def _tenor_to_years(self, tenor: str) -> float:
        """Convert tenor to years"""
        mapping = {
            '1W': 1/52, '2W': 2/52, '3W': 3/52,
            '1M': 1/12, '2M': 2/12, '3M': 3/12,
            '4M': 4/12, '6M': 6/12, '9M': 9/12,
            '1Y': 1.0
        }
        return mapping.get(tenor, 1/12)

# ==================== Strategy ====================
class OptionsArbitrageStrategy:
    """
    Main trading strategy that:
    1. Prices options using Black-Scholes at various strikes
    2. Compares with model prices (GVV, SABR, etc.)
    3. Trades mispricings
    4. Delta hedges positions
    """

    def __init__(self, model_name: str, use_iv_filter: bool = False,
                 delta_model: str = None):
        self.model_name = model_name
        self.use_iv_filter = use_iv_filter
        self.delta_model = delta_model or Config.DELTA_MODEL
        self.book = TradingBook()
        self.sabr_model = SABRModel()
        self.lgbm_model = LGBMPricer()

        # Initialize delta model if different from BS
        self.delta_sabr_model = SABRModel() if self.delta_model == 'SABR' else None

        # Track performance
        self.equity_curve = []
        self.trades_log = []
        self.hedges_log = []
        self.daily_stats = []

    def _get_rates_for_pair(self, pair: str, market_data: MarketData) -> Tuple[float, float]:
        """Get appropriate interest rates for currency pair"""
        if pair == 'USDJPY':
            r_d = market_data.rates.get('USD', 0.05)
            r_f = market_data.rates.get('JPY', 0.01)
        elif pair == 'GBPNZD':
            r_d = market_data.rates.get('GBP', 0.04)
            r_f = market_data.rates.get('NZD', 0.03)
        elif pair == 'USDCAD':
            r_d = market_data.rates.get('USD', 0.05)
            r_f = market_data.rates.get('CAD', 0.04)
        else:
            r_d = 0.05
            r_f = 0.01
        return r_d, r_f

    def _calculate_delta(self, spot: float, strike: float, T: float, r_d: float, r_f: float,
                        vol: float, option_type: str, market_data: MarketData,
                        pair: str, tenor: str) -> float:
        """Calculate delta using the specified model"""

        if self.delta_model == 'BS' or self.delta_model == 'GK':
            # Black-Scholes / Garman-Kohlhagen delta
            return BlackScholes.delta(spot, strike, T, r_d, r_f, vol, option_type)

        elif self.delta_model == 'GVV':
            # GVV delta calculation
            rr_25 = market_data.rr_25[pair].get(tenor, 0)
            bf_25 = market_data.bf_25[pair].get(tenor, 0)

            # Calculate GVV-adjusted volatility for this strike
            moneyness = strike / spot
            if moneyness < 0.95:
                smile_vol = vol + bf_25 - rr_25/2 + 0.002 * (0.95 - moneyness)
            elif moneyness > 1.05:
                smile_vol = vol + bf_25 + rr_25/2 + 0.002 * (moneyness - 1.05)
            else:
                skew_adjustment = rr_25 * (moneyness - 1.0) * 4
                smile_vol = vol + bf_25 + skew_adjustment

            # Use adjusted vol for delta
            return BlackScholes.delta(spot, strike, T, r_d, r_f, smile_vol, option_type)

        elif self.delta_model == 'SABR':
            # SABR delta calculation
            if self.delta_sabr_model.alpha is None:
                # Calibrate SABR if not done
                strikes = np.array([spot*0.9, spot, spot*1.1])
                vols = np.array([vol*1.1, vol, vol*1.1])
                self.delta_sabr_model.calibrate(strikes, vols, spot, T)

            # Calculate SABR vol for this strike
            F = spot * np.exp((r_d - r_f) * T)
            if abs(F - strike) < 0.001:
                sabr_vol = self.delta_sabr_model.alpha
            else:
                z = (self.delta_sabr_model.nu/self.delta_sabr_model.alpha) * \
                    (F*strike)**((1-self.delta_sabr_model.beta)/2) * np.log(F/strike)
                x = np.log((np.sqrt(1 - 2*self.delta_sabr_model.rho*z + z**2) + z -
                          self.delta_sabr_model.rho)/(1 - self.delta_sabr_model.rho))
                sabr_vol = self.delta_sabr_model.alpha * (z/x if abs(x) > 0.001 else 1.0)

            # Use SABR vol for delta
            return BlackScholes.delta(spot, strike, T, r_d, r_f, sabr_vol, option_type)

        else:
            # Default to Black-Scholes
            return BlackScholes.delta(spot, strike, T, r_d, r_f, vol, option_type)

    def run_daily(self, market_data: MarketData, historical_data: Dict[str, pd.DataFrame]):
        """Execute daily trading logic"""

        # 1. Update existing positions
        self._update_positions(market_data)

        # 2. Find new trading opportunities
        signals = self._find_opportunities(market_data, historical_data)

        # 3. Execute trades
        self._execute_trades(signals, market_data)

        # 4. Delta hedge
        self._delta_hedge(market_data)

        # 5. Calculate interest on cash/margin
        self._calculate_interest(market_data)

        # 6. Record daily statistics
        self._record_stats(market_data)

    def _find_opportunities(self, market_data: MarketData,
                           historical_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Find mispriced options to trade"""
        signals = []

        for pair in Config.PAIRS:
            if pair not in market_data.spot:
                continue

            spot = market_data.spot[pair]
            r_d, r_f = self._get_rates_for_pair(pair, market_data)

            # Check IV filter if enabled
            if self.use_iv_filter and pair in historical_data:
                hist = historical_data[pair]
                if len(hist) < Config.IV_MA_DAYS:
                    continue

            for tenor in Config.TENORS:
                if tenor not in market_data.atm_vols[pair]:
                    continue

                atm_vol = market_data.atm_vols[pair][tenor]
                T = self._tenor_to_years(tenor)

                # Apply IV filter
                if self.use_iv_filter:
                    iv_ma = hist[f'atm_vol_{tenor}'].tail(Config.IV_MA_DAYS).mean()
                    if pd.isna(iv_ma):
                        continue

                # Evaluate at different deltas
                for delta in Config.DELTAS:
                    for option_type in ['call', 'put']:
                        # Calculate strike from delta
                        adj_delta = delta if option_type == 'call' else -delta
                        strike = BlackScholes.strike_from_delta(
                            spot, adj_delta, T, r_d, r_f, atm_vol, option_type
                        )

                        # For GVV model: compare smile-adjusted price vs flat ATM vol price
                        # For other models: compare model price vs market smile price
                        if self.model_name == 'GVV':
                            # Market price with flat ATM vol (no smile)
                            market_price = BlackScholes.price(
                                spot, strike, T, r_d, r_f, atm_vol, option_type
                            )
                            # GVV price with smile adjustment
                            model_price = self._calculate_model_price(
                                market_data, pair, spot, strike, T, r_d, r_f,
                                tenor, option_type, historical_data
                            )
                        else:
                            # Get market implied vol from smile
                            market_vol = self._get_smile_vol(
                                market_data, pair, tenor, strike/spot
                            )
                            # Market price with smile
                            market_price = BlackScholes.price(
                                spot, strike, T, r_d, r_f, market_vol, option_type
                            )
                            # Model price
                            model_price = self._calculate_model_price(
                                market_data, pair, spot, strike, T, r_d, r_f,
                                tenor, option_type, historical_data
                            )

                        # Check for mispricing
                        if market_price > 0:
                            mispricing = (model_price - market_price) / market_price

                            # Lower threshold for GVV to generate more signals
                            threshold = Config.MISPRICING_THRESHOLD * 0.5 if self.model_name == 'GVV' else Config.MISPRICING_THRESHOLD

                            if abs(mispricing) > threshold:
                                # Apply IV filter
                                if self.use_iv_filter:
                                    if mispricing > 0 and atm_vol < iv_ma:
                                        continue  # Skip long if IV below MA
                                    if mispricing < 0 and atm_vol > iv_ma:
                                        continue  # Skip short if IV above MA

                                # Use appropriate vol for the signal
                                signal_vol = atm_vol if self.model_name == 'GVV' else market_vol

                                signal = {
                                    'pair': pair,
                                    'tenor': tenor,
                                    'strike': strike,
                                    'option_type': option_type,
                                    'spot': spot,
                                    'market_price': market_price,
                                    'model_price': model_price,
                                    'mispricing': mispricing,
                                    'direction': 1 if mispricing > 0 else -1,
                                    'vol': signal_vol,
                                    'delta': delta,
                                    'r_d': r_d,
                                    'r_f': r_f
                                }
                                signals.append(signal)

        # Sort by absolute mispricing and limit positions
        signals.sort(key=lambda x: abs(x['mispricing']), reverse=True)
        max_new = Config.MAX_POSITIONS - self.book.get_num_positions()
        return signals[:max_new]

    def _calculate_model_price(self, market_data: MarketData, pair: str,
                              S: float, K: float, T: float, r_d: float, r_f: float,
                              tenor: str, option_type: str,
                              historical_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate theoretical price using selected model"""

        if self.model_name == 'GVV':
            rr_25 = market_data.rr_25[pair].get(tenor, 0)
            bf_25 = market_data.bf_25[pair].get(tenor, 0)
            atm_vol = market_data.atm_vols[pair].get(tenor, 0.1)
            return GVVModel.price(S, K, T, r_d, r_f, atm_vol, rr_25, bf_25, option_type)

        elif self.model_name == 'SABR':
            # Use properly calibrated SABR model
            return self.sabr_model.price(S, K, T, r_d, r_f, pair, tenor, option_type)

        elif self.model_name == 'LGBM':
            features = self.lgbm_model.prepare_features(
                market_data, pair, K, tenor, option_type,
                historical_data.get(pair)
            )
            return self.lgbm_model.price(S, K, T, r_d, r_f, features, pair)

        else:  # GK/Black-Scholes
            atm_vol = market_data.atm_vols[pair].get(tenor, 0.1)
            return BlackScholes.price(S, K, T, r_d, r_f, atm_vol, option_type)

    def _get_smile_vol(self, market_data: MarketData, pair: str,
                      tenor: str, moneyness: float) -> float:
        """Get implied vol from smile for given moneyness"""
        atm_vol = market_data.atm_vols[pair].get(tenor, 0.1)
        rr_25 = market_data.rr_25[pair].get(tenor, 0)
        bf_25 = market_data.bf_25[pair].get(tenor, 0)

        # Simple smile interpolation
        if moneyness < 0.95:  # OTM put
            return atm_vol + bf_25 - rr_25/2
        elif moneyness > 1.05:  # OTM call
            return atm_vol + bf_25 + rr_25/2
        else:  # Near ATM
            skew = rr_25 * (moneyness - 1.0) * 4
            return atm_vol + bf_25 + skew

    def _execute_trades(self, signals: List[Dict], market_data: MarketData):
        """Execute trades based on signals with realistic transaction costs"""
        for signal in signals:
            # Get rates from signal
            r_d = signal['r_d']
            r_f = signal['r_f']

            # Calculate option price with transaction costs
            T = self._tenor_to_years(signal['tenor'])
            is_buy = signal['direction'] > 0

            # Get price with vol spread
            trade_price, vol_cost = BlackScholes.price_with_vol_spread(
                signal['spot'], signal['strike'], T, r_d, r_f,
                signal['vol'], signal['option_type'], signal['tenor'], is_buy
            )

            # Calculate notional - this should be a reasonable size per trade
            # Use a fraction of capital based on position limits
            max_position_value = Config.INITIAL_CAPITAL * 0.02  # 2% per position
            notional = min(max_position_value, Config.INITIAL_CAPITAL * 0.01)  # Start with 1% of capital

            # Number of contracts (each contract is for notional/spot units of currency)
            num_contracts = notional / signal['spot']

            # Total premium for the options
            total_option_premium = trade_price * notional

            # Add brokerage and clearing costs
            brokerage_cost = notional * Config.TRANSACTION_COSTS['brokerage']
            clearing_cost = notional * Config.TRANSACTION_COSTS['clearing']
            total_transaction_cost = vol_cost * notional + brokerage_cost + clearing_cost

            # Total cost including all fees
            if is_buy:
                total_cost = total_option_premium + total_transaction_cost
            else:
                total_cost = total_transaction_cost  # For selling, we receive premium

            # Check margin/capital
            if signal['direction'] > 0:  # Buying
                if self.book.cash < total_cost:
                    continue  # Not enough cash
            else:  # Selling (need margin)
                required_margin = total_option_premium * 0.2  # 20% margin for short
                if self.book.used_margin + required_margin > Config.INITIAL_CAPITAL * Config.MAX_LEVERAGE:
                    continue  # Exceeds leverage limit

            # Create position
            expiry = market_data.date + pd.Timedelta(days=int(T*365))

            position = OptionPosition(
                entry_date=market_data.date,
                expiry_date=expiry,
                pair=signal['pair'],
                strike=signal['strike'],
                option_type=signal['option_type'],
                position_size=signal['direction'] * num_contracts,  # Signed number of contracts
                entry_price=trade_price,
                entry_spot=signal['spot'],
                entry_vol=signal['vol'],
                model_used=self.model_name,
                transaction_costs=total_transaction_cost
            )

            # Update greeks
            position.current_delta = self._calculate_delta(
                signal['spot'], signal['strike'], T, r_d, r_f,
                signal['vol'], signal['option_type'], market_data,
                signal['pair'], signal['tenor']
            )
            position.current_vega = BlackScholes.vega(
                signal['spot'], signal['strike'], T, r_d, r_f, signal['vol']
            )

            # Add to book
            self.book.options.append(position)
            self.book.total_transaction_costs += total_transaction_cost

            # Update cash/margin
            if signal['direction'] > 0:
                self.book.cash -= total_cost
            else:
                self.book.cash += total_option_premium - total_transaction_cost
                self.book.used_margin += required_margin

            # Log trade
            self.trades_log.append({
                'date': market_data.date,
                'pair': signal['pair'],
                'strike': signal['strike'],
                'option_type': signal['option_type'],
                'direction': signal['direction'],
                'notional': notional,
                'num_contracts': num_contracts,
                'premium': total_option_premium,
                'transaction_costs': total_transaction_cost,
                'mispricing': signal['mispricing']
            })

    def _delta_hedge(self, market_data: MarketData):
        """Delta hedge the portfolio to stay within limits"""
        for pair in Config.PAIRS:
            if pair not in market_data.spot:
                continue

            net_delta = self.book.get_net_delta(pair)
            spot_price = market_data.spot[pair]

            # Check if delta exceeds limits (in dollar terms)
            delta_limit = Config.MAX_DELTA_EXPOSURE * Config.INITIAL_CAPITAL

            if abs(net_delta) > delta_limit:
                # Need to hedge - calculate spot amount needed
                # Delta hedge amount is in units of base currency
                hedge_amount = -net_delta / spot_price

                # Calculate transaction cost for spot hedge
                spot_spread = Config.TRANSACTION_COSTS['spot'].get(pair, 0.0002)
                hedge_cost = abs(hedge_amount * spot_price) * spot_spread

                # Update spot hedge position (in units of base currency)
                if pair not in self.book.spot_hedge:
                    self.book.spot_hedge[pair] = 0
                self.book.spot_hedge[pair] += hedge_amount

                # Update cash and costs
                self.book.cash -= hedge_cost
                self.book.total_transaction_costs += hedge_cost

                # Log hedge
                self.hedges_log.append({
                    'date': market_data.date,
                    'pair': pair,
                    'hedge_amount': hedge_amount,
                    'hedge_delta': hedge_amount * spot_price,  # Dollar delta hedged
                    'spot_price': spot_price,
                    'cost': hedge_cost,
                    'net_delta_before': net_delta,
                    'net_delta_after': self.book.get_net_delta(pair)
                })

    def _update_positions(self, market_data: MarketData):
        """Update existing positions with current market data"""
        positions_to_remove = []

        for i, pos in enumerate(self.book.options):
            # Check if expired
            if market_data.date >= pos.expiry_date:
                # Calculate final P&L
                spot = market_data.spot[pos.pair]
                if pos.option_type == 'call':
                    payoff = max(0, spot - pos.strike)
                else:
                    payoff = max(0, pos.strike - spot)

                pos.pnl = (payoff - pos.entry_price) * pos.position_size * pos.entry_spot - pos.transaction_costs
                self.book.cash += payoff * pos.position_size * pos.entry_spot

                if pos.position_size < 0:  # Release margin for shorts
                    self.book.used_margin -= pos.entry_price * pos.entry_spot * 0.2

                positions_to_remove.append(i)
            else:
                # Update greeks and mark-to-market
                spot = market_data.spot[pos.pair]
                T = (pos.expiry_date - market_data.date).days / 365
                vol = self._get_smile_vol(market_data, pos.pair,
                                         self._years_to_tenor(T), pos.strike/spot)

                r_d, r_f = self._get_rates_for_pair(pos.pair, market_data)

                pos.current_delta = self._calculate_delta(
                    spot, pos.strike, T, r_d, r_f, vol, pos.option_type,
                    market_data, pos.pair, self._years_to_tenor(T)
                )
                pos.current_vega = BlackScholes.vega(
                    spot, pos.strike, T, r_d, r_f, vol
                )
                pos.current_value = BlackScholes.price(
                    spot, pos.strike, T, r_d, r_f, vol, pos.option_type
                )
                pos.pnl = (pos.current_value - pos.entry_price) * pos.position_size * pos.entry_spot - pos.transaction_costs

        # Remove expired positions
        for i in reversed(positions_to_remove):
            del self.book.options[i]

    def _calculate_interest(self, market_data: MarketData):
        """Calculate daily interest on cash/margin"""
        daily_rate = market_data.rates.get('USD', 0.05) / 365

        if self.book.cash > 0:
            # Earn interest on cash
            self.book.cash *= (1 + daily_rate)
        else:
            # Pay interest on borrowed funds
            self.book.cash *= (1 + daily_rate * 1.5)  # Higher rate for borrowing

    def _record_stats(self, market_data: MarketData):
        """Record daily statistics"""
        total_value = self.book.cash + sum(pos.current_value * pos.position_size * pos.entry_spot
                                          for pos in self.book.options)

        self.equity_curve.append({
            'date': market_data.date,
            'equity': total_value,
            'cash': self.book.cash,
            'num_positions': self.book.get_num_positions(),
            'net_delta': sum(self.book.get_net_delta(pair) for pair in Config.PAIRS),
            'net_vega': self.book.get_net_vega(),
            'used_margin': self.book.used_margin,
            'total_transaction_costs': self.book.total_transaction_costs
        })

    def _tenor_to_years(self, tenor: str) -> float:
        """Convert tenor to years"""
        mapping = {
            '1W': 1/52, '2W': 2/52, '3W': 3/52,
            '1M': 1/12, '2M': 2/12, '3M': 3/12,
            '4M': 4/12, '6M': 6/12, '9M': 9/12,
            '1Y': 1.0
        }
        return mapping.get(tenor, 1/12)

    def _years_to_tenor(self, years: float) -> str:
        """Convert years to nearest tenor"""
        if years <= 1/52:
            return '1W'
        elif years <= 1/12:
            return '1M'
        elif years <= 3/12:
            return '3M'
        elif years <= 6/12:
            return '6M'
        else:
            return '1Y'

# ==================== Data Loader ====================
class DataLoader:
    """Load and prepare market data"""

    @staticmethod
    def load_fx_data(path: Path) -> pd.DataFrame:
        """Load FX data from parquet file"""
        return pd.read_parquet(path)

    @staticmethod
    def load_curves_data(path: Path) -> pd.DataFrame:
        """Load discount curves from parquet file"""
        if path.exists():
            return pd.read_parquet(path)
        else:
            print(f"Warning: Discount curves not found at {path}")
            return None

    @staticmethod
    def get_rates_from_curves(curves_data: pd.DataFrame, date: pd.Timestamp) -> Dict[str, float]:
        """Extract rates for each currency from discount curves"""
        if curves_data is None:
            # Fallback to default rates
            return {'USD': 0.05, 'JPY': 0.01, 'GBP': 0.04, 'NZD': 0.03, 'CAD': 0.04}

        # Get rates for the specific date
        date_curves = curves_data[curves_data['date'] == date]

        if date_curves.empty:
            # Try nearest date
            nearest_date = curves_data['date'].iloc[(curves_data['date'] - date).abs().argsort()[:1]].values[0]
            date_curves = curves_data[curves_data['date'] == nearest_date]

        if date_curves.empty:
            return {'USD': 0.05, 'JPY': 0.01, 'GBP': 0.04, 'NZD': 0.03, 'CAD': 0.04}

        # Get 3M rate as representative
        rates = {}

        # USD rate from the curves
        usd_3m = date_curves[date_curves['tenor'] == '3M']['interpolated_rate'].values
        rates['USD'] = usd_3m[0] if len(usd_3m) > 0 else 0.05

        # For other currencies, apply spreads (simplified)
        rates['JPY'] = rates['USD'] - 0.04  # Japan typically lower rates
        rates['GBP'] = rates['USD'] - 0.01  # UK similar to US
        rates['NZD'] = rates['USD'] - 0.02  # NZ slightly lower
        rates['CAD'] = rates['USD'] - 0.01  # Canada similar to US

        return rates

    @staticmethod
    def prepare_market_data(fx_data: pd.DataFrame, date: pd.Timestamp,
                           curves_data: pd.DataFrame = None) -> MarketData:
        """Prepare market data for a specific date"""
        if date not in fx_data.index:
            return None

        row = fx_data.loc[date]

        # Get rates from discount curves
        rates = DataLoader.get_rates_from_curves(curves_data, date) if curves_data is not None else {
            'USD': 0.05, 'JPY': 0.01, 'GBP': 0.04, 'NZD': 0.03, 'CAD': 0.04
        }

        market = MarketData(
            date=date,
            spot={},
            forwards={},
            atm_vols={},
            rr_25={},
            bf_25={},
            rr_10={},
            bf_10={},
            rates=rates
        )

        # Parse data for each pair
        for pair in Config.PAIRS:
            # Spot
            spot_col = f'{pair} Curncy'
            if spot_col in row.index:
                market.spot[pair] = row[spot_col]

            # Forwards, vols, etc.
            market.forwards[pair] = {}
            market.atm_vols[pair] = {}
            market.rr_25[pair] = {}
            market.bf_25[pair] = {}
            market.rr_10[pair] = {}
            market.bf_10[pair] = {}

            for tenor in Config.TENORS:
                # ATM vol
                vol_col = f'{pair}V{tenor} Curncy'
                if vol_col in row.index:
                    market.atm_vols[pair][tenor] = row[vol_col] / 100

                # Risk reversal and butterfly
                rr_col = f'{pair}25R{tenor} Curncy'
                if rr_col in row.index:
                    market.rr_25[pair][tenor] = row[rr_col] / 100

                bf_col = f'{pair}25B{tenor} Curncy'
                if bf_col in row.index:
                    market.bf_25[pair][tenor] = row[bf_col] / 100

        return market

# Rest of the code remains the same (Backtester class and main function)...
# [Continue with the rest of the original code from line 906 onwards]

# ==================== Backtester ====================
class Backtester:
    """Run backtests for all models"""

    def __init__(self, fx_data: pd.DataFrame, curves_data: pd.DataFrame = None):
        self.fx_data = fx_data
        self.curves_data = curves_data
        self.results = {}
        self.trained_lgbm = None
        self.trained_sabr = None

    def train_models(self, train_data: pd.DataFrame):
        """Train ML and calibrate models using training data"""
        print("\n" + "=" * 80)
        print("TRAINING MODELS ON HISTORICAL DATA")
        print("=" * 80)

        # Train LightGBM
        self.trained_lgbm = LGBMPricer()
        self.trained_lgbm.train(train_data)

        # Calibrate SABR
        self.trained_sabr = SABRModel()
        self.trained_sabr.calibrate_from_training_data(train_data)

        print("\nModel training complete!")

    def run_all_models(self, start_date: pd.Timestamp, end_date: pd.Timestamp,
                      delta_model: str = None):
        """Run backtest for all models"""
        models = ['GK', 'GVV', 'SABR', 'LGBM']

        # Use provided delta model or default from config
        delta_model = delta_model or Config.DELTA_MODEL

        for model in models:
            print(f"\nRunning backtest for {model} (Delta model: {delta_model})...")
            strategy = OptionsArbitrageStrategy(model_name=model, delta_model=delta_model)

            # Inject trained models
            if model == 'LGBM' and self.trained_lgbm is not None:
                strategy.lgbm_model = self.trained_lgbm
            elif model == 'SABR' and self.trained_sabr is not None:
                strategy.sabr_model = self.trained_sabr

            self._run_single_backtest(strategy, start_date, end_date)
            self.results[f'{model}_delta{delta_model}'] = strategy.equity_curve

        # Also run GK with IV filter using default delta model
        print(f"\nRunning backtest for GK with IV filter (Delta model: {delta_model})...")
        strategy = OptionsArbitrageStrategy(model_name='GK', use_iv_filter=True,
                                           delta_model=delta_model)
        self._run_single_backtest(strategy, start_date, end_date)
        self.results[f'GK_IVFilter_delta{delta_model}'] = strategy.equity_curve

        # Additional: GK with IV filter using SABR delta model
        print(f"\nRunning backtest for GK with IV filter (Delta model: SABR)...")
        strategy = OptionsArbitrageStrategy(model_name='GK', use_iv_filter=True,
                                           delta_model='SABR')
        # Inject trained SABR for delta calculation
        if self.trained_sabr is not None:
            strategy.delta_sabr_model = self.trained_sabr
        self._run_single_backtest(strategy, start_date, end_date)
        self.results[f'GK_IVFilter_deltaSABR'] = strategy.equity_curve

    def _run_single_backtest(self, strategy: OptionsArbitrageStrategy,
                            start_date: pd.Timestamp, end_date: pd.Timestamp):
        """Run backtest for a single strategy"""
        dates = self.fx_data.index[(self.fx_data.index >= start_date) &
                                   (self.fx_data.index <= end_date)]

        # Keep historical data for features
        historical_data = {}
        for pair in Config.PAIRS:
            historical_data[pair] = pd.DataFrame()

        for i, date in enumerate(dates):
            if i % 50 == 0:
                print(f"  Processing {i}/{len(dates)} days...")

            # Prepare market data with curves
            market_data = DataLoader.prepare_market_data(self.fx_data, date, self.curves_data)
            if market_data is None:
                continue

            # Update historical data
            for pair in Config.PAIRS:
                if pair in market_data.spot:
                    new_row = pd.DataFrame({
                        'date': [date],
                        'spot': [market_data.spot[pair]]
                    })
                    for tenor in Config.TENORS:
                        if tenor in market_data.atm_vols[pair]:
                            new_row[f'atm_vol_{tenor}'] = market_data.atm_vols[pair][tenor]

                    historical_data[pair] = pd.concat([historical_data[pair], new_row],
                                                     ignore_index=True)

            # Run daily strategy
            strategy.run_daily(market_data, historical_data)

    def calculate_performance_metrics(self) -> pd.DataFrame:
        """Calculate performance metrics for all strategies"""
        metrics = []

        for model, equity_curve in self.results.items():
            if not equity_curve:
                continue

            df = pd.DataFrame(equity_curve)
            returns = df['equity'].pct_change().dropna()

            # Calculate transaction costs impact
            total_costs = df['total_transaction_costs'].iloc[-1] if 'total_transaction_costs' in df else 0

            metrics.append({
                'Model': model,
                'Total Return': (df['equity'].iloc[-1] / Config.INITIAL_CAPITAL - 1) * 100,
                'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                'Max Drawdown': ((df['equity'] / df['equity'].expanding().max() - 1).min()) * 100,
                'Avg Positions': df['num_positions'].mean(),
                'Total Trades': len([e for e in equity_curve if e['num_positions'] > 0]),
                'Transaction Costs': total_costs,
                'Costs as % of Capital': (total_costs / Config.INITIAL_CAPITAL) * 100
            })

        return pd.DataFrame(metrics)

    def plot_results(self):
        """Plot equity curves and statistics"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Equity curves
        ax = axes[0, 0]
        for model, equity_curve in self.results.items():
            if equity_curve:
                df = pd.DataFrame(equity_curve)
                ax.plot(df['date'], df['equity'], label=model)
        ax.set_title('Equity Curves')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.legend()
        ax.grid(True)

        # Drawdown
        ax = axes[0, 1]
        for model, equity_curve in self.results.items():
            if equity_curve:
                df = pd.DataFrame(equity_curve)
                dd = (df['equity'] / df['equity'].expanding().max() - 1) * 100
                ax.plot(df['date'], dd, label=model)
        ax.set_title('Drawdown')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(True)

        # Rolling Delta Exposure
        ax = axes[1, 0]
        for model, equity_curve in self.results.items():
            if equity_curve:
                df = pd.DataFrame(equity_curve)
                ax.plot(df['date'], df['net_delta'], label=model, alpha=0.7)
        ax.set_title('Rolling Delta Exposure')
        ax.set_xlabel('Date')
        ax.set_ylabel('Net Delta ($)')
        ax.axhline(y=Config.MAX_DELTA_EXPOSURE * Config.INITIAL_CAPITAL,
                   color='r', linestyle='--', alpha=0.5, label='Max Delta Limit')
        ax.axhline(y=-Config.MAX_DELTA_EXPOSURE * Config.INITIAL_CAPITAL,
                   color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.legend()
        ax.grid(True)

        # Transaction costs over time
        ax = axes[1, 1]
        for model, equity_curve in self.results.items():
            if equity_curve:
                df = pd.DataFrame(equity_curve)
                if 'total_transaction_costs' in df:
                    ax.plot(df['date'], df['total_transaction_costs'], label=model)
        ax.set_title('Cumulative Transaction Costs')
        ax.set_xlabel('Date')
        ax.set_ylabel('Transaction Costs ($)')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(Config.RESULTS_DIR / 'backtest_results.png')
        plt.show()

# ==================== Main Execution ====================
def main():
    """Main execution function"""
    print("=" * 80)
    print("FX OPTIONS TRADING SYSTEM")
    print("=" * 80)

    # Create directories
    Config.DATA_DIR.mkdir(exist_ok=True)
    Config.RESULTS_DIR.mkdir(exist_ok=True)

    # Load FX data
    print("\nLoading FX data...")
    fx_data = DataLoader.load_fx_data(Config.FX_DATA_PATH)
    print(f"Loaded {len(fx_data)} days of data")
    print(f"Date range: {fx_data.index[0]} to {fx_data.index[-1]}")

    # Load discount curves
    print("\nLoading discount curves...")
    curves_data = DataLoader.load_curves_data(Config.CURVES_PATH)
    if curves_data is not None:
        print(f"Loaded {len(curves_data)} curve points")
        print(f"Tenors available: {sorted(curves_data['tenor'].unique())}")
    else:
        print("Discount curves not found. Using default rates.")

    # Split data
    n_days = len(fx_data)
    train_end = int(n_days * 0.8)

    train_data = fx_data.iloc[:train_end]
    test_data = fx_data.iloc[train_end:]

    print(f"\nTrain period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")

    # Display transaction cost structure
    print("\n" + "=" * 80)
    print("TRANSACTION COST STRUCTURE")
    print("=" * 80)
    print("\nSpot FX Spreads:")
    for pair, spread in Config.TRANSACTION_COSTS['spot'].items():
        print(f"  {pair}: {spread*10000:.1f} pips")

    print("\nOption Volatility Spreads:")
    for tenor, spread in Config.TRANSACTION_COSTS['options'].items():
        print(f"  {tenor}: {spread*100:.0f} bps")

    print(f"\nBrokerage: {Config.TRANSACTION_COSTS['brokerage']*10000:.1f} bps")
    print(f"Clearing: {Config.TRANSACTION_COSTS['clearing']*10000:.1f} bps")

    # Initialize backtester
    print(f"\nUsing delta model: {Config.DELTA_MODEL}")
    backtester = Backtester(fx_data, curves_data)

    # TRAIN MODELS USING TRAINING DATA
    backtester.train_models(train_data)

    # Run backtests on TEST data
    print("\n" + "=" * 80)
    print("RUNNING BACKTESTS ON TEST DATA")
    print("=" * 80)
    backtester.run_all_models(test_data.index[0], test_data.index[-1],
                              delta_model=Config.DELTA_MODEL)

    # Calculate and display metrics
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    metrics = backtester.calculate_performance_metrics()
    print(metrics.to_string(index=False))

    # Save results
    metrics.to_csv(Config.RESULTS_DIR / 'performance_metrics.csv', index=False)

    # Plot results
    print("\nGenerating plots...")
    backtester.plot_results()

    print("\n✅ Analysis complete!")
    print(f"Results saved to {Config.RESULTS_DIR}")

if __name__ == "__main__":
    main()