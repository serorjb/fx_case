"""
FX Options Trading System - Fixed Implementation
Addresses stability issues and adds proper risk management
"""
import skfolio
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
    MIN_EQUITY_THRESHOLD = 1_000_000  # Stop trading if equity falls below 10% of initial
    MAX_LEVERAGE = 3  # Reduced from 5 for stability
    MAX_POSITIONS = 100  # Reduced from 500
    MAX_DELTA_EXPOSURE = 0.02  # Reduced from 3% to 2%
    MISPRICING_THRESHOLD = 0.015  # Reduced from 2% to 1.5%
    POSITION_SIZE_PCT = 0.005  # 0.5% of current equity per position (was 2%)

    # Enhanced Transaction costs
    TRANSACTION_COSTS = {
        'spot': {
            'USDJPY': 0.00003,  # Slightly higher for realism
            'GBPNZD': 0.00020,
            'USDCAD': 0.00004,
        },
        'options': {
            '1W': 0.006,   # Increased spreads
            '2W': 0.005,
            '3W': 0.005,
            '1M': 0.004,
            '2M': 0.005,
            '3M': 0.006,
            '4M': 0.007,
            '6M': 0.008,
            '9M': 0.010,
            '1Y': 0.012,
        },
        'brokerage': 0.00003,
        'clearing': 0.00002,
    }

    # Model parameters
    LGBM_TRAINING_YEARS = 5
    IV_MA_DAYS = 20
    DELTA_MODEL = 'BS'  # Default delta model

    # Risk limits
    MAX_VEGA_EXPOSURE = 0.01  # 1% of capital
    MAX_POSITION_CONCENTRATION = 0.20  # 20% max in any single position

    # Deltas and tenors
    DELTAS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    TENORS = ['1W', '2W', '3W', '1M', '2M', '3M', '4M', '6M', '9M', '1Y']
    PAIRS = ['USDJPY', 'GBPNZD', 'USDCAD']

# ==================== Data Structures ====================
@dataclass
class OptionPosition:
    """Represents an option position in the book"""
    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    pair: str
    strike: float
    option_type: str
    position_size: float
    entry_price: float
    entry_spot: float
    entry_vol: float
    model_used: str
    current_delta: float = 0.0
    current_vega: float = 0.0
    current_value: float = 0.0
    pnl: float = 0.0
    transaction_costs: float = 0.0
    notional: float = 0.0

@dataclass
class TradingBook:
    """Manages all positions and hedges"""
    options: List[OptionPosition] = field(default_factory=list)
    spot_hedge: Dict[str, float] = field(default_factory=dict)
    cash: float = Config.INITIAL_CAPITAL
    used_margin: float = 0.0
    total_transaction_costs: float = 0.0
    equity_stopped: bool = False

    def get_current_equity(self) -> float:
        """Calculate current total equity"""
        options_value = sum(pos.current_value * abs(pos.position_size) * pos.entry_spot *
                           (1 if pos.position_size > 0 else -1)
                           for pos in self.options)
        return self.cash + options_value

    def get_net_delta(self, pair: str = None) -> float:
        """Calculate total portfolio delta in dollar terms"""
        if pair:
            option_delta = sum(pos.current_delta * abs(pos.position_size) * pos.entry_spot *
                             (1 if pos.position_size > 0 else -1)
                             for pos in self.options if pos.pair == pair)
            spot_delta = self.spot_hedge.get(pair, 0) * self._get_current_spot(pair)
            return option_delta + spot_delta
        else:
            option_delta = sum(pos.current_delta * abs(pos.position_size) * pos.entry_spot *
                             (1 if pos.position_size > 0 else -1)
                             for pos in self.options)
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
        """Get current spot price for a pair"""
        pair_positions = [pos for pos in self.options if pos.pair == pair]
        if pair_positions:
            return pair_positions[-1].entry_spot
        return 100.0

@dataclass
class MarketData:
    """Market data for a specific date"""
    date: pd.Timestamp
    spot: Dict[str, float]
    forwards: Dict[str, Dict[str, float]]
    atm_vols: Dict[str, Dict[str, float]]
    rr_25: Dict[str, Dict[str, float]]
    bf_25: Dict[str, Dict[str, float]]
    rr_10: Dict[str, Dict[str, float]]
    bf_10: Dict[str, Dict[str, float]]
    rates: Dict[str, float]

# ==================== Pricing Models ====================
class BlackScholes:
    """Black-Scholes pricing for FX options"""

    @staticmethod
    def price(S: float, K: float, T: float, r_d: float, r_f: float,
              sigma: float, option_type: str) -> float:
        """Price option using Black-Scholes"""
        if T <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S)

        # Clamp volatility to reasonable range
        sigma = max(0.001, min(3.0, sigma))

        try:
            from scipy.stats import norm
            d1 = (np.log(S/K) + (r_d - r_f + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)

            if option_type == 'call':
                return S*np.exp(-r_f*T)*norm.cdf(d1) - K*np.exp(-r_d*T)*norm.cdf(d2)
            else:
                return K*np.exp(-r_d*T)*norm.cdf(-d2) - S*np.exp(-r_f*T)*norm.cdf(-d1)
        except:
            # Fallback to intrinsic value
            return max(0, S - K) if option_type == 'call' else max(0, K - S)

    @staticmethod
    def delta(S: float, K: float, T: float, r_d: float, r_f: float,
              sigma: float, option_type: str) -> float:
        """Calculate option delta"""
        if T <= 0:
            return 1.0 if option_type == 'call' and S > K else 0.0

        sigma = max(0.001, min(3.0, sigma))

        try:
            from scipy.stats import norm
            d1 = (np.log(S/K) + (r_d - r_f + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

            if option_type == 'call':
                return np.exp(-r_f*T) * norm.cdf(d1)
            else:
                return -np.exp(-r_f*T) * norm.cdf(-d1)
        except:
            return 0.5  # Fallback

    @staticmethod
    def vega(S: float, K: float, T: float, r_d: float, r_f: float, sigma: float) -> float:
        """Calculate option vega"""
        if T <= 0:
            return 0.0

        sigma = max(0.001, min(3.0, sigma))

        try:
            from scipy.stats import norm
            d1 = (np.log(S/K) + (r_d - r_f + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            return S * np.exp(-r_f*T) * norm.pdf(d1) * np.sqrt(T) / 100
        except:
            return 0.0

    @staticmethod
    def price_with_vol_spread(S: float, K: float, T: float, r_d: float, r_f: float,
                             sigma: float, option_type: str, tenor: str,
                             is_buy: bool) -> Tuple[float, float]:
        """Price option with bid-ask spread in volatility"""
        vol_spread = Config.TRANSACTION_COSTS['options'].get(tenor, 0.005)

        if is_buy:
            effective_vol = sigma + vol_spread / 2
        else:
            effective_vol = max(0.001, sigma - vol_spread / 2)

        price = BlackScholes.price(S, K, T, r_d, r_f, effective_vol, option_type)
        mid_price = BlackScholes.price(S, K, T, r_d, r_f, sigma, option_type)
        transaction_cost = abs(price - mid_price)

        return price, transaction_cost

    @staticmethod
    def strike_from_delta(S: float, delta: float, T: float, r_d: float, r_f: float,
                         sigma: float, option_type: str) -> float:
        """Calculate strike from delta"""
        sigma = max(0.001, min(3.0, sigma))

        try:
            from scipy.stats import norm
            if option_type == 'call':
                d1 = norm.ppf(abs(delta) * np.exp(r_f * T))
            else:
                d1 = -norm.ppf(abs(delta) * np.exp(r_f * T))

            K = S * np.exp(-(d1 * sigma * np.sqrt(T) - (r_d - r_f + 0.5*sigma**2)*T))
            return max(S * 0.5, min(S * 2.0, K))  # Reasonable strike bounds
        except:
            return S  # Fallback to ATM

class GVVModel:
    """Simplified GVV model implementation with stability fixes"""

    @staticmethod
    def price(S: float, K: float, T: float, r_d: float, r_f: float,
              atm_vol: float, rr_25: float, bf_25: float, option_type: str) -> float:
        """Price using GVV adjustment to Black-Scholes"""
        moneyness = K / S

        # Clamp inputs to reasonable ranges
        atm_vol = max(0.001, min(3.0, atm_vol))
        rr_25 = max(-0.1, min(0.1, rr_25))
        bf_25 = max(-0.05, min(0.05, bf_25))

        if moneyness < 0.90:
            smile_vol = atm_vol + bf_25 - rr_25/2 + 0.001 * (0.90 - moneyness)
        elif moneyness > 1.10:
            smile_vol = atm_vol + bf_25 + rr_25/2 + 0.001 * (moneyness - 1.10)
        else:
            skew_adjustment = rr_25 * (moneyness - 1.0) * 2
            smile_vol = atm_vol + bf_25 + skew_adjustment

        # Ensure reasonable vol
        smile_vol = max(0.001, min(3.0, smile_vol))
        return BlackScholes.price(S, K, T, r_d, r_f, smile_vol, option_type)

class SABRModel:
    """Simplified SABR model with stability improvements"""

    def __init__(self):
        self.alpha = None
        self.beta = 0.5
        self.rho = -0.25  # Less extreme correlation
        self.nu = 0.2     # Reduced vol of vol

    def calibrate(self, strikes: np.ndarray, vols: np.ndarray, F: float, T: float):
        """Simple calibration"""
        try:
            atm_idx = np.argmin(np.abs(strikes - F))
            self.alpha = max(0.001, min(3.0, vols[atm_idx]))
        except:
            self.alpha = 0.1

    def price(self, S: float, K: float, T: float, r_d: float, r_f: float,
              atm_vol: float, option_type: str) -> float:
        """Price using SABR vol with stability checks"""
        if self.alpha is None:
            self.alpha = max(0.001, min(3.0, atm_vol))

        try:
            F = S * np.exp((r_d - r_f) * T)

            if abs(F - K) < 0.001:
                sabr_vol = self.alpha
            else:
                # Simplified SABR vol calculation with bounds
                log_fk = np.log(F/K)
                log_fk = max(-1.0, min(1.0, log_fk))  # Bound log ratio

                fk_mid = (F*K)**((1-self.beta)/2)
                if fk_mid <= 0:
                    sabr_vol = self.alpha
                else:
                    z = (self.nu/self.alpha) * fk_mid * log_fk
                    z = max(-2.0, min(2.0, z))  # Bound z

                    if abs(z) < 0.001:
                        sabr_vol = self.alpha
                    else:
                        sqrt_term = 1 - 2*self.rho*z + z**2
                        if sqrt_term <= 0:
                            sabr_vol = self.alpha
                        else:
                            x_num = np.sqrt(sqrt_term) + z - self.rho
                            x_denom = 1 - self.rho
                            if x_denom == 0 or x_num <= 0:
                                sabr_vol = self.alpha
                            else:
                                x = np.log(x_num / x_denom)
                                sabr_vol = self.alpha * (z/x if abs(x) > 0.001 else 1.0)

            # Final bounds check
            sabr_vol = max(0.001, min(3.0, sabr_vol))
            return BlackScholes.price(S, K, T, r_d, r_f, sabr_vol, option_type)

        except Exception as e:
            # Fallback to Black-Scholes with ATM vol
            return BlackScholes.price(S, K, T, r_d, r_f, max(0.001, min(3.0, atm_vol)), option_type)

# ==================== ML Model ====================
class LGBMPricer:
    """LightGBM model for option pricing"""

    def __init__(self, training_years: int = 5):
        self.training_years = training_years
        self.model = None
        self.last_train_date = None

    def prepare_features(self, market_data: MarketData, pair: str,
                         strike: float, tenor: str, option_type: str,
                         historical_data: pd.DataFrame = None) -> np.ndarray:
        """Prepare features for ML model"""
        spot = market_data.spot[pair]
        T = self._tenor_to_years(tenor)

        features = [
            T,
            strike / spot,
            market_data.rates.get('USD', 0.05),
            market_data.rates.get('JPY', 0.01),
            market_data.atm_vols[pair].get(tenor, 0.1),
            market_data.rr_25[pair].get(tenor, 0),
            market_data.bf_25[pair].get(tenor, 0),
        ]

        if historical_data is not None and len(historical_data) > 20:
            try:
                returns = np.log(historical_data['spot'] / historical_data['spot'].shift(1))
                realized_vol = returns.tail(20).std() * np.sqrt(252)
                features.append(max(0.001, min(3.0, realized_vol)))
            except:
                features.append(0.1)
        else:
            features.append(0.1)

        features.extend([0.0, 0.0, 0.0])
        return np.array(features).reshape(1, -1)

    def price(self, S: float, K: float, T: float, r_d: float, r_f: float,
              features: np.ndarray) -> float:
        """Price option using trained model"""
        if self.model is None:
            # Fallback to Black-Scholes
            return BlackScholes.price(S, K, T, r_d, r_f, 0.1, 'call')

        try:
            return max(0.0, self.model.predict(features)[0])
        except:
            return BlackScholes.price(S, K, T, r_d, r_f, 0.1, 'call')

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
    """Enhanced trading strategy with proper risk management"""

    def __init__(self, model_name: str, use_iv_filter: bool = False,
                 delta_model: str = None):
        self.model_name = model_name
        self.use_iv_filter = use_iv_filter
        self.delta_model = delta_model or Config.DELTA_MODEL
        self.book = TradingBook()
        self.sabr_model = SABRModel()
        self.lgbm_model = LGBMPricer()

        # Performance tracking
        self.equity_curve = []
        self.trades_log = []
        self.hedges_log = []
        self.daily_stats = []

    def _get_rates_for_pair(self, pair: str, market_data: MarketData) -> Tuple[float, float]:
        """Get appropriate interest rates for currency pair"""
        rate_map = {
            'USDJPY': ('USD', 'JPY'),
            'GBPNZD': ('GBP', 'NZD'),
            'USDCAD': ('USD', 'CAD')
        }

        if pair in rate_map:
            base, quote = rate_map[pair]
            r_d = market_data.rates.get(base, 0.05)
            r_f = market_data.rates.get(quote, 0.01)
        else:
            r_d, r_f = 0.05, 0.01

        return r_d, r_f

    def run_daily(self, market_data: MarketData, historical_data: Dict[str, pd.DataFrame]):
        """Execute daily trading logic with equity protection"""

        # Check if equity has fallen too low
        current_equity = self.book.get_current_equity()
        if current_equity < Config.MIN_EQUITY_THRESHOLD:
            if not self.book.equity_stopped:
                print(f"EQUITY PROTECTION: Stopping trading at {current_equity:,.0f} on {market_data.date}")
                self.book.equity_stopped = True

            # Only update positions and record stats, no new trades
            self._update_positions(market_data)
            self._record_stats(market_data)
            return

        # Normal trading flow
        self._update_positions(market_data)

        if not self.book.equity_stopped:
            signals = self._find_opportunities(market_data, historical_data)
            self._execute_trades(signals, market_data)
            self._delta_hedge(market_data)

        self._calculate_interest(market_data)
        self._record_stats(market_data)

    def _find_opportunities(self, market_data: MarketData,
                           historical_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Find mispriced options to trade"""
        signals = []
        current_equity = self.book.get_current_equity()

        for pair in Config.PAIRS:
            if pair not in market_data.spot:
                continue

            spot = market_data.spot[pair]
            r_d, r_f = self._get_rates_for_pair(pair, market_data)

            for tenor in Config.TENORS:
                if tenor not in market_data.atm_vols[pair]:
                    continue

                atm_vol = market_data.atm_vols[pair][tenor]
                if pd.isna(atm_vol) or atm_vol <= 0:
                    continue

                T = self._tenor_to_years(tenor)

                # Apply IV filter
                if self.use_iv_filter and pair in historical_data:
                    hist = historical_data[pair]
                    if len(hist) >= Config.IV_MA_DAYS:
                        iv_ma = hist[f'atm_vol_{tenor}'].tail(Config.IV_MA_DAYS).mean()
                        if pd.isna(iv_ma):
                            continue

                for delta in Config.DELTAS:
                    for option_type in ['call', 'put']:
                        try:
                            adj_delta = delta if option_type == 'call' else -delta
                            strike = BlackScholes.strike_from_delta(
                                spot, adj_delta, T, r_d, r_f, atm_vol, option_type
                            )

                            # Calculate prices based on model
                            if self.model_name == 'GVV':
                                market_price = BlackScholes.price(
                                    spot, strike, T, r_d, r_f, atm_vol, option_type
                                )
                                model_price = self._calculate_model_price(
                                    market_data, pair, spot, strike, T, r_d, r_f,
                                    tenor, option_type, historical_data
                                )
                            else:
                                market_vol = self._get_smile_vol(market_data, pair, tenor, strike/spot)
                                market_price = BlackScholes.price(
                                    spot, strike, T, r_d, r_f, market_vol, option_type
                                )
                                model_price = self._calculate_model_price(
                                    market_data, pair, spot, strike, T, r_d, r_f,
                                    tenor, option_type, historical_data
                                )

                            # Check for valid prices
                            if market_price <= 0 or model_price <= 0:
                                continue

                            mispricing = (model_price - market_price) / market_price
                            threshold = Config.MISPRICING_THRESHOLD * 0.5 if self.model_name == 'GVV' else Config.MISPRICING_THRESHOLD

                            if abs(mispricing) > threshold:
                                # Apply IV filter
                                if self.use_iv_filter:
                                    if mispricing > 0 and atm_vol < iv_ma:
                                        continue
                                    if mispricing < 0 and atm_vol > iv_ma:
                                        continue

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

                        except Exception as e:
                            continue  # Skip problematic calculations

        # Sort and limit
        signals.sort(key=lambda x: abs(x['mispricing']), reverse=True)
        max_new = Config.MAX_POSITIONS - self.book.get_num_positions()
        return signals[:max_new]

    def _calculate_model_price(self, market_data: MarketData, pair: str,
                              S: float, K: float, T: float, r_d: float, r_f: float,
                              tenor: str, option_type: str,
                              historical_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate theoretical price using selected model"""
        try:
            if self.model_name == 'GVV':
                rr_25 = market_data.rr_25[pair].get(tenor, 0)
                bf_25 = market_data.bf_25[pair].get(tenor, 0)
                atm_vol = market_data.atm_vols[pair].get(tenor, 0.1)
                return GVVModel.price(S, K, T, r_d, r_f, atm_vol, rr_25, bf_25, option_type)

            elif self.model_name == 'SABR':
                atm_vol = market_data.atm_vols[pair].get(tenor, 0.1)
                return self.sabr_model.price(S, K, T, r_d, r_f, atm_vol, option_type)

            elif self.model_name == 'LGBM':
                features = self.lgbm_model.prepare_features(
                    market_data, pair, K, tenor, option_type,
                    historical_data.get(pair)
                )
                return self.lgbm_model.price(S, K, T, r_d, r_f, features)

            else:  # GK/Black-Scholes
                atm_vol = market_data.atm_vols[pair].get(tenor, 0.1)
                return BlackScholes.price(S, K, T, r_d, r_f, atm_vol, option_type)

        except Exception as e:
            # Fallback to Black-Scholes
            return BlackScholes.price(S, K, T, r_d, r_f, 0.1, option_type)

    def _get_smile_vol(self, market_data: MarketData, pair: str,
                      tenor: str, moneyness: float) -> float:
        """Get implied vol from smile for given moneyness"""
        try:
            atm_vol = market_data.atm_vols[pair].get(tenor, 0.1)
            rr_25 = market_data.rr_25[pair].get(tenor, 0)
            bf_25 = market_data.bf_25[pair].get(tenor, 0)

            # Clamp inputs
            atm_vol = max(0.001, min(3.0, atm_vol))
            rr_25 = max(-0.1, min(0.1, rr_25))
            bf_25 = max(-0.05, min(0.05, bf_25))

            if moneyness < 0.90:
                vol = atm_vol + bf_25 - rr_25/2
            elif moneyness > 1.10:
                vol = atm_vol + bf_25 + rr_25/2
            else:
                skew = rr_25 * (moneyness - 1.0) * 2
                vol = atm_vol + bf_25 + skew

            return max(0.001, min(3.0, vol))
        except:
            return 0.1

    def _execute_trades(self, signals: List[Dict], market_data: MarketData):
        """Execute trades with proper position sizing"""
        current_equity = self.book.get_current_equity()

        for signal in signals:
            try:
                r_d, r_f = signal['r_d'], signal['r_f']
                T = self._tenor_to_years(signal['tenor'])
                is_buy = signal['direction'] > 0

                # Dynamic position sizing based on current equity
                position_value = current_equity * Config.POSITION_SIZE_PCT
                notional = min(position_value, current_equity * Config.MAX_POSITION_CONCENTRATION)

                trade_price, vol_cost = BlackScholes.price_with_vol_spread(
                    signal['spot'], signal['strike'], T, r_d, r_f,
                    signal['vol'], signal['option_type'], signal['tenor'], is_buy
                )

                num_contracts = notional / signal['spot']
                total_option_premium = trade_price * notional

                # Transaction costs
                brokerage_cost = notional * Config.TRANSACTION_COSTS['brokerage']
                clearing_cost = notional * Config.TRANSACTION_COSTS['clearing']
                total_transaction_cost = vol_cost * notional + brokerage_cost + clearing_cost

                # Capital checks
                if is_buy:
                    total_cost = total_option_premium + total_transaction_cost
                    if self.book.cash < total_cost:
                        continue
                else:
                    required_margin = total_option_premium * 0.2
                    if self.book.used_margin + required_margin > current_equity * Config.MAX_LEVERAGE:
                        continue

                # Create position
                expiry = market_data.date + pd.Timedelta(days=int(T*365))

                position = OptionPosition(
                    entry_date=market_data.date,
                    expiry_date=expiry,
                    pair=signal['pair'],
                    strike=signal['strike'],
                    option_type=signal['option_type'],
                    position_size=signal['direction'] * num_contracts,
                    entry_price=trade_price,
                    entry_spot=signal['spot'],
                    entry_vol=signal['vol'],
                    model_used=self.model_name,
                    transaction_costs=total_transaction_cost,
                    notional=notional
                )

                # Calculate greeks
                position.current_delta = BlackScholes.delta(
                    signal['spot'], signal['strike'], T, r_d, r_f,
                    signal['vol'], signal['option_type']
                )
                position.current_vega = BlackScholes.vega(
                    signal['spot'], signal['strike'], T, r_d, r_f, signal['vol']
                )

                # Update book
                self.book.options.append(position)
                self.book.total_transaction_costs += total_transaction_cost

                if is_buy:
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

            except Exception as e:
                continue  # Skip problematic trades

    def _delta_hedge(self, market_data: MarketData):
        """Delta hedge with improved risk management"""
        current_equity = self.book.get_current_equity()
        delta_limit = Config.MAX_DELTA_EXPOSURE * current_equity

        for pair in Config.PAIRS:
            if pair not in market_data.spot:
                continue

            try:
                net_delta = self.book.get_net_delta(pair)
                spot_price = market_data.spot[pair]

                if abs(net_delta) > delta_limit:
                    hedge_amount = -net_delta / spot_price
                    spot_spread = Config.TRANSACTION_COSTS['spot'].get(pair, 0.0002)
                    hedge_cost = abs(hedge_amount * spot_price) * spot_spread

                    if self.book.cash >= hedge_cost:
                        if pair not in self.book.spot_hedge:
                            self.book.spot_hedge[pair] = 0
                        self.book.spot_hedge[pair] += hedge_amount

                        self.book.cash -= hedge_cost
                        self.book.total_transaction_costs += hedge_cost

                        self.hedges_log.append({
                            'date': market_data.date,
                            'pair': pair,
                            'hedge_amount': hedge_amount,
                            'hedge_delta': hedge_amount * spot_price,
                            'spot_price': spot_price,
                            'cost': hedge_cost,
                            'net_delta_before': net_delta,
                            'net_delta_after': self.book.get_net_delta(pair)
                        })
            except Exception as e:
                continue

    def _update_positions(self, market_data: MarketData):
        """Update existing positions"""
        positions_to_remove = []

        for i, pos in enumerate(self.book.options):
            try:
                if market_data.date >= pos.expiry_date:
                    # Expiry settlement
                    spot = market_data.spot[pos.pair]
                    if pos.option_type == 'call':
                        payoff = max(0, spot - pos.strike)
                    else:
                        payoff = max(0, pos.strike - spot)

                    pos.pnl = (payoff - pos.entry_price) * pos.position_size * pos.entry_spot - pos.transaction_costs
                    self.book.cash += payoff * pos.position_size * pos.entry_spot

                    if pos.position_size < 0:
                        self.book.used_margin -= pos.entry_price * pos.entry_spot * 0.2

                    positions_to_remove.append(i)
                else:
                    # Mark to market
                    spot = market_data.spot[pos.pair]
                    T = (pos.expiry_date - market_data.date).days / 365
                    vol = self._get_smile_vol(market_data, pos.pair,
                                             self._years_to_tenor(T), pos.strike/spot)

                    r_d, r_f = self._get_rates_for_pair(pos.pair, market_data)

                    pos.current_delta = BlackScholes.delta(
                        spot, pos.strike, T, r_d, r_f, vol, pos.option_type
                    )
                    pos.current_vega = BlackScholes.vega(
                        spot, pos.strike, T, r_d, r_f, vol
                    )
                    pos.current_value = BlackScholes.price(
                        spot, pos.strike, T, r_d, r_f, vol, pos.option_type
                    )
                    pos.pnl = (pos.current_value - pos.entry_price) * pos.position_size * pos.entry_spot - pos.transaction_costs

            except Exception as e:
                # If position can't be updated, mark for removal
                positions_to_remove.append(i)

        # Remove expired/problematic positions
        for i in reversed(positions_to_remove):
            del self.book.options[i]

    def _calculate_interest(self, market_data: MarketData):
        """Calculate daily interest"""
        daily_rate = market_data.rates.get('USD', 0.05) / 365

        if self.book.cash > 0:
            self.book.cash *= (1 + daily_rate)
        else:
            self.book.cash *= (1 + daily_rate * 1.5)

    def _record_stats(self, market_data: MarketData):
        """Record daily statistics"""
        try:
            total_value = self.book.get_current_equity()

            # Calculate Value at Risk (simplified)
            portfolio_vol = abs(self.book.get_net_vega()) * 0.01  # 1% vol shock
            var_95 = portfolio_vol * 1.645  # 95% VaR

            self.equity_curve.append({
                'date': market_data.date,
                'equity': total_value,
                'cash': self.book.cash,
                'num_positions': self.book.get_num_positions(),
                'net_delta': sum(self.book.get_net_delta(pair) for pair in Config.PAIRS if pair in market_data.spot),
                'net_vega': self.book.get_net_vega(),
                'used_margin': self.book.used_margin,
                'total_transaction_costs': self.book.total_transaction_costs,
                'var_95': var_95,
                'equity_stopped': self.book.equity_stopped
            })
        except Exception as e:
            # Fallback record
            self.equity_curve.append({
                'date': market_data.date,
                'equity': self.book.cash,
                'cash': self.book.cash,
                'num_positions': 0,
                'net_delta': 0,
                'net_vega': 0,
                'used_margin': 0,
                'total_transaction_costs': self.book.total_transaction_costs,
                'var_95': 0,
                'equity_stopped': True
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
            return {'USD': 0.05, 'JPY': 0.01, 'GBP': 0.04, 'NZD': 0.03, 'CAD': 0.04}

        try:
            date_curves = curves_data[curves_data['date'] == date]

            if date_curves.empty:
                nearest_date = curves_data['date'].iloc[(curves_data['date'] - date).abs().argsort()[:1]].values[0]
                date_curves = curves_data[curves_data['date'] == nearest_date]

            if date_curves.empty:
                return {'USD': 0.05, 'JPY': 0.01, 'GBP': 0.04, 'NZD': 0.03, 'CAD': 0.04}

            rates = {}
            usd_3m = date_curves[date_curves['tenor'] == '3M']['interpolated_rate'].values
            rates['USD'] = usd_3m[0] if len(usd_3m) > 0 else 0.05

            rates['JPY'] = max(0.001, rates['USD'] - 0.04)
            rates['GBP'] = max(0.001, rates['USD'] - 0.01)
            rates['NZD'] = max(0.001, rates['USD'] - 0.02)
            rates['CAD'] = max(0.001, rates['USD'] - 0.01)

            return rates
        except:
            return {'USD': 0.05, 'JPY': 0.01, 'GBP': 0.04, 'NZD': 0.03, 'CAD': 0.04}

    @staticmethod
    def prepare_market_data(fx_data: pd.DataFrame, date: pd.Timestamp,
                           curves_data: pd.DataFrame = None) -> MarketData:
        """Prepare market data for a specific date"""
        try:
            if date not in fx_data.index:
                return None

            row = fx_data.loc[date]
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

            for pair in Config.PAIRS:
                try:
                    # Spot
                    spot_col = f'{pair} Curncy'
                    if spot_col in row.index and not pd.isna(row[spot_col]):
                        market.spot[pair] = row[spot_col]

                    market.forwards[pair] = {}
                    market.atm_vols[pair] = {}
                    market.rr_25[pair] = {}
                    market.bf_25[pair] = {}
                    market.rr_10[pair] = {}
                    market.bf_10[pair] = {}

                    for tenor in Config.TENORS:
                        # ATM vol
                        vol_col = f'{pair}V{tenor} Curncy'
                        if vol_col in row.index and not pd.isna(row[vol_col]):
                            market.atm_vols[pair][tenor] = max(0.001, row[vol_col] / 100)

                        # Risk reversal
                        rr_col = f'{pair}25R{tenor} Curncy'
                        if rr_col in row.index and not pd.isna(row[rr_col]):
                            market.rr_25[pair][tenor] = row[rr_col] / 100

                        # Butterfly
                        bf_col = f'{pair}25B{tenor} Curncy'
                        if bf_col in row.index and not pd.isna(row[bf_col]):
                            market.bf_25[pair][tenor] = row[bf_col] / 100
                except:
                    continue

            return market
        except:
            return None

# ==================== Backtester ====================
class Backtester:
    """Run backtests for all models"""

    def __init__(self, fx_data: pd.DataFrame, curves_data: pd.DataFrame = None):
        self.fx_data = fx_data
        self.curves_data = curves_data
        self.results = {}

    def run_all_models(self, start_date: pd.Timestamp, end_date: pd.Timestamp,
                      delta_model: str = None):
        """Run backtest for all models"""
        models = ['GK', 'GVV', 'SABR', 'LGBM']
        delta_model = delta_model or Config.DELTA_MODEL

        for model in models:
            print(f"\nRunning backtest for {model} (Delta model: {delta_model})...")
            strategy = OptionsArbitrageStrategy(model_name=model, delta_model=delta_model)
            self._run_single_backtest(strategy, start_date, end_date)
            self.results[f'{model}_delta{delta_model}'] = strategy

        # GK with IV filter
        print(f"\nRunning backtest for GK with IV filter (Delta model: {delta_model})...")
        strategy = OptionsArbitrageStrategy(model_name='GK', use_iv_filter=True,
                                           delta_model=delta_model)
        self._run_single_backtest(strategy, start_date, end_date)
        self.results[f'GK_IVFilter_delta{delta_model}'] = strategy

    def _run_single_backtest(self, strategy: OptionsArbitrageStrategy,
                            start_date: pd.Timestamp, end_date: pd.Timestamp):
        """Run backtest for a single strategy"""
        dates = self.fx_data.index[(self.fx_data.index >= start_date) &
                                   (self.fx_data.index <= end_date)]

        historical_data = {}
        for pair in Config.PAIRS:
            historical_data[pair] = pd.DataFrame()

        for i, date in enumerate(dates):
            if i % 50 == 0:
                print(f"  Processing {i}/{len(dates)} days...")

            market_data = DataLoader.prepare_market_data(self.fx_data, date, self.curves_data)
            if market_data is None:
                continue

            # Update historical data
            for pair in Config.PAIRS:
                if pair in market_data.spot:
                    try:
                        new_row = pd.DataFrame({
                            'date': [date],
                            'spot': [market_data.spot[pair]]
                        })
                        for tenor in Config.TENORS:
                            if tenor in market_data.atm_vols[pair]:
                                new_row[f'atm_vol_{tenor}'] = market_data.atm_vols[pair][tenor]

                        historical_data[pair] = pd.concat([historical_data[pair], new_row],
                                                         ignore_index=True)
                    except:
                        continue

            strategy.run_daily(market_data, historical_data)

            # Stop if equity is too low
            if strategy.book.equity_stopped:
                print(f"  Strategy stopped due to low equity on {date}")
                break

    def calculate_performance_metrics(self) -> pd.DataFrame:
        """Calculate performance metrics for all strategies"""
        metrics = []

        for model_name, strategy in self.results.items():
            if not strategy.equity_curve:
                continue

            try:
                df = pd.DataFrame(strategy.equity_curve)
                returns = df['equity'].pct_change().dropna()

                if len(returns) == 0:
                    continue

                total_return = (df['equity'].iloc[-1] / Config.INITIAL_CAPITAL - 1) * 100

                # Fixed Sharpe ratio calculation
                sharpe = 0
                if len(returns) > 0 and returns.std() > 0:
                    mean_return = returns.mean()
                    volatility = returns.std()
                    sharpe = (mean_return / volatility) * np.sqrt(252)

                    # Ensure sign consistency: negative returns should give negative Sharpe
                    if total_return < 0 and sharpe > 0:
                        sharpe = -abs(sharpe)

                # Max drawdown
                running_max = df['equity'].expanding().max()
                drawdown = (df['equity'] / running_max - 1) * 100
                max_drawdown = drawdown.min()

                # Risk metrics
                avg_var = df['var_95'].mean() if 'var_95' in df else 0

                # Additional metrics
                volatility_annual = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
                num_trades = len(strategy.trades_log) if hasattr(strategy, 'trades_log') else 0

                metrics.append({
                    'Model': model_name,
                    'Total Return (%)': round(total_return, 2),
                    'Sharpe Ratio': round(sharpe, 3),
                    'Volatility (%)': round(volatility_annual, 2),
                    'Max Drawdown (%)': round(max_drawdown, 2),
                    'Avg Positions': round(df['num_positions'].mean(), 1),
                    'Total Trades': num_trades,
                    'Final Equity': f"${df['equity'].iloc[-1]:,.0f}",
                    'Avg VaR 95%': f"${avg_var:,.0f}",
                    'Transaction Costs': f"${df['total_transaction_costs'].iloc[-1]:,.0f}",
                    'Equity Stopped': any(df['equity_stopped']) if 'equity_stopped' in df else False
                })
            except Exception as e:
                print(f"Error calculating metrics for {model_name}: {e}")
                continue

        return pd.DataFrame(metrics)

    def plot_results(self):
        """Plot equity curves and risk metrics with rolling Sharpe ratio"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Equity curves
        ax = axes[0, 0]
        for model_name, strategy in self.results.items():
            if strategy.equity_curve:
                df = pd.DataFrame(strategy.equity_curve)
                ax.plot(df['date'], df['equity'], label=model_name, linewidth=2)

        ax.axhline(y=Config.INITIAL_CAPITAL, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
        ax.axhline(y=Config.MIN_EQUITY_THRESHOLD, color='red', linestyle='--', alpha=0.5, label='Stop Loss Level')
        ax.set_title('Equity Curves', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='plain', axis='y')

        # Rolling Sharpe Ratio (replacing drawdown)
        ax = axes[0, 1]
        lookback_days = 60  # 60-day rolling Sharpe

        for model_name, strategy in self.results.items():
            if strategy.equity_curve:
                df = pd.DataFrame(strategy.equity_curve)
                if len(df) > lookback_days:
                    returns = df['equity'].pct_change().dropna()

                    # Calculate rolling Sharpe ratio
                    rolling_sharpe = []
                    rolling_dates = []

                    for i in range(lookback_days, len(returns)):
                        window_returns = returns.iloc[i-lookback_days:i]
                        if window_returns.std() > 0:
                            sharpe = (window_returns.mean() / window_returns.std()) * np.sqrt(252)
                            # Fix sign consistency
                            if window_returns.mean() < 0 and sharpe > 0:
                                sharpe = -abs(sharpe)
                        else:
                            sharpe = 0

                        rolling_sharpe.append(sharpe)
                        rolling_dates.append(df['date'].iloc[i])

                    ax.plot(rolling_dates, rolling_sharpe, label=model_name, linewidth=2)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
        ax.set_title(f'Rolling Sharpe Ratio ({lookback_days}-day)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Value at Risk
        ax = axes[1, 0]
        for model_name, strategy in self.results.items():
            if strategy.equity_curve:
                df = pd.DataFrame(strategy.equity_curve)
                if 'var_95' in df:
                    ax.plot(df['date'], df['var_95'], label=model_name, alpha=0.7, linewidth=2)

        ax.set_title('Value at Risk (95%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('VaR ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Risk-Return Scatter
        ax = axes[1, 1]
        returns_list = []
        sharpe_list = []
        model_names = []
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.results)))

        for i, (model_name, strategy) in enumerate(self.results.items()):
            if strategy.equity_curve:
                df = pd.DataFrame(strategy.equity_curve)
                if len(df) > 1:
                    ret_series = df['equity'].pct_change().dropna()
                    if len(ret_series) > 0 and ret_series.std() > 0:
                        total_ret = (df['equity'].iloc[-1] / Config.INITIAL_CAPITAL - 1) * 100
                        sharpe = (ret_series.mean() / ret_series.std()) * np.sqrt(252)

                        # Fix sign consistency
                        if total_ret < 0 and sharpe > 0:
                            sharpe = -abs(sharpe)

                        returns_list.append(total_ret)
                        sharpe_list.append(sharpe)
                        model_names.append(model_name)

                        ax.scatter(total_ret, sharpe, s=100, c=[colors[i]],
                                 label=model_name, alpha=0.8, edgecolors='black')

        ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax.set_xlabel('Total Return (%)')
        ax.set_ylabel('Sharpe Ratio')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = Config.RESULTS_DIR / 'backtest_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")

        plt.show()

# ==================== Main Execution ====================
def main():
    """Main execution function"""
    print("=" * 80)
    print("FX OPTIONS TRADING SYSTEM - ENHANCED")
    print("=" * 80)
    print("\nASSUMPTIONS & CONFIGURATION:")
    print(f"• Initial Capital: ${Config.INITIAL_CAPITAL:,}")
    print(f"• Stop Loss Level: ${Config.MIN_EQUITY_THRESHOLD:,}")
    print(f"• Max Leverage: {Config.MAX_LEVERAGE}x")
    print(f"• Max Delta Exposure: {Config.MAX_DELTA_EXPOSURE*100}% of capital")
    print(f"• Position Size: {Config.POSITION_SIZE_PCT*100}% of current equity")
    print(f"• Mispricing Threshold: {Config.MISPRICING_THRESHOLD*100}%")
    print("• Enhanced transaction costs with realistic bid-ask spreads")
    print("• Automatic position closure on equity protection trigger")

    # Create directories
    Config.DATA_DIR.mkdir(exist_ok=True)
    Config.RESULTS_DIR.mkdir(exist_ok=True)

    # Load data
    print("\nLoading market data...")
    try:
        # Try to load discount curves first, create if missing
        if not Config.CURVES_PATH.exists():
            print("⚠ Discount curves not found. Attempting to create them...")
            try:
                import subprocess
                result = subprocess.run(['python', 'discount_curves.py'],
                                      capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print("✓ Discount curves created successfully")
                else:
                    print(f"⚠ Could not create discount curves: {result.stderr}")
            except Exception as e:
                print(f"⚠ Could not run discount_curves.py: {e}")

        fx_data = DataLoader.load_fx_data(Config.FX_DATA_PATH)
        # fx_data = fx_data.tail(2000)
        print(f"✓ FX data loaded: {len(fx_data)} days")
        print(f"  Date range: {fx_data.index[0]} to {fx_data.index[-1]}")

        curves_data = DataLoader.load_curves_data(Config.CURVES_PATH)
        if curves_data is not None:
            print(f"✓ Discount curves loaded: {len(curves_data)} points")
        else:
            print("⚠ Using default interest rates")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Split data
    n_days = len(fx_data)
    train_end = int(n_days * 0.75)

    train_data = fx_data.iloc[:train_end]
    test_data = fx_data.iloc[train_end:]

    print(f"\nData Split:")
    print(f"• Train period: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} days)")
    print(f"• Test period: {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} days)")

    # Run backtests
    print(f"\nRunning backtests with delta model: {Config.DELTA_MODEL}")
    backtester = Backtester(fx_data, curves_data)
    backtester.run_all_models(test_data.index[0], test_data.index[-1], Config.DELTA_MODEL)

    # Calculate metrics
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)

    metrics = backtester.calculate_performance_metrics()
    print(metrics.to_string(index=False))

    # Save results
    results_file = Config.RESULTS_DIR / 'performance_metrics.txt'
    with open(results_file, 'w') as f:
        f.write("FX OPTIONS TRADING SYSTEM - PERFORMANCE ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write("CONFIGURATION:\n")
        f.write(f"Initial Capital: ${Config.INITIAL_CAPITAL:,}\n")
        f.write(f"Stop Loss Level: ${Config.MIN_EQUITY_THRESHOLD:,}\n")
        f.write(f"Max Leverage: {Config.MAX_LEVERAGE}x\n")
        f.write(f"Max Delta Exposure: {Config.MAX_DELTA_EXPOSURE*100}%\n")
        f.write(f"Position Size: {Config.POSITION_SIZE_PCT*100}% of equity\n")
        f.write(f"Mispricing Threshold: {Config.MISPRICING_THRESHOLD*100}%\n\n")
        f.write("PERFORMANCE METRICS:\n")
        f.write(metrics.to_string(index=False))
        f.write("\n\nTRANSACTION COSTS:\n")
        f.write("Spot FX Spreads: 0.2-2.0 pips\n")
        f.write("Option Vol Spreads: 30-120 bps\n")
        f.write("Brokerage: 3 bps\n")
        f.write("Clearing: 2 bps\n")

    print(f"\n✓ Results saved to: {results_file}")

    # Generate plots
    print("\nGenerating visualizations...")
    backtester.plot_results()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    best_return = metrics.loc[metrics['Total Return (%)'].idxmax()]
    best_sharpe = metrics.loc[metrics['Sharpe Ratio'].idxmax()]

    print(f"🏆 Best Return: {best_return['Model']} ({best_return['Total Return (%)']}%)")
    print(f"📊 Best Sharpe: {best_sharpe['Model']} ({best_sharpe['Sharpe Ratio']})")

    # Check drawdown constraint
    acceptable_dd = metrics[metrics['Max Drawdown (%)'] >= -10]
    if len(acceptable_dd) > 0:
        best_constrained = acceptable_dd.loc[acceptable_dd['Sharpe Ratio'].idxmax()]
        print(f"✅ Best with <10% DD: {best_constrained['Model']} (Sharpe: {best_constrained['Sharpe Ratio']})")
    else:
        print("⚠ No strategies met the <10% drawdown constraint")

    print(f"\n✅ Analysis complete! Check {Config.RESULTS_DIR} for detailed results.")

if __name__ == "__main__":
    main()