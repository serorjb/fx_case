"""
FX Options Trading System - Refactored Implementation
Focuses on GK, SABR, and GK with IV Filter strategies
Implements proper transaction costs and ensures no look-ahead bias
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings
import matplotlib.pyplot as plt

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
    MAX_POSITIONS = 100  # Reduced for better risk management
    MAX_DELTA_EXPOSURE = 0.02  # 2% max delta exposure
    MISPRICING_THRESHOLD = 0.015  # 1.5% mispricing to trade
    MAX_DRAWDOWN = 0.10  # 10% max drawdown as per requirements

    # Transaction costs - properly structured
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
    IV_MA_DAYS = 20  # For IV filter strategy
    VOL_REGIME_WINDOW = 60  # Days to calculate volatility regime

    # Deltas to evaluate (as absolute values)
    DELTAS = [0.10, 0.25, 0.40, 0.50]  # Focus on key strikes

    # Tenors to trade
    TENORS = ['1W', '2W', '1M', '2M', '3M', '6M', '1Y']

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
    entry_price: float  # option premium
    entry_spot: float
    entry_vol: float
    model_used: str
    tenor: str
    delta_at_entry: float
    current_delta: float = 0.0
    current_vega: float = 0.0
    current_gamma: float = 0.0
    current_value: float = 0.0
    pnl: float = 0.0
    days_held: int = 0


@dataclass
class TradingBook:
    """Manages all positions and hedges"""
    options: List[OptionPosition] = field(default_factory=list)
    spot_hedges: Dict[str, float] = field(default_factory=dict)  # Spot hedge per currency pair
    cash: float = Config.INITIAL_CAPITAL
    used_margin: float = 0.0
    peak_equity: float = Config.INITIAL_CAPITAL
    current_drawdown: float = 0.0

    def get_net_delta(self, pair: str = None) -> float:
        """Calculate total portfolio delta"""
        if pair:
            option_delta = sum(
                pos.current_delta * pos.position_size
                for pos in self.options if pos.pair == pair
            )
            return option_delta + self.spot_hedges.get(pair, 0)
        else:
            total_delta = 0
            for curr_pair in Config.PAIRS:
                total_delta += self.get_net_delta(curr_pair)
            return total_delta

    def get_net_vega(self) -> float:
        """Calculate total portfolio vega"""
        return sum(pos.current_vega * pos.position_size for pos in self.options)

    def get_net_gamma(self) -> float:
        """Calculate total portfolio gamma"""
        return sum(pos.current_gamma * pos.position_size for pos in self.options)

    def update_drawdown(self, current_equity: float):
        """Update peak equity and drawdown"""
        self.peak_equity = max(self.peak_equity, current_equity)
        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity


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
    """Black-Scholes/Garman-Kohlhagen pricing for FX options"""

    @staticmethod
    def price(S: float, K: float, T: float, r_d: float, r_f: float,
              sigma: float, option_type: str) -> float:
        """Price option using Black-Scholes"""
        if T <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S)

        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            return S * np.exp(-r_f * T) * norm.cdf(d1) - K * np.exp(-r_d * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r_d * T) * norm.cdf(-d2) - S * np.exp(-r_f * T) * norm.cdf(-d1)

    @staticmethod
    def delta(S: float, K: float, T: float, r_d: float, r_f: float,
              sigma: float, option_type: str) -> float:
        """Calculate option delta"""
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if K > S else 0.0

        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        if option_type == 'call':
            return np.exp(-r_f * T) * norm.cdf(d1)
        else:
            return -np.exp(-r_f * T) * norm.cdf(-d1)

    @staticmethod
    def vega(S: float, K: float, T: float, r_d: float, r_f: float, sigma: float) -> float:
        """Calculate option vega (per 1% vol change)"""
        if T <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-r_f * T) * norm.pdf(d1) * np.sqrt(T) / 100

    @staticmethod
    def gamma(S: float, K: float, T: float, r_d: float, r_f: float, sigma: float) -> float:
        """Calculate option gamma"""
        if T <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return np.exp(-r_f * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def implied_vol(option_price: float, S: float, K: float, T: float,
                   r_d: float, r_f: float, option_type: str) -> float:
        """Calculate implied volatility using Newton-Raphson"""
        if T <= 0:
            return 0.0

        # Initial guess
        sigma = 0.2

        for _ in range(50):
            price = BlackScholes.price(S, K, T, r_d, r_f, sigma, option_type)
            vega = BlackScholes.vega(S, K, T, r_d, r_f, sigma) * 100  # Convert back

            diff = option_price - price
            if abs(diff) < 1e-6:
                break

            if abs(vega) < 1e-10:
                break

            sigma += diff / vega
            sigma = max(0.001, min(sigma, 5.0))  # Keep in reasonable bounds

        return sigma

    @staticmethod
    def strike_from_delta(S: float, delta: float, T: float, r_d: float, r_f: float,
                         sigma: float, option_type: str) -> float:
        """Calculate strike from delta"""
        if option_type == 'call':
            d1 = norm.ppf(delta * np.exp(r_f * T))
        else:
            d1 = -norm.ppf(-delta * np.exp(r_f * T))

        K = S * np.exp(-(d1 * sigma * np.sqrt(T) - (r_d - r_f + 0.5 * sigma**2) * T))
        return K


class SABRModel:
    """SABR model for volatility smile"""

    def __init__(self):
        self.alpha = 0.1
        self.beta = 0.5  # Fixed for FX
        self.rho = -0.3
        self.nu = 0.3

    def calibrate(self, F: float, T: float, atm_vol: float, rr_25: float, bf_25: float):
        """Calibrate SABR to market smile"""
        # Set alpha to ATM vol as starting point
        self.alpha = atm_vol

        # Adjust rho based on risk reversal (skew)
        self.rho = np.clip(-rr_25 * 10, -0.9, 0.9)  # Scale and clip

        # Adjust nu based on butterfly (convexity)
        self.nu = np.clip(bf_25 * 20 + 0.3, 0.1, 1.0)  # Scale and shift

    def vol(self, F: float, K: float, T: float) -> float:
        """Calculate SABR implied volatility"""
        if abs(F - K) < 0.001 * F:  # ATM
            return self.alpha

        # Hagan's approximation
        FK_mid = np.sqrt(F * K)
        log_FK = np.log(F / K)

        z = (self.nu / self.alpha) * FK_mid**(1 - self.beta) * log_FK
        x_z = np.log((np.sqrt(1 - 2*self.rho*z + z**2) + z - self.rho) / (1 - self.rho))

        if abs(x_z) < 0.001:
            series_exp = 1.0
        else:
            series_exp = z / x_z

        # First order approximation
        sabr_vol = self.alpha * series_exp

        # Add correction terms
        term1 = ((1 - self.beta)**2 / 24) * (self.alpha**2 / FK_mid**(2*(1-self.beta)))
        term2 = 0.25 * self.rho * self.beta * self.nu * self.alpha / FK_mid**(1-self.beta)
        term3 = (2 - 3*self.rho**2) * self.nu**2 / 24

        sabr_vol *= (1 + (term1 + term2 + term3) * T)

        return sabr_vol

    def price(self, S: float, K: float, T: float, r_d: float, r_f: float,
              atm_vol: float, rr_25: float, bf_25: float, option_type: str) -> float:
        """Price option using SABR vol"""
        F = S * np.exp((r_d - r_f) * T)

        # Calibrate to market smile
        self.calibrate(F, T, atm_vol, rr_25, bf_25)

        # Get SABR vol
        sabr_vol = self.vol(F, K, T)

        # Price with Black-Scholes using SABR vol
        return BlackScholes.price(S, K, T, r_d, r_f, sabr_vol, option_type)


# ==================== Volatility Surface ====================
class VolatilitySurface:
    """Construct and interpolate volatility surface from market quotes"""

    @staticmethod
    def get_smile_vol(atm_vol: float, rr_25: float, bf_25: float,
                     rr_10: float, bf_10: float, delta: float,
                     option_type: str) -> float:
        """Get implied vol for given delta from smile"""

        # Convert delta to signed delta for interpolation
        signed_delta = delta if option_type == 'call' else -delta

        # Market quotes give us vols at specific deltas
        # ATM ~ 50 delta, 25 delta, 10 delta

        if abs(signed_delta) <= 0.10:
            # Near 10 delta - use 10 delta quotes
            if option_type == 'put':
                return atm_vol + bf_10 - rr_10/2
            else:
                return atm_vol + bf_10 + rr_10/2

        elif abs(signed_delta) <= 0.25:
            # Between 10 and 25 delta - interpolate
            if option_type == 'put':
                vol_10 = atm_vol + bf_10 - rr_10/2
                vol_25 = atm_vol + bf_25 - rr_25/2
            else:
                vol_10 = atm_vol + bf_10 + rr_10/2
                vol_25 = atm_vol + bf_25 + rr_25/2

            # Linear interpolation
            weight = (abs(signed_delta) - 0.10) / 0.15
            return vol_10 * (1 - weight) + vol_25 * weight

        elif abs(signed_delta) <= 0.50:
            # Between 25 and 50 delta
            if option_type == 'put':
                vol_25 = atm_vol + bf_25 - rr_25/2
            else:
                vol_25 = atm_vol + bf_25 + rr_25/2

            # Interpolate to ATM
            weight = (abs(signed_delta) - 0.25) / 0.25
            return vol_25 * (1 - weight) + atm_vol * weight

        else:
            # Beyond 50 delta - extrapolate carefully
            if option_type == 'call':
                vol_25 = atm_vol + bf_25 + rr_25/2
            else:
                vol_25 = atm_vol + bf_25 - rr_25/2

            # Modest extrapolation
            return atm_vol + (atm_vol - vol_25) * (abs(signed_delta) - 0.50) / 0.25


# ==================== Risk Management ====================
class RiskManager:
    """Manage portfolio risk and position sizing"""

    def __init__(self, max_drawdown: float = 0.10):
        self.max_drawdown = max_drawdown
        self.current_var = 0.0
        self.position_limits = {
            'per_trade': 0.02,  # Max 2% risk per trade
            'per_pair': 0.10,   # Max 10% exposure per pair
            'total': 0.30       # Max 30% total exposure
        }

    def check_position_size(self, book: TradingBook, proposed_premium: float,
                           pair: str) -> bool:
        """Check if proposed position meets risk limits"""
        current_equity = book.cash + sum(
            pos.current_value * pos.position_size * pos.entry_spot
            for pos in book.options
        )

        # Check drawdown limit
        if book.current_drawdown >= self.max_drawdown * 0.8:  # 80% of max
            return False

        # Check per-trade limit
        if proposed_premium > current_equity * self.position_limits['per_trade']:
            return False

        # Check per-pair limit
        pair_exposure = sum(
            abs(pos.current_value * pos.position_size * pos.entry_spot)
            for pos in book.options if pos.pair == pair
        )
        if pair_exposure + proposed_premium > current_equity * self.position_limits['per_pair']:
            return False

        # Check total exposure
        total_exposure = sum(
            abs(pos.current_value * pos.position_size * pos.entry_spot)
            for pos in book.options
        )
        if total_exposure + proposed_premium > current_equity * self.position_limits['total']:
            return False

        return True

    def calculate_position_size(self, book: TradingBook, signal_strength: float,
                               vol_regime: str) -> float:
        """Calculate position size based on signal and regime"""
        base_size = 1.0

        # Adjust for signal strength
        base_size *= min(abs(signal_strength) / 0.05, 2.0)  # Cap at 2x for strong signals

        # Adjust for volatility regime
        if vol_regime == 'high':
            base_size *= 0.5  # Reduce in high vol
        elif vol_regime == 'low':
            base_size *= 1.2  # Increase in low vol

        # Adjust for current drawdown
        if book.current_drawdown > 0.05:
            base_size *= (1 - book.current_drawdown / self.max_drawdown)

        return np.clip(base_size, 0.1, 2.0)


# ==================== Trading Strategy ====================
class OptionsArbitrageStrategy:
    """
    Main trading strategy:
    1. Prices options using Black-Scholes/SABR
    2. Identifies mispricings
    3. Trades and delta hedges positions
    """

    def __init__(self, model_name: str, use_iv_filter: bool = False):
        self.model_name = model_name
        self.use_iv_filter = use_iv_filter
        self.book = TradingBook()
        self.sabr_model = SABRModel()
        self.risk_manager = RiskManager(Config.MAX_DRAWDOWN)

        # Initialize spot hedges
        for pair in Config.PAIRS:
            self.book.spot_hedges[pair] = 0.0

        # Track performance
        self.equity_curve = []
        self.trades_log = []
        self.hedges_log = []
        self.vol_regimes = []

        # Record initial state
        self.initial_equity_recorded = False

    def run_daily(self, market_data: MarketData, historical_vol: Dict[str, pd.Series]):
        """Execute daily trading logic"""

        # Record initial equity on first day
        if not self.initial_equity_recorded:
            self.equity_curve.append({
                'date': market_data.date,
                'equity': Config.INITIAL_CAPITAL,
                'cash': Config.INITIAL_CAPITAL,
                'options_value': 0,
                'spot_hedge_value': 0,
                'num_positions': 0,
                'net_delta': 0,
                'net_vega': 0,
                'net_gamma': 0,
                'used_margin': 0,
                'drawdown': 0
            })
            self.initial_equity_recorded = True
            return  # Don't trade on first day, just establish baseline

        # 1. Determine volatility regime
        vol_regime = self._determine_vol_regime(market_data, historical_vol)
        self.vol_regimes.append({'date': market_data.date, 'regime': vol_regime})

        # 2. Update existing positions
        self._update_positions(market_data)

        # 3. Check for stops/exits
        self._check_exits(market_data)

        # 4. Find new opportunities (only if not in drawdown)
        if self.book.current_drawdown < Config.MAX_DRAWDOWN * 0.8:
            signals = self._find_opportunities(market_data, historical_vol)
            self._execute_trades(signals, market_data, vol_regime)

        # 5. Delta hedge
        self._delta_hedge(market_data)

        # 6. Update P&L and statistics
        self._update_pnl(market_data)

        # 7. Record daily statistics
        self._record_stats(market_data)

    def _determine_vol_regime(self, market_data: MarketData,
                             historical_vol: Dict[str, pd.Series]) -> str:
        """Determine current volatility regime"""
        if not historical_vol or 'USDJPY' not in historical_vol:
            return 'normal'

        hist = historical_vol['USDJPY']
        if len(hist) < Config.VOL_REGIME_WINDOW:
            return 'normal'

        current_vol = np.mean([
            market_data.atm_vols[pair].get('1M', 0.1)
            for pair in Config.PAIRS if pair in market_data.atm_vols
        ])

        recent_vols = hist.tail(Config.VOL_REGIME_WINDOW)
        percentile = (recent_vols < current_vol).sum() / len(recent_vols)

        if percentile > 0.8:
            return 'high'
        elif percentile < 0.2:
            return 'low'
        else:
            return 'normal'

    def _find_opportunities(self, market_data: MarketData,
                          historical_vol: Dict[str, pd.Series]) -> List[Dict]:
        """Find mispriced options to trade"""
        signals = []

        for pair in Config.PAIRS:
            if pair not in market_data.spot:
                continue

            spot = market_data.spot[pair]
            r_d, r_f = self._get_rates(pair, market_data)

            # Check IV filter if enabled
            if self.use_iv_filter and pair in historical_vol:
                hist = historical_vol[pair]
                if len(hist) < Config.IV_MA_DAYS:
                    continue
                iv_ma = hist.tail(Config.IV_MA_DAYS).mean()

            for tenor in Config.TENORS:
                if tenor not in market_data.atm_vols[pair]:
                    continue

                atm_vol = market_data.atm_vols[pair][tenor]
                T = self._tenor_to_years(tenor)

                # Skip if too close to expiry
                if T < 1/252:  # Less than 1 day
                    continue

                # Get smile parameters
                rr_25 = market_data.rr_25[pair].get(tenor, 0)
                bf_25 = market_data.bf_25[pair].get(tenor, 0)
                rr_10 = market_data.rr_10[pair].get(tenor, 0)
                bf_10 = market_data.bf_10[pair].get(tenor, 0)

                for delta in Config.DELTAS:
                    for option_type in ['call', 'put']:
                        # Calculate strike from delta
                        adj_delta = delta if option_type == 'call' else -delta
                        strike = BlackScholes.strike_from_delta(
                            spot, adj_delta, T, r_d, r_f, atm_vol, option_type
                        )

                        # Get market vol from smile
                        market_vol = VolatilitySurface.get_smile_vol(
                            atm_vol, rr_25, bf_25, rr_10, bf_10,
                            delta, option_type
                        )

                        # Calculate market price
                        market_price = BlackScholes.price(
                            spot, strike, T, r_d, r_f, market_vol, option_type
                        )

                        # Calculate model price
                        if self.model_name == 'SABR':
                            model_price = self.sabr_model.price(
                                spot, strike, T, r_d, r_f,
                                atm_vol, rr_25, bf_25, option_type
                            )
                        else:  # GK model
                            # For GK, we use ATM vol (simplified)
                            model_price = BlackScholes.price(
                                spot, strike, T, r_d, r_f, atm_vol, option_type
                            )

                        # Check for mispricing
                        if market_price > 0 and model_price > 0:
                            mispricing = (model_price - market_price) / market_price

                            if abs(mispricing) > Config.MISPRICING_THRESHOLD:
                                # Apply IV filter
                                if self.use_iv_filter:
                                    if mispricing > 0 and atm_vol < iv_ma * 0.95:
                                        continue  # Skip long if IV below MA
                                    if mispricing < 0 and atm_vol > iv_ma * 1.05:
                                        continue  # Skip short if IV above MA

                                signal = {
                                    'pair': pair,
                                    'tenor': tenor,
                                    'strike': strike,
                                    'option_type': option_type,
                                    'spot': spot,
                                    'market_price': market_price,
                                    'model_price': model_price,
                                    'market_vol': market_vol,
                                    'mispricing': mispricing,
                                    'direction': 1 if mispricing > 0 else -1,
                                    'delta': delta,
                                    'time_to_expiry': T
                                }
                                signals.append(signal)

        # Sort by absolute mispricing
        signals.sort(key=lambda x: abs(x['mispricing']), reverse=True)

        # Limit number of new positions
        max_new = max(0, Config.MAX_POSITIONS - len(self.book.options))
        return signals[:max_new]

    def _execute_trades(self, signals: List[Dict], market_data: MarketData,
                       vol_regime: str):
        """Execute trades based on signals"""
        for signal in signals:
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                self.book, signal['mispricing'], vol_regime
            )

            # Calculate premium with transaction costs
            notional = signal['spot'] * 1_000_000  # 1M notional per unit

            # Option spread cost (half-spread for entry)
            vol_spread = Config.TRANSACTION_COSTS['options'][signal['tenor']]
            adjusted_vol = signal['market_vol'] + vol_spread * signal['direction'] * 0.5

            # Recalculate price with adjusted vol
            r_d, r_f = self._get_rates(signal['pair'], market_data)
            adjusted_price = BlackScholes.price(
                signal['spot'], signal['strike'], signal['time_to_expiry'],
                r_d, r_f, adjusted_vol, signal['option_type']
            )

            # Total premium including all costs
            option_premium = adjusted_price * notional * position_size
            brokerage = notional * Config.TRANSACTION_COSTS['brokerage'] * position_size
            clearing = notional * Config.TRANSACTION_COSTS['clearing'] * position_size
            total_cost = option_premium + brokerage + clearing

            # Check risk limits
            if not self.risk_manager.check_position_size(
                self.book, abs(total_cost), signal['pair']
            ):
                continue

            # Check capital
            if signal['direction'] > 0:  # Buying
                if self.book.cash < total_cost:
                    continue
            else:  # Selling - need margin
                required_margin = abs(total_cost) * 0.3  # 30% margin
                if self.book.used_margin + required_margin > Config.INITIAL_CAPITAL * Config.MAX_LEVERAGE:
                    continue

            # Create position
            expiry = market_data.date + pd.Timedelta(days=int(signal['time_to_expiry'] * 365))

            position = OptionPosition(
                entry_date=market_data.date,
                expiry_date=expiry,
                pair=signal['pair'],
                strike=signal['strike'],
                option_type=signal['option_type'],
                position_size=position_size * signal['direction'],
                entry_price=adjusted_price,
                entry_spot=signal['spot'],
                entry_vol=adjusted_vol,
                model_used=self.model_name,
                tenor=signal['tenor'],
                delta_at_entry=signal['delta'] * signal['direction']
            )

            # Calculate initial Greeks
            position.current_delta = BlackScholes.delta(
                signal['spot'], signal['strike'], signal['time_to_expiry'],
                r_d, r_f, adjusted_vol, signal['option_type']
            ) * position_size * signal['direction']

            position.current_vega = BlackScholes.vega(
                signal['spot'], signal['strike'], signal['time_to_expiry'],
                r_d, r_f, adjusted_vol
            ) * position_size * abs(signal['direction'])

            position.current_gamma = BlackScholes.gamma(
                signal['spot'], signal['strike'], signal['time_to_expiry'],
                r_d, r_f, adjusted_vol
            ) * position_size * abs(signal['direction'])

            # Add to book
            self.book.options.append(position)

            # Update cash/margin
            if signal['direction'] > 0:
                self.book.cash -= total_cost
            else:
                self.book.cash += option_premium - brokerage - clearing
                self.book.used_margin += required_margin

            # Log trade
            self.trades_log.append({
                'date': market_data.date,
                'pair': signal['pair'],
                'tenor': signal['tenor'],
                'strike': signal['strike'],
                'option_type': signal['option_type'],
                'direction': signal['direction'],
                'size': position_size,
                'premium': option_premium,
                'total_cost': total_cost,
                'mispricing': signal['mispricing'],
                'vol_regime': vol_regime
            })

    def _delta_hedge(self, market_data: MarketData):
        """Delta hedge the portfolio by currency pair"""
        for pair in Config.PAIRS:
            if pair not in market_data.spot:
                continue

            # Calculate net delta for this pair
            net_delta = self.book.get_net_delta(pair)

            # Check if hedge needed (threshold based on notional)
            threshold = Config.MAX_DELTA_EXPOSURE * Config.INITIAL_CAPITAL / market_data.spot[pair]

            if abs(net_delta) > threshold:
                # Calculate hedge amount
                hedge_amount = -net_delta

                # Calculate transaction cost
                spot_spread = Config.TRANSACTION_COSTS['spot'][pair]
                spot_price = market_data.spot[pair]

                # Cost includes half-spread
                hedge_cost = abs(hedge_amount) * spot_price * (spot_spread * 0.5)
                hedge_cost += abs(hedge_amount) * spot_price * Config.TRANSACTION_COSTS['brokerage']

                # Execute hedge
                self.book.spot_hedges[pair] += hedge_amount
                self.book.cash -= hedge_cost

                # Log hedge
                self.hedges_log.append({
                    'date': market_data.date,
                    'pair': pair,
                    'hedge_amount': hedge_amount,
                    'spot_price': spot_price,
                    'cost': hedge_cost,
                    'net_delta_before': net_delta,
                    'net_delta_after': self.book.get_net_delta(pair)
                })

    def _update_positions(self, market_data: MarketData):
        """Update existing positions with current market data"""
        positions_to_remove = []

        for i, pos in enumerate(self.book.options):
            # Update days held
            pos.days_held = (market_data.date - pos.entry_date).days

            # Check if expired
            if market_data.date >= pos.expiry_date:
                # Calculate final payoff
                spot = market_data.spot.get(pos.pair, pos.entry_spot)
                if pos.option_type == 'call':
                    payoff = max(0, spot - pos.strike)
                else:
                    payoff = max(0, pos.strike - spot)

                # Final P&L
                final_value = payoff
                pos.pnl = (final_value - pos.entry_price) * abs(pos.position_size) * 1_000_000

                # Update cash
                self.book.cash += final_value * pos.position_size * 1_000_000

                # Release margin if short
                if pos.position_size < 0:
                    self.book.used_margin -= abs(pos.entry_price * pos.position_size * 1_000_000 * 0.3)

                positions_to_remove.append(i)
            else:
                # Update mark-to-market
                spot = market_data.spot.get(pos.pair, pos.entry_spot)
                T = max(0, (pos.expiry_date - market_data.date).days / 365)

                if T > 0:
                    # Get current market vol
                    tenor = self._years_to_tenor(T)
                    if tenor in market_data.atm_vols.get(pos.pair, {}):
                        atm_vol = market_data.atm_vols[pos.pair][tenor]
                        rr_25 = market_data.rr_25[pos.pair].get(tenor, 0)
                        bf_25 = market_data.bf_25[pos.pair].get(tenor, 0)
                        rr_10 = market_data.rr_10[pos.pair].get(tenor, 0)
                        bf_10 = market_data.bf_10[pos.pair].get(tenor, 0)

                        # Calculate current delta to get vol
                        current_delta = abs(BlackScholes.delta(
                            spot, pos.strike, T, *self._get_rates(pos.pair, market_data),
                            atm_vol, pos.option_type
                        ))

                        current_vol = VolatilitySurface.get_smile_vol(
                            atm_vol, rr_25, bf_25, rr_10, bf_10,
                            current_delta, pos.option_type
                        )
                    else:
                        current_vol = pos.entry_vol

                    r_d, r_f = self._get_rates(pos.pair, market_data)

                    # Update Greeks
                    pos.current_delta = BlackScholes.delta(
                        spot, pos.strike, T, r_d, r_f, current_vol, pos.option_type
                    )
                    pos.current_vega = BlackScholes.vega(
                        spot, pos.strike, T, r_d, r_f, current_vol
                    )
                    pos.current_gamma = BlackScholes.gamma(
                        spot, pos.strike, T, r_d, r_f, current_vol
                    )

                    # Update value
                    pos.current_value = BlackScholes.price(
                        spot, pos.strike, T, r_d, r_f, current_vol, pos.option_type
                    )

                    # Update P&L
                    pos.pnl = (pos.current_value - pos.entry_price) * pos.position_size * 1_000_000

        # Remove expired positions
        for i in reversed(positions_to_remove):
            del self.book.options[i]

    def _check_exits(self, market_data: MarketData):
        """Check for stop-loss or take-profit exits"""
        positions_to_close = []

        for i, pos in enumerate(self.book.options):
            # Stop-loss: -50% of premium
            if pos.pnl < -0.5 * abs(pos.entry_price * pos.position_size * 1_000_000):
                positions_to_close.append(i)
                continue

            # Take-profit: +100% of premium
            if pos.pnl > 1.0 * abs(pos.entry_price * pos.position_size * 1_000_000):
                positions_to_close.append(i)
                continue

            # Time decay exit: close if less than 2 days to expiry
            days_to_expiry = (pos.expiry_date - market_data.date).days
            if days_to_expiry <= 2:
                positions_to_close.append(i)

        # Close positions
        for i in reversed(positions_to_close):
            pos = self.book.options[i]

            # Calculate exit costs
            spot = market_data.spot.get(pos.pair, pos.entry_spot)
            vol_spread = Config.TRANSACTION_COSTS['options'][pos.tenor]
            exit_cost = abs(pos.current_value * pos.position_size * 1_000_000 * vol_spread * 0.5)

            # Update cash
            if pos.position_size > 0:  # Was long, now selling
                self.book.cash += pos.current_value * pos.position_size * 1_000_000 - exit_cost
            else:  # Was short, now buying back
                self.book.cash -= pos.current_value * abs(pos.position_size) * 1_000_000 + exit_cost
                self.book.used_margin -= abs(pos.entry_price * pos.position_size * 1_000_000 * 0.3)

            del self.book.options[i]

    def _update_pnl(self, market_data: MarketData):
        """Update P&L including spot hedges"""
        # Update spot hedge P&L
        for pair in Config.PAIRS:
            if pair in market_data.spot and self.book.spot_hedges[pair] != 0:
                # Simple daily funding cost for spot position
                funding_rate = 0.0001  # 1 bp per day
                funding_cost = abs(self.book.spot_hedges[pair]) * market_data.spot[pair] * funding_rate
                self.book.cash -= funding_cost

    def _record_stats(self, market_data: MarketData):
        """Record daily statistics"""
        # Calculate total portfolio value
        options_value = sum(
            pos.current_value * pos.position_size * 1_000_000
            for pos in self.book.options
        )

        spot_hedge_value = sum(
            self.book.spot_hedges[pair] * market_data.spot.get(pair, 0)
            for pair in Config.PAIRS
        )

        total_equity = self.book.cash + options_value + spot_hedge_value

        # Update drawdown
        self.book.update_drawdown(total_equity)

        # Record stats
        self.equity_curve.append({
            'date': market_data.date,
            'equity': total_equity,
            'cash': self.book.cash,
            'options_value': options_value,
            'spot_hedge_value': spot_hedge_value,
            'num_positions': len(self.book.options),
            'net_delta': self.book.get_net_delta(),
            'net_vega': self.book.get_net_vega(),
            'net_gamma': self.book.get_net_gamma(),
            'used_margin': self.book.used_margin,
            'drawdown': self.book.current_drawdown
        })

    def _get_rates(self, pair: str, market_data: MarketData) -> Tuple[float, float]:
        """Get interest rates for currency pair"""
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
            r_f = 0.02
        return r_d, r_f

    def _tenor_to_years(self, tenor: str) -> float:
        """Convert tenor string to years"""
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
        elif years <= 2/52:
            return '2W'
        elif years <= 3/52:
            return '3W'
        elif years <= 1/12:
            return '1M'
        elif years <= 2/12:
            return '2M'
        elif years <= 3/12:
            return '3M'
        elif years <= 4/12:
            return '4M'
        elif years <= 6/12:
            return '6M'
        elif years <= 9/12:
            return '9M'
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
    def prepare_market_data(fx_data: pd.DataFrame, date: pd.Timestamp) -> MarketData:
        """Prepare market data for specific date - no look-ahead"""
        if date not in fx_data.index:
            return None

        # Only use data available up to this date
        row = fx_data.loc[date]

        # Default rates
        rates = {
            'USD': 0.05,
            'JPY': 0.01,
            'GBP': 0.04,
            'NZD': 0.03,
            'CAD': 0.04
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
            # Spot price
            spot_col = f'{pair} Curncy'
            if spot_col in row.index and not pd.isna(row[spot_col]):
                market.spot[pair] = row[spot_col]

            # Initialize dicts
            market.atm_vols[pair] = {}
            market.rr_25[pair] = {}
            market.bf_25[pair] = {}
            market.rr_10[pair] = {}
            market.bf_10[pair] = {}

            for tenor in Config.TENORS:
                # ATM vol
                vol_col = f'{pair}V{tenor} Curncy'
                if vol_col in row.index and not pd.isna(row[vol_col]):
                    market.atm_vols[pair][tenor] = row[vol_col] / 100

                # 25-delta risk reversal
                rr25_col = f'{pair}25R{tenor} Curncy'
                if rr25_col in row.index and not pd.isna(row[rr25_col]):
                    market.rr_25[pair][tenor] = row[rr25_col] / 100

                # 25-delta butterfly
                bf25_col = f'{pair}25B{tenor} Curncy'
                if bf25_col in row.index and not pd.isna(row[bf25_col]):
                    market.bf_25[pair][tenor] = row[bf25_col] / 100

                # 10-delta risk reversal
                rr10_col = f'{pair}10R{tenor} Curncy'
                if rr10_col in row.index and not pd.isna(row[rr10_col]):
                    market.rr_10[pair][tenor] = row[rr10_col] / 100

                # 10-delta butterfly
                bf10_col = f'{pair}10B{tenor} Curncy'
                if bf10_col in row.index and not pd.isna(row[bf10_col]):
                    market.bf_10[pair][tenor] = row[bf10_col] / 100

        return market


# ==================== Backtester ====================
class Backtester:
    """Run backtests for models"""

    def __init__(self, fx_data: pd.DataFrame):
        self.fx_data = fx_data
        self.results = {}

    def run_backtest(self, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """Run backtest for all models"""
        models = [
            ('GK', False),       # Garman-Kohlhagen
            ('SABR', False),     # SABR model
            ('GK_IVFilter', True)  # GK with IV filter
        ]

        for model_name, use_filter in models:
            print(f"\nRunning backtest for {model_name}...")

            # Use correct model name for strategy
            actual_model = 'GK' if model_name == 'GK_IVFilter' else model_name
            strategy = OptionsArbitrageStrategy(
                model_name=actual_model,
                use_iv_filter=use_filter
            )

            self._run_single_backtest(strategy, start_date, end_date)
            self.results[model_name] = {
                'equity_curve': strategy.equity_curve,
                'trades': strategy.trades_log,
                'hedges': strategy.hedges_log,
                'vol_regimes': strategy.vol_regimes
            }

    def _run_single_backtest(self, strategy: OptionsArbitrageStrategy,
                            start_date: pd.Timestamp, end_date: pd.Timestamp):
        """Run backtest for single strategy"""
        # Get date range
        dates = self.fx_data.index[
            (self.fx_data.index >= start_date) &
            (self.fx_data.index <= end_date)
        ]

        # Build historical volatility data (no look-ahead)
        historical_vol = {}
        for pair in Config.PAIRS:
            historical_vol[pair] = pd.Series(dtype=float)

        for i, date in enumerate(dates):
            if i % 50 == 0:
                print(f"  Processing {i}/{len(dates)} days...")

            # Prepare market data (only current day)
            market_data = DataLoader.prepare_market_data(self.fx_data, date)
            if market_data is None:
                continue

            # Run strategy
            strategy.run_daily(market_data, historical_vol)

            # Update historical vol (for next day)
            for pair in Config.PAIRS:
                if pair in market_data.atm_vols:
                    # Use 1M ATM vol as representative
                    if '1M' in market_data.atm_vols[pair]:
                        vol_value = market_data.atm_vols[pair]['1M']
                        historical_vol[pair] = pd.concat([
                            historical_vol[pair],
                            pd.Series([vol_value], index=[date])
                        ])

    def calculate_metrics(self) -> pd.DataFrame:
        """Calculate performance metrics"""
        metrics = []

        # Find normalization start date (when all strategies reached capacity)
        max_capacity_dates = {}
        for model_name, results in self.results.items():
            if results['equity_curve']:
                df = pd.DataFrame(results['equity_curve'])
                capacity_threshold = Config.MAX_POSITIONS * 0.8
                mask = df['num_positions'] >= capacity_threshold
                if mask.any():
                    max_capacity_dates[model_name] = df.loc[mask.idxmax(), 'date']

        if max_capacity_dates:
            start_date = max(max_capacity_dates.values())
        else:
            # Fallback: use 30 days after start
            start_date = None
            for results in self.results.values():
                if results['equity_curve'] and len(results['equity_curve']) > 30:
                    start_date = results['equity_curve'][30]['date']
                    break

        for model_name, results in self.results.items():
            equity_curve = results['equity_curve']
            if not equity_curve:
                continue

            df = pd.DataFrame(equity_curve)

            # Filter from steady state if we have a start date
            if start_date:
                df = df[df['date'] >= start_date].copy()
                if not df.empty:
                    # Normalize equity from this point
                    initial_equity = df.iloc[0]['equity']
                    df['equity'] = (df['equity'] / initial_equity) * Config.INITIAL_CAPITAL

            if df.empty:
                continue

            # Calculate returns
            df['returns'] = df['equity'].pct_change()

            # Calculate metrics from normalized data
            total_return = (df['equity'].iloc[-1] / Config.INITIAL_CAPITAL - 1) * 100

            # Sharpe ratio
            if df['returns'].std() > 0:
                sharpe = df['returns'].mean() / df['returns'].std() * np.sqrt(252)
            else:
                sharpe = 0

            # Max drawdown (recalculated)
            running_max = df['equity'].expanding().max()
            drawdown = ((df['equity'] - running_max) / running_max).min()
            max_dd = abs(drawdown) * 100

            # Trade statistics
            trades = results['trades']
            num_trades = len(trades)
            if num_trades > 0:
                trades_df = pd.DataFrame(trades)
                # Only count trades after steady state
                if start_date:
                    trades_df = trades_df[trades_df['date'] >= start_date]
                    num_trades = len(trades_df)
                if len(trades_df) > 0:
                    win_rate = (trades_df['mispricing'] * trades_df['direction'] > 0).mean() * 100
                else:
                    win_rate = 0
            else:
                win_rate = 0

            metrics.append({
                'Model': model_name,
                'Total Return (%)': round(total_return, 2),
                'Sharpe Ratio': round(sharpe, 2),
                'Max Drawdown (%)': round(max_dd, 2),
                'Num Trades': num_trades,
                'Win Rate (%)': round(win_rate, 1),
                'Avg Positions': round(df['num_positions'].mean(), 1)
            })

        return pd.DataFrame(metrics)

    def plot_results(self):
        """Plot equity curves and drawdowns"""
        # Find when all strategies first reach max capacity
        max_capacity_dates = {}
        for model_name, results in self.results.items():
            if results['equity_curve']:
                df = pd.DataFrame(results['equity_curve'])
                # Find first date where positions reach 80% of max capacity
                capacity_threshold = Config.MAX_POSITIONS * 0.8
                mask = df['num_positions'] >= capacity_threshold
                if mask.any():
                    max_capacity_dates[model_name] = df.loc[mask.idxmax(), 'date']

        # Use the latest date when all models reached capacity
        if max_capacity_dates:
            start_date = max(max_capacity_dates.values())
        else:
            # Fallback: start after 30 days to allow position buildup
            start_date = None
            for results in self.results.values():
                if results['equity_curve'] and len(results['equity_curve']) > 30:
                    start_date = results['equity_curve'][30]['date']
                    break

        if start_date is None:
            print("Warning: Not enough data to plot normalized equity curves")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Prepare normalized equity curves
        normalized_data = {}
        for model_name, results in self.results.items():
            if results['equity_curve']:
                df = pd.DataFrame(results['equity_curve'])
                # Filter to start from the normalization date
                df_filtered = df[df['date'] >= start_date].copy()
                if not df_filtered.empty:
                    # Normalize to start at 10M
                    initial_equity = df_filtered.iloc[0]['equity']
                    df_filtered['normalized_equity'] = (df_filtered['equity'] / initial_equity) * Config.INITIAL_CAPITAL
                    normalized_data[model_name] = df_filtered

        # Plot normalized equity curves
        ax = axes[0]
        for model_name, df in normalized_data.items():
            ax.plot(df['date'], df['normalized_equity'] / 1e6, label=model_name, linewidth=2)

        ax.set_title('Equity Curves (Normalized from Steady State)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($M)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5, label='Initial Capital')

        # Plot drawdowns (recalculated from normalized equity)
        ax = axes[1]
        for model_name, df in normalized_data.items():
            # Recalculate drawdown from normalized equity
            running_max = df['normalized_equity'].expanding().max()
            drawdown = (df['normalized_equity'] - running_max) / running_max
            ax.fill_between(df['date'], 0, drawdown * 100,
                           label=model_name, alpha=0.5)

        ax.set_title('Drawdowns (from Steady State)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-15, 1)

        # Add max drawdown line
        ax.axhline(y=-10, color='r', linestyle='--', alpha=0.5, label='10% Max DD')

        # Add text annotation about normalization
        fig.text(0.5, 0.02, f'Note: Performance normalized from {start_date.strftime("%Y-%m-%d")} when all strategies reached steady state',
                ha='center', fontsize=10, style='italic')

        plt.tight_layout()
        plt.savefig(Config.RESULTS_DIR / 'backtest_results.png', dpi=100, bbox_inches='tight')
        plt.show()

    def analyze_regimes(self):
        """Analyze performance in different volatility regimes"""
        for model_name, results in self.results.items():
            if not results['equity_curve'] or not results['vol_regimes']:
                continue

            print(f"\n{model_name} - Volatility Regime Analysis:")

            # Merge equity and regime data
            equity_df = pd.DataFrame(results['equity_curve'])
            regime_df = pd.DataFrame(results['vol_regimes'])

            merged = pd.merge(equity_df, regime_df, on='date', how='inner')
            merged['returns'] = merged['equity'].pct_change()

            # Calculate metrics by regime
            for regime in ['low', 'normal', 'high']:
                regime_data = merged[merged['regime'] == regime]
                if len(regime_data) > 1:
                    avg_return = regime_data['returns'].mean() * 252 * 100
                    vol = regime_data['returns'].std() * np.sqrt(252) * 100
                    sharpe = avg_return / vol if vol > 0 else 0

                    print(f"  {regime.capitalize()} Vol: "
                          f"Return={avg_return:.1f}%, "
                          f"Vol={vol:.1f}%, "
                          f"Sharpe={sharpe:.2f}")


# ==================== Main Execution ====================
def main():
    """Main execution function"""
    print("=" * 80)
    print("FX OPTIONS TRADING SYSTEM - REFACTORED")
    print("=" * 80)

    # Create directories
    Config.DATA_DIR.mkdir(exist_ok=True)
    Config.RESULTS_DIR.mkdir(exist_ok=True)

    # Load data
    print("\nLoading FX data...")
    fx_data = DataLoader.load_fx_data(Config.FX_DATA_PATH)
    print(f"Loaded {len(fx_data)} days of data")
    print(f"Date range: {fx_data.index[0]} to {fx_data.index[-1]}")

    # Use 75/25 train/test split
    n_days = len(fx_data)
    split_idx = int(n_days * 0.75)

    train_end_date = fx_data.index[split_idx]
    test_start_date = fx_data.index[split_idx + 1]
    test_end_date = fx_data.index[-1]

    print(f"\nIn-sample period: {fx_data.index[0]} to {train_end_date}")
    print(f"Out-of-sample period: {test_start_date} to {test_end_date}")

    # Run backtest on out-of-sample data
    print("\n" + "=" * 80)
    print("RUNNING OUT-OF-SAMPLE BACKTEST")
    print("=" * 80)

    backtester = Backtester(fx_data)
    backtester.run_backtest(test_start_date, test_end_date)

    # Calculate metrics
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    metrics = backtester.calculate_metrics()
    print(metrics.to_string(index=False))

    # Save results
    metrics.to_csv(Config.RESULTS_DIR / 'performance_metrics.csv', index=False)

    # Analyze by volatility regime
    print("\n" + "=" * 80)
    print("VOLATILITY REGIME ANALYSIS")
    print("=" * 80)
    backtester.analyze_regimes()

    # Plot results
    print("\nGenerating plots...")
    backtester.plot_results()

    # Export detailed results
    for model_name, results in backtester.results.items():
        # Save equity curve
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df.to_csv(Config.RESULTS_DIR / f'{model_name}_equity.csv', index=False)

        # Save trades
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            trades_df.to_csv(Config.RESULTS_DIR / f'{model_name}_trades.csv', index=False)

    print(f"\n✅ Analysis complete! Results saved to {Config.RESULTS_DIR}")


if __name__ == "__main__":
    main()