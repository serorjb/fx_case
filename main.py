# main.py
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
    MAX_LEVERAGE = 10
    MAX_POSITIONS = 500
    MAX_DELTA_EXPOSURE = 0.04  # 4% max delta at end of day
    MISPRICING_THRESHOLD = 0.025  # 2.5% mispricing to trade

    # Transaction costs
    SPOT_SPREAD = 0.0002  # 2 bps
    OPTION_SPREAD = 0.0005  # 5 bps

    # Model parameters
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


@dataclass
class TradingBook:
    """Manages all positions and hedges"""
    options: List[OptionPosition] = field(default_factory=list)
    spot_hedge: float = 0.0  # Spot position for delta hedging
    cash: float = Config.INITIAL_CAPITAL
    used_margin: float = 0.0

    def get_net_delta(self) -> float:
        """Calculate total portfolio delta"""
        option_delta = sum(pos.current_delta * pos.position_size for pos in self.options)
        return option_delta + self.spot_hedge

    def get_net_vega(self) -> float:
        """Calculate total portfolio vega"""
        return sum(pos.current_vega * pos.position_size for pos in self.options)

    def get_num_positions(self) -> int:
        """Count active positions"""
        return len(self.options)


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

        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
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
            return 1.0 if option_type == 'call' and S > K else 0.0

        from scipy.stats import norm

        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        if option_type == 'call':
            return np.exp(-r_f * T) * norm.cdf(d1)
        else:
            return -np.exp(-r_f * T) * norm.cdf(-d1)

    @staticmethod
    def vega(S: float, K: float, T: float, r_d: float, r_f: float, sigma: float) -> float:
        """Calculate option vega"""
        if T <= 0:
            return 0.0

        from scipy.stats import norm

        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-r_f * T) * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change

    @staticmethod
    def implied_vol_from_delta(delta: float, T: float, r_f: float,
                               option_type: str, spot_vol: float) -> float:
        """Get implied vol for a given delta strike"""
        # Simplified - in practice would use proper inversion
        if option_type == 'call':
            if delta > 0.5:  # ITM call
                return spot_vol * (1 - 0.1 * (delta - 0.5))
            else:  # OTM call
                return spot_vol * (1 + 0.1 * (0.5 - delta))
        else:  # put
            if delta < -0.5:  # ITM put
                return spot_vol * (1 - 0.1 * (abs(delta) - 0.5))
            else:  # OTM put
                return spot_vol * (1 + 0.1 * (0.5 - abs(delta)))

    @staticmethod
    def strike_from_delta(S: float, delta: float, T: float, r_d: float, r_f: float,
                          sigma: float, option_type: str) -> float:
        """Calculate strike from delta"""
        from scipy.stats import norm

        if option_type == 'call':
            d1 = norm.ppf(delta * np.exp(r_f * T))
        else:
            d1 = -norm.ppf(-delta * np.exp(r_f * T))

        K = S * np.exp(-(d1 * sigma * np.sqrt(T) - (r_d - r_f + 0.5 * sigma ** 2) * T))
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
            smile_vol = atm_vol + bf_25 - rr_25 / 2 + 0.002 * (0.95 - moneyness)
        elif moneyness > 1.05:  # Deep OTM call / ITM put region
            smile_vol = atm_vol + bf_25 + rr_25 / 2 + 0.002 * (moneyness - 1.05)
        else:  # Near ATM
            skew_adjustment = rr_25 * (moneyness - 1.0) * 4  # Linear skew
            smile_vol = atm_vol + bf_25 + skew_adjustment

        # Price with adjusted vol
        return BlackScholes.price(S, K, T, r_d, r_f, smile_vol, option_type)


class SABRModel:
    """Simplified SABR model"""

    def __init__(self):
        self.alpha = None
        self.beta = 0.5  # Fixed for FX
        self.rho = -0.3
        self.nu = 0.3

    def calibrate(self, strikes: np.ndarray, vols: np.ndarray, F: float, T: float):
        """Simple calibration - just set alpha to ATM vol"""
        atm_idx = np.argmin(np.abs(strikes - F))
        self.alpha = vols[atm_idx]

    def price(self, S: float, K: float, T: float, r_d: float, r_f: float,
              atm_vol: float, option_type: str) -> float:
        """Price using SABR vol"""
        if self.alpha is None:
            self.alpha = atm_vol

        # Simplified SABR vol (Hagan approximation)
        F = S * np.exp((r_d - r_f) * T)
        if abs(F - K) < 0.001:
            sabr_vol = self.alpha
        else:
            z = (self.nu / self.alpha) * (F * K) ** ((1 - self.beta) / 2) * np.log(F / K)
            x = np.log((np.sqrt(1 - 2 * self.rho * z + z ** 2) + z - self.rho) / (1 - self.rho))
            sabr_vol = self.alpha * (z / x if abs(x) > 0.001 else 1.0)

        return BlackScholes.price(S, K, T, r_d, r_f, sabr_vol, option_type)


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

        # Basic features
        features = [
            T,  # Time to maturity
            strike / spot,  # Moneyness
            market_data.rates.get('USD', 0.05),  # Domestic rate
            market_data.rates.get('JPY', 0.01),  # Foreign rate
            market_data.atm_vols[pair].get(tenor, 0.1),  # ATM vol
            market_data.rr_25[pair].get(tenor, 0),  # Risk reversal
            market_data.bf_25[pair].get(tenor, 0),  # Butterfly
        ]

        # Add historical vol if available
        if historical_data is not None and len(historical_data) > 20:
            returns = np.log(historical_data['spot'] / historical_data['spot'].shift(1))
            realized_vol = returns.tail(20).std() * np.sqrt(252)
            features.append(realized_vol)
        else:
            features.append(0.1)  # Default RV

        # Add other model prices as features (would be calculated separately)
        features.extend([0.0, 0.0, 0.0])  # Placeholder for GK, GVV, SABR prices

        return np.array(features).reshape(1, -1)

    def train(self, training_data: pd.DataFrame):
        """Train the model on historical data"""
        # Simplified - would implement full training logic
        import lightgbm as lgb

        # Prepare training data
        X = training_data[['moneyness', 'time_to_maturity', 'atm_vol', 'rr', 'bf']].values
        y = training_data['option_price'].values

        # Train model
        self.model = lgb.LGBMRegressor()
        self.model.fit(X, y)

    def price(self, S: float, K: float, T: float, r_d: float, r_f: float,
              features: np.ndarray) -> float:
        """Price option using trained model"""
        if self.model is None:
            # Fallback to Black-Scholes if not trained
            return BlackScholes.price(S, K, T, r_d, r_f, 0.1, 'call')

        return self.model.predict(features)[0]

    def _tenor_to_years(self, tenor: str) -> float:
        """Convert tenor to years"""
        mapping = {
            '1W': 1 / 52, '2W': 2 / 52, '3W': 3 / 52,
            '1M': 1 / 12, '2M': 2 / 12, '3M': 3 / 12,
            '4M': 4 / 12, '6M': 6 / 12, '9M': 9 / 12,
            '1Y': 1.0
        }
        return mapping.get(tenor, 1 / 12)


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
        self.delta_model = delta_model or Config.DELTA_MODEL  # Use config default if not specified
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
                smile_vol = vol + bf_25 - rr_25 / 2 + 0.002 * (0.95 - moneyness)
            elif moneyness > 1.05:
                smile_vol = vol + bf_25 + rr_25 / 2 + 0.002 * (moneyness - 1.05)
            else:
                skew_adjustment = rr_25 * (moneyness - 1.0) * 4
                smile_vol = vol + bf_25 + skew_adjustment

            # Use adjusted vol for delta
            return BlackScholes.delta(spot, strike, T, r_d, r_f, smile_vol, option_type)

        elif self.delta_model == 'SABR':
            # SABR delta calculation
            if self.delta_sabr_model.alpha is None:
                # Calibrate SABR if not done
                strikes = np.array([spot * 0.9, spot, spot * 1.1])
                vols = np.array([vol * 1.1, vol, vol * 1.1])
                self.delta_sabr_model.calibrate(strikes, vols, spot, T)

            # Calculate SABR vol for this strike
            F = spot * np.exp((r_d - r_f) * T)
            if abs(F - strike) < 0.001:
                sabr_vol = self.delta_sabr_model.alpha
            else:
                z = (self.delta_sabr_model.nu / self.delta_sabr_model.alpha) * \
                    (F * strike) ** ((1 - self.delta_sabr_model.beta) / 2) * np.log(F / strike)
                x = np.log((np.sqrt(1 - 2 * self.delta_sabr_model.rho * z + z ** 2) + z -
                            self.delta_sabr_model.rho) / (1 - self.delta_sabr_model.rho))
                sabr_vol = self.delta_sabr_model.alpha * (z / x if abs(x) > 0.001 else 1.0)

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

            # Get proper rates from curves for this pair
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

                        # Get market implied vol (from smile)
                        market_vol = self._get_smile_vol(
                            market_data, pair, tenor, strike / spot
                        )

                        # Calculate market price using Black-Scholes
                        market_price = BlackScholes.price(
                            spot, strike, T, r_d, r_f, market_vol, option_type
                        )

                        # Calculate model price
                        model_price = self._calculate_model_price(
                            market_data, pair, spot, strike, T, r_d, r_f,
                            tenor, option_type, historical_data
                        )

                        # Check for mispricing
                        if market_price > 0:
                            mispricing = (model_price - market_price) / market_price

                            if abs(mispricing) > Config.MISPRICING_THRESHOLD:
                                # Apply IV filter
                                if self.use_iv_filter:
                                    if mispricing > 0 and atm_vol < iv_ma:
                                        continue  # Skip long if IV below MA
                                    if mispricing < 0 and atm_vol > iv_ma:
                                        continue  # Skip short if IV above MA

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
                                    'vol': market_vol,
                                    'delta': delta
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
            atm_vol = market_data.atm_vols[pair].get(tenor, 0.1)
            # Calibrate SABR if needed
            if self.sabr_model.alpha is None:
                strikes = np.array([S * 0.9, S, S * 1.1])
                vols = np.array([atm_vol * 1.1, atm_vol, atm_vol * 1.1])
                self.sabr_model.calibrate(strikes, vols, S, T)
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

    def _get_smile_vol(self, market_data: MarketData, pair: str,
                       tenor: str, moneyness: float) -> float:
        """Get implied vol from smile for given moneyness"""
        atm_vol = market_data.atm_vols[pair].get(tenor, 0.1)
        rr_25 = market_data.rr_25[pair].get(tenor, 0)
        bf_25 = market_data.bf_25[pair].get(tenor, 0)

        # Simple smile interpolation
        if moneyness < 0.95:  # OTM put
            return atm_vol + bf_25 - rr_25 / 2
        elif moneyness > 1.05:  # OTM call
            return atm_vol + bf_25 + rr_25 / 2
        else:  # Near ATM
            skew = rr_25 * (moneyness - 1.0) * 4
            return atm_vol + bf_25 + skew

    def _execute_trades(self, signals: List[Dict], market_data: MarketData):
        """Execute trades based on signals"""
        for signal in signals:
            # Check margin/capital
            premium = signal['market_price'] * signal['spot'] * abs(signal['direction'])

            if signal['direction'] > 0:  # Buying
                if self.book.cash < premium:
                    continue  # Not enough cash
            else:  # Selling (need margin)
                required_margin = premium * 0.2  # 20% margin for short
                if self.book.used_margin + required_margin > Config.INITIAL_CAPITAL * Config.MAX_LEVERAGE:
                    continue  # Exceeds leverage limit

            # Create position
            T = self._tenor_to_years(signal['tenor'])
            expiry = market_data.date + pd.Timedelta(days=int(T * 365))

            position = OptionPosition(
                entry_date=market_data.date,
                expiry_date=expiry,
                pair=signal['pair'],
                strike=signal['strike'],
                option_type=signal['option_type'],
                position_size=signal['direction'],
                entry_price=signal['market_price'],
                entry_spot=signal['spot'],
                entry_vol=signal['vol'],
                model_used=self.model_name
            )

            # Update greeks
            position.current_delta = BlackScholes.delta(
                signal['spot'], signal['strike'], T, 0.05, 0.01,
                signal['vol'], signal['option_type']
            )
            position.current_vega = BlackScholes.vega(
                signal['spot'], signal['strike'], T, 0.05, 0.01, signal['vol']
            )

            # Add to book
            self.book.options.append(position)

            # Update cash/margin
            if signal['direction'] > 0:
                self.book.cash -= premium * (1 + Config.OPTION_SPREAD)
            else:
                self.book.cash += premium * (1 - Config.OPTION_SPREAD)
                self.book.used_margin += premium * 0.2

            # Log trade
            self.trades_log.append({
                'date': market_data.date,
                'pair': signal['pair'],
                'strike': signal['strike'],
                'option_type': signal['option_type'],
                'direction': signal['direction'],
                'premium': premium,
                'mispricing': signal['mispricing']
            })

    def _delta_hedge(self, market_data: MarketData):
        """Delta hedge the portfolio to stay within limits"""
        net_delta = self.book.get_net_delta()

        if abs(net_delta) > Config.MAX_DELTA_EXPOSURE * Config.INITIAL_CAPITAL:
            # Need to hedge
            hedge_amount = -net_delta

            # Execute spot hedge (simplified - assumes single pair)
            spot_price = list(market_data.spot.values())[0]
            hedge_cost = abs(hedge_amount) * spot_price * Config.SPOT_SPREAD

            self.book.spot_hedge += hedge_amount
            self.book.cash -= hedge_cost

            # Log hedge
            self.hedges_log.append({
                'date': market_data.date,
                'hedge_amount': hedge_amount,
                'spot_price': spot_price,
                'cost': hedge_cost
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

                pos.pnl = (payoff - pos.entry_price) * pos.position_size * pos.entry_spot
                self.book.cash += payoff * pos.position_size * pos.entry_spot

                if pos.position_size < 0:  # Release margin for shorts
                    self.book.used_margin -= pos.entry_price * pos.entry_spot * 0.2

                positions_to_remove.append(i)
            else:
                # Update greeks and mark-to-market
                spot = market_data.spot[pos.pair]
                T = (pos.expiry_date - market_data.date).days / 365
                vol = self._get_smile_vol(market_data, pos.pair,
                                          self._years_to_tenor(T), pos.strike / spot)

                # Get proper rates for the pair
                if pos.pair == 'USDJPY':
                    r_d = market_data.rates.get('USD', 0.05)
                    r_f = market_data.rates.get('JPY', 0.01)
                elif pos.pair == 'GBPNZD':
                    r_d = market_data.rates.get('GBP', 0.04)
                    r_f = market_data.rates.get('NZD', 0.03)
                elif pos.pair == 'USDCAD':
                    r_d = market_data.rates.get('USD', 0.05)
                    r_f = market_data.rates.get('CAD', 0.04)
                else:
                    r_d = 0.05
                    r_f = 0.01

                pos.current_delta = BlackScholes.delta(
                    spot, pos.strike, T, r_d, r_f, vol, pos.option_type
                )
                pos.current_vega = BlackScholes.vega(
                    spot, pos.strike, T, r_d, r_f, vol
                )
                pos.current_value = BlackScholes.price(
                    spot, pos.strike, T, r_d, r_f, vol, pos.option_type
                )
                pos.pnl = (pos.current_value - pos.entry_price) * pos.position_size * pos.entry_spot

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
            'net_delta': self.book.get_net_delta(),
            'net_vega': self.book.get_net_vega(),
            'used_margin': self.book.used_margin
        })

    def _tenor_to_years(self, tenor: str) -> float:
        """Convert tenor to years"""
        mapping = {
            '1W': 1 / 52, '2W': 2 / 52, '3W': 3 / 52,
            '1M': 1 / 12, '2M': 2 / 12, '3M': 3 / 12,
            '4M': 4 / 12, '6M': 6 / 12, '9M': 9 / 12,
            '1Y': 1.0
        }
        return mapping.get(tenor, 1 / 12)

    def _years_to_tenor(self, years: float) -> str:
        """Convert years to nearest tenor"""
        if years <= 1 / 52:
            return '1W'
        elif years <= 1 / 12:
            return '1M'
        elif years <= 3 / 12:
            return '3M'
        elif years <= 6 / 12:
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

        # Get 3M rate as representative (could use different tenors for different currencies)
        rates = {}

        # USD rate from the curves
        usd_3m = date_curves[date_curves['tenor'] == '3M']['interpolated_rate'].values
        rates['USD'] = usd_3m[0] if len(usd_3m) > 0 else 0.05

        # For other currencies, apply spreads (simplified - in practice would have separate curves)
        rates['JPY'] = rates['USD'] - 0.04  # Japan typically lower rates
        rates['GBP'] = rates['USD'] - 0.01  # UK similar to US
        rates['NZD'] = rates['USD'] - 0.02  # NZ slightly lower
        rates['CAD'] = rates['USD'] - 0.01  # Canada similar to US

        return rates

    @staticmethod
    def get_forward_points(curves_data: pd.DataFrame, date: pd.Timestamp,
                           pair: str, tenor: str) -> float:
        """Calculate forward points from interest rate differential"""
        if curves_data is None:
            return 0.0

        # Get rates for both currencies
        rates = DataLoader.get_rates_from_curves(curves_data, date)

        # Parse currency pair
        if pair == 'USDJPY':
            r_base = rates['USD']
            r_quote = rates['JPY']
        elif pair == 'GBPNZD':
            r_base = rates['GBP']
            r_quote = rates['NZD']
        elif pair == 'USDCAD':
            r_base = rates['USD']
            r_quote = rates['CAD']
        else:
            return 0.0

        # Get tenor in years
        tenor_map = {
            '1W': 1 / 52, '2W': 2 / 52, '3W': 3 / 52,
            '1M': 1 / 12, '2M': 2 / 12, '3M': 3 / 12,
            '4M': 4 / 12, '6M': 6 / 12, '9M': 9 / 12,
            '1Y': 1.0
        }
        T = tenor_map.get(tenor, 1 / 12)

        # Calculate forward points (simplified)
        # Forward = Spot * exp((r_base - r_quote) * T)
        # Forward points = Forward - Spot = Spot * (exp((r_base - r_quote) * T) - 1)
        # This is approximate - actual market forward points include bid-ask and other factors

        # For now, return simplified calculation
        # In practice, would read from market data
        forward_factor = np.exp((r_base - r_quote) * T) - 1

        return forward_factor

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


# ==================== Backtester ====================
class Backtester:
    """Run backtests for GK, SABR, and GK_IVFilter models"""

    def __init__(self, fx_data: pd.DataFrame, curves_data: pd.DataFrame = None):
        self.fx_data = fx_data
        self.curves_data = curves_data
        self.results = {}

    def run_all_models(self, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """Run backtest for GK, SABR, and GK with IV filter"""
        models = ['GK', 'SABR']

        for model in models:
            print(f"\nRunning backtest for {model}...")
            strategy = OptionsArbitrageStrategy(model_name=model)
            self._run_single_backtest(strategy, start_date, end_date)
            self.results[model] = strategy.equity_curve

        # Also run GK with IV filter
        print(f"\nRunning backtest for GK with IV filter...")
        strategy = OptionsArbitrageStrategy(model_name='GK', use_iv_filter=True)
        self._run_single_backtest(strategy, start_date, end_date)
        self.results['GK_IVFilter'] = strategy.equity_curve

    def _run_single_backtest(self, strategy: OptionsArbitrageStrategy,
                             start_date: pd.Timestamp, end_date: pd.Timestamp):
        """Run backtest for a single strategy"""
        dates = self.fx_data.index[(self.fx_data.index >= start_date) &
                                   (self.fx_data.index <= end_date)]

        historical_data = {pair: pd.DataFrame() for pair in Config.PAIRS}

        for i, date in enumerate(dates):
            if i % 50 == 0:
                print(f"  Processing {i}/{len(dates)} days...")

            market_data = DataLoader.prepare_market_data(self.fx_data, date, self.curves_data)
            if market_data is None:
                continue

            for pair in Config.PAIRS:
                if pair in market_data.spot:
                    new_row = pd.DataFrame({
                        'date': [date],
                        'spot': [market_data.spot[pair]]
                    })
                    for tenor in Config.TENORS:
                        if tenor in market_data.atm_vols[pair]:
                            new_row[f'atm_vol_{tenor}'] = market_data.atm_vols[pair][tenor]
                    historical_data[pair] = pd.concat([historical_data[pair], new_row], ignore_index=True)

            strategy.run_daily(market_data, historical_data)

    def calculate_performance_metrics(self) -> pd.DataFrame:
        """Calculate performance metrics for all strategies"""
        metrics = []

        for model, equity_curve in self.results.items():
            if not equity_curve:
                continue

            df = pd.DataFrame(equity_curve)
            returns = df['equity'].pct_change().dropna()

            metrics.append({
                'Model': model,
                'Total Return': (df['equity'].iloc[-1] / Config.INITIAL_CAPITAL - 1) * 100,
                'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                'Max Drawdown': ((df['equity'] / df['equity'].expanding().max() - 1).min()) * 100,
                # 'Avg Positions': df['num_positions'].mean(),
                # 'Total Trades': len([e for e in equity_curve if e['num_positions'] > 0])
            })

        return pd.DataFrame(metrics)

    def plot_results(self):
        """Plot equity curves and statistics for GK, SABR, and GK_IVFilter"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Only plot GK, SABR, and GK_IVFilter
        plot_models = ['GK', 'SABR', 'GK_IVFilter']

        # Equity curves
        ax = axes[0]
        for model in plot_models:
            equity_curve = self.results.get(model)
            if equity_curve:
                df = pd.DataFrame(equity_curve)
                ax.plot(df['date'], df['equity'], label=model)
        ax.set_title('Equity Curves')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.legend()
        ax.grid(True)

        # Drawdown
        ax = axes[1]
        for model in plot_models:
            equity_curve = self.results.get(model)
            if equity_curve:
                df = pd.DataFrame(equity_curve)
                dd = (df['equity'] / df['equity'].expanding().max() - 1) * 100
                ax.plot(df['date'], dd, label=model)
        ax.set_title('Drawdown')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
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
        print("Warning: Using default rates instead of curves")

    # Split data
    n_days = len(fx_data)
    train_end = int(n_days * 0.85)

    train_data = fx_data.iloc[:train_end]
    test_data = fx_data.iloc[train_end:]

    print(f"\nTrain period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")

    # Run backtests with curves data
    backtester = Backtester(fx_data, curves_data)
    backtester.run_all_models(test_data.index[0], test_data.index[-1])

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