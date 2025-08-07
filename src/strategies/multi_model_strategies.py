# src/strategies/multi_model_strategies.py
"""
Multi-model trading strategies for FX options
Implements GVV, SABR, ML-based, and consensus strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class Signal:
    """Trading signal with metadata"""
    timestamp: pd.Timestamp
    pair: str
    direction: str  # 'buy' or 'sell'
    option_type: str  # 'call' or 'put'
    strength: float  # Signal strength [0, 1]
    model: str  # Model that generated the signal
    confidence: float  # Confidence level
    metadata: Dict = None


class BaseModelStrategy:
    """Base class for model-based trading strategies"""

    def __init__(self, name: str = "BaseStrategy", enable_hedging: bool = False):
        self.name = name
        self.enable_hedging = enable_hedging
        self.positions = {}
        self.signals = []
        self.hedge_history = []

    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame = None) -> List[Signal]:
        """Generate trading signals - to be implemented by subclasses"""
        raise NotImplementedError

    def calculate_position_size(self, signal: Signal, capital: float) -> float:
        """Calculate position size based on signal strength and Kelly criterion"""
        base_size = capital * 0.02  # 2% base position

        # Adjust by signal strength
        position_size = base_size * signal.strength

        # Apply Kelly criterion if we have confidence
        if signal.confidence > 0:
            kelly_fraction = min(signal.confidence * signal.strength, 0.25)  # Cap at 25%
            position_size = capital * kelly_fraction * 0.1  # Conservative Kelly

        return min(position_size, capital * 0.05)  # Max 5% per position

    def should_exit_position(self, position: Dict, current_data: pd.Series) -> bool:
        """Determine if position should be exited"""
        # Exit if position has been held for more than 20 days
        if (current_data.name - position['entry_date']).days > 20:
            return True

        # Exit if stop loss hit (10% loss)
        if position.get('pnl', 0) < -position['size'] * 0.1:
            return True

        # Exit if take profit hit (20% gain)
        if position.get('pnl', 0) > position['size'] * 0.2:
            return True

        return False


class GVVArbitrageStrategy(BaseModelStrategy):
    """
    Garman-Kohlhagen-Vanna-Volga arbitrage strategy
    Trades mispricings between GVV model and market prices
    """

    def __init__(self, lookback_window: int = 20, mispricing_threshold: float = 2.0,
                 enable_hedging: bool = False):
        super().__init__(name="GVV_Arbitrage", enable_hedging=enable_hedging)
        self.lookback_window = lookback_window
        self.mispricing_threshold = mispricing_threshold

    def calculate_gvv_price(self, data: pd.Series) -> Tuple[float, float]:
        """Calculate GVV model price for options"""
        # Extract parameters
        S = data.get('spot', 100)
        r_d = data.get('rate_domestic', 0.05)
        r_f = data.get('rate_foreign', 0.01)
        T = 1 / 12  # 1 month option

        # Market quotes
        atm_vol = data.get('atm_vol_1M', 0.1)
        rr_25 = data.get('rr_25_1M', 0)
        bf_25 = data.get('bf_25_1M', 0)

        # Simple GVV approximation
        # In practice, this would use the full GVV model
        K_atm = S * np.exp((r_d - r_f) * T)

        # Volatility smile adjustment
        call_vol_25d = atm_vol + bf_25 + rr_25 / 2
        put_vol_25d = atm_vol + bf_25 - rr_25 / 2

        # Black-Scholes with smile-adjusted vol
        d1_call = (np.log(S / K_atm) + (r_d - r_f + 0.5 * call_vol_25d ** 2) * T) / (call_vol_25d * np.sqrt(T))
        d2_call = d1_call - call_vol_25d * np.sqrt(T)

        from scipy.stats import norm
        call_price = S * np.exp(-r_f * T) * norm.cdf(d1_call) - K_atm * np.exp(-r_d * T) * norm.cdf(d2_call)

        d1_put = (np.log(S / K_atm) + (r_d - r_f + 0.5 * put_vol_25d ** 2) * T) / (put_vol_25d * np.sqrt(T))
        d2_put = d1_put - put_vol_25d * np.sqrt(T)

        put_price = K_atm * np.exp(-r_d * T) * norm.cdf(-d2_put) - S * np.exp(-r_f * T) * norm.cdf(-d1_put)

        return call_price, put_price

    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame = None) -> List[Signal]:
        """Generate GVV arbitrage signals"""
        signals = []

        if len(data) < self.lookback_window:
            return signals

        # Get latest data
        latest = data.iloc[-1]
        historical = data.iloc[-self.lookback_window:]

        # Calculate GVV prices
        try:
            gvv_call, gvv_put = self.calculate_gvv_price(latest)
        except Exception as e:
            return signals

        # Get market implied prices (using ATM vol as proxy)
        market_call = latest.get('spot', 100) * latest.get('atm_vol_1M', 0.1) * 0.4 * np.sqrt(1 / 12)
        market_put = market_call  # Simplified

        # Calculate mispricings
        call_mispricing = (gvv_call - market_call) / market_call if market_call > 0 else 0
        put_mispricing = (gvv_put - market_put) / market_put if market_put > 0 else 0

        # Calculate z-scores
        historical_call_mispricing = []
        historical_put_mispricing = []

        for _, row in historical.iterrows():
            try:
                h_gvv_call, h_gvv_put = self.calculate_gvv_price(row)
                h_market_call = row.get('spot', 100) * row.get('atm_vol_1M', 0.1) * 0.4 * np.sqrt(1 / 12)
                h_market_put = h_market_call

                historical_call_mispricing.append(
                    (h_gvv_call - h_market_call) / h_market_call if h_market_call > 0 else 0)
                historical_put_mispricing.append((h_gvv_put - h_market_put) / h_market_put if h_market_put > 0 else 0)
            except:
                continue

        if len(historical_call_mispricing) > 0:
            mean_call = np.mean(historical_call_mispricing)
            std_call = np.std(historical_call_mispricing)
            if std_call > 0:
                z_call = (call_mispricing - mean_call) / std_call

                if abs(z_call) > self.mispricing_threshold:
                    signal = Signal(
                        timestamp=data.index[-1],
                        pair=data.columns[0] if len(data.columns) > 0 else "UNKNOWN",
                        direction='sell' if z_call > 0 else 'buy',  # Sell if overpriced
                        option_type='call',
                        strength=min(abs(z_call) / 3.0, 1.0),
                        model='GVV',
                        confidence=0.6 + 0.1 * min(abs(z_call) - self.mispricing_threshold, 2.0),
                        metadata={'mispricing': call_mispricing, 'z_score': z_call}
                    )
                    signals.append(signal)

        if len(historical_put_mispricing) > 0:
            mean_put = np.mean(historical_put_mispricing)
            std_put = np.std(historical_put_mispricing)
            if std_put > 0:
                z_put = (put_mispricing - mean_put) / std_put

                if abs(z_put) > self.mispricing_threshold:
                    signal = Signal(
                        timestamp=data.index[-1],
                        pair=data.columns[0] if len(data.columns) > 0 else "UNKNOWN",
                        direction='sell' if z_put > 0 else 'buy',
                        option_type='put',
                        strength=min(abs(z_put) / 3.0, 1.0),
                        model='GVV',
                        confidence=0.6 + 0.1 * min(abs(z_put) - self.mispricing_threshold, 2.0),
                        metadata={'mispricing': put_mispricing, 'z_score': z_put}
                    )
                    signals.append(signal)

        return signals


class SABRVolatilityStrategy(BaseModelStrategy):
    """
    SABR model-based volatility trading strategy
    Trades based on SABR calibration and volatility surface analysis
    """

    def __init__(self, calibration_window: int = 60, vol_threshold: float = 2.0,
                 enable_hedging: bool = False):
        super().__init__(name="SABR_Volatility", enable_hedging=enable_hedging)
        self.calibration_window = calibration_window
        self.vol_threshold = vol_threshold
        self.sabr_params = {}

    def calibrate_sabr(self, data: pd.DataFrame) -> Dict:
        """Calibrate SABR model parameters"""
        # Simplified SABR calibration
        # In practice, this would use full SABR calibration

        atm_vols = data['atm_vol_1M'].dropna()
        if len(atm_vols) < 20:
            return None

        # Estimate SABR parameters
        alpha = atm_vols.mean()  # Initial vol
        beta = 0.5  # CEV parameter (0.5 for FX typically)
        rho = data[['atm_vol_1M', 'spot']].corr().iloc[0, 1] if 'spot' in data.columns else -0.3
        nu = atm_vols.std() * np.sqrt(252)  # Vol of vol

        return {
            'alpha': alpha,
            'beta': beta,
            'rho': rho,
            'nu': nu
        }

    def calculate_sabr_vol(self, params: Dict, K: float, F: float, T: float) -> float:
        """Calculate SABR implied volatility"""
        if params is None:
            return 0.1  # Default vol

        alpha = params['alpha']
        beta = params['beta']
        rho = params['rho']
        nu = params['nu']

        # SABR formula (simplified)
        if K == F:
            # ATM case
            vol = alpha * (1 + (nu ** 2 / 24) * T)
        else:
            # General case (simplified)
            z = (nu / alpha) * (F * K) ** ((1 - beta) / 2) * np.log(F / K)
            x = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

            vol = alpha * (z / x if abs(x) > 0.001 else 1.0)

        return vol

    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame = None) -> List[Signal]:
        """Generate SABR-based trading signals"""
        signals = []

        if len(data) < self.calibration_window:
            return signals

        # Calibrate SABR model
        calibration_data = data.iloc[-self.calibration_window:]
        sabr_params = self.calibrate_sabr(calibration_data)

        if sabr_params is None:
            return signals

        latest = data.iloc[-1]
        spot = latest.get('spot', 100)

        # Calculate SABR vols for different strikes
        strikes = [spot * m for m in [0.9, 0.95, 1.0, 1.05, 1.1]]
        T = 1 / 12  # 1 month

        sabr_vols = []
        for K in strikes:
            sabr_vol = self.calculate_sabr_vol(sabr_params, K, spot, T)
            sabr_vols.append(sabr_vol)

        # Compare with market vols
        market_atm_vol = latest.get('atm_vol_1M', 0.1)
        sabr_atm_vol = sabr_vols[2]  # ATM strike

        # Calculate vol spread
        vol_spread = sabr_atm_vol - market_atm_vol

        # Historical vol spreads for z-score
        historical_spreads = []
        for i in range(self.calibration_window, len(data)):
            hist_window = data.iloc[i - self.calibration_window:i]
            hist_params = self.calibrate_sabr(hist_window)
            if hist_params:
                hist_spot = data.iloc[i]['spot']
                hist_sabr_vol = self.calculate_sabr_vol(hist_params, hist_spot, hist_spot, T)
                hist_market_vol = data.iloc[i].get('atm_vol_1M', 0.1)
                historical_spreads.append(hist_sabr_vol - hist_market_vol)

        if len(historical_spreads) > 10:
            mean_spread = np.mean(historical_spreads)
            std_spread = np.std(historical_spreads)

            if std_spread > 0:
                z_score = (vol_spread - mean_spread) / std_spread

                if abs(z_score) > self.vol_threshold:
                    # Generate signal based on vol mispricing
                    signal = Signal(
                        timestamp=data.index[-1],
                        pair=data.columns[0] if len(data.columns) > 0 else "UNKNOWN",
                        direction='buy' if z_score < 0 else 'sell',  # Buy vol if underpriced
                        option_type='straddle',  # Trade volatility via straddle
                        strength=min(abs(z_score) / 3.0, 1.0),
                        model='SABR',
                        confidence=0.5 + 0.15 * min(abs(z_score) - self.vol_threshold, 2.0),
                        metadata={
                            'vol_spread': vol_spread,
                            'z_score': z_score,
                            'sabr_params': sabr_params
                        }
                    )
                    signals.append(signal)

        # Also check for skew trading opportunities
        if len(sabr_vols) == 5:
            skew = sabr_vols[0] - sabr_vols[4]  # 90% - 110% skew

            if abs(skew) > 0.02:  # Significant skew
                signal = Signal(
                    timestamp=data.index[-1],
                    pair=data.columns[0] if len(data.columns) > 0 else "UNKNOWN",
                    direction='buy' if skew > 0 else 'sell',
                    option_type='risk_reversal',  # Trade skew via risk reversal
                    strength=min(abs(skew) / 0.05, 1.0),
                    model='SABR_Skew',
                    confidence=0.6,
                    metadata={'skew': skew}
                )
                signals.append(signal)

        return signals


class MLSignalStrategy(BaseModelStrategy):
    """
    Machine learning-based trading strategy
    Uses trained ML models to generate signals
    """

    def __init__(self, ml_model: Any = None, signal_threshold: float = 0.6,
                 enable_hedging: bool = False):
        super().__init__(name="ML_Signal", enable_hedging=enable_hedging)
        self.ml_model = ml_model
        self.signal_threshold = signal_threshold

    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame = None) -> List[Signal]:
        """Generate ML-based trading signals"""
        signals = []

        if self.ml_model is None or features is None or len(features) == 0:
            return signals

        try:
            # Get latest features
            latest_features = features.iloc[-1:].values.reshape(1, -1)

            # Get model prediction
            if hasattr(self.ml_model, 'predict_proba'):
                # Classification model
                proba = self.ml_model.predict_proba(latest_features)[0]

                # Assuming binary classification (0: down, 1: up)
                if len(proba) >= 2:
                    up_prob = proba[1]
                    down_prob = proba[0]

                    if up_prob > self.signal_threshold:
                        signal = Signal(
                            timestamp=data.index[-1],
                            pair=data.columns[0] if len(data.columns) > 0 else "UNKNOWN",
                            direction='buy',
                            option_type='call',
                            strength=(up_prob - 0.5) * 2,  # Scale to [0, 1]
                            model='ML',
                            confidence=up_prob,
                            metadata={'probability': up_prob}
                        )
                        signals.append(signal)

                    elif down_prob > self.signal_threshold:
                        signal = Signal(
                            timestamp=data.index[-1],
                            pair=data.columns[0] if len(data.columns) > 0 else "UNKNOWN",
                            direction='buy',
                            option_type='put',
                            strength=(down_prob - 0.5) * 2,
                            model='ML',
                            confidence=down_prob,
                            metadata={'probability': down_prob}
                        )
                        signals.append(signal)

            elif hasattr(self.ml_model, 'predict'):
                # Regression model
                prediction = self.ml_model.predict(latest_features)[0]

                # Convert regression output to signal
                if abs(prediction) > 0.001:  # Threshold for significant move
                    signal = Signal(
                        timestamp=data.index[-1],
                        pair=data.columns[0] if len(data.columns) > 0 else "UNKNOWN",
                        direction='buy',
                        option_type='call' if prediction > 0 else 'put',
                        strength=min(abs(prediction) * 100, 1.0),
                        model='ML',
                        confidence=0.5 + min(abs(prediction) * 50, 0.4),
                        metadata={'prediction': prediction}
                    )
                    signals.append(signal)

        except Exception as e:
            # Silent fail - no signals if ML model fails
            pass

        return signals


class ModelConsensusStrategy(BaseModelStrategy):
    """
    Consensus strategy that combines signals from multiple models
    Only trades when multiple models agree
    """

    def __init__(self, models: List[str] = None, consensus_threshold: float = 0.6,
                 enable_hedging: bool = False):
        super().__init__(name="Model_Consensus", enable_hedging=enable_hedging)
        self.models = models or ['gvv', 'sabr', 'ml']
        self.consensus_threshold = consensus_threshold
        self.sub_strategies = {}

        # Initialize sub-strategies
        if 'gvv' in self.models:
            self.sub_strategies['gvv'] = GVVArbitrageStrategy(enable_hedging=False)
        if 'sabr' in self.models:
            self.sub_strategies['sabr'] = SABRVolatilityStrategy(enable_hedging=False)

    def add_ml_strategy(self, ml_model: Any):
        """Add ML strategy to consensus"""
        if 'ml' in self.models:
            self.sub_strategies['ml'] = MLSignalStrategy(ml_model=ml_model, enable_hedging=False)

    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame = None) -> List[Signal]:
        """Generate consensus signals from multiple models"""
        all_signals = []
        model_signals = {}

        # Collect signals from all sub-strategies
        for model_name, strategy in self.sub_strategies.items():
            try:
                if model_name == 'ml' and features is not None:
                    signals = strategy.generate_signals(data, features)
                else:
                    signals = strategy.generate_signals(data, None)
                model_signals[model_name] = signals
            except:
                model_signals[model_name] = []

        # Find consensus signals
        consensus_signals = []

        # Group signals by option type and direction
        signal_groups = {}
        for model_name, signals in model_signals.items():
            for signal in signals:
                key = (signal.option_type, signal.direction)
                if key not in signal_groups:
                    signal_groups[key] = []
                signal_groups[key].append((model_name, signal))

        # Check for consensus
        for (option_type, direction), group in signal_groups.items():
            if len(group) >= len(self.sub_strategies) * self.consensus_threshold:
                # Consensus achieved
                avg_strength = np.mean([s.strength for _, s in group])
                avg_confidence = np.mean([s.confidence for _, s in group])

                consensus_signal = Signal(
                    timestamp=data.index[-1],
                    pair=group[0][1].pair,
                    direction=direction,
                    option_type=option_type,
                    strength=avg_strength,
                    model='Consensus',
                    confidence=avg_confidence,
                    metadata={
                        'models_agree': [m for m, _ in group],
                        'num_models': len(group)
                    }
                )
                consensus_signals.append(consensus_signal)

        return consensus_signals


class CarryArbitrageStrategy(BaseModelStrategy):
    """
    Carry trade arbitrage strategy
    Exploits interest rate differentials with option hedging
    """

    def __init__(self, min_carry: float = 0.02, enable_hedging: bool = True):
        super().__init__(name="Carry_Arbitrage", enable_hedging=enable_hedging)
        self.min_carry = min_carry

    def calculate_carry(self, data: pd.Series) -> float:
        """Calculate carry from interest rate differential"""
        r_domestic = data.get('rate_domestic', 0.05)
        r_foreign = data.get('rate_foreign', 0.01)
        return r_domestic - r_foreign

    def generate_signals(self, data: pd.DataFrame, features: pd.DataFrame = None) -> List[Signal]:
        """Generate carry arbitrage signals"""
        signals = []

        if len(data) < 20:
            return signals

        latest = data.iloc[-1]
        carry = self.calculate_carry(latest)

        # Look for significant carry opportunities
        if abs(carry) > self.min_carry:
            # Positive carry: receive domestic, pay foreign
            # Hedge with options to protect against adverse FX moves

            signal = Signal(
                timestamp=data.index[-1],
                pair=data.columns[0] if len(data.columns) > 0 else "UNKNOWN",
                direction='buy' if carry > 0 else 'sell',
                option_type='put' if carry > 0 else 'call',  # Protective option
                strength=min(abs(carry) / 0.05, 1.0),
                model='Carry',
                confidence=0.7,
                metadata={'carry': carry}
            )
            signals.append(signal)

        return signals


def create_all_model_strategies(enable_hedging: bool = False) -> List[BaseModelStrategy]:
    """
    Create all model-based strategies
    """
    strategies = []

    # GVV Arbitrage
    gvv_strategy = GVVArbitrageStrategy(
        lookback_window=20,
        mispricing_threshold=1.5,
        enable_hedging=enable_hedging
    )
    strategies.append(gvv_strategy)

    # SABR Volatility
    sabr_strategy = SABRVolatilityStrategy(
        calibration_window=60,
        vol_threshold=2.0,
        enable_hedging=enable_hedging
    )
    strategies.append(sabr_strategy)

    # Model Consensus
    consensus_strategy = ModelConsensusStrategy(
        models=['gvv', 'sabr'],
        consensus_threshold=0.6,
        enable_hedging=enable_hedging
    )
    strategies.append(consensus_strategy)

    # Carry Arbitrage
    carry_strategy = CarryArbitrageStrategy(
        min_carry=0.02,
        enable_hedging=enable_hedging
    )
    strategies.append(carry_strategy)

    return strategies


class MultiModelPortfolio:
    """
    Portfolio manager for multiple model strategies
    Handles position sizing, risk management, and rebalancing
    """

    def __init__(self, strategies: List[BaseModelStrategy],
                 initial_capital: float = 1000000,
                 max_positions: int = 10,
                 max_exposure: float = 0.3):
        self.strategies = strategies
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_exposure = max_exposure
        self.positions = {}
        self.performance_history = []

    def allocate_capital(self, signals: List[Signal]) -> Dict[Signal, float]:
        """Allocate capital across signals based on strength and confidence"""
        if not signals:
            return {}

        # Calculate total weight
        total_weight = sum([s.strength * s.confidence for s in signals])

        if total_weight == 0:
            return {}

        # Available capital (considering max exposure)
        available_capital = self.capital * self.max_exposure

        # Allocate proportionally
        allocations = {}
        for signal in signals:
            weight = (signal.strength * signal.confidence) / total_weight
            allocation = available_capital * weight / len(signals)
            allocations[signal] = min(allocation, self.capital * 0.05)  # Max 5% per signal

        return allocations

    def execute_signals(self, signals: List[Signal], data: pd.DataFrame) -> List[Dict]:
        """Execute trading signals and return trades"""
        trades = []

        # Get capital allocations
        allocations = self.allocate_capital(signals)

        for signal, size in allocations.items():
            if len(self.positions) >= self.max_positions:
                break  # Max positions reached

            trade = {
                'timestamp': signal.timestamp,
                'pair': signal.pair,
                'direction': signal.direction,
                'option_type': signal.option_type,
                'size': size,
                'model': signal.model,
                'confidence': signal.confidence,
                'entry_price': data.iloc[-1].get('spot', 100)
            }

            trades.append(trade)

            # Update positions
            position_key = f"{signal.pair}_{signal.option_type}_{signal.timestamp}"
            self.positions[position_key] = trade

        return trades

    def update_portfolio(self, data: pd.DataFrame) -> Dict:
        """Update portfolio positions and calculate P&L"""
        total_pnl = 0
        closed_positions = []

        current_price = data.iloc[-1].get('spot', 100)

        for key, position in list(self.positions.items()):
            # Simple P&L calculation (would be more complex with actual option pricing)
            price_change = (current_price - position['entry_price']) / position['entry_price']

            if position['direction'] == 'buy':
                pnl = position['size'] * price_change
            else:
                pnl = -position['size'] * price_change

            position['pnl'] = pnl
            total_pnl += pnl

            # Check exit conditions
            days_held = (data.index[-1] - position['timestamp']).days
            if days_held > 20 or abs(pnl) > position['size'] * 0.2:
                closed_positions.append(key)

        # Remove closed positions
        for key in closed_positions:
            del self.positions[key]

        # Update capital
        self.capital = self.initial_capital + total_pnl

        return {
            'capital': self.capital,
            'total_pnl': total_pnl,
            'num_positions': len(self.positions),
            'return': (self.capital - self.initial_capital) / self.initial_capital
        }