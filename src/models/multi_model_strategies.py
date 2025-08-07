"""
Multi-model trading strategies for FX options - FIXED VERSION
Implements GVV, SABR, ML-based, and consensus strategies with improved signal generation
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

    def generate_signals(self, data: Dict[str, pd.DataFrame], features: Dict[str, pd.DataFrame] = None,
                         current_date: pd.Timestamp = None) -> List[Signal]:
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


class GVVArbitrageStrategy(BaseModelStrategy):
    """
    Garman-Kohlhagen-Vanna-Volga arbitrage strategy
    Trades mispricings between GVV model and market prices
    """

    def __init__(self, lookback_window: int = 20, mispricing_threshold: float = 1.5,
                 enable_hedging: bool = False):
        super().__init__(name="GVV_Arbitrage", enable_hedging=enable_hedging)
        self.lookback_window = lookback_window
        self.mispricing_threshold = mispricing_threshold

    def calculate_gvv_price(self, spot: float, atm_vol: float, rr_25: float,
                            bf_25: float, r_d: float = 0.05, r_f: float = 0.01) -> Tuple[float, float]:
        """Calculate GVV model price for options"""
        from scipy.stats import norm

        T = 1 / 12  # 1 month option
        K_atm = spot * np.exp((r_d - r_f) * T)

        # Volatility smile adjustment
        call_vol_25d = atm_vol + bf_25 + rr_25 / 2
        put_vol_25d = atm_vol + bf_25 - rr_25 / 2

        # Ensure positive volatilities
        call_vol_25d = max(call_vol_25d, 0.001)
        put_vol_25d = max(put_vol_25d, 0.001)

        # Black-Scholes with smile-adjusted vol
        d1_call = (np.log(spot / K_atm) + (r_d - r_f + 0.5 * call_vol_25d ** 2) * T) / (call_vol_25d * np.sqrt(T))
        d2_call = d1_call - call_vol_25d * np.sqrt(T)

        call_price = spot * np.exp(-r_f * T) * norm.cdf(d1_call) - K_atm * np.exp(-r_d * T) * norm.cdf(d2_call)

        d1_put = (np.log(spot / K_atm) + (r_d - r_f + 0.5 * put_vol_25d ** 2) * T) / (put_vol_25d * np.sqrt(T))
        d2_put = d1_put - put_vol_25d * np.sqrt(T)

        put_price = K_atm * np.exp(-r_d * T) * norm.cdf(-d2_put) - spot * np.exp(-r_f * T) * norm.cdf(-d1_put)

        return max(call_price, 0.0001), max(put_price, 0.0001)

    def generate_signals(self, data: Dict[str, pd.DataFrame], features: Dict[str, pd.DataFrame] = None,
                         current_date: pd.Timestamp = None) -> List[Signal]:
        """Generate GVV arbitrage signals"""
        signals = []

        for pair, pair_data in data.items():
            if current_date:
                # Get data up to current date
                pair_data = pair_data[:current_date]

            if len(pair_data) < self.lookback_window:
                continue

            # Get latest data
            latest = pair_data.iloc[-1]
            historical = pair_data.iloc[-self.lookback_window:]

            # Extract required fields with defaults
            spot = latest.get('spot', 100)
            atm_vol = latest.get('atm_vol_1M', 0.1)
            rr_25 = latest.get('rr_25_1M', 0)
            bf_25 = latest.get('bf_25_1M', 0)

            # Skip if missing critical data
            if pd.isna(spot) or pd.isna(atm_vol):
                continue

            # Replace NaN with defaults
            rr_25 = 0 if pd.isna(rr_25) else rr_25
            bf_25 = 0 if pd.isna(bf_25) else bf_25

            # Calculate GVV prices
            try:
                gvv_call, gvv_put = self.calculate_gvv_price(spot, atm_vol, rr_25, bf_25)
            except Exception as e:
                continue

            # Calculate market implied prices (simplified)
            market_call = spot * atm_vol * 0.4 * np.sqrt(1 / 12)
            market_put = market_call * (1 + rr_25 * 10)  # Adjust for risk reversal

            # Ensure positive prices
            market_call = max(market_call, 0.0001)
            market_put = max(market_put, 0.0001)

            # Calculate mispricings
            call_mispricing = (gvv_call - market_call) / market_call
            put_mispricing = (gvv_put - market_put) / market_put

            # Calculate historical mispricings for z-score
            historical_call_mispricing = []
            historical_put_mispricing = []

            for _, row in historical.iterrows():
                try:
                    h_spot = row.get('spot', 100)
                    h_atm_vol = row.get('atm_vol_1M', 0.1)
                    h_rr_25 = row.get('rr_25_1M', 0)
                    h_bf_25 = row.get('bf_25_1M', 0)

                    if pd.isna(h_spot) or pd.isna(h_atm_vol):
                        continue

                    h_rr_25 = 0 if pd.isna(h_rr_25) else h_rr_25
                    h_bf_25 = 0 if pd.isna(h_bf_25) else h_bf_25

                    h_gvv_call, h_gvv_put = self.calculate_gvv_price(h_spot, h_atm_vol, h_rr_25, h_bf_25)
                    h_market_call = h_spot * h_atm_vol * 0.4 * np.sqrt(1 / 12)
                    h_market_put = h_market_call * (1 + h_rr_25 * 10)

                    h_market_call = max(h_market_call, 0.0001)
                    h_market_put = max(h_market_put, 0.0001)

                    historical_call_mispricing.append((h_gvv_call - h_market_call) / h_market_call)
                    historical_put_mispricing.append((h_gvv_put - h_market_put) / h_market_put)
                except:
                    continue

            # Generate signals based on z-scores
            if len(historical_call_mispricing) > 5:  # Need at least 5 data points
                mean_call = np.mean(historical_call_mispricing)
                std_call = np.std(historical_call_mispricing)

                if std_call > 0.001:  # Avoid division by zero
                    z_call = (call_mispricing - mean_call) / std_call

                    if abs(z_call) > self.mispricing_threshold:
                        signal = Signal(
                            timestamp=pair_data.index[-1],
                            pair=pair,
                            direction='sell' if z_call > 0 else 'buy',  # Sell if overpriced
                            option_type='call',
                            strength=min(abs(z_call) / 3.0, 1.0),
                            model='GVV',
                            confidence=0.6 + 0.1 * min(abs(z_call) - self.mispricing_threshold, 2.0),
                            metadata={'mispricing': call_mispricing, 'z_score': z_call}
                        )
                        signals.append(signal)

            if len(historical_put_mispricing) > 5:
                mean_put = np.mean(historical_put_mispricing)
                std_put = np.std(historical_put_mispricing)

                if std_put > 0.001:
                    z_put = (put_mispricing - mean_put) / std_put

                    if abs(z_put) > self.mispricing_threshold:
                        signal = Signal(
                            timestamp=pair_data.index[-1],
                            pair=pair,
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

    def __init__(self, calibration_window: int = 60, vol_threshold: float = 1.5,
                 enable_hedging: bool = False):
        super().__init__(name="SABR_Volatility", enable_hedging=enable_hedging)
        self.calibration_window = calibration_window
        self.vol_threshold = vol_threshold  # Lowered threshold
        self.sabr_params = {}

    def calibrate_sabr(self, data: pd.DataFrame) -> Dict:
        """Calibrate SABR model parameters"""
        atm_vols = data['atm_vol_1M'].dropna()
        if len(atm_vols) < 20:
            return None

        # Estimate SABR parameters
        alpha = atm_vols.mean()  # Initial vol
        beta = 0.5  # CEV parameter (0.5 for FX typically)

        # Calculate correlation between vol and spot
        if 'spot' in data.columns:
            vol_spot_corr = data[['atm_vol_1M', 'spot']].dropna().corr()
            if len(vol_spot_corr) == 2:
                rho = vol_spot_corr.iloc[0, 1]
            else:
                rho = -0.3
        else:
            rho = -0.3

        nu = atm_vols.std() * np.sqrt(252)  # Vol of vol

        return {
            'alpha': alpha,
            'beta': beta,
            'rho': np.clip(rho, -0.99, 0.99),  # Ensure valid range
            'nu': max(nu, 0.01)  # Ensure positive
        }

    def calculate_sabr_vol(self, params: Dict, K: float, F: float, T: float) -> float:
        """Calculate SABR implied volatility"""
        if params is None:
            return 0.1  # Default vol

        alpha = params['alpha']
        beta = params['beta']
        rho = params['rho']
        nu = params['nu']

        # Prevent numerical issues
        if abs(K - F) < 0.001:
            # ATM case
            vol = alpha * (1 + (nu ** 2 / 24) * T)
        else:
            # General case (simplified)
            try:
                z = (nu / alpha) * (F * K) ** ((1 - beta) / 2) * np.log(F / K)
                x_arg = (np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho)

                # Ensure positive argument for log
                if x_arg > 0:
                    x = np.log(x_arg)
                    vol = alpha * (z / x if abs(x) > 0.001 else 1.0)
                else:
                    vol = alpha  # Fallback to base vol
            except:
                vol = alpha  # Fallback to base vol

        return max(vol, 0.001)  # Ensure positive

    def generate_signals(self, data: Dict[str, pd.DataFrame], features: Dict[str, pd.DataFrame] = None,
                         current_date: pd.Timestamp = None) -> List[Signal]:
        """Generate SABR-based trading signals"""
        signals = []

        for pair, pair_data in data.items():
            if current_date:
                pair_data = pair_data[:current_date]

            if len(pair_data) < self.calibration_window:
                continue

            # Calibrate SABR model
            calibration_data = pair_data.iloc[-self.calibration_window:]
            sabr_params = self.calibrate_sabr(calibration_data)

            if sabr_params is None:
                continue

            latest = pair_data.iloc[-1]
            spot = latest.get('spot', 100)

            if pd.isna(spot):
                continue

            # Calculate SABR vol for ATM
            T = 1 / 12  # 1 month
            sabr_atm_vol = self.calculate_sabr_vol(sabr_params, spot, spot, T)

            # Compare with market vol
            market_atm_vol = latest.get('atm_vol_1M', 0.1)
            if pd.isna(market_atm_vol):
                continue

            # Calculate vol spread
            vol_spread = sabr_atm_vol - market_atm_vol

            # Calculate rolling mean and std of spreads
            # Use recent history for more responsive signals
            recent_window = min(20, len(pair_data) - self.calibration_window)
            if recent_window < 5:
                continue

            # Simple moving statistics
            historical_spreads = []
            for i in range(recent_window):
                idx = len(pair_data) - recent_window + i - 1
                if idx >= self.calibration_window:
                    hist_data = pair_data.iloc[idx - self.calibration_window:idx]
                    hist_params = self.calibrate_sabr(hist_data)
                    if hist_params:
                        hist_spot = pair_data.iloc[idx]['spot']
                        if not pd.isna(hist_spot):
                            hist_sabr_vol = self.calculate_sabr_vol(hist_params, hist_spot, hist_spot, T)
                            hist_market_vol = pair_data.iloc[idx].get('atm_vol_1M', 0.1)
                            if not pd.isna(hist_market_vol):
                                historical_spreads.append(hist_sabr_vol - hist_market_vol)

            if len(historical_spreads) >= 5:
                mean_spread = np.mean(historical_spreads)
                std_spread = np.std(historical_spreads)

                if std_spread > 0.001:
                    z_score = (vol_spread - mean_spread) / std_spread

                    # Generate signal if z-score exceeds threshold
                    if abs(z_score) > self.vol_threshold:
                        signal = Signal(
                            timestamp=pair_data.index[-1],
                            pair=pair,
                            direction='buy' if z_score < 0 else 'sell',  # Buy vol if underpriced
                            option_type='straddle',  # Trade volatility via straddle
                            strength=min(abs(z_score) / 2.5, 1.0),
                            model='SABR',
                            confidence=0.5 + 0.15 * min(abs(z_score) - self.vol_threshold, 2.0),
                            metadata={
                                'vol_spread': vol_spread,
                                'z_score': z_score,
                                'sabr_vol': sabr_atm_vol,
                                'market_vol': market_atm_vol
                            }
                        )
                        signals.append(signal)

        return signals


class SimpleVolatilityStrategy(BaseModelStrategy):
    """
    Simple volatility mean reversion strategy
    Trades when implied volatility deviates from historical average
    """

    def __init__(self, lookback_window: int = 30, zscore_threshold: float = 1.5,
                 enable_hedging: bool = False):
        super().__init__(name="Simple_Vol", enable_hedging=enable_hedging)
        self.lookback_window = lookback_window
        self.zscore_threshold = zscore_threshold

    def generate_signals(self, data: Dict[str, pd.DataFrame], features: Dict[str, pd.DataFrame] = None,
                         current_date: pd.Timestamp = None) -> List[Signal]:
        """Generate simple volatility trading signals"""
        signals = []

        for pair, pair_data in data.items():
            if current_date:
                pair_data = pair_data[:current_date]

            if len(pair_data) < self.lookback_window:
                continue

            # Get ATM volatilities
            atm_vols = pair_data['atm_vol_1M'].dropna()

            if len(atm_vols) < self.lookback_window:
                continue

            # Calculate z-score
            recent_vols = atm_vols.iloc[-self.lookback_window:]
            current_vol = atm_vols.iloc[-1]

            mean_vol = recent_vols.mean()
            std_vol = recent_vols.std()

            if std_vol > 0.001:
                z_score = (current_vol - mean_vol) / std_vol

                if abs(z_score) > self.zscore_threshold:
                    signal = Signal(
                        timestamp=pair_data.index[-1],
                        pair=pair,
                        direction='sell' if z_score > 0 else 'buy',  # Sell high vol, buy low vol
                        option_type='straddle',
                        strength=min(abs(z_score) / 2.5, 1.0),
                        model='SimpleVol',
                        confidence=0.6,
                        metadata={'z_score': z_score, 'current_vol': current_vol, 'mean_vol': mean_vol}
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

    def generate_signals(self, data: Dict[str, pd.DataFrame], features: Dict[str, pd.DataFrame] = None,
                         current_date: pd.Timestamp = None) -> List[Signal]:
        """Generate ML-based trading signals"""
        signals = []

        if self.ml_model is None or features is None:
            return signals

        # ML model expects features for a specific pair
        # This would be called with pair-specific data in practice
        for pair, pair_features in features.items():
            if pair not in data:
                continue

            if current_date:
                pair_features = pair_features[:current_date]

            if len(pair_features) == 0:
                continue

            try:
                # Get latest features
                latest_features = pair_features.iloc[-1:].values.reshape(1, -1)

                # Get model prediction
                if hasattr(self.ml_model, 'predict_proba'):
                    # Classification model
                    proba = self.ml_model.predict_proba(latest_features)[0]

                    if len(proba) >= 2:
                        up_prob = proba[1]
                        down_prob = proba[0]

                        if up_prob > self.signal_threshold:
                            signal = Signal(
                                timestamp=pair_features.index[-1],
                                pair=pair,
                                direction='buy',
                                option_type='call',
                                strength=(up_prob - 0.5) * 2,
                                model='ML',
                                confidence=up_prob,
                                metadata={'probability': up_prob}
                            )
                            signals.append(signal)

                        elif down_prob > self.signal_threshold:
                            signal = Signal(
                                timestamp=pair_features.index[-1],
                                pair=pair,
                                direction='buy',
                                option_type='put',
                                strength=(down_prob - 0.5) * 2,
                                model='ML',
                                confidence=down_prob,
                                metadata={'probability': down_prob}
                            )
                            signals.append(signal)

            except Exception as e:
                continue

        return signals


class ModelConsensusStrategy(BaseModelStrategy):
    """
    Consensus strategy that combines signals from multiple models
    Only trades when multiple models agree
    """

    def __init__(self, models: List[str] = None, consensus_threshold: float = 0.5,
                 enable_hedging: bool = False):
        super().__init__(name="Model_Consensus", enable_hedging=enable_hedging)
        self.models = models or ['gvv', 'sabr', 'simple_vol']
        self.consensus_threshold = consensus_threshold
        self.sub_strategies = {}

        # Initialize sub-strategies
        if 'gvv' in self.models:
            self.sub_strategies['gvv'] = GVVArbitrageStrategy(
                lookback_window=20,
                mispricing_threshold=1.5,
                enable_hedging=False
            )
        if 'sabr' in self.models:
            self.sub_strategies['sabr'] = SABRVolatilityStrategy(
                calibration_window=60,
                vol_threshold=1.5,
                enable_hedging=False
            )
        if 'simple_vol' in self.models:
            self.sub_strategies['simple_vol'] = SimpleVolatilityStrategy(
                lookback_window=30,
                zscore_threshold=1.5,
                enable_hedging=False
            )

    def generate_signals(self, data: Dict[str, pd.DataFrame], features: Dict[str, pd.DataFrame] = None,
                         current_date: pd.Timestamp = None) -> List[Signal]:
        """Generate consensus signals from multiple models"""
        model_signals = {}

        # Collect signals from all sub-strategies
        for model_name, strategy in self.sub_strategies.items():
            try:
                signals = strategy.generate_signals(data, features, current_date)
                model_signals[model_name] = signals
            except:
                model_signals[model_name] = []

        # Find consensus signals
        consensus_signals = []

        # Group signals by pair and option type
        signal_groups = {}
        for model_name, signals in model_signals.items():
            for signal in signals:
                key = (signal.pair, signal.option_type, signal.direction)
                if key not in signal_groups:
                    signal_groups[key] = []
                signal_groups[key].append((model_name, signal))

        # Check for consensus
        min_models = max(2, int(len(self.sub_strategies) * self.consensus_threshold))

        for (pair, option_type, direction), group in signal_groups.items():
            if len(group) >= min_models:
                # Consensus achieved
                avg_strength = np.mean([s.strength for _, s in group])
                avg_confidence = np.mean([s.confidence for _, s in group])

                consensus_signal = Signal(
                    timestamp=group[0][1].timestamp,
                    pair=pair,
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


def create_all_model_strategies(enable_hedging: bool = False) -> List[BaseModelStrategy]:
    """
    Create all model-based strategies with appropriate parameters
    """
    strategies = []

    # GVV Arbitrage - more sensitive
    gvv_strategy = GVVArbitrageStrategy(
        lookback_window=20,
        mispricing_threshold=1.2,  # Lower threshold for more signals
        enable_hedging=enable_hedging
    )
    strategies.append(gvv_strategy)

    # SABR Volatility - more sensitive
    sabr_strategy = SABRVolatilityStrategy(
        calibration_window=40,  # Shorter window for faster adaptation
        vol_threshold=1.2,  # Lower threshold
        enable_hedging=enable_hedging
    )
    strategies.append(sabr_strategy)

    # Simple Volatility - baseline strategy
    simple_vol_strategy = SimpleVolatilityStrategy(
        lookback_window=20,
        zscore_threshold=1.2,
        enable_hedging=enable_hedging
    )
    strategies.append(simple_vol_strategy)

    # Model Consensus
    consensus_strategy = ModelConsensusStrategy(
        models=['gvv', 'sabr', 'simple_vol'],
        consensus_threshold=0.5,  # Only need 2 out of 3 models
        enable_hedging=enable_hedging
    )
    strategies.append(consensus_strategy)

    return strategies