# src/strategies/multi_model_strategies.py
"""
Multi-model volatility trading strategies
Each pricing model (GK, GVV, SABR) and ML generates its own signals
GARCH is used as input/feature, not as a pricing model
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from strategies.base_strategy import BaseStrategy, Signal, Position
from models.garman_kohlhagen import GarmanKohlhagen
from models.gvv_model import GVVModel
from models.sabr_model import SABRModel
from models.ml_models import LightGBMModel
from models.garch_models import GARCHForecaster
from hedging import DeltaHedger


class ModelBasedVolStrategy(BaseStrategy):
    """
    Base class for model-based volatility trading
    Each pricing model identifies mispriced options and trades them
    """

    def __init__(self, model_name: str, model_instance,
                 mispricing_threshold: float = 0.005,  # 0.5% of spot
                 min_edge: float = 0.001,  # 0.1% minimum edge
                 enable_hedging: bool = True):
        """
        Initialize model-based strategy

        Parameters:
        -----------
        model_name : str
            Name of the model (GK, GVV, SABR, ML)
        model_instance : object
            The pricing model instance
        mispricing_threshold : float
            Minimum mispricing to trade (as fraction of spot)
        min_edge : float
            Minimum expected edge to trade
        enable_hedging : bool
            Whether to delta hedge
        """
        super().__init__(f"Model_{model_name}")
        self.model = model_instance
        self.model_name = model_name
        self.mispricing_threshold = mispricing_threshold
        self.min_edge = min_edge
        self.enable_hedging = enable_hedging

        if enable_hedging:
            self.hedger = DeltaHedger(hedge_threshold=0.01)

        # Track model-specific metrics
        self.model_prices = []
        self.market_prices = []
        self.mispricings = []

    def calculate_model_price(self, data_row: pd.Series, tenor: str,
                              garch_vol: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate option price using the model
        Returns (model_price, market_price)
        """
        # Get market data
        S = data_row['spot']

        # ATM strike
        K = S

        # Time to maturity
        tenor_map = {'1W': 7 / 365, '2W': 14 / 365, '1M': 30 / 365, '3M': 90 / 365,
                     '6M': 180 / 365, '9M': 270 / 365, '1Y': 1.0}
        T = tenor_map.get(tenor, 30 / 365)

        if T <= 0:
            return None, None

        # Simplified rates (should use discount curves from data)
        r_d = 0.05  # USD rate
        r_f = 0.01  # Foreign rate

        # Get market implied volatility
        vol_col = f'atm_vol_{tenor}'
        if vol_col not in data_row.index or pd.isna(data_row[vol_col]):
            return None, None

        market_vol = data_row[vol_col]

        # Calculate model price based on model type
        try:
            if self.model_name == 'GK':
                # Standard Black-Scholes with market vol
                model_price = self.model.price_option(S, K, T, r_d, r_f, market_vol, 'call')

            elif self.model_name == 'GK_GARCH':
                # GK with GARCH forecasted volatility instead of market IV
                if garch_vol is not None and not pd.isna(garch_vol):
                    model_price = self.model.price_option(S, K, T, r_d, r_f, garch_vol, 'call')
                else:
                    return None, None

            elif self.model_name == 'GVV':
                # GVV with smile
                rr_col = f'rr_25_{tenor}'
                bf_col = f'bf_25_{tenor}'

                if rr_col in data_row.index and bf_col in data_row.index:
                    rr = data_row[rr_col] if not pd.isna(data_row[rr_col]) else 0
                    bf = data_row[bf_col] if not pd.isna(data_row[bf_col]) else 0

                    model_price = self.model.price_option(
                        S, K, T, r_d, r_f, market_vol, 'call',
                        rr_25=rr, bf_25=bf
                    )
                else:
                    # Fallback to flat vol if no smile data
                    model_price = self.model.price_option(S, K, T, r_d, r_f, market_vol, 'call')

            elif self.model_name == 'SABR':
                # SABR model - needs calibration to smile
                # For simplicity, using approximation based on ATM vol and smile
                try:
                    # Calibrate SABR (simplified - should use full smile)
                    F = S * np.exp((r_d - r_f) * T)

                    # Create synthetic strikes for calibration
                    strikes = np.array([K * 0.95, K, K * 1.05])

                    # Get vols (using butterfly and RR if available)
                    rr_col = f'rr_25_{tenor}'
                    bf_col = f'bf_25_{tenor}'

                    if rr_col in data_row.index and bf_col in data_row.index:
                        rr = data_row[rr_col] if not pd.isna(data_row[rr_col]) else 0
                        bf = data_row[bf_col] if not pd.isna(data_row[bf_col]) else 0

                        # Approximate smile
                        vol_put = market_vol + bf - 0.5 * rr
                        vol_call = market_vol + bf + 0.5 * rr
                        vols = np.array([vol_put, market_vol, vol_call])
                    else:
                        # Flat vol
                        vols = np.array([market_vol, market_vol, market_vol])

                    # Calibrate and price
                    self.model.calibrate(F, strikes, T, vols)
                    model_price = self.model.price_option(S, K, T, r_d, r_f, 'call')
                except:
                    # If calibration fails, skip
                    return None, None

            else:
                return None, None

            # Calculate market price using GK with market IV (benchmark)
            gk = GarmanKohlhagen()
            market_price = gk.price_option(S, K, T, r_d, r_f, market_vol, 'call')

            return model_price, market_price

        except Exception as e:
            return None, None

    def generate_signals(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> List[Signal]:
        """
        Generate trading signals based on model mispricing
        """
        signals = []

        for pair, pair_data in data.items():
            if date not in pair_data.index:
                continue

            current_data = pair_data.loc[date]

            # Get GARCH forecast if using GK_GARCH
            garch_vol = None
            if self.model_name == 'GK_GARCH':
                garch_col = f'garch_forecast_{pair}'
                if garch_col in pair_data.columns:
                    garch_vol = pair_data.loc[date, garch_col] if date in pair_data.index else None

            # Check each tenor
            for tenor in ['1M', '3M', '6M']:
                model_price, market_price = self.calculate_model_price(current_data, tenor, garch_vol)

                if model_price is None or market_price is None:
                    continue

                # Calculate mispricing
                S = current_data['spot']
                mispricing = (model_price - market_price) / S
                mispricing_pct = abs(mispricing)

                # Store for analysis
                self.model_prices.append(model_price)
                self.market_prices.append(market_price)
                self.mispricings.append(mispricing)

                # Generate signal if mispricing exceeds threshold
                if mispricing_pct > self.mispricing_threshold:
                    # Determine direction
                    if model_price > market_price:
                        # Model says option is worth more - buy
                        direction = 1
                        trade_type = 'buy_option'
                    else:
                        # Model says option is worth less - sell
                        direction = -1
                        trade_type = 'sell_option'

                    # Calculate expected edge
                    expected_edge = mispricing_pct

                    # Confidence based on mispricing magnitude
                    confidence = min(mispricing_pct / (self.mispricing_threshold * 3), 1.0)

                    # Additional filters based on market conditions
                    vol_col = f'atm_vol_{tenor}'
                    if vol_col in current_data:
                        current_vol = current_data[vol_col]

                        # Increase confidence if vol is at extremes
                        vol_percentile_col = f'iv_{tenor}_pct_rank'
                        if vol_percentile_col in pair_data.columns and date in pair_data.index:
                            vol_pct = pair_data.loc[date, vol_percentile_col]
                            if not pd.isna(vol_pct):
                                if vol_pct > 0.8 or vol_pct < 0.2:
                                    confidence *= 1.2

                    # Cap confidence
                    confidence = min(confidence, 1.0)

                    # Only trade if edge exceeds minimum
                    if expected_edge >= self.min_edge:
                        signal = Signal(
                            pair=pair,
                            tenor=tenor,
                            direction=direction,
                            confidence=confidence,
                            expected_edge=expected_edge,
                            strategy_name=self.name,
                            signal_type='option',
                            metadata={
                                'trade_type': trade_type,
                                'model': self.model_name,
                                'model_price': model_price,
                                'market_price': market_price,
                                'mispricing': mispricing,
                                'strike': S  # ATM
                            }
                        )

                        signals.append(signal)

        return signals

    def calculate_position_size(self, signal: Signal, capital: float,
                                current_positions: List[Position] = None) -> float:
        """
        Calculate position size based on Kelly criterion
        """
        # Base size as percentage of capital
        base_size = capital * 0.01  # 1% per trade

        # Adjust by confidence
        size = base_size * signal.confidence

        # Adjust by expected edge (simplified Kelly)
        kelly_fraction = min(signal.expected_edge / 0.05, 0.25)  # Cap at 25%
        size = size * kelly_fraction

        # Risk management adjustments
        if current_positions:
            # Count positions from same model
            model_positions = [p for p in current_positions
                               if hasattr(p, 'metadata') and p.metadata.get('model') == self.model_name]

            if len(model_positions) > 3:
                size *= 0.7  # Reduce by 30% if too many positions
            if len(model_positions) > 5:
                size *= 0.5  # Reduce by 50%

        # Cap maximum position size
        max_size = capital * 0.03  # Max 3% per position

        return min(size, max_size)


class MLSignalStrategy(BaseStrategy):
    """
    Strategy based on ML model predictions
    """

    def __init__(self, ml_model: Optional[LightGBMModel] = None,
                 signal_threshold: float = 0.6,
                 enable_hedging: bool = True):
        """
        Initialize ML-based strategy

        Parameters:
        -----------
        ml_model : LightGBMModel
            Trained ML model
        signal_threshold : float
            Minimum probability to generate signal
        """
        super().__init__("ML_Strategy")
        self.ml_model = ml_model
        self.signal_threshold = signal_threshold
        self.enable_hedging = enable_hedging

        if enable_hedging:
            self.hedger = DeltaHedger(hedge_threshold=0.01)

    def generate_signals(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp,
                         features: Dict[str, pd.DataFrame] = None) -> List[Signal]:
        """
        Generate signals from ML model predictions
        """
        signals = []

        if self.ml_model is None or features is None:
            return signals

        for pair, pair_features in features.items():
            if date not in pair_features.index:
                continue

            # Get features for current date
            current_features = pair_features.loc[date:date]

            if len(current_features) == 0:
                continue

            try:
                # Get model prediction
                predictions = self.ml_model.predict(current_features)

                # For classification model, predictions are probabilities
                if len(predictions.shape) > 1:
                    # Multi-class: [sell, hold, buy]
                    prob_sell = predictions[0, 0]
                    prob_hold = predictions[0, 1]
                    prob_buy = predictions[0, 2]

                    # Generate signal based on highest probability
                    if prob_buy > self.signal_threshold and prob_buy > prob_sell:
                        direction = 1
                        confidence = prob_buy
                    elif prob_sell > self.signal_threshold and prob_sell > prob_buy:
                        direction = -1
                        confidence = prob_sell
                    else:
                        continue  # No signal
                else:
                    # Regression model - use prediction directly
                    prediction = predictions[0]
                    if abs(prediction) < 0.001:  # Minimum threshold
                        continue

                    direction = 1 if prediction > 0 else -1
                    confidence = min(abs(prediction) * 100, 1.0)

                # Create signal for most liquid tenor
                signal = Signal(
                    pair=pair,
                    tenor='3M',  # Use 3M as default
                    direction=direction,
                    confidence=confidence,
                    expected_edge=abs(prediction) if len(predictions.shape) == 1 else 0.01,
                    strategy_name=self.name,
                    signal_type='option',
                    metadata={
                        'model': 'LightGBM',
                        'prediction': prediction if len(predictions.shape) == 1 else prob_buy - prob_sell
                    }
                )

                signals.append(signal)

            except Exception as e:
                continue

        return signals

    def calculate_position_size(self, signal: Signal, capital: float,
                                current_positions: List[Position] = None) -> float:
        """
        Calculate position size for ML signals
        """
        # Conservative sizing for ML signals
        base_size = capital * 0.005  # 0.5% per trade

        # Adjust by confidence
        size = base_size * signal.confidence

        # Cap maximum
        max_size = capital * 0.02  # Max 2% per position

        return min(size, max_size)


def create_all_model_strategies(enable_hedging: bool = True) -> List[BaseStrategy]:
    """
    Create all model-based strategies
    """
    strategies = []

    # 1. Garman-Kohlhagen with market IV
    gk_model = GarmanKohlhagen()
    gk_strategy = ModelBasedVolStrategy(
        model_name="GK",
        model_instance=gk_model,
        mispricing_threshold=0.005,  # 0.5% threshold
        min_edge=0.001,
        enable_hedging=enable_hedging
    )
    strategies.append(gk_strategy)

    # 2. Garman-Kohlhagen with GARCH vol
    gk_garch_strategy = ModelBasedVolStrategy(
        model_name="GK_GARCH",
        model_instance=gk_model,  # Same model, different vol input
        mispricing_threshold=0.007,  # Slightly higher threshold
        min_edge=0.002,
        enable_hedging=enable_hedging
    )
    strategies.append(gk_garch_strategy)

    # 3. GVV Model (with smile)
    gvv_model = GVVModel()
    gvv_strategy = ModelBasedVolStrategy(
        model_name="GVV",
        model_instance=gvv_model,
        mispricing_threshold=0.004,  # Lower threshold as it includes smile
        min_edge=0.001,
        enable_hedging=enable_hedging
    )
    strategies.append(gvv_strategy)

    # 4. SABR Model
    sabr_model = SABRModel()
    sabr_strategy = ModelBasedVolStrategy(
        model_name="SABR",
        model_instance=sabr_model,
        mispricing_threshold=0.006,
        min_edge=0.0015,
        enable_hedging=enable_hedging
    )
    strategies.append(sabr_strategy)

    # 5. ML Strategy (if model available)
    # This would be added after training

    return strategies