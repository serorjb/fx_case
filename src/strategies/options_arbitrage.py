# src/strategies/options_arbitrage.py
"""
Options Arbitrage Strategy Based on Model Mispricing
This is the CORRECT implementation based on the assignment requirements
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')


@dataclass
class OptionTrade:
    """Represents an option trade opportunity"""
    timestamp: pd.Timestamp
    pair: str
    tenor: str
    strike: float
    option_type: str  # 'call' or 'put'
    direction: int  # 1: buy, -1: sell
    model_price: float
    market_price: float
    mispricing: float  # model_price - market_price
    mispricing_pct: float  # (model_price - market_price) / market_price
    confidence: float
    model_used: str  # 'GVV', 'SABR', etc.
    delta: float  # For hedging
    vega: float  # For risk management
    metadata: dict = None


class OptionsArbitrageStrategy:
    """
    Options Arbitrage Strategy Based on Pricing Model Mispricings

    This strategy:
    1. Uses market quotes (ATM vol, RR, BF) to construct volatility surface
    2. Prices options using GVV/SABR models
    3. Compares model prices with market-implied prices
    4. Trades mispricings and hedges delta
    """

    def __init__(self,
                 pricing_model: str = 'GVV',
                 mispricing_threshold: float = 0.02,  # 2% mispricing to trade
                 max_positions: int = 20,
                 enable_hedging: bool = True,
                 transaction_cost: float = 0.0005):  # 5 bps
        """
        Parameters:
        -----------
        pricing_model : str
            Model to use for pricing ('GVV', 'SABR', 'BS')
        mispricing_threshold : float
            Minimum mispricing (as %) to generate signal
        max_positions : int
            Maximum number of concurrent positions
        enable_hedging : bool
            Whether to delta hedge positions
        transaction_cost : float
            Transaction cost as fraction of notional
        """
        self.name = f"OptionsArbitrage_{pricing_model}"
        self.pricing_model = pricing_model
        self.mispricing_threshold = mispricing_threshold
        self.max_positions = max_positions
        self.enable_hedging = enable_hedging
        self.transaction_cost = transaction_cost

        # Initialize pricing models
        self._initialize_models()

        # Track positions
        self.positions = []
        self.hedge_positions = []
        self.closed_trades = []

    def _initialize_models(self):
        """Initialize pricing models"""
        from src.models.gk_model import GarmanKohlhagen
        from src.models.gvv_model import GVVModel
        from src.models.sabr_model import SABRModel

        self.gk_model = GarmanKohlhagen()
        self.gvv_model = GVVModel()
        self.sabr_model = SABRModel()

    def construct_volatility_surface(self, data_row: pd.Series, tenor: str) -> Dict:
        """
        Construct volatility surface from market quotes

        Returns dictionary with:
        - strikes: list of strikes
        - vols: list of implied vols
        - spot: current spot
        """
        spot = data_row['spot']

        # Get market quotes
        atm_vol = data_row.get(f'atm_vol_{tenor}', 0.10)
        rr_25 = data_row.get(f'rr_25_{tenor}', 0)
        bf_25 = data_row.get(f'bf_25_{tenor}', 0)
        rr_10 = data_row.get(f'rr_10_{tenor}', 0)
        bf_10 = data_row.get(f'bf_10_{tenor}', 0)

        # Calculate implied vols for different strikes
        # 25-delta quotes
        vol_25_call = atm_vol + bf_25 + rr_25 / 2
        vol_25_put = atm_vol + bf_25 - rr_25 / 2

        # 10-delta quotes (if available)
        if not pd.isna(rr_10) and not pd.isna(bf_10):
            vol_10_call = atm_vol + bf_10 + rr_10 / 2
            vol_10_put = atm_vol + bf_10 - rr_10 / 2
        else:
            # Extrapolate
            vol_10_call = vol_25_call + (vol_25_call - atm_vol) * 0.5
            vol_10_put = vol_25_put + (vol_25_put - atm_vol) * 0.5

        # Calculate strikes from deltas (simplified)
        # In practice, would use proper delta-to-strike conversion
        T = self._tenor_to_years(tenor)

        # ATM strike
        K_atm = spot  # Simplified ATM

        # 25-delta strikes (approximate)
        K_25_call = spot * np.exp(0.5 * vol_25_call * np.sqrt(T))
        K_25_put = spot * np.exp(-0.5 * vol_25_put * np.sqrt(T))

        # 10-delta strikes (approximate)
        K_10_call = spot * np.exp(1.0 * vol_10_call * np.sqrt(T))
        K_10_put = spot * np.exp(-1.0 * vol_10_put * np.sqrt(T))

        return {
            'spot': spot,
            'strikes': [K_10_put, K_25_put, K_atm, K_25_call, K_10_call],
            'vols': [vol_10_put, vol_25_put, atm_vol, vol_25_call, vol_10_call],
            'atm_vol': atm_vol,
            'rr_25': rr_25,
            'bf_25': bf_25
        }

    def calculate_market_price(self, surface: Dict, strike: float, tenor: str,
                               option_type: str = 'call') -> float:
        """
        Calculate market-implied option price from volatility surface
        Uses linear interpolation of implied vol
        """
        # Interpolate volatility for the given strike
        strikes = np.array(surface['strikes'])
        vols = np.array(surface['vols'])

        # Linear interpolation
        if strike <= strikes[0]:
            implied_vol = vols[0]
        elif strike >= strikes[-1]:
            implied_vol = vols[-1]
        else:
            implied_vol = np.interp(strike, strikes, vols)

        # Price using Black-Scholes with interpolated vol
        spot = surface['spot']
        T = self._tenor_to_years(tenor)
        r_d = 0.05  # Simplified rates
        r_f = 0.01

        market_price = self.gk_model.price_option(
            spot, strike, T, r_d, r_f, implied_vol, option_type
        )

        return market_price

    def calculate_model_price(self, surface: Dict, strike: float, tenor: str,
                              option_type: str = 'call') -> Tuple[float, float, float]:
        """
        Calculate theoretical price using specified model
        Returns: (price, delta, vega)
        """
        spot = surface['spot']
        T = self._tenor_to_years(tenor)
        r_d = 0.05  # Simplified rates
        r_f = 0.01

        if self.pricing_model == 'GVV':
            # Use GVV model with market quotes
            price = self.gvv_model.price_option(
                spot, strike, T, r_d, r_f,
                surface['atm_vol'], option_type,
                rr_25=surface['rr_25'],
                bf_25=surface['bf_25']
            )

            # Calculate Greeks
            greeks = self.gvv_model.calculate_greeks(
                spot, strike, T, r_d, r_f,
                surface['atm_vol'], option_type,
                rr_25=surface['rr_25'],
                bf_25=surface['bf_25']
            )

        elif self.pricing_model == 'SABR':
            # Calibrate SABR to the surface
            strikes = np.array(surface['strikes'])
            vols = np.array(surface['vols'])

            try:
                self.sabr_model.calibrate(spot, strikes, T, vols)
                price = self.sabr_model.price_option(
                    spot, strike, T, r_d, r_f, surface['atm_vol'], option_type
                )
                greeks = self.sabr_model.calculate_greeks(
                    spot, strike, T, r_d, r_f, surface['atm_vol'], option_type
                )
            except:
                # Fallback to Black-Scholes
                price = self.gk_model.price_option(
                    spot, strike, T, r_d, r_f, surface['atm_vol'], option_type
                )
                greeks = self.gk_model.calculate_greeks(
                    spot, strike, T, r_d, r_f, surface['atm_vol'], option_type
                )

        else:  # Black-Scholes
            price = self.gk_model.price_option(
                spot, strike, T, r_d, r_f, surface['atm_vol'], option_type
            )
            greeks = self.gk_model.calculate_greeks(
                spot, strike, T, r_d, r_f, surface['atm_vol'], option_type
            )

        delta = greeks.get('delta', 0)
        vega = greeks.get('vega', 0)

        return price, delta, vega

    def find_arbitrage_opportunities(self, data: pd.DataFrame,
                                     date: pd.Timestamp) -> List[OptionTrade]:
        """
        Find mispriced options by comparing model vs market prices
        """
        trades = []

        # Get current data
        if date not in data.index:
            return trades

        current_data = data.loc[date]

        # Check different tenors
        for tenor in ['1M', '3M', '6M']:
            # Check if we have required data
            if f'atm_vol_{tenor}' not in data.columns:
                continue

            if pd.isna(current_data[f'atm_vol_{tenor}']):
                continue

            # Construct volatility surface
            surface = self.construct_volatility_surface(current_data, tenor)

            # Test different strikes around ATM
            spot = surface['spot']
            test_strikes = [
                spot * 0.95,  # 5% OTM put
                spot * 0.98,  # 2% OTM put
                spot,  # ATM
                spot * 1.02,  # 2% OTM call
                spot * 1.05  # 5% OTM call
            ]

            for strike in test_strikes:
                for option_type in ['call', 'put']:
                    # Calculate prices
                    market_price = self.calculate_market_price(
                        surface, strike, tenor, option_type
                    )

                    model_price, delta, vega = self.calculate_model_price(
                        surface, strike, tenor, option_type
                    )

                    # Calculate mispricing
                    mispricing = model_price - market_price
                    mispricing_pct = mispricing / market_price if market_price > 0 else 0

                    # Check if mispricing exceeds threshold
                    if abs(mispricing_pct) > self.mispricing_threshold:
                        # Determine trade direction
                        if mispricing > 0:
                            # Model price > Market price: Market undervalued, BUY
                            direction = 1
                        else:
                            # Model price < Market price: Market overvalued, SELL
                            direction = -1

                        # Calculate confidence based on mispricing magnitude
                        confidence = min(abs(mispricing_pct) / 0.05, 1.0)

                        trade = OptionTrade(
                            timestamp=date,
                            pair='FX',  # Would be specific pair in practice
                            tenor=tenor,
                            strike=strike,
                            option_type=option_type,
                            direction=direction,
                            model_price=model_price,
                            market_price=market_price,
                            mispricing=mispricing,
                            mispricing_pct=mispricing_pct,
                            confidence=confidence,
                            model_used=self.pricing_model,
                            delta=delta,
                            vega=vega,
                            metadata={
                                'spot': spot,
                                'atm_vol': surface['atm_vol'],
                                'moneyness': strike / spot
                            }
                        )

                        trades.append(trade)

        return trades

    def generate_signals(self, data: Dict[str, pd.DataFrame],
                         date: pd.Timestamp) -> List:
        """
        Generate trading signals for all currency pairs
        Compatible with backtesting engine
        """
        all_signals = []

        for pair, pair_data in data.items():
            # Find arbitrage opportunities
            trades = self.find_arbitrage_opportunities(pair_data, date)

            # Convert to signals format expected by backtester
            for trade in trades:
                # Only take highest confidence trades
                if trade.confidence > 0.6 and len(all_signals) < self.max_positions:
                    signal = type('Signal', (), {
                        'timestamp': trade.timestamp,
                        'pair': pair,
                        'tenor': trade.tenor,
                        'direction': trade.direction,
                        'strength': trade.confidence,
                        'confidence': trade.confidence,
                        'expected_edge': abs(trade.mispricing_pct),
                        'strategy_name': self.name,
                        'signal_type': 'option',
                        'option_type': trade.option_type,
                        'metadata': {
                            'strike': trade.strike,
                            'model_price': trade.model_price,
                            'market_price': trade.market_price,
                            'mispricing_pct': trade.mispricing_pct,
                            'delta': trade.delta,
                            'vega': trade.vega
                        }
                    })()

                    all_signals.append(signal)

                    # Debug output
                    if len(all_signals) <= 3:
                        print(f"  Signal: {pair} {trade.tenor} {trade.option_type} "
                              f"K={trade.strike:.2f} | "
                              f"Model: {trade.model_price:.4f} Market: {trade.market_price:.4f} | "
                              f"Mispricing: {trade.mispricing_pct:.2%}")

        return all_signals

    def calculate_position_size(self, signal, capital: float,
                                current_positions: List = None) -> float:
        """
        Calculate position size based on Kelly criterion and risk limits
        """
        # Base size as percentage of capital
        base_size = capital * 0.01  # 1% base position

        # Adjust by confidence and expected edge
        kelly_fraction = signal.confidence * signal.expected_edge
        kelly_fraction = min(kelly_fraction, 0.25)  # Cap at 25%

        size = base_size * (1 + kelly_fraction)

        # Adjust for existing positions
        if current_positions:
            num_positions = len(current_positions)
            if num_positions > 10:
                size *= 0.5  # Reduce size when many positions

        # Cap maximum size
        max_size = capital * 0.03  # 3% max per position

        # Account for transaction costs
        size = size * (1 - self.transaction_cost)

        return min(size, max_size)

    def execute_hedge(self, position, data: pd.DataFrame, date: pd.Timestamp) -> Dict:
        """
        Execute delta hedge for a position
        """
        if not self.enable_hedging:
            return {}

        # Get position delta from metadata
        delta = position.metadata.get('delta', 0)
        spot = data.loc[date, 'spot'] if date in data.index else 100

        # Calculate hedge size
        hedge_size = -delta * position.size / spot  # Opposite sign to neutralize

        hedge = {
            'type': 'spot',
            'size': hedge_size,
            'entry_price': spot,
            'entry_date': date
        }

        self.hedge_positions.append(hedge)

        return hedge

    def should_exit_position(self, position, current_data: pd.Series,
                             current_date: pd.Timestamp) -> bool:
        """
        Determine if position should be closed
        """
        # Time-based exit
        entry_date = position.entry_date if hasattr(position, 'entry_date') else current_date
        days_held = (current_date - entry_date).days

        tenor_days = {'1M': 30, '3M': 90, '6M': 180}.get(position.tenor, 30)

        # Exit at 50% of tenor
        if days_held > tenor_days * 0.5:
            return True

        # Exit if mispricing has reversed
        if hasattr(position, 'metadata'):
            original_mispricing = position.metadata.get('mispricing_pct', 0)

            # Recalculate current mispricing (simplified)
            # In practice, would recalculate properly
            if abs(original_mispricing) < 0.01:  # Mispricing has converged
                return True

        return False

    def _tenor_to_years(self, tenor: str) -> float:
        """Convert tenor string to years"""
        mapping = {
            '1W': 1 / 52, '2W': 2 / 52, '3W': 3 / 52,
            '1M': 1 / 12, '2M': 2 / 12, '3M': 3 / 12,
            '4M': 4 / 12, '6M': 6 / 12, '9M': 9 / 12,
            '1Y': 1.0
        }
        return mapping.get(tenor, 1 / 12)

    def get_performance_summary(self) -> Dict:
        """Get strategy performance summary"""
        return {
            'total_trades': len(self.closed_trades),
            'total_hedges': len(self.hedge_positions),
            'model_used': self.pricing_model,
            'mispricing_threshold': self.mispricing_threshold
        }


# Backward compatibility
VolatilityArbitrageStrategy = OptionsArbitrageStrategy