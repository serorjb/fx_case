"""
Gamma-Vanna-Volga (GVV) model for FX options
"""
import numpy as np
from typing import Dict, Tuple
from .garman_kohlhagen import GarmanKohlhagen


class GVVModel:
    """
    Gamma-Vanna-Volga model for pricing FX options with smile
    """

    def __init__(self):
        self.name = "GVV"
        self.gk = GarmanKohlhagen()

    def price_option(self, S: float, K: float, T: float, r_d: float, r_f: float,
                     atm_vol: float, rr_25: float, bf_25: float,
                     option_type: str = 'call') -> Tuple[float, Dict]:
        """
        Price option using GVV model

        Parameters:
        -----------
        S : float
            Spot rate
        K : float
            Strike
        T : float
            Time to maturity
        r_d, r_f : float
            Domestic and foreign rates
        atm_vol : float
            ATM volatility
        rr_25 : float
            25-delta risk reversal
        bf_25 : float
            25-delta butterfly
        """

        # Calculate forward
        F = S * np.exp((r_d - r_f) * T)

        # ATM strike is typically forward
        K_atm = F

        # Calculate 25-delta strikes using approximate delta-strike relationship
        vol_25_call = atm_vol + bf_25 + 0.5 * rr_25
        vol_25_put = atm_vol + bf_25 - 0.5 * rr_25

        # Approximate 25-delta strikes
        K_25_call = F * np.exp(0.5 * vol_25_call * np.sqrt(T) * norm.ppf(0.75))
        K_25_put = F * np.exp(-0.5 * vol_25_put * np.sqrt(T) * norm.ppf(0.75))

        # Calculate weights for GVV
        weights = self._calculate_gvv_weights(S, K, T, r_d, r_f, K_atm, K_25_put, K_25_call)

        # Calculate ATM price
        price_atm = self.gk.price_option(S, K, T, r_d, r_f, atm_vol, option_type)

        # Calculate vega
        greeks_atm = self.gk.calculate_greeks(S, K, T, r_d, r_f, atm_vol, option_type)
        vega = greeks_atm['vega']

        # Calculate adjustments
        if K <= K_25_put:
            implied_vol = vol_25_put
        elif K >= K_25_call:
            implied_vol = vol_25_call
        else:
            # Interpolate
            if K <= K_atm:
                alpha = (K_atm - K) / (K_atm - K_25_put)
                implied_vol = atm_vol * (1 - alpha) + vol_25_put * alpha
            else:
                alpha = (K - K_atm) / (K_25_call - K_atm)
                implied_vol = atm_vol * (1 - alpha) + vol_25_call * alpha

        # Price with implied vol
        price = self.gk.price_option(S, K, T, r_d, r_f, implied_vol, option_type)

        # Calculate Greeks with implied vol
        greeks = self.gk.calculate_greeks(S, K, T, r_d, r_f, implied_vol, option_type)

        return price, greeks

    def _calculate_gvv_weights(self, S: float, K: float, T: float, r_d: float, r_f: float,
                               K_atm: float, K_put: float, K_call: float) -> Dict[str, float]:
        """Calculate GVV weights based on market strangle"""

        # Calculate vegas for the three strikes
        vega_atm = self.gk.calculate_greeks(S, K_atm, T, r_d, r_f, 0.15, 'call')['vega']
        vega_put = self.gk.calculate_greeks(S, K_put, T, r_d, r_f, 0.15, 'put')['vega']
        vega_call = self.gk.calculate_greeks(S, K_call, T, r_d, r_f, 0.15, 'call')['vega']
        vega_k = self.gk.calculate_greeks(S, K, T, r_d, r_f, 0.15, 'call')['vega']

        # Calculate weights
        total_vega = vega_atm + vega_put + vega_call

        weights = {
            'atm': vega_atm / total_vega if total_vega > 0 else 0.33,
            'put': vega_put / total_vega if total_vega > 0 else 0.33,
            'call': vega_call / total_vega if total_vega > 0 else 0.34
        }

        return weights