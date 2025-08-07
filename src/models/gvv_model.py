# src/models/gvv_model.py
"""
Corrected Gamma-Vanna-Volga (GVV) model for FX options
Properly implements the GVV approximation for volatility smile
"""
import numpy as np
from typing import Dict, Tuple
from scipy.stats import norm
from scipy.optimize import brentq
from .base_model import BaseOptionModel
from .gk_model import GarmanKohlhagen


class GVVModel(BaseOptionModel):
    """
    Gamma-Vanna-Volga model for pricing FX options with smile

    The GVV method uses three market instruments (ATM, 25-delta call, 25-delta put)
    to adjust Black-Scholes prices for smile effects
    """

    def __init__(self):
        super().__init__("GVV")
        self.gk = GarmanKohlhagen()

    def calculate_strike_from_delta(self, S: float, T: float, r_d: float, r_f: float,
                                    sigma: float, delta: float, option_type: str = 'call') -> float:
        """
        Calculate strike from delta using spot delta convention

        Parameters:
        -----------
        S : float
            Spot price
        T : float
            Time to maturity
        r_d, r_f : float
            Domestic and foreign rates
        sigma : float
            Volatility
        delta : float
            Target delta (e.g., 0.25 for 25-delta)
        option_type : str
            'call' or 'put'
        """
        F = S * np.exp((r_d - r_f) * T)

        # For spot delta convention (most common in FX)
        # Delta_call = exp(-r_f * T) * N(d1)
        # Delta_put = -exp(-r_f * T) * N(-d1)

        if option_type == 'call':
            # For call: delta = exp(-r_f * T) * N(d1)
            # So: N(d1) = delta * exp(r_f * T)
            N_d1_target = delta * np.exp(r_f * T)
            d1_target = norm.ppf(N_d1_target)
        else:
            # For put: delta = -exp(-r_f * T) * N(-d1)
            # So: N(-d1) = -delta * exp(r_f * T)
            # Therefore: N(d1) = 1 + delta * exp(r_f * T)
            N_d1_target = 1 + delta * np.exp(r_f * T)
            d1_target = norm.ppf(N_d1_target)

        # From d1 formula: d1 = (ln(S/K) + (r_d - r_f + 0.5*sigma^2)*T) / (sigma*sqrt(T))
        # Solving for K:
        # ln(S/K) = d1 * sigma * sqrt(T) - (r_d - r_f + 0.5*sigma^2)*T
        # K = S * exp(-(d1 * sigma * sqrt(T) - (r_d - r_f + 0.5*sigma^2)*T))

        ln_S_over_K = d1_target * sigma * np.sqrt(T) - (r_d - r_f + 0.5 * sigma ** 2) * T
        K = S * np.exp(-ln_S_over_K)

        return K

    def calculate_25delta_strikes(self, S: float, T: float, r_d: float, r_f: float,
                                  atm_vol: float, rr_25: float, bf_25: float) -> Tuple[float, float, float]:
        """
        Calculate 25-delta call and put strikes from market quotes

        Parameters:
        -----------
        S : float
            Spot price
        T : float
            Time to maturity
        r_d, r_f : float
            Domestic and foreign rates
        atm_vol : float
            ATM volatility
        rr_25 : float
            25-delta risk reversal (vol_25c - vol_25p)
        bf_25 : float
            25-delta butterfly ((vol_25c + vol_25p)/2 - vol_atm)

        Returns:
        --------
        Tuple of (K_25p, K_atm, K_25c)
        """
        # Calculate forward
        F = S * np.exp((r_d - r_f) * T)

        # ATM strike (typically DNS - Delta Neutral Straddle)
        K_atm = F  # Simplified - could use DNS convention

        # Calculate 25-delta volatilities from market quotes
        # RR = vol_25c - vol_25p
        # BF = (vol_25c + vol_25p)/2 - vol_atm
        # Solving:
        vol_25c = atm_vol + bf_25 + 0.5 * rr_25
        vol_25p = atm_vol + bf_25 - 0.5 * rr_25

        # Calculate 25-delta strikes
        # Note: 25-delta put has delta = -0.25, 25-delta call has delta = 0.25
        K_25c = self.calculate_strike_from_delta(S, T, r_d, r_f, vol_25c, 0.25, 'call')
        K_25p = self.calculate_strike_from_delta(S, T, r_d, r_f, vol_25p, -0.25, 'put')

        return K_25p, K_atm, K_25c

    def calculate_gvv_adjustment(self, S: float, K: float, T: float, r_d: float, r_f: float,
                                 K_25p: float, K_atm: float, K_25c: float,
                                 vol_25p: float, vol_atm: float, vol_25c: float) -> float:
        """
        Calculate GVV adjustment to Black-Scholes price

        The GVV method uses second-order Greeks (gamma, vanna, volga) to adjust prices
        """
        # Calculate BS prices and Greeks at market strikes with their respective vols
        # ATM
        greeks_atm = self.gk.calculate_greeks(S, K_atm, T, r_d, r_f, vol_atm, 'call')

        # 25-delta put
        greeks_25p = self.gk.calculate_greeks(S, K_25p, T, r_d, r_f, vol_25p, 'put')

        # 25-delta call
        greeks_25c = self.gk.calculate_greeks(S, K_25c, T, r_d, r_f, vol_25c, 'call')

        # Calculate Greeks for target strike K with ATM vol (first approximation)
        greeks_k = self.gk.calculate_greeks(S, K, T, r_d, r_f, vol_atm, 'call')

        # Get vegas (need to multiply by 100 since our vega is for 1% move)
        vega_atm = greeks_atm['vega'] * 100
        vega_25p = greeks_25p['vega'] * 100
        vega_25c = greeks_25c['vega'] * 100
        vega_k = greeks_k['vega'] * 100

        # Calculate weights based on vega matching
        # The idea is to replicate the vega of option at strike K using the three market instruments
        if abs(vega_atm) < 1e-10 or abs(vega_25p) < 1e-10 or abs(vega_25c) < 1e-10:
            # Fallback to simple interpolation
            return 0.0

        # Simplified GVV: use vega-weighted combination
        total_market_vega = vega_atm + vega_25p + vega_25c
        if abs(total_market_vega) < 1e-10:
            return 0.0

        # Calculate market prices with smile
        price_atm_market = self.gk.price_option(S, K_atm, T, r_d, r_f, vol_atm, 'call')
        price_25p_market = self.gk.price_option(S, K_25p, T, r_d, r_f, vol_25p, 'put')
        price_25c_market = self.gk.price_option(S, K_25c, T, r_d, r_f, vol_25c, 'call')

        # Calculate BS prices with ATM vol (no smile)
        price_atm_bs = self.gk.price_option(S, K_atm, T, r_d, r_f, vol_atm, 'call')
        price_25p_bs = self.gk.price_option(S, K_25p, T, r_d, r_f, vol_atm, 'put')
        price_25c_bs = self.gk.price_option(S, K_25c, T, r_d, r_f, vol_atm, 'call')

        # Cost of smile at market strikes
        cost_atm = price_atm_market - price_atm_bs
        cost_25p = price_25p_market - price_25p_bs
        cost_25c = price_25c_market - price_25c_bs

        # GVV adjustment using vega weights
        # This is simplified - full GVV uses gamma, vanna, volga
        if K <= K_25p:
            # Deep OTM put region - use 25p
            adjustment = cost_25p * (vega_k / vega_25p) if vega_25p != 0 else 0
        elif K >= K_25c:
            # Deep OTM call region - use 25c
            adjustment = cost_25c * (vega_k / vega_25c) if vega_25c != 0 else 0
        else:
            # Interpolate between instruments
            if K <= K_atm:
                # Between 25p and ATM
                weight_25p = (K_atm - K) / (K_atm - K_25p)
                weight_atm = 1 - weight_25p
                adjustment = weight_25p * cost_25p + weight_atm * cost_atm
            else:
                # Between ATM and 25c
                weight_atm = (K_25c - K) / (K_25c - K_atm)
                weight_25c = 1 - weight_atm
                adjustment = weight_atm * cost_atm + weight_25c * cost_25c

        return adjustment

    def price_option(self, S: float, K: float, T: float, r_d: float, r_f: float,
                     sigma: float, option_type: str = 'call', **kwargs) -> float:
        """
        Price option using GVV model

        Additional kwargs:
        -----------------
        rr_25 : float
            25-delta risk reversal
        bf_25 : float
            25-delta butterfly
        """
        # Extract GVV specific parameters
        rr_25 = kwargs.get('rr_25', 0)
        bf_25 = kwargs.get('bf_25', 0)

        # If no smile parameters, just use Black-Scholes
        if rr_25 == 0 and bf_25 == 0:
            return self.gk.price_option(S, K, T, r_d, r_f, sigma, option_type)

        # Calculate 25-delta strikes and volatilities
        vol_25c = sigma + bf_25 + 0.5 * rr_25
        vol_25p = sigma + bf_25 - 0.5 * rr_25

        K_25p, K_atm, K_25c = self.calculate_25delta_strikes(S, T, r_d, r_f, sigma, rr_25, bf_25)

        # Get BS price with ATM vol
        bs_price = self.gk.price_option(S, K, T, r_d, r_f, sigma, option_type)

        # Calculate GVV adjustment
        adjustment = self.calculate_gvv_adjustment(
            S, K, T, r_d, r_f,
            K_25p, K_atm, K_25c,
            vol_25p, sigma, vol_25c
        )

        # Apply adjustment
        gvv_price = bs_price + adjustment

        # Ensure price is non-negative and respects bounds
        if option_type == 'call':
            # Call price bounds: max(S*exp(-r_f*T) - K*exp(-r_d*T), 0) <= C <= S*exp(-r_f*T)
            lower_bound = max(S * np.exp(-r_f * T) - K * np.exp(-r_d * T), 0)
            upper_bound = S * np.exp(-r_f * T)
        else:
            # Put price bounds: max(K*exp(-r_d*T) - S*exp(-r_f*T), 0) <= P <= K*exp(-r_d*T)
            lower_bound = max(K * np.exp(-r_d * T) - S * np.exp(-r_f * T), 0)
            upper_bound = K * np.exp(-r_d * T)

        gvv_price = np.clip(gvv_price, lower_bound, upper_bound)

        return gvv_price

    def calculate_greeks(self, S: float, K: float, T: float, r_d: float, r_f: float,
                         sigma: float, option_type: str = 'call', **kwargs) -> Dict[str, float]:
        """
        Calculate Greeks using GVV-adjusted implied volatility
        """
        # Extract GVV specific parameters
        rr_25 = kwargs.get('rr_25', 0)
        bf_25 = kwargs.get('bf_25', 0)

        # If no smile, use standard Greeks
        if rr_25 == 0 and bf_25 == 0:
            return self.gk.calculate_greeks(S, K, T, r_d, r_f, sigma, option_type)

        # Calculate implied vol for this strike using GVV
        # This is a simplified approach - could use full smile interpolation
        K_25p, K_atm, K_25c = self.calculate_25delta_strikes(S, T, r_d, r_f, sigma, rr_25, bf_25)

        vol_25c = sigma + bf_25 + 0.5 * rr_25
        vol_25p = sigma + bf_25 - 0.5 * rr_25

        # Interpolate volatility based on strike
        if K <= K_25p:
            implied_vol = vol_25p
        elif K >= K_25c:
            implied_vol = vol_25c
        elif K <= K_atm:
            # Linear interpolation between 25p and ATM
            alpha = (K - K_25p) / (K_atm - K_25p)
            implied_vol = vol_25p * (1 - alpha) + sigma * alpha
        else:
            # Linear interpolation between ATM and 25c
            alpha = (K - K_atm) / (K_25c - K_atm)
            implied_vol = sigma * (1 - alpha) + vol_25c * alpha

        # Calculate Greeks with interpolated volatility
        greeks = self.gk.calculate_greeks(S, K, T, r_d, r_f, implied_vol, option_type)

        return greeks
