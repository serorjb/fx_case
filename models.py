"""
Option pricing models: Garman-Kohlhagen, GVV, and SABR
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar, brentq
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseModel(ABC):
    """Base class for all pricing models"""

    @abstractmethod
    def price(self, spot: float, strike: float, T: float, r_d: float, r_f: float,
              sigma: float, is_call: bool = True, **kwargs) -> float:
        """Price an option"""
        pass

    def calculate_greeks(self, spot: float, strike: float, T: float, r_d: float, r_f: float,
                         sigma: float, is_call: bool = True) -> dict:
        """Calculate option Greeks"""
        # Use finite differences for Greeks
        eps = 0.01

        base_price = self.price(spot, strike, T, r_d, r_f, sigma, is_call)

        # Delta
        price_up = self.price(spot * (1 + eps), strike, T, r_d, r_f, sigma, is_call)
        delta = (price_up - base_price) / (spot * eps)

        # Gamma
        price_down = self.price(spot * (1 - eps), strike, T, r_d, r_f, sigma, is_call)
        gamma = (price_up - 2 * base_price + price_down) / (spot * eps) ** 2

        # Vega
        vega_up = self.price(spot, strike, T, r_d, r_f, sigma + eps, is_call)
        vega = (vega_up - base_price) / eps

        # Theta (1 day)
        if T > 1 / 365:
            price_tminus = self.price(spot, strike, T - 1 / 365, r_d, r_f, sigma, is_call)
            theta = price_tminus - base_price
        else:
            theta = 0

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta
        }


class GarmanKohlhagen(BaseModel):
    """Garman-Kohlhagen model for FX options"""

    def price(self, spot: float, strike: float, T: float, r_d: float, r_f: float,
              sigma: float, is_call: bool = True, **kwargs) -> float:
        """Price using Black-Scholes for FX"""

        if T <= 0:
            return max(0, spot - strike) if is_call else max(0, strike - spot)

        d1 = (np.log(spot / strike) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if is_call:
            price = spot * np.exp(-r_f * T) * norm.cdf(d1) - strike * np.exp(-r_d * T) * norm.cdf(d2)
        else:
            price = strike * np.exp(-r_d * T) * norm.cdf(-d2) - spot * np.exp(-r_f * T) * norm.cdf(-d1)

        return max(0, price)


class GVVModel(BaseModel):
    """Gamma-Vanna-Volga model"""

    def __init__(self):
        self.gk = GarmanKohlhagen()

    def price(self, spot: float, strike: float, T: float, r_d: float, r_f: float,
              sigma: float, is_call: bool = True, **kwargs) -> float:
        """Price using GVV adjustment"""

        # Get smile parameters
        rr = kwargs.get('rr', 0)  # Risk reversal
        bf = kwargs.get('bf', 0)  # Butterfly

        # Base GK price
        base_price = self.gk.price(spot, strike, T, r_d, r_f, sigma, is_call)

        if rr == 0 and bf == 0:
            return base_price

        # Calculate 25-delta strikes
        K_25p, K_atm, K_25c = self._calculate_pillar_strikes(spot, T, r_d, r_f, sigma, rr, bf)

        # Calculate volatilities at pillar strikes
        vol_25c = sigma + bf + 0.5 * rr
        vol_25p = sigma + bf - 0.5 * rr

        # GVV adjustment using simplified weights
        if strike <= K_25p:
            adjustment_vol = vol_25p
        elif strike >= K_25c:
            adjustment_vol = vol_25c
        elif strike <= K_atm:
            # Interpolate between 25p and ATM
            w = (strike - K_25p) / (K_atm - K_25p)
            adjustment_vol = vol_25p * (1 - w) + sigma * w
        else:
            # Interpolate between ATM and 25c
            w = (strike - K_atm) / (K_25c - K_atm)
            adjustment_vol = sigma * (1 - w) + vol_25c * w

        # Recalculate with adjusted volatility
        adjusted_price = self.gk.price(spot, strike, T, r_d, r_f, adjustment_vol, is_call)

        return adjusted_price

    def _calculate_pillar_strikes(self, spot: float, T: float, r_d: float, r_f: float,
                                  sigma: float, rr: float, bf: float) -> Tuple[float, float, float]:
        """Calculate 25-delta put, ATM, and 25-delta call strikes"""

        # Forward price
        F = spot * np.exp((r_d - r_f) * T)

        # ATM strike (simplified as forward)
        K_atm = F

        # 25-delta strikes (simplified calculation)
        vol_25c = sigma + bf + 0.5 * rr
        vol_25p = sigma + bf - 0.5 * rr

        # Using approximate formula for 25-delta strikes
        d1_25c = norm.ppf(0.25 * np.exp(r_f * T))
        K_25c = F * np.exp(-d1_25c * vol_25c * np.sqrt(T) + 0.5 * vol_25c ** 2 * T)

        d1_25p = -norm.ppf(0.25 * np.exp(r_f * T))
        K_25p = F * np.exp(-d1_25p * vol_25p * np.sqrt(T) + 0.5 * vol_25p ** 2 * T)

        return K_25p, K_atm, K_25c


class SABRModel(BaseModel):
    """SABR model for volatility smile"""

    def __init__(self, beta: float = 0.5):
        self.beta = beta
        self.gk = GarmanKohlhagen()

    def price(self, spot: float, strike: float, T: float, r_d: float, r_f: float,
              sigma: float, is_call: bool = True, **kwargs) -> float:
        """Price using SABR volatility"""

        # Get SABR parameters
        rr = kwargs.get('rr', 0)
        bf = kwargs.get('bf', 0)

        # Calibrate SABR parameters from market quotes
        alpha, rho, nu = self._calibrate_sabr(spot, T, sigma, rr, bf)

        # Calculate SABR implied volatility
        sabr_vol = self._sabr_vol(spot, strike, T, alpha, self.beta, rho, nu)

        # Price using GK with SABR vol
        return self.gk.price(spot, strike, T, r_d, r_f, sabr_vol, is_call)

    def _calibrate_sabr(self, F: float, T: float, atm_vol: float, rr: float, bf: float) -> Tuple[float, float, float]:
        """Calibrate SABR parameters from market quotes"""

        # Initial guess
        alpha = atm_vol
        rho = -0.3 if rr < 0 else 0.3  # Correlation based on risk reversal sign
        nu = 0.3  # Vol of vol

        # Simplified calibration - adjust based on smile
        if abs(rr) > 0:
            rho = np.clip(-rr * 10, -0.9, 0.9)

        if bf > 0:
            nu = np.clip(bf * 20, 0.1, 1.0)

        return alpha, rho, nu

    def _sabr_vol(self, F: float, K: float, T: float, alpha: float, beta: float,
                  rho: float, nu: float) -> float:
        """Calculate SABR implied volatility using Hagan's approximation"""

        if abs(F - K) < 1e-10:
            # ATM case
            return alpha * (1 + T * ((2 - 3 * rho ** 2) * nu ** 2 / 24))

        # General case
        FK_beta = (F * K) ** ((1 - beta) / 2)
        z = (nu / alpha) * FK_beta * np.log(F / K)
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        if abs(x_z) < 1e-10:
            return alpha / FK_beta

        A = alpha / (FK_beta * (1 + (1 - beta) ** 2 * np.log(F / K) ** 2 / 24))
        B = 1 + T * ((1 - beta) ** 2 * alpha ** 2 / (24 * FK_beta ** 2) +
                     rho * beta * nu * alpha / (4 * FK_beta) +
                     nu ** 2 * (2 - 3 * rho ** 2) / 24)

        return A * (z / x_z) * B