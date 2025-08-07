"""
Analysis script for detailed examination
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.data_loader import FXDataLoader
from src.features import FeatureEngineer
from src.models.garman_kohlhagen import GarmanKohlhagen
from src.models.gvv_model import GVVModel
from src.models.sabr_model import SABRModel


def analyze_pricing_models():
    """
    Compare different pricing models
    """
    print("Analyzing pricing models...")

    # Load data
    loader = FXDataLoader(
        Path('data/FX.parquet'),
        Path('data/discount_curves.parquet')
    )
    loader.load_data()
    data = loader.process_all_pairs()

    # Initialize models
    gk = GarmanKohlhagen()
    gvv = GVVModel()
    sabr = SABRModel()

    # Test parameters
    S = 150  # USDJPY spot
    K = 152  # Strike
    T = 0.25  # 3 months
    r_d = 0.05  # USD rate
    r_f = -0.001  # JPY rate
    atm_vol = 0.10
    rr_25 = -0.01
    bf_25 = 0.002

    # Price with different models
    gk_price = gk.price_option(S, K, T, r_d, r_f, atm_vol, 'call')
    gvv_price, gvv_greeks = gvv.price_option(S, K, T, r_d, r_f, atm_vol, rr_25, bf_25, 'call')

    print(f"\nPricing comparison for USDJPY call option:")
    print(f"Spot: {S}, Strike: {K}, Maturity: {T} years")
    print(f"Garman-Kohlhagen price: ${gk_price:.4f}")
    print(f"GVV price: ${gvv_price:.4f}")
    print(f"Difference: ${abs(gk_price - gvv_price):.4f}")

    # Calculate Greeks
    greeks = gk.calculate_greeks(S, K, T, r_d, r_f, atm_vol, 'call')
    print(f"\nGreeks (Garman-Kohlhagen):")
    for name, value in greeks.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    analyze_pricing_models()