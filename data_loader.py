"""
===============================================================================
data_loader.py - Data loading and preprocessing
===============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from scipy.interpolate import interp1d

from config import Config


class DataLoader:
    """Load and preprocess FX data"""

    def __init__(self, config: Config):
        self.config = config
        self.data = {}
        self.curves = None

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load FX data for all pairs"""

        # Load main FX data
        fx_data = pd.read_parquet(self.config.FX_DATA_FILE)

        # Process each currency pair
        for pair in self.config.CURRENCY_PAIRS:
            pair_data = self._process_pair_data(fx_data, pair)
            if pair_data is not None and len(pair_data) > 0:
                self.data[pair] = pair_data
                print(f"  Loaded {pair}: {len(pair_data)} days")

        return self.data

    def _process_pair_data(self, fx_data: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Process data for a single currency pair"""

        # Find columns for this pair
        pair_cols = [col for col in fx_data.columns if pair in col]

        if not pair_cols:
            return None

        # Create DataFrame for this pair
        pair_df = pd.DataFrame(index=fx_data.index)

        # Extract spot price
        spot_col = f'{pair} Curncy'
        if spot_col in fx_data.columns:
            pair_df['spot'] = fx_data[spot_col]

        # Extract forward points and volatilities for each tenor
        for tenor in self.config.TENORS:
            # Forward points
            fwd_col = f'{pair}{tenor} Curncy'
            if fwd_col in fx_data.columns:
                pair_df[f'fwd_points_{tenor}'] = fx_data[fwd_col]

            # ATM volatility
            atm_col = f'{pair}V{tenor} Curncy'
            if atm_col in fx_data.columns:
                pair_df[f'atm_vol_{tenor}'] = fx_data[atm_col] / 100  # Convert to decimal

            # 25-delta risk reversal
            rr25_col = f'{pair}25R{tenor} Curncy'
            if rr25_col in fx_data.columns:
                pair_df[f'rr_25_{tenor}'] = fx_data[rr25_col] / 100

            # 25-delta butterfly
            bf25_col = f'{pair}25B{tenor} Curncy'
            if bf25_col in fx_data.columns:
                pair_df[f'bf_25_{tenor}'] = fx_data[bf25_col] / 100

            # 10-delta risk reversal
            rr10_col = f'{pair}10R{tenor} Curncy'
            if rr10_col in fx_data.columns:
                pair_df[f'rr_10_{tenor}'] = fx_data[rr10_col] / 100

            # 10-delta butterfly
            bf10_col = f'{pair}10B{tenor} Curncy'
            if bf10_col in fx_data.columns:
                pair_df[f'bf_10_{tenor}'] = fx_data[bf10_col] / 100

        # Calculate realized volatility
        if 'spot' in pair_df.columns:
            log_returns = np.log(pair_df['spot'] / pair_df['spot'].shift(1))
            pair_df['realized_vol_20d'] = log_returns.rolling(20).std() * np.sqrt(252)

        # Forward fill missing data
        pair_df = pair_df.fillna(method='ffill')

        # Drop rows with no spot price
        pair_df = pair_df.dropna(subset=['spot'])

        return pair_df

    def build_discount_curves(self) -> pd.DataFrame:
        """Build or load discount curves"""

        if self.config.DISCOUNT_CURVES_FILE.exists():
            curves = pd.read_parquet(self.config.DISCOUNT_CURVES_FILE)
        else:
            # Build simple curves
            curves = self._build_simple_curves()

        self.curves = curves
        return curves

    def _build_simple_curves(self) -> pd.DataFrame:
        """Build simple discount curves"""

        # Use first pair's dates as reference
        if self.data:
            dates = list(self.data.values())[0].index
        else:
            return pd.DataFrame()

        # Simple flat curves
        curves = pd.DataFrame(index=dates)
        curves['rate'] = 0.05  # 5% domestic rate
        curves['rate_foreign'] = 0.02  # 2% foreign rate

        return curves
