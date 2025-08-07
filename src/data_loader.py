import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class FXDataLoader:
    '''Load and preprocess FX options data'''

    def __init__(self, fx_path: Path, discount_path: Path):
        self.fx_path = fx_path
        self.discount_path = discount_path
        self.fx_data = None
        self.discount_data = None
        self.processed_data = {}

    def load_data(self) -> None:
        '''Load FX and discount curve data'''
        print("Loading FX data...")
        self.fx_data = pd.read_parquet(self.fx_path)

        # todo: delete this line once done with testing
        self.fx_data = self.fx_data.tail(1650)

        print("Loading discount curves...")
        self.discount_data = pd.read_parquet(self.discount_path)

        print(f"FX data shape: {self.fx_data.shape}")
        print(f"Date range: {self.fx_data.index.min()} to {self.fx_data.index.max()}")

    def parse_columns(self) -> Dict[str, Dict[str, List[str]]]:
        '''Parse and categorize columns by currency pair and data type'''
        columns_dict = {}

        for pair in ['USDJPY', 'GBPNZD', 'USDCAD']:
            columns_dict[pair] = {
                'spot': [],
                'forwards': [],
                'atm_vols': [],
                'rr_25': [],
                'rr_10': [],
                'bf_25': [],
                'bf_10': []
            }

            for col in self.fx_data.columns:
                if pair not in col:
                    continue

                col_clean = col.replace(' Curncy', '')

                # Spot
                if col_clean == pair:
                    columns_dict[pair]['spot'].append(col)
                # Forwards
                elif any(tenor in col_clean for tenor in ['1W', '2W', '3W', '1M', '2M', '3M', '4M', '6M', '9M',
                                                          '12M']) and 'V' not in col_clean and 'R' not in col_clean and 'B' not in col_clean:
                    columns_dict[pair]['forwards'].append(col)
                # ATM Vols
                elif 'V' in col_clean:
                    columns_dict[pair]['atm_vols'].append(col)
                # 25 Delta Risk Reversals
                elif '25R' in col_clean:
                    columns_dict[pair]['rr_25'].append(col)
                # 10 Delta Risk Reversals
                elif '10R' in col_clean:
                    columns_dict[pair]['rr_10'].append(col)
                # 25 Delta Butterflies
                elif '25B' in col_clean:
                    columns_dict[pair]['bf_25'].append(col)
                # 10 Delta Butterflies
                elif '10B' in col_clean:
                    columns_dict[pair]['bf_10'].append(col)

        return columns_dict

    def process_pair_data(self, pair: str) -> pd.DataFrame:
        '''Process data for a specific currency pair'''
        columns = self.parse_columns()[pair]

        # Create dataframe for this pair
        pair_data = pd.DataFrame(index=self.fx_data.index)

        # Add spot
        if columns['spot']:
            pair_data['spot'] = self.fx_data[columns['spot'][0]]

        # Add forwards and calculate forward points
        for fwd_col in columns['forwards']:
            tenor = self._extract_tenor(fwd_col)
            if tenor:
                pair_data[f'fwd_{tenor}'] = self.fx_data[fwd_col]
                # Forward points are typically in pips (need to divide by 10000 for major pairs)
                if pair == 'USDJPY':
                    pair_data[f'fwd_points_{tenor}'] = self.fx_data[fwd_col] / 100
                else:
                    pair_data[f'fwd_points_{tenor}'] = self.fx_data[fwd_col] / 10000

        # Add ATM vols (already in percentage)
        for vol_col in columns['atm_vols']:
            tenor = self._extract_tenor(vol_col)
            if tenor:
                pair_data[f'atm_vol_{tenor}'] = self.fx_data[vol_col] / 100

        # Add risk reversals and butterflies (in vol points)
        for rr_col in columns['rr_25']:
            tenor = self._extract_tenor(rr_col)
            if tenor:
                pair_data[f'rr_25_{tenor}'] = self.fx_data[rr_col] / 100

        for rr_col in columns['rr_10']:
            tenor = self._extract_tenor(rr_col)
            if tenor:
                pair_data[f'rr_10_{tenor}'] = self.fx_data[rr_col] / 100

        for bf_col in columns['bf_25']:
            tenor = self._extract_tenor(bf_col)
            if tenor:
                pair_data[f'bf_25_{tenor}'] = self.fx_data[bf_col] / 100

        for bf_col in columns['bf_10']:
            tenor = self._extract_tenor(bf_col)
            if tenor:
                pair_data[f'bf_10_{tenor}'] = self.fx_data[bf_col] / 100

        # Calculate forward outright prices
        for tenor in self._get_tenors_from_columns(columns['forwards']):
            if f'fwd_points_{tenor}' in pair_data.columns:
                pair_data[f'forward_{tenor}'] = pair_data['spot'] + pair_data[f'fwd_points_{tenor}']

        return pair_data

    def _extract_tenor(self, col_name: str) -> Optional[str]:
        '''Extract tenor from column name'''
        col_clean = col_name.replace(' Curncy', '')
        tenors = ['1W', '2W', '3W', '1M', '2M', '3M', '4M', '6M', '9M', '12M', '1Y']

        for tenor in tenors:
            if tenor in col_clean:
                return tenor.replace('12M', '1Y')  # Normalize 12M to 1Y
        return None

    def _get_tenors_from_columns(self, columns: List[str]) -> List[str]:
        '''Get unique tenors from column list'''
        tenors = []
        for col in columns:
            tenor = self._extract_tenor(col)
            if tenor and tenor not in tenors:
                tenors.append(tenor)
        return tenors

    def get_discount_factors(self, date: pd.Timestamp, tenor: str) -> float:
        '''Get discount factor for a given date and tenor'''
        if self.discount_data is None:
            return 1.0

        try:
            mask = (self.discount_data['date'] == date) & (self.discount_data['tenor'] == tenor)
            df_val = self.discount_data.loc[mask, 'discount_factor'].values
            return df_val[0] if len(df_val) > 0 else 1.0
        except:
            return 1.0

    def process_all_pairs(self) -> Dict[str, pd.DataFrame]:
        '''Process all currency pairs'''
        processed = {}
        for pair in ['USDJPY', 'GBPNZD', 'USDCAD']:
            print(f"Processing {pair}...")
            processed[pair] = self.process_pair_data(pair)
            print(f"  Shape: {processed[pair].shape}")
            print(f"  Columns: {len(processed[pair].columns)}")

        self.processed_data = processed
        return processed