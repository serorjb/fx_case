"""
Enhanced feature engineering with GARCH volatility forecasting
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Import from existing modules
from src.features import HMMRegimeDetector, FeatureEngineer
from src.models.garch_models import GARCHForecaster, VolatilityComparator, create_time_features


class EnhancedFeatureEngineer(FeatureEngineer):
    """
    Enhanced feature engineering with GARCH and multivariate analysis
    """

    def __init__(self):
        super().__init__()
        self.vol_comparator = VolatilityComparator()
        self.mv_garch = GARCHForecaster(model_type='GARCH')
        self.garch_features = {}

    def create_garch_features(self, data: pd.DataFrame, pair: str) -> pd.DataFrame:
        """
        Create GARCH-based volatility features
        """
        # Get basic GARCH features
        garch_features = self.vol_comparator.compare_vol_models(data, pair)

        # Add to cache
        self.garch_features[pair] = garch_features

        return garch_features

    def create_multivariate_garch_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create features from multivariate GARCH model
        """
        # Prepare returns for all pairs
        returns_dict = {}
        for pair, data in data_dict.items():
            returns = self.mv_garch.prepare_returns(data['spot'])
            returns_dict[pair] = returns

        # Fit multivariate GARCH
        mv_results = self.mv_garch.fit_multivariate_garch(returns_dict)

        # Create features DataFrame
        features = pd.DataFrame()

        # Portfolio volatility forecast
        if 'forecasts' in mv_results:
            portfolio_vols = pd.Series(mv_results['forecasts'])
            features['mv_garch_portfolio_vol'] = portfolio_vols

        # Correlation features
        if 'correlations' in mv_results:
            for date, corr_matrix in mv_results['correlations'].items():
                # Average correlation
                off_diag = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                features.loc[date, 'avg_correlation'] = off_diag.mean()
                features.loc[date, 'max_correlation'] = off_diag.max()
                features.loc[date, 'min_correlation'] = off_diag.min()

                # Specific pair correlations
                pairs = list(corr_matrix.index)
                for i, pair1 in enumerate(pairs):
                    for j, pair2 in enumerate(pairs):
                        if i < j:
                            features.loc[date, f'corr_{pair1}_{pair2}'] = corr_matrix.loc[pair1, pair2]

        return features

    def create_volatility_comparison_features(self, data: pd.DataFrame, pair: str,
                                              models: Dict) -> pd.DataFrame:
        """
        Compare option prices using different volatility inputs
        """
        features = self.vol_comparator.simulate_option_prices_with_garch(data, pair, models)

        # Add volatility smile features with GARCH
        for tenor in ['1M', '3M', '6M']:
            if f'garch_forecast_{pair}' in self.garch_features.get(pair, {}).columns:
                garch_vol = self.garch_features[pair][f'garch_forecast_{pair}']

                # Compare with RR and BF
                if f'rr_25_{tenor}' in data.columns:
                    features[f'rr_vs_garch_{tenor}_{pair}'] = data[f'rr_25_{tenor}'] / garch_vol.replace(0, np.nan)

                if f'bf_25_{tenor}' in data.columns:
                    features[f'bf_vs_garch_{tenor}_{pair}'] = data[f'bf_25_{tenor}'] / garch_vol.replace(0, np.nan)

        return features

    def create_all_enhanced_features(self, data_dict: Dict[str, pd.DataFrame],
                                     models: Dict = None) -> Dict[str, pd.DataFrame]:
        """
        Create all enhanced features including GARCH
        """
        all_features = {}

        for pair, data in data_dict.items():
            print(f"Creating enhanced features for {pair}...")

            # Base features (with no look-ahead bias)
            base_features = super().create_all_features(data, pair, models)

            # GARCH features
            garch_features = self.create_garch_features(data, pair)

            # Volatility comparison features
            if models:
                vol_comp_features = self.create_volatility_comparison_features(data, pair, models)
            else:
                vol_comp_features = pd.DataFrame(index=data.index)

            # Time features
            time_features = create_time_features(data)

            # Combine all features
            combined = pd.concat([
                base_features,
                garch_features,
                vol_comp_features,
                time_features
            ], axis=1)

            # Remove duplicates
            combined = combined.loc[:, ~combined.columns.duplicated()]

            all_features[pair] = combined

        # Add multivariate features
        print("Creating multivariate GARCH features...")
        mv_features = self.create_multivariate_garch_features(data_dict)

        # Add multivariate features to each pair's features
        for pair in all_features:
            all_features[pair] = pd.concat([all_features[pair], mv_features], axis=1)

        return all_features