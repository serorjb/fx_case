# main.py
"""
Main execution script for FX Options Trading System
Enhanced version with multi-model approach including GVV, SABR, ML models, and Delta Hedging
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Import configurations
from config.settings import *

# Import data and curves modules
from src.data_loader import FXDataLoader
from src.curves import DiscountCurveBuilder, process_discount_curves

# Import feature engineering modules
from src.features_enhanced import EnhancedFeatureEngineer

# Import models
from src.models.gk_model import GarmanKohlhagen
from src.models.gvv_model import GVVModel
from src.models.sabr_model import SABRModel
from src.models.ml_models import LightGBMModel

# Import strategies
from src.strategies.multi_model_strategies import (
    create_all_model_strategies,
    MLSignalStrategy,
    GVVArbitrageStrategy,
    SABRVolatilityStrategy,
    ModelConsensusStrategy
)
from src.strategies.volatility_arbitrage import VolatilityArbitrageStrategy
from src.strategies.carry_strategy import CarryToVolStrategy

# Import backtesting and risk management
from src.backtesting.engine import BacktestingEngine
from src.backtesting.performance import PerformanceAnalyzer
from src.risk_management import RiskManager
from src.hedging import DeltaHedger

# Import visualization
from src.visualization.plots import TradingVisualizer
from src.visualization.multi_model_plots import MultiModelVisualizer
from src.visualization.garch_plots import GARCHVisualizer


class FXOptionsSystem:
    """
    Main system class for FX Options trading
    """

    def __init__(self):
        """Initialize the FX Options Trading System"""
        self.loader = None
        self.data = None
        self.features = None
        self.models = None
        self.ml_models = None
        self.strategies = None
        self.results = None

    def initialize_system(self) -> bool:
        """Initialize all system components"""
        print("=" * 80)
        print(" FX OPTIONS TRADING SYSTEM - MULTI-MODEL ENHANCED VERSION ".center(80))
        print("=" * 80)
        print(f"\n📅 Start time: {datetime.now()}")
        print(f"⚙️  Configuration:")
        print(f"   - Curve Method: {CURVE_METHOD}")
        print(f"   - HMM States: {HMM_STATES}")
        print(f"   - Delta Hedging: {'Enabled' if ENABLE_DELTA_HEDGING else 'Disabled'}")
        print(f"   - Initial Capital: ${INITIAL_CAPITAL:,.0f}")
        print(f"   - Max Drawdown Limit: {MAX_DRAWDOWN:.1%}")
        print("-" * 80)

        # Create necessary directories
        for directory in [PLOTS_DIR, RESULTS_DIR, DATA_DIR]:
            directory.mkdir(exist_ok=True, parents=True)

        # Check for data files
        if not FX_DATA_PATH.exists():
            raise FileNotFoundError(f"❌ FX data not found at {FX_DATA_PATH}")

        # Process discount curves if needed
        if not DISCOUNT_CURVES_PATH.exists():
            print("\n📊 Processing discount curves...")
            try:
                process_discount_curves(
                    data_dir=DATA_DIR / "FRED",
                    output_dir=DATA_DIR,
                    plot_dir=PLOTS_DIR
                )
                print("   ✓ Discount curves processed successfully")
            except Exception as e:
                print(f"   ❌ Error processing discount curves: {e}")
                return False

        return True

    def load_and_prepare_data(self) -> Tuple[FXDataLoader, Dict]:
        """Load FX data and discount curves"""
        print("\n" + "=" * 80)
        print("1️⃣  LOADING AND PREPARING DATA")
        print("-" * 80)

        # Initialize data loader
        self.loader = FXDataLoader(FX_DATA_PATH, DISCOUNT_CURVES_PATH)
        self.loader.load_data()

        # Process all currency pairs
        self.data = self.loader.process_all_pairs()

        # Data quality check
        print("\n📊 Data Summary:")
        valid_pairs = []
        for pair in CURRENCY_PAIRS:
            if pair not in self.data:
                print(f"   ⚠️  {pair}: Not found in data")
                continue

            pair_data = self.data[pair]
            print(f"   ✓ {pair}: {pair_data.shape[0]:,} days, {pair_data.shape[1]:,} features")

            # Check for critical columns
            required_cols = ['spot', 'atm_vol_1M', 'rr_25_1M', 'bf_25_1M']
            missing = [col for col in required_cols if col not in pair_data.columns]
            if missing:
                print(f"      ⚠️  Missing columns: {missing}")
            else:
                valid_pairs.append(pair)

        print(f"\n✅ Successfully loaded {len(valid_pairs)} currency pairs")
        return self.loader, self.data

    def create_enhanced_features(self) -> Tuple[Dict, Dict]:
        """Create all enhanced features including GARCH and HMM"""
        print("\n" + "=" * 80)
        print("2️⃣  CREATING ENHANCED FEATURES")
        print("-" * 80)

        # Initialize pricing models for feature generation
        self.models = {
            'gk': GarmanKohlhagen(),
            'gvv': GVVModel(),
            'sabr': SABRModel()
        }

        # Initialize enhanced feature engineer
        enhanced_fe = EnhancedFeatureEngineer()

        # Create features for all pairs
        print("\n📈 Generating features...")
        self.features = enhanced_fe.create_all_enhanced_features(self.data, self.models)

        # Print feature summary
        print("\n📊 Feature Summary:")
        for pair, features in self.features.items():
            if features is None or features.empty:
                print(f"   ⚠️  {pair}: No features created")
                continue

            print(f"   ✓ {pair}: {features.shape[1]:,} total features")

            # Analyze feature groups
            garch_features = [col for col in features.columns if 'garch' in col.lower()]
            hmm_features = [col for col in features.columns if 'hmm' in col.lower() or 'regime' in col.lower()]
            time_features = [col for col in features.columns if
                             any(t in col.lower() for t in ['day', 'month', 'quarter', 'year'])]
            model_features = [col for col in features.columns if any(m in col.lower() for m in ['gvv', 'sabr', 'gk'])]

            print(f"      - GARCH features: {len(garch_features)}")
            print(f"      - HMM features: {len(hmm_features)}")
            print(f"      - Time features: {len(time_features)}")
            print(f"      - Model features: {len(model_features)}")

        return self.features, self.models

    def split_data(self) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
        """Split data into train, validation, and test sets"""
        print("\n" + "=" * 80)
        print("3️⃣  SPLITTING DATA")
        print("-" * 80)

        # Use USDJPY as reference for dates
        reference_pair = 'USDJPY'
        if reference_pair not in self.data:
            reference_pair = list(self.data.keys())[0]
            print(f"   ⚠️  Using {reference_pair} as reference (USDJPY not available)")

        all_dates = self.data[reference_pair].index
        n_dates = len(all_dates)

        # Calculate split points
        train_end = int(n_dates * TRAIN_RATIO)
        val_end = int(n_dates * (TRAIN_RATIO + VALIDATION_RATIO))

        train_dates = all_dates[:train_end]
        val_dates = all_dates[train_end:val_end]
        test_dates = all_dates[val_end:]

        print(f"\n📅 Data Split:")
        print(f"   ✓ Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates):,} days)")
        print(f"   ✓ Val:   {val_dates[0].date()} to {val_dates[-1].date()} ({len(val_dates):,} days)")
        print(f"   ✓ Test:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates):,} days)")

        return train_dates, val_dates, test_dates

    def train_ml_models(self, train_dates: pd.DatetimeIndex, val_dates: pd.DatetimeIndex) -> Dict:
        """Train LightGBM models for each currency pair"""
        print("\n" + "=" * 80)
        print("4️⃣  TRAINING MACHINE LEARNING MODELS")
        print("-" * 80)

        self.ml_models = {}

        for pair in CURRENCY_PAIRS:
            if pair not in self.features or self.features[pair] is None:
                print(f"\n   ⚠️  Skipping {pair}: No features available")
                continue

            print(f"\n🔧 Training LightGBM for {pair}...")

            try:
                # Get features and create target
                pair_features = self.features[pair]

                # Create target: next day return (classification: up/down)
                returns = self.data[pair]['spot'].pct_change().shift(-1)
                target = (returns > 0).astype(int)  # Binary classification

                # Align features and target
                combined = pd.concat([pair_features, target.rename('target')], axis=1)
                combined = combined.dropna()

                # Split data
                train_data = combined.loc[combined.index.intersection(train_dates)]
                val_data = combined.loc[combined.index.intersection(val_dates)]

                if len(train_data) < 100 or len(val_data) < 50:
                    print(f"   ⚠️  Insufficient data for {pair}")
                    continue

                # Separate features and target
                feature_cols = [col for col in train_data.columns if col != 'target']
                X_train = train_data[feature_cols]
                y_train = train_data['target']
                X_val = val_data[feature_cols]
                y_val = val_data['target']

                # Initialize and train model
                lgb_model = LightGBMModel(model_type='signal')
                lgb_model.train_signal_model(X_train, y_train, X_val, y_val)

                # Save model
                model_path = RESULTS_DIR / f'lgb_model_{pair}.pkl'
                lgb_model.save_model(str(model_path))
                self.ml_models[pair] = lgb_model

                print(f"   ✓ Model trained successfully")

                # Print top features
                if hasattr(lgb_model, 'feature_importance') and lgb_model.feature_importance is not None:
                    top_features = lgb_model.feature_importance.head(5)
                    print(f"   📊 Top 5 features:")
                    for idx, row in top_features.iterrows():
                        print(f"      - {row['feature']}: {row['importance']:.1f}")

            except Exception as e:
                print(f"   ❌ Error training model for {pair}: {str(e)}")
                continue

        print(f"\n✅ Successfully trained {len(self.ml_models)} ML models")
        return self.ml_models

    def initialize_multi_model_strategies(self) -> List:
        """Initialize all model-based trading strategies"""
        print("\n" + "=" * 80)
        print("5️⃣  INITIALIZING MULTI-MODEL STRATEGIES")
        print("-" * 80)

        self.strategies = []

        # 1. GVV Arbitrage Strategy
        gvv_strategy = GVVArbitrageStrategy(
            lookback_window=20,
            mispricing_threshold=1.5,
            enable_hedging=ENABLE_DELTA_HEDGING
        )
        self.strategies.append(gvv_strategy)
        print(f"   ✓ GVV Arbitrage Strategy")

        # 2. SABR Volatility Strategy
        sabr_strategy = SABRVolatilityStrategy(
            calibration_window=60,
            vol_threshold=2.0,
            enable_hedging=ENABLE_DELTA_HEDGING
        )
        self.strategies.append(sabr_strategy)
        print(f"   ✓ SABR Volatility Strategy")

        # 3. Model Consensus Strategy
        consensus_strategy = ModelConsensusStrategy(
            models=['gvv', 'sabr'],
            consensus_threshold=0.7,
            enable_hedging=ENABLE_DELTA_HEDGING
        )
        self.strategies.append(consensus_strategy)
        print(f"   ✓ Model Consensus Strategy")

        # 4. ML Signal Strategies (one per pair)
        if self.ml_models:
            for pair, ml_model in self.ml_models.items():
                ml_strategy = MLSignalStrategy(
                    ml_model=ml_model,
                    signal_threshold=0.6,
                    enable_hedging=ENABLE_DELTA_HEDGING
                )
                ml_strategy.name = f"ML_{pair}"
                self.strategies.append(ml_strategy)
                print(f"   ✓ ML Strategy for {pair}")

        # 5. Traditional strategies for comparison
        vol_arb = VolatilityArbitrageStrategy(
            lookback_window=20,
            zscore_threshold=1.5,
            enable_hedging=ENABLE_DELTA_HEDGING,
            hedge_threshold=HEDGE_THRESHOLD
        )
        self.strategies.append(vol_arb)
        print(f"   ✓ Volatility Arbitrage (Traditional)")

        carry_vol = CarryToVolStrategy(min_ratio=0.3)
        self.strategies.append(carry_vol)
        print(f"   ✓ Carry to Volatility")

        print(f"\n✅ Total strategies initialized: {len(self.strategies)}")
        return self.strategies

    def run_multi_model_backtest(self, test_dates: pd.DatetimeIndex) -> Tuple[Dict, Dict, pd.DataFrame]:
        """Run separate backtests for each strategy and combine results"""
        print("\n" + "=" * 80)
        print("6️⃣  RUNNING MULTI-MODEL BACKTEST")
        print("-" * 80)
        print(f"\n📅 Test Period: {test_dates[0].date()} to {test_dates[-1].date()}")
        print(f"💰 Initial Capital: ${INITIAL_CAPITAL:,.0f}")
        print(f"📉 Max Drawdown Limit: {MAX_DRAWDOWN:.1%}")

        all_results = {}
        all_trades = []

        # Run backtest for each strategy separately
        for i, strategy in enumerate(self.strategies, 1):
            print(f"\n[{i}/{len(self.strategies)}] Testing {strategy.name}...")

            try:
                # Initialize engine for this strategy
                engine = BacktestingEngine(initial_capital=INITIAL_CAPITAL)

                # Run backtest with single strategy
                backtest_results = engine.run_backtest(
                    strategies=[strategy],
                    data=self.data,
                    start_date=test_dates[0],
                    end_date=test_dates[-1],
                    rebalance_freq=REBALANCE_FREQUENCY
                )

                # Store results
                all_results[strategy.name] = backtest_results

                # Add strategy name to trades
                if 'trades' in backtest_results and len(backtest_results['trades']) > 0:
                    trades = backtest_results['trades'].copy()
                    trades['model'] = strategy.name
                    all_trades.append(trades)

                # Print summary for this strategy
                perf = backtest_results.get('performance', {})
                print(f"   📊 Results:")
                print(f"      - Return: {perf.get('total_return', 0):.2%}")
                print(f"      - Sharpe: {perf.get('sharpe_ratio', 0):.2f}")
                print(f"      - Max DD: {perf.get('max_drawdown', 0):.2%}")
                print(f"      - Trades: {perf.get('total_trades', 0)}")

            except Exception as e:
                print(f"   ❌ Error in {strategy.name}: {str(e)}")
                continue

        # Combine all trades
        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)
        else:
            combined_trades = pd.DataFrame()

        # Create combined portfolio
        combined_results = self.combine_strategy_results(all_results)

        print("\n" + "-" * 80)
        print("📊 COMBINED PORTFOLIO RESULTS:")
        combined_perf = combined_results.get('performance', {})
        print(f"   - Total Return: {combined_perf.get('total_return', 0):.2%}")
        print(f"   - Sharpe Ratio: {combined_perf.get('sharpe_ratio', 0):.2f}")
        print(f"   - Max Drawdown: {combined_perf.get('max_drawdown', 0):.2%}")

        return all_results, combined_results, combined_trades

    def combine_strategy_results(self, all_results: Dict) -> Dict:
        """Combine results from multiple strategies with equal weighting"""

        # Get all equity curves
        equity_curves = {}
        for name, result in all_results.items():
            if 'results' in result and len(result['results']) > 0:
                equity_curves[name] = result['results']['equity']

        if len(equity_curves) == 0:
            return {
                'results': pd.DataFrame(),
                'performance': {},
                'trades': pd.DataFrame()
            }

        # Align all equity curves to same dates
        equity_df = pd.DataFrame(equity_curves)
        equity_df = equity_df.fillna(method='ffill').fillna(INITIAL_CAPITAL)

        # Calculate combined equity (equal weight)
        combined_equity = equity_df.mean(axis=1)

        # Calculate returns
        combined_returns = combined_equity.pct_change().fillna(0)

        # Create combined results DataFrame
        combined_results_df = pd.DataFrame({
            'equity': combined_equity,
            'returns': combined_returns
        })

        # Calculate performance metrics
        analyzer = PerformanceAnalyzer()
        combined_metrics = analyzer.calculate_metrics(
            combined_returns,
            combined_equity
        )

        return {
            'results': combined_results_df,
            'performance': combined_metrics,
            'individual_results': all_results
        }

    def generate_comprehensive_visualizations(self, all_results: Dict, combined_results: Dict) -> None:
        """Generate all visualizations for multi-model analysis"""
        print("\n" + "=" * 80)
        print("7️⃣  GENERATING VISUALIZATIONS")
        print("-" * 80)

        # Create visualizers
        multi_viz = MultiModelVisualizer(output_dir=str(PLOTS_DIR))
        trading_viz = TradingVisualizer(results_dir=str(PLOTS_DIR))
        garch_viz = GARCHVisualizer(output_dir=str(PLOTS_DIR))

        print("\n📊 Creating multi-model visualizations...")

        try:
            # 1. Multi-strategy equity curves
            multi_viz.plot_multi_strategy_equity_curves(all_results)
            print("   ✓ Multi-strategy equity curves")
        except Exception as e:
            print(f"   ⚠️  Could not create equity curves: {e}")

        try:
            # 2. Performance comparison
            multi_viz.plot_performance_comparison(all_results)
            print("   ✓ Performance comparison")
        except Exception as e:
            print(f"   ⚠️  Could not create performance comparison: {e}")

        try:
            # 3. Strategy correlation matrix
            multi_viz.plot_strategy_correlations(all_results)
            print("   ✓ Strategy correlation matrix")
        except Exception as e:
            print(f"   ⚠️  Could not create correlation matrix: {e}")

        try:
            # 4. Model mispricing analysis
            multi_viz.plot_model_mispricings(all_results)
            print("   ✓ Model mispricing analysis")
        except Exception as e:
            print(f"   ⚠️  Could not create mispricing analysis: {e}")

        # 5. Combined portfolio analysis
        if len(combined_results.get('results', [])) > 0:
            try:
                trading_viz.plot_equity_curve(
                    combined_results['results'],
                    title="Combined Multi-Model Portfolio"
                )
                print("   ✓ Combined portfolio equity curve")

                trading_viz.plot_rolling_sharpe(combined_results['results'], window=60)
                print("   ✓ Rolling Sharpe ratio")

                trading_viz.plot_returns_distribution(combined_results['results'])
                print("   ✓ Returns distribution")
            except Exception as e:
                print(f"   ⚠️  Could not create combined portfolio plots: {e}")

        # 6. GARCH analysis
        try:
            garch_viz.create_comprehensive_report(self.data, self.features)
            print("   ✓ GARCH analysis report")
        except Exception as e:
            print(f"   ⚠️  Could not create GARCH report: {e}")

        # 7. Volatility surfaces for key pairs
        print("\n📊 Creating volatility surfaces...")
        for pair in ['USDJPY', 'EURUSD', 'GBPUSD']:
            if pair in self.data:
                try:
                    latest_date = self.data[pair].index[-1]
                    trading_viz.plot_volatility_surface(self.data[pair], pair, latest_date)
                    print(f"   ✓ {pair} volatility surface")
                except Exception as e:
                    print(f"   ⚠️  Could not create vol surface for {pair}: {e}")

    def save_all_results(self, all_results: Dict, combined_results: Dict, combined_trades: pd.DataFrame) -> None:
        """Save all results to files"""
        print("\n" + "=" * 80)
        print("8️⃣  SAVING RESULTS")
        print("-" * 80)

        # Save combined results
        if len(combined_results.get('results', [])) > 0:
            combined_results['results'].to_csv(RESULTS_DIR / 'combined_portfolio_results.csv')
            print(f"   ✓ Combined portfolio results saved")

        # Save individual strategy results
        for strategy_name, results in all_results.items():
            if 'results' in results and len(results['results']) > 0:
                filename = f"strategy_{strategy_name.replace(' ', '_').lower()}_results.csv"
                results['results'].to_csv(RESULTS_DIR / filename)

        print(f"   ✓ Individual strategy results saved")

        # Save all trades
        if len(combined_trades) > 0:
            combined_trades.to_csv(RESULTS_DIR / 'all_trades.csv', index=False)
            print(f"   ✓ All trades saved ({len(combined_trades):,} total)")

        # Save performance summary
        perf_summary = []
        for name, results in all_results.items():
            if 'performance' in results:
                perf = results['performance'].copy()
                perf['strategy'] = name
                perf_summary.append(perf)

        if perf_summary:
            perf_df = pd.DataFrame(perf_summary)
            perf_df.to_csv(RESULTS_DIR / 'performance_summary.csv', index=False)
            print(f"   ✓ Performance summary saved")

        # Save ML models metadata
        if self.ml_models:
            ml_metadata = {
                pair: {
                    'trained': True,
                    'model_path': str(RESULTS_DIR / f'lgb_model_{pair}.pkl')
                }
                for pair in self.ml_models.keys()
            }
            pd.DataFrame(ml_metadata).T.to_csv(RESULTS_DIR / 'ml_models_metadata.csv')
            print(f"   ✓ ML models metadata saved")

    def run(self) -> Tuple[Dict, Dict]:
        """Main execution function"""
        try:
            # 1. Initialize system
            if not self.initialize_system():
                raise RuntimeError("System initialization failed")

            # 2. Load and prepare data
            self.loader, self.data = self.load_and_prepare_data()

            # 3. Create enhanced features
            self.features, self.models = self.create_enhanced_features()

            # 4. Split data
            train_dates, val_dates, test_dates = self.split_data()

            # 5. Train ML models
            self.ml_models = self.train_ml_models(train_dates, val_dates)

            # 6. Initialize multi-model strategies
            self.strategies = self.initialize_multi_model_strategies()

            # 7. Run multi-model backtest
            all_results, combined_results, combined_trades = self.run_multi_model_backtest(test_dates)

            # 8. Generate visualizations
            self.generate_comprehensive_visualizations(all_results, combined_results)

            # 9. Save results
            self.save_all_results(all_results, combined_results, combined_trades)

            # Final summary
            self.print_final_summary(combined_results)

            return all_results, combined_results

        except Exception as e:
            print(f"\n❌ Critical error in system execution: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def print_final_summary(self, combined_results: Dict) -> None:
        """Print final execution summary"""
        print("\n" + "=" * 80)
        print(" ✅ FX OPTIONS TRADING SYSTEM EXECUTION COMPLETE ".center(80))
        print("=" * 80)

        perf = combined_results.get('performance', {})

        print(f"\n📊 FINAL PORTFOLIO SUMMARY:")
        print(f"   💰 Total Return: {perf.get('total_return', 0):.2%}")
        print(f"   📈 Annualized Return: {perf.get('annualized_return', 0):.2%}")
        print(f"   📊 Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
        print(f"   📉 Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
        print(f"   🎯 Win Rate: {perf.get('win_rate', 0):.1%}")

        # Check constraints
        print(f"\n✅ CONSTRAINT VALIDATION:")
        if abs(perf.get('max_drawdown', 0)) < MAX_DRAWDOWN:
            print(f"   ✓ Drawdown constraint SATISFIED (limit: {MAX_DRAWDOWN:.1%})")
        else:
            print(f"   ❌ Drawdown constraint VIOLATED (limit: {MAX_DRAWDOWN:.1%})")

        if perf.get('sharpe_ratio', 0) > 1.0:
            print(f"   ✓ Sharpe ratio > 1.0 ACHIEVED")
        else:
            print(f"   ⚠️  Sharpe ratio below target of 1.0")

        print(f"\n📁 Results saved to:")
        print(f"   - Plots: {PLOTS_DIR}")
        print(f"   - Data: {RESULTS_DIR}")

        print(f"\n⏱️  Completion time: {datetime.now()}")


def main():
    """Main entry point"""
    system = FXOptionsSystem()
    all_results, combined_results = system.run()
    return all_results, combined_results


if __name__ == "__main__":
    all_results, combined_results = main()