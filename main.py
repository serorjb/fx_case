# main.py
"""
Main execution script for FX Options Trading System
Complete version with all enhancements including GARCH, HMM, and Delta Hedging
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
warnings.filterwarnings('ignore')

# Import configurations
from config.settings import *

# Import data and curves modules
from src.data_loader import FXDataLoader
from src.curves import DiscountCurveBuilder, process_discount_curves

# Import feature engineering modules
from src.features_enhanced import EnhancedFeatureEngineer
from src.models.garch_models import create_time_features

# Import models
from src.models.gk_model import GarmanKohlhagen
from src.models.gvv_model import GVVModel
from src.models.sabr_model import SABRModel
from src.models.ml_models import LightGBMModel

# Import strategies
from src.strategies.volatility_arbitrage import VolatilityArbitrageStrategy
from src.strategies.carry_strategy import CarryToVolStrategy

# Import backtesting and risk management
from src.backtesting.engine import BacktestingEngine
from src.backtesting.performance import PerformanceAnalyzer
from src.risk_management import RiskManager
from src.hedging import DeltaHedger

# Import visualization
from src.visualization.plots import TradingVisualizer
from src.visualization.garch_plots import GARCHVisualizer

def initialize_system():
    """
    Initialize all system components
    """
    print("=" * 60)
    print("FX OPTIONS TRADING SYSTEM - ENHANCED VERSION")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    print(f"Configuration: {CURVE_METHOD}, HMM States: {HMM_STATES}")
    print(f"Delta Hedging: {ENABLE_DELTA_HEDGING}")
    print("-" * 60)

    # Create necessary directories
    PLOTS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Check for data files
    if not FX_DATA_PATH.exists():
        raise FileNotFoundError(f"FX data not found at {FX_DATA_PATH}")

    # Process discount curves if needed
    if not DISCOUNT_CURVES_PATH.exists():
        print("\n📊 Processing discount curves...")
        process_discount_curves(
            data_dir=DATA_DIR / "FRED",
            output_dir=DATA_DIR,
            plot_dir=PLOTS_DIR
        )

    return True

def load_and_prepare_data():
    """
    Load FX data and discount curves
    """
    print("\n1️⃣ Loading and preparing data...")

    # Initialize data loader
    loader = FXDataLoader(FX_DATA_PATH, DISCOUNT_CURVES_PATH)
    loader.load_data()

    # Process all currency pairs
    data = loader.process_all_pairs()

    # Data quality check
    for pair in CURRENCY_PAIRS:
        if pair not in data:
            print(f"  ⚠️ Warning: {pair} not found in data")
            continue

        pair_data = data[pair]
        print(f"  ✓ {pair}: {pair_data.shape[0]} days, {pair_data.shape[1]} features")

        # Check for critical columns
        required_cols = ['spot', 'atm_vol_1M', 'rr_25_1M', 'bf_25_1M']
        missing = [col for col in required_cols if col not in pair_data.columns]
        if missing:
            print(f"    ⚠️ Missing columns: {missing}")

    return loader, data

def create_enhanced_features(data, loader):
    """
    Create all enhanced features including GARCH and HMM
    """
    print("\n2️⃣ Creating enhanced features...")

    # Initialize pricing models for feature generation
    models = {
        'gk': GarmanKohlhagen(),
        'gvv': GVVModel(),
        'sabr': SABRModel()
    }

    # Initialize enhanced feature engineer
    enhanced_fe = EnhancedFeatureEngineer()

    # Create features for all pairs
    print("  Creating GARCH, HMM, and model-based features...")
    all_features = enhanced_fe.create_all_enhanced_features(data, models)

    # Print feature summary
    for pair, features in all_features.items():
        print(f"  ✓ {pair}: {features.shape[1]} features created")

        # Check for key feature groups
        garch_features = [col for col in features.columns if 'garch' in col.lower()]
        hmm_features = [col for col in features.columns if 'hmm' in col.lower() or 'regime' in col.lower()]
        time_features = [col for col in features.columns if 'day' in col.lower() or 'month' in col.lower()]

        print(f"    - GARCH features: {len(garch_features)}")
        print(f"    - HMM features: {len(hmm_features)}")
        print(f"    - Time features: {len(time_features)}")

    return all_features, models

def split_data(data, features):
    """
    Split data into train, validation, and test sets
    """
    print("\n3️⃣ Splitting data...")

    # Use USDJPY as reference for dates
    reference_pair = 'USDJPY'
    if reference_pair not in data:
        reference_pair = list(data.keys())[0]

    all_dates = data[reference_pair].index
    n_dates = len(all_dates)

    # Calculate split points
    train_end = int(n_dates * TRAIN_RATIO)
    val_end = int(n_dates * (TRAIN_RATIO + VALIDATION_RATIO))

    train_dates = all_dates[:train_end]
    val_dates = all_dates[train_end:val_end]
    test_dates = all_dates[val_end:]

    print(f"  ✓ Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} days)")
    print(f"  ✓ Val:   {val_dates[0].date()} to {val_dates[-1].date()} ({len(val_dates)} days)")
    print(f"  ✓ Test:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")

    return train_dates, val_dates, test_dates

def train_ml_models(data, features, train_dates, val_dates):
    """
    Train LightGBM models for each currency pair
    """
    print("\n4️⃣ Training machine learning models...")

    ml_models = {}

    for pair in CURRENCY_PAIRS:
        if pair not in features:
            continue

        print(f"  Training LightGBM for {pair}...")

        # Get features and target
        pair_features = features[pair]

        # Create target: next day return
        target = data[pair]['spot'].pct_change().shift(-1)

        # Align features and target
        combined = pd.concat([pair_features, target.rename('target')], axis=1)
        combined = combined.dropna()

        # Split data
        train_data = combined.loc[combined.index.intersection(train_dates)]
        val_data = combined.loc[combined.index.intersection(val_dates)]

        if len(train_data) < 100 or len(val_data) < 50:
            print(f"    ⚠️ Insufficient data for {pair}")
            continue

        # Separate features and target
        feature_cols = [col for col in train_data.columns if col != 'target']
        X_train = train_data[feature_cols]
        y_train = train_data['target']
        X_val = val_data[feature_cols]
        y_val = val_data['target']

        # Initialize and train model
        lgb_model = LightGBMModel(model_type='signal')

        try:
            lgb_model.train_signal_model(X_train, y_train, X_val, y_val)

            # Save model
            model_path = RESULTS_DIR / f'lgb_model_{pair}'
            lgb_model.save_model(str(model_path))

            ml_models[pair] = lgb_model

            # Print feature importance (top 10)
            if lgb_model.feature_importance is not None:
                top_features = lgb_model.feature_importance.head(10)
                print(f"    Top features for {pair}:")
                for idx, row in top_features.iterrows():
                    print(f"      - {row['feature']}: {row['importance']:.2f}")

        except Exception as e:
            print(f"    ❌ Error training model for {pair}: {e}")

    return ml_models

def initialize_strategies():
    """
    Initialize trading strategies
    """
    print("\n5️⃣ Initializing trading strategies...")

    strategies = []

    # Volatility Arbitrage with delta hedging
    vol_arb = VolatilityArbitrageStrategy(
        lookback_window=20,
        zscore_threshold=2.0,
        enable_hedging=ENABLE_DELTA_HEDGING,
        hedge_threshold=HEDGE_THRESHOLD
    )
    strategies.append(vol_arb)
    print(f"  ✓ Volatility Arbitrage (hedging: {ENABLE_DELTA_HEDGING})")

    # Carry to Volatility strategy
    carry_vol = CarryToVolStrategy(min_ratio=0.5)
    strategies.append(carry_vol)
    print(f"  ✓ Carry to Volatility")

    return strategies

def run_backtest(strategies, data, test_dates):
    """
    Run backtesting engine
    """
    print("\n6️⃣ Running backtest...")
    print(f"  Period: {test_dates[0].date()} to {test_dates[-1].date()}")
    print(f"  Initial capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Max drawdown limit: {MAX_DRAWDOWN:.1%}")

    # Initialize backtesting engine
    engine = BacktestingEngine(initial_capital=INITIAL_CAPITAL)

    # Run backtest
    backtest_results = engine.run_backtest(
        strategies=strategies,
        data=data,
        start_date=test_dates[0],
        end_date=test_dates[-1],
        rebalance_freq=REBALANCE_FREQUENCY
    )

    # Print summary statistics
    perf = backtest_results['performance']
    print("\n  Backtest Results:")
    print(f"  ✓ Total Return: {perf.get('total_return', 0):.2%}")
    print(f"  ✓ Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
    print(f"  ✓ Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
    print(f"  ✓ Win Rate: {perf.get('win_rate', 0):.2%}")
    print(f"  ✓ Total Trades: {perf.get('total_trades', 0)}")

    # Check if drawdown limit was breached
    if abs(perf.get('max_drawdown', 0)) > MAX_DRAWDOWN:
        print(f"  ⚠️ Warning: Drawdown limit breached!")

    return backtest_results

def generate_visualizations(backtest_results, data, features):
    """
    Generate all visualizations
    """
    print("\n7️⃣ Generating visualizations...")

    # Standard trading visualizations
    visualizer = TradingVisualizer(results_dir=str(PLOTS_DIR))

    # Equity curve
    visualizer.plot_equity_curve(backtest_results['results'])
    print("  ✓ Equity curve")

    # Rolling Sharpe
    visualizer.plot_rolling_sharpe(backtest_results['results'], window=60)
    print("  ✓ Rolling Sharpe ratio")

    # Returns distribution
    visualizer.plot_returns_distribution(backtest_results['results'])
    print("  ✓ Returns distribution")

    # Strategy performance
    if len(backtest_results.get('trades', [])) > 0:
        visualizer.plot_strategy_performance(backtest_results['trades'])
        print("  ✓ Strategy performance")

    # Volatility surfaces
    latest_date = backtest_results['results'].index[-1]
    for pair in CURRENCY_PAIRS:
        if pair in data:
            try:
                visualizer.plot_volatility_surface(data[pair], pair, latest_date)
                print(f"  ✓ Volatility surface for {pair}")
            except Exception as e:
                print(f"  ⚠️ Could not create vol surface for {pair}: {e}")

    # GARCH visualizations
    garch_viz = GARCHVisualizer(output_dir=str(PLOTS_DIR))
    garch_viz.create_comprehensive_report(data, features)
    print("  ✓ GARCH analysis plots")

    # Performance report
    visualizer.create_performance_report(
        backtest_results['performance'],
        backtest_results.get('trades', pd.DataFrame())
    )
    print("  ✓ Performance report")

def analyze_results(backtest_results, strategies):
    """
    Perform detailed analysis of results
    """
    print("\n8️⃣ Analyzing results...")

    # Performance analyzer
    analyzer = PerformanceAnalyzer()

    # Calculate detailed metrics
    returns = backtest_results['results']['returns'].fillna(0)
    equity = backtest_results['results']['equity']
    trades = backtest_results.get('trades', pd.DataFrame())

    detailed_metrics = analyzer.calculate_metrics(returns, equity, trades)

    print("\n  Detailed Performance Metrics:")
    print("  " + "-" * 40)

    # Return metrics
    print("  Returns:")
    print(f"    Total Return: {detailed_metrics.get('total_return', 0):.2%}")
    print(f"    Annualized Return: {detailed_metrics.get('annualized_return', 0):.2%}")
    print(f"    Volatility: {detailed_metrics.get('volatility', 0):.2%}")

    # Risk metrics
    print("  Risk Metrics:")
    print(f"    Sharpe Ratio: {detailed_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"    Sortino Ratio: {detailed_metrics.get('sortino_ratio', 0):.3f}")
    print(f"    Calmar Ratio: {detailed_metrics.get('calmar_ratio', 0):.3f}")
    print(f"    Max Drawdown: {detailed_metrics.get('max_drawdown', 0):.2%}")
    print(f"    Max DD Duration: {detailed_metrics.get('max_drawdown_duration', 0)} days")

    # Risk measures
    print("  Risk Measures:")
    print(f"    VaR (95%): {detailed_metrics.get('var_95', 0):.3%}")
    print(f"    CVaR (95%): {detailed_metrics.get('cvar_95', 0):.3%}")
    print(f"    Skewness: {detailed_metrics.get('skewness', 0):.3f}")
    print(f"    Kurtosis: {detailed_metrics.get('kurtosis', 0):.3f}")

    # Trading statistics
    if len(trades) > 0:
        print("  Trading Statistics:")
        print(f"    Total Trades: {detailed_metrics.get('total_trades', 0)}")
        print(f"    Win Rate: {detailed_metrics.get('win_rate', 0):.2%}")
        print(f"    Avg Win: ${detailed_metrics.get('avg_win', 0):,.2f}")
        print(f"    Avg Loss: ${detailed_metrics.get('avg_loss', 0):,.2f}")
        print(f"    Profit Factor: {detailed_metrics.get('profit_factor', 0):.2f}")
        print(f"    Expectancy: ${detailed_metrics.get('expectancy', 0):,.2f}")

        # Strategy breakdown
        if 'strategy' in trades.columns:
            print("\n  Strategy Breakdown:")
            for strategy in trades['strategy'].unique():
                strategy_trades = trades[trades['strategy'] == strategy]
                strategy_pnl = strategy_trades['pnl'].sum()
                strategy_count = len(strategy_trades)
                strategy_winrate = (strategy_trades['pnl'] > 0).mean()

                print(f"    {strategy}:")
                print(f"      Trades: {strategy_count}")
                print(f"      Total P&L: ${strategy_pnl:,.2f}")
                print(f"      Win Rate: {strategy_winrate:.2%}")

    # Check hedging statistics if enabled
    if ENABLE_DELTA_HEDGING:
        print("\n  Hedging Statistics:")
        for strategy in strategies:
            if hasattr(strategy, 'hedge_history') and strategy.hedge_history:
                total_hedges = sum([len(h.get('hedges_executed', []))
                                  for h in strategy.hedge_history])
                total_hedge_cost = sum([h.get('hedge_cost', 0)
                                      for h in strategy.hedge_history])

                print(f"    {strategy.name}:")
                print(f"      Total Hedges: {total_hedges}")
                print(f"      Total Hedge Cost: ${total_hedge_cost:,.2f}")

    return detailed_metrics

def save_results(backtest_results, detailed_metrics, features):
    """
    Save all results to files
    """
    print("\n9️⃣ Saving results...")

    # Save backtest results
    results_df = backtest_results['results']
    results_df.to_csv(RESULTS_DIR / 'backtest_results.csv')
    print(f"  ✓ Backtest results saved to {RESULTS_DIR / 'backtest_results.csv'}")

    # Save trades
    if 'trades' in backtest_results and len(backtest_results['trades']) > 0:
        backtest_results['trades'].to_csv(RESULTS_DIR / 'trades.csv', index=False)
        print(f"  ✓ Trades saved to {RESULTS_DIR / 'trades.csv'}")

    # Save performance metrics
    metrics_df = pd.DataFrame([detailed_metrics])
    metrics_df.to_csv(RESULTS_DIR / 'performance_metrics.csv', index=False)
    print(f"  ✓ Performance metrics saved to {RESULTS_DIR / 'performance_metrics.csv'}")

    # Save feature importance for each pair
    for pair, pair_features in features.items():
        # Save sample of features for analysis
        pair_features.tail(100).to_csv(RESULTS_DIR / f'features_sample_{pair}.csv')

    print(f"  ✓ Feature samples saved")

def main():
    """
    Main execution function
    """
    try:
        # Initialize system
        initialize_system()

        # Load data
        loader, data = load_and_prepare_data()

        # Create enhanced features
        features, models = create_enhanced_features(data, loader)

        # Split data
        train_dates, val_dates, test_dates = split_data(data, features)

        # Train ML models
        ml_models = train_ml_models(data, features, train_dates, val_dates)

        # Initialize strategies
        strategies = initialize_strategies()

        # Run backtest
        backtest_results = run_backtest(strategies, data, test_dates)

        # Generate visualizations
        generate_visualizations(backtest_results, data, features)

        # Analyze results
        detailed_metrics = analyze_results(backtest_results, strategies)

        # Save results
        save_results(backtest_results, detailed_metrics, features)

        # Final summary
        print("\n" + "=" * 60)
        print("✅ FX OPTIONS TRADING SYSTEM EXECUTION COMPLETE")
        print("=" * 60)

        print(f"\n📊 Final Summary:")
        print(f"  Total Return: {backtest_results['performance']['total_return']:.2%}")
        print(f"  Sharpe Ratio: {backtest_results['performance']['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {backtest_results['performance']['max_drawdown']:.2%}")

        if abs(backtest_results['performance']['max_drawdown']) < MAX_DRAWDOWN:
            print(f"  ✅ Drawdown constraint satisfied")
        else:
            print(f"  ❌ Drawdown constraint violated")

        print(f"\n📁 Results saved to:")
        print(f"  - Plots: {PLOTS_DIR}")
        print(f"  - Data: {RESULTS_DIR}")

        print(f"\n⏱️ Execution time: {datetime.now()}")

        return backtest_results, detailed_metrics

    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    backtest_results, metrics = main()
