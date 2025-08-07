"""
Main execution script for FX Options Trading System
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Import configurations
from config.settings import *

# Import modules
from src.data_loader import FXDataLoader
from src.strategies.volatility_arbitrage import VolatilityArbitrageStrategy
from src.strategies.carry_strategy import CarryToVolStrategy
from src.backtesting.engine import BacktestingEngine
from src.visualization.plots import TradingVisualizer
from src.models.ml_models import LightGBMModel
from src.features_enhanced import EnhancedFeatureEngineer
from src.models.garch_models import GARCHForecaster, VolatilityComparator
from src.visualization.garch_plots import GARCHVisualizer

def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("FX OPTIONS TRADING SYSTEM")
    print("=" * 60)

    # 1. Load Data
    print("\n1. Loading data...")
    loader = FXDataLoader(FX_DATA_PATH, DISCOUNT_CURVES_PATH)
    loader.load_data()
    data = loader.process_all_pairs()

    # 2. Split data for train/test
    print("\n2. Splitting data...")
    all_dates = data['USDJPY'].index
    n_dates = len(all_dates)

    train_end = int(n_dates * TRAIN_RATIO)
    val_end = int(n_dates * (TRAIN_RATIO + VALIDATION_RATIO))

    train_dates = all_dates[:train_end]
    val_dates = all_dates[train_end:val_end]
    test_dates = all_dates[val_end:]

    print(f"Train: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
    print(f"Validation: {val_dates[0]} to {val_dates[-1]} ({len(val_dates)} days)")
    print(f"Test: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

    # 3. Train ML Model (LightGBM)
    print("\n3. Training LightGBM model...")
    lgb_model = LightGBMModel(model_type='signal')

    # Prepare features for one pair as example
    pair_data = data['USDJPY']
    features = lgb_model.prepare_features(pair_data, 'USDJPY')

    # Create target (next day return)
    target = pair_data['spot'].pct_change().shift(-1)

    # Split features
    X_train = features.loc[train_dates].dropna()
    y_train = target.loc[X_train.index]
    X_val = features.loc[val_dates].dropna()
    y_val = target.loc[X_val.index]

    # Train model
    lgb_model.train_signal_model(X_train, y_train, X_val, y_val)

    # Save model
    lgb_model.save_model(str(RESULTS_DIR / 'lgb_model'))
    print("LightGBM model trained and saved")

    # 4. Initialize Strategies
    print("\n4. Initializing strategies...")
    strategies = [
        VolatilityArbitrageStrategy(lookback_window=20, zscore_threshold=2.0),
        CarryToVolStrategy(min_ratio=0.5)
    ]

    # 5. Run Backtest
    print("\n5. Running backtest...")
    engine = BacktestingEngine(initial_capital=INITIAL_CAPITAL)

    backtest_results = engine.run_backtest(
        strategies=strategies,
        data=data,
        start_date=test_dates[0],
        end_date=test_dates[-1],
        rebalance_freq=REBALANCE_FREQUENCY
    )

    # 6. Generate Visualizations
    print("\n6. Generating visualizations...")
    visualizer = TradingVisualizer(results_dir=str(PLOTS_DIR))

    # Plot equity curve
    visualizer.plot_equity_curve(backtest_results['results'])

    # Plot rolling Sharpe
    visualizer.plot_rolling_sharpe(backtest_results['results'])

    # Plot returns distribution
    visualizer.plot_returns_distribution(backtest_results['results'])

    # Plot strategy performance
    if len(backtest_results['trades']) > 0:
        visualizer.plot_strategy_performance(backtest_results['trades'])

    # Plot volatility surface for latest date
    latest_date = test_dates[-1]
    for pair in CURRENCY_PAIRS:
        visualizer.plot_volatility_surface(data[pair], pair, latest_date)

    # 7. Generate Performance Report
    print("\n7. Generating performance report...")
    visualizer.create_performance_report(
        backtest_results['performance'],
        backtest_results['trades']
    )

    # 8. Print Summary
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)

    perf = backtest_results['performance']
    print(f"Total Return: {perf['total_return']:.2%}")
    print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {perf['max_drawdown']:.2%}")
    print(f"Win Rate: {perf['win_rate']:.2%}")
    print(f"Total Trades: {perf['total_trades']}")

    print("\nAll results saved to:")
    print(f"  - Plots: {PLOTS_DIR}")
    print(f"  - Reports: {RESULTS_DIR}")

    print("\n✅ FX Options Trading System execution complete!")


if __name__ == "__main__":
    main()