# src/backtesting/engine.py
"""
Fixed backtesting engine using skfolio with proper data handling
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import skfolio components
try:
    from skfolio.optimization import MeanRisk, EqualWeighted
    from skfolio.measures import RiskMeasure, RatioMeasure
    from skfolio import Population, Portfolio
    from skfolio.preprocessing import prices_to_returns
    SKFOLIO_AVAILABLE = True
except ImportError:
    SKFOLIO_AVAILABLE = False
    print("Warning: skfolio not available. Using simplified portfolio optimization.")

class BacktestingEngine:
    """
    Backtesting engine for FX options strategies
    """

    def __init__(self, initial_capital: float = 10000000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.returns = []
        self.daily_pnl = []

    def run_backtest(self, strategies: List, data: Dict[str, pd.DataFrame],
                     start_date: pd.Timestamp, end_date: pd.Timestamp,
                     rebalance_freq: str = 'D') -> Dict:
        """
        Run backtest for multiple strategies
        Fixed version with proper data alignment
        """
        # Get trading dates
        all_dates = data[list(data.keys())[0]].index
        trading_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]

        if len(trading_dates) == 0:
            print(f"Warning: No trading dates found between {start_date} and {end_date}")
            return {
                'results': pd.DataFrame(),
                'performance': {},
                'trades': pd.DataFrame()
            }

        # Initialize results storage with lists
        dates_list = []
        equity_list = []
        returns_list = []
        positions_list = []
        signals_list = []

        # Initialize capital
        current_capital = self.initial_capital
        prev_capital = self.initial_capital

        # Rebalance dates
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)

        print(f"  Running backtest from {trading_dates[0].date()} to {trading_dates[-1].date()}")
        print(f"  Total trading days: {len(trading_dates)}")

        for i, date in enumerate(trading_dates):
            daily_pnl = 0
            all_signals = []

            # Generate signals from all strategies
            for strategy in strategies:
                try:
                    signals = strategy.generate_signals(data, date)
                    all_signals.extend(signals)
                except Exception as e:
                    print(f"  ⚠️ Strategy {strategy.name} failed on {date.date()}: {e}")
                    # Continue if strategy fails for this date
                    pass

            # Portfolio optimization using skfolio (if available and rebalance date)
            if date in rebalance_dates and len(all_signals) > 0:
                if SKFOLIO_AVAILABLE:
                    # Create portfolio from signals
                    portfolio_returns = self._signals_to_returns(all_signals, data, date)

                    if portfolio_returns is not None and len(portfolio_returns.columns) > 1:
                        # Use Mean-Risk optimization
                        model = MeanRisk(
                            risk_measure=RiskMeasure.VARIANCE,
                            risk_free_rate=0.02
                        )

                        try:
                            portfolio = model.fit(portfolio_returns)
                            weights = portfolio.weights_

                            # Execute trades based on weights
                            self._execute_trades(all_signals, weights, date, data)
                        except:
                            # Fall back to equal weighting if optimization fails
                            weights = np.ones(len(all_signals)) / len(all_signals)
                            self._execute_trades(all_signals, weights, date, data)
                else:
                    # Simple equal weighting if skfolio not available
                    if len(all_signals) > 0:
                        weights = np.ones(len(all_signals)) / len(all_signals)
                        self._execute_trades(all_signals, weights, date, data)

            # Update positions and calculate P&L
            daily_pnl = self._update_positions(data, date)
            current_capital += daily_pnl

            # Calculate return
            if prev_capital != 0:
                daily_return = daily_pnl / prev_capital
            else:
                daily_return = 0

            # Record results for this date
            dates_list.append(date)
            equity_list.append(current_capital)
            returns_list.append(daily_return)
            positions_list.append(len(self.positions))
            signals_list.append(len(all_signals))

            # Update previous capital
            prev_capital = current_capital

            # Check drawdown constraint
            if current_capital < self.initial_capital * 0.9:
                print(f"  ⚠️ Drawdown limit reached on {date.date()}. Stopping backtest.")
                break

            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(trading_dates)} days...")

        # Create results DataFrame
        if len(dates_list) > 0:
            results_df = pd.DataFrame({
                'dates': dates_list,
                'equity': equity_list,
                'returns': returns_list,
                'positions': positions_list,
                'signals': signals_list
            })
            results_df.set_index('dates', inplace=True)
        else:
            results_df = pd.DataFrame()

        # Calculate performance metrics
        if len(results_df) > 0:
            performance = self.calculate_performance_metrics(results_df)
        else:
            performance = {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0
            }

        # Create trades DataFrame
        if len(self.closed_trades) > 0:
            trades_df = pd.DataFrame(self.closed_trades)
        else:
            trades_df = pd.DataFrame()

        return {
            'results': results_df,
            'performance': performance,
            'trades': trades_df
        }

    def _signals_to_returns(self, signals: List, data: Dict[str, pd.DataFrame],
                           date: pd.Timestamp, lookback: int = 60) -> Optional[pd.DataFrame]:
        """
        Convert signals to expected returns for portfolio optimization
        """
        if len(signals) == 0:
            return None

        returns_data = {}

        for i, signal in enumerate(signals):
            pair_data = data.get(signal.pair)
            if pair_data is None:
                continue

            # Get historical returns for this signal
            try:
                end_idx = pair_data.index.get_loc(date)
            except KeyError:
                continue

            start_idx = max(0, end_idx - lookback)

            if start_idx >= end_idx:
                continue

            # Calculate returns based on signal type
            hist_data = pair_data.iloc[start_idx:end_idx]

            if len(hist_data) < 2:
                continue

            # Simple returns based on spot movement
            returns = hist_data['spot'].pct_change().dropna()

            # Adjust returns by signal direction
            returns = returns * signal.direction

            # Store with unique identifier
            returns_data[f"{signal.pair}_{signal.tenor}_{i}"] = returns

        if len(returns_data) == 0:
            return None

        # Create DataFrame
        returns_df = pd.DataFrame(returns_data)

        # Handle missing data
        returns_df = returns_df.fillna(0)

        # Need at least 2 observations for optimization
        returns_df = returns_df.dropna(how='all')

        return returns_df if len(returns_df) > 1 else None

    def _execute_trades(self, signals: List, weights: np.ndarray,
                       date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> None:
        """
        Execute trades based on signals and weights
        """
        for signal, weight in zip(signals, weights):
            if weight > 0.01:  # Minimum weight threshold
                position_size = self.capital * weight

                pair_data = data.get(signal.pair)
                if pair_data is None or date not in pair_data.index:
                    continue

                # Create position (simplified)
                position = {
                    'pair': signal.pair,
                    'tenor': signal.tenor,
                    'direction': signal.direction,
                    'size': position_size,
                    'entry_date': date,
                    'entry_price': pair_data.loc[date, 'spot'],
                    'strategy': signal.strategy_name
                }

                self.positions.append(position)

    def _update_positions(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
        """
        Update positions and calculate P&L
        """
        daily_pnl = 0
        positions_to_close = []

        for i, position in enumerate(self.positions):
            pair_data = data.get(position['pair'])

            if pair_data is None or date not in pair_data.index:
                continue

            current_price = pair_data.loc[date, 'spot']

            # Simple P&L calculation
            price_change = (current_price - position['entry_price']) / position['entry_price']
            position_pnl = position['size'] * price_change * position['direction']

            # Check exit conditions (simplified)
            days_held = (date - position['entry_date']).days

            # Exit after tenor expires or stop loss (3% loss)
            tenor_days = {'1W': 7, '2W': 14, '1M': 30, '3M': 90, '6M': 180, '1Y': 365}
            max_days = tenor_days.get(position['tenor'], 30)

            if days_held > max_days or position_pnl < -position['size'] * 0.03:
                positions_to_close.append(i)
                self.closed_trades.append({
                    **position,
                    'exit_date': date,
                    'exit_price': current_price,
                    'pnl': position_pnl,
                    'days_held': days_held
                })
                daily_pnl += position_pnl

        # Remove closed positions (in reverse order to maintain indices)
        for i in sorted(positions_to_close, reverse=True):
            del self.positions[i]

        return daily_pnl

    def calculate_performance_metrics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics
        """
        if len(results_df) == 0:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_trades': 0
            }

        returns = results_df['returns'].fillna(0)
        equity = results_df['equity']

        # Calculate metrics
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital

        # Sharpe ratio
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0

        # Max drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Annualized return
        n_days = len(returns)
        if n_days > 0:
            annualized_return = (1 + total_return) ** (252 / n_days) - 1
        else:
            annualized_return = 0

        # Win rate and trade statistics
        if len(self.closed_trades) > 0:
            trades_df = pd.DataFrame(self.closed_trades)
            win_rate = (trades_df['pnl'] > 0).mean()

            wins = trades_df[trades_df['pnl'] > 0]['pnl']
            losses = trades_df[trades_df['pnl'] < 0]['pnl']

            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0
            total_trades = len(trades_df)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            total_trades = 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': total_trades
        }