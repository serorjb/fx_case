"""
Backtesting engine using skfolio
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Import skfolio components
from skfolio.optimization import MeanRisk, EqualWeighted
from skfolio.measures import RiskMeasure, RatioMeasure
from skfolio import Population, Portfolio
from skfolio.preprocessing import prices_to_returns


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
                     rebalance_freq: str = 'W') -> Dict:
        """
        Run backtest for multiple strategies
        """
        # Get trading dates
        all_dates = data[list(data.keys())[0]].index
        trading_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]

        # Initialize results storage
        results = {
            'dates': [],
            'equity': [],
            'returns': [],
            'positions': [],
            'signals': [],
            'trades': []
        }

        # Rebalance dates
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)

        for date in trading_dates:
            daily_pnl = 0
            all_signals = []

            # Generate signals from all strategies
            for strategy in strategies:
                signals = strategy.generate_signals(data, date)
                all_signals.extend(signals)

            # Portfolio optimization using skfolio
            if date in rebalance_dates and len(all_signals) > 0:
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

            # Update positions and calculate P&L
            daily_pnl = self._update_positions(data, date)
            self.capital += daily_pnl

            # Record results
            results['dates'].append(date)
            results['equity'].append(self.capital)
            results['returns'].append(daily_pnl / (self.capital - daily_pnl) if self.capital != daily_pnl else 0)
            results['positions'].append(len(self.positions))
            results['signals'].append(len(all_signals))

            # Check drawdown constraint
            if self.capital < self.initial_capital * 0.9:
                print(f"Drawdown limit reached on {date}. Stopping backtest.")
                break

        # Calculate performance metrics
        results_df = pd.DataFrame(results)
        results_df.set_index('dates', inplace=True)

        performance = self.calculate_performance_metrics(results_df)

        return {
            'results': results_df,
            'performance': performance,
            'trades': pd.DataFrame(self.closed_trades)
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
            pair_data = data[signal.pair]

            # Get historical returns for this signal
            end_idx = pair_data.index.get_loc(date)
            start_idx = max(0, end_idx - lookback)

            if start_idx >= end_idx:
                continue

            # Calculate returns based on signal type
            hist_data = pair_data.iloc[start_idx:end_idx]

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
        returns_df = returns_df.dropna()

        return returns_df if len(returns_df) > 0 else None

    def _execute_trades(self, signals: List, weights: np.ndarray,
                        date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> None:
        """
        Execute trades based on signals and weights
        """
        for signal, weight in zip(signals, weights):
            if weight > 0.01:  # Minimum weight threshold
                position_size = self.capital * weight

                # Create position (simplified)
                position = {
                    'pair': signal.pair,
                    'tenor': signal.tenor,
                    'direction': signal.direction,
                    'size': position_size,
                    'entry_date': date,
                    'entry_price': data[signal.pair].loc[date, 'spot'],
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
            pair_data = data[position['pair']]

            if date not in pair_data.index:
                continue

            current_price = pair_data.loc[date, 'spot']

            # Simple P&L calculation
            price_change = (current_price - position['entry_price']) / position['entry_price']
            position_pnl = position['size'] * price_change * position['direction']

            # Check exit conditions (simplified)
            days_held = (date - position['entry_date']).days

            # Exit after tenor expires or stop loss
            if days_held > 30 or position_pnl < -position['size'] * 0.02:
                positions_to_close.append(i)
                self.closed_trades.append({
                    **position,
                    'exit_date': date,
                    'exit_price': current_price,
                    'pnl': position_pnl,
                    'days_held': days_held
                })
                daily_pnl += position_pnl

        # Remove closed positions
        for i in sorted(positions_to_close, reverse=True):
            del self.positions[i]

        return daily_pnl

    def calculate_performance_metrics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics
        """
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

        # Win rate
        if len(self.closed_trades) > 0:
            trades_df = pd.DataFrame(self.closed_trades)
            win_rate = (trades_df['pnl'] > 0).mean()
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0

        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(self.closed_trades)
        }