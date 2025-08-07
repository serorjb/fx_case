# src/backtesting/performance.py
"""
Performance analysis module for backtesting results
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for backtest results
    """

    def __init__(self):
        self.metrics = {}

    def calculate_metrics(self, returns: pd.Series, equity: pd.Series,
                          trades: pd.DataFrame = None) -> Dict:
        """
        Calculate comprehensive performance metrics

        Parameters:
        -----------
        returns : pd.Series
            Daily returns
        equity : pd.Series
            Equity curve
        trades : pd.DataFrame
            Trade history (optional)

        Returns:
        --------
        Dict : Performance metrics
        """
        metrics = {}

        # Return metrics
        metrics['total_return'] = self._calculate_total_return(equity)
        metrics['annualized_return'] = self._annualized_return(returns)
        metrics['volatility'] = self._calculate_volatility(returns)

        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = self._sharpe_ratio(returns)
        metrics['sortino_ratio'] = self._sortino_ratio(returns)
        metrics['calmar_ratio'] = self._calmar_ratio(equity)
        metrics['omega_ratio'] = self._omega_ratio(returns)

        # Drawdown metrics
        dd_stats = self._drawdown_statistics(equity)
        metrics.update(dd_stats)

        # Trade statistics
        if trades is not None and len(trades) > 0:
            trade_stats = self._trade_statistics(trades)
            metrics.update(trade_stats)

        # Risk metrics
        metrics['var_95'] = self._calculate_var(returns, 0.95)
        metrics['cvar_95'] = self._calculate_cvar(returns, 0.95)
        metrics['max_1d_loss'] = returns.min() if len(returns) > 0 else 0
        metrics['max_1d_gain'] = returns.max() if len(returns) > 0 else 0

        # Distribution metrics
        if len(returns) > 3:
            metrics['skewness'] = stats.skew(returns.dropna())
            metrics['kurtosis'] = stats.kurtosis(returns.dropna())
        else:
            metrics['skewness'] = 0
            metrics['kurtosis'] = 0

        metrics['hit_ratio'] = (returns > 0).mean() if len(returns) > 0 else 0

        # Consistency metrics
        metrics['positive_months'] = self._positive_months(returns)
        metrics['best_month'] = self._best_month(returns)
        metrics['worst_month'] = self._worst_month(returns)

        return metrics

    def _calculate_total_return(self, equity: pd.Series) -> float:
        """Calculate total return from equity curve"""
        if len(equity) < 2:
            return 0
        return (equity.iloc[-1] / equity.iloc[0] - 1)

    def _annualized_return(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0

        # Compound return
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)

        if n_periods == 0:
            return 0

        # Annualize
        years = n_periods / periods_per_year
        if years <= 0:
            return total_return

        return (1 + total_return) ** (1 / years) - 1

    def _calculate_volatility(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized volatility"""
        if len(returns) < 2:
            return 0
        return returns.std() * np.sqrt(periods_per_year)

    def _sharpe_ratio(self, returns: pd.Series, risk_free: float = 0.02,
                      periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0

        excess_returns = returns - risk_free / periods_per_year

        if returns.std() == 0:
            return 0

        return excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)

    def _sortino_ratio(self, returns: pd.Series, risk_free: float = 0.02,
                       periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        if len(returns) < 2:
            return 0

        excess_returns = returns - risk_free / periods_per_year
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return np.inf if excess_returns.mean() > 0 else 0

        downside_std = downside_returns.std()

        if downside_std == 0:
            return 0

        return excess_returns.mean() / downside_std * np.sqrt(periods_per_year)

    def _calmar_ratio(self, equity: pd.Series) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        if len(equity) < 2:
            return 0

        total_return = self._calculate_total_return(equity)
        max_dd = self._max_drawdown(equity)

        if max_dd == 0:
            return np.inf if total_return > 0 else 0

        # Annualize return
        n_years = len(equity) / 252
        if n_years <= 0:
            annual_return = total_return
        else:
            annual_return = (1 + total_return) ** (1 / n_years) - 1

        return annual_return / abs(max_dd)

    def _omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio"""
        if len(returns) == 0:
            return 0

        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = -excess_returns[excess_returns < 0].sum()

        if losses == 0:
            return np.inf if gains > 0 else 0

        return gains / losses

    def _max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(equity) < 2:
            return 0

        # Calculate running maximum
        running_max = equity.expanding().max()

        # Calculate drawdown
        drawdown = (equity - running_max) / running_max

        return drawdown.min()

    def _drawdown_statistics(self, equity: pd.Series) -> Dict:
        """Calculate comprehensive drawdown statistics"""
        if len(equity) < 2:
            return {
                'max_drawdown': 0,
                'avg_drawdown': 0,
                'max_drawdown_duration': 0,
                'recovery_time': 0
            }

        # Calculate drawdown series
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max

        # Find drawdown periods
        is_drawdown = drawdown < 0

        # Calculate statistics
        stats = {
            'max_drawdown': drawdown.min(),
            'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        }

        # Calculate drawdown durations
        if is_drawdown.any():
            # Find start and end of drawdown periods
            drawdown_starts = (~is_drawdown).shift(1) & is_drawdown
            drawdown_ends = is_drawdown.shift(1) & (~is_drawdown)

            # Calculate max duration
            max_duration = 0
            current_duration = 0

            for i, in_dd in enumerate(is_drawdown):
                if in_dd:
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
                else:
                    current_duration = 0

            stats['max_drawdown_duration'] = max_duration

            # Calculate recovery time (time to recover from max drawdown)
            max_dd_idx = drawdown.idxmin()
            if max_dd_idx in equity.index:
                max_dd_value = running_max[max_dd_idx]
                recovery_mask = (equity[max_dd_idx:] >= max_dd_value)
                if recovery_mask.any():
                    recovery_idx = recovery_mask.idxmax()
                    stats['recovery_time'] = (recovery_idx - max_dd_idx).days
                else:
                    stats['recovery_time'] = np.inf  # Never recovered
            else:
                stats['recovery_time'] = 0
        else:
            stats['max_drawdown_duration'] = 0
            stats['recovery_time'] = 0

        return stats

    def _trade_statistics(self, trades: pd.DataFrame) -> Dict:
        """Calculate trade-level statistics"""
        stats = {}

        if 'pnl' not in trades.columns:
            return stats

        pnls = trades['pnl'].dropna()

        if len(pnls) == 0:
            return stats

        # Basic statistics
        stats['total_trades'] = len(trades)
        stats['total_pnl'] = pnls.sum()
        stats['avg_pnl'] = pnls.mean()
        stats['median_pnl'] = pnls.median()
        stats['std_pnl'] = pnls.std()

        # Win/loss statistics
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        stats['num_wins'] = len(wins)
        stats['num_losses'] = len(losses)
        stats['win_rate'] = len(wins) / len(pnls) if len(pnls) > 0 else 0

        stats['avg_win'] = wins.mean() if len(wins) > 0 else 0
        stats['avg_loss'] = losses.mean() if len(losses) > 0 else 0
        stats['max_win'] = wins.max() if len(wins) > 0 else 0
        stats['max_loss'] = losses.min() if len(losses) > 0 else 0

        # Risk-reward metrics
        if stats['avg_loss'] != 0:
            stats['risk_reward_ratio'] = abs(stats['avg_win'] / stats['avg_loss'])
        else:
            stats['risk_reward_ratio'] = np.inf if stats['avg_win'] > 0 else 0

        # Profit factor
        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0

        if total_losses > 0:
            stats['profit_factor'] = total_wins / total_losses
        else:
            stats['profit_factor'] = np.inf if total_wins > 0 else 0

        # Expectancy
        stats['expectancy'] = (stats['win_rate'] * stats['avg_win'] +
                               (1 - stats['win_rate']) * stats['avg_loss'])

        # Consecutive wins/losses
        if len(pnls) > 0:
            stats['max_consecutive_wins'] = self._max_consecutive(pnls > 0, True)
            stats['max_consecutive_losses'] = self._max_consecutive(pnls < 0, True)

        # Average holding period
        if 'days_held' in trades.columns:
            stats['avg_holding_period'] = trades['days_held'].mean()
            stats['median_holding_period'] = trades['days_held'].median()

        # By strategy breakdown if available
        if 'strategy' in trades.columns:
            strategy_stats = {}
            for strategy in trades['strategy'].unique():
                strategy_trades = trades[trades['strategy'] == strategy]
                strategy_pnl = strategy_trades['pnl'].sum()
                strategy_count = len(strategy_trades)
                strategy_winrate = (strategy_trades['pnl'] > 0).mean()

                strategy_stats[strategy] = {
                    'total_pnl': strategy_pnl,
                    'num_trades': strategy_count,
                    'win_rate': strategy_winrate
                }
            stats['strategy_breakdown'] = strategy_stats

        return stats

    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0
        return np.percentile(returns.dropna(), (1 - confidence) * 100)

    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self._calculate_var(returns, confidence)
        conditional_returns = returns[returns <= var]
        return conditional_returns.mean() if len(conditional_returns) > 0 else var

    def _positive_months(self, returns: pd.Series) -> float:
        """Calculate percentage of positive months"""
        if len(returns) == 0:
            return 0

        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        return (monthly_returns > 0).mean()

    def _best_month(self, returns: pd.Series) -> float:
        """Calculate best monthly return"""
        if len(returns) == 0:
            return 0

        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        return monthly_returns.max() if len(monthly_returns) > 0 else 0

    def _worst_month(self, returns: pd.Series) -> float:
        """Calculate worst monthly return"""
        if len(returns) == 0:
            return 0

        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        return monthly_returns.min() if len(monthly_returns) > 0 else 0

    def _max_consecutive(self, series: pd.Series, value: bool) -> int:
        """Calculate maximum consecutive occurrences of a value"""
        max_count = 0
        current_count = 0

        for val in series:
            if val == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def create_performance_report(self, metrics: Dict) -> str:
        """
        Create formatted performance report
        """
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 60)

        # Return Metrics
        report.append("\nRETURN METRICS:")
        report.append("-" * 30)
        report.append(f"Total Return: {metrics.get('total_return', 0):.2%}")
        report.append(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        report.append(f"Volatility: {metrics.get('volatility', 0):.2%}")

        # Risk-Adjusted Returns
        report.append("\nRISK-ADJUSTED RETURNS:")
        report.append("-" * 30)
        report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        report.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
        report.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
        report.append(f"Omega Ratio: {metrics.get('omega_ratio', 0):.3f}")

        # Drawdown Analysis
        report.append("\nDRAWDOWN ANALYSIS:")
        report.append("-" * 30)
        report.append(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"Avg Drawdown: {metrics.get('avg_drawdown', 0):.2%}")
        report.append(f"Max DD Duration: {metrics.get('max_drawdown_duration', 0)} days")
        report.append(f"Recovery Time: {metrics.get('recovery_time', 0)} days")

        # Risk Metrics
        report.append("\nRISK METRICS:")
        report.append("-" * 30)
        report.append(f"VaR (95%): {metrics.get('var_95', 0):.3%}")
        report.append(f"CVaR (95%): {metrics.get('cvar_95', 0):.3%}")
        report.append(f"Max Daily Loss: {metrics.get('max_1d_loss', 0):.3%}")
        report.append(f"Max Daily Gain: {metrics.get('max_1d_gain', 0):.3%}")

        # Distribution
        report.append("\nDISTRIBUTION:")
        report.append("-" * 30)
        report.append(f"Skewness: {metrics.get('skewness', 0):.3f}")
        report.append(f"Kurtosis: {metrics.get('kurtosis', 0):.3f}")
        report.append(f"Hit Ratio: {metrics.get('hit_ratio', 0):.2%}")

        # Trading Statistics
        if 'total_trades' in metrics:
            report.append("\nTRADING STATISTICS:")
            report.append("-" * 30)
            report.append(f"Total Trades: {metrics.get('total_trades', 0)}")
            report.append(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
            report.append(f"Avg Win: ${metrics.get('avg_win', 0):,.2f}")
            report.append(f"Avg Loss: ${metrics.get('avg_loss', 0):,.2f}")
            report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            report.append(f"Expectancy: ${metrics.get('expectancy', 0):,.2f}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)
