"""
Backtesting engine for FX options strategies
Handles position management, P&L calculation, and performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from config import Config
from portfolio import Portfolio, Position
from risk_manager import RiskManager
from models import BaseModel


@dataclass
class BacktestResults:
    """Container for backtest results"""
    equity_curve: pd.DataFrame
    trades: List[Dict]
    metrics: Dict
    greeks_history: pd.DataFrame
    margin_history: pd.DataFrame


class Backtester:
    """Main backtesting engine"""

    def __init__(self, model: BaseModel, config: Config, data: Dict,
                 curves: pd.DataFrame, use_iv_filter: bool = False):
        self.model = model
        self.config = config
        self.data = data
        self.curves = curves
        self.use_iv_filter = use_iv_filter

        self.portfolio = Portfolio(config)
        self.risk_manager = RiskManager(config)

        # History tracking
        self.equity_history = []
        self.trade_history = []
        self.greeks_history = []
        self.margin_history = []
        self.daily_pnl = []

        # IV moving averages for filter strategy
        self.iv_ma = {}
        self._initialize_iv_ma()

    def _initialize_iv_ma(self):
        """Initialize IV moving averages for filter strategy"""
        if not self.use_iv_filter:
            return

        for pair in self.config.CURRENCY_PAIRS:
            if pair not in self.data:
                continue

            self.iv_ma[pair] = {}
            for tenor in self.config.TENORS:
                col = f'atm_vol_{tenor}'
                if col in self.data[pair].columns:
                    # Calculate rolling MA
                    self.iv_ma[pair][tenor] = self.data[pair][col].rolling(
                        window=self.config.IV_MA_WINDOW
                    ).mean()

    def run(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict:
        """Run backtest for specified period"""

        # Get trading dates
        trading_dates = self.data['USDJPY'].loc[start_date:end_date].index

        print(f"  Backtesting from {start_date.date()} to {end_date.date()}")
        print(f"  Total days: {len(trading_dates)}")

        for i, date in enumerate(trading_dates):
            # 1. Update portfolio with current market data
            self.portfolio.mark_to_market(date, self.data, self.curves)

            # 2. Check and close expired positions
            expired = self.portfolio.close_expired_positions(date)
            if expired:
                self.trade_history.extend(expired)

            # 3. Generate new signals
            signals = self._generate_signals(date)

            # 4. Execute trades based on signals
            new_trades = self._execute_signals(signals, date)

            # 5. Calculate and execute delta hedges
            hedge_trades = self.risk_manager.delta_hedge(
                self.portfolio, self.data, date
            )

            # 6. Update portfolio with new trades
            for trade in new_trades + hedge_trades:
                self.portfolio.add_position(trade)

            # 7. Record daily metrics
            self._record_daily_metrics(date)

            # 8. Check risk limits
            if self._check_risk_limits():
                print(f"    Risk limit breached on {date.date()}")
                break

            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(trading_dates)} days")

        return self._compile_results()

    def _generate_signals(self, date: pd.Timestamp) -> List[Dict]:
        """Generate trading signals for all pairs and tenors"""
        signals = []

        for pair in self.config.CURRENCY_PAIRS:
            if pair not in self.data:
                continue

            pair_data = self.data[pair]
            if date not in pair_data.index:
                continue

            # Get current market data
            current = pair_data.loc[date]
            spot = current['spot']

            # Get historical data for signal generation
            hist_end = pair_data.index.get_loc(date)
            hist_start = max(0, hist_end - self.config.LOOKBACK_DAYS * 2)
            historical = pair_data.iloc[hist_start:hist_end]

            if len(historical) < self.config.LOOKBACK_DAYS:
                continue

            # Generate signals for each tenor
            for tenor in self.config.TENORS:
                # Get ATM volatility
                atm_vol_col = f'atm_vol_{tenor}'
                if atm_vol_col not in current or pd.isna(current[atm_vol_col]):
                    continue

                atm_vol = current[atm_vol_col]

                # Calculate signal based on model
                for delta in self.config.DELTAS:
                    signal = self._calculate_signal(
                        pair, tenor, delta, spot, atm_vol,
                        historical, date
                    )

                    if signal:
                        signals.append(signal)

        return signals

    def _calculate_signal(self, pair: str, tenor: str, delta: int,
                          spot: float, atm_vol: float, historical: pd.DataFrame,
                          date: pd.Timestamp) -> Optional[Dict]:
        """Calculate trading signal for specific option"""

        # Calculate strike from delta
        is_call = delta >= 50
        strike = self._delta_to_strike(spot, delta, atm_vol, tenor, is_call)

        # Get market data for smile
        rr_col = f'rr_25_{tenor}'
        bf_col = f'bf_25_{tenor}'

        rr = historical[rr_col].iloc[-1] if rr_col in historical else 0
        bf = historical[bf_col].iloc[-1] if bf_col in historical else 0

        # Calculate market price (simplified - using smile adjustment)
        market_vol = self._get_smile_vol(atm_vol, delta, rr, bf)

        # Calculate model price
        T = self.config.TENOR_YEARS[tenor]
        r_d, r_f = self._get_rates(date, tenor)

        model_price = self.model.price(
            spot=spot,
            strike=strike,
            T=T,
            r_d=r_d,
            r_f=r_f,
            sigma=atm_vol,
            is_call=is_call,
            rr=rr,
            bf=bf
        )

        # Calculate market price using Black-Scholes with market vol
        from models import GarmanKohlhagen
        gk = GarmanKohlhagen()
        market_price = gk.price(
            spot=spot,
            strike=strike,
            T=T,
            r_d=r_d,
            r_f=r_f,
            sigma=market_vol,
            is_call=is_call
        )

        # Check for IV filter condition
        if self.use_iv_filter:
            if not self._check_iv_filter(pair, tenor, atm_vol, date, market_price > model_price):
                return None

        # Calculate mispricing
        if market_price > 0:
            mispricing = (model_price - market_price) / market_price
        else:
            return None

        # Generate signal if mispricing exceeds threshold
        if abs(mispricing) > self.config.SIGNAL_THRESHOLD / 100:
            return {
                'date': date,
                'pair': pair,
                'tenor': tenor,
                'strike': strike,
                'delta': delta,
                'is_call': is_call,
                'direction': 'buy' if mispricing > 0 else 'sell',
                'model_price': model_price,
                'market_price': market_price,
                'mispricing': mispricing,
                'spot': spot,
                'atm_vol': atm_vol,
                'market_vol': market_vol
            }

        return None

    def _check_iv_filter(self, pair: str, tenor: str, current_iv: float,
                         date: pd.Timestamp, is_underpriced: bool) -> bool:
        """Check IV filter condition for trades"""
        if pair not in self.iv_ma or tenor not in self.iv_ma[pair]:
            return True  # No filter if MA not available

        iv_ma_series = self.iv_ma[pair][tenor]
        if date not in iv_ma_series.index:
            return True

        ma_value = iv_ma_series.loc[date]
        if pd.isna(ma_value):
            return True

        # For long positions (underpriced), only trade if IV > MA
        # For short positions (overpriced), only trade if IV < MA
        if is_underpriced:  # Want to go long
            return current_iv > ma_value
        else:  # Want to go short
            return current_iv < ma_value

    def _execute_signals(self, signals: List[Dict], date: pd.Timestamp) -> List[Position]:
        """Execute trades based on signals"""
        trades = []

        # Check position limits
        current_positions = len(self.portfolio.positions)
        available_slots = self.config.MAX_POSITIONS - current_positions

        if available_slots <= 0:
            return trades

        # Sort signals by absolute mispricing
        signals.sort(key=lambda x: abs(x['mispricing']), reverse=True)

        # Execute top signals
        for signal in signals[:available_slots]:
            # Calculate position size
            position_size = self._calculate_position_size(signal)

            # Calculate premium with transaction costs
            premium = signal['market_price'] * position_size
            transaction_cost = self.config.get_transaction_cost('option', premium)

            # Create position
            position = Position(
                date=date,
                pair=signal['pair'],
                tenor=signal['tenor'],
                strike=signal['strike'],
                is_call=signal['is_call'],
                direction=1 if signal['direction'] == 'buy' else -1,
                size=position_size,
                entry_price=signal['market_price'],
                entry_spot=signal['spot'],
                entry_vol=signal['market_vol'],
                model_price=signal['model_price'],
                delta_at_entry=signal['delta'] / 100
            )

            # Update cash for premium and costs
            if signal['direction'] == 'buy':
                self.portfolio.cash -= (premium + transaction_cost)
            else:
                self.portfolio.cash += (premium - transaction_cost)

            trades.append(position)

        return trades

    def _delta_to_strike(self, spot: float, delta: int, vol: float,
                         tenor: str, is_call: bool) -> float:
        """Convert delta to strike price"""
        from scipy.stats import norm

        T = self.config.TENOR_YEARS[tenor]

        # Simplified conversion (assuming r_d = r_f)
        if is_call:
            d1 = norm.ppf(delta / 100)
        else:
            d1 = -norm.ppf(1 - delta / 100)

        strike = spot * np.exp(-d1 * vol * np.sqrt(T) + 0.5 * vol ** 2 * T)

        return strike

    def _get_smile_vol(self, atm_vol: float, delta: int,
                       rr: float, bf: float) -> float:
        """Get volatility from smile parameters"""
        # Simplified smile interpolation
        if delta == 50:  # ATM
            return atm_vol

        # Linear interpolation based on delta
        if delta < 50:  # Put side
            delta_ratio = (50 - delta) / 50
            adjustment = -rr * delta_ratio * 0.5 + bf * delta_ratio
        else:  # Call side
            delta_ratio = (delta - 50) / 50
            adjustment = rr * delta_ratio * 0.5 + bf * delta_ratio

        return atm_vol + adjustment

    def _get_rates(self, date: pd.Timestamp, tenor: str) -> Tuple[float, float]:
        """Get interest rates from curves"""
        if self.curves is not None and date in self.curves.index:
            # Simplified - use same rate for both currencies
            rate = self.curves.loc[date, 'rate'] if 'rate' in self.curves else 0.05
            return rate, rate * 0.5  # Foreign rate lower
        return 0.05, 0.02  # Default rates

    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on signal strength"""
        # Base size as percentage of capital
        base_size = self.portfolio.cash * self.config.POSITION_SIZE_PCT

        # Adjust by signal strength (mispricing)
        size_multiplier = min(abs(signal['mispricing']) * 10, 2.0)

        position_size = base_size * size_multiplier

        # Apply maximum position size limit
        max_size = self.portfolio.cash * self.config.MAX_POSITION_SIZE_PCT

        return min(position_size, max_size)

    def _record_daily_metrics(self, date: pd.Timestamp):
        """Record daily portfolio metrics"""

        # Portfolio value and P&L
        portfolio_value = self.portfolio.get_total_value()
        daily_pnl = portfolio_value - (self.equity_history[-1] if self.equity_history else self.config.INITIAL_CAPITAL)

        self.equity_history.append(portfolio_value)
        self.daily_pnl.append(daily_pnl)

        # Greeks
        greeks = self.portfolio.get_portfolio_greeks()
        greeks['date'] = date
        self.greeks_history.append(greeks)

        # Margin usage
        margin_used = max(0, -self.portfolio.cash)
        margin_ratio = margin_used / portfolio_value if portfolio_value > 0 else 0

        self.margin_history.append({
            'date': date,
            'margin_used': margin_used,
            'margin_ratio': margin_ratio,
            'cash': self.portfolio.cash,
            'portfolio_value': portfolio_value
        })

    def _check_risk_limits(self) -> bool:
        """Check if any risk limits are breached"""
        if not self.equity_history:
            return False

        # Check drawdown
        peak = max(self.equity_history)
        current = self.equity_history[-1]
        drawdown = (peak - current) / peak

        if drawdown > self.config.MAX_DRAWDOWN:
            return True

        # Check leverage
        if self.margin_history:
            current_margin = self.margin_history[-1]
            leverage = current_margin['margin_ratio'] * self.config.MAX_LEVERAGE
            if leverage > self.config.MAX_LEVERAGE:
                return True

        return False

    def _compile_results(self) -> Dict:
        """Compile backtest results"""

        # Create equity curve DataFrame
        dates = self.data['USDJPY'].index[:len(self.equity_history)]
        equity_df = pd.DataFrame({
            'date': dates,
            'equity': self.equity_history,
            'daily_pnl': self.daily_pnl
        }).set_index('date')

        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()

        # Calculate metrics
        metrics = self._calculate_performance_metrics(equity_df)

        # Add trading statistics
        if self.trade_history:
            trade_stats = self._calculate_trade_statistics()
            metrics.update(trade_stats)

        # Greeks history DataFrame
        greeks_df = pd.DataFrame(self.greeks_history).set_index('date') if self.greeks_history else pd.DataFrame()

        # Margin history DataFrame
        margin_df = pd.DataFrame(self.margin_history).set_index('date') if self.margin_history else pd.DataFrame()

        return {
            'equity_curve': equity_df,
            'trades': self.trade_history,
            'metrics': metrics,
            'greeks_history': greeks_df,
            'margin_history': margin_df
        }

    def _calculate_performance_metrics(self, equity_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        returns = equity_df['returns'].dropna()

        if len(returns) == 0:
            return {}

        # Basic metrics
        total_return = (equity_df['equity'].iloc[-1] / self.config.INITIAL_CAPITAL) - 1

        # Risk metrics
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino = returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

        # Drawdown
        rolling_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Delta hedge effectiveness
        if self.greeks_history:
            greeks_df = pd.DataFrame(self.greeks_history)
            avg_delta = greeks_df['delta'].abs().mean() if 'delta' in greeks_df else 0
        else:
            avg_delta = 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'avg_delta_hedge': avg_delta
        }

    def _calculate_trade_statistics(self) -> Dict:
        """Calculate trade statistics"""
        if not self.trade_history:
            return {}

        trades_df = pd.DataFrame(self.trade_history)

        # Calculate P&L for each trade
        if 'exit_price' in trades_df.columns:
            trades_df['pnl'] = (trades_df['exit_price'] - trades_df['entry_price']) * trades_df['size'] * trades_df[
                'direction']

            # Win rate
            winning_trades = trades_df[trades_df['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0

            return {
                'total_trades': len(trades_df),
                'win_rate': win_rate,
                'avg_pnl': trades_df['pnl'].mean(),
                'total_pnl': trades_df['pnl'].sum()
            }

        return {'total_trades': len(trades_df)}