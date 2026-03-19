"""
Risk Management Module for KIMITRADE
Implements MDD controls, position sizing, and risk limits
"""

import json
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
import pandas as pd

class RiskManager:
    """
    Implements research-backed risk controls:
    - Drawdown ladder (5%→90%, 10%→75%, 15%→50%, 20%→25%, 25%→HALT)
    - Daily/weekly/monthly loss limits
    - Consecutive loss control
    - Volatility-based sizing
    """
    
    def __init__(self, starting_capital: float = 500000):
        self.starting_capital = starting_capital
        self.peak_capital = starting_capital
        self.current_capital = starting_capital
        
        # Tracking
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.equity_history = [starting_capital]
        
        # Settings
        self.dd_tiers = [
            (5.0, 0.90, "DD_TIER1"),
            (10.0, 0.75, "DD_TIER2"),
            (15.0, 0.50, "DD_TIER3"),
            (20.0, 0.25, "DD_TIER4"),
            (25.0, 0.00, "DD_HALT")
        ]
        
        self.limits = {
            'daily': 3.0,    # 3% daily loss limit
            'weekly': 5.0,   # 5% weekly loss limit
            'monthly': 8.0  # 8% monthly loss limit
        }
        
        self.consecutive_loss_limit = 3
        self.consecutive_loss_multiplier = 0.5
        
        # Current tracking periods
        self.current_day = date.today()
        self.current_week = datetime.now().isocalendar()[1]
        self.current_month = date.today().month
        
    def update_capital(self, pnl: float):
        """Update capital after trade and track metrics"""
        self.current_capital += pnl
        self.equity_history.append(self.current_capital)
        
        # Update peak and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Update period P&L
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.monthly_pnl += pnl
    
    def check_period_reset(self):
        """Check if we need to reset daily/weekly/monthly tracking"""
        today = date.today()
        current_week = datetime.now().isocalendar()[1]
        current_month = today.month
        
        if today != self.current_day:
            self.daily_pnl = 0.0
            self.current_day = today
            
        if current_week != self.current_week:
            self.weekly_pnl = 0.0
            self.current_week = current_week
            
        if current_month != self.current_month:
            self.monthly_pnl = 0.0
            self.current_month = current_month
    
    def get_drawdown(self) -> Tuple[float, float]:
        """Return current drawdown in amount and percentage"""
        dd_amount = self.peak_capital - self.current_capital
        dd_pct = (dd_amount / self.peak_capital) * 100 if self.peak_capital > 0 else 0
        return dd_amount, dd_pct
    
    def get_position_size_multiplier(self, current_atr: float = 0) -> Tuple[float, str]:
        """
        Calculate position size multiplier based on risk controls
        Returns: (multiplier, reason)
        """
        self.check_period_reset()
        
        dd_amount, dd_pct = self.get_drawdown()
        
        # 1. Drawdown Ladder
        for threshold, multiplier, reason in self.dd_tiers:
            if dd_pct >= threshold:
                if multiplier == 0.0:
                    return 0.0, f"{reason} ({dd_pct:.1f}% DD) - TRADING HALTED"
                return multiplier, f"{reason} ({dd_pct:.1f}% DD)"
        
        # 2. Daily Loss Limit
        daily_loss_pct = (-self.daily_pnl / self.current_capital) * 100
        if daily_loss_pct >= self.limits['daily']:
            return 0.0, f"DAILY_LIMIT ({daily_loss_pct:.1f}% loss)"
        
        # 3. Weekly Loss Limit
        weekly_loss_pct = (-self.weekly_pnl / self.current_capital) * 100
        if weekly_loss_pct >= self.limits['weekly']:
            return 0.0, f"WEEKLY_LIMIT ({weekly_loss_pct:.1f}% loss)"
        
        # 4. Monthly Loss Limit
        monthly_loss_pct = (-self.monthly_pnl / self.current_capital) * 100
        if monthly_loss_pct >= self.limits['monthly']:
            return 0.0, f"MONTHLY_LIMIT ({monthly_loss_pct:.1f}% loss)"
        
        # 5. Consecutive Loss Control
        if self.consecutive_losses >= self.consecutive_loss_limit:
            return self.consecutive_loss_multiplier, f"CONSEC_LOSS ({self.consecutive_losses})"
        
        # 6. Equity Curve Filter (20-period MA)
        if len(self.equity_history) >= 20:
            recent_equity = self.equity_history[-20:]
            equity_ma = sum(recent_equity) / len(recent_equity)
            if self.current_capital < equity_ma:
                return 0.5, "BELOW_EQ_MA"
        
        # 7. Volatility Filter (if ATR provided)
        # This would need historical ATR values to implement properly
        
        return 1.0, "FULL_SIZE"
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed"""
        multiplier, reason = self.get_position_size_multiplier()
        if multiplier <= 0:
            return False, reason
        return True, reason
    
    def calculate_position_size(self, risk_per_trade_pct: float, 
                                entry_price: float, stop_loss: float,
                                lot_size: int = 1) -> int:
        """
        Calculate number of lots based on risk
        """
        multiplier, reason = self.get_position_size_multiplier()
        
        if multiplier <= 0:
            return 0
        
        # Risk amount
        risk_amount = self.current_capital * (risk_per_trade_pct / 100) * multiplier
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return 0
        
        # Number of shares
        num_shares = int(risk_amount / risk_per_share)
        
        # Convert to lots
        num_lots = max(1, num_shares // lot_size)
        
        return num_lots
    
    def get_status_report(self) -> Dict:
        """Get current risk status"""
        dd_amount, dd_pct = self.get_drawdown()
        can_trade, reason = self.can_trade()
        
        return {
            'capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'drawdown_amount': dd_amount,
            'drawdown_pct': dd_pct,
            'consecutive_losses': self.consecutive_losses,
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'monthly_pnl': self.monthly_pnl,
            'can_trade': can_trade,
            'reason': reason
        }


class PositionSizer:
    """
    Advanced position sizing methods
    """
    
    @staticmethod
    def fixed_fractional(capital: float, risk_pct: float, 
                        entry: float, stop: float, lot_size: int = 1) -> int:
        """Fixed fractional sizing (e.g., 1% risk per trade)"""
        risk_amount = capital * (risk_pct / 100)
        risk_per_unit = abs(entry - stop)
        
        if risk_per_unit <= 0:
            return 0
        
        units = int(risk_amount / risk_per_unit)
        return max(1, units // lot_size)
    
    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float,
                       fraction: float = 0.25) -> float:
        """
        Fractional Kelly Criterion
        f* = (p*b - q) / b
        where p = win rate, q = loss rate, b = win/loss ratio
        """
        if avg_loss == 0 or win_rate >= 1:
            return 0
        
        loss_rate = 1 - win_rate
        b = avg_win / avg_loss
        
        kelly = (win_rate * b - loss_rate) / b
        
        # Fractional Kelly for safety
        return max(0, kelly * fraction)
    
    @staticmethod
    def volatility_based(capital: float, risk_pct: float, atr: float,
                        atr_multiplier: float = 2.0, lot_size: int = 1) -> int:
        """ATR-based position sizing"""
        risk_amount = capital * (risk_pct / 100)
        risk_per_unit = atr * atr_multiplier
        
        if risk_per_unit <= 0:
            return 0
        
        units = int(risk_amount / risk_per_unit)
        return max(1, units // lot_size)


class TradeLogger:
    """
    Comprehensive trade logging for analysis
    """
    
    def __init__(self, log_file: str = 'trades.json'):
        self.log_file = log_file
        self.trades = []
        self.load_trades()
    
    def load_trades(self):
        try:
            with open(self.log_file, 'r') as f:
                self.trades = json.load(f)
        except FileNotFoundError:
            self.trades = []
    
    def save_trades(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.trades, f, indent=2)
    
    def log_trade(self, trade: Dict):
        trade['timestamp'] = datetime.now().isoformat()
        self.trades.append(trade)
        self.save_trades()
    
    def get_stats(self) -> Dict:
        if not self.trades:
            return {}
        
        df = pd.DataFrame(self.trades)
        
        winners = df[df['pnl'] > 0]
        losers = df[df['pnl'] <= 0]
        
        total_trades = len(df)
        win_rate = len(winners) / total_trades if total_trades > 0 else 0
        
        avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0
        
        total_pnl = df['pnl'].sum()
        profit_factor = abs(winners['pnl'].sum()) / abs(losers['pnl'].sum()) if len(losers) > 0 and losers['pnl'].sum() != 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_win': df['pnl'].max(),
            'max_loss': df['pnl'].min()
        }
