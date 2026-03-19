"""
ALTERNATIVE 1: Intraday Momentum Strategy
Based on: "Market Intraday Momentum" (Gao et al., Journal of Financial Economics)

Logic:
- First 30-min return (9:15-9:45 AM) predicts last 30-min direction
- If first 30-min is positive: go LONG at 11 AM
- If first 30-min is negative: go SHORT at 11 AM
- Exit at 3:15 PM or opposite signal

Performance:
- 6.3% annual return vs -0.5% buy-and-hold
- Works in both bull and bear markets
- R² ~1.6-2%, up to 4-7% in recessions
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import sys
sys.path.append('..')
from utils.indicators import calculate_first_30min_momentum
from utils.risk_manager import RiskManager, PositionSizer
from utils.paper_trading import PaperTradingEngine

@dataclass
class MomentumConfig:
    """Configuration for Intraday Momentum Strategy"""
    
    # Entry timing
    ENTRY_TIME = time(11, 0)  # Enter at 11 AM
    EXIT_TIME = time(15, 15)  # Exit at 3:15 PM
    
    # Risk management (improvements over original)
    USE_SL = True  # Enable SL (contradicts original research but safer)
    SL_ATR_MULTIPLIER = 2.0  # 2x ATR for SL
    TARGET_RR = 1.5  # 1.5:1 risk reward
    
    # Position sizing
    RISK_PER_TRADE_PCT = 1.0  # 1% risk per trade
    
    # Momentum threshold
    MIN_MOMENTUM_PCT = 0.15  # Min 0.15% momentum to trade
    
    # Market reversal check (NEW - not in original)
    EXIT_ON_REVERSAL = True
    REVERSAL_THRESHOLD = 0.3  # Exit if momentum reverses by 0.3%
    
    # Max trades
    MAX_TRADES_PER_DAY = 2


class IntradayMomentumStrategy:
    """
    Intraday Momentum Strategy implementation
    """
    
    def __init__(self, capital: float = 500000):
        self.config = MomentumConfig()
        self.engine = PaperTradingEngine(capital)
        self.risk_manager = RiskManager(capital)
        self.today_trades = 0
    
    def calculate_first_30min_return(self, df: pd.DataFrame) -> float:
        """
        Calculate first 30-minute return (9:15-9:45 AM)
        """
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['time'] = df_copy['date'].dt.time
        df_copy['date_only'] = df_copy['date'].dt.date
        
        today = df_copy['date_only'].iloc[-1]
        today_data = df_copy[df_copy['date_only'] == today]
        
        # Get 9:15 to 9:45 data
        first_30 = today_data[
            (today_data['time'] >= time(9, 15)) &
            (today_data['time'] <= time(9, 45))
        ]
        
        if len(first_30) < 2:
            return 0.0
        
        first_price = first_30.iloc[0]['open']
        last_price = first_30.iloc[-1]['close']
        
        return ((last_price - first_price) / first_price) * 100
    
    def generate_signal(self, df: pd.DataFrame, nifty_df: pd.DataFrame) -> Optional[Dict]:
        """
        Generate trading signal based on first 30-min momentum
        """
        # Check if within entry window
        current_time = datetime.now().time()
        if current_time < self.config.ENTRY_TIME:
            return None
        
        if current_time > time(13, 30):  # Don't enter after 1:30 PM
            return None
        
        # Check daily trade limit
        if self.today_trades >= self.config.MAX_TRADES_PER_DAY:
            return None
        
        # Check risk manager
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            return None
        
        # Calculate momentum
        momentum = self.calculate_first_30min_return(df)
        
        # Check minimum momentum threshold
        if abs(momentum) < self.config.MIN_MOMENTUM_PCT:
            return None
        
        # Determine direction
        if momentum > 0:
            trade_type = "LONG"
            direction = "positive"
        else:
            trade_type = "SHORT"
            direction = "negative"
        
        # Get entry price (current price)
        entry_price = df.iloc[-1]['close']
        
        # Calculate SL and Target
        from utils.indicators import calculate_atr
        atr = calculate_atr(df).iloc[-1]
        
        sl_distance = atr * self.config.SL_ATR_MULTIPLIER
        
        if trade_type == "LONG":
            sl_price = entry_price - sl_distance
            target_price = entry_price + (sl_distance * self.config.TARGET_RR)
        else:
            sl_price = entry_price + sl_distance
            target_price = entry_price - (sl_distance * self.config.TARGET_RR)
        
        # Calculate position size
        quantity = self.risk_manager.calculate_position_size(
            self.config.RISK_PER_TRADE_PCT,
            entry_price,
            sl_price,
            lot_size=1  # Will be overridden with actual lot size
        )
        
        return {
            'symbol': 'NIFTY' if 'nifty' in nifty_df else df.iloc[-1].get('symbol', 'UNKNOWN'),
            'trade_type': trade_type,
            'direction': direction,
            'momentum': momentum,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'target_price': target_price,
            'quantity': quantity,
            'atr': atr,
            'reasons': [f'First 30-min momentum: {momentum:+.2f}%']
        }
    
    def check_exit(self, trade_id: str, current_price: float, 
                   current_time: time, momentum_now: float) -> Optional[str]:
        """
        Check if trade should be exited
        Includes reversal detection (improvement over original)
        """
        if trade_id not in self.engine.active_trades:
            return None
        
        trade = self.engine.active_trades[trade_id]
        
        # Check SL
        if trade.sl_price:
            if trade.trade_type == "LONG" and current_price <= trade.sl_price:
                return "SL_HIT"
            if trade.trade_type == "SHORT" and current_price >= trade.sl_price:
                return "SL_HIT"
        
        # Check Target
        if trade.target_price:
            if trade.trade_type == "LONG" and current_price >= trade.target_price:
                return "TARGET_HIT"
            if trade.trade_type == "SHORT" and current_price <= trade.target_price:
                return "TARGET_HIT"
        
        # Check Time Exit
        if current_time >= self.config.EXIT_TIME:
            return "TIME_EXIT"
        
        # Check Reversal (NEW - improvement)
        if self.config.EXIT_ON_REVERSAL:
            # If momentum reversed significantly, exit
            if trade.trade_type == "LONG" and momentum_now < -self.config.REVERSAL_THRESHOLD:
                return "MOMENTUM_REVERSAL"
            if trade.trade_type == "SHORT" and momentum_now > self.config.REVERSAL_THRESHOLD:
                return "MOMENTUM_REVERSAL"
        
        return None
    
    def execute_signal(self, signal: Dict, lot_size: int = 1) -> Optional[str]:
        """
        Execute a trading signal
        """
        # Recalculate quantity with actual lot size
        quantity = self.risk_manager.calculate_position_size(
            self.config.RISK_PER_TRADE_PCT,
            signal['entry_price'],
            signal['sl_price'],
            lot_size
        )
        
        if quantity <= 0:
            return None
        
        # Enter trade
        trade = self.engine.enter_trade(
            symbol=signal['symbol'],
            trade_type=signal['trade_type'],
            entry_price=signal['entry_price'],
            quantity=quantity * lot_size,
            sl_price=signal['sl_price'],
            target_price=signal['target_price']
        )
        
        self.today_trades += 1
        
        return trade.id
    
    def update_trades(self, price_data: Dict[str, float], 
                     momentum_data: Dict[str, float]) -> List[Dict]:
        """
        Update all active trades and check for exits
        """
        current_time = datetime.now().time()
        exited = []
        
        for trade_id in list(self.engine.active_trades.keys()):
            trade = self.engine.active_trades[trade_id]
            
            if trade.symbol not in price_data:
                continue
            
            current_price = price_data[trade.symbol]
            momentum_now = momentum_data.get(trade.symbol, 0)
            
            exit_reason = self.check_exit(trade_id, current_price, current_time, momentum_now)
            
            if exit_reason:
                exited_trade = self.engine.exit_trade(trade_id, current_price, exit_reason)
                
                # Update risk manager
                self.risk_manager.update_capital(exited_trade.pnl)
                
                exited.append(exited_trade.to_dict())
        
        return exited
    
    def get_status(self) -> Dict:
        """Get strategy status"""
        return {
            'strategy': 'Intraday Momentum',
            'config': {
                'entry_time': self.config.ENTRY_TIME.strftime('%H:%M'),
                'exit_time': self.config.EXIT_TIME.strftime('%H:%M'),
                'use_sl': self.config.USE_SL,
                'sl_atr_mult': self.config.SL_ATR_MULTIPLIER,
                'risk_per_trade': self.config.RISK_PER_TRADE_PCT
            },
            'risk_status': self.risk_manager.get_status_report(),
            'portfolio': self.engine.get_portfolio_summary()
        }
    
    def reset_daily(self):
        """Reset daily counters"""
        self.today_trades = 0


# ============== BACKTESTER ==============

class IntradayMomentumBacktester:
    """
    Backtester for Intraday Momentum Strategy
    """
    
    def __init__(self, capital: float = 500000):
        self.strategy = IntradayMomentumStrategy(capital)
    
    def backtest(self, data: pd.DataFrame, nifty_data: pd.DataFrame) -> Dict:
        """
        Backtest on historical data
        """
        trades = []
        
        # Group by date
        dates = data['date'].dt.date.unique()
        
        for date in dates:
            day_data = data[data['date'].dt.date == date]
            nifty_day = nifty_data[nifty_data['date'].dt.date == date]
            
            if len(day_data) < 30 or len(nifty_day) < 30:
                continue
            
            # Calculate first 30-min momentum
            momentum = self.strategy.calculate_first_30min_return(day_data)
            
            # Check if we have a signal
            if abs(momentum) < self.strategy.config.MIN_MOMENTUM_PCT:
                continue
            
            # Get 11 AM entry price
            entry_data = day_data[day_data['date'].dt.time >= time(11, 0)]
            if len(entry_data) == 0:
                continue
            
            entry_price = entry_data.iloc[0]['close']
            
            # Determine direction
            trade_type = "LONG" if momentum > 0 else "SHORT"
            
            # Calculate SL and Target
            from utils.indicators import calculate_atr
            atr = calculate_atr(day_data).iloc[-1]
            sl_distance = atr * self.strategy.config.SL_ATR_MULTIPLIER
            
            if trade_type == "LONG":
                sl_price = entry_price - sl_distance
                target_price = entry_price + (sl_distance * self.strategy.config.TARGET_RR)
            else:
                sl_price = entry_price + sl_distance
                target_price = entry_price - (sl_distance * self.strategy.config.TARGET_RR)
            
            # Simulate until exit
            exit_data = day_data[day_data['date'].dt.time >= time(11, 0)]
            exit_price = None
            exit_reason = "TIME_EXIT"
            
            for _, row in exit_data.iterrows():
                current_price = row['close']
                current_time = row['date'].time()
                
                # Check SL
                if trade_type == "LONG" and current_price <= sl_price:
                    exit_price = current_price
                    exit_reason = "SL_HIT"
                    break
                if trade_type == "SHORT" and current_price >= sl_price:
                    exit_price = current_price
                    exit_reason = "SL_HIT"
                    break
                
                # Check Target
                if trade_type == "LONG" and current_price >= target_price:
                    exit_price = current_price
                    exit_reason = "TARGET_HIT"
                    break
                if trade_type == "SHORT" and current_price <= target_price:
                    exit_price = current_price
                    exit_reason = "TARGET_HIT"
                    break
                
                # Check Time Exit
                if current_time >= time(15, 15):
                    exit_price = current_price
                    exit_reason = "TIME_EXIT"
                    break
            
            if exit_price:
                # Calculate P&L
                if trade_type == "LONG":
                    pnl = exit_price - entry_price
                else:
                    pnl = entry_price - exit_price
                
                trades.append({
                    'date': date,
                    'type': trade_type,
                    'momentum': momentum,
                    'entry': entry_price,
                    'exit': exit_price,
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'pnl_pct': (pnl / entry_price) * 100
                })
        
        # Calculate statistics
        if trades:
            df = pd.DataFrame(trades)
            winners = df[df['pnl'] > 0]
            losers = df[df['pnl'] <= 0]
            
            return {
                'total_trades': len(df),
                'win_rate': len(winners) / len(df),
                'total_pnl': df['pnl'].sum(),
                'avg_pnl': df['pnl'].mean(),
                'avg_win': winners['pnl'].mean() if len(winners) > 0 else 0,
                'avg_loss': losers['pnl'].mean() if len(losers) > 0 else 0,
                'profit_factor': abs(winners['pnl'].sum()) / abs(losers['pnl'].sum()) if len(losers) > 0 else 0,
                'trades': trades
            }
        
        return {'total_trades': 0}
