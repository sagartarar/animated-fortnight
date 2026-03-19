"""
Paper Trading Engine for KIMITRADE
Simulates real trading without real money
"""

import json
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class TradeStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"

class ExitReason(Enum):
    TARGET_HIT = "target_hit"
    SL_HIT = "sl_hit"
    TIME_EXIT = "time_exit"
    MANUAL = "manual"
    REVERSAL = "reversal"

@dataclass
class PaperTrade:
    """Represents a paper trade"""
    id: str
    symbol: str
    trade_type: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: datetime
    quantity: int
    
    # Exit details
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    
    # Orders
    sl_price: Optional[float] = None
    target_price: Optional[float] = None
    
    # Status
    status: str = "open"
    
    # P&L
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'trade_type': self.trade_type,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'quantity': self.quantity,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_reason': self.exit_reason,
            'sl_price': self.sl_price,
            'target_price': self.target_price,
            'status': self.status,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct
        }


class PaperTradingEngine:
    """
    Paper trading engine that simulates trade execution
    """
    
    def __init__(self, initial_capital: float = 500000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades: List[PaperTrade] = []
        self.active_trades: Dict[str, PaperTrade] = {}
        self.trade_counter = 0
    
    def generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        self.trade_counter += 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"TRADE_{timestamp}_{self.trade_counter}"
    
    def enter_trade(self, symbol: str, trade_type: str, entry_price: float,
                   quantity: int, sl_price: Optional[float] = None,
                   target_price: Optional[float] = None) -> PaperTrade:
        """
        Enter a new paper trade
        """
        trade_id = self.generate_trade_id()
        
        trade = PaperTrade(
            id=trade_id,
            symbol=symbol,
            trade_type=trade_type,
            entry_price=entry_price,
            entry_time=datetime.now(),
            quantity=quantity,
            sl_price=sl_price,
            target_price=target_price,
            status="open"
        )
        
        self.trades.append(trade)
        self.active_trades[trade_id] = trade
        
        return trade
    
    def check_exit_conditions(self, trade_id: str, current_price: float,
                             current_time: time, nifty_change: float) -> Optional[str]:
        """
        Check if trade should be exited
        Returns exit reason if conditions met, None otherwise
        """
        if trade_id not in self.active_trades:
            return None
        
        trade = self.active_trades[trade_id]
        
        # Check SL
        if trade.sl_price:
            if trade.trade_type == "LONG" and current_price <= trade.sl_price:
                return ExitReason.SL_HIT.value
            if trade.trade_type == "SHORT" and current_price >= trade.sl_price:
                return ExitReason.SL_HIT.value
        
        # Check Target
        if trade.target_price:
            if trade.trade_type == "LONG" and current_price >= trade.target_price:
                return ExitReason.TARGET_HIT.value
            if trade.trade_type == "SHORT" and current_price <= trade.target_price:
                return ExitReason.TARGET_HIT.value
        
        # Check Time Exit (3:15 PM)
        if current_time >= time(15, 15):
            return ExitReason.TIME_EXIT.value
        
        # Check Market Reversal (for shorts, exit if Nifty turns positive)
        if trade.trade_type == "SHORT" and nifty_change > 0.3:
            return ExitReason.REVERSAL.value
        
        return None
    
    def exit_trade(self, trade_id: str, exit_price: float, reason: str) -> PaperTrade:
        """
        Exit a paper trade
        """
        if trade_id not in self.active_trades:
            raise ValueError(f"Trade {trade_id} not found")
        
        trade = self.active_trades[trade_id]
        
        # Calculate P&L
        if trade.trade_type == "LONG":
            pnl = (exit_price - trade.entry_price) * trade.quantity
        else:  # SHORT
            pnl = (trade.entry_price - exit_price) * trade.quantity
        
        pnl_pct = (pnl / (trade.entry_price * trade.quantity)) * 100
        
        # Update trade
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.exit_reason = reason
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.status = "closed"
        
        # Update capital
        self.current_capital += pnl
        
        # Remove from active trades
        del self.active_trades[trade_id]
        
        return trade
    
    def update_active_trades(self, price_data: Dict[str, float],
                            nifty_change: float) -> List[Dict]:
        """
        Update all active trades and check for exits
        Returns list of exited trades
        """
        exited_trades = []
        current_time = datetime.now().time()
        
        for trade_id in list(self.active_trades.keys()):
            trade = self.active_trades[trade_id]
            
            if trade.symbol not in price_data:
                continue
            
            current_price = price_data[trade.symbol]
            
            # Check exit conditions
            exit_reason = self.check_exit_conditions(trade_id, current_price, 
                                                      current_time, nifty_change)
            
            if exit_reason:
                exited_trade = self.exit_trade(trade_id, current_price, exit_reason)
                exited_trades.append(exited_trade.to_dict())
        
        return exited_trades
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get portfolio summary
        """
        closed_trades = [t for t in self.trades if t.status == "closed"]
        
        if not closed_trades:
            return {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'total_pnl': 0,
                'total_trades': 0,
                'win_rate': 0,
                'active_trades': len(self.active_trades)
            }
        
        winners = [t for t in closed_trades if t.pnl > 0]
        losers = [t for t in closed_trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in closed_trades)
        win_rate = len(winners) / len(closed_trades) if closed_trades else 0
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_pnl': total_pnl,
            'total_return_pct': (total_pnl / self.initial_capital) * 100,
            'total_trades': len(closed_trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': win_rate,
            'avg_win': sum(t.pnl for t in winners) / len(winners) if winners else 0,
            'avg_loss': sum(t.pnl for t in losers) / len(losers) if losers else 0,
            'active_trades': len(self.active_trades),
            'active_trade_details': [t.to_dict() for t in self.active_trades.values()]
        }
    
    def save_state(self, filepath: str):
        """Save trading state to file"""
        state = {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'trades': [t.to_dict() for t in self.trades],
            'trade_counter': self.trade_counter
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load trading state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.initial_capital = state['initial_capital']
            self.current_capital = state['current_capital']
            self.trade_counter = state['trade_counter']
            
            # Reconstruct trades
            self.trades = []
            self.active_trades = {}
            
            for trade_dict in state['trades']:
                trade = PaperTrade(
                    id=trade_dict['id'],
                    symbol=trade_dict['symbol'],
                    trade_type=trade_dict['trade_type'],
                    entry_price=trade_dict['entry_price'],
                    entry_time=datetime.fromisoformat(trade_dict['entry_time']) if trade_dict['entry_time'] else None,
                    quantity=trade_dict['quantity'],
                    exit_price=trade_dict.get('exit_price'),
                    exit_time=datetime.fromisoformat(trade_dict['exit_time']) if trade_dict.get('exit_time') else None,
                    exit_reason=trade_dict.get('exit_reason'),
                    sl_price=trade_dict.get('sl_price'),
                    target_price=trade_dict.get('target_price'),
                    status=trade_dict.get('status', 'open'),
                    pnl=trade_dict.get('pnl'),
                    pnl_pct=trade_dict.get('pnl_pct')
                )
                
                self.trades.append(trade)
                if trade.status == "open":
                    self.active_trades[trade.id] = trade
        
        except FileNotFoundError:
            pass


class TradeSimulator:
    """
    Simulates trades using historical data for backtesting
    """
    
    def __init__(self, initial_capital: float = 500000):
        self.engine = PaperTradingEngine(initial_capital)
        self.risk_manager = None  # Will be set externally
    
    def simulate_trade(self, symbol: str, trade_type: str, entry_time: datetime,
                      entry_price: float, exit_time: datetime, exit_price: float,
                      quantity: int, reason: str) -> PaperTrade:
        """
        Simulate a completed trade
        """
        trade_id = self.engine.generate_trade_id()
        
        trade = PaperTrade(
            id=trade_id,
            symbol=symbol,
            trade_type=trade_type,
            entry_price=entry_price,
            entry_time=entry_time,
            quantity=quantity,
            exit_price=exit_price,
            exit_time=exit_time,
            exit_reason=reason,
            status="closed"
        )
        
        # Calculate P&L
        if trade_type == "LONG":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        trade.pnl = pnl
        trade.pnl_pct = (pnl / (entry_price * quantity)) * 100
        
        self.engine.trades.append(trade)
        self.engine.current_capital += pnl
        
        return trade
