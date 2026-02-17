#!/usr/bin/env python3
"""
Kite Connect Day Trading Engine
Semi-auto mode: Suggests trades, waits for approval before execution.
Continuous monitoring of positions with SL/Target alerts.
Capital: ₹30,000 | Max Risk per Trade: 2% (₹600)
"""

import json
import os
import sys
import logging
import time
import threading
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
import pytz

# Check dependencies
try:
    from kiteconnect import KiteConnect
    import pandas as pd
    import numpy as np
except ImportError:
    print("Installing dependencies...")
    os.system("pip install --user kiteconnect pandas numpy")
    from kiteconnect import KiteConnect
    import pandas as pd
    import numpy as np

# ============== LOGGING SETUP ==============
LOG_DIR = "/u/tarar/repos/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create loggers
def setup_logging():
    """Setup comprehensive logging for the trading engine"""
    
    # Main trading log - all activities
    trade_logger = logging.getLogger('trading')
    trade_logger.setLevel(logging.DEBUG)
    
    # API log - all Kite API requests/responses
    api_logger = logging.getLogger('api')
    api_logger.setLevel(logging.DEBUG)
    
    # Signal log - all strategy signals
    signal_logger = logging.getLogger('signals')
    signal_logger.setLevel(logging.DEBUG)
    
    # Order log - all order placements and executions
    order_logger = logging.getLogger('orders')
    order_logger.setLevel(logging.DEBUG)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    json_formatter = logging.Formatter('%(message)s')
    
    # File handlers with rotation (10MB max, keep 5 backups)
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Trading log handler
    trade_handler = RotatingFileHandler(
        f"{LOG_DIR}/trading_{today}.log",
        maxBytes=10*1024*1024,
        backupCount=5
    )
    trade_handler.setFormatter(detailed_formatter)
    trade_logger.addHandler(trade_handler)
    
    # API log handler (JSON format for easy parsing)
    api_handler = RotatingFileHandler(
        f"{LOG_DIR}/api_{today}.log",
        maxBytes=10*1024*1024,
        backupCount=5
    )
    api_handler.setFormatter(detailed_formatter)
    api_logger.addHandler(api_handler)
    
    # Signal log handler
    signal_handler = RotatingFileHandler(
        f"{LOG_DIR}/signals_{today}.log",
        maxBytes=10*1024*1024,
        backupCount=5
    )
    signal_handler.setFormatter(detailed_formatter)
    signal_logger.addHandler(signal_handler)
    
    # Order log handler
    order_handler = RotatingFileHandler(
        f"{LOG_DIR}/orders_{today}.log",
        maxBytes=10*1024*1024,
        backupCount=5
    )
    order_handler.setFormatter(detailed_formatter)
    order_logger.addHandler(order_handler)
    
    # Also log to console for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    trade_logger.addHandler(console_handler)
    
    return trade_logger, api_logger, signal_logger, order_logger

# Initialize loggers
trade_log, api_log, signal_log, order_log = setup_logging()

def log_api_call(method, endpoint, params=None, response=None, error=None):
    """Log API request and response"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "method": method,
        "endpoint": endpoint,
        "params": params,
        "response_summary": str(response)[:500] if response else None,
        "error": str(error) if error else None
    }
    if error:
        api_log.error(f"API {method} {endpoint} | Params: {params} | Error: {error}")
    else:
        api_log.info(f"API {method} {endpoint} | Params: {params} | Response: {str(response)[:200]}...")
    return log_entry

def log_signal(symbol, strategy, signal_type, reason, indicators):
    """Log trading signal"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "strategy": strategy,
        "signal": signal_type,
        "reason": reason,
        "indicators": indicators
    }
    signal_log.info(f"SIGNAL | {symbol} | {strategy} | {signal_type} | {reason}")
    return log_entry

def log_order(action, symbol, qty, price, order_type, order_id=None, status=None, error=None):
    """Log order placement and execution"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "symbol": symbol,
        "quantity": qty,
        "price": price,
        "order_type": order_type,
        "order_id": order_id,
        "status": status,
        "error": str(error) if error else None
    }
    if error:
        order_log.error(f"ORDER {action} | {symbol} | Qty: {qty} | Error: {error}")
    else:
        order_log.info(f"ORDER {action} | {symbol} | Qty: {qty} | Price: {price} | ID: {order_id} | Status: {status}")
    return log_entry

# ============== TRADEBOOK FUNCTIONS ==============

def get_tradebook_path(date=None):
    """Get tradebook file path for a specific date"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    return f"{TRADEBOOK_DIR}/tradebook_{date}.json"

def load_tradebook(date=None):
    """Load tradebook for a specific date"""
    path = get_tradebook_path(date)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return {"date": date, "trades": [], "summary": {}}
    return {"date": date or datetime.now().strftime('%Y-%m-%d'), "trades": [], "summary": {}}

def save_tradebook(tradebook):
    """Save tradebook to file"""
    path = get_tradebook_path(tradebook.get("date"))
    with open(path, 'w') as f:
        json.dump(tradebook, f, indent=2)
    trade_log.info(f"Tradebook saved: {path}")

def record_trade(trade_data):
    """
    Record a trade in the tradebook with full details and reasoning.
    
    trade_data should contain:
    - symbol: Stock symbol
    - action: BUY/SELL
    - quantity: Number of shares
    - entry_price: Entry price
    - stop_loss: Stop loss price
    - target: Target price
    - strategy: Strategy name that generated the signal
    - reason: Why the trade was taken (signal reason)
    - indicators: Technical indicator values at entry
    - order_ids: Dict with entry, sl, target order IDs
    - margin_used: Margin utilized for this trade
    - expected_risk: Expected risk amount
    - expected_reward: Expected reward amount
    """
    tradebook = load_tradebook()
    
    trade_entry = {
        "trade_id": len(tradebook["trades"]) + 1,
        "timestamp": datetime.now().isoformat(),
        "time_ist": now_ist().strftime('%Y-%m-%d %H:%M:%S'),
        
        # Trade details
        "symbol": trade_data.get("symbol"),
        "action": trade_data.get("action"),  # BUY or SELL
        "quantity": trade_data.get("quantity"),
        "entry_price": trade_data.get("entry_price"),
        "position_value": trade_data.get("quantity", 0) * trade_data.get("entry_price", 0),
        
        # Risk management
        "stop_loss": trade_data.get("stop_loss"),
        "target": trade_data.get("target"),
        "expected_risk": trade_data.get("expected_risk"),
        "expected_reward": trade_data.get("expected_reward"),
        "risk_reward_ratio": trade_data.get("risk_reward_ratio"),
        
        # Strategy and reasoning
        "strategy": trade_data.get("strategy"),
        "reason": trade_data.get("reason"),
        "indicators": trade_data.get("indicators", {}),
        
        # Order details
        "order_ids": trade_data.get("order_ids", {}),
        "margin_used": trade_data.get("margin_used"),
        
        # Status tracking
        "status": "OPEN",
        "exit_price": None,
        "exit_time": None,
        "exit_reason": None,
        "realized_pnl": None,
        "realized_pnl_pct": None
    }
    
    tradebook["trades"].append(trade_entry)
    
    # Update summary
    update_tradebook_summary(tradebook)
    
    save_tradebook(tradebook)
    trade_log.info(f"Trade recorded in tradebook: {trade_entry['trade_id']} | {trade_entry['symbol']} | {trade_entry['action']}")
    
    return trade_entry

def update_trade_exit(symbol, exit_price, exit_reason="MANUAL"):
    """
    Update a trade when it exits (SL hit, Target hit, or manual exit)
    
    exit_reason: SL_HIT, TARGET_HIT, MANUAL, EOD_SQUAREOFF
    """
    tradebook = load_tradebook()
    
    for trade in reversed(tradebook["trades"]):  # Start from most recent
        if trade["symbol"] == symbol and trade["status"] == "OPEN":
            trade["status"] = "CLOSED"
            trade["exit_price"] = exit_price
            trade["exit_time"] = datetime.now().isoformat()
            trade["exit_reason"] = exit_reason
            
            # Calculate P&L
            if trade["action"] == "BUY":
                pnl = (exit_price - trade["entry_price"]) * trade["quantity"]
                pnl_pct = ((exit_price - trade["entry_price"]) / trade["entry_price"]) * 100
            else:  # SELL (short)
                pnl = (trade["entry_price"] - exit_price) * trade["quantity"]
                pnl_pct = ((trade["entry_price"] - exit_price) / trade["entry_price"]) * 100
            
            trade["realized_pnl"] = round(pnl, 2)
            trade["realized_pnl_pct"] = round(pnl_pct, 2)
            
            # Update summary
            update_tradebook_summary(tradebook)
            save_tradebook(tradebook)
            
            trade_log.info(f"Trade exit recorded: {symbol} | {exit_reason} | P&L: ₹{pnl:.2f} ({pnl_pct:.2f}%)")
            return trade
    
    return None

def update_tradebook_summary(tradebook):
    """Update tradebook summary statistics"""
    trades = tradebook["trades"]
    closed_trades = [t for t in trades if t["status"] == "CLOSED"]
    open_trades = [t for t in trades if t["status"] == "OPEN"]
    
    if closed_trades:
        winning_trades = [t for t in closed_trades if t.get("realized_pnl", 0) > 0]
        losing_trades = [t for t in closed_trades if t.get("realized_pnl", 0) < 0]
        
        total_pnl = sum(t.get("realized_pnl", 0) for t in closed_trades)
        total_wins = sum(t.get("realized_pnl", 0) for t in winning_trades)
        total_losses = sum(t.get("realized_pnl", 0) for t in losing_trades)
        
        tradebook["summary"] = {
            "total_trades": len(trades),
            "open_trades": len(open_trades),
            "closed_trades": len(closed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round(len(winning_trades) / len(closed_trades) * 100, 2) if closed_trades else 0,
            "total_pnl": round(total_pnl, 2),
            "total_wins": round(total_wins, 2),
            "total_losses": round(total_losses, 2),
            "avg_win": round(total_wins / len(winning_trades), 2) if winning_trades else 0,
            "avg_loss": round(total_losses / len(losing_trades), 2) if losing_trades else 0,
            "profit_factor": round(abs(total_wins / total_losses), 2) if total_losses != 0 else float('inf'),
            "largest_win": max((t.get("realized_pnl", 0) for t in closed_trades), default=0),
            "largest_loss": min((t.get("realized_pnl", 0) for t in closed_trades), default=0),
            "strategies_used": list(set(t.get("strategy", "Unknown") for t in trades)),
            "last_updated": datetime.now().isoformat()
        }
    else:
        tradebook["summary"] = {
            "total_trades": len(trades),
            "open_trades": len(open_trades),
            "closed_trades": 0,
            "total_pnl": 0,
            "last_updated": datetime.now().isoformat()
        }

def display_tradebook(date=None):
    """Display tradebook for a specific date"""
    tradebook = load_tradebook(date)
    
    print("\n" + "═" * 90)
    print(f"  📒 TRADEBOOK - {tradebook.get('date', 'Today')}")
    print("═" * 90)
    
    if not tradebook["trades"]:
        print("  No trades recorded for this date.")
        return
    
    # Display trades
    print(f"\n  {'#':<3} {'Time':<8} {'Symbol':<12} {'Action':<6} {'Qty':>5} {'Entry':>9} {'SL':>9} {'Target':>9} {'Status':<8} {'P&L':>10}")
    print("  " + "-" * 88)
    
    for trade in tradebook["trades"]:
        time_str = trade.get("time_ist", "")[-8:-3] if trade.get("time_ist") else ""  # HH:MM
        pnl_str = f"₹{trade.get('realized_pnl', 0):,.2f}" if trade.get("realized_pnl") is not None else "---"
        status_emoji = "✅" if trade["status"] == "CLOSED" and trade.get("realized_pnl", 0) > 0 else \
                       "❌" if trade["status"] == "CLOSED" and trade.get("realized_pnl", 0) < 0 else "🔄"
        
        print(f"  {trade['trade_id']:<3} {time_str:<8} {trade['symbol']:<12} {trade['action']:<6} "
              f"{trade['quantity']:>5} ₹{trade['entry_price']:>7.2f} ₹{trade['stop_loss']:>7.2f} "
              f"₹{trade['target']:>7.2f} {status_emoji}{trade['status']:<7} {pnl_str:>10}")
    
    # Display summary
    summary = tradebook.get("summary", {})
    if summary:
        print("\n  " + "-" * 88)
        print("  📊 SUMMARY")
        print(f"     Total Trades: {summary.get('total_trades', 0)} | "
              f"Open: {summary.get('open_trades', 0)} | "
              f"Closed: {summary.get('closed_trades', 0)}")
        
        if summary.get('closed_trades', 0) > 0:
            print(f"     Win Rate: {summary.get('win_rate', 0):.1f}% | "
                  f"Winners: {summary.get('winning_trades', 0)} | "
                  f"Losers: {summary.get('losing_trades', 0)}")
            print(f"     Total P&L: ₹{summary.get('total_pnl', 0):,.2f} | "
                  f"Profit Factor: {summary.get('profit_factor', 0):.2f}")
            print(f"     Avg Win: ₹{summary.get('avg_win', 0):,.2f} | "
                  f"Avg Loss: ₹{summary.get('avg_loss', 0):,.2f}")
            print(f"     Largest Win: ₹{summary.get('largest_win', 0):,.2f} | "
                  f"Largest Loss: ₹{summary.get('largest_loss', 0):,.2f}")
        
        if summary.get('strategies_used'):
            print(f"     Strategies: {', '.join(summary['strategies_used'])}")
    
    print("═" * 90)

def display_trade_details(trade_id, date=None):
    """Display detailed information about a specific trade"""
    tradebook = load_tradebook(date)
    
    trade = None
    for t in tradebook["trades"]:
        if t["trade_id"] == trade_id:
            trade = t
            break
    
    if not trade:
        print(f"  Trade #{trade_id} not found.")
        return
    
    print("\n" + "═" * 70)
    print(f"  📋 TRADE DETAILS - #{trade['trade_id']}")
    print("═" * 70)
    
    print(f"\n  📌 BASIC INFO")
    print(f"     Symbol: {trade['symbol']}")
    print(f"     Action: {trade['action']}")
    print(f"     Quantity: {trade['quantity']}")
    print(f"     Entry Time: {trade.get('time_ist', 'N/A')}")
    print(f"     Entry Price: ₹{trade['entry_price']:.2f}")
    print(f"     Position Value: ₹{trade.get('position_value', 0):,.2f}")
    
    print(f"\n  🎯 RISK MANAGEMENT")
    print(f"     Stop Loss: ₹{trade['stop_loss']:.2f}")
    print(f"     Target: ₹{trade['target']:.2f}")
    print(f"     Expected Risk: ₹{trade.get('expected_risk', 'N/A')}")
    print(f"     Expected Reward: ₹{trade.get('expected_reward', 'N/A')}")
    print(f"     Risk:Reward: 1:{trade.get('risk_reward_ratio', 'N/A')}")
    
    print(f"\n  🧠 STRATEGY & REASONING")
    print(f"     Strategy: {trade.get('strategy', 'N/A')}")
    print(f"     Reason: {trade.get('reason', 'N/A')}")
    
    if trade.get('indicators'):
        print(f"\n  📊 INDICATORS AT ENTRY")
        for key, value in trade['indicators'].items():
            if isinstance(value, float):
                print(f"     {key}: {value:.2f}")
            else:
                print(f"     {key}: {value}")
    
    print(f"\n  📝 ORDER IDs")
    order_ids = trade.get('order_ids', {})
    print(f"     Entry: {order_ids.get('entry', 'N/A')}")
    print(f"     Stop Loss: {order_ids.get('sl', 'N/A')}")
    print(f"     Target: {order_ids.get('target', 'N/A')}")
    
    print(f"\n  📈 STATUS")
    print(f"     Status: {trade['status']}")
    if trade['status'] == "CLOSED":
        print(f"     Exit Price: ₹{trade.get('exit_price', 'N/A')}")
        print(f"     Exit Time: {trade.get('exit_time', 'N/A')}")
        print(f"     Exit Reason: {trade.get('exit_reason', 'N/A')}")
        pnl = trade.get('realized_pnl', 0)
        pnl_pct = trade.get('realized_pnl_pct', 0)
        emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
        print(f"     Realized P&L: {emoji} ₹{pnl:,.2f} ({pnl_pct:+.2f}%)")
    
    print("═" * 70)

def list_tradebook_dates():
    """List all available tradebook dates"""
    import glob
    files = glob.glob(f"{TRADEBOOK_DIR}/tradebook_*.json")
    dates = []
    for f in files:
        # Extract date from filename
        basename = os.path.basename(f)
        date = basename.replace("tradebook_", "").replace(".json", "")
        dates.append(date)
    return sorted(dates, reverse=True)

# ============== CONFIGURATION ==============
CREDS_FILE = "/u/tarar/repos/.kite_creds.json"
SESSION_FILE = "/u/tarar/repos/.kite_session.json"

CAPITAL = 30000
MAX_RISK_PER_TRADE = 0.02  # 2%
MAX_RISK_AMOUNT = CAPITAL * MAX_RISK_PER_TRADE  # ₹600
MAX_DAILY_LOSS = 0.05  # 5% kill switch
MAX_CONCURRENT_TRADES = 3
MIN_RR_RATIO = 2.0  # Minimum reward:risk ratio

# Monitoring settings
MONITOR_INTERVAL = 30  # seconds between position checks
SCAN_INTERVAL = 300  # seconds between new signal scans (5 min)

IST = pytz.timezone('Asia/Kolkata')

# Position tracking file for SL/Target
POSITIONS_FILE = "/u/tarar/repos/.tracked_positions.json"

# Tradebook file - stores all trades with reasoning
TRADEBOOK_DIR = "/u/tarar/repos/tradebook"
os.makedirs(TRADEBOOK_DIR, exist_ok=True)

# High liquidity F&O stocks for day trading
WATCHLIST = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
    "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "ASIANPAINT",
    "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO",
    "TATASTEEL", "POWERGRID", "NTPC", "M&M", "JSWSTEEL"
]

# NSE Holidays 2026
NSE_HOLIDAYS_2026 = [
    "2026-01-26", "2026-03-10", "2026-03-17", "2026-04-02",
    "2026-04-03", "2026-04-06", "2026-04-14", "2026-05-01",
    "2026-07-17", "2026-08-15", "2026-08-26", "2026-10-02",
    "2026-10-20", "2026-10-21", "2026-11-05", "2026-12-25"
]

# ============== HELPER FUNCTIONS ==============

# Cache for instrument data (tick size, lot size, etc.)
INSTRUMENT_CACHE = {}

def now_ist():
    """Get current time in IST"""
    return datetime.now(IST)

def get_instrument_info(kite, symbol, exchange="NSE"):
    """Get instrument info including tick size from Kite"""
    global INSTRUMENT_CACHE
    
    cache_key = f"{exchange}:{symbol}"
    if cache_key in INSTRUMENT_CACHE:
        return INSTRUMENT_CACHE[cache_key]
    
    try:
        instruments = kite.instruments(exchange)
        for inst in instruments:
            if inst['tradingsymbol'] == symbol:
                INSTRUMENT_CACHE[cache_key] = inst
                api_log.debug(f"Instrument info for {symbol}: tick_size={inst.get('tick_size')}, lot_size={inst.get('lot_size')}")
                return inst
    except Exception as e:
        api_log.error(f"Failed to get instrument info for {symbol}: {e}")
    
    return None

def round_to_tick_size(price, tick_size):
    """Round price to the nearest tick size"""
    if tick_size <= 0:
        return round(price, 2)
    return round(round(price / tick_size) * tick_size, 2)

def get_tick_size(kite, symbol, exchange="NSE"):
    """Get tick size for a symbol"""
    inst = get_instrument_info(kite, symbol, exchange)
    if inst:
        return inst.get('tick_size', 0.05)
    return 0.05  # Default tick size

def check_available_margin(kite):
    """Check available margin for trading"""
    try:
        margins = kite.margins()
        equity = margins.get('equity', {})
        available = equity.get('available', {})
        
        cash = available.get('live_balance', 0)
        collateral = available.get('collateral', 0)
        
        # Get utilized margin
        utilized = equity.get('utilised', {})
        used = utilized.get('debits', 0)
        
        total_available = cash + collateral - used
        
        api_log.info(f"Margin check: Cash=₹{cash:.2f}, Collateral=₹{collateral:.2f}, Used=₹{used:.2f}, Available=₹{total_available:.2f}")
        return total_available
    except Exception as e:
        api_log.error(f"Failed to check margin: {e}")
        return 0

def get_order_margin_required(kite, symbol, qty, transaction_type, product="MIS"):
    """Get margin required for an order using Kite's order margins API"""
    try:
        order_params = [{
            "exchange": "NSE",
            "tradingsymbol": symbol,
            "transaction_type": transaction_type,
            "variety": "regular",
            "product": product,
            "order_type": "MARKET",
            "quantity": qty
        }]
        
        margin_info = kite.order_margins(order_params)
        if margin_info and len(margin_info) > 0:
            total_margin = margin_info[0].get('total', 0)
            api_log.info(f"Margin required for {symbol} x {qty}: ₹{total_margin:.2f}")
            return total_margin
    except Exception as e:
        api_log.warning(f"Could not get order margin for {symbol}: {e}")
        # Fallback: estimate based on position value (assume 20% margin for MIS)
        try:
            ltp = get_ltp(kite, symbol)
            if ltp:
                estimated = ltp * qty * 0.2  # 20% margin estimate
                api_log.info(f"Estimated margin for {symbol} x {qty}: ₹{estimated:.2f}")
                return estimated
        except:
            pass
    
    return float('inf')  # Return infinity if we can't determine margin

def validate_order_before_placement(kite, symbol, qty, ltp, transaction_type):
    """
    Validate order before placement:
    1. Check available margin
    2. Check position value doesn't exceed capital
    3. Return adjusted quantity if needed
    """
    # Get available margin
    available_margin = check_available_margin(kite)
    
    # Get required margin for this order
    required_margin = get_order_margin_required(kite, symbol, qty, transaction_type)
    
    position_value = qty * ltp
    
    validation = {
        "valid": True,
        "original_qty": qty,
        "adjusted_qty": qty,
        "available_margin": available_margin,
        "required_margin": required_margin,
        "position_value": position_value,
        "reason": ""
    }
    
    # Check if we have enough margin
    if required_margin > available_margin:
        # Try to adjust quantity to fit available margin
        if available_margin > 0 and ltp > 0:
            # For MIS, margin is roughly 20% of position value
            max_position_value = available_margin / 0.2
            adjusted_qty = int(max_position_value / ltp)
            
            if adjusted_qty < 1:
                validation["valid"] = False
                validation["reason"] = f"Insufficient margin. Need ₹{required_margin:.2f}, have ₹{available_margin:.2f}"
            else:
                validation["adjusted_qty"] = adjusted_qty
                validation["reason"] = f"Quantity adjusted from {qty} to {adjusted_qty} due to margin"
                order_log.warning(f"{symbol}: {validation['reason']}")
        else:
            validation["valid"] = False
            validation["reason"] = "No available margin"
    
    # Check if position value exceeds 90% of capital (safety limit)
    max_position = CAPITAL * 0.9
    if validation["adjusted_qty"] * ltp > max_position:
        validation["adjusted_qty"] = int(max_position / ltp)
        validation["reason"] = f"Quantity limited to {validation['adjusted_qty']} (90% of capital)"
    
    trade_log.info(f"Order validation for {symbol}: valid={validation['valid']}, qty={validation['adjusted_qty']}, reason={validation['reason']}")
    return validation

def is_market_open():
    """Check if NSE market is currently open"""
    now = now_ist()
    
    # Check weekend
    if now.weekday() >= 5:
        return False, "Weekend"
    
    # Check holiday
    today_str = now.strftime("%Y-%m-%d")
    if today_str in NSE_HOLIDAYS_2026:
        return False, "NSE Holiday"
    
    # Check market hours (9:15 AM - 3:30 PM IST)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if now < market_open:
        return False, f"Pre-market (opens at 9:15 AM)"
    if now > market_close:
        return False, "Market closed (after 3:30 PM)"
    
    return True, "Market Open"

def is_safe_trading_window():
    """Check if we're in safe trading hours (avoid first 15 min and last 20 min)"""
    now = now_ist()
    safe_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    safe_end = now.replace(hour=15, minute=10, second=0, microsecond=0)
    
    if now < safe_start:
        return False, "Avoid first 15 minutes - high volatility"
    if now > safe_end:
        return False, "Avoid last 20 minutes - square off zone"
    
    return True, "Safe trading window"

def load_session():
    """Load saved Kite session"""
    if not os.path.exists(SESSION_FILE):
        return None
    with open(SESSION_FILE, 'r') as f:
        return json.load(f)

def init_kite():
    """Initialize Kite Connect with saved session"""
    trade_log.info("="*60)
    trade_log.info("TRADING ENGINE STARTUP")
    trade_log.info("="*60)
    
    session = load_session()
    if not session:
        trade_log.error("No session found. Run kite_login.py first!")
        print("❌ No session found. Run kite_login.py first!")
        sys.exit(1)
    
    trade_log.info(f"Loaded session for API key: {session['api_key'][:8]}...")
    
    kite = KiteConnect(api_key=session['api_key'])
    kite.set_access_token(session['access_token'])
    
    # Test connection
    try:
        api_log.info("Testing connection - fetching profile")
        profile = kite.profile()
        log_api_call("GET", "profile", response=profile)
        trade_log.info(f"Connected as: {profile['user_name']} ({profile['email']})")
        print(f"✅ Connected as: {profile['user_name']}")
        return kite
    except Exception as e:
        log_api_call("GET", "profile", error=e)
        trade_log.error(f"Session expired or invalid: {e}")
        print(f"❌ Session expired or invalid: {e}")
        print("Run kite_login.py to refresh session")
        sys.exit(1)

def get_historical_data(kite, symbol, interval="5minute", days=5):
    """Fetch historical data from Kite"""
    try:
        # Get instrument token
        api_log.debug(f"Fetching instruments for NSE to find {symbol}")
        instruments = kite.instruments("NSE")
        token = None
        for inst in instruments:
            if inst['tradingsymbol'] == symbol:
                token = inst['instrument_token']
                break
        
        if not token:
            api_log.warning(f"Instrument token not found for {symbol}")
            return None
        
        # Fetch historical data
        to_date = now_ist()
        from_date = to_date - timedelta(days=days)
        
        params = {
            "instrument_token": token,
            "from_date": from_date.strftime('%Y-%m-%d'),
            "to_date": to_date.strftime('%Y-%m-%d'),
            "interval": interval
        }
        api_log.info(f"Fetching historical data for {symbol} | {params}")
        
        data = kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )
        
        log_api_call("GET", "historical_data", params=params, response=f"{len(data)} candles")
        
        df = pd.DataFrame(data)
        if len(df) > 0:
            df.set_index('date', inplace=True)
            api_log.debug(f"{symbol}: Got {len(df)} candles, latest: {df.index[-1]}")
        return df
    except Exception as e:
        log_api_call("GET", "historical_data", params={"symbol": symbol}, error=e)
        print(f"  ⚠️ Error fetching {symbol}: {e}")
        return None

def get_ltp(kite, symbol):
    """Get Last Traded Price"""
    try:
        quote = kite.quote(f"NSE:{symbol}")
        ltp = quote[f"NSE:{symbol}"]["last_price"]
        api_log.debug(f"LTP {symbol}: ₹{ltp}")
        return ltp
    except Exception as e:
        api_log.warning(f"Failed to get LTP for {symbol}: {e}")
        return None

# ============== POSITION TRACKING ==============

def load_tracked_positions():
    """Load tracked positions with SL/Target from file"""
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_tracked_positions(positions):
    """Save tracked positions to file"""
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(positions, f, indent=2)
    trade_log.debug(f"Saved {len(positions)} tracked positions")

def add_tracked_position(symbol, qty, entry_price, stop_loss, target, position_type="LONG"):
    """Add a position to tracking"""
    positions = load_tracked_positions()
    positions[symbol] = {
        "symbol": symbol,
        "quantity": qty,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target": target,
        "type": position_type,  # LONG or SHORT
        "entry_time": datetime.now().isoformat(),
        "status": "ACTIVE"
    }
    save_tracked_positions(positions)
    trade_log.info(f"Tracking position: {symbol} | {position_type} | Entry: ₹{entry_price} | SL: ₹{stop_loss} | Target: ₹{target}")

def remove_tracked_position(symbol):
    """Remove a position from tracking"""
    positions = load_tracked_positions()
    if symbol in positions:
        del positions[symbol]
        save_tracked_positions(positions)
        trade_log.info(f"Removed tracking for {symbol}")

def get_positions_with_sl_target(kite):
    """
    Get current positions and merge with tracked SL/Target info.
    Also auto-detect positions that need SL/Target setup.
    """
    try:
        positions = kite.positions()
        day_positions = positions.get('day', [])
        net_positions = positions.get('net', [])
        
        # Combine day and net positions
        all_positions = []
        seen = set()
        
        for pos in day_positions + net_positions:
            if pos['quantity'] != 0 and pos['tradingsymbol'] not in seen:
                seen.add(pos['tradingsymbol'])
                all_positions.append(pos)
        
        # Load tracked positions
        tracked = load_tracked_positions()
        
        # Merge tracking info
        result = []
        for pos in all_positions:
            symbol = pos['tradingsymbol']
            qty = pos['quantity']
            avg_price = pos['average_price']
            ltp = pos['last_price']
            pnl = pos['pnl']
            
            position_data = {
                "symbol": symbol,
                "quantity": qty,
                "average_price": avg_price,
                "ltp": ltp,
                "pnl": pnl,
                "pnl_pct": ((ltp - avg_price) / avg_price * 100) if avg_price > 0 else 0,
                "type": "LONG" if qty > 0 else "SHORT",
                "product": pos.get('product', 'CNC'),
                "exchange": pos.get('exchange', 'NSE')
            }
            
            # Check if we have tracking info
            if symbol in tracked:
                t = tracked[symbol]
                position_data["stop_loss"] = t.get("stop_loss")
                position_data["target"] = t.get("target")
                position_data["tracked"] = True
                
                # Calculate distance to SL and Target
                if position_data["type"] == "LONG":
                    position_data["sl_distance_pct"] = ((ltp - t["stop_loss"]) / ltp * 100) if t.get("stop_loss") else None
                    position_data["target_distance_pct"] = ((t["target"] - ltp) / ltp * 100) if t.get("target") else None
                else:  # SHORT
                    position_data["sl_distance_pct"] = ((t["stop_loss"] - ltp) / ltp * 100) if t.get("stop_loss") else None
                    position_data["target_distance_pct"] = ((ltp - t["target"]) / ltp * 100) if t.get("target") else None
            else:
                position_data["tracked"] = False
                position_data["stop_loss"] = None
                position_data["target"] = None
            
            result.append(position_data)
        
        return result
    except Exception as e:
        trade_log.error(f"Error fetching positions: {e}")
        return []

def cancel_opposite_order(kite, symbol, order_type_to_cancel):
    """
    When SL is hit, cancel Target order and vice versa.
    This implements OCO (One-Cancels-Other) logic.
    
    order_type_to_cancel: 'SL' or 'TARGET'
    """
    try:
        orders = kite.orders()
        for order in orders:
            if order['tradingsymbol'] == symbol and order['status'] in ['TRIGGER PENDING', 'OPEN']:
                # Determine if this is the order we want to cancel
                is_sl_order = order['order_type'] == 'SL-M'
                is_target_order = order['order_type'] == 'LIMIT'
                
                should_cancel = False
                if order_type_to_cancel == 'SL' and is_sl_order:
                    should_cancel = True
                elif order_type_to_cancel == 'TARGET' and is_target_order:
                    should_cancel = True
                
                if should_cancel:
                    try:
                        kite.cancel_order(
                            variety=order['variety'],
                            order_id=order['order_id']
                        )
                        order_log.info(f"OCO: Cancelled {order_type_to_cancel} order {order['order_id']} for {symbol}")
                        return True
                    except Exception as e:
                        order_log.error(f"Failed to cancel {order_type_to_cancel} order for {symbol}: {e}")
    except Exception as e:
        order_log.error(f"Error in cancel_opposite_order for {symbol}: {e}")
    return False

def check_sl_target_hits(kite, positions, auto_update_tradebook=True):
    """
    Check if any position has hit SL or Target.
    Returns list of alerts.
    If auto_update_tradebook=True, automatically records exit in tradebook.
    """
    alerts = []
    
    for pos in positions:
        if not pos.get("tracked"):
            continue
        
        symbol = pos["symbol"]
        ltp = pos["ltp"]
        sl = pos.get("stop_loss")
        target = pos.get("target")
        pos_type = pos["type"]
        
        if pos_type == "LONG":
            # Check SL hit
            if sl and ltp <= sl:
                alerts.append({
                    "type": "SL_HIT",
                    "symbol": symbol,
                    "ltp": ltp,
                    "sl": sl,
                    "message": f"🔴 STOP LOSS HIT: {symbol} @ ₹{ltp:.2f} (SL was ₹{sl:.2f})"
                })
                # Auto-update tradebook and cancel opposite order
                if auto_update_tradebook:
                    update_trade_exit(symbol, ltp, "SL_HIT")
                    cancel_opposite_order(kite, symbol, 'TARGET')  # Cancel target since SL hit
                    remove_tracked_position(symbol)  # Remove from tracking
                    
            # Check Target hit
            elif target and ltp >= target:
                alerts.append({
                    "type": "TARGET_HIT", 
                    "symbol": symbol,
                    "ltp": ltp,
                    "target": target,
                    "message": f"🎯 TARGET HIT: {symbol} @ ₹{ltp:.2f} (Target was ₹{target:.2f})"
                })
                # Auto-update tradebook and cancel opposite order
                if auto_update_tradebook:
                    update_trade_exit(symbol, ltp, "TARGET_HIT")
                    cancel_opposite_order(kite, symbol, 'SL')  # Cancel SL since target hit
                    remove_tracked_position(symbol)  # Remove from tracking
                    
            # Check approaching SL (within 0.5%)
            elif sl and ltp <= sl * 1.005:
                alerts.append({
                    "type": "APPROACHING_SL",
                    "symbol": symbol,
                    "ltp": ltp,
                    "sl": sl,
                    "message": f"⚠️ APPROACHING SL: {symbol} @ ₹{ltp:.2f} (SL: ₹{sl:.2f})"
                })
            # Check approaching Target (within 0.5%)
            elif target and ltp >= target * 0.995:
                alerts.append({
                    "type": "APPROACHING_TARGET",
                    "symbol": symbol,
                    "ltp": ltp,
                    "target": target,
                    "message": f"📈 APPROACHING TARGET: {symbol} @ ₹{ltp:.2f} (Target: ₹{target:.2f})"
                })
        
        else:  # SHORT position
            # Check SL hit
            if sl and ltp >= sl:
                alerts.append({
                    "type": "SL_HIT",
                    "symbol": symbol,
                    "ltp": ltp,
                    "sl": sl,
                    "message": f"🔴 STOP LOSS HIT: {symbol} @ ₹{ltp:.2f} (SL was ₹{sl:.2f})"
                })
                # Auto-update tradebook and cancel opposite order
                if auto_update_tradebook:
                    update_trade_exit(symbol, ltp, "SL_HIT")
                    cancel_opposite_order(kite, symbol, 'TARGET')  # Cancel target since SL hit
                    remove_tracked_position(symbol)  # Remove from tracking
                    
            # Check Target hit
            elif target and ltp <= target:
                alerts.append({
                    "type": "TARGET_HIT",
                    "symbol": symbol,
                    "ltp": ltp,
                    "target": target,
                    "message": f"🎯 TARGET HIT: {symbol} @ ₹{ltp:.2f} (Target was ₹{target:.2f})"
                })
                # Auto-update tradebook and cancel opposite order
                if auto_update_tradebook:
                    update_trade_exit(symbol, ltp, "TARGET_HIT")
                    cancel_opposite_order(kite, symbol, 'SL')  # Cancel SL since target hit
                    remove_tracked_position(symbol)  # Remove from tracking
                    
            # Check approaching SL
            elif sl and ltp >= sl * 0.995:
                alerts.append({
                    "type": "APPROACHING_SL",
                    "symbol": symbol,
                    "ltp": ltp,
                    "sl": sl,
                    "message": f"⚠️ APPROACHING SL: {symbol} @ ₹{ltp:.2f} (SL: ₹{sl:.2f})"
                })
            # Check approaching Target
            elif target and ltp <= target * 1.005:
                alerts.append({
                    "type": "APPROACHING_TARGET",
                    "symbol": symbol,
                    "ltp": ltp,
                    "target": target,
                    "message": f"📈 APPROACHING TARGET: {symbol} @ ₹{ltp:.2f} (Target: ₹{target:.2f})"
                })
    
    return alerts

def display_positions_dashboard(positions, total_pnl):
    """Display positions in a nice dashboard format"""
    now = now_ist()
    
    # Clear screen for fresh display
    if sys.stdout.isatty():
        os.system('clear')
    
    print("\n" + "═" * 80)
    print(f"  📊 LIVE POSITION MONITOR | {now.strftime('%Y-%m-%d %H:%M:%S')} IST")
    print("═" * 80)
    
    if not positions:
        print("\n  No open positions.")
    else:
        print(f"\n  {'Symbol':<12} {'Type':<6} {'Qty':>6} {'Entry':>10} {'LTP':>10} {'P&L':>12} {'SL':>10} {'Target':>10}")
        print("  " + "-" * 76)
        
        for pos in positions:
            pnl_str = f"₹{pos['pnl']:,.2f}"
            pnl_color = "🟢" if pos['pnl'] >= 0 else "🔴"
            
            sl_str = f"₹{pos['stop_loss']:.2f}" if pos.get('stop_loss') else "---"
            tgt_str = f"₹{pos['target']:.2f}" if pos.get('target') else "---"
            tracked_mark = "✓" if pos.get('tracked') else "⚠"
            
            print(f"  {pos['symbol']:<12} {pos['type']:<6} {abs(pos['quantity']):>6} "
                  f"₹{pos['average_price']:>8.2f} ₹{pos['ltp']:>8.2f} "
                  f"{pnl_color}{pnl_str:>10} {sl_str:>10} {tgt_str:>10} {tracked_mark}")
    
    print("\n  " + "-" * 76)
    pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
    print(f"  Total P&L: {pnl_emoji} ₹{total_pnl:,.2f}")
    
    # Check kill switch
    if total_pnl < -CAPITAL * MAX_DAILY_LOSS:
        print(f"\n  🚨 KILL SWITCH WARNING: Loss exceeds {MAX_DAILY_LOSS*100}% of capital!")
    
    print("═" * 80)

def setup_sl_target_interactive(kite, pos):
    """Interactively set up SL and Target for a position"""
    symbol = pos['symbol']
    ltp = pos['ltp']
    avg_price = pos['average_price']
    qty = pos['quantity']
    pos_type = pos['type']
    
    print(f"\n┌─ Setup SL/Target for {symbol} ───────────────────────────")
    print(f"│  Type: {pos_type}")
    print(f"│  Quantity: {abs(qty)}")
    print(f"│  Entry Price: ₹{avg_price:.2f}")
    print(f"│  Current LTP: ₹{ltp:.2f}")
    print(f"│  Current P&L: ₹{pos['pnl']:.2f}")
    print(f"└──────────────────────────────────────────────────────────")
    
    # Suggest SL/Target based on ATR or percentage
    if pos_type == "LONG":
        suggested_sl = round(avg_price * 0.99, 2)  # 1% below entry
        suggested_target = round(avg_price * 1.02, 2)  # 2% above entry
    else:
        suggested_sl = round(avg_price * 1.01, 2)  # 1% above entry
        suggested_target = round(avg_price * 0.98, 2)  # 2% below entry
    
    print(f"\n  Suggested SL: ₹{suggested_sl} | Suggested Target: ₹{suggested_target}")
    
    try:
        sl_input = input(f"  Enter Stop Loss (or press Enter for ₹{suggested_sl}): ").strip()
        sl = float(sl_input) if sl_input else suggested_sl
        
        target_input = input(f"  Enter Target (or press Enter for ₹{suggested_target}): ").strip()
        target = float(target_input) if target_input else suggested_target
        
        # Validate
        if pos_type == "LONG":
            if sl >= ltp:
                print(f"  ⚠️ Warning: SL (₹{sl}) is above current price (₹{ltp})")
            if target <= ltp:
                print(f"  ⚠️ Warning: Target (₹{target}) is below current price (₹{ltp})")
        else:
            if sl <= ltp:
                print(f"  ⚠️ Warning: SL (₹{sl}) is below current price (₹{ltp})")
            if target >= ltp:
                print(f"  ⚠️ Warning: Target (₹{target}) is above current price (₹{ltp})")
        
        # Confirm
        confirm = input(f"  Confirm SL: ₹{sl}, Target: ₹{target}? (y/n): ").strip().lower()
        if confirm == 'y':
            add_tracked_position(symbol, qty, avg_price, sl, target, pos_type)
            print(f"  ✅ SL/Target set for {symbol}")
            return True
        else:
            print(f"  ❌ Cancelled")
            return False
            
    except (ValueError, EOFError) as e:
        print(f"  ❌ Invalid input: {e}")
        return False

# ============== TECHNICAL INDICATORS ==============

def calculate_rsi(df, period=14):
    """Calculate RSI"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(df, period):
    """Calculate EMA"""
    return df['close'].ewm(span=period, adjust=False).mean()

def calculate_vwap(df):
    """Calculate VWAP"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap

def calculate_supertrend(df, period=10, multiplier=3):
    """Calculate Supertrend"""
    hl2 = (df['high'] + df['low']) / 2
    atr = df['high'].rolling(period).max() - df['low'].rolling(period).min()
    atr = atr.rolling(period).mean()
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(period, len(df)):
        if df['close'].iloc[i] > upper_band.iloc[i-1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1  # Bullish
        elif df['close'].iloc[i] < lower_band.iloc[i-1]:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1  # Bearish
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]
    
    return supertrend, direction

def calculate_atr(df, period=14):
    """Calculate ATR for stop loss"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ============== TRADING STRATEGIES ==============

def strategy_vwap_rsi_reversal(df, ltp, symbol=""):
    """
    Strategy 1: VWAP + RSI Reversal
    BUY: Price below VWAP + RSI < 30 (oversold)
    SELL: Price above VWAP + RSI > 70 (overbought)
    """
    if len(df) < 20:
        return None
    
    vwap = calculate_vwap(df).iloc[-1]
    rsi = calculate_rsi(df).iloc[-1]
    
    indicators = {"vwap": round(vwap, 2), "rsi": round(rsi, 2), "ltp": ltp}
    signal = None
    reason = ""
    
    if ltp < vwap * 0.995 and rsi < 30:
        signal = "BUY"
        reason = f"Price below VWAP ({ltp:.2f} < {vwap:.2f}), RSI oversold ({rsi:.1f})"
        log_signal(symbol, "VWAP+RSI Reversal", signal, reason, indicators)
    elif ltp > vwap * 1.005 and rsi > 70:
        signal = "SELL"
        reason = f"Price above VWAP ({ltp:.2f} > {vwap:.2f}), RSI overbought ({rsi:.1f})"
        log_signal(symbol, "VWAP+RSI Reversal", signal, reason, indicators)
    
    return {"signal": signal, "reason": reason, "strategy": "VWAP+RSI Reversal", "indicators": indicators}

def strategy_ema_crossover(df, ltp, symbol=""):
    """
    Strategy 2: EMA 9/21 Crossover
    BUY: EMA9 crosses above EMA21
    SELL: EMA9 crosses below EMA21
    """
    if len(df) < 25:
        return None
    
    ema9 = calculate_ema(df, 9)
    ema21 = calculate_ema(df, 21)
    
    # Check for crossover in last 2 candles
    prev_diff = ema9.iloc[-2] - ema21.iloc[-2]
    curr_diff = ema9.iloc[-1] - ema21.iloc[-1]
    
    indicators = {
        "ema9": round(ema9.iloc[-1], 2), 
        "ema21": round(ema21.iloc[-1], 2),
        "prev_diff": round(prev_diff, 2),
        "curr_diff": round(curr_diff, 2),
        "ltp": ltp
    }
    signal = None
    reason = ""
    
    if prev_diff < 0 and curr_diff > 0:
        signal = "BUY"
        reason = f"EMA9 ({ema9.iloc[-1]:.2f}) crossed above EMA21 ({ema21.iloc[-1]:.2f})"
        log_signal(symbol, "EMA 9/21 Crossover", signal, reason, indicators)
    elif prev_diff > 0 and curr_diff < 0:
        signal = "SELL"
        reason = f"EMA9 ({ema9.iloc[-1]:.2f}) crossed below EMA21 ({ema21.iloc[-1]:.2f})"
        log_signal(symbol, "EMA 9/21 Crossover", signal, reason, indicators)
    
    return {"signal": signal, "reason": reason, "strategy": "EMA 9/21 Crossover", "indicators": indicators}

def strategy_supertrend_scalp(df, ltp, symbol=""):
    """
    Strategy 3: Supertrend Scalping
    BUY: Supertrend turns bullish
    SELL: Supertrend turns bearish
    """
    if len(df) < 15:
        return None
    
    supertrend, direction = calculate_supertrend(df)
    
    signal = None
    reason = ""
    indicators = {"ltp": ltp, "supertrend": None, "direction": None}
    
    # Check for direction change
    if len(direction.dropna()) >= 2:
        prev_dir = direction.dropna().iloc[-2]
        curr_dir = direction.dropna().iloc[-1]
        st_value = supertrend.dropna().iloc[-1] if len(supertrend.dropna()) > 0 else None
        
        indicators["supertrend"] = round(st_value, 2) if st_value else None
        indicators["prev_direction"] = int(prev_dir)
        indicators["curr_direction"] = int(curr_dir)
        
        if prev_dir == -1 and curr_dir == 1:
            signal = "BUY"
            reason = "Supertrend turned BULLISH"
            log_signal(symbol, "Supertrend Scalp", signal, reason, indicators)
        elif prev_dir == 1 and curr_dir == -1:
            signal = "SELL"
            reason = "Supertrend turned BEARISH"
            log_signal(symbol, "Supertrend Scalp", signal, reason, indicators)
    
    return {"signal": signal, "reason": reason, "strategy": "Supertrend Scalp", "indicators": indicators}

def strategy_orb(df, ltp, symbol=""):
    """
    Strategy 4: Opening Range Breakout (ORB)
    Uses first 15 minutes range for breakout levels
    """
    if len(df) < 5:
        return None
    
    now = now_ist()
    
    # Only valid between 9:30 AM and 11:30 AM
    if now.hour < 9 or (now.hour == 9 and now.minute < 30):
        return None
    if now.hour > 11 or (now.hour == 11 and now.minute > 30):
        return None
    
    # Get today's data only
    today = now.date()
    today_data = df[df.index.date == today] if hasattr(df.index, 'date') else df.tail(20)
    
    if len(today_data) < 3:
        return None
    
    # First 3 candles (15 min on 5-min chart)
    opening_range = today_data.head(3)
    orb_high = opening_range['high'].max()
    orb_low = opening_range['low'].min()
    
    indicators = {
        "ltp": ltp,
        "orb_high": round(orb_high, 2),
        "orb_low": round(orb_low, 2),
        "orb_range": round(orb_high - orb_low, 2)
    }
    signal = None
    reason = ""
    
    if ltp > orb_high * 1.002:  # Breakout above with 0.2% buffer
        signal = "BUY"
        reason = f"ORB Breakout above {orb_high:.2f}, LTP: {ltp:.2f}"
        log_signal(symbol, "Opening Range Breakout", signal, reason, indicators)
    elif ltp < orb_low * 0.998:  # Breakdown below with 0.2% buffer
        signal = "SELL"
        reason = f"ORB Breakdown below {orb_low:.2f}, LTP: {ltp:.2f}"
        log_signal(symbol, "Opening Range Breakout", signal, reason, indicators)
    
    return {"signal": signal, "reason": reason, "strategy": "Opening Range Breakout", "indicators": indicators}

# ============== POSITION SIZING ==============

def calculate_position_size(ltp, stop_loss_pct=0.01):
    """
    Risk-based position sizing
    Risk exactly ₹600 per trade (2% of ₹30k)
    """
    stop_loss_amount = ltp * stop_loss_pct
    qty = int(MAX_RISK_AMOUNT / stop_loss_amount)
    
    position_value = qty * ltp
    
    # Ensure we don't exceed capital
    if position_value > CAPITAL * 0.9:  # Max 90% per trade
        qty = int((CAPITAL * 0.9) / ltp)
    
    return max(1, qty)

def calculate_costs(qty, price, is_intraday=True):
    """Calculate all trading costs"""
    turnover = qty * price * 2  # Buy + Sell
    
    brokerage = min(40, turnover * 0.0003)  # ₹20 per side or 0.03%
    stt = turnover * 0.000125 if is_intraday else turnover * 0.001  # STT
    exchange = turnover * 0.0000345  # NSE charges
    gst = brokerage * 0.18  # GST on brokerage
    sebi = turnover * 0.000001  # SEBI charges
    stamp = qty * price * 0.00003  # Stamp duty on buy side
    
    total = brokerage + stt + exchange + gst + sebi + stamp
    return round(total, 2)

# ============== MAIN ENGINE ==============

def scan_for_signals(kite, check_margin=True):
    """
    Scan watchlist for trade signals.
    
    Args:
        kite: Kite Connect instance
        check_margin: If True, only return signals we can afford to trade
    
    Returns:
        List of viable signals with margin info
    """
    trade_log.info("="*60)
    trade_log.info(f"SCANNING FOR TRADE SIGNALS - {now_ist().strftime('%Y-%m-%d %H:%M:%S')} IST")
    trade_log.info(f"Watchlist: {len(WATCHLIST)} stocks")
    trade_log.info("="*60)
    
    print("\n" + "=" * 70)
    print(f"🔍 SCANNING FOR TRADE SIGNALS - {now_ist().strftime('%Y-%m-%d %H:%M:%S')} IST")
    print("=" * 70)
    
    # ============ STEP 1: CHECK AVAILABLE FUNDS FIRST ============
    available_margin = check_available_margin(kite)
    trade_log.info(f"Available margin: ₹{available_margin:,.2f}")
    print(f"\n  💰 Available Funds: ₹{available_margin:,.2f}")
    
    if available_margin <= 0:
        print(f"  ❌ No funds available for trading!")
        trade_log.warning("No funds available - skipping signal scan")
        return []
    
    # Minimum margin needed for any trade (rough estimate)
    min_margin_needed = 5000  # ₹5000 minimum to take any position
    if available_margin < min_margin_needed:
        print(f"  ⚠️ Low funds! Need at least ₹{min_margin_needed:,.2f} for trading")
        trade_log.warning(f"Low funds: ₹{available_margin:,.2f} < ₹{min_margin_needed:,.2f}")
    
    # Track margin as we find signals
    remaining_margin = available_margin
    
    # ============ STEP 2: COUNT EXISTING POSITIONS ============
    try:
        positions = kite.positions()
        day_positions = [p for p in positions.get('day', []) if p['quantity'] != 0]
        open_position_count = len(day_positions)
        print(f"  📊 Open Positions: {open_position_count} / {MAX_CONCURRENT_TRADES} max")
        
        if open_position_count >= MAX_CONCURRENT_TRADES:
            print(f"  ⚠️ Max concurrent trades reached! Close positions before taking new trades.")
            trade_log.warning(f"Max concurrent trades ({MAX_CONCURRENT_TRADES}) reached")
            return []
    except Exception as e:
        trade_log.error(f"Failed to check positions: {e}")
        open_position_count = 0
    
    slots_available = MAX_CONCURRENT_TRADES - open_position_count
    print(f"  🎯 Trade Slots Available: {slots_available}")
    print("=" * 70)
    
    signals = []
    skipped_due_to_margin = 0
    
    for symbol in WATCHLIST:
        # Stop if we've found enough signals for available slots
        if len(signals) >= slots_available:
            trade_log.info(f"Found enough signals ({len(signals)}) for available slots ({slots_available})")
            break
            
        print(f"  Analyzing {symbol}...", end=" ")
        trade_log.debug(f"Analyzing {symbol}")
        
        # Get data
        df = get_historical_data(kite, symbol, interval="5minute", days=3)
        if df is None or len(df) < 20:
            trade_log.debug(f"{symbol}: Insufficient data (got {len(df) if df is not None else 0} candles)")
            print("❌ Insufficient data")
            continue
        
        ltp = get_ltp(kite, symbol)
        if ltp is None:
            trade_log.debug(f"{symbol}: Failed to get LTP")
            print("❌ No quote")
            continue
        
        trade_log.debug(f"{symbol}: LTP=₹{ltp}, Candles={len(df)}")
        
        # Run all strategies
        strategies = [
            strategy_vwap_rsi_reversal(df, ltp, symbol),
            strategy_ema_crossover(df, ltp, symbol),
            strategy_supertrend_scalp(df, ltp, symbol),
            strategy_orb(df, ltp, symbol)
        ]
        
        found_signal = False
        for result in strategies:
            if result and result['signal']:
                # Calculate position details
                atr = calculate_atr(df).iloc[-1]
                sl_pct = max(0.005, min(0.02, atr / ltp))  # ATR-based SL, 0.5%-2%
                
                if result['signal'] == "BUY":
                    stop_loss = round(ltp * (1 - sl_pct), 2)
                    target = round(ltp * (1 + sl_pct * MIN_RR_RATIO), 2)
                else:
                    stop_loss = round(ltp * (1 + sl_pct), 2)
                    target = round(ltp * (1 - sl_pct * MIN_RR_RATIO), 2)
                
                qty = calculate_position_size(ltp, sl_pct)
                costs = calculate_costs(qty, ltp)
                position_value = qty * ltp
                potential_profit = abs(target - ltp) * qty
                potential_loss = abs(ltp - stop_loss) * qty
                
                # ============ STEP 3: CHECK IF WE CAN AFFORD THIS TRADE ============
                if check_margin:
                    # Get actual margin required from Kite
                    txn_type = "BUY" if result['signal'] == "BUY" else "SELL"
                    required_margin = get_order_margin_required(kite, symbol, qty, txn_type)
                    
                    # Check if we have enough remaining margin
                    if required_margin > remaining_margin:
                        # Try to adjust quantity to fit available margin
                        adjusted_qty = int(remaining_margin / (required_margin / qty)) if required_margin > 0 else 0
                        
                        if adjusted_qty < 1:
                            trade_log.info(f"{symbol}: Signal found but insufficient margin (need ₹{required_margin:,.2f}, have ₹{remaining_margin:,.2f})")
                            print(f"⚠️ Signal but no margin (need ₹{required_margin:,.0f})")
                            skipped_due_to_margin += 1
                            continue
                        else:
                            # Adjust quantity to fit margin
                            qty = adjusted_qty
                            position_value = qty * ltp
                            potential_profit = abs(target - ltp) * qty
                            potential_loss = abs(ltp - stop_loss) * qty
                            required_margin = get_order_margin_required(kite, symbol, qty, txn_type)
                            trade_log.info(f"{symbol}: Quantity adjusted to {qty} to fit margin")
                else:
                    required_margin = position_value * 0.2  # Estimate 20% margin
                
                signal_data = {
                    "symbol": symbol,
                    "signal": result['signal'],
                    "strategy": result['strategy'],
                    "reason": result['reason'],
                    "ltp": ltp,
                    "qty": qty,
                    "stop_loss": stop_loss,
                    "target": target,
                    "position_value": position_value,
                    "costs": costs,
                    "potential_profit": potential_profit,
                    "potential_loss": potential_loss,
                    "rr_ratio": potential_profit / potential_loss if potential_loss > 0 else 0,
                    "indicators": result.get('indicators', {}),
                    "margin_required": required_margin,
                    "margin_available": remaining_margin
                }
                signals.append(signal_data)
                found_signal = True
                
                # Deduct margin for next signal check
                remaining_margin -= required_margin
                
                # Log the complete signal
                trade_log.info(f"SIGNAL FOUND: {symbol} | {result['signal']} | {result['strategy']}")
                trade_log.info(f"  LTP: ₹{ltp} | Qty: {qty} | Value: ₹{position_value:.2f}")
                trade_log.info(f"  SL: ₹{stop_loss} | Target: ₹{target} | R:R={signal_data['rr_ratio']:.2f}")
                trade_log.info(f"  Margin Required: ₹{required_margin:,.2f} | Remaining: ₹{remaining_margin:,.2f}")
                trade_log.info(f"  Reason: {result['reason']}")
                
                print(f"✅ {result['signal']} (₹{required_margin:,.0f} margin)")
                break
        
        if not found_signal:
            print("No signal")
    
    # ============ STEP 4: SUMMARY ============
    trade_log.info(f"Scan complete: {len(signals)} viable signals found")
    if skipped_due_to_margin > 0:
        trade_log.info(f"Skipped {skipped_due_to_margin} signals due to insufficient margin")
        print(f"\n  ⚠️ {skipped_due_to_margin} signal(s) skipped due to insufficient margin")
    
    print(f"\n  📊 Scan Summary:")
    print(f"     Viable Signals: {len(signals)}")
    print(f"     Initial Margin: ₹{available_margin:,.2f}")
    print(f"     After Signals: ₹{remaining_margin:,.2f}")
    
    return signals

def display_signals(signals):
    """Display found signals in a formatted way"""
    if not signals:
        print("\n📭 No trade signals found at this time.")
        return
    
    print(f"\n{'='*70}")
    print(f"📊 TRADE SIGNALS FOUND: {len(signals)}")
    print(f"{'='*70}")
    
    for i, sig in enumerate(signals, 1):
        print(f"\n┌─ Signal #{i}: {sig['symbol']} ─────────────────────────────────")
        print(f"│  Type: {'🟢 BUY' if sig['signal'] == 'BUY' else '🔴 SELL'}")
        print(f"│  Strategy: {sig['strategy']}")
        print(f"│  Reason: {sig['reason']}")
        print(f"├─ Trade Details ──────────────────────────────────────────────")
        print(f"│  LTP: ₹{sig['ltp']:,.2f}")
        print(f"│  Quantity: {sig['qty']} shares")
        print(f"│  Position Value: ₹{sig['position_value']:,.2f}")
        print(f"│  Stop Loss: ₹{sig['stop_loss']:,.2f}")
        print(f"│  Target: ₹{sig['target']:,.2f}")
        print(f"├─ Risk/Reward ────────────────────────────────────────────────")
        print(f"│  Potential Loss: ₹{sig['potential_loss']:,.2f}")
        print(f"│  Potential Profit: ₹{sig['potential_profit']:,.2f}")
        print(f"│  Risk:Reward: 1:{sig['rr_ratio']:.2f}")
        print(f"│  Trading Costs: ₹{sig['costs']:.2f}")
        print(f"├─ Margin ─────────────────────────────────────────────────────")
        margin_req = sig.get('margin_required', 0)
        margin_avail = sig.get('margin_available', 0)
        print(f"│  Margin Required: ₹{margin_req:,.2f}")
        print(f"│  Margin Available: ₹{margin_avail:,.2f}")
        if margin_req > 0:
            margin_pct = (margin_req / margin_avail * 100) if margin_avail > 0 else 100
            print(f"│  Margin Usage: {margin_pct:.1f}%")
        print(f"└──────────────────────────────────────────────────────────────")

def execute_trade(kite, signal):
    """
    Execute a trade with STRICT safety checks:
    1. Validate margin availability
    2. Adjust quantity if needed
    3. Round SL/Target to tick size
    4. Place entry order
    5. Place SL order (SL-M) - MANDATORY, exit position if fails
    6. Place Target order (LIMIT) - MANDATORY, exit position if fails
    7. Track position and record in tradebook
    
    CRITICAL: If SL or Target order fails, the position will be IMMEDIATELY EXITED
    to prevent unprotected positions.
    """
    symbol = signal['symbol']
    qty = signal['qty']
    ltp = signal['ltp']
    
    order_log.info("="*60)
    order_log.info(f"EXECUTING TRADE: {symbol}")
    order_log.info(f"Signal: {signal['signal']} | Strategy: {signal['strategy']}")
    order_log.info(f"Original Qty: {qty} | LTP: ₹{ltp}")
    order_log.info("="*60)
    
    # Step 1: Validate margin and adjust quantity
    transaction_type = "BUY" if signal['signal'] == "BUY" else "SELL"
    validation = validate_order_before_placement(kite, symbol, qty, ltp, transaction_type)
    
    if not validation['valid']:
        order_log.error(f"Order validation failed: {validation['reason']}")
        print(f"  ❌ Cannot place order: {validation['reason']}")
        return False
    
    # Use adjusted quantity
    qty = validation['adjusted_qty']
    if qty != signal['qty']:
        print(f"  ⚠️ Quantity adjusted: {signal['qty']} → {qty} ({validation['reason']})")
    
    # Step 2: Get tick size and round SL/Target
    tick_size = get_tick_size(kite, symbol)
    stop_loss = round_to_tick_size(signal['stop_loss'], tick_size)
    target = round_to_tick_size(signal['target'], tick_size)
    
    order_log.info(f"Tick size for {symbol}: {tick_size}")
    order_log.info(f"Adjusted - Qty: {qty} | SL: ₹{stop_loss} | Target: ₹{target}")
    
    print(f"  📊 Order Details:")
    print(f"     Symbol: {symbol}")
    print(f"     Quantity: {qty}")
    print(f"     Entry (Market): ~₹{ltp:.2f}")
    print(f"     Stop Loss: ₹{stop_loss}")
    print(f"     Target: ₹{target}")
    print(f"     Tick Size: {tick_size}")
    
    entry_order_id = None
    sl_order_id = None
    target_order_id = None
    
    # Determine exit transaction type (opposite of entry)
    exit_txn_type = kite.TRANSACTION_TYPE_SELL if signal['signal'] == "BUY" else kite.TRANSACTION_TYPE_BUY
    entry_txn_type = kite.TRANSACTION_TYPE_BUY if signal['signal'] == "BUY" else kite.TRANSACTION_TYPE_SELL
    
    try:
        # ============ STEP 3: PLACE ENTRY ORDER ============
        order_params = {
            "variety": kite.VARIETY_REGULAR,
            "exchange": kite.EXCHANGE_NSE,
            "tradingsymbol": symbol,
            "transaction_type": entry_txn_type,
            "quantity": qty,
            "product": kite.PRODUCT_MIS,
            "order_type": kite.ORDER_TYPE_MARKET
        }
        order_log.info(f"Placing {signal['signal']} order: {order_params}")
        
        entry_order_id = kite.place_order(**order_params)
        log_order(signal['signal'], symbol, qty, ltp, "MARKET", entry_order_id, "PLACED")
        print(f"  ✅ {signal['signal']} order placed! Order ID: {entry_order_id}")
        
        # Wait briefly for order to be processed
        time.sleep(0.5)
        
        # ============ STEP 4: PLACE STOP LOSS ORDER (MANDATORY) ============
        sl_params = {
            "variety": kite.VARIETY_REGULAR,
            "exchange": kite.EXCHANGE_NSE,
            "tradingsymbol": symbol,
            "transaction_type": exit_txn_type,
            "quantity": qty,
            "product": kite.PRODUCT_MIS,
            "order_type": kite.ORDER_TYPE_SLM,
            "trigger_price": stop_loss
        }
        order_log.info(f"Placing SL-M order: {sl_params}")
        
        try:
            sl_order_id = kite.place_order(**sl_params)
            log_order(f"SL-M {exit_txn_type}", symbol, qty, stop_loss, "SL-M", sl_order_id, "PLACED")
            print(f"  ✅ Stop Loss order placed at ₹{stop_loss}! Order ID: {sl_order_id}")
        except Exception as sl_error:
            # CRITICAL: SL order failed - IMMEDIATELY EXIT THE POSITION
            order_log.critical(f"🚨 SL ORDER FAILED for {symbol}: {sl_error}")
            order_log.critical(f"🚨 EMERGENCY EXIT: Closing position to prevent unprotected trade")
            print(f"  🚨 CRITICAL: Stop Loss order FAILED: {sl_error}")
            print(f"  🚨 EMERGENCY: Exiting position immediately to prevent unprotected trade!")
            
            # Place emergency exit order
            try:
                emergency_exit = kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=kite.EXCHANGE_NSE,
                    tradingsymbol=symbol,
                    transaction_type=exit_txn_type,
                    quantity=qty,
                    product=kite.PRODUCT_MIS,
                    order_type=kite.ORDER_TYPE_MARKET
                )
                order_log.info(f"Emergency exit order placed: {emergency_exit}")
                print(f"  ⚠️ Position closed via emergency exit. Order ID: {emergency_exit}")
            except Exception as exit_error:
                order_log.critical(f"🚨🚨 EMERGENCY EXIT ALSO FAILED: {exit_error}")
                print(f"  🚨🚨 CRITICAL: Emergency exit FAILED! Manual intervention required!")
                print(f"       Go to Kite app and close {symbol} position IMMEDIATELY!")
            
            return False
        
        # ============ STEP 5: PLACE TARGET ORDER (MANDATORY) ============
        target_params = {
            "variety": kite.VARIETY_REGULAR,
            "exchange": kite.EXCHANGE_NSE,
            "tradingsymbol": symbol,
            "transaction_type": exit_txn_type,
            "quantity": qty,
            "product": kite.PRODUCT_MIS,
            "order_type": kite.ORDER_TYPE_LIMIT,
            "price": target
        }
        order_log.info(f"Placing TARGET order: {target_params}")
        
        try:
            target_order_id = kite.place_order(**target_params)
            log_order(f"TARGET {exit_txn_type}", symbol, qty, target, "LIMIT", target_order_id, "PLACED")
            print(f"  ✅ Target order placed at ₹{target}! Order ID: {target_order_id}")
        except Exception as target_error:
            # Target order failed - position has SL but no target
            # This is less critical but still concerning
            order_log.error(f"⚠️ TARGET ORDER FAILED for {symbol}: {target_error}")
            print(f"  ⚠️ WARNING: Target order FAILED: {target_error}")
            print(f"  ⚠️ Position has SL protection but NO target order!")
            print(f"  ⚠️ You must manually set a target or monitor the position!")
            # Don't exit here since SL is in place, but warn user
        
        # ============ STEP 6: VERIFY ALL ORDERS ARE IN PLACE ============
        order_log.info("Verifying orders...")
        
        # Check if SL and Target orders are properly placed
        orders_verified = True
        try:
            all_orders = kite.orders()
            sl_found = any(o['order_id'] == sl_order_id and o['status'] in ['TRIGGER PENDING', 'OPEN', 'COMPLETE'] for o in all_orders)
            target_found = target_order_id is None or any(o['order_id'] == target_order_id and o['status'] in ['OPEN', 'COMPLETE'] for o in all_orders)
            
            if not sl_found:
                order_log.error(f"SL order {sl_order_id} not found in order book!")
                orders_verified = False
            if target_order_id and not target_found:
                order_log.warning(f"Target order {target_order_id} not found in order book!")
        except Exception as verify_error:
            order_log.warning(f"Could not verify orders: {verify_error}")
        
        # ============ STEP 7: TRACK AND RECORD ============
        pos_type = "LONG" if signal['signal'] == "BUY" else "SHORT"
        add_tracked_position(symbol, qty, ltp, stop_loss, target, pos_type)
        
        # Record trade in tradebook with full reasoning
        expected_risk = abs(ltp - stop_loss) * qty
        expected_reward = abs(target - ltp) * qty
        
        trade_record = record_trade({
            "symbol": symbol,
            "action": signal['signal'],
            "quantity": qty,
            "entry_price": ltp,
            "stop_loss": stop_loss,
            "target": target,
            "strategy": signal.get('strategy', 'Unknown'),
            "reason": signal.get('reason', 'No reason provided'),
            "indicators": signal.get('indicators', {}),
            "order_ids": {
                "entry": entry_order_id,
                "sl": sl_order_id,
                "target": target_order_id
            },
            "margin_used": validation.get('required_margin', 0),
            "expected_risk": round(expected_risk, 2),
            "expected_reward": round(expected_reward, 2),
            "risk_reward_ratio": round(expected_reward / expected_risk, 2) if expected_risk > 0 else 0,
            "sl_order_placed": sl_order_id is not None,
            "target_order_placed": target_order_id is not None
        })
        
        order_log.info(f"Trade execution SUCCESS for {symbol}")
        order_log.info(f"  Entry: {entry_order_id} | SL: {sl_order_id} | Target: {target_order_id}")
        order_log.info(f"  Tradebook entry: #{trade_record['trade_id']}")
        
        # Final status message
        if sl_order_id and target_order_id:
            print(f"\n  ✅ TRADE COMPLETE: {symbol} is FULLY PROTECTED")
            print(f"     Entry: {entry_order_id}")
            print(f"     SL Order: {sl_order_id} @ ₹{stop_loss}")
            print(f"     Target Order: {target_order_id} @ ₹{target}")
        elif sl_order_id:
            print(f"\n  ⚠️ TRADE PARTIAL: {symbol} has SL but NO TARGET")
            print(f"     Monitor position manually or set target in Kite app!")
        
        return {
            "success": True,
            "symbol": symbol,
            "qty": qty,
            "entry_order_id": entry_order_id,
            "sl_order_id": sl_order_id,
            "target_order_id": target_order_id,
            "stop_loss": stop_loss,
            "target": target,
            "trade_id": trade_record['trade_id'],
            "fully_protected": sl_order_id is not None and target_order_id is not None
        }
        
    except Exception as e:
        log_order(signal['signal'], symbol, qty, ltp, "MARKET", error=e)
        order_log.error(f"Trade execution FAILED for {symbol}: {e}")
        print(f"  ❌ Order failed: {e}")
        return False

def show_current_positions(kite):
    """Display current positions and P&L"""
    print("\n" + "=" * 70)
    print("📈 CURRENT POSITIONS")
    print("=" * 70)
    
    try:
        api_log.info("Fetching positions")
        positions = kite.positions()
        log_api_call("GET", "positions", response=positions)
        day_positions = positions.get('day', [])
        
        if not day_positions:
            trade_log.info("No open positions today")
            print("No open positions today.")
            return
        
        total_pnl = 0
        trade_log.info("Current Positions:")
        for pos in day_positions:
            if pos['quantity'] != 0:
                pnl = pos['pnl']
                total_pnl += pnl
                pnl_color = "🟢" if pnl >= 0 else "🔴"
                trade_log.info(f"  {pos['tradingsymbol']}: {pos['quantity']} qty @ ₹{pos['average_price']:.2f} | P&L: ₹{pnl:,.2f}")
                print(f"  {pos['tradingsymbol']}: {pos['quantity']} qty @ ₹{pos['average_price']:.2f} | P&L: {pnl_color} ₹{pnl:,.2f}")
        
        trade_log.info(f"Total Day P&L: ₹{total_pnl:,.2f}")
        print(f"\n  Total Day P&L: {'🟢' if total_pnl >= 0 else '🔴'} ₹{total_pnl:,.2f}")
        
        # Kill switch warning
        if total_pnl < -CAPITAL * MAX_DAILY_LOSS:
            trade_log.warning(f"KILL SWITCH TRIGGERED! Daily loss ₹{total_pnl:,.2f} exceeds {MAX_DAILY_LOSS*100}% limit!")
            print(f"\n⚠️  WARNING: Daily loss exceeds {MAX_DAILY_LOSS*100}% limit!")
            print("    Consider stopping trading for today.")
    except Exception as e:
        log_api_call("GET", "positions", error=e)
        print(f"Error fetching positions: {e}")

def show_margins(kite):
    """Display available margins"""
    print("\n" + "=" * 70)
    print("💰 MARGIN STATUS")
    print("=" * 70)
    
    try:
        api_log.info("Fetching margins")
        margins = kite.margins()
        log_api_call("GET", "margins", response=margins)
        equity = margins.get('equity', {})
        
        available = equity.get('available', {})
        utilized = equity.get('utilised', {})
        
        cash = available.get('live_balance', 0)
        collateral = available.get('collateral', 0)
        total_available = cash + collateral
        
        used = utilized.get('debits', 0)
        
        trade_log.info(f"Margin Status: Cash=₹{cash:,.2f}, Collateral=₹{collateral:,.2f}, Used=₹{used:,.2f}")
        
        print(f"  Cash Balance: ₹{cash:,.2f}")
        print(f"  Collateral: ₹{collateral:,.2f}")
        print(f"  Total Available: ₹{total_available:,.2f}")
        print(f"  Utilized: ₹{used:,.2f}")
    except Exception as e:
        log_api_call("GET", "margins", error=e)
        print(f"Error fetching margins: {e}")

def monitor_positions_continuous(kite):
    """
    Continuous position monitoring loop.
    - Monitors live positions and P&L
    - Alerts when price approaches or hits SL/Target
    - Does NOT scan for new signals (use option 3 for that)
    """
    trade_log.info("Starting continuous position monitoring")
    print("\n" + "=" * 80)
    print("  🔄 LIVE POSITION MONITORING")
    print(f"     Refresh interval: {MONITOR_INTERVAL} seconds")
    print("     Monitors: Positions, P&L, SL/Target proximity")
    print("     Press Ctrl+C to stop and return to menu")
    print("=" * 80)
    
    try:
        while True:
            # Check market status
            is_open, status = is_market_open()
            
            # Get positions with SL/Target info
            positions = get_positions_with_sl_target(kite)
            total_pnl = sum(p['pnl'] for p in positions)
            
            # Display dashboard
            display_positions_dashboard(positions, total_pnl)
            
            # Show market status
            print(f"\n  📅 Market: {status}")
            
            # Check for SL/Target hits
            alerts = check_sl_target_hits(kite, positions)
            if alerts:
                print("\n  ⚡ ALERTS:")
                for alert in alerts:
                    print(f"     {alert['message']}")
                    trade_log.warning(alert['message'])
                    
                    # Log to order log for critical alerts
                    if alert['type'] in ['SL_HIT', 'TARGET_HIT']:
                        order_log.warning(alert['message'])
            
            # Check for untracked positions
            untracked = [p for p in positions if not p.get('tracked')]
            if untracked:
                print(f"\n  ⚠️ {len(untracked)} position(s) without SL/Target tracking")
                for p in untracked:
                    print(f"     • {p['symbol']}: {p['type']} {abs(p['quantity'])} @ ₹{p['average_price']:.2f}")
            
            # Show pending orders count
            try:
                orders = kite.orders()
                pending = [o for o in orders if o['status'] in ['TRIGGER PENDING', 'OPEN']]
                if pending:
                    sl_orders = len([o for o in pending if o['order_type'] == 'SL-M'])
                    target_orders = len([o for o in pending if o['order_type'] == 'LIMIT'])
                    print(f"\n  📋 Pending Orders: {len(pending)} (SL: {sl_orders}, Target: {target_orders})")
            except:
                pass
            
            # Auto square-off warning
            now_time = now_ist()
            if is_open and now_time.hour == 15 and now_time.minute >= 10:
                print("\n  🚨 SQUARE-OFF ZONE: Consider closing intraday positions!")
            
            # Kill switch check
            if total_pnl < -CAPITAL * MAX_DAILY_LOSS:
                trade_log.critical(f"KILL SWITCH: Daily loss ₹{total_pnl:,.2f} exceeds limit!")
                print("\n  🚨🚨🚨 KILL SWITCH TRIGGERED! STOP TRADING! 🚨🚨🚨")
            
            # Show available margin
            try:
                available = check_available_margin(kite)
                print(f"\n  💰 Available Margin: ₹{available:,.2f}")
            except:
                pass
            
            print(f"\n  Next refresh in {MONITOR_INTERVAL}s... (Ctrl+C for menu)")
            time.sleep(MONITOR_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n  ⏹ Monitoring paused. Returning to menu...")
        return

def show_menu():
    """Display main menu"""
    print("\n" + "═" * 60)
    print("  📋 TRADING ENGINE MENU")
    print("═" * 60)
    print("  [1] View current positions & P&L")
    print("  [2] Set up SL/Target for existing positions")
    print("  [3] Scan & execute new trades")
    print("  [4] Start live monitoring (auto SL/Target alerts)")
    print("  [5] Show margin & fund status")
    print("  [6] View all tracked SL/Target levels")
    print("  [7] Remove position from tracking")
    print("  [8] View pending orders")
    print("  [9] View tradebook (trade history & reasoning)")
    print("  [0] Exit")
    print("═" * 60)

def main():
    trade_log.info("="*60)
    trade_log.info("KITE CONNECT DAY TRADING ENGINE STARTED")
    trade_log.info(f"Capital: ₹{CAPITAL:,} | Max Risk/Trade: ₹{MAX_RISK_AMOUNT:,}")
    trade_log.info(f"Log directory: {LOG_DIR}")
    trade_log.info("="*60)
    
    print("\n" + "═" * 70)
    print("  🚀 KITE CONNECT DAY TRADING ENGINE")
    print(f"     Capital: ₹{CAPITAL:,} | Max Risk/Trade: ₹{MAX_RISK_AMOUNT:,}")
    print(f"     Logs: {LOG_DIR}/")
    print("═" * 70)
    
    # Check market status
    is_open, status = is_market_open()
    trade_log.info(f"Market Status: {status}")
    trade_log.info(f"Current Time: {now_ist().strftime('%Y-%m-%d %H:%M:%S')} IST")
    print(f"\n  📅 Market Status: {status}")
    print(f"  ⏰ Current Time: {now_ist().strftime('%Y-%m-%d %H:%M:%S')} IST")
    
    # Initialize Kite
    kite = init_kite()
    
    # Main menu loop
    while True:
        show_menu()
        
        try:
            choice = input("  Enter choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting...")
            break
        
        if choice == '1':
            # View positions
            positions = get_positions_with_sl_target(kite)
            total_pnl = sum(p['pnl'] for p in positions)
            display_positions_dashboard(positions, total_pnl)
            
            # Check alerts
            alerts = check_sl_target_hits(kite, positions)
            if alerts:
                print("\n  ⚡ ALERTS:")
                for alert in alerts:
                    print(f"     {alert['message']}")
            
            input("\n  Press Enter to continue...")
        
        elif choice == '2':
            # Set up SL/Target
            positions = get_positions_with_sl_target(kite)
            if not positions:
                print("\n  No open positions to configure.")
            else:
                print("\n  Select position to set SL/Target:")
                for i, pos in enumerate(positions, 1):
                    tracked = "✓" if pos.get('tracked') else "⚠"
                    print(f"    [{i}] {pos['symbol']} - {pos['type']} {abs(pos['quantity'])} @ ₹{pos['average_price']:.2f} {tracked}")
                print("    [0] Cancel")
                
                try:
                    idx = int(input("  Select: ").strip())
                    if 1 <= idx <= len(positions):
                        setup_sl_target_interactive(kite, positions[idx-1])
                except (ValueError, IndexError):
                    print("  Invalid selection")
            
            input("\n  Press Enter to continue...")
        
        elif choice == '3':
            # Scan for signals and execute trades
            is_open, status = is_market_open()  # Refresh market status
            
            if not is_open:
                print(f"\n  ⚠️ Market is closed ({status}). Running in simulation mode...")
            
            # Scan for signals (this now checks margin and filters signals we can afford)
            signals = scan_for_signals(kite, check_margin=True)
            display_signals(signals)
            
            # Trade approval
            if signals:
                if not is_open:
                    print("\n  ⚠️ Market closed - orders will not be placed")
                    input("\n  Press Enter to continue...")
                    continue
                
                print("\n  🎯 TRADE APPROVAL")
                print(f"      Note: All signals shown are within your available margin")
                
                for i, sig in enumerate(signals, 1):
                    # Get tick size for display
                    tick_size = get_tick_size(kite, sig['symbol'])
                    sl_rounded = round_to_tick_size(sig['stop_loss'], tick_size)
                    target_rounded = round_to_tick_size(sig['target'], tick_size)
                    margin_req = sig.get('margin_required', 0)
                    
                    print(f"\n  [{i}] Execute {sig['signal']} on {sig['symbol']}?")
                    print(f"      Qty: {sig['qty']} | Value: ₹{sig['position_value']:,.2f} | Margin: ₹{margin_req:,.2f}")
                    print(f"      SL: ₹{sl_rounded} | Target: ₹{target_rounded} (tick: {tick_size})")
                    
                    response = input(f"      Approve? (y/n/skip all): ").strip().lower()
                    
                    if response == 'y':
                        trade_log.info(f"USER APPROVED trade: {sig['signal']} {sig['symbol']}")
                        # execute_trade handles: margin re-check, tick size, SL, Target, and auto-tracking
                        result = execute_trade(kite, sig)
                        if result:
                            print(f"\n  ✅ Trade executed successfully!")
                            # Refresh available margin
                            available_margin = check_available_margin(kite)
                            print(f"      Remaining margin: ₹{available_margin:,.2f}")
                    elif response == 'skip all':
                        print("  Skipping remaining signals...")
                        break
                    else:
                        trade_log.info(f"USER REJECTED trade: {sig['signal']} {sig['symbol']}")
                        print("      Skipped.")
            
            input("\n  Press Enter to continue...")
        
        elif choice == '4':
            # Continuous monitoring
            monitor_positions_continuous(kite)
        
        elif choice == '5':
            # Show margins
            show_margins(kite)
            input("\n  Press Enter to continue...")
        
        elif choice == '6':
            # View tracked positions
            tracked = load_tracked_positions()
            if not tracked:
                print("\n  No positions being tracked.")
            else:
                print("\n  📋 TRACKED POSITIONS (SL/Target):")
                print(f"  {'Symbol':<12} {'Type':<6} {'Entry':>10} {'SL':>10} {'Target':>10} {'Status':<10}")
                print("  " + "-" * 60)
                for symbol, pos in tracked.items():
                    print(f"  {symbol:<12} {pos['type']:<6} ₹{pos['entry_price']:>8.2f} "
                          f"₹{pos['stop_loss']:>8.2f} ₹{pos['target']:>8.2f} {pos['status']:<10}")
            input("\n  Press Enter to continue...")
        
        elif choice == '7':
            # Remove tracking
            tracked = load_tracked_positions()
            if not tracked:
                print("\n  No positions being tracked.")
            else:
                print("\n  Select position to remove tracking:")
                symbols = list(tracked.keys())
                for i, symbol in enumerate(symbols, 1):
                    print(f"    [{i}] {symbol}")
                print("    [0] Cancel")
                
                try:
                    idx = int(input("  Select: ").strip())
                    if 1 <= idx <= len(symbols):
                        remove_tracked_position(symbols[idx-1])
                        print(f"  ✅ Removed tracking for {symbols[idx-1]}")
                except (ValueError, IndexError):
                    print("  Invalid selection")
            
            input("\n  Press Enter to continue...")
        
        elif choice == '8':
            # View pending orders (SL and Target orders)
            print("\n  📋 PENDING ORDERS")
            print("  " + "=" * 70)
            try:
                orders = kite.orders()
                pending_orders = [o for o in orders if o['status'] in ['TRIGGER PENDING', 'OPEN']]
                
                if not pending_orders:
                    print("  No pending orders.")
                else:
                    print(f"  {'Symbol':<12} {'Type':<6} {'Qty':>6} {'Price':>10} {'Trigger':>10} {'Status':<18}")
                    print("  " + "-" * 70)
                    for order in pending_orders:
                        symbol = order['tradingsymbol']
                        txn_type = order['transaction_type']
                        qty = order['quantity']
                        price = order.get('price', 0) or 0
                        trigger = order.get('trigger_price', 0) or 0
                        status = order['status']
                        order_type = order['order_type']
                        
                        price_str = f"₹{price:.2f}" if price > 0 else "MKT"
                        trigger_str = f"₹{trigger:.2f}" if trigger > 0 else "---"
                        
                        # Identify order purpose
                        if order_type == "SL-M":
                            purpose = "SL"
                        elif order_type == "LIMIT" and status == "OPEN":
                            purpose = "TARGET"
                        else:
                            purpose = order_type
                        
                        print(f"  {symbol:<12} {txn_type:<6} {qty:>6} {price_str:>10} {trigger_str:>10} {status:<18} [{purpose}]")
                    
                    print(f"\n  Total pending orders: {len(pending_orders)}")
                    
                    # Option to cancel orders
                    cancel = input("\n  Cancel an order? (enter order # or 'n'): ").strip().lower()
                    if cancel != 'n' and cancel.isdigit():
                        idx = int(cancel) - 1
                        if 0 <= idx < len(pending_orders):
                            order_to_cancel = pending_orders[idx]
                            try:
                                kite.cancel_order(
                                    variety=order_to_cancel['variety'],
                                    order_id=order_to_cancel['order_id']
                                )
                                print(f"  ✅ Order {order_to_cancel['order_id']} cancelled!")
                                order_log.info(f"Order cancelled: {order_to_cancel['order_id']} {order_to_cancel['tradingsymbol']}")
                            except Exception as e:
                                print(f"  ❌ Failed to cancel: {e}")
            except Exception as e:
                print(f"  Error fetching orders: {e}")
            
            input("\n  Press Enter to continue...")
        
        elif choice == '9':
            # View tradebook
            print("\n  📒 TRADEBOOK OPTIONS")
            print("  [1] View today's tradebook")
            print("  [2] View tradebook for specific date")
            print("  [3] View trade details")
            print("  [4] List all tradebook dates")
            print("  [5] Mark trade as closed (manual exit)")
            print("  [0] Back to main menu")
            
            tb_choice = input("\n  Select option: ").strip()
            
            if tb_choice == '1':
                display_tradebook()
            
            elif tb_choice == '2':
                dates = list_tradebook_dates()
                if dates:
                    print("\n  Available dates:")
                    for i, d in enumerate(dates[:10], 1):  # Show last 10
                        print(f"    [{i}] {d}")
                    idx = input("  Select date #: ").strip()
                    if idx.isdigit() and 1 <= int(idx) <= len(dates):
                        display_tradebook(dates[int(idx)-1])
                else:
                    print("  No tradebook history found.")
            
            elif tb_choice == '3':
                date_input = input("  Enter date (YYYY-MM-DD) or press Enter for today: ").strip()
                date = date_input if date_input else None
                tradebook = load_tradebook(date)
                
                if tradebook["trades"]:
                    print(f"\n  Trades for {tradebook['date']}:")
                    for t in tradebook["trades"]:
                        print(f"    [{t['trade_id']}] {t['symbol']} - {t['action']} - {t['status']}")
                    
                    trade_id = input("  Enter trade # for details: ").strip()
                    if trade_id.isdigit():
                        display_trade_details(int(trade_id), date)
                else:
                    print("  No trades found for this date.")
            
            elif tb_choice == '4':
                dates = list_tradebook_dates()
                if dates:
                    print("\n  📅 Available tradebook dates:")
                    for d in dates:
                        tb = load_tradebook(d)
                        trade_count = len(tb.get("trades", []))
                        total_pnl = tb.get("summary", {}).get("total_pnl", 0)
                        pnl_str = f"₹{total_pnl:+,.2f}" if total_pnl != 0 else "---"
                        print(f"    {d}: {trade_count} trades | P&L: {pnl_str}")
                else:
                    print("  No tradebook history found.")
            
            elif tb_choice == '5':
                # Mark trade as closed manually
                tradebook = load_tradebook()
                open_trades = [t for t in tradebook["trades"] if t["status"] == "OPEN"]
                
                if not open_trades:
                    print("  No open trades to close.")
                else:
                    print("\n  Open trades:")
                    for t in open_trades:
                        print(f"    [{t['trade_id']}] {t['symbol']} - {t['action']} @ ₹{t['entry_price']:.2f}")
                    
                    trade_id = input("  Enter trade # to close: ").strip()
                    if trade_id.isdigit():
                        trade = next((t for t in open_trades if t['trade_id'] == int(trade_id)), None)
                        if trade:
                            exit_price = input(f"  Enter exit price for {trade['symbol']}: ").strip()
                            if exit_price:
                                try:
                                    exit_price = float(exit_price)
                                    exit_reason = input("  Exit reason (SL_HIT/TARGET_HIT/MANUAL/EOD_SQUAREOFF): ").strip().upper()
                                    if exit_reason not in ['SL_HIT', 'TARGET_HIT', 'MANUAL', 'EOD_SQUAREOFF']:
                                        exit_reason = 'MANUAL'
                                    
                                    result = update_trade_exit(trade['symbol'], exit_price, exit_reason)
                                    if result:
                                        pnl = result.get('realized_pnl', 0)
                                        print(f"  ✅ Trade closed! P&L: ₹{pnl:,.2f}")
                                except ValueError:
                                    print("  Invalid price.")
            
            input("\n  Press Enter to continue...")
        
        elif choice == '0':
            print("\n  👋 Goodbye! Happy trading!")
            trade_log.info("Trading engine session ended by user")
            break
        
        else:
            print("\n  ❌ Invalid choice. Try again.")
    
    trade_log.info("="*60)
    trade_log.info("TRADING ENGINE SESSION COMPLETE")
    trade_log.info("="*60)

if __name__ == "__main__":
    main()
