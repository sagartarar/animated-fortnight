#!/usr/bin/env python3
"""
VWAP + RSI Reversal Strategy Backtest
=====================================
Backtests the VWAP+RSI reversal strategy on Nifty F&O stocks
across multiple timeframes: 5min, 15min, 60min (1hr), day

Author: Trading System
Date: February 2026
"""

import json
import os
import sys
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
import pytz

# ============== CONFIGURATION ==============

IST = pytz.timezone('Asia/Kolkata')

# Nifty F&O Stocks (Top 25 by volume)
FNO_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
    "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "ASIANPAINT",
    "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO",
    "TATASTEEL", "POWERGRID", "NTPC", "M&M", "JSWSTEEL",
    "BAJAJFINSV", "ADANIENT", "ADANIPORTS", "TATAMOTORS", "ONGC",
    "NESTLEIND", "COALINDIA", "TECHM", "INDUSINDBK", "BRITANNIA"
]

# Timeframes to test
TIMEFRAMES = {
    "5minute": {"display": "5 Min", "days_fetch": 60, "candles_per_day": 75},
    "15minute": {"display": "15 Min", "days_fetch": 100, "candles_per_day": 25},
    "60minute": {"display": "1 Hour", "days_fetch": 365, "candles_per_day": 6},
    "day": {"display": "Daily", "days_fetch": 365, "candles_per_day": 1}
}

# Backtest Parameters
INITIAL_CAPITAL = 30000  # ₹30,000
MAX_RISK_PER_TRADE = 0.02  # 2% of capital
MAX_RISK_AMOUNT = 600  # ₹600 max risk per trade
MIN_RR_RATIO = 2.0  # Minimum Risk:Reward ratio

# Strategy Parameters (same as live trading)
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
VWAP_THRESHOLD = 0.005  # 0.5% deviation from VWAP

# Credentials
CREDS_FILE = "/u/tarar/repos/.kite_creds.json"
SESSION_FILE = "/u/tarar/repos/.kite_session.json"


# ============== INDICATOR FUNCTIONS ==============

def calculate_vwap(df):
    """Calculate VWAP (Volume Weighted Average Price)"""
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        # If no volume data, return simple moving average
        return df['close'].rolling(window=20).mean()
    
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap


def calculate_rsi(df, period=14):
    """Calculate RSI (Relative Strength Index)"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(df, period=14):
    """Calculate ATR (Average True Range)"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


# ============== KITE CONNECT SETUP ==============

def load_session():
    """Load Kite Connect session"""
    if not os.path.exists(CREDS_FILE):
        print("❌ Credentials file not found!")
        sys.exit(1)
    
    with open(CREDS_FILE) as f:
        creds = json.load(f)
    
    if not os.path.exists(SESSION_FILE):
        print("❌ Session file not found! Run kite_login.py first.")
        sys.exit(1)
    
    with open(SESSION_FILE) as f:
        session = json.load(f)
    
    kite = KiteConnect(api_key=creds['api_key'])
    kite.set_access_token(session['access_token'])
    
    # Verify session
    try:
        kite.profile()
        print("✅ Kite Connect session active")
        return kite
    except Exception as e:
        print(f"❌ Session invalid: {e}")
        sys.exit(1)


def get_historical_data(kite, symbol, interval, from_date, to_date):
    """
    Fetch historical data from Kite Connect
    
    For longer periods, we need to fetch in chunks due to API limits:
    - 5min/15min: max 60 days per call
    - 60min: max 365 days per call
    - day: max 2000 days per call
    """
    try:
        instrument_token = None
        instruments = kite.instruments("NSE")
        
        for inst in instruments:
            if inst['tradingsymbol'] == symbol:
                instrument_token = inst['instrument_token']
                break
        
        if not instrument_token:
            return None
        
        # Determine chunk size based on interval
        if interval in ["5minute", "15minute"]:
            chunk_days = 55  # Stay under 60 day limit
        else:
            chunk_days = 300  # Larger chunks for hourly/daily
        
        all_data = []
        current_from = from_date
        
        while current_from < to_date:
            current_to = min(current_from + timedelta(days=chunk_days), to_date)
            
            try:
                data = kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=current_from.strftime("%Y-%m-%d"),
                    to_date=current_to.strftime("%Y-%m-%d"),
                    interval=interval
                )
                all_data.extend(data)
            except Exception as e:
                print(f"    Chunk error {current_from} to {current_to}: {e}")
            
            current_from = current_to + timedelta(days=1)
        
        if not all_data:
            return None
        
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date']).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"    Error fetching {symbol}: {e}")
        return None


# ============== STRATEGY IMPLEMENTATION ==============

def vwap_rsi_strategy(df, idx):
    """
    VWAP + RSI Reversal Strategy
    
    BUY Signal: Price below VWAP (by 0.5%+) AND RSI < 30 (oversold)
    SELL Signal: Price above VWAP (by 0.5%+) AND RSI > 70 (overbought)
    
    Returns: signal ('BUY', 'SELL', or None), indicators dict
    """
    if idx < 20:  # Need at least 20 candles for indicators
        return None, {}
    
    # Get data up to current candle (no look-ahead bias)
    data_slice = df.iloc[:idx+1].copy()
    
    # Calculate indicators
    vwap = calculate_vwap(data_slice).iloc[-1]
    rsi = calculate_rsi(data_slice).iloc[-1]
    close = data_slice['close'].iloc[-1]
    
    if pd.isna(vwap) or pd.isna(rsi):
        return None, {}
    
    indicators = {
        "vwap": round(vwap, 2),
        "rsi": round(rsi, 2),
        "close": round(close, 2)
    }
    
    # BUY: Price below VWAP by threshold AND RSI oversold
    if close < vwap * (1 - VWAP_THRESHOLD) and rsi < RSI_OVERSOLD:
        return "BUY", indicators
    
    # SELL: Price above VWAP by threshold AND RSI overbought
    if close > vwap * (1 + VWAP_THRESHOLD) and rsi > RSI_OVERBOUGHT:
        return "SELL", indicators
    
    return None, indicators


# ============== BACKTEST ENGINE ==============

class BacktestEngine:
    """Engine to run backtests with proper position management"""
    
    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades = []
        self.current_position = None
        self.equity_curve = []
    
    def reset(self):
        """Reset for new backtest"""
        self.capital = self.initial_capital
        self.trades = []
        self.current_position = None
        self.equity_curve = []
    
    def calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on fixed risk"""
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0
        
        max_risk = min(MAX_RISK_AMOUNT, self.capital * MAX_RISK_PER_TRADE)
        qty = int(max_risk / risk_per_share)
        
        # Check if we can afford it
        position_value = qty * entry_price
        if position_value > self.capital * 0.9:  # Max 90% of capital
            qty = int((self.capital * 0.9) / entry_price)
        
        return max(1, qty)
    
    def calculate_costs(self, qty, price, is_intraday=True):
        """Calculate trading costs"""
        turnover = qty * price * 2  # Buy + Sell
        
        brokerage = min(40, turnover * 0.0003)  # 0.03% or ₹20 per order
        stt = turnover * 0.00025 if is_intraday else turnover * 0.001
        exchange = turnover * 0.0000345
        gst = brokerage * 0.18
        sebi = turnover * 0.000001
        stamp = turnover * 0.00003
        
        return brokerage + stt + exchange + gst + sebi + stamp
    
    def enter_trade(self, date, symbol, signal, entry_price, stop_loss, target, indicators):
        """Enter a new trade"""
        if self.current_position is not None:
            return False  # Already in a trade
        
        qty = self.calculate_position_size(entry_price, stop_loss)
        if qty < 1:
            return False
        
        self.current_position = {
            "symbol": symbol,
            "signal": signal,
            "entry_date": date,
            "entry_price": entry_price,
            "qty": qty,
            "stop_loss": stop_loss,
            "target": target,
            "indicators": indicators
        }
        return True
    
    def check_exit(self, date, high, low, close):
        """Check if current position should be exited"""
        if self.current_position is None:
            return False
        
        pos = self.current_position
        exit_price = None
        exit_reason = None
        
        if pos['signal'] == "BUY":
            # Check stop loss hit
            if low <= pos['stop_loss']:
                exit_price = pos['stop_loss']
                exit_reason = "Stop Loss"
            # Check target hit
            elif high >= pos['target']:
                exit_price = pos['target']
                exit_reason = "Target"
        else:  # SELL (short)
            # Check stop loss hit
            if high >= pos['stop_loss']:
                exit_price = pos['stop_loss']
                exit_reason = "Stop Loss"
            # Check target hit
            elif low <= pos['target']:
                exit_price = pos['target']
                exit_reason = "Target"
        
        if exit_price is not None:
            self.exit_trade(date, exit_price, exit_reason)
            return True
        
        return False
    
    def exit_trade(self, date, exit_price, reason):
        """Exit current trade and record results"""
        if self.current_position is None:
            return
        
        pos = self.current_position
        
        # Calculate P&L
        if pos['signal'] == "BUY":
            gross_pnl = (exit_price - pos['entry_price']) * pos['qty']
        else:  # SELL
            gross_pnl = (pos['entry_price'] - exit_price) * pos['qty']
        
        costs = self.calculate_costs(pos['qty'], pos['entry_price'])
        net_pnl = gross_pnl - costs
        
        # Update capital
        self.capital += net_pnl
        
        # Record trade
        trade = {
            "symbol": pos['symbol'],
            "signal": pos['signal'],
            "entry_date": pos['entry_date'],
            "entry_price": pos['entry_price'],
            "exit_date": date,
            "exit_price": exit_price,
            "qty": pos['qty'],
            "stop_loss": pos['stop_loss'],
            "target": pos['target'],
            "exit_reason": reason,
            "gross_pnl": round(gross_pnl, 2),
            "costs": round(costs, 2),
            "net_pnl": round(net_pnl, 2),
            "capital_after": round(self.capital, 2),
            "indicators": pos['indicators']
        }
        self.trades.append(trade)
        
        # Record equity
        self.equity_curve.append({
            "date": date,
            "capital": self.capital
        })
        
        # Clear position
        self.current_position = None
    
    def force_exit_eod(self, date, close):
        """Force exit at end of day for intraday"""
        if self.current_position is not None:
            self.exit_trade(date, close, "EOD Exit")
    
    def get_metrics(self):
        """Calculate backtest metrics"""
        if not self.trades:
            return {
                "total_trades": 0,
                "winners": 0,
                "losers": 0,
                "win_rate": 0,
                "gross_pnl": 0,
                "total_costs": 0,
                "net_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "profit_factor": 0,
                "expectancy": 0,
                "max_drawdown": 0,
                "max_drawdown_pct": 0,
                "roi": 0,
                "sharpe_ratio": 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        winners = trades_df[trades_df['net_pnl'] > 0]
        losers = trades_df[trades_df['net_pnl'] <= 0]
        
        total_trades = len(trades_df)
        num_winners = len(winners)
        num_losers = len(losers)
        win_rate = (num_winners / total_trades * 100) if total_trades > 0 else 0
        
        gross_pnl = trades_df['gross_pnl'].sum()
        total_costs = trades_df['costs'].sum()
        net_pnl = trades_df['net_pnl'].sum()
        
        avg_win = winners['net_pnl'].mean() if len(winners) > 0 else 0
        avg_loss = losers['net_pnl'].mean() if len(losers) > 0 else 0
        
        largest_win = winners['net_pnl'].max() if len(winners) > 0 else 0
        largest_loss = losers['net_pnl'].min() if len(losers) > 0 else 0
        
        total_wins = winners['net_pnl'].sum() if len(winners) > 0 else 0
        total_losses = abs(losers['net_pnl'].sum()) if len(losers) > 0 else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
        
        expectancy = trades_df['net_pnl'].mean()
        
        # Calculate max drawdown
        equity = pd.Series([self.initial_capital] + [t['capital_after'] for t in self.trades])
        peak = equity.cummax()
        drawdown = (equity - peak)
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / peak.max() * 100) if peak.max() > 0 else 0
        
        # ROI
        roi = ((self.capital - self.initial_capital) / self.initial_capital * 100)
        
        # Sharpe Ratio (simplified, assuming 0 risk-free rate)
        returns = trades_df['net_pnl'] / self.initial_capital
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        return {
            "total_trades": total_trades,
            "winners": num_winners,
            "losers": num_losers,
            "win_rate": round(win_rate, 2),
            "gross_pnl": round(gross_pnl, 2),
            "total_costs": round(total_costs, 2),
            "net_pnl": round(net_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "largest_win": round(largest_win, 2),
            "largest_loss": round(largest_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "expectancy": round(expectancy, 2),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "roi": round(roi, 2),
            "sharpe_ratio": round(sharpe, 2)
        }


def run_backtest(kite, symbol, df, timeframe, is_intraday=True):
    """
    Run backtest on a single stock for a given timeframe
    
    Args:
        kite: Kite Connect instance
        symbol: Stock symbol
        df: Historical data DataFrame
        timeframe: Timeframe string
        is_intraday: If True, force exit at EOD
    
    Returns:
        BacktestEngine with results
    """
    engine = BacktestEngine()
    
    if df is None or len(df) < 50:
        return engine
    
    # Track the last trading day for EOD exit
    last_date = None
    
    for idx in range(20, len(df)):
        row = df.iloc[idx]
        current_date = row['date']
        
        # For intraday, check if day changed (force exit)
        if is_intraday and last_date is not None:
            if current_date.date() != last_date.date():
                # New day - force exit any open position from previous day
                if engine.current_position is not None:
                    prev_row = df.iloc[idx-1]
                    engine.force_exit_eod(prev_row['date'], prev_row['close'])
        
        last_date = current_date
        
        # Check existing position for exit
        if engine.current_position is not None:
            engine.check_exit(current_date, row['high'], row['low'], row['close'])
        
        # If no position, check for new signal
        if engine.current_position is None:
            signal, indicators = vwap_rsi_strategy(df, idx)
            
            if signal:
                entry_price = row['close']
                atr = calculate_atr(df.iloc[:idx+1]).iloc[-1]
                
                # Calculate SL and Target based on ATR
                sl_distance = max(atr * 1.5, entry_price * 0.005)  # Min 0.5% SL
                target_distance = sl_distance * MIN_RR_RATIO
                
                if signal == "BUY":
                    stop_loss = round(entry_price - sl_distance, 2)
                    target = round(entry_price + target_distance, 2)
                else:  # SELL
                    stop_loss = round(entry_price + sl_distance, 2)
                    target = round(entry_price - target_distance, 2)
                
                engine.enter_trade(
                    date=current_date,
                    symbol=symbol,
                    signal=signal,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target=target,
                    indicators=indicators
                )
    
    # Force exit any remaining position
    if engine.current_position is not None and len(df) > 0:
        engine.force_exit_eod(df.iloc[-1]['date'], df.iloc[-1]['close'])
    
    return engine


# ============== MAIN BACKTEST RUNNER ==============

def main():
    print("\n" + "=" * 80)
    print("🔬 VWAP + RSI REVERSAL STRATEGY BACKTEST")
    print("=" * 80)
    print(f"\n📅 Backtest Period: 1 Year (from {(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')})")
    print(f"💰 Initial Capital: ₹{INITIAL_CAPITAL:,}")
    print(f"📊 Stocks: {len(FNO_STOCKS)} Nifty F&O stocks")
    print(f"⏱️ Timeframes: {', '.join([v['display'] for v in TIMEFRAMES.values()])}")
    print(f"\n🎯 Strategy Rules:")
    print(f"   BUY: Price < VWAP * {1-VWAP_THRESHOLD:.3f} AND RSI < {RSI_OVERSOLD}")
    print(f"   SELL: Price > VWAP * {1+VWAP_THRESHOLD:.3f} AND RSI > {RSI_OVERBOUGHT}")
    print(f"   Risk per trade: ₹{MAX_RISK_AMOUNT} (max {MAX_RISK_PER_TRADE*100}%)")
    print(f"   Min R:R ratio: 1:{MIN_RR_RATIO}")
    
    # Load Kite session
    print("\n" + "-" * 80)
    print("🔐 Loading Kite Connect session...")
    kite = load_session()
    
    # Calculate date range
    to_date = datetime.now()
    from_date = to_date - timedelta(days=365)
    
    # Results storage
    all_results = {}
    
    for tf_key, tf_config in TIMEFRAMES.items():
        print("\n" + "=" * 80)
        print(f"⏱️ TESTING TIMEFRAME: {tf_config['display']}")
        print("=" * 80)
        
        is_intraday = tf_key != "day"
        
        # Aggregate results for this timeframe
        combined_trades = []
        stocks_tested = 0
        stocks_with_data = 0
        
        for i, symbol in enumerate(FNO_STOCKS):
            print(f"\n  [{i+1}/{len(FNO_STOCKS)}] {symbol}...", end=" ", flush=True)
            
            # Fetch data
            df = get_historical_data(kite, symbol, tf_key, from_date, to_date)
            
            if df is None or len(df) < 50:
                print(f"❌ Insufficient data ({len(df) if df is not None else 0} candles)")
                continue
            
            stocks_with_data += 1
            print(f"📊 {len(df)} candles", end=" ", flush=True)
            
            # Run backtest
            engine = run_backtest(kite, symbol, df, tf_key, is_intraday)
            metrics = engine.get_metrics()
            
            if metrics['total_trades'] > 0:
                combined_trades.extend(engine.trades)
                stocks_tested += 1
                print(f"✅ {metrics['total_trades']} trades, Net: ₹{metrics['net_pnl']:,.2f}", flush=True)
            else:
                print(f"⚪ No signals", flush=True)
        
        # Calculate combined metrics for this timeframe
        if combined_trades:
            combined_engine = BacktestEngine()
            combined_engine.trades = combined_trades
            combined_engine.capital = INITIAL_CAPITAL + sum(t['net_pnl'] for t in combined_trades)
            combined_metrics = combined_engine.get_metrics()
            
            all_results[tf_key] = {
                "config": tf_config,
                "stocks_tested": stocks_tested,
                "stocks_with_data": stocks_with_data,
                "metrics": combined_metrics,
                "trades": combined_trades
            }
        else:
            all_results[tf_key] = {
                "config": tf_config,
                "stocks_tested": 0,
                "stocks_with_data": stocks_with_data,
                "metrics": BacktestEngine().get_metrics(),
                "trades": []
            }
    
    # ============== DISPLAY RESULTS ==============
    
    print("\n\n" + "=" * 100)
    print("📊 BACKTEST RESULTS SUMMARY - VWAP + RSI REVERSAL STRATEGY")
    print("=" * 100)
    
    # Summary table header
    print(f"\n{'Timeframe':<12} {'Trades':<8} {'Win%':<8} {'Net P&L':<14} {'Costs':<10} {'P.Factor':<10} {'Expectancy':<12} {'Max DD%':<10} {'ROI%':<10}")
    print("-" * 100)
    
    best_tf = None
    best_roi = float('-inf')
    
    for tf_key, result in all_results.items():
        m = result['metrics']
        tf_name = result['config']['display']
        
        print(f"{tf_name:<12} {m['total_trades']:<8} {m['win_rate']:<8.1f} ₹{m['net_pnl']:<12,.2f} ₹{m['total_costs']:<8,.2f} {m['profit_factor']:<10.2f} ₹{m['expectancy']:<10,.2f} {m['max_drawdown_pct']:<10.2f} {m['roi']:<10.2f}")
        
        if m['roi'] > best_roi and m['total_trades'] > 10:
            best_roi = m['roi']
            best_tf = tf_key
    
    print("-" * 100)
    
    # Detailed metrics for each timeframe
    print("\n\n📈 DETAILED METRICS BY TIMEFRAME")
    print("=" * 100)
    
    for tf_key, result in all_results.items():
        m = result['metrics']
        tf_name = result['config']['display']
        
        print(f"\n┌─ {tf_name} Timeframe ─────────────────────────────────────────────────────")
        print(f"│")
        print(f"│  Stocks Analyzed: {result['stocks_with_data']}")
        print(f"│  Stocks with Signals: {result['stocks_tested']}")
        print(f"│")
        print(f"│  📊 Trade Statistics:")
        print(f"│     Total Trades: {m['total_trades']}")
        print(f"│     Winners: {m['winners']} | Losers: {m['losers']}")
        print(f"│     Win Rate: {m['win_rate']:.1f}%")
        print(f"│")
        print(f"│  💰 P&L:")
        print(f"│     Gross P&L: ₹{m['gross_pnl']:,.2f}")
        print(f"│     Trading Costs: ₹{m['total_costs']:,.2f}")
        print(f"│     Net P&L: ₹{m['net_pnl']:,.2f}")
        print(f"│     ROI: {m['roi']:.2f}%")
        print(f"│")
        print(f"│  📉 Risk Metrics:")
        print(f"│     Avg Win: ₹{m['avg_win']:,.2f}")
        print(f"│     Avg Loss: ₹{m['avg_loss']:,.2f}")
        print(f"│     Largest Win: ₹{m['largest_win']:,.2f}")
        print(f"│     Largest Loss: ₹{m['largest_loss']:,.2f}")
        print(f"│     Max Drawdown: ₹{m['max_drawdown']:,.2f} ({m['max_drawdown_pct']:.2f}%)")
        print(f"│")
        print(f"│  📐 Performance Ratios:")
        print(f"│     Profit Factor: {m['profit_factor']:.2f}")
        print(f"│     Expectancy: ₹{m['expectancy']:,.2f}")
        print(f"│     Sharpe Ratio: {m['sharpe_ratio']:.2f}")
        print(f"└───────────────────────────────────────────────────────────────────────────")
    
    # Best timeframe
    if best_tf:
        print(f"\n\n🏆 BEST PERFORMING TIMEFRAME: {all_results[best_tf]['config']['display']}")
        print(f"   ROI: {all_results[best_tf]['metrics']['roi']:.2f}%")
        print(f"   Profit Factor: {all_results[best_tf]['metrics']['profit_factor']:.2f}")
        print(f"   Win Rate: {all_results[best_tf]['metrics']['win_rate']:.1f}%")
    
    # Signal distribution analysis
    print("\n\n📊 SIGNAL TYPE ANALYSIS")
    print("=" * 80)
    
    for tf_key, result in all_results.items():
        if not result['trades']:
            continue
        
        trades_df = pd.DataFrame(result['trades'])
        buy_trades = trades_df[trades_df['signal'] == 'BUY']
        sell_trades = trades_df[trades_df['signal'] == 'SELL']
        
        print(f"\n{result['config']['display']}:")
        print(f"  BUY signals:  {len(buy_trades):3d} trades, Net P&L: ₹{buy_trades['net_pnl'].sum():,.2f}, Win Rate: {(len(buy_trades[buy_trades['net_pnl'] > 0]) / len(buy_trades) * 100):.1f}%" if len(buy_trades) > 0 else "  BUY signals:  0 trades")
        print(f"  SELL signals: {len(sell_trades):3d} trades, Net P&L: ₹{sell_trades['net_pnl'].sum():,.2f}, Win Rate: {(len(sell_trades[sell_trades['net_pnl'] > 0]) / len(sell_trades) * 100):.1f}%" if len(sell_trades) > 0 else "  SELL signals: 0 trades")
    
    # Exit reason analysis
    print("\n\n📊 EXIT REASON ANALYSIS")
    print("=" * 80)
    
    for tf_key, result in all_results.items():
        if not result['trades']:
            continue
        
        trades_df = pd.DataFrame(result['trades'])
        
        print(f"\n{result['config']['display']}:")
        for reason in trades_df['exit_reason'].unique():
            reason_trades = trades_df[trades_df['exit_reason'] == reason]
            print(f"  {reason:12s}: {len(reason_trades):3d} trades ({len(reason_trades)/len(trades_df)*100:.1f}%), Net P&L: ₹{reason_trades['net_pnl'].sum():,.2f}")
    
    # Save results to file
    results_file = f"/u/tarar/repos/backtest_vwap_rsi_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert to serializable format
    save_results = {}
    for tf_key, result in all_results.items():
        save_results[tf_key] = {
            "config": result['config'],
            "stocks_tested": result['stocks_tested'],
            "stocks_with_data": result['stocks_with_data'],
            "metrics": result['metrics'],
            "trade_count": len(result['trades'])
        }
    
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    
    print(f"\n\n💾 Results saved to: {results_file}")
    print("\n" + "=" * 80)
    print("✅ BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
