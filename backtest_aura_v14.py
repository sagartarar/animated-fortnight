#!/usr/bin/env python3
"""
AURA V14 Strategy Backtest
==========================
Converts the TradingView Pine Script Aura V14 indicator into a Python backtest.

Components:
1. Alpha Trend (MFI-based dynamic support/resistance)
2. Magic Trend (CCI-based trend)
3. UT Bot (ATR trailing stop)
4. Consensus Engine (RSI, MFI, ADX, EMA50)
5. Volume Filter

BUY Signal: All bullish conditions align + volume confirmation
SELL Signal: All bearish conditions align + volume confirmation

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

# Aura V14 Parameters (matching Pine Script)
ATR_PERIOD = 1
ATR_MULTIPLIER = 5.0
ALPHA_PERIOD = 14
ALPHA_MULTIPLIER = 1.0
MAGIC_PERIOD = 20
MAGIC_MULTIPLIER = 1.0

# Backtest Parameters
INITIAL_CAPITAL = 30000  # ₹30,000
MAX_RISK_PER_TRADE = 0.02  # 2% of capital
MAX_RISK_AMOUNT = 600  # ₹600 max risk per trade
MIN_RR_RATIO = 2.0  # Minimum Risk:Reward ratio

# Timeframes to test
TIMEFRAMES = {
    "15minute": {"display": "15 Min", "days_fetch": 100, "is_intraday": True},
    "60minute": {"display": "1 Hour", "days_fetch": 365, "is_intraday": True},
    "day": {"display": "Daily", "days_fetch": 365, "is_intraday": False}
}

# Nifty 200 Stocks (Top by market cap and liquidity)
NIFTY_200 = [
    # Nifty 50
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
    "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "ASIANPAINT",
    "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO",
    "TATASTEEL", "POWERGRID", "NTPC", "M&M", "JSWSTEEL",
    "BAJAJFINSV", "ADANIENT", "ADANIPORTS", "ONGC", "NESTLEIND",
    "COALINDIA", "TECHM", "INDUSINDBK", "BRITANNIA", "HINDALCO",
    "DRREDDY", "DIVISLAB", "CIPLA", "EICHERMOT", "GRASIM",
    "APOLLOHOSP", "BPCL", "TATACONSUM", "HEROMOTOCO", "SHREECEM",
    "SBILIFE", "HDFCLIFE", "UPL", "BAJAJ-AUTO", "TATAMOTORS",
    # Nifty Next 50
    "BANKBARODA", "PNB", "CANBK", "IOC", "GAIL",
    "VEDL", "JINDALSTEL", "DLF", "SAIL", "NMDC",
    "PIDILITIND", "HAVELLS", "SIEMENS", "ABB", "GODREJCP",
    "DABUR", "MARICO", "COLPAL", "PGHH", "BERGEPAINT",
    "INDIGO", "TRENT", "ZOMATO", "PAYTM", "NYKAA",
    "POLICYBZR", "DELHIVERY", "IRCTC", "LODHA", "MANKIND",
    "MAXHEALTH", "FORTIS", "AUROPHARMA", "TORNTPHARM", "LUPIN",
    "ZYDUSLIFE", "BIOCON", "ALKEM", "IPCALAB", "LAURUSLABS",
    "TATAPOWER", "ADANIGREEN", "ADANIPOWER", "NHPC", "SJVN",
    "RECLTD", "PFC", "IRFC", "RVNL", "BEL",
    # Nifty Midcap Select
    "MPHASIS", "LTIM", "PERSISTENT", "COFORGE", "LTTS",
    "HAPPSTMNDS", "TATAELXSI", "ROUTE", "SONACOMS", "MOTHERSON",
    "BOSCHLTD", "MRF", "ASHOKLEY", "ESCORTS", "EXIDEIND",
    "AMARAJABAT", "BALKRISIND", "APOLLOTYRE", "CEAT", "TVSMTR",
    "CHOLAFIN", "MUTHOOTFIN", "MANAPPURAM", "IIFL", "LICHSGFIN",
    "CANFINHOME", "ABCAPITAL", "M&MFIN", "SHRIRAMFIN", "SUNDARMFIN",
    "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "RBLBANK", "AUBANK",
    "INDIANB", "CENTRALBK", "UNIONBANK", "MAHABANK", "UCOBANK",
    # Additional Nifty 200
    "OBEROIRLTY", "PRESTIGE", "BRIGADE", "SOBHA", "GODREJPROP",
    "PHOENIXLTD", "LICI", "SBICARD", "HDFCAMC", "ICICIGI",
    "ICICIPRULI", "STARHEALTH", "NIACL", "GICRE", "IIFLWAM",
    "CRISIL", "ICRA", "CARERATING", "METROPOLIS", "LALPATHLAB",
    "DMART", "TATACOMM", "IDEA", "MTNL", "RCOM",
    "ABFRL", "PAGEIND", "RELAXO", "BATAINDIA", "CAMPUS",
    "JUBLFOOD", "DEVYANI", "WESTLIFE", "ZOMATO", "SWIGGY",
    "PATANJALI", "EMAMILTD", "JYOTHYLAB", "VGUARD", "CROMPTON",
    "VOLTAS", "BLUESTAR", "WHIRLPOOL", "DIXON", "AMBER",
    "POLYCAB", "KAYNES", "CLEAN", "AFFLE", "LATENTVIEW",
    "TANLA", "KPITTECH", "MASTEK", "CYIENT", "BIRLASOFT",
    "ZENSAR", "NIITLTD", "INTELLECT", "NEWGEN", "DATAPATTNS",
    "PIIND", "ATUL", "DEEPAKNTR", "AARTI", "FINEORG",
    "CLEAN", "NAVINFLUOR", "SRF", "FLUOROCHEM", "GUJGASLTD",
    "PETRONET", "GSPL", "IGL", "MGL", "ATGL"
]

# Remove duplicates and keep unique
NIFTY_200 = list(dict.fromkeys(NIFTY_200))

# Credentials
CREDS_FILE = "/u/tarar/repos/.kite_creds.json"
SESSION_FILE = "/u/tarar/repos/.kite_session.json"


# ============== INDICATOR FUNCTIONS ==============

def calculate_tr(df):
    """Calculate True Range"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr


def calculate_atr(df, period=14):
    """Calculate ATR (Average True Range)"""
    tr = calculate_tr(df)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_rsi(df, period=14):
    """Calculate RSI (Relative Strength Index)"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_mfi(df, period=14):
    """Calculate Money Flow Index"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    delta = typical_price.diff()
    positive_flow = money_flow.where(delta > 0, 0).rolling(window=period).sum()
    negative_flow = money_flow.where(delta < 0, 0).rolling(window=period).sum()
    
    mfi_ratio = positive_flow / negative_flow
    mfi = 100 - (100 / (1 + mfi_ratio))
    return mfi


def calculate_cci(df, period=20):
    """Calculate Commodity Channel Index"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mad)
    return cci


def calculate_ema(df, period):
    """Calculate Exponential Moving Average"""
    return df['close'].ewm(span=period, adjust=False).mean()


def calculate_sma(series, period):
    """Calculate Simple Moving Average"""
    return series.rolling(window=period).mean()


def calculate_dmi(df, period=14):
    """
    Calculate DMI (Directional Movement Index)
    Returns: plus_di, minus_di, adx
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Calculate TR
    tr = calculate_tr(df)
    
    # Smoothed values
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return plus_di, minus_di, adx


# ============== AURA V14 STRATEGY ==============

def calculate_alpha_trend(df, period=14, multiplier=1.0):
    """
    Calculate Alpha Trend Line (MFI-based dynamic support/resistance)
    
    Logic:
    - When MFI >= 50 (bullish): Use lower band (support)
    - When MFI < 50 (bearish): Use upper band (resistance)
    - Line only moves in favorable direction (ratchets)
    """
    mfi = calculate_mfi(df, period)
    at_atr = calculate_sma(calculate_tr(df), period)
    
    alpha_up = df['low'] - (at_atr * multiplier)
    alpha_dn = df['high'] + (at_atr * multiplier)
    
    alpha_line = pd.Series(index=df.index, dtype=float)
    alpha_line.iloc[0] = df['close'].iloc[0]
    
    for i in range(1, len(df)):
        prev_line = alpha_line.iloc[i-1]
        
        if mfi.iloc[i] >= 50:
            # Bullish - use support, only move up
            alpha_line.iloc[i] = max(prev_line, alpha_up.iloc[i]) if not pd.isna(alpha_up.iloc[i]) else prev_line
        else:
            # Bearish - use resistance, only move down
            alpha_line.iloc[i] = min(prev_line, alpha_dn.iloc[i]) if not pd.isna(alpha_dn.iloc[i]) else prev_line
    
    return alpha_line


def calculate_ut_bot_trailing_stop(df, atr_period=1, multiplier=5.0):
    """
    Calculate UT Bot trailing stop (n_loss in Pine Script)
    
    Logic:
    - Trailing stop that follows price
    - Uses ATR for distance calculation
    - Ratchets in trend direction
    """
    atr = calculate_atr(df, atr_period)
    x_atr = atr * multiplier
    
    n_loss = pd.Series(index=df.index, dtype=float)
    n_loss.iloc[0] = df['close'].iloc[0]
    
    for i in range(1, len(df)):
        close = df['close'].iloc[i]
        close_prev = df['close'].iloc[i-1]
        prev_l = n_loss.iloc[i-1]
        x = x_atr.iloc[i] if not pd.isna(x_atr.iloc[i]) else 0
        
        if close > prev_l and close_prev > prev_l:
            # Uptrend - raise stop
            n_loss.iloc[i] = max(prev_l, close - x)
        elif close < prev_l and close_prev < prev_l:
            # Downtrend - lower stop
            n_loss.iloc[i] = min(prev_l, close + x)
        elif close > prev_l:
            # Turned bullish
            n_loss.iloc[i] = close - x
        else:
            # Turned bearish
            n_loss.iloc[i] = close + x
    
    return n_loss


def aura_v14_strategy(df, idx):
    """
    AURA V14 Strategy - Full implementation
    
    BUY Signal requires ALL:
    1. Consensus bullish (bull_sc >= 3)
    2. Alpha Trend bullish (close > alpha_line)
    3. Magic Trend bullish (CCI > 0)
    4. UT Bot bullish (close > n_loss)
    5. Volume confirmation (volume > 20-SMA volume)
    
    SELL Signal requires ALL:
    1. Consensus bearish (bull_sc <= 1)
    2. Alpha Trend bearish (close <= alpha_line)
    3. Magic Trend bearish (CCI <= 0)
    4. UT Bot bearish (close < n_loss)
    5. Volume confirmation
    
    Returns: signal ('BUY', 'SELL', or None), indicators dict
    """
    min_bars = max(50, ALPHA_PERIOD + 20, MAGIC_PERIOD + 5)
    
    if idx < min_bars:
        return None, {}
    
    # Get data up to current candle (no look-ahead bias)
    data = df.iloc[:idx+1].copy()
    
    # Calculate all indicators
    close = data['close'].iloc[-1]
    volume = data['volume'].iloc[-1]
    
    # 1. Consensus Engine
    rsi = calculate_rsi(data, 14).iloc[-1]
    mfi = calculate_mfi(data, 14).iloc[-1]
    plus_di, minus_di, adx = calculate_dmi(data, 14)
    adx_val = adx.iloc[-1]
    ema50 = calculate_ema(data, 50).iloc[-1]
    
    bull_score = 0
    if rsi > 50:
        bull_score += 1
    if mfi > 50:
        bull_score += 1
    if adx_val > 30:
        bull_score += 1
    if close > ema50:
        bull_score += 1
    
    # 2. Alpha Trend
    alpha_line = calculate_alpha_trend(data, ALPHA_PERIOD, ALPHA_MULTIPLIER)
    at_bull = close > alpha_line.iloc[-1]
    
    # 3. Magic Trend (CCI-based)
    cci = calculate_cci(data, MAGIC_PERIOD).iloc[-1]
    magic_bull = cci > 0
    
    # 4. UT Bot Trailing Stop
    n_loss = calculate_ut_bot_trailing_stop(data, ATR_PERIOD, ATR_MULTIPLIER)
    ut_bull = close > n_loss.iloc[-1]
    
    # 5. Volume Filter
    vol_sma = calculate_sma(data['volume'], 20).iloc[-1]
    vol_ok = volume > vol_sma
    
    # Store indicators
    indicators = {
        "close": round(close, 2),
        "rsi": round(rsi, 2) if not pd.isna(rsi) else 0,
        "mfi": round(mfi, 2) if not pd.isna(mfi) else 0,
        "adx": round(adx_val, 2) if not pd.isna(adx_val) else 0,
        "cci": round(cci, 2) if not pd.isna(cci) else 0,
        "ema50": round(ema50, 2) if not pd.isna(ema50) else 0,
        "alpha_line": round(alpha_line.iloc[-1], 2) if not pd.isna(alpha_line.iloc[-1]) else 0,
        "ut_stop": round(n_loss.iloc[-1], 2) if not pd.isna(n_loss.iloc[-1]) else 0,
        "bull_score": bull_score,
        "at_bull": at_bull,
        "magic_bull": magic_bull,
        "ut_bull": ut_bull,
        "vol_ok": vol_ok
    }
    
    # Generate signals
    is_bull = bull_score >= 3 and at_bull and magic_bull and ut_bull and vol_ok
    is_bear = bull_score <= 1 and not at_bull and not magic_bull and not ut_bull and vol_ok
    
    if is_bull:
        return "BUY", indicators
    elif is_bear:
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
        
        return max(1, qty) if qty > 0 else 0
    
    def calculate_costs(self, qty, price, is_intraday=True):
        """Calculate trading costs"""
        turnover = qty * price * 2  # Buy + Sell
        
        brokerage = min(40, turnover * 0.0003)
        stt = turnover * 0.00025 if is_intraday else turnover * 0.001
        exchange = turnover * 0.0000345
        gst = brokerage * 0.18
        sebi = turnover * 0.000001
        stamp = turnover * 0.00003
        
        return brokerage + stt + exchange + gst + sebi + stamp
    
    def enter_trade(self, date, symbol, signal, entry_price, stop_loss, target, indicators):
        """Enter a new trade"""
        if self.current_position is not None:
            return False
        
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
            if low <= pos['stop_loss']:
                exit_price = pos['stop_loss']
                exit_reason = "Stop Loss"
            elif high >= pos['target']:
                exit_price = pos['target']
                exit_reason = "Target"
        else:  # SELL
            if high >= pos['stop_loss']:
                exit_price = pos['stop_loss']
                exit_reason = "Stop Loss"
            elif low <= pos['target']:
                exit_price = pos['target']
                exit_reason = "Target"
        
        if exit_price is not None:
            self.exit_trade(date, exit_price, exit_reason)
            return True
        
        return False
    
    def exit_trade(self, date, exit_price, reason, is_intraday=True):
        """Exit current trade and record results"""
        if self.current_position is None:
            return
        
        pos = self.current_position
        
        if pos['signal'] == "BUY":
            gross_pnl = (exit_price - pos['entry_price']) * pos['qty']
        else:
            gross_pnl = (pos['entry_price'] - exit_price) * pos['qty']
        
        costs = self.calculate_costs(pos['qty'], pos['entry_price'], is_intraday)
        net_pnl = gross_pnl - costs
        
        self.capital += net_pnl
        
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
        
        self.equity_curve.append({
            "date": date,
            "capital": self.capital
        })
        
        self.current_position = None
    
    def force_exit_eod(self, date, close, is_intraday=True):
        """Force exit at end of day for intraday"""
        if self.current_position is not None:
            self.exit_trade(date, close, "EOD Exit", is_intraday)
    
    def get_metrics(self):
        """Calculate backtest metrics"""
        if not self.trades:
            return {
                "total_trades": 0, "winners": 0, "losers": 0, "win_rate": 0,
                "gross_pnl": 0, "total_costs": 0, "net_pnl": 0,
                "avg_win": 0, "avg_loss": 0, "largest_win": 0, "largest_loss": 0,
                "profit_factor": 0, "expectancy": 0,
                "max_drawdown": 0, "max_drawdown_pct": 0, "roi": 0, "sharpe_ratio": 0
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
        
        equity = pd.Series([self.initial_capital] + [t['capital_after'] for t in self.trades])
        peak = equity.cummax()
        drawdown = (equity - peak)
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / peak.max() * 100) if peak.max() > 0 else 0
        
        roi = ((self.capital - self.initial_capital) / self.initial_capital * 100)
        
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


# ============== KITE CONNECT ==============

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
    
    try:
        kite.profile()
        print("✅ Kite Connect session active")
        return kite
    except Exception as e:
        print(f"❌ Session invalid: {e}")
        sys.exit(1)


def get_historical_data(kite, symbol, interval, from_date, to_date):
    """Fetch historical data from Kite Connect"""
    try:
        instrument_token = None
        instruments = kite.instruments("NSE")
        
        for inst in instruments:
            if inst['tradingsymbol'] == symbol:
                instrument_token = inst['instrument_token']
                break
        
        if not instrument_token:
            return None
        
        # Chunk size based on interval
        if interval in ["5minute", "15minute"]:
            chunk_days = 55
        else:
            chunk_days = 300
        
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
                pass
            
            current_from = current_to + timedelta(days=1)
        
        if not all_data:
            return None
        
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df.drop_duplicates(subset=['date']).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        return None


def run_backtest(kite, symbol, df, timeframe, is_intraday=True):
    """Run backtest on a single stock"""
    engine = BacktestEngine()
    
    if df is None or len(df) < 60:
        return engine
    
    last_date = None
    
    for idx in range(50, len(df)):
        row = df.iloc[idx]
        current_date = row['date']
        
        # EOD exit for intraday
        if is_intraday and last_date is not None:
            if current_date.date() != last_date.date():
                if engine.current_position is not None:
                    prev_row = df.iloc[idx-1]
                    engine.force_exit_eod(prev_row['date'], prev_row['close'], is_intraday)
        
        last_date = current_date
        
        # Check exit
        if engine.current_position is not None:
            engine.check_exit(current_date, row['high'], row['low'], row['close'])
        
        # Check new signal
        if engine.current_position is None:
            signal, indicators = aura_v14_strategy(df, idx)
            
            if signal:
                entry_price = row['close']
                atr = calculate_atr(df.iloc[:idx+1]).iloc[-1]
                
                sl_distance = max(atr * 1.5, entry_price * 0.01)  # Min 1% SL
                target_distance = sl_distance * MIN_RR_RATIO
                
                if signal == "BUY":
                    stop_loss = round(entry_price - sl_distance, 2)
                    target = round(entry_price + target_distance, 2)
                else:
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
    
    # Final exit
    if engine.current_position is not None and len(df) > 0:
        engine.force_exit_eod(df.iloc[-1]['date'], df.iloc[-1]['close'], is_intraday)
    
    return engine


# ============== MAIN ==============

def main():
    print("\n" + "=" * 80)
    print("🔮 AURA V14 STRATEGY BACKTEST")
    print("=" * 80)
    print(f"\n📅 Backtest Period: 1 Year")
    print(f"💰 Initial Capital: ₹{INITIAL_CAPITAL:,}")
    print(f"📊 Stocks: {len(NIFTY_200)} Nifty 200 stocks")
    print(f"⏱️ Timeframes: {', '.join([v['display'] for v in TIMEFRAMES.values()])}")
    print(f"\n🎯 Aura V14 Strategy Components:")
    print(f"   • Alpha Trend (MFI-based, period={ALPHA_PERIOD})")
    print(f"   • Magic Trend (CCI-based, period={MAGIC_PERIOD})")
    print(f"   • UT Bot (ATR trailing stop, period={ATR_PERIOD}, mult={ATR_MULTIPLIER})")
    print(f"   • Consensus (RSI>50, MFI>50, ADX>30, Close>EMA50)")
    print(f"   • Volume Filter (> 20-SMA)")
    print(f"\n📋 Signal Rules:")
    print(f"   BUY: All 5 components bullish")
    print(f"   SELL: All 5 components bearish")
    
    # Load session
    print("\n" + "-" * 80)
    print("🔐 Loading Kite Connect session...")
    kite = load_session()
    
    # Date range
    to_date = datetime.now()
    from_date = to_date - timedelta(days=365)
    
    all_results = {}
    
    for tf_key, tf_config in TIMEFRAMES.items():
        print("\n" + "=" * 80)
        print(f"⏱️ TESTING TIMEFRAME: {tf_config['display']}")
        print("=" * 80)
        
        is_intraday = tf_config['is_intraday']
        combined_trades = []
        stocks_tested = 0
        stocks_with_data = 0
        
        for i, symbol in enumerate(NIFTY_200):
            print(f"\r  [{i+1}/{len(NIFTY_200)}] {symbol}...", end=" ", flush=True)
            
            df = get_historical_data(kite, symbol, tf_key, from_date, to_date)
            
            if df is None or len(df) < 60:
                continue
            
            stocks_with_data += 1
            
            engine = run_backtest(kite, symbol, df, tf_key, is_intraday)
            metrics = engine.get_metrics()
            
            if metrics['total_trades'] > 0:
                combined_trades.extend(engine.trades)
                stocks_tested += 1
                print(f"✅ {metrics['total_trades']} trades, Net: ₹{metrics['net_pnl']:,.2f}", flush=True)
            else:
                print(f"⚪ No signals", flush=True)
        
        # Combined metrics
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
    print("📊 BACKTEST RESULTS SUMMARY - AURA V14 STRATEGY")
    print("=" * 100)
    
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
    
    # Detailed metrics
    print("\n\n📈 DETAILED METRICS BY TIMEFRAME")
    print("=" * 100)
    
    for tf_key, result in all_results.items():
        m = result['metrics']
        tf_name = result['config']['display']
        
        print(f"\n┌─ {tf_name} Timeframe ─────────────────────────────────────────────────────")
        print(f"│  Stocks Analyzed: {result['stocks_with_data']}")
        print(f"│  Stocks with Signals: {result['stocks_tested']}")
        print(f"│  📊 Trade Statistics:")
        print(f"│     Total Trades: {m['total_trades']}")
        print(f"│     Winners: {m['winners']} | Losers: {m['losers']}")
        print(f"│     Win Rate: {m['win_rate']:.1f}%")
        print(f"│  💰 P&L:")
        print(f"│     Gross P&L: ₹{m['gross_pnl']:,.2f}")
        print(f"│     Trading Costs: ₹{m['total_costs']:,.2f}")
        print(f"│     Net P&L: ₹{m['net_pnl']:,.2f}")
        print(f"│     ROI: {m['roi']:.2f}%")
        print(f"│  📉 Risk Metrics:")
        print(f"│     Avg Win: ₹{m['avg_win']:,.2f} | Avg Loss: ₹{m['avg_loss']:,.2f}")
        print(f"│     Max Drawdown: {m['max_drawdown_pct']:.2f}%")
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
    
    # Signal analysis
    print("\n\n📊 SIGNAL TYPE ANALYSIS")
    print("=" * 80)
    
    for tf_key, result in all_results.items():
        if not result['trades']:
            continue
        
        trades_df = pd.DataFrame(result['trades'])
        buy_trades = trades_df[trades_df['signal'] == 'BUY']
        sell_trades = trades_df[trades_df['signal'] == 'SELL']
        
        print(f"\n{result['config']['display']}:")
        if len(buy_trades) > 0:
            buy_wr = len(buy_trades[buy_trades['net_pnl'] > 0]) / len(buy_trades) * 100
            print(f"  BUY signals:  {len(buy_trades):3d} trades, Net P&L: ₹{buy_trades['net_pnl'].sum():,.2f}, Win Rate: {buy_wr:.1f}%")
        if len(sell_trades) > 0:
            sell_wr = len(sell_trades[sell_trades['net_pnl'] > 0]) / len(sell_trades) * 100
            print(f"  SELL signals: {len(sell_trades):3d} trades, Net P&L: ₹{sell_trades['net_pnl'].sum():,.2f}, Win Rate: {sell_wr:.1f}%")
    
    # Save results
    results_file = f"/u/tarar/repos/backtest_aura_v14_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
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
