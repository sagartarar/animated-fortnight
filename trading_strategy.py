"""
OPTIMIZED INTRADAY TRADING STRATEGY
=====================================
Backtested Results (36 trading days, Nifty 100):
- 3,679 trades
- 48.6% win rate  
- ₹13,37,257 net profit
- ₹363 per trade average

KEY RULES:
1. Trade F&O Futures only (not equity - charges kill profits)
2. Entry time: 11:00 AM to 2:45 PM only (avoid morning volatility)
3. Exit by 3:15 PM (mandatory)
4. Score >= 7 for entry
5. Risk:Reward = 1:1.5 using ATR-based SL/Target
6. Max 2 trades per stock per day
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from kiteconnect import KiteConnect
import pytz

IST = pytz.timezone('Asia/Kolkata')

# ============== CONFIGURATION ==============

class StrategyConfig:
    # Entry timing
    ENTRY_START = time(11, 0)   # 11:00 AM - KEY OPTIMIZATION
    ENTRY_END = time(14, 45)    # 2:45 PM
    EXIT_TIME = time(15, 15)    # 3:15 PM mandatory exit
    
    # Scoring
    MIN_SCORE = 7
    
    # Risk management
    ATR_SL_MULTIPLIER = 1.5     # SL = ATR * 1.5
    RISK_REWARD_RATIO = 1.5    # Target = SL * 1.5
    MAX_TRADES_PER_DAY = 2     # Per stock
    
    # Position sizing
    RISK_PER_TRADE = 2000      # ₹2000 risk per trade
    
    # Timeframe
    INTERVAL = '15minute'

# ============== NIFTY 100 STOCKS ==============

NIFTY_100 = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'SBIN',
    'BHARTIARTL', 'KOTAKBANK', 'ITC', 'LT', 'AXISBANK', 'BAJFINANCE', 'MARUTI',
    'ASIANPAINT', 'HCLTECH', 'SUNPHARMA', 'TITAN', 'WIPRO', 'TATASTEEL',
    'POWERGRID', 'NTPC', 'M&M', 'JSWSTEEL', 'ADANIENT', 'BAJAJFINSV',
    'TECHM', 'INDUSINDBK', 'HINDALCO', 'DRREDDY', 'CIPLA', 'TATAPOWER',
    'DLF', 'BANKBARODA', 'PNB', 'GRASIM', 'ULTRACEMCO', 'TATAMOTORS',
    'ONGC', 'COALINDIA', 'NESTLEIND', 'BRITANNIA', 'APOLLOHOSP', 'DIVISLAB',
    'EICHERMOT', 'ADANIPORTS', 'BPCL', 'HEROMOTOCO', 'SHREECEM', 'DABUR',
    'TATACONSUM', 'BAJAJ-AUTO', 'SBILIFE', 'HDFCLIFE', 'GODREJCP', 'PIDILITIND',
    'SIEMENS', 'HAVELLS', 'MOTHERSON', 'AMBUJACEM', 'ACC', 'BERGEPAINT',
    'MCDOWELL-N', 'CHOLAFIN', 'MARICO', 'INDIGO', 'BANDHANBNK', 'VEDL',
    'GAIL', 'IOC', 'LUPIN', 'TORNTPHARM', 'ZYDUSLIFE', 'AUROPHARMA',
    'BIOCON', 'IDFCFIRSTB', 'PEL', 'JINDALSTEL', 'TRENT', 'VOLTAS',
    'MPHASIS', 'LTIM', 'COFORGE', 'PERSISTENT', 'LICI', 'IRCTC',
    'INDUSTOWER', 'NAUKRI', 'PAYTM', 'ZOMATO', 'POLICYBZR', 'DMART',
    'ADANIGREEN', 'ADANITRANS', 'NHPC', 'RECLTD', 'PFC', 'CANBK',
    'FEDERALBNK', 'IDBI'
]

# ============== INDICATORS ==============

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_vwap(df):
    """Calculate Volume Weighted Average Price (resets daily)"""
    df = df.copy()
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_volume'] = df['typical_price'] * df['volume']
    df['date_only'] = df['date'].dt.date
    df['cum_tp_vol'] = df.groupby('date_only')['tp_volume'].cumsum()
    df['cum_vol'] = df.groupby('date_only')['volume'].cumsum()
    return df['cum_tp_vol'] / df['cum_vol']

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_supertrend(df, period=10, multiplier=3):
    """Calculate Supertrend indicator"""
    df = df.copy()
    atr = calculate_atr(df, period)
    
    hl2 = (df['high'] + df['low']) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = -1
        else:
            if df['close'].iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lowerband.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upperband.iloc[i]
                direction.iloc[i] = -1
    
    return direction, atr

def prepare_data(df):
    """Add all indicators to dataframe"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date
    df['time'] = df['date'].dt.time
    
    df['rsi'] = calculate_rsi(df['close'])
    df['ema9'] = calculate_ema(df['close'], 9)
    df['ema21'] = calculate_ema(df['close'], 21)
    df['vwap'] = calculate_vwap(df)
    df['supertrend'], df['atr'] = calculate_supertrend(df)
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    return df

# ============== SCORING SYSTEM ==============

def score_buy_trade(row, nifty_change):
    """
    Score a potential BUY trade (0-10 scale)
    Higher score = higher confidence
    """
    score = 0
    reasons = []
    
    # Nifty alignment (0-2)
    if nifty_change > 0.3:
        score += 2
        reasons.append('Nifty strong')
    elif nifty_change > 0.1:
        score += 1
        reasons.append('Nifty positive')
    
    # RSI zone (0-2)
    if 35 <= row['rsi'] <= 55:
        score += 2
        reasons.append('RSI optimal')
    elif 30 <= row['rsi'] < 35 or 55 < row['rsi'] <= 65:
        score += 1
        reasons.append('RSI acceptable')
    
    # VWAP (0-1)
    if row['close'] > row['vwap']:
        score += 1
        reasons.append('Above VWAP')
    
    # Supertrend (0-2)
    if row['supertrend'] == 1:
        score += 2
        reasons.append('Supertrend bullish')
    
    # EMA alignment (0-2)
    if row['ema9'] > row['ema21']:
        score += 1
        reasons.append('EMA bullish')
    if row['close'] > row['ema9']:
        score += 1
        reasons.append('Price > EMA9')
    
    return score, reasons

def score_short_trade(row, nifty_change):
    """
    Score a potential SHORT trade (0-10 scale)
    """
    score = 0
    reasons = []
    
    # Nifty alignment (0-2)
    if nifty_change < -0.3:
        score += 2
        reasons.append('Nifty weak')
    elif nifty_change < -0.1:
        score += 1
        reasons.append('Nifty negative')
    
    # RSI zone (0-2)
    if 55 <= row['rsi'] <= 70:
        score += 2
        reasons.append('RSI optimal')
    elif 70 < row['rsi'] <= 80:
        score += 1
        reasons.append('RSI overbought')
    
    # VWAP (0-1)
    if row['close'] < row['vwap']:
        score += 1
        reasons.append('Below VWAP')
    
    # Supertrend (0-2)
    if row['supertrend'] == -1:
        score += 2
        reasons.append('Supertrend bearish')
    
    # EMA alignment (0-2)
    if row['ema9'] < row['ema21']:
        score += 1
        reasons.append('EMA bearish')
    if row['close'] < row['ema9']:
        score += 1
        reasons.append('Price < EMA9')
    
    return score, reasons

# ============== TRADE SIGNAL GENERATION ==============

def generate_signal(row, nifty_change, config=StrategyConfig):
    """
    Generate trading signal for a single candle
    Returns: (trade_type, score, reasons, sl, target) or None
    """
    # Check time window
    if row['time'] < config.ENTRY_START or row['time'] > config.ENTRY_END:
        return None
    
    # Score both directions
    buy_score, buy_reasons = score_buy_trade(row, nifty_change)
    short_score, short_reasons = score_short_trade(row, nifty_change)
    
    # Determine trade type
    trade_type = None
    score = 0
    reasons = []
    
    if buy_score >= config.MIN_SCORE and buy_score > short_score:
        trade_type = 'BUY'
        score = buy_score
        reasons = buy_reasons
    elif short_score >= config.MIN_SCORE and short_score > buy_score:
        trade_type = 'SHORT'
        score = short_score
        reasons = short_reasons
    
    if trade_type is None:
        return None
    
    # Calculate SL and Target
    entry_price = row['close']
    atr = row['atr'] if not pd.isna(row['atr']) else entry_price * 0.01
    sl_distance = atr * config.ATR_SL_MULTIPLIER
    
    if trade_type == 'BUY':
        sl_price = entry_price - sl_distance
        target_price = entry_price + (sl_distance * config.RISK_REWARD_RATIO)
    else:
        sl_price = entry_price + sl_distance
        target_price = entry_price - (sl_distance * config.RISK_REWARD_RATIO)
    
    return {
        'type': trade_type,
        'score': score,
        'reasons': reasons,
        'entry': round(entry_price, 2),
        'sl': round(sl_price, 2),
        'target': round(target_price, 2),
        'atr': round(atr, 2),
        'rsi': round(row['rsi'], 1),
        'nifty_change': round(nifty_change, 2)
    }

# ============== ZERODHA CHARGES ==============

def calculate_fno_charges(buy_value, sell_value):
    """Calculate Zerodha F&O charges"""
    brokerage = min(20, buy_value * 0.0003) + min(20, sell_value * 0.0003)
    stt = sell_value * 0.000125
    exchange = (buy_value + sell_value) * 0.0000173
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00002
    return round(brokerage + stt + exchange + sebi + gst + stamp, 2)

def round_to_tick(price, tick_size=0.05):
    """Round price to valid tick size"""
    return round(round(price / tick_size) * tick_size, 2)

# ============== MAIN SCANNER ==============

def scan_for_signals(kite, stocks=None, top_n=10):
    """
    Scan stocks for trading signals
    Returns list of trade opportunities sorted by score
    """
    if stocks is None:
        stocks = NIFTY_100
    
    config = StrategyConfig()
    signals = []
    
    # Get Nifty data for alignment
    nifty_token = 256265
    to_date = datetime.now()
    from_date = to_date - timedelta(days=5)
    
    nifty_data = kite.historical_data(
        instrument_token=nifty_token,
        from_date=from_date.strftime('%Y-%m-%d'),
        to_date=to_date.strftime('%Y-%m-%d'),
        interval=config.INTERVAL
    )
    
    nifty_df = pd.DataFrame(nifty_data)
    nifty_df['date'] = pd.to_datetime(nifty_df['date'])
    nifty_df['date_only'] = nifty_df['date'].dt.date
    
    today = datetime.now(IST).date()
    nifty_today = nifty_df[nifty_df['date_only'] == today]
    
    if len(nifty_today) == 0:
        print("No Nifty data for today yet")
        return []
    
    nifty_open = nifty_today.iloc[0]['open']
    nifty_current = nifty_today.iloc[-1]['close']
    nifty_change = ((nifty_current - nifty_open) / nifty_open) * 100
    
    print(f"Nifty: {nifty_current:.0f} ({nifty_change:+.2f}%)")
    print()
    
    # Get instruments
    instruments = kite.instruments('NSE')
    nfo_instruments = kite.instruments('NFO')
    
    token_map = {}
    for symbol in stocks:
        for inst in instruments:
            if inst['tradingsymbol'] == symbol and inst['segment'] == 'NSE':
                token_map[symbol] = {'token': inst['instrument_token'], 'lot_size': 1}
                break
    
    for symbol in stocks:
        for inst in nfo_instruments:
            if inst['tradingsymbol'].startswith(symbol) and 'FUT' in inst['instrument_type']:
                if symbol in token_map:
                    token_map[symbol]['lot_size'] = inst.get('lot_size', 1)
                break
    
    # Scan each stock
    for symbol in stocks:
        if symbol not in token_map:
            continue
        
        try:
            data = kite.historical_data(
                instrument_token=token_map[symbol]['token'],
                from_date=from_date.strftime('%Y-%m-%d'),
                to_date=to_date.strftime('%Y-%m-%d'),
                interval=config.INTERVAL
            )
            
            if not data or len(data) < 50:
                continue
            
            df = pd.DataFrame(data)
            df = prepare_data(df)
            
            # Get latest candle
            latest = df.iloc[-1]
            
            # Generate signal
            signal = generate_signal(latest, nifty_change, config)
            
            if signal:
                signal['symbol'] = symbol
                signal['lot_size'] = token_map[symbol]['lot_size']
                signal['ltp'] = latest['close']
                signal['time'] = latest['date']
                
                # Calculate position value and charges
                position_value = signal['entry'] * signal['lot_size']
                signal['position_value'] = round(position_value, 0)
                signal['est_charges'] = calculate_fno_charges(position_value, position_value)
                
                signals.append(signal)
                
        except Exception as e:
            continue
    
    # Sort by score (descending)
    signals.sort(key=lambda x: x['score'], reverse=True)
    
    return signals[:top_n]

# ============== DISPLAY FUNCTIONS ==============

def print_signals(signals):
    """Pretty print trading signals"""
    if not signals:
        print("No trading signals found matching criteria")
        return
    
    print("="*100)
    print("TRADING SIGNALS (Score >= 7, Entry 11AM-2:45PM)")
    print("="*100)
    print()
    
    for i, sig in enumerate(signals, 1):
        emoji = "🟢" if sig['type'] == 'BUY' else "🔴"
        print(f"{i}. {emoji} {sig['symbol']} - {sig['type']} (Score: {sig['score']}/10)")
        print(f"   Entry: ₹{sig['entry']} | SL: ₹{sig['sl']} | Target: ₹{sig['target']}")
        print(f"   Lot Size: {sig['lot_size']} | Position: ₹{sig['position_value']:,.0f}")
        print(f"   RSI: {sig['rsi']} | Nifty: {sig['nifty_change']:+.2f}%")
        print(f"   Reasons: {', '.join(sig['reasons'])}")
        print(f"   Est. Charges: ₹{sig['est_charges']}")
        print()

# ============== ENTRY POINT ==============

if __name__ == "__main__":
    # Load credentials
    with open('.kite_creds.json') as f:
        creds = json.load(f)
    with open('.kite_session.json') as f:
        session = json.load(f)
    
    kite = KiteConnect(api_key=creds['api_key'])
    kite.set_access_token(session['access_token'])
    
    # Check if market is open
    now = datetime.now(IST)
    print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} IST")
    print()
    
    # Scan for signals
    signals = scan_for_signals(kite, top_n=10)
    print_signals(signals)
