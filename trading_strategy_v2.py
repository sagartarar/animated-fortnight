"""
RESEARCH-OPTIMIZED INTRADAY TRADING STRATEGY V2
=================================================
Based on 10-Year Backtest (Feb 2015 - Feb 2026):
- 123,866 trades across 196 stocks
- 49.2% win rate
- ₹61.5L net profit
- Profit Factor: 1.07

KEY FINDINGS FROM RESEARCH:
1. SHORT-ONLY is profitable; BUY strategies lose money
2. ADX filter (>25) identifies trending markets
3. Time exit outperforms SL for this strategy
4. Entry window: 11:00 AM - 2:00 PM optimal
5. Score >= 7 gives best total P&L
6. Trailing stops HURT performance (avoid)
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
    # Entry timing (research-optimized)
    ENTRY_START = time(11, 0)   # 11:00 AM
    ENTRY_END = time(14, 0)     # 2:00 PM (earlier cutoff for time exit)
    EXIT_TIME = time(15, 15)    # 3:15 PM mandatory exit
    
    # Trade direction (CRITICAL: SHORT-ONLY)
    TRADE_DIRECTION = 'SHORT_ONLY'  # Options: 'SHORT_ONLY', 'BUY_ONLY', 'BOTH'
    
    # Scoring
    MIN_SCORE = 7  # Research: 7 gives best total P&L
    
    # ADX filter (market regime)
    USE_ADX_FILTER = True
    MIN_ADX = 25.0   # Only trade when ADX > 25 (trending)
    MAX_ADX = 50.0   # Avoid exhaustion moves
    
    # RSI sweet spots (research-based)
    USE_RSI_FILTER = False  # Optional - reduces trades but improves WR
    RSI_SHORT_MIN = 45.0
    RSI_SHORT_MAX = 70.0
    RSI_BUY_MIN = 40.0
    RSI_BUY_MAX = 65.0
    
    # Risk management
    USE_SL = False  # Research: time exit > SL for this strategy
    ATR_SL_MULTIPLIER = 2.0  # If using SL, wider is better
    RISK_REWARD_RATIO = 1.5
    MAX_TRADES_PER_DAY = 1  # Per stock
    
    # Position sizing
    RISK_PER_TRADE = 2000  # ₹2000 risk per trade
    
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

def calculate_adx(df, period=14):
    """Calculate ADX (Average Directional Index) - CRITICAL FILTER"""
    df = df.copy()
    
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff().multiply(-1)
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr = calculate_atr(df, 1) * period  # TR
    
    atr = calculate_atr(df, period)
    
    plus_di = 100 * (plus_dm.rolling(period).sum() / (atr * period))
    minus_di = 100 * (minus_dm.rolling(period).sum() / (atr * period))
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    
    return adx, plus_di, minus_di

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
    df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df)
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    return df

# ============== SCORING SYSTEM ==============

def score_short_trade(row, nifty_change, config=StrategyConfig):
    """
    Score a potential SHORT trade (0-12 scale)
    Based on research-optimized criteria
    """
    score = 0
    reasons = []
    
    # MUST have negative market direction
    if nifty_change >= 0:
        return 0, ["Market not bearish"]
    
    # ADX filter - only trade in trending markets
    if config.USE_ADX_FILTER:
        if pd.isna(row['adx']) or row['adx'] < config.MIN_ADX:
            return 0, [f"ADX too low ({row['adx']:.1f})"]
        if row['adx'] > config.MAX_ADX:
            return 0, [f"ADX exhaustion ({row['adx']:.1f})"]
        # Bonus for -DI > +DI (bearish directional strength)
        if row['minus_di'] > row['plus_di']:
            score += 1
            reasons.append('-DI dominates')
    
    # RSI filter
    if config.USE_RSI_FILTER:
        if pd.isna(row['rsi']) or row['rsi'] < config.RSI_SHORT_MIN or row['rsi'] > config.RSI_SHORT_MAX:
            return 0, [f"RSI out of range ({row['rsi']:.1f})"]
    
    # Nifty alignment (0-3)
    if nifty_change < -0.5:
        score += 3
        reasons.append('Nifty very weak')
    elif nifty_change < -0.3:
        score += 2
        reasons.append('Nifty weak')
    elif nifty_change < -0.1:
        score += 1
        reasons.append('Nifty negative')
    
    # RSI zone (0-2)
    if not pd.isna(row['rsi']):
        if 60 <= row['rsi'] <= 75:
            score += 2
            reasons.append('RSI optimal short')
        elif row['rsi'] > 75:
            score += 1
            reasons.append('RSI overbought')
    
    # VWAP (0-1)
    if not pd.isna(row['vwap']) and row['close'] < row['vwap']:
        score += 1
        reasons.append('Below VWAP')
    
    # Supertrend (0-2)
    if row['supertrend'] == -1:
        score += 2
        reasons.append('Supertrend bearish')
    
    # EMA alignment (0-2)
    if not pd.isna(row['ema9']) and not pd.isna(row['ema21']):
        if row['ema9'] < row['ema21']:
            score += 1
            reasons.append('EMA bearish')
        if row['close'] < row['ema9']:
            score += 1
            reasons.append('Price < EMA9')
    
    return score, reasons

def score_buy_trade(row, nifty_change, config=StrategyConfig):
    """
    Score a potential BUY trade (0-12 scale)
    NOTE: Research shows BUY trades are NOT profitable - use with caution
    """
    score = 0
    reasons = []
    
    # MUST have positive market direction
    if nifty_change <= 0:
        return 0, ["Market not bullish"]
    
    # ADX filter
    if config.USE_ADX_FILTER:
        if pd.isna(row['adx']) or row['adx'] < config.MIN_ADX:
            return 0, [f"ADX too low ({row['adx']:.1f})"]
        if row['adx'] > config.MAX_ADX:
            return 0, [f"ADX exhaustion ({row['adx']:.1f})"]
        if row['plus_di'] > row['minus_di']:
            score += 1
            reasons.append('+DI dominates')
    
    # RSI filter
    if config.USE_RSI_FILTER:
        if pd.isna(row['rsi']) or row['rsi'] < config.RSI_BUY_MIN or row['rsi'] > config.RSI_BUY_MAX:
            return 0, [f"RSI out of range ({row['rsi']:.1f})"]
    
    # Nifty alignment (0-3)
    if nifty_change > 0.5:
        score += 3
        reasons.append('Nifty very strong')
    elif nifty_change > 0.3:
        score += 2
        reasons.append('Nifty strong')
    elif nifty_change > 0.1:
        score += 1
        reasons.append('Nifty positive')
    
    # RSI zone (0-2)
    if not pd.isna(row['rsi']):
        if 35 <= row['rsi'] <= 50:
            score += 2
            reasons.append('RSI optimal')
        elif 30 <= row['rsi'] < 35:
            score += 1
            reasons.append('RSI oversold')
    
    # VWAP (0-1)
    if not pd.isna(row['vwap']) and row['close'] > row['vwap']:
        score += 1
        reasons.append('Above VWAP')
    
    # Supertrend (0-2)
    if row['supertrend'] == 1:
        score += 2
        reasons.append('Supertrend bullish')
    
    # EMA alignment (0-2)
    if not pd.isna(row['ema9']) and not pd.isna(row['ema21']):
        if row['ema9'] > row['ema21']:
            score += 1
            reasons.append('EMA bullish')
        if row['close'] > row['ema9']:
            score += 1
            reasons.append('Price > EMA9')
    
    return score, reasons

# ============== TRADE SIGNAL GENERATION ==============

def generate_signal(row, nifty_change, config=StrategyConfig):
    """
    Generate trading signal for a single candle
    Returns: signal dict or None
    """
    # Check time window
    if row['time'] < config.ENTRY_START or row['time'] > config.ENTRY_END:
        return None
    
    trade_type = None
    score = 0
    reasons = []
    
    # Based on configured direction
    if config.TRADE_DIRECTION == 'SHORT_ONLY':
        short_score, short_reasons = score_short_trade(row, nifty_change, config)
        if short_score >= config.MIN_SCORE:
            trade_type = 'SHORT'
            score = short_score
            reasons = short_reasons
    
    elif config.TRADE_DIRECTION == 'BUY_ONLY':
        buy_score, buy_reasons = score_buy_trade(row, nifty_change, config)
        if buy_score >= config.MIN_SCORE:
            trade_type = 'BUY'
            score = buy_score
            reasons = buy_reasons
    
    else:  # BOTH
        buy_score, buy_reasons = score_buy_trade(row, nifty_change, config)
        short_score, short_reasons = score_short_trade(row, nifty_change, config)
        
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
        'sl': round(sl_price, 2) if config.USE_SL else None,
        'target': round(target_price, 2) if config.USE_SL else None,
        'exit_strategy': 'TIME_EXIT (3:15 PM)' if not config.USE_SL else 'SL/TGT or TIME',
        'atr': round(atr, 2),
        'adx': round(row['adx'], 1) if not pd.isna(row['adx']) else None,
        'rsi': round(row['rsi'], 1) if not pd.isna(row['rsi']) else None,
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
    Scan stocks for trading signals (SHORT-only by default)
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
    print(f"Strategy Mode: {config.TRADE_DIRECTION}")
    
    # For SHORT-only, need negative Nifty
    if config.TRADE_DIRECTION == 'SHORT_ONLY' and nifty_change >= 0:
        print("\n⚠️  Nifty is positive - no SHORT signals possible")
        print("   SHORT-only strategy requires negative market")
        return []
    
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
    
    config = StrategyConfig()
    
    print("="*100)
    print(f"TRADING SIGNALS ({config.TRADE_DIRECTION}, Score >= {config.MIN_SCORE})")
    print(f"Entry: {config.ENTRY_START} - {config.ENTRY_END}")
    print(f"Exit: {config.EXIT_TIME} (Time-based)" if not config.USE_SL else f"Exit: SL/TGT or {config.EXIT_TIME}")
    print("="*100)
    print()
    
    for i, sig in enumerate(signals, 1):
        emoji = "🟢" if sig['type'] == 'BUY' else "🔴"
        print(f"{i}. {emoji} {sig['symbol']} - {sig['type']} (Score: {sig['score']}/12)")
        print(f"   Entry: ₹{sig['entry']}")
        if sig['sl']:
            print(f"   SL: ₹{sig['sl']} | Target: ₹{sig['target']}")
        else:
            print(f"   Exit: TIME-BASED at 3:15 PM")
        print(f"   Lot Size: {sig['lot_size']} | Position: ₹{sig['position_value']:,.0f}")
        print(f"   ADX: {sig['adx']} | RSI: {sig['rsi']} | Nifty: {sig['nifty_change']:+.2f}%")
        print(f"   Reasons: {', '.join(sig['reasons'])}")
        print(f"   Est. Charges: ₹{sig['est_charges']}")
        print()

def print_strategy_info():
    """Print strategy configuration"""
    config = StrategyConfig()
    
    print("="*60)
    print("RESEARCH-OPTIMIZED STRATEGY V2")
    print("="*60)
    print()
    print("10-YEAR BACKTEST RESULTS:")
    print("├─ Total Trades: 123,866")
    print("├─ Win Rate: 49.2%")
    print("├─ Net P&L: ₹61.5L")
    print("├─ Profit Factor: 1.07")
    print("└─ Max Drawdown: ₹24.3L")
    print()
    print("CONFIGURATION:")
    print(f"├─ Direction: {config.TRADE_DIRECTION}")
    print(f"├─ Entry: {config.ENTRY_START} - {config.ENTRY_END}")
    print(f"├─ Exit: {config.EXIT_TIME}")
    print(f"├─ Min Score: {config.MIN_SCORE}")
    print(f"├─ ADX Filter: {config.MIN_ADX}-{config.MAX_ADX}" if config.USE_ADX_FILTER else "├─ ADX Filter: OFF")
    print(f"├─ Stop Loss: {'ON' if config.USE_SL else 'OFF (time exit)'}")
    print(f"└─ Max Trades/Day: {config.MAX_TRADES_PER_DAY}")
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
    
    # Print strategy info
    print_strategy_info()
    
    # Check if market is open
    now = datetime.now(IST)
    print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} IST")
    print()
    
    # Scan for signals
    signals = scan_for_signals(kite, top_n=10)
    print_signals(signals)
