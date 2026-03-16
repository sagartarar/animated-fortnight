"""
OPTIMIZED STRATEGY BACKTEST
Based on data-driven insights from 4,005 trades analysis

KEY FINDINGS APPLIED:
1. Avoid 10:00 AM hour (-₹4.4L loss) - Wait till 11:00 AM
2. BUY only when RSI 30-50 (+₹7.2L combined)
3. Avoid SHORT trades in general (-₹1.4L loss) OR only with strict criteria
4. Score >= 8 for better win rate
5. Nifty alignment is critical for BUY
6. TIME_EXIT and EOD trades are profitable - let winners run
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from kiteconnect import KiteConnect
import pytz
import warnings
warnings.filterwarnings('ignore')

IST = pytz.timezone('Asia/Kolkata')

# Load Kite credentials
with open('.kite_creds.json') as f:
    creds = json.load(f)
with open('.kite_session.json') as f:
    session = json.load(f)

kite = KiteConnect(api_key=creds['api_key'])
kite.set_access_token(session['access_token'])

# ============== ZERODHA F&O CHARGES ==============

def calculate_fno_charges(buy_value, sell_value):
    brokerage = min(20, buy_value * 0.0003) + min(20, sell_value * 0.0003)
    stt = sell_value * 0.000125
    exchange = (buy_value + sell_value) * 0.0000173
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00002
    return brokerage + stt + exchange + sebi + gst + stamp

# ============== INDICATORS ==============

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_vwap(df):
    df = df.copy()
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_volume'] = df['typical_price'] * df['volume']
    df['date_only'] = df['date'].dt.date
    df['cum_tp_vol'] = df.groupby('date_only')['tp_volume'].cumsum()
    df['cum_vol'] = df.groupby('date_only')['volume'].cumsum()
    return df['cum_tp_vol'] / df['cum_vol']

def calculate_supertrend(df, period=10, multiplier=3):
    df = df.copy()
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
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

# ============== OPTIMIZED SCORING ==============

def score_buy_trade(row, nifty_change):
    """
    OPTIMIZED BUY SCORING based on backtest analysis
    - Only BUY (SHORT has negative expectancy)
    - RSI 30-50 is the sweet spot
    - Nifty alignment critical
    """
    score = 0
    reasons = []
    
    # MANDATORY: Nifty must be positive (BUY + Nifty UP = +₹4L)
    if nifty_change <= 0:
        return 0, []  # No BUY in falling market
    
    # Nifty strength (0-3)
    if nifty_change > 0.5:
        score += 3
        reasons.append('Nifty very strong')
    elif nifty_change > 0.3:
        score += 2
        reasons.append('Nifty strong')
    elif nifty_change > 0.1:
        score += 1
        reasons.append('Nifty positive')
    
    # RSI - CRITICAL (30-50 zone is most profitable)
    if 30 <= row['rsi'] <= 40:
        score += 3  # Best zone: +₹4.5L
        reasons.append('RSI sweet spot')
    elif 40 < row['rsi'] <= 50:
        score += 2  # Good zone: +₹2.7L
        reasons.append('RSI good')
    elif 25 <= row['rsi'] < 30:
        score += 1  # Oversold bounce potential
        reasons.append('RSI oversold')
    else:
        return 0, []  # Avoid RSI > 50 or < 25
    
    # VWAP (0-2)
    if row['close'] > row['vwap'] * 1.002:
        score += 2
        reasons.append('Strong above VWAP')
    elif row['close'] > row['vwap']:
        score += 1
        reasons.append('Above VWAP')
    
    # Supertrend (0-2)
    if row['supertrend'] == 1:
        score += 2
        reasons.append('Supertrend bullish')
    
    # EMA alignment (0-2)
    if row['ema9'] > row['ema21'] and row['close'] > row['ema9']:
        score += 2
        reasons.append('EMA bullish stack')
    elif row['ema9'] > row['ema21']:
        score += 1
        reasons.append('EMA bullish')
    
    # Volume confirmation (0-1)
    if row.get('volume_ratio', 1) > 1.2:
        score += 1
        reasons.append('High volume')
    
    return score, reasons

def score_short_trade(row, nifty_change):
    """
    VERY STRICT SHORT SCORING
    SHORT trades overall lost money, so be very selective
    """
    score = 0
    reasons = []
    
    # MANDATORY: Nifty must be strongly negative
    if nifty_change >= -0.3:
        return 0, []  # No SHORT unless market clearly falling
    
    # Nifty weakness (0-3)
    if nifty_change < -0.7:
        score += 3
        reasons.append('Nifty crashing')
    elif nifty_change < -0.5:
        score += 2
        reasons.append('Nifty weak')
    else:
        score += 1
        reasons.append('Nifty negative')
    
    # RSI - must be overbought
    if 65 <= row['rsi'] <= 75:
        score += 2
        reasons.append('RSI overbought')
    elif 75 < row['rsi'] <= 80:
        score += 3
        reasons.append('RSI extreme')
    else:
        return 0, []  # Only short overbought stocks
    
    # Below VWAP
    if row['close'] < row['vwap'] * 0.998:
        score += 2
        reasons.append('Weak below VWAP')
    elif row['close'] < row['vwap']:
        score += 1
    
    # Supertrend bearish
    if row['supertrend'] == -1:
        score += 2
        reasons.append('Supertrend bearish')
    
    # EMA bearish
    if row['ema9'] < row['ema21'] and row['close'] < row['ema9']:
        score += 2
        reasons.append('EMA bearish stack')
    elif row['ema9'] < row['ema21']:
        score += 1
    
    return score, reasons

# ============== STOCKS ==============

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

# Stocks to AVOID (from previous backtest - consistently losing)
AVOID_STOCKS = ['BIOCON', 'TECHM', 'HINDUNILVR', 'DRREDDY', 'ONGC', 'INDUSTOWER', 'VEDL', 'AUROPHARMA', 'ASIANPAINT', 'MOTHERSON']

# Preferred stocks (consistently winning)
PREFER_STOCKS = ['PAYTM', 'CANBK', 'EICHERMOT', 'PNB', 'MARICO', 'TORNTPHARM', 'DLF', 'BAJFINANCE', 'PFC', 'PIDILITIND']

# ============== MAIN BACKTEST ==============

print("="*100)
print("OPTIMIZED STRATEGY BACKTEST")
print("="*100)
print("""
OPTIMIZATIONS APPLIED:
1. Entry after 11:00 AM (avoid -₹4.4L morning losses)
2. BUY only when RSI 30-50 (best performing zone)
3. BUY only when Nifty positive (aligned trades)
4. Score >= 8 required (higher quality)
5. Very strict SHORT criteria (only when Nifty < -0.3% and RSI > 65)
6. Avoid consistently losing stocks
7. Better R:R ratio (1:2 for BUY, tighter SL)
8. Max 1 trade per stock per day
""")

# Get instruments
instruments = kite.instruments('NSE')
nfo_instruments = kite.instruments('NFO')

token_map = {}
for symbol in NIFTY_100:
    if symbol in AVOID_STOCKS:
        continue
    for inst in instruments:
        if inst['tradingsymbol'] == symbol and inst['segment'] == 'NSE':
            token_map[symbol] = {'token': inst['instrument_token'], 'lot_size': 1}
            break

for symbol in NIFTY_100:
    for inst in nfo_instruments:
        if inst['tradingsymbol'].startswith(symbol) and 'FUT' in inst['instrument_type']:
            if symbol in token_map:
                token_map[symbol]['lot_size'] = inst.get('lot_size', 1)
            break

print(f"Stocks: {len(token_map)} (excluded {len(AVOID_STOCKS)} losers)")

# Get Nifty data
nifty_token = 256265
to_date = datetime.now()
from_date = to_date - timedelta(days=55)

nifty_data = kite.historical_data(
    instrument_token=nifty_token,
    from_date=from_date.strftime('%Y-%m-%d'),
    to_date=to_date.strftime('%Y-%m-%d'),
    interval='15minute'
)

nifty_df = pd.DataFrame(nifty_data)
nifty_df['date'] = pd.to_datetime(nifty_df['date'])
nifty_df['date_only'] = nifty_df['date'].dt.date

nifty_daily_open = nifty_df.groupby('date_only')['open'].first().to_dict()
nifty_df['nifty_day_open'] = nifty_df['date_only'].map(nifty_daily_open)
nifty_df['nifty_change'] = ((nifty_df['close'] - nifty_df['nifty_day_open']) / nifty_df['nifty_day_open']) * 100

nifty_lookup = nifty_df.set_index('date')[['nifty_change', 'close']].to_dict('index')

print(f"Period: {nifty_df['date'].min().date()} to {nifty_df['date'].max().date()}")
print()

# Parameters
MIN_BUY_SCORE = 8
MIN_SHORT_SCORE = 10  # Very strict for shorts
BUY_RR = 2.0  # 1:2 risk-reward for BUY
SHORT_RR = 1.5

all_trades = []

print("Running optimized backtest...")
for i, symbol in enumerate(NIFTY_100):
    if symbol not in token_map:
        continue
    
    is_preferred = symbol in PREFER_STOCKS
    print(f"\r  [{i+1}/{len(NIFTY_100)}] {symbol}{'*' if is_preferred else ''}...", end='', flush=True)
    
    try:
        data = kite.historical_data(
            instrument_token=token_map[symbol]['token'],
            from_date=from_date.strftime('%Y-%m-%d'),
            to_date=to_date.strftime('%Y-%m-%d'),
            interval='15minute'
        )
        
        if not data or len(data) < 100:
            continue
        
        df = pd.DataFrame(data)
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
        df['nifty_change'] = df['date'].map(lambda x: nifty_lookup.get(x, {}).get('nifty_change', 0))
        
        # OPTIMIZED: Entry from 11:00 AM to 2:30 PM
        entry_start = time(11, 0)  # Changed from 10:15
        entry_end = time(14, 30)   # Slightly earlier
        
        daily_trades = {}
        lot_size = token_map[symbol].get('lot_size', 1)
        
        for idx in range(50, len(df) - 8):
            row = df.iloc[idx]
            
            if row['time'] < entry_start or row['time'] > entry_end:
                continue
            
            date_key = row['date_only']
            if daily_trades.get(date_key, 0) >= 1:  # Max 1 per day
                continue
            
            if pd.isna(row['rsi']) or pd.isna(row['vwap']) or pd.isna(row['nifty_change']):
                continue
            
            # Score trades
            buy_score, buy_reasons = score_buy_trade(row, row['nifty_change'])
            short_score, short_reasons = score_short_trade(row, row['nifty_change'])
            
            # Bonus for preferred stocks
            if is_preferred:
                buy_score += 1
                short_score += 1
            
            trade_type = None
            score = 0
            reasons = []
            rr_ratio = 1.5
            
            if buy_score >= MIN_BUY_SCORE:
                trade_type = 'BUY'
                score = buy_score
                reasons = buy_reasons
                rr_ratio = BUY_RR
            elif short_score >= MIN_SHORT_SCORE:
                trade_type = 'SHORT'
                score = short_score
                reasons = short_reasons
                rr_ratio = SHORT_RR
            
            if trade_type is None:
                continue
            
            entry_price = row['close']
            entry_time = row['date']
            atr = row['atr'] if not pd.isna(row['atr']) else entry_price * 0.01
            
            # Tighter SL for better R:R
            sl_distance = atr * 1.2  # Tighter than before
            
            if trade_type == 'BUY':
                sl_price = entry_price - sl_distance
                target_price = entry_price + (sl_distance * rr_ratio)
            else:
                sl_price = entry_price + sl_distance
                target_price = entry_price - (sl_distance * rr_ratio)
            
            # Simulate exit
            exit_price = None
            exit_time = None
            exit_reason = None
            
            for j in range(idx + 1, min(idx + 16, len(df))):
                future = df.iloc[j]
                
                if future['time'] >= time(15, 15):
                    exit_price = future['close']
                    exit_time = future['date']
                    exit_reason = 'TIME_EXIT'
                    break
                
                if trade_type == 'BUY':
                    if future['low'] <= sl_price:
                        exit_price = sl_price
                        exit_time = future['date']
                        exit_reason = 'SL'
                        break
                    if future['high'] >= target_price:
                        exit_price = target_price
                        exit_time = future['date']
                        exit_reason = 'TARGET'
                        break
                else:
                    if future['high'] >= sl_price:
                        exit_price = sl_price
                        exit_time = future['date']
                        exit_reason = 'SL'
                        break
                    if future['low'] <= target_price:
                        exit_price = target_price
                        exit_time = future['date']
                        exit_reason = 'TARGET'
                        break
            
            if exit_price is None:
                last_candle = df.iloc[min(idx + 15, len(df) - 1)]
                exit_price = last_candle['close']
                exit_time = last_candle['date']
                exit_reason = 'EOD'
            
            # Calculate P&L
            if trade_type == 'BUY':
                gross_pnl = (exit_price - entry_price) * lot_size
            else:
                gross_pnl = (entry_price - exit_price) * lot_size
            
            buy_val = entry_price * lot_size if trade_type == 'BUY' else exit_price * lot_size
            sell_val = exit_price * lot_size if trade_type == 'BUY' else entry_price * lot_size
            charges = calculate_fno_charges(buy_val, sell_val)
            
            net_pnl = gross_pnl - charges
            
            trade = {
                'date': date_key,
                'symbol': symbol,
                'preferred': is_preferred,
                'type': trade_type,
                'score': score,
                'entry_time': entry_time,
                'entry_price': round(entry_price, 2),
                'qty': lot_size,
                'sl_price': round(sl_price, 2),
                'target_price': round(target_price, 2),
                'exit_time': exit_time,
                'exit_price': round(exit_price, 2),
                'exit_reason': exit_reason,
                'gross_pnl': round(gross_pnl, 2),
                'charges': round(charges, 2),
                'net_pnl': round(net_pnl, 2),
                'nifty_change': round(row['nifty_change'], 2),
                'rsi': round(row['rsi'], 1),
                'reasons': ', '.join(reasons[:3])
            }
            
            all_trades.append(trade)
            daily_trades[date_key] = daily_trades.get(date_key, 0) + 1
            
    except Exception as e:
        continue

print(f"\n\nTotal trades: {len(all_trades)}")

if all_trades:
    df_trades = pd.DataFrame(all_trades)
    
    total = len(df_trades)
    winners = len(df_trades[df_trades['net_pnl'] > 0])
    win_rate = winners / total * 100
    
    gross = df_trades['gross_pnl'].sum()
    charges = df_trades['charges'].sum()
    net = df_trades['net_pnl'].sum()
    
    print()
    print("="*100)
    print("OPTIMIZED STRATEGY RESULTS")
    print("="*100)
    print(f"""
Period: {df_trades['date'].min()} to {df_trades['date'].max()}
Trading Days: {df_trades['date'].nunique()}

Total Trades: {total}
Winners: {winners} | Losers: {total - winners}
Win Rate: {win_rate:.1f}%

Gross P&L: ₹{gross:,.0f}
Charges: ₹{charges:,.0f}
NET P&L: ₹{net:,.0f}

Avg Winner: ₹{df_trades[df_trades['net_pnl'] > 0]['net_pnl'].mean():,.0f}
Avg Loser: ₹{df_trades[df_trades['net_pnl'] <= 0]['net_pnl'].mean():,.0f}
Avg Trade: ₹{df_trades['net_pnl'].mean():,.0f}
""")
    
    # Compare with original
    print("="*100)
    print("COMPARISON WITH ORIGINAL STRATEGY")
    print("="*100)
    
    original_trades = 4005
    original_win_rate = 46.6
    original_gross = 902324
    original_charges = 692678
    original_net = 209646
    
    print(f"""
                    ORIGINAL        OPTIMIZED       CHANGE
Trades:             {original_trades:>8}        {total:>8}        {total - original_trades:>+8}
Win Rate:           {original_win_rate:>7.1f}%        {win_rate:>7.1f}%        {win_rate - original_win_rate:>+7.1f}%
Gross P&L:        ₹{original_gross:>9,}      ₹{gross:>9,.0f}      ₹{gross - original_gross:>+9,.0f}
Charges:          ₹{original_charges:>9,}      ₹{charges:>9,.0f}      ₹{charges - original_charges:>+9,.0f}
NET P&L:          ₹{original_net:>9,}      ₹{net:>9,.0f}      ₹{net - original_net:>+9,.0f}

Profit per Trade: ₹{original_net/original_trades:>9,.0f}      ₹{net/total:>9,.0f}      ₹{net/total - original_net/original_trades:>+9,.0f}
""")
    
    # By exit reason
    print("BY EXIT REASON:")
    for reason in ['TARGET', 'SL', 'TIME_EXIT', 'EOD']:
        subset = df_trades[df_trades['exit_reason'] == reason]
        if len(subset) > 0:
            wr = len(subset[subset['net_pnl'] > 0]) / len(subset) * 100
            print(f"  {reason:>10}: {len(subset):>4} trades | WR: {wr:>5.1f}% | Net: ₹{subset['net_pnl'].sum():>10,.0f}")
    
    # Export
    df_export = df_trades.copy()
    df_export['entry_time'] = df_export['entry_time'].astype(str)
    df_export['exit_time'] = df_export['exit_time'].astype(str)
    df_export['date'] = df_export['date'].astype(str)
    
    with pd.ExcelWriter('backtest_optimized_strategy.xlsx', engine='openpyxl') as writer:
        df_export.to_excel(writer, sheet_name='All Trades', index=False)
        
        daily = df_export.groupby('date').agg({
            'net_pnl': 'sum',
            'gross_pnl': 'sum',
            'charges': 'sum',
            'symbol': 'count'
        }).rename(columns={'symbol': 'num_trades'}).reset_index()
        daily['cumulative_pnl'] = daily['net_pnl'].cumsum()
        daily.to_excel(writer, sheet_name='Daily Summary', index=False)
    
    print()
    print(f"✅ Results saved to: backtest_optimized_strategy.xlsx")

else:
    print("No trades with optimized criteria!")
