"""
OPTIMIZED Intraday Backtest with 15-minute data
- Higher score threshold (9+) 
- Fewer trades = lower charges
- Better risk-reward (2:1)
- Only trade stocks with good win rates
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

# ============== ZERODHA CHARGES ==============

def calculate_charges(buy_value, sell_value):
    """Calculate exact Zerodha intraday charges"""
    total_turnover = buy_value + sell_value
    
    brokerage_buy = min(20, buy_value * 0.0003)
    brokerage_sell = min(20, sell_value * 0.0003)
    brokerage = brokerage_buy + brokerage_sell
    
    stt = sell_value * 0.00025
    exchange = total_turnover * 0.0000307
    sebi = total_turnover * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00015
    
    total_charges = brokerage + stt + exchange + sebi + gst + stamp
    
    return {
        'brokerage': round(brokerage, 2),
        'stt': round(stt, 2),
        'exchange': round(exchange, 2),
        'sebi': round(sebi, 2),
        'gst': round(gst, 2),
        'stamp': round(stamp, 2),
        'total_charges': round(total_charges, 2)
    }

# ============== INDICATORS ==============

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_vwap(df):
    df = df.copy()
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_volume'] = df['typical_price'] * df['volume']
    df['date_only'] = df['date'].dt.date
    df['cum_tp_vol'] = df.groupby('date_only')['tp_volume'].cumsum()
    df['cum_vol'] = df.groupby('date_only')['volume'].cumsum()
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
    return df['vwap']

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

# ============== OPTIMIZED SCORING (STRICTER) ==============

def score_trade(row, nifty_change, trade_type):
    score = 0
    reasons = []
    
    if trade_type == 'BUY':
        # Nifty alignment (0-3) - MANDATORY positive for BUY
        if nifty_change > 0.5:
            score += 3
            reasons.append('Nifty strong trend')
        elif nifty_change > 0.3:
            score += 2
            reasons.append('Nifty bullish')
        elif nifty_change > 0.1:
            score += 1
            reasons.append('Nifty positive')
        else:
            return 0, []  # NO BUY without Nifty support
        
        # RSI (0-2) - Optimal zone only
        if 40 <= row['rsi'] <= 55:
            score += 2
            reasons.append('RSI optimal')
        elif 35 <= row['rsi'] < 40:
            score += 1
        
        # VWAP (0-2)
        if row['close'] > row['vwap'] * 1.002:  # Needs clear break
            score += 2
            reasons.append('Above VWAP')
        elif row['close'] > row['vwap']:
            score += 1
        
        # Supertrend (0-2)
        if row['supertrend'] == 1:
            score += 2
            reasons.append('Supertrend BUY')
        
        # EMA (0-2)
        if row['ema9'] > row['ema21']:
            score += 1
            reasons.append('EMA bullish')
        if row['close'] > row['ema9']:
            score += 1
            
        # Momentum (0-1)
        if row.get('momentum', 0) > 0.3:
            score += 1
            reasons.append('Strong momentum')
            
    else:  # SHORT
        # Nifty alignment - MANDATORY negative for SHORT
        if nifty_change < -0.5:
            score += 3
            reasons.append('Nifty weak trend')
        elif nifty_change < -0.3:
            score += 2
            reasons.append('Nifty bearish')
        elif nifty_change < -0.1:
            score += 1
            reasons.append('Nifty negative')
        else:
            return 0, []  # NO SHORT without Nifty weakness
        
        if 55 <= row['rsi'] <= 70:
            score += 2
            reasons.append('RSI optimal')
        elif 70 < row['rsi'] <= 75:
            score += 1
        
        if row['close'] < row['vwap'] * 0.998:
            score += 2
            reasons.append('Below VWAP')
        elif row['close'] < row['vwap']:
            score += 1
        
        if row['supertrend'] == -1:
            score += 2
            reasons.append('Supertrend SELL')
        
        if row['ema9'] < row['ema21']:
            score += 1
            reasons.append('EMA bearish')
        if row['close'] < row['ema9']:
            score += 1
            
        if row.get('momentum', 0) < -0.3:
            score += 1
            reasons.append('Strong down momentum')
    
    return score, reasons

# ============== STOCKS WITH GOOD WIN RATES (from previous backtest) ==============

STOCKS = [
    'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 
    'AXISBANK', 'BAJFINANCE', 'SUNPHARMA', 'WIPRO', 'TATASTEEL',
    'POWERGRID', 'NTPC', 'JSWSTEEL', 'TECHM', 'INDUSINDBK',
    'HINDALCO', 'CIPLA', 'BANKBARODA', 'PNB', 'ULTRACEMCO'
]

# ============== MAIN BACKTEST ==============

print("="*80)
print("OPTIMIZED INTRADAY BACKTEST - 15 MIN TIMEFRAME")
print("Score >= 9 | 2:1 R:R | Strict Nifty Alignment | Max 1 trade/day")
print("="*80)
print()

# Get Nifty 15min data
print("Fetching Nifty 50 data (15min)...")
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

print(f"Nifty data: {len(nifty_df)} candles, {nifty_df['date_only'].nunique()} days")
print()

# Get instruments
instruments = kite.instruments('NSE')
token_map = {}
for symbol in STOCKS:
    for inst in instruments:
        if inst['tradingsymbol'] == symbol:
            token_map[symbol] = inst['instrument_token']
            break

print(f"Found {len(token_map)} stocks")
print()

CAPITAL = 100000
RISK_PER_TRADE = 2000
POSITION_SIZE = 200000
MIN_SCORE = 9  # Higher threshold
RR_RATIO = 2.0  # Better risk-reward

all_trades = []

print("Running backtest...")
for i, symbol in enumerate(STOCKS):
    if symbol not in token_map:
        continue
    
    print(f"\r  [{i+1}/{len(STOCKS)}] {symbol}...", end='', flush=True)
    
    try:
        data = kite.historical_data(
            instrument_token=token_map[symbol],
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
        df['momentum'] = df['close'].pct_change(4) * 100
        df['nifty_change'] = df['date'].map(lambda x: nifty_lookup.get(x, {}).get('nifty_change', 0))
        
        entry_start = time(10, 30)  # Later entry (avoid opening volatility)
        entry_end = time(14, 15)    # Earlier cutoff
        
        daily_trades = {}
        
        for idx in range(50, len(df) - 8):
            row = df.iloc[idx]
            
            if row['time'] < entry_start or row['time'] > entry_end:
                continue
            
            date_key = row['date_only']
            if daily_trades.get(date_key, 0) >= 1:  # Only 1 trade per day
                continue
            
            if pd.isna(row['rsi']) or pd.isna(row['vwap']) or pd.isna(row['nifty_change']):
                continue
            
            buy_score, buy_reasons = score_trade(row, row['nifty_change'], 'BUY')
            sell_score, sell_reasons = score_trade(row, row['nifty_change'], 'SHORT')
            
            trade_type = None
            score = 0
            reasons = []
            
            if buy_score >= MIN_SCORE and buy_score > sell_score:
                trade_type = 'BUY'
                score = buy_score
                reasons = buy_reasons
            elif sell_score >= MIN_SCORE and sell_score > buy_score:
                trade_type = 'SHORT'
                score = sell_score
                reasons = sell_reasons
            
            if trade_type is None:
                continue
            
            entry_price = row['close']
            entry_time = row['date']
            atr = row['atr'] if not pd.isna(row['atr']) else entry_price * 0.01
            
            sl_distance = atr * 1.0  # Tighter SL
            qty = int(RISK_PER_TRADE / sl_distance)
            qty = max(1, min(qty, int(POSITION_SIZE / entry_price)))
            
            if trade_type == 'BUY':
                sl_price = entry_price - sl_distance
                target_price = entry_price + (sl_distance * RR_RATIO)  # 2:1 R:R
            else:
                sl_price = entry_price + sl_distance
                target_price = entry_price - (sl_distance * RR_RATIO)
            
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
            
            if trade_type == 'BUY':
                gross_pnl = (exit_price - entry_price) * qty
            else:
                gross_pnl = (entry_price - exit_price) * qty
            
            buy_value = entry_price * qty if trade_type == 'BUY' else exit_price * qty
            sell_value = exit_price * qty if trade_type == 'BUY' else entry_price * qty
            charges = calculate_charges(buy_value, sell_value)
            
            net_pnl = gross_pnl - charges['total_charges']
            
            trade = {
                'date': date_key,
                'symbol': symbol,
                'type': trade_type,
                'score': score,
                'entry_time': entry_time,
                'entry_price': round(entry_price, 2),
                'qty': qty,
                'sl_price': round(sl_price, 2),
                'target_price': round(target_price, 2),
                'exit_time': exit_time,
                'exit_price': round(exit_price, 2),
                'exit_reason': exit_reason,
                'gross_pnl': round(gross_pnl, 2),
                'brokerage': charges['brokerage'],
                'stt': charges['stt'],
                'other_charges': round(charges['exchange'] + charges['sebi'] + charges['gst'] + charges['stamp'], 2),
                'total_charges': charges['total_charges'],
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
    
    print()
    print("="*80)
    print("OPTIMIZED BACKTEST RESULTS (WITH ZERODHA CHARGES)")
    print("="*80)
    
    total_trades = len(df_trades)
    winners = len(df_trades[df_trades['net_pnl'] > 0])
    losers = len(df_trades[df_trades['net_pnl'] <= 0])
    win_rate = winners / total_trades * 100 if total_trades > 0 else 0
    
    total_gross = df_trades['gross_pnl'].sum()
    total_charges = df_trades['total_charges'].sum()
    total_net = df_trades['net_pnl'].sum()
    
    avg_winner = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].mean() if winners > 0 else 0
    avg_loser = df_trades[df_trades['net_pnl'] <= 0]['net_pnl'].mean() if losers > 0 else 0
    
    # Calculate expectancy
    expectancy = (win_rate/100 * avg_winner) + ((100-win_rate)/100 * avg_loser)
    
    print(f"""
Period: {df_trades['date'].min()} to {df_trades['date'].max()}
Trading Days: {df_trades['date'].nunique()}

Total Trades: {total_trades}
Winners: {winners} | Losers: {losers}
Win Rate: {win_rate:.1f}%

Gross P&L: ₹{total_gross:,.2f}
Total Charges: ₹{total_charges:,.2f}

NET P&L: ₹{total_net:,.2f}

Avg Winner: ₹{avg_winner:,.2f}
Avg Loser: ₹{avg_loser:,.2f}
Expectancy per Trade: ₹{expectancy:,.2f}

Avg Charges/Trade: ₹{total_charges/total_trades:,.2f}
""")
    
    print("="*80)
    print("BY EXIT REASON")
    print("="*80)
    for reason in ['TARGET', 'SL', 'TIME_EXIT', 'EOD']:
        subset = df_trades[df_trades['exit_reason'] == reason]
        if len(subset) > 0:
            net = subset['net_pnl'].sum()
            wr = len(subset[subset['net_pnl'] > 0]) / len(subset) * 100
            print(f"{reason}: {len(subset)} trades | Win Rate: {wr:.1f}% | Net P&L: ₹{net:,.2f}")
    
    # Export to Excel
    excel_file = 'intraday_backtest_optimized.xlsx'
    
    df_export = df_trades.copy()
    df_export['entry_time'] = df_export['entry_time'].astype(str)
    df_export['exit_time'] = df_export['exit_time'].astype(str)
    df_export['date'] = df_export['date'].astype(str)
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df_export.to_excel(writer, sheet_name='All Trades', index=False)
        
        daily = df_export.groupby('date').agg({
            'net_pnl': 'sum',
            'gross_pnl': 'sum',
            'total_charges': 'sum',
            'symbol': 'count'
        }).rename(columns={'symbol': 'num_trades'}).reset_index()
        daily['cumulative_pnl'] = daily['net_pnl'].cumsum()
        daily.to_excel(writer, sheet_name='Daily Summary', index=False)
    
    print()
    print(f"✅ Trades exported to: {excel_file}")

else:
    print("No trades generated with strict criteria!")
