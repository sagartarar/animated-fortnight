"""
REAL Intraday Backtest with 15-minute data
- Exact entry/exit timestamps
- Zerodha charges included
- Exports to Excel for verification
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
    
    # Brokerage: Rs 20 per order or 0.03% whichever is lower (2 orders: buy + sell)
    brokerage_buy = min(20, buy_value * 0.0003)
    brokerage_sell = min(20, sell_value * 0.0003)
    brokerage = brokerage_buy + brokerage_sell
    
    # STT: 0.025% on sell side only (intraday)
    stt = sell_value * 0.00025
    
    # Exchange Transaction Charges: 0.00307% (NSE)
    exchange = total_turnover * 0.0000307
    
    # SEBI Charges: Rs 10 per crore
    sebi = total_turnover * 0.000001
    
    # GST: 18% on (brokerage + exchange + sebi)
    gst = (brokerage + exchange + sebi) * 0.18
    
    # Stamp Duty: 0.015% on buy side
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
    """Calculate VWAP - resets each day"""
    df = df.copy()
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_volume'] = df['typical_price'] * df['volume']
    
    # Group by date and calculate cumulative
    df['date_only'] = df['date'].dt.date
    df['cum_tp_vol'] = df.groupby('date_only')['tp_volume'].cumsum()
    df['cum_vol'] = df.groupby('date_only')['volume'].cumsum()
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
    
    return df['vwap']

def calculate_supertrend(df, period=10, multiplier=3):
    df = df.copy()
    
    # ATR
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

# ============== SCORING ==============

def score_trade(row, nifty_change, trade_type):
    score = 0
    reasons = []
    
    if trade_type == 'BUY':
        # Nifty alignment (0-2)
        if nifty_change > 0.3:
            score += 2
            reasons.append('Nifty strong')
        elif nifty_change > 0.1:
            score += 1
            reasons.append('Nifty positive')
        
        # RSI (0-2)
        if 35 <= row['rsi'] <= 55:
            score += 2
            reasons.append('RSI sweet spot')
        elif 30 <= row['rsi'] < 35 or 55 < row['rsi'] <= 65:
            score += 1
        
        # VWAP (0-1)
        if row['close'] > row['vwap']:
            score += 1
            reasons.append('Above VWAP')
        
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
            
    else:  # SHORT
        if nifty_change < -0.3:
            score += 2
            reasons.append('Nifty weak')
        elif nifty_change < -0.1:
            score += 1
        
        if 55 <= row['rsi'] <= 70:
            score += 2
            reasons.append('RSI sweet spot')
        elif 70 < row['rsi'] <= 80:
            score += 1
        
        if row['close'] < row['vwap']:
            score += 1
            reasons.append('Below VWAP')
        
        if row['supertrend'] == -1:
            score += 2
            reasons.append('Supertrend SELL')
        
        if row['ema9'] < row['ema21']:
            score += 1
            reasons.append('EMA bearish')
        if row['close'] < row['ema9']:
            score += 1
    
    return score, reasons

# ============== STOCKS ==============

# Using top liquid F&O stocks
STOCKS = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'SBIN',
    'BHARTIARTL', 'KOTAKBANK', 'ITC', 'LT', 'AXISBANK', 'BAJFINANCE', 'MARUTI',
    'ASIANPAINT', 'HCLTECH', 'SUNPHARMA', 'TITAN', 'WIPRO', 'TATASTEEL',
    'POWERGRID', 'NTPC', 'M&M', 'JSWSTEEL', 'ADANIENT', 'BAJAJFINSV',
    'TECHM', 'INDUSINDBK', 'HINDALCO', 'DRREDDY', 'CIPLA', 'TATAPOWER',
    'DLF', 'BANKBARODA', 'PNB', 'GRASIM', 'ULTRACEMCO', 'TATAMOTORS'
]

# ============== MAIN BACKTEST ==============

print("="*80)
print("INTRADAY BACKTEST - 15 MIN TIMEFRAME")
print("="*80)
print()

# Get Nifty 15min data
print("Fetching Nifty 50 data (15min)...")
nifty_token = 256265
to_date = datetime.now()
from_date = to_date - timedelta(days=55)  # ~55 days of 15min data

nifty_data = kite.historical_data(
    instrument_token=nifty_token,
    from_date=from_date.strftime('%Y-%m-%d'),
    to_date=to_date.strftime('%Y-%m-%d'),
    interval='15minute'
)

nifty_df = pd.DataFrame(nifty_data)
nifty_df['date'] = pd.to_datetime(nifty_df['date'])
nifty_df['date_only'] = nifty_df['date'].dt.date

# Calculate daily open for Nifty change
nifty_daily_open = nifty_df.groupby('date_only')['open'].first().to_dict()
nifty_df['nifty_day_open'] = nifty_df['date_only'].map(nifty_daily_open)
nifty_df['nifty_change'] = ((nifty_df['close'] - nifty_df['nifty_day_open']) / nifty_df['nifty_day_open']) * 100

nifty_lookup = nifty_df.set_index('date')[['nifty_change', 'close']].to_dict('index')

print(f"Nifty data: {len(nifty_df)} candles, {nifty_df['date_only'].nunique()} days")
print(f"Date range: {nifty_df['date'].min()} to {nifty_df['date'].max()}")
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

# Capital and risk parameters
CAPITAL = 100000
RISK_PER_TRADE = 2000  # 2%
POSITION_SIZE = 200000  # Approx position value

all_trades = []

print("Running backtest...")
for i, symbol in enumerate(STOCKS):
    if symbol not in token_map:
        continue
    
    print(f"\r  [{i+1}/{len(STOCKS)}] {symbol}...", end='', flush=True)
    
    try:
        # Fetch 15min data
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
        
        # Calculate indicators
        df['rsi'] = calculate_rsi(df['close'])
        df['ema9'] = calculate_ema(df['close'], 9)
        df['ema21'] = calculate_ema(df['close'], 21)
        df['vwap'] = calculate_vwap(df)
        df['supertrend'], df['atr'] = calculate_supertrend(df)
        
        # Get Nifty change for each candle
        df['nifty_change'] = df['date'].map(lambda x: nifty_lookup.get(x, {}).get('nifty_change', 0))
        
        # Trading hours: 10:15 AM to 3:00 PM (entry), exit by 3:15 PM
        entry_start = time(10, 15)
        entry_end = time(14, 45)
        exit_time = time(15, 15)
        
        # Track daily trades (max 2 per day)
        daily_trades = {}
        
        for idx in range(50, len(df) - 4):  # Need future candles for exit
            row = df.iloc[idx]
            
            # Skip if outside trading hours
            if row['time'] < entry_start or row['time'] > entry_end:
                continue
            
            # Skip if already have 2 trades today
            date_key = row['date_only']
            if daily_trades.get(date_key, 0) >= 2:
                continue
            
            # Skip if indicators not ready
            if pd.isna(row['rsi']) or pd.isna(row['vwap']) or pd.isna(row['nifty_change']):
                continue
            
            # Score BUY and SHORT
            buy_score, buy_reasons = score_trade(row, row['nifty_change'], 'BUY')
            sell_score, sell_reasons = score_trade(row, row['nifty_change'], 'SHORT')
            
            trade_type = None
            score = 0
            reasons = []
            
            # Only take trades with score >= 7
            if buy_score >= 7 and buy_score > sell_score:
                trade_type = 'BUY'
                score = buy_score
                reasons = buy_reasons
            elif sell_score >= 7 and sell_score > buy_score:
                trade_type = 'SHORT'
                score = sell_score
                reasons = sell_reasons
            
            if trade_type is None:
                continue
            
            # Entry
            entry_price = row['close']
            entry_time = row['date']
            atr = row['atr'] if not pd.isna(row['atr']) else entry_price * 0.01
            
            # Position sizing based on ATR
            sl_distance = atr * 1.5
            qty = int(RISK_PER_TRADE / sl_distance)
            qty = max(1, min(qty, int(POSITION_SIZE / entry_price)))
            
            if trade_type == 'BUY':
                sl_price = entry_price - sl_distance
                target_price = entry_price + (sl_distance * 1.5)
            else:
                sl_price = entry_price + sl_distance
                target_price = entry_price - (sl_distance * 1.5)
            
            # Simulate trade execution - look at future candles
            exit_price = None
            exit_time = None
            exit_reason = None
            
            for j in range(idx + 1, min(idx + 20, len(df))):  # Max 20 candles (~5 hours)
                future = df.iloc[j]
                
                # Force exit at 3:15 PM
                if future['time'] >= time(15, 15):
                    exit_price = future['close']
                    exit_time = future['date']
                    exit_reason = 'TIME_EXIT'
                    break
                
                if trade_type == 'BUY':
                    # Check SL first (worst case)
                    if future['low'] <= sl_price:
                        exit_price = sl_price
                        exit_time = future['date']
                        exit_reason = 'SL'
                        break
                    # Check target
                    if future['high'] >= target_price:
                        exit_price = target_price
                        exit_time = future['date']
                        exit_reason = 'TARGET'
                        break
                else:  # SHORT
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
            
            # If no exit found, use last candle
            if exit_price is None:
                last_candle = df.iloc[min(idx + 19, len(df) - 1)]
                exit_price = last_candle['close']
                exit_time = last_candle['date']
                exit_reason = 'EOD'
            
            # Calculate P&L
            if trade_type == 'BUY':
                gross_pnl = (exit_price - entry_price) * qty
            else:
                gross_pnl = (entry_price - exit_price) * qty
            
            # Calculate charges
            buy_value = entry_price * qty if trade_type == 'BUY' else exit_price * qty
            sell_value = exit_price * qty if trade_type == 'BUY' else entry_price * qty
            charges = calculate_charges(buy_value, sell_value)
            
            net_pnl = gross_pnl - charges['total_charges']
            
            # Record trade
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
    # Create DataFrame
    df_trades = pd.DataFrame(all_trades)
    
    # Summary statistics
    print()
    print("="*80)
    print("BACKTEST RESULTS (WITH ZERODHA CHARGES)")
    print("="*80)
    
    total_trades = len(df_trades)
    winners = len(df_trades[df_trades['net_pnl'] > 0])
    losers = len(df_trades[df_trades['net_pnl'] <= 0])
    win_rate = winners / total_trades * 100
    
    total_gross = df_trades['gross_pnl'].sum()
    total_charges = df_trades['total_charges'].sum()
    total_net = df_trades['net_pnl'].sum()
    
    avg_winner = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].mean() if winners > 0 else 0
    avg_loser = df_trades[df_trades['net_pnl'] <= 0]['net_pnl'].mean() if losers > 0 else 0
    
    print(f"""
Period: {df_trades['date'].min()} to {df_trades['date'].max()}
Trading Days: {df_trades['date'].nunique()}

Total Trades: {total_trades}
Winners: {winners} | Losers: {losers}
Win Rate: {win_rate:.1f}%

Gross P&L: ₹{total_gross:,.2f}
Total Charges: ₹{total_charges:,.2f}
  - Brokerage: ₹{df_trades['brokerage'].sum():,.2f}
  - STT: ₹{df_trades['stt'].sum():,.2f}
  - Other: ₹{df_trades['other_charges'].sum():,.2f}

NET P&L: ₹{total_net:,.2f}

Avg Winner: ₹{avg_winner:,.2f}
Avg Loser: ₹{avg_loser:,.2f}
Avg Trade: ₹{df_trades['net_pnl'].mean():,.2f}
""")
    
    # By trade type
    print("="*80)
    print("BY TRADE TYPE")
    print("="*80)
    for t in ['BUY', 'SHORT']:
        subset = df_trades[df_trades['type'] == t]
        if len(subset) > 0:
            wr = len(subset[subset['net_pnl'] > 0]) / len(subset) * 100
            net = subset['net_pnl'].sum()
            print(f"{t}: {len(subset)} trades | Win Rate: {wr:.1f}% | Net P&L: ₹{net:,.2f}")
    
    # By exit reason
    print()
    print("="*80)
    print("BY EXIT REASON")
    print("="*80)
    for reason in ['TARGET', 'SL', 'TIME_EXIT', 'EOD']:
        subset = df_trades[df_trades['exit_reason'] == reason]
        if len(subset) > 0:
            net = subset['net_pnl'].sum()
            print(f"{reason}: {len(subset)} trades | Net P&L: ₹{net:,.2f}")
    
    # By score
    print()
    print("="*80)
    print("BY SCORE")
    print("="*80)
    for score in sorted(df_trades['score'].unique()):
        subset = df_trades[df_trades['score'] == score]
        if len(subset) >= 5:
            wr = len(subset[subset['net_pnl'] > 0]) / len(subset) * 100
            net = subset['net_pnl'].sum()
            print(f"Score {score}: {len(subset)} trades | Win Rate: {wr:.1f}% | Net P&L: ₹{net:,.2f}")
    
    # Export to Excel
    excel_file = 'intraday_backtest_trades.xlsx'
    
    # Convert timezone-aware datetimes to strings for Excel
    df_export = df_trades.copy()
    df_export['entry_time'] = df_export['entry_time'].astype(str)
    df_export['exit_time'] = df_export['exit_time'].astype(str)
    df_export['date'] = df_export['date'].astype(str)
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # All trades
        df_export.to_excel(writer, sheet_name='All Trades', index=False)
        
        # Daily summary
        daily = df_export.groupby('date').agg({
            'net_pnl': 'sum',
            'gross_pnl': 'sum',
            'total_charges': 'sum',
            'symbol': 'count'
        }).rename(columns={'symbol': 'num_trades'}).reset_index()
        daily['cumulative_pnl'] = daily['net_pnl'].cumsum()
        daily.to_excel(writer, sheet_name='Daily Summary', index=False)
        
        # Summary stats
        summary = pd.DataFrame({
            'Metric': [
                'Period Start', 'Period End', 'Trading Days',
                'Total Trades', 'Winners', 'Losers', 'Win Rate',
                'Gross P&L', 'Total Charges', 'Net P&L',
                'Avg Winner', 'Avg Loser', 'Profit Factor',
                'Max Drawdown', 'Best Day', 'Worst Day'
            ],
            'Value': [
                str(df_trades['date'].min()),
                str(df_trades['date'].max()),
                df_trades['date'].nunique(),
                total_trades, winners, losers, f"{win_rate:.1f}%",
                f"₹{total_gross:,.2f}", f"₹{total_charges:,.2f}", f"₹{total_net:,.2f}",
                f"₹{avg_winner:,.2f}", f"₹{avg_loser:,.2f}",
                f"{abs(total_gross) / abs(df_trades[df_trades['gross_pnl'] < 0]['gross_pnl'].sum()):.2f}" if df_trades[df_trades['gross_pnl'] < 0]['gross_pnl'].sum() != 0 else "N/A",
                f"₹{daily['cumulative_pnl'].min():,.2f}",
                f"₹{daily['net_pnl'].max():,.2f}",
                f"₹{daily['net_pnl'].min():,.2f}"
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print()
    print(f"✅ Detailed trades exported to: {excel_file}")
    print()
    print("Excel contains:")
    print("  - Sheet 1: All Trades (every trade with timestamps, prices, charges)")
    print("  - Sheet 2: Daily Summary (P&L by day)")
    print("  - Sheet 3: Summary Statistics")

else:
    print("No trades generated!")
