"""
Backtest Quantified Trading Rules on Nifty 100 Stocks
Uses daily data for 1 year + available intraday data for validation
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

# ============== INDICATORS ==============

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_supertrend(df, period=10, multiplier=3):
    atr = calculate_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = 1
        else:
            if df['close'].iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lowerband.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upperband.iloc[i]
                direction.iloc[i] = -1
    
    return direction

# ============== NIFTY 100 STOCKS ==============

NIFTY_100 = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'SBIN',
    'BHARTIARTL', 'KOTAKBANK', 'ITC', 'LT', 'AXISBANK', 'BAJFINANCE', 'MARUTI',
    'ASIANPAINT', 'HCLTECH', 'SUNPHARMA', 'TITAN', 'WIPRO', 'TATASTEEL',
    'POWERGRID', 'NTPC', 'M&M', 'JSWSTEEL', 'ADANIENT', 'BAJAJFINSV', 'ONGC',
    'COALINDIA', 'TECHM', 'INDUSINDBK', 'HINDALCO', 'DRREDDY', 'CIPLA',
    'TATAPOWER', 'BEL', 'VEDL', 'JINDALSTEL', 'DLF', 'BANKBARODA', 'PNB',
    'GRASIM', 'NESTLEIND', 'ULTRACEMCO', 'BRITANNIA', 'DIVISLAB',
    'EICHERMOT', 'APOLLOHOSP', 'HEROMOTOCO', 'DABUR', 'PIDILITIND', 'GODREJCP',
    'BAJAJ-AUTO', 'HAVELLS', 'SIEMENS', 'SHREECEM', 'AMBUJACEM', 'ACC',
    'INDIGO', 'TRENT', 'ADANIPORTS',
    'GAIL', 'IOC', 'BPCL', 'HDFCLIFE', 'SBILIFE', 'ICICIPRULI', 'ICICIGI',
    'HDFCAMC', 'NAUKRI', 'MUTHOOTFIN', 'CHOLAFIN', 'SHRIRAMFIN', 'PFC', 'RECLTD',
    'TATACONSUM', 'MARICO', 'COLPAL', 'BERGEPAINT', 'PAGEIND', 'ASTRAL',
    'POLYCAB', 'CUMMINSIND', 'ABB', 'VOLTAS', 'CROMPTON',
    'LUPIN', 'AUROPHARMA', 'TORNTPHARM', 'ALKEM', 'BIOCON', 'PERSISTENT',
    'MPHASIS', 'COFORGE'
]

# ============== SCORING FUNCTION ==============

def score_trade(row, nifty_change, trade_type='BUY'):
    """
    Score a trade based on quantified rules (0-10)
    """
    score = 0
    reasons = []
    
    rsi = row['rsi']
    supertrend = row['supertrend']
    day_change = row['day_change']
    ema9 = row['ema9']
    ema21 = row['ema21']
    close = row['close']
    
    if trade_type == 'BUY':
        # Market Direction (0-2 pts)
        if nifty_change > 0.5:
            score += 2
            reasons.append('Nifty strong')
        elif nifty_change > 0.2:
            score += 1
            reasons.append('Nifty positive')
        
        # RSI Zone (0-2 pts)
        if 40 <= rsi <= 65:
            score += 2
            reasons.append('RSI sweet spot')
        elif 35 <= rsi < 40 or 65 < rsi <= 70:
            score += 1
            reasons.append('RSI acceptable')
        
        # Supertrend (0-1 pt)
        if supertrend == 1:
            score += 1
            reasons.append('Supertrend BUY')
        
        # EMA alignment (0-2 pts)
        if ema9 > ema21:
            score += 1
            reasons.append('EMA bullish')
        if close > ema9:
            score += 1
            reasons.append('Above EMA9')
        
        # Momentum (0-2 pts)
        if day_change > 1:
            score += 2
            reasons.append('Strong momentum')
        elif day_change > 0.3:
            score += 1
            reasons.append('Positive momentum')
            
    else:  # SHORT
        # Market Direction (0-2 pts)
        if nifty_change < -0.5:
            score += 2
            reasons.append('Nifty weak')
        elif nifty_change < -0.2:
            score += 1
            reasons.append('Nifty negative')
        
        # RSI Zone (0-2 pts)
        if 45 <= rsi <= 70:
            score += 2
            reasons.append('RSI sweet spot')
        elif 70 < rsi <= 80:
            score += 1
            reasons.append('RSI overbought')
        
        # Supertrend (0-1 pt)
        if supertrend == -1:
            score += 1
            reasons.append('Supertrend SELL')
        
        # EMA alignment (0-2 pts)
        if ema9 < ema21:
            score += 1
            reasons.append('EMA bearish')
        if close < ema9:
            score += 1
            reasons.append('Below EMA9')
        
        # Momentum (0-2 pts)
        if day_change < -1:
            score += 2
            reasons.append('Strong down momentum')
        elif day_change < -0.3:
            score += 1
            reasons.append('Negative momentum')
    
    return score, reasons

# ============== BACKTEST ENGINE ==============

def backtest_stock(kite, symbol, token, nifty_data):
    """
    Backtest quantified rules on a single stock
    """
    try:
        to_date = datetime.now()
        from_date = to_date - timedelta(days=365)
        
        data = kite.historical_data(
            instrument_token=token,
            from_date=from_date.strftime('%Y-%m-%d'),
            to_date=to_date.strftime('%Y-%m-%d'),
            interval='day'
        )
        
        if not data or len(data) < 50:
            return None
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Calculate indicators
        df['rsi'] = calculate_rsi(df['close'])
        df['ema9'] = calculate_ema(df['close'], 9)
        df['ema21'] = calculate_ema(df['close'], 21)
        df['atr'] = calculate_atr(df)
        df['supertrend'] = calculate_supertrend(df)
        df['day_change'] = df['close'].pct_change() * 100
        
        # Merge with Nifty data
        df = df.merge(nifty_data[['date', 'nifty_change']], on='date', how='left')
        
        trades = []
        
        # Skip first 30 rows for indicator warmup
        for i in range(30, len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            if pd.isna(row['rsi']) or pd.isna(row['nifty_change']):
                continue
            
            # Check for BUY signal
            buy_score, buy_reasons = score_trade(row, row['nifty_change'], 'BUY')
            
            # Check for SHORT signal
            sell_score, sell_reasons = score_trade(row, row['nifty_change'], 'SHORT')
            
            # Only take trades with score >= 5
            if buy_score >= 5 and buy_score > sell_score:
                # Simulate entry at close, exit next day
                entry = row['close']
                sl = entry * 0.99  # 1% SL
                target = entry * 1.015  # 1.5% target
                
                # Check outcome
                next_high = next_row['high']
                next_low = next_row['low']
                next_close = next_row['close']
                
                if next_low <= sl:
                    exit_price = sl
                    result = 'SL'
                elif next_high >= target:
                    exit_price = target
                    result = 'TARGET'
                else:
                    exit_price = next_close
                    result = 'EXIT'
                
                pnl_pct = ((exit_price - entry) / entry) * 100
                
                trades.append({
                    'date': row['date'],
                    'symbol': symbol,
                    'type': 'BUY',
                    'score': buy_score,
                    'entry': entry,
                    'exit': exit_price,
                    'result': result,
                    'pnl_pct': pnl_pct,
                    'nifty_aligned': row['nifty_change'] > 0.2,
                    'rsi': row['rsi']
                })
                
            elif sell_score >= 5 and sell_score > buy_score:
                entry = row['close']
                sl = entry * 1.01  # 1% SL
                target = entry * 0.985  # 1.5% target
                
                next_high = next_row['high']
                next_low = next_row['low']
                next_close = next_row['close']
                
                if next_high >= sl:
                    exit_price = sl
                    result = 'SL'
                elif next_low <= target:
                    exit_price = target
                    result = 'TARGET'
                else:
                    exit_price = next_close
                    result = 'EXIT'
                
                pnl_pct = ((entry - exit_price) / entry) * 100
                
                trades.append({
                    'date': row['date'],
                    'symbol': symbol,
                    'type': 'SHORT',
                    'score': sell_score,
                    'entry': entry,
                    'exit': exit_price,
                    'result': result,
                    'pnl_pct': pnl_pct,
                    'nifty_aligned': row['nifty_change'] < -0.2,
                    'rsi': row['rsi']
                })
        
        return trades
        
    except Exception as e:
        return None

# ============== MAIN ==============

print("="*80)
print("BACKTEST: Quantified Trading Rules on Nifty 100 Stocks")
print("="*80)
print()

# Get Nifty data first
print("Fetching Nifty 50 data...")
nifty_token = 256265

to_date = datetime.now()
from_date = to_date - timedelta(days=365)

nifty_data = kite.historical_data(
    instrument_token=nifty_token,
    from_date=from_date.strftime('%Y-%m-%d'),
    to_date=to_date.strftime('%Y-%m-%d'),
    interval='day'
)

nifty_df = pd.DataFrame(nifty_data)
nifty_df['date'] = pd.to_datetime(nifty_df['date']).dt.date
nifty_df['nifty_change'] = nifty_df['close'].pct_change() * 100
nifty_df = nifty_df[['date', 'nifty_change', 'close']].rename(columns={'close': 'nifty_close'})

print(f"Nifty data: {len(nifty_df)} days")
print()

# Get instrument tokens
print("Fetching instruments...")
instruments = kite.instruments('NSE')
token_map = {}
for symbol in NIFTY_100:
    for inst in instruments:
        if inst['tradingsymbol'] == symbol:
            token_map[symbol] = inst['instrument_token']
            break

print(f"Found {len(token_map)} stocks")
print()

# Backtest each stock
all_trades = []
print("Running backtest...")

for i, symbol in enumerate(NIFTY_100):
    if symbol not in token_map:
        continue
    
    print(f"\r  [{i+1}/{len(NIFTY_100)}] {symbol}...", end='', flush=True)
    
    trades = backtest_stock(kite, symbol, token_map[symbol], nifty_df)
    if trades:
        all_trades.extend(trades)

print(f"\n\nTotal trades generated: {len(all_trades)}")
print()

# Convert to DataFrame for analysis
if all_trades:
    df = pd.DataFrame(all_trades)
    
    print("="*80)
    print("OVERALL RESULTS")
    print("="*80)
    
    total_trades = len(df)
    winners = len(df[df['pnl_pct'] > 0])
    losers = len(df[df['pnl_pct'] <= 0])
    win_rate = winners / total_trades * 100
    
    avg_win = df[df['pnl_pct'] > 0]['pnl_pct'].mean() if winners > 0 else 0
    avg_loss = df[df['pnl_pct'] <= 0]['pnl_pct'].mean() if losers > 0 else 0
    
    total_pnl = df['pnl_pct'].sum()
    avg_pnl = df['pnl_pct'].mean()
    
    gross_profit = df[df['pnl_pct'] > 0]['pnl_pct'].sum()
    gross_loss = abs(df[df['pnl_pct'] <= 0]['pnl_pct'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    print(f"""
Total Trades: {total_trades}
Winners: {winners} | Losers: {losers}
Win Rate: {win_rate:.1f}%

Avg Win: +{avg_win:.2f}%
Avg Loss: {avg_loss:.2f}%
Avg P&L per trade: {avg_pnl:.2f}%

Total P&L: {total_pnl:.1f}%
Profit Factor: {profit_factor:.2f}

Gross Profit: +{gross_profit:.1f}%
Gross Loss: -{gross_loss:.1f}%
""")
    
    # By trade type
    print("="*80)
    print("BY TRADE TYPE")
    print("="*80)
    
    for trade_type in ['BUY', 'SHORT']:
        subset = df[df['type'] == trade_type]
        if len(subset) > 0:
            wr = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
            avg = subset['pnl_pct'].mean()
            print(f"{trade_type}: {len(subset)} trades | Win Rate: {wr:.1f}% | Avg P&L: {avg:.2f}%")
    
    # By score
    print()
    print("="*80)
    print("BY SCORE (Higher score = Better signal)")
    print("="*80)
    
    for score in sorted(df['score'].unique()):
        subset = df[df['score'] == score]
        if len(subset) >= 10:  # Only show meaningful groups
            wr = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
            avg = subset['pnl_pct'].mean()
            print(f"Score {score}: {len(subset):4d} trades | Win Rate: {wr:.1f}% | Avg P&L: {avg:+.2f}%")
    
    # By Nifty alignment
    print()
    print("="*80)
    print("BY MARKET ALIGNMENT")
    print("="*80)
    
    aligned = df[df['nifty_aligned'] == True]
    not_aligned = df[df['nifty_aligned'] == False]
    
    if len(aligned) > 0:
        wr = len(aligned[aligned['pnl_pct'] > 0]) / len(aligned) * 100
        avg = aligned['pnl_pct'].mean()
        print(f"Aligned with Nifty:     {len(aligned):4d} trades | Win Rate: {wr:.1f}% | Avg P&L: {avg:+.2f}%")
    
    if len(not_aligned) > 0:
        wr = len(not_aligned[not_aligned['pnl_pct'] > 0]) / len(not_aligned) * 100
        avg = not_aligned['pnl_pct'].mean()
        print(f"Against Nifty:          {len(not_aligned):4d} trades | Win Rate: {wr:.1f}% | Avg P&L: {avg:+.2f}%")
    
    # By RSI zone
    print()
    print("="*80)
    print("BY RSI ZONE")
    print("="*80)
    
    rsi_zones = [
        ('Oversold (<35)', df['rsi'] < 35),
        ('Low (35-45)', (df['rsi'] >= 35) & (df['rsi'] < 45)),
        ('Sweet Spot (45-65)', (df['rsi'] >= 45) & (df['rsi'] <= 65)),
        ('High (65-75)', (df['rsi'] > 65) & (df['rsi'] <= 75)),
        ('Overbought (>75)', df['rsi'] > 75)
    ]
    
    for zone_name, condition in rsi_zones:
        subset = df[condition]
        if len(subset) >= 10:
            wr = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
            avg = subset['pnl_pct'].mean()
            print(f"{zone_name:25s}: {len(subset):4d} trades | Win Rate: {wr:.1f}% | Avg P&L: {avg:+.2f}%")
    
    # By result type
    print()
    print("="*80)
    print("BY EXIT TYPE")
    print("="*80)
    
    for result in ['TARGET', 'SL', 'EXIT']:
        subset = df[df['result'] == result]
        if len(subset) > 0:
            pct = len(subset) / len(df) * 100
            avg = subset['pnl_pct'].mean()
            print(f"{result:8s}: {len(subset):4d} trades ({pct:.1f}%) | Avg P&L: {avg:+.2f}%")
    
    # Top performing stocks
    print()
    print("="*80)
    print("TOP 10 STOCKS BY TOTAL P&L")
    print("="*80)
    
    stock_perf = df.groupby('symbol').agg({
        'pnl_pct': ['sum', 'count', 'mean'],
    }).round(2)
    stock_perf.columns = ['total_pnl', 'trades', 'avg_pnl']
    stock_perf = stock_perf.sort_values('total_pnl', ascending=False).head(10)
    
    for symbol, row in stock_perf.iterrows():
        print(f"{symbol:12s}: {row['trades']:3.0f} trades | Total: {row['total_pnl']:+6.1f}% | Avg: {row['avg_pnl']:+.2f}%")
    
    # Worst performing stocks
    print()
    print("="*80)
    print("BOTTOM 10 STOCKS BY TOTAL P&L")
    print("="*80)
    
    stock_perf = df.groupby('symbol').agg({
        'pnl_pct': ['sum', 'count', 'mean'],
    }).round(2)
    stock_perf.columns = ['total_pnl', 'trades', 'avg_pnl']
    stock_perf = stock_perf.sort_values('total_pnl', ascending=True).head(10)
    
    for symbol, row in stock_perf.iterrows():
        print(f"{symbol:12s}: {row['trades']:3.0f} trades | Total: {row['total_pnl']:+6.1f}% | Avg: {row['avg_pnl']:+.2f}%")
    
    # Save results
    df.to_csv('backtest_results.csv', index=False)
    print()
    print("Results saved to backtest_results.csv")
    
    # Final recommendations
    print()
    print("="*80)
    print("KEY INSIGHTS FROM BACKTEST")
    print("="*80)
    
    # Find optimal score threshold
    print("\nOptimal Score Threshold Analysis:")
    for min_score in [5, 6, 7, 8]:
        subset = df[df['score'] >= min_score]
        if len(subset) >= 20:
            wr = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
            avg = subset['pnl_pct'].mean()
            total = subset['pnl_pct'].sum()
            print(f"  Score >= {min_score}: {len(subset):4d} trades | Win Rate: {wr:.1f}% | Avg: {avg:+.2f}% | Total: {total:+.1f}%")

else:
    print("No trades generated!")
