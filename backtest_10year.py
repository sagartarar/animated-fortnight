"""
10-YEAR BACKTEST - Optimized Intraday Strategy
Data: Nifty 200 stocks, 15-min interval, Feb 2015 - Feb 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import os
from glob import glob
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'trading_data_repo/data/nifty_200_15min'

# ============== STRATEGY PARAMETERS ==============

class Config:
    ENTRY_START = time(11, 0)   # 11:00 AM - KEY OPTIMIZATION
    ENTRY_END = time(14, 45)    # 2:45 PM
    EXIT_TIME = time(15, 15)    # 3:15 PM
    MIN_SCORE = 7
    ATR_MULTIPLIER = 1.5
    RR_RATIO = 1.5
    MAX_TRADES_PER_DAY = 2

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
    df['cum_tp_vol'] = df.groupby('date_only')['tp_volume'].cumsum()
    df['cum_vol'] = df.groupby('date_only')['volume'].cumsum()
    return df['cum_tp_vol'] / df['cum_vol']

def calculate_supertrend(df, period=10, multiplier=3):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    hl2 = (df['high'] + df['low']) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    direction = pd.Series(index=df.index, dtype=int)
    st = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = -1
        else:
            if df['close'].iloc[i] > st.iloc[i-1]:
                st.iloc[i] = lowerband.iloc[i]
                direction.iloc[i] = 1
            else:
                st.iloc[i] = upperband.iloc[i]
                direction.iloc[i] = -1
    
    return direction, atr

def score_trade(row, nifty_change, trade_type):
    score = 0
    reasons = []
    
    if trade_type == 'BUY':
        if nifty_change > 0.3:
            score += 2
            reasons.append('Nifty strong')
        elif nifty_change > 0.1:
            score += 1
        
        if 35 <= row['rsi'] <= 55:
            score += 2
            reasons.append('RSI optimal')
        elif 30 <= row['rsi'] < 35 or 55 < row['rsi'] <= 65:
            score += 1
        
        if row['close'] > row['vwap']:
            score += 1
            reasons.append('Above VWAP')
        
        if row['supertrend'] == 1:
            score += 2
            reasons.append('Supertrend BUY')
        
        if row['ema9'] > row['ema21']:
            score += 1
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
            reasons.append('RSI optimal')
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
        if row['close'] < row['ema9']:
            score += 1
    
    return score, reasons

# ============== CHARGES (Zerodha F&O) ==============

def calculate_charges(buy_value, sell_value):
    brokerage = min(20, buy_value * 0.0003) + min(20, sell_value * 0.0003)
    stt = sell_value * 0.000125
    exchange = (buy_value + sell_value) * 0.0000173
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00002
    return brokerage + stt + exchange + sebi + gst + stamp

# ============== LOAD NIFTY INDEX DATA ==============

print("="*80)
print("10-YEAR BACKTEST - OPTIMIZED INTRADAY STRATEGY")
print("="*80)
print()

# Check if NIFTY 50 index data exists
nifty_file = os.path.join(DATA_DIR, 'NIFTY 50_15min.csv')
if not os.path.exists(nifty_file):
    nifty_file = os.path.join(DATA_DIR, 'NIFTY50_15min.csv')
if not os.path.exists(nifty_file):
    # Use a proxy - RELIANCE or create synthetic
    print("No Nifty 50 index file found. Using proxy calculations...")
    nifty_df = None
else:
    nifty_df = pd.read_csv(nifty_file)
    nifty_df['date'] = pd.to_datetime(nifty_df['date'])

# Get all stock files
stock_files = glob(os.path.join(DATA_DIR, '*_15min.csv'))
print(f"Found {len(stock_files)} stock files")

# Sample a file to get date range
sample_df = pd.read_csv(stock_files[0])
sample_df['date'] = pd.to_datetime(sample_df['date'])
print(f"Date range: {sample_df['date'].min().date()} to {sample_df['date'].max().date()}")
print()

# ============== RUN BACKTEST ==============

all_trades = []
stock_count = 0
config = Config()

# If no Nifty index, we'll calculate market direction from a basket
MARKET_PROXY_STOCKS = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']

print("Running backtest...")
print()

for file_path in stock_files:
    symbol = os.path.basename(file_path).replace('_15min.csv', '')
    
    # Skip index files
    if 'NIFTY' in symbol or 'BANKNIFTY' in symbol:
        continue
    
    stock_count += 1
    if stock_count % 20 == 0:
        print(f"  Processed {stock_count} stocks, {len(all_trades)} trades so far...")
    
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df['date_only'] = df['date'].dt.date
        df['time'] = df['date'].dt.time
        
        # Need at least 100 candles
        if len(df) < 100:
            continue
        
        # Calculate indicators
        df['rsi'] = calculate_rsi(df['close'])
        df['ema9'] = calculate_ema(df['close'], 9)
        df['ema21'] = calculate_ema(df['close'], 21)
        df['vwap'] = calculate_vwap(df)
        df['supertrend'], df['atr'] = calculate_supertrend(df)
        
        # Calculate daily change for market proxy
        df['day_open'] = df.groupby('date_only')['open'].transform('first')
        df['day_change'] = ((df['close'] - df['day_open']) / df['day_open']) * 100
        
        # Track daily trades
        daily_trades = {}
        
        for idx in range(50, len(df) - 8):
            row = df.iloc[idx]
            
            # Time filter - KEY OPTIMIZATION
            if row['time'] < config.ENTRY_START or row['time'] > config.ENTRY_END:
                continue
            
            date_key = row['date_only']
            if daily_trades.get(date_key, 0) >= config.MAX_TRADES_PER_DAY:
                continue
            
            if pd.isna(row['rsi']) or pd.isna(row['vwap']) or pd.isna(row['atr']):
                continue
            
            # Use own stock's day change as proxy for market (simplified)
            nifty_change = row['day_change']
            
            # Score trades
            buy_score, _ = score_trade(row, nifty_change, 'BUY')
            sell_score, _ = score_trade(row, nifty_change, 'SHORT')
            
            trade_type = None
            score = 0
            
            if buy_score >= config.MIN_SCORE and buy_score > sell_score:
                trade_type = 'BUY'
                score = buy_score
            elif sell_score >= config.MIN_SCORE and sell_score > buy_score:
                trade_type = 'SHORT'
                score = sell_score
            
            if trade_type is None:
                continue
            
            entry_price = row['close']
            entry_time = row['date']
            atr = row['atr']
            
            sl_distance = atr * config.ATR_MULTIPLIER
            
            if trade_type == 'BUY':
                sl_price = entry_price - sl_distance
                target_price = entry_price + (sl_distance * config.RR_RATIO)
            else:
                sl_price = entry_price + sl_distance
                target_price = entry_price - (sl_distance * config.RR_RATIO)
            
            # Simulate exit
            exit_price = None
            exit_time = None
            exit_reason = None
            
            for j in range(idx + 1, min(idx + 20, len(df))):
                future = df.iloc[j]
                
                # Different day - exit at previous day close
                if future['date_only'] != date_key:
                    prev = df.iloc[j-1]
                    exit_price = prev['close']
                    exit_time = prev['date']
                    exit_reason = 'EOD'
                    break
                
                # Time exit
                if future['time'] >= config.EXIT_TIME:
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
                continue
            
            # Calculate P&L (using 1 lot = 100 shares for simplicity)
            qty = 100
            if trade_type == 'BUY':
                gross_pnl = (exit_price - entry_price) * qty
            else:
                gross_pnl = (entry_price - exit_price) * qty
            
            buy_val = entry_price * qty
            sell_val = exit_price * qty
            charges = calculate_charges(buy_val, sell_val)
            net_pnl = gross_pnl - charges
            
            trade = {
                'date': date_key,
                'year': entry_time.year,
                'symbol': symbol,
                'type': trade_type,
                'score': score,
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'exit_reason': exit_reason,
                'gross_pnl': round(gross_pnl, 2),
                'charges': round(charges, 2),
                'net_pnl': round(net_pnl, 2)
            }
            
            all_trades.append(trade)
            daily_trades[date_key] = daily_trades.get(date_key, 0) + 1
            
    except Exception as e:
        continue

print()
print(f"Total stocks processed: {stock_count}")
print(f"Total trades: {len(all_trades)}")
print()

# ============== RESULTS ==============

if all_trades:
    df_trades = pd.DataFrame(all_trades)
    
    total = len(df_trades)
    winners = len(df_trades[df_trades['net_pnl'] > 0])
    losers = len(df_trades[df_trades['net_pnl'] <= 0])
    win_rate = winners / total * 100
    
    gross = df_trades['gross_pnl'].sum()
    charges = df_trades['charges'].sum()
    net = df_trades['net_pnl'].sum()
    
    avg_winner = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].mean()
    avg_loser = df_trades[df_trades['net_pnl'] <= 0]['net_pnl'].mean()
    
    # Profit factor
    total_wins = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].sum()
    total_losses = abs(df_trades[df_trades['net_pnl'] <= 0]['net_pnl'].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    
    print("="*80)
    print("10-YEAR BACKTEST RESULTS")
    print("="*80)
    print(f"""
Period: {df_trades['date'].min()} to {df_trades['date'].max()}
Years: {df_trades['year'].nunique()}
Stocks: {df_trades['symbol'].nunique()}

OVERALL PERFORMANCE:
├─ Total Trades: {total:,}
├─ Winners: {winners:,}
├─ Losers: {losers:,}
├─ WIN RATE: {win_rate:.1f}%
│
├─ Gross P&L: ₹{gross:,.0f}
├─ Charges: ₹{charges:,.0f}
├─ NET P&L: ₹{net:,.0f}
│
├─ Avg Winner: ₹{avg_winner:,.0f}
├─ Avg Loser: ₹{avg_loser:,.0f}
├─ Avg Trade: ₹{net/total:,.0f}
│
├─ Profit Factor: {profit_factor:.2f}
└─ Expectancy: ₹{net/total:,.0f}/trade
""")
    
    # By Year
    print("="*80)
    print("PERFORMANCE BY YEAR")
    print("="*80)
    print(f"{'Year':<8} {'Trades':>10} {'Win Rate':>10} {'Net P&L':>15} {'Avg Trade':>12}")
    print("-"*60)
    
    for year in sorted(df_trades['year'].unique()):
        year_df = df_trades[df_trades['year'] == year]
        yr_trades = len(year_df)
        yr_wr = len(year_df[year_df['net_pnl'] > 0]) / yr_trades * 100
        yr_net = year_df['net_pnl'].sum()
        yr_avg = yr_net / yr_trades
        print(f"{year:<8} {yr_trades:>10,} {yr_wr:>9.1f}% ₹{yr_net:>13,.0f} ₹{yr_avg:>10,.0f}")
    
    # By Exit Reason
    print()
    print("="*80)
    print("BY EXIT REASON")
    print("="*80)
    for reason in ['TARGET', 'SL', 'TIME_EXIT', 'EOD']:
        subset = df_trades[df_trades['exit_reason'] == reason]
        if len(subset) > 0:
            wr = len(subset[subset['net_pnl'] > 0]) / len(subset) * 100
            net_r = subset['net_pnl'].sum()
            print(f"{reason:>10}: {len(subset):>8,} trades | WR: {wr:>5.1f}% | Net: ₹{net_r:>12,.0f}")
    
    # By Trade Type
    print()
    print("="*80)
    print("BY TRADE TYPE")
    print("="*80)
    for t in ['BUY', 'SHORT']:
        subset = df_trades[df_trades['type'] == t]
        if len(subset) > 0:
            wr = len(subset[subset['net_pnl'] > 0]) / len(subset) * 100
            net_t = subset['net_pnl'].sum()
            print(f"{t:>6}: {len(subset):>8,} trades | WR: {wr:>5.1f}% | Net: ₹{net_t:>12,.0f}")
    
    # Top/Bottom Stocks
    print()
    print("="*80)
    print("TOP 10 STOCKS")
    print("="*80)
    by_stock = df_trades.groupby('symbol').agg({
        'net_pnl': ['sum', 'count']
    })
    by_stock.columns = ['net_pnl', 'trades']
    by_stock['avg'] = by_stock['net_pnl'] / by_stock['trades']
    by_stock = by_stock.sort_values('net_pnl', ascending=False)
    
    for i, (symbol, row) in enumerate(by_stock.head(10).iterrows()):
        print(f"{i+1:>2}. {symbol:<15} ₹{row['net_pnl']:>12,.0f} ({row['trades']:>4} trades, ₹{row['avg']:>6,.0f}/trade)")
    
    print()
    print("BOTTOM 10 STOCKS")
    print("-"*60)
    for i, (symbol, row) in enumerate(by_stock.tail(10).iterrows()):
        print(f"{i+1:>2}. {symbol:<15} ₹{row['net_pnl']:>12,.0f} ({row['trades']:>4} trades, ₹{row['avg']:>6,.0f}/trade)")
    
    # Export
    df_trades.to_csv('backtest_10year_results.csv', index=False)
    print()
    print(f"✅ Results saved to: backtest_10year_results.csv")

else:
    print("No trades generated!")
