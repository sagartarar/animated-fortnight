"""
INTRADAY BACKTEST - FUTURES (F&O)
- Lower charges than equity
- Nifty 100 stocks (futures contracts)
- 15-minute timeframe
- Exact trades exported to Excel
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

def calculate_fno_charges(buy_value, sell_value, lot_size):
    """Calculate exact Zerodha FUTURES charges (much lower than equity)"""
    total_turnover = buy_value + sell_value
    
    # Brokerage: Rs 20 per order or 0.03% whichever is lower
    brokerage_buy = min(20, buy_value * 0.0003)
    brokerage_sell = min(20, sell_value * 0.0003)
    brokerage = brokerage_buy + brokerage_sell
    
    # STT: 0.0125% on sell side only (FUTURES - much lower than equity's 0.025%)
    stt = sell_value * 0.000125
    
    # Exchange Transaction Charges: 0.00173% for NSE Futures
    exchange = total_turnover * 0.0000173
    
    # SEBI Charges: Rs 10 per crore
    sebi = total_turnover * 0.000001
    
    # GST: 18% on (brokerage + exchange + sebi)
    gst = (brokerage + exchange + sebi) * 0.18
    
    # Stamp Duty: 0.002% on buy side (much lower for futures)
    stamp = buy_value * 0.00002
    
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

# ============== ZERODHA EQUITY CHARGES (for comparison) ==============

def calculate_equity_charges(buy_value, sell_value):
    """Calculate exact Zerodha EQUITY intraday charges"""
    total_turnover = buy_value + sell_value
    
    brokerage_buy = min(20, buy_value * 0.0003)
    brokerage_sell = min(20, sell_value * 0.0003)
    brokerage = brokerage_buy + brokerage_sell
    
    # STT: 0.025% on sell side (EQUITY - higher)
    stt = sell_value * 0.00025
    
    # Exchange: 0.00307% (NSE)
    exchange = total_turnover * 0.0000307
    
    sebi = total_turnover * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    
    # Stamp: 0.015% on buy (higher for equity)
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

# ============== SCORING ==============

def score_trade(row, nifty_change, trade_type):
    score = 0
    reasons = []
    
    if trade_type == 'BUY':
        if nifty_change > 0.3:
            score += 2
            reasons.append('Nifty strong')
        elif nifty_change > 0.1:
            score += 1
            reasons.append('Nifty positive')
        
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
            reasons.append('EMA bearish')
        if row['close'] < row['ema9']:
            score += 1
    
    return score, reasons

# ============== GET NIFTY 100 STOCKS ==============

print("="*80)
print("INTRADAY BACKTEST - F&O vs EQUITY COMPARISON")
print("Nifty 100 Stocks | 15-min Timeframe | ~55 days")
print("="*80)
print()

# Fetch all instruments
print("Fetching instrument list...")
instruments = kite.instruments('NSE')
nfo_instruments = kite.instruments('NFO')

# Get Nifty 100 constituents (F&O eligible stocks)
# These are all major F&O stocks
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

# Get spot tokens for Nifty 100
token_map = {}
for symbol in NIFTY_100:
    for inst in instruments:
        if inst['tradingsymbol'] == symbol and inst['segment'] == 'NSE':
            token_map[symbol] = {
                'token': inst['instrument_token'],
                'lot_size': 1  # For spot, we'll use qty based on position size
            }
            break

# Get current month futures lot sizes
current_month = datetime.now().strftime('%y%b').upper()  # e.g., '26FEB'
for symbol in NIFTY_100:
    for inst in nfo_instruments:
        if inst['tradingsymbol'].startswith(symbol) and 'FUT' in inst['instrument_type']:
            if symbol in token_map:
                token_map[symbol]['lot_size'] = inst.get('lot_size', 1)
            break

print(f"Found {len(token_map)} Nifty 100 stocks with tokens")

# Get Nifty 15min data
print("Fetching Nifty 50 index data (15min)...")
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

print(f"Nifty data: {len(nifty_df)} candles, {nifty_df['date_only'].nunique()} trading days")
print(f"Date range: {nifty_df['date'].min().date()} to {nifty_df['date'].max().date()}")
print()

# ============== BACKTEST PARAMETERS ==============

CAPITAL = 100000
RISK_PER_TRADE = 2000
POSITION_SIZE = 200000  # Position value for equity
MIN_SCORE = 7

# ============== RUN BACKTEST ==============

all_trades_equity = []
all_trades_fno = []

print(f"Running backtest on {len(token_map)} stocks...")
print()

for i, symbol in enumerate(NIFTY_100):
    if symbol not in token_map:
        continue
    
    print(f"\r  [{i+1}/{len(NIFTY_100)}] {symbol}...", end='', flush=True)
    
    try:
        # Fetch 15min data (spot data - works for backtesting both)
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
        
        # Calculate indicators
        df['rsi'] = calculate_rsi(df['close'])
        df['ema9'] = calculate_ema(df['close'], 9)
        df['ema21'] = calculate_ema(df['close'], 21)
        df['vwap'] = calculate_vwap(df)
        df['supertrend'], df['atr'] = calculate_supertrend(df)
        df['nifty_change'] = df['date'].map(lambda x: nifty_lookup.get(x, {}).get('nifty_change', 0))
        
        entry_start = time(10, 15)
        entry_end = time(14, 45)
        
        daily_trades = {}
        lot_size = token_map[symbol].get('lot_size', 1)
        
        for idx in range(50, len(df) - 4):
            row = df.iloc[idx]
            
            if row['time'] < entry_start or row['time'] > entry_end:
                continue
            
            date_key = row['date_only']
            if daily_trades.get(date_key, 0) >= 2:
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
            
            sl_distance = atr * 1.5
            
            # For equity: calculate qty based on risk
            equity_qty = int(RISK_PER_TRADE / sl_distance)
            equity_qty = max(1, min(equity_qty, int(POSITION_SIZE / entry_price)))
            
            # For F&O: use lot size (minimum 1 lot)
            fno_qty = lot_size
            
            if trade_type == 'BUY':
                sl_price = entry_price - sl_distance
                target_price = entry_price + (sl_distance * 1.5)
            else:
                sl_price = entry_price + sl_distance
                target_price = entry_price - (sl_distance * 1.5)
            
            # Simulate exit
            exit_price = None
            exit_time = None
            exit_reason = None
            
            for j in range(idx + 1, min(idx + 20, len(df))):
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
                last_candle = df.iloc[min(idx + 19, len(df) - 1)]
                exit_price = last_candle['close']
                exit_time = last_candle['date']
                exit_reason = 'EOD'
            
            # ===== EQUITY TRADE =====
            if trade_type == 'BUY':
                equity_gross = (exit_price - entry_price) * equity_qty
            else:
                equity_gross = (entry_price - exit_price) * equity_qty
            
            eq_buy_val = entry_price * equity_qty if trade_type == 'BUY' else exit_price * equity_qty
            eq_sell_val = exit_price * equity_qty if trade_type == 'BUY' else entry_price * equity_qty
            eq_charges = calculate_equity_charges(eq_buy_val, eq_sell_val)
            
            equity_net = equity_gross - eq_charges['total_charges']
            
            equity_trade = {
                'date': date_key,
                'symbol': symbol,
                'type': trade_type,
                'score': score,
                'entry_time': entry_time,
                'entry_price': round(entry_price, 2),
                'qty': equity_qty,
                'sl_price': round(sl_price, 2),
                'target_price': round(target_price, 2),
                'exit_time': exit_time,
                'exit_price': round(exit_price, 2),
                'exit_reason': exit_reason,
                'gross_pnl': round(equity_gross, 2),
                'brokerage': eq_charges['brokerage'],
                'stt': eq_charges['stt'],
                'other_charges': round(eq_charges['exchange'] + eq_charges['sebi'] + eq_charges['gst'] + eq_charges['stamp'], 2),
                'total_charges': eq_charges['total_charges'],
                'net_pnl': round(equity_net, 2),
                'nifty_change': round(row['nifty_change'], 2),
                'rsi': round(row['rsi'], 1),
                'reasons': ', '.join(reasons[:3])
            }
            all_trades_equity.append(equity_trade)
            
            # ===== F&O TRADE =====
            if trade_type == 'BUY':
                fno_gross = (exit_price - entry_price) * fno_qty
            else:
                fno_gross = (entry_price - exit_price) * fno_qty
            
            fno_buy_val = entry_price * fno_qty if trade_type == 'BUY' else exit_price * fno_qty
            fno_sell_val = exit_price * fno_qty if trade_type == 'BUY' else entry_price * fno_qty
            fno_charges = calculate_fno_charges(fno_buy_val, fno_sell_val, fno_qty)
            
            fno_net = fno_gross - fno_charges['total_charges']
            
            fno_trade = {
                'date': date_key,
                'symbol': symbol,
                'type': trade_type,
                'score': score,
                'lot_size': fno_qty,
                'entry_time': entry_time,
                'entry_price': round(entry_price, 2),
                'qty': fno_qty,
                'sl_price': round(sl_price, 2),
                'target_price': round(target_price, 2),
                'exit_time': exit_time,
                'exit_price': round(exit_price, 2),
                'exit_reason': exit_reason,
                'gross_pnl': round(fno_gross, 2),
                'brokerage': fno_charges['brokerage'],
                'stt': fno_charges['stt'],
                'other_charges': round(fno_charges['exchange'] + fno_charges['sebi'] + fno_charges['gst'] + fno_charges['stamp'], 2),
                'total_charges': fno_charges['total_charges'],
                'net_pnl': round(fno_net, 2),
                'nifty_change': round(row['nifty_change'], 2),
                'rsi': round(row['rsi'], 1),
                'reasons': ', '.join(reasons[:3])
            }
            all_trades_fno.append(fno_trade)
            
            daily_trades[date_key] = daily_trades.get(date_key, 0) + 1
            
    except Exception as e:
        continue

print(f"\n\nEquity Trades: {len(all_trades_equity)}")
print(f"F&O Trades: {len(all_trades_fno)}")

# ============== RESULTS ==============

def print_results(trades, title):
    if not trades:
        print(f"\n{title}: No trades!")
        return None
    
    df = pd.DataFrame(trades)
    
    total = len(df)
    winners = len(df[df['net_pnl'] > 0])
    losers = len(df[df['net_pnl'] <= 0])
    win_rate = winners / total * 100
    
    gross = df['gross_pnl'].sum()
    charges = df['total_charges'].sum()
    net = df['net_pnl'].sum()
    
    avg_win = df[df['net_pnl'] > 0]['net_pnl'].mean() if winners > 0 else 0
    avg_loss = df[df['net_pnl'] <= 0]['net_pnl'].mean() if losers > 0 else 0
    
    print(f"""
{'='*80}
{title}
{'='*80}
Period: {df['date'].min()} to {df['date'].max()}
Trading Days: {df['date'].nunique()}

Total Trades: {total}
Winners: {winners} | Losers: {losers}
Win Rate: {win_rate:.1f}%

Gross P&L: ₹{gross:,.2f}
Total Charges: ₹{charges:,.2f}
  - Brokerage: ₹{df['brokerage'].sum():,.2f}
  - STT: ₹{df['stt'].sum():,.2f}
  - Other: ₹{df['other_charges'].sum():,.2f}

NET P&L: ₹{net:,.2f}

Avg Winner: ₹{avg_win:,.2f}
Avg Loser: ₹{avg_loss:,.2f}
Avg Charges/Trade: ₹{charges/total:,.2f}
""")
    
    return df

print("\n")
df_equity = print_results(all_trades_equity, "EQUITY INTRADAY BACKTEST (Nifty 100)")
df_fno = print_results(all_trades_fno, "F&O FUTURES BACKTEST (Nifty 100)")

# ============== COMPARISON ==============

if df_equity is not None and df_fno is not None:
    print("="*80)
    print("EQUITY vs F&O COMPARISON")
    print("="*80)
    
    eq_charges = df_equity['total_charges'].sum()
    fno_charges = df_fno['total_charges'].sum()
    
    print(f"""
                        EQUITY          F&O (FUTURES)
Trades:                 {len(df_equity):,}            {len(df_fno):,}
Gross P&L:              ₹{df_equity['gross_pnl'].sum():>12,.2f}  ₹{df_fno['gross_pnl'].sum():>12,.2f}
Total Charges:          ₹{eq_charges:>12,.2f}  ₹{fno_charges:>12,.2f}
NET P&L:                ₹{df_equity['net_pnl'].sum():>12,.2f}  ₹{df_fno['net_pnl'].sum():>12,.2f}

Charges Saved in F&O:   ₹{eq_charges - fno_charges:,.2f}
Avg Charge/Trade:       ₹{eq_charges/len(df_equity):.2f}         ₹{fno_charges/len(df_fno):.2f}
""")

# ============== EXPORT TO EXCEL ==============

if df_equity is not None:
    # Equity Excel
    df_eq_export = df_equity.copy()
    df_eq_export['entry_time'] = df_eq_export['entry_time'].astype(str)
    df_eq_export['exit_time'] = df_eq_export['exit_time'].astype(str)
    df_eq_export['date'] = df_eq_export['date'].astype(str)
    
    with pd.ExcelWriter('backtest_equity_nifty100.xlsx', engine='openpyxl') as writer:
        df_eq_export.to_excel(writer, sheet_name='All Trades', index=False)
        
        daily = df_eq_export.groupby('date').agg({
            'net_pnl': 'sum',
            'gross_pnl': 'sum',
            'total_charges': 'sum',
            'symbol': 'count'
        }).rename(columns={'symbol': 'num_trades'}).reset_index()
        daily['cumulative_pnl'] = daily['net_pnl'].cumsum()
        daily.to_excel(writer, sheet_name='Daily Summary', index=False)
        
        # By stock
        by_stock = df_eq_export.groupby('symbol').agg({
            'net_pnl': ['sum', 'count', lambda x: (x > 0).sum() / len(x) * 100]
        }).round(2)
        by_stock.columns = ['Net_PnL', 'Trades', 'WinRate']
        by_stock = by_stock.sort_values('Net_PnL', ascending=False).reset_index()
        by_stock.to_excel(writer, sheet_name='By Stock', index=False)
    
    print(f"\n✅ Equity trades exported to: backtest_equity_nifty100.xlsx")

if df_fno is not None:
    # F&O Excel
    df_fno_export = df_fno.copy()
    df_fno_export['entry_time'] = df_fno_export['entry_time'].astype(str)
    df_fno_export['exit_time'] = df_fno_export['exit_time'].astype(str)
    df_fno_export['date'] = df_fno_export['date'].astype(str)
    
    with pd.ExcelWriter('backtest_fno_nifty100.xlsx', engine='openpyxl') as writer:
        df_fno_export.to_excel(writer, sheet_name='All Trades', index=False)
        
        daily = df_fno_export.groupby('date').agg({
            'net_pnl': 'sum',
            'gross_pnl': 'sum',
            'total_charges': 'sum',
            'symbol': 'count'
        }).rename(columns={'symbol': 'num_trades'}).reset_index()
        daily['cumulative_pnl'] = daily['net_pnl'].cumsum()
        daily.to_excel(writer, sheet_name='Daily Summary', index=False)
        
        # By stock
        by_stock = df_fno_export.groupby('symbol').agg({
            'net_pnl': ['sum', 'count', lambda x: (x > 0).sum() / len(x) * 100]
        }).round(2)
        by_stock.columns = ['Net_PnL', 'Trades', 'WinRate']
        by_stock = by_stock.sort_values('Net_PnL', ascending=False).reset_index()
        by_stock.to_excel(writer, sheet_name='By Stock', index=False)
    
    print(f"✅ F&O trades exported to: backtest_fno_nifty100.xlsx")

print("\n" + "="*80)
print("Excel files contain:")
print("  - Sheet 1: All Trades (every trade with exact timestamps, prices, charges)")
print("  - Sheet 2: Daily Summary (P&L by day with cumulative)")
print("  - Sheet 3: By Stock (performance ranking)")
print("="*80)
