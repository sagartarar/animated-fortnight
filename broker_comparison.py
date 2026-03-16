"""
BROKER COMPARISON - Same trades, different charge structures
Compare: Zerodha, Dhan, Fyers, Angel One, Groww, 5paisa, Upstox
Using actual trades from backtest
"""

import pandas as pd
import numpy as np

# Load the actual trades from backtest
df_equity = pd.read_excel('backtest_equity_nifty100.xlsx', sheet_name='All Trades')
df_fno = pd.read_excel('backtest_fno_nifty100.xlsx', sheet_name='All Trades')

print("="*100)
print("BROKER COMPARISON - SAME 4,005 TRADES, DIFFERENT BROKERS")
print("="*100)
print()

# ============== BROKER CHARGE STRUCTURES ==============

def calculate_zerodha_equity(buy_value, sell_value):
    """Zerodha Equity Intraday"""
    brokerage = min(20, buy_value * 0.0003) + min(20, sell_value * 0.0003)
    stt = sell_value * 0.00025  # 0.025%
    exchange = (buy_value + sell_value) * 0.0000307
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00015
    return brokerage + stt + exchange + sebi + gst + stamp

def calculate_zerodha_fno(buy_value, sell_value):
    """Zerodha Futures"""
    brokerage = min(20, buy_value * 0.0003) + min(20, sell_value * 0.0003)
    stt = sell_value * 0.000125  # 0.0125%
    exchange = (buy_value + sell_value) * 0.0000173
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00002
    return brokerage + stt + exchange + sebi + gst + stamp

def calculate_dhan_equity(buy_value, sell_value):
    """Dhan - Rs 20 flat per order (same as Zerodha but better app)"""
    brokerage = min(20, buy_value * 0.0003) + min(20, sell_value * 0.0003)
    stt = sell_value * 0.00025
    exchange = (buy_value + sell_value) * 0.0000307
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00015
    return brokerage + stt + exchange + sebi + gst + stamp

def calculate_dhan_fno(buy_value, sell_value):
    """Dhan Futures - Rs 20 flat"""
    brokerage = min(20, buy_value * 0.0003) + min(20, sell_value * 0.0003)
    stt = sell_value * 0.000125
    exchange = (buy_value + sell_value) * 0.0000173
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00002
    return brokerage + stt + exchange + sebi + gst + stamp

def calculate_fyers_equity(buy_value, sell_value):
    """Fyers - Rs 20 flat per order"""
    brokerage = 20 + 20  # Rs 20 per order (flat, not 0.03%)
    stt = sell_value * 0.00025
    exchange = (buy_value + sell_value) * 0.0000307
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00015
    return brokerage + stt + exchange + sebi + gst + stamp

def calculate_fyers_fno(buy_value, sell_value):
    """Fyers Futures - Rs 20 flat"""
    brokerage = 20 + 20
    stt = sell_value * 0.000125
    exchange = (buy_value + sell_value) * 0.0000173
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00002
    return brokerage + stt + exchange + sebi + gst + stamp

def calculate_angelone_equity(buy_value, sell_value):
    """Angel One - Rs 20 flat per order"""
    brokerage = 20 + 20
    stt = sell_value * 0.00025
    exchange = (buy_value + sell_value) * 0.0000307
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00015
    return brokerage + stt + exchange + sebi + gst + stamp

def calculate_angelone_fno(buy_value, sell_value):
    """Angel One Futures - Rs 20 flat"""
    brokerage = 20 + 20
    stt = sell_value * 0.000125
    exchange = (buy_value + sell_value) * 0.0000173
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00002
    return brokerage + stt + exchange + sebi + gst + stamp

def calculate_groww_equity(buy_value, sell_value):
    """Groww - Rs 20 flat per order"""
    brokerage = 20 + 20
    stt = sell_value * 0.00025
    exchange = (buy_value + sell_value) * 0.0000307
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00015
    return brokerage + stt + exchange + sebi + gst + stamp

def calculate_groww_fno(buy_value, sell_value):
    """Groww Futures - Rs 20 flat"""
    brokerage = 20 + 20
    stt = sell_value * 0.000125
    exchange = (buy_value + sell_value) * 0.0000173
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00002
    return brokerage + stt + exchange + sebi + gst + stamp

def calculate_5paisa_equity(buy_value, sell_value):
    """5paisa - Rs 20 flat OR Rs 0 with Power Investor Pack (Rs 999/month)"""
    brokerage = 20 + 20  # Standard
    stt = sell_value * 0.00025
    exchange = (buy_value + sell_value) * 0.0000307
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00015
    return brokerage + stt + exchange + sebi + gst + stamp

def calculate_5paisa_zero_equity(buy_value, sell_value):
    """5paisa with Power Investor Pack - Rs 0 brokerage"""
    brokerage = 0  # Zero brokerage with Rs 999/month subscription
    stt = sell_value * 0.00025
    exchange = (buy_value + sell_value) * 0.0000307
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00015
    return brokerage + stt + exchange + sebi + gst + stamp

def calculate_5paisa_zero_fno(buy_value, sell_value):
    """5paisa with Power Investor Pack - Rs 0 brokerage for F&O"""
    brokerage = 0
    stt = sell_value * 0.000125
    exchange = (buy_value + sell_value) * 0.0000173
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00002
    return brokerage + stt + exchange + sebi + gst + stamp

def calculate_upstox_equity(buy_value, sell_value):
    """Upstox - Rs 20 flat per order"""
    brokerage = 20 + 20
    stt = sell_value * 0.00025
    exchange = (buy_value + sell_value) * 0.0000307
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00015
    return brokerage + stt + exchange + sebi + gst + stamp

def calculate_upstox_fno(buy_value, sell_value):
    """Upstox Futures - Rs 20 flat"""
    brokerage = 20 + 20
    stt = sell_value * 0.000125
    exchange = (buy_value + sell_value) * 0.0000173
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00002
    return brokerage + stt + exchange + sebi + gst + stamp

def calculate_mstock_equity(buy_value, sell_value):
    """MStock (Mirae Asset) - Rs 0 brokerage (FREE for first year, then Rs 999/year)"""
    brokerage = 0
    stt = sell_value * 0.00025
    exchange = (buy_value + sell_value) * 0.0000307
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00015
    return brokerage + stt + exchange + sebi + gst + stamp

def calculate_mstock_fno(buy_value, sell_value):
    """MStock Futures - Rs 0 brokerage"""
    brokerage = 0
    stt = sell_value * 0.000125
    exchange = (buy_value + sell_value) * 0.0000173
    sebi = (buy_value + sell_value) * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = buy_value * 0.00002
    return brokerage + stt + exchange + sebi + gst + stamp

# ============== CALCULATE FOR ALL TRADES ==============

# Get gross P&L from trades
equity_gross = df_equity['gross_pnl'].sum()
fno_gross = df_fno['gross_pnl'].sum()

print(f"Total Trades: {len(df_equity)}")
print(f"Period: {df_equity['date'].min()} to {df_equity['date'].max()}")
print()

# Calculate charges for each broker
brokers_equity = {
    'Zerodha': calculate_zerodha_equity,
    'Dhan': calculate_dhan_equity,
    'Fyers': calculate_fyers_equity,
    'Angel One': calculate_angelone_equity,
    'Groww': calculate_groww_equity,
    '5paisa (Standard)': calculate_5paisa_equity,
    '5paisa (Zero - ₹999/mo)': calculate_5paisa_zero_equity,
    'Upstox': calculate_upstox_equity,
    'MStock (Zero - ₹999/yr)': calculate_mstock_equity,
}

brokers_fno = {
    'Zerodha': calculate_zerodha_fno,
    'Dhan': calculate_dhan_fno,
    'Fyers': calculate_fyers_fno,
    'Angel One': calculate_angelone_fno,
    'Groww': calculate_groww_fno,
    '5paisa (Zero - ₹999/mo)': calculate_5paisa_zero_fno,
    'Upstox': calculate_upstox_fno,
    'MStock (Zero - ₹999/yr)': calculate_mstock_fno,
}

# ============== EQUITY COMPARISON ==============

print("="*100)
print("EQUITY INTRADAY - BROKER COMPARISON")
print("="*100)
print()

equity_results = []

for broker, calc_func in brokers_equity.items():
    total_charges = 0
    for _, row in df_equity.iterrows():
        # Calculate buy and sell values
        if row['type'] == 'BUY':
            buy_val = row['entry_price'] * row['qty']
            sell_val = row['exit_price'] * row['qty']
        else:
            buy_val = row['exit_price'] * row['qty']
            sell_val = row['entry_price'] * row['qty']
        
        charges = calc_func(buy_val, sell_val)
        total_charges += charges
    
    net_pnl = equity_gross - total_charges
    
    # Add subscription cost if applicable
    subscription = 0
    if '999/mo' in broker:
        subscription = 999 * 2  # ~2 months in our backtest period
    elif '999/yr' in broker:
        subscription = 999 / 6  # Prorated for 2 months
    
    net_pnl_after_sub = net_pnl - subscription
    
    equity_results.append({
        'Broker': broker,
        'Gross P&L': equity_gross,
        'Total Charges': total_charges,
        'Subscription': subscription,
        'Net P&L': net_pnl_after_sub,
        'Avg Charge/Trade': total_charges / len(df_equity)
    })

equity_df = pd.DataFrame(equity_results)
equity_df = equity_df.sort_values('Net P&L', ascending=False)

print(f"Gross P&L (before charges): ₹{equity_gross:,.2f}")
print()
print(f"{'Broker':<30} {'Total Charges':>15} {'Subscription':>12} {'Net P&L':>15} {'Charge/Trade':>12}")
print("-"*90)

for _, row in equity_df.iterrows():
    status = "✅ PROFIT" if row['Net P&L'] > 0 else "❌ LOSS"
    print(f"{row['Broker']:<30} ₹{row['Total Charges']:>12,.0f} ₹{row['Subscription']:>10,.0f} ₹{row['Net P&L']:>12,.0f}  ₹{row['Avg Charge/Trade']:>9.2f}")

# ============== F&O COMPARISON ==============

print()
print("="*100)
print("F&O FUTURES - BROKER COMPARISON")
print("="*100)
print()

fno_results = []

for broker, calc_func in brokers_fno.items():
    total_charges = 0
    for _, row in df_fno.iterrows():
        if row['type'] == 'BUY':
            buy_val = row['entry_price'] * row['qty']
            sell_val = row['exit_price'] * row['qty']
        else:
            buy_val = row['exit_price'] * row['qty']
            sell_val = row['entry_price'] * row['qty']
        
        charges = calc_func(buy_val, sell_val)
        total_charges += charges
    
    net_pnl = fno_gross - total_charges
    
    subscription = 0
    if '999/mo' in broker:
        subscription = 999 * 2
    elif '999/yr' in broker:
        subscription = 999 / 6
    
    net_pnl_after_sub = net_pnl - subscription
    
    fno_results.append({
        'Broker': broker,
        'Gross P&L': fno_gross,
        'Total Charges': total_charges,
        'Subscription': subscription,
        'Net P&L': net_pnl_after_sub,
        'Avg Charge/Trade': total_charges / len(df_fno)
    })

fno_df = pd.DataFrame(fno_results)
fno_df = fno_df.sort_values('Net P&L', ascending=False)

print(f"Gross P&L (before charges): ₹{fno_gross:,.2f}")
print()
print(f"{'Broker':<30} {'Total Charges':>15} {'Subscription':>12} {'Net P&L':>15} {'Charge/Trade':>12}")
print("-"*90)

for _, row in fno_df.iterrows():
    status = "✅ PROFIT" if row['Net P&L'] > 0 else "❌ LOSS"
    print(f"{row['Broker']:<30} ₹{row['Total Charges']:>12,.0f} ₹{row['Subscription']:>10,.0f} ₹{row['Net P&L']:>12,.0f}  ₹{row['Avg Charge/Trade']:>9.2f}")

# ============== SAVINGS ANALYSIS ==============

print()
print("="*100)
print("SAVINGS ANALYSIS - vs Zerodha")
print("="*100)
print()

zerodha_eq = equity_df[equity_df['Broker'] == 'Zerodha']['Total Charges'].values[0]
zerodha_fno = fno_df[fno_df['Broker'] == 'Zerodha']['Total Charges'].values[0]

print("EQUITY:")
for _, row in equity_df.iterrows():
    savings = zerodha_eq - row['Total Charges']
    if savings != 0:
        print(f"  {row['Broker']:<28} Savings: ₹{savings:>10,.0f}  ({savings/zerodha_eq*100:.1f}%)")

print()
print("F&O:")
for _, row in fno_df.iterrows():
    savings = zerodha_fno - row['Total Charges']
    if savings != 0:
        print(f"  {row['Broker']:<28} Savings: ₹{savings:>10,.0f}  ({savings/zerodha_fno*100:.1f}%)")

# ============== BREAK-EVEN ANALYSIS ==============

print()
print("="*100)
print("BREAK-EVEN ANALYSIS - Minimum Gross P&L needed for Profit")
print("="*100)
print()

print("EQUITY (4,005 trades):")
for _, row in equity_df.iterrows():
    breakeven = row['Total Charges'] + row['Subscription']
    trades_for_profit = breakeven / len(df_equity)
    print(f"  {row['Broker']:<28} Need ₹{breakeven:>10,.0f} gross profit (₹{trades_for_profit:.2f}/trade)")

print()
print("F&O (4,005 trades):")
for _, row in fno_df.iterrows():
    breakeven = row['Total Charges'] + row['Subscription']
    trades_for_profit = breakeven / len(df_fno)
    print(f"  {row['Broker']:<28} Need ₹{breakeven:>10,.0f} gross profit (₹{trades_for_profit:.2f}/trade)")

# ============== RECOMMENDATION ==============

print()
print("="*100)
print("RECOMMENDATION")
print("="*100)

best_equity = equity_df.iloc[0]
best_fno = fno_df.iloc[0]

print(f"""
FOR EQUITY INTRADAY:
  Best Broker: {best_equity['Broker']}
  Net P&L: ₹{best_equity['Net P&L']:,.0f}
  vs Zerodha: ₹{best_equity['Net P&L'] - equity_df[equity_df['Broker'] == 'Zerodha']['Net P&L'].values[0]:+,.0f}
  
FOR F&O FUTURES:
  Best Broker: {best_fno['Broker']}
  Net P&L: ₹{best_fno['Net P&L']:,.0f}
  vs Zerodha: ₹{best_fno['Net P&L'] - fno_df[fno_df['Broker'] == 'Zerodha']['Net P&L'].values[0]:+,.0f}

NOTE: Even with zero-brokerage brokers, EQUITY INTRADAY still LOSES money
because of unavoidable STT, stamp duty, and exchange charges.

The ONLY way to be profitable with this strategy is F&O FUTURES.
""")

# Export comparison to Excel
with pd.ExcelWriter('broker_comparison.xlsx', engine='openpyxl') as writer:
    equity_df.to_excel(writer, sheet_name='Equity Comparison', index=False)
    fno_df.to_excel(writer, sheet_name='FnO Comparison', index=False)

print("✅ Comparison saved to: broker_comparison.xlsx")
