"""
OCO (One-Cancels-Other) Trade Monitor
Monitors active trades and cancels SL when Target hits (or vice versa)
"""

import json
import time
from datetime import datetime, time as dtime
from kiteconnect import KiteConnect
import pytz

IST = pytz.timezone('Asia/Kolkata')

# Load credentials
with open('.kite_creds.json') as f:
    creds = json.load(f)
with open('.kite_session.json') as f:
    session = json.load(f)

kite = KiteConnect(api_key=creds['api_key'])
kite.set_access_token(session['access_token'])

# Active trades to monitor
ACTIVE_TRADES = [
    {
        'symbol': 'TATACONSUM26MARFUT',
        'qty': 550,
        'entry_price': 1085.10,
        'sl_order_id': '260316190859965',
        'target_order_id': '260316190861910',
        'sl_price': 1068.3,
        'target_price': 1097.0,
        'type': 'BUY'
    },
    {
        'symbol': 'NHPC26MARFUT',
        'qty': 6400,
        'entry_price': 75.43,
        'sl_order_id': '260316190860188',
        'target_order_id': '260316190860210',
        'sl_price': 74.5,
        'target_price': 76.65,
        'type': 'BUY'
    }
]

def get_order_status(order_id):
    """Get status of an order"""
    try:
        orders = kite.orders()
        for o in orders:
            if str(o['order_id']) == str(order_id):
                return o['status']
        return 'NOT_FOUND'
    except:
        return 'ERROR'

def cancel_order(order_id):
    """Cancel an open order"""
    try:
        kite.cancel_order(variety=kite.VARIETY_REGULAR, order_id=order_id)
        return True
    except Exception as e:
        print(f"  Could not cancel {order_id}: {e}")
        return False

def get_position_pnl(symbol):
    """Get current P&L for a position"""
    try:
        positions = kite.positions()
        for p in positions.get('net', []):
            if p['tradingsymbol'] == symbol:
                return p['pnl'], p['last_price'], p['quantity']
        return 0, 0, 0
    except:
        return 0, 0, 0

def monitor_once():
    """Check all trades once and handle OCO"""
    now = datetime.now(IST)
    print(f"\n[{now.strftime('%H:%M:%S')}] Checking positions...")
    
    completed_trades = []
    
    for trade in ACTIVE_TRADES:
        symbol = trade['symbol']
        sl_status = get_order_status(trade['sl_order_id'])
        target_status = get_order_status(trade['target_order_id'])
        pnl, ltp, qty = get_position_pnl(symbol)
        
        emoji = "🟢" if pnl >= 0 else "🔴"
        print(f"  {emoji} {symbol}: LTP ₹{ltp:.2f} | P&L ₹{pnl:,.0f} | SL: {sl_status} | TGT: {target_status}")
        
        # Check if SL hit (SL order completed)
        if sl_status == 'COMPLETE':
            print(f"  ❌ SL HIT for {symbol}! Cancelling target order...")
            cancel_order(trade['target_order_id'])
            completed_trades.append(trade)
            
        # Check if Target hit (Target order completed)
        elif target_status == 'COMPLETE':
            print(f"  ✅ TARGET HIT for {symbol}! Cancelling SL order...")
            cancel_order(trade['sl_order_id'])
            completed_trades.append(trade)
        
        # Check if position is closed (qty = 0)
        elif qty == 0:
            print(f"  ⚠️  Position closed for {symbol}. Cleaning up orders...")
            cancel_order(trade['sl_order_id'])
            cancel_order(trade['target_order_id'])
            completed_trades.append(trade)
    
    # Remove completed trades
    for t in completed_trades:
        if t in ACTIVE_TRADES:
            ACTIVE_TRADES.remove(t)
    
    return len(ACTIVE_TRADES) > 0

def main():
    print("="*80)
    print("OCO TRADE MONITOR")
    print("="*80)
    print(f"Monitoring {len(ACTIVE_TRADES)} trades")
    print("Press Ctrl+C to stop")
    print()
    
    # Summary
    for t in ACTIVE_TRADES:
        print(f"  {t['symbol']}: Entry ₹{t['entry_price']} | SL ₹{t['sl_price']} | TGT ₹{t['target_price']}")
    
    exit_time = dtime(15, 20)  # Stop monitoring at 3:20 PM
    
    try:
        while True:
            now = datetime.now(IST)
            
            # Check if market is closed
            if now.time() > exit_time:
                print("\n⏰ Market closing. Exiting monitor.")
                break
            
            # Monitor trades
            has_active = monitor_once()
            
            if not has_active:
                print("\n✅ All trades completed!")
                break
            
            # Wait before next check
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitor stopped by user")
        
    # Final summary
    print("\n" + "="*80)
    print("FINAL POSITION STATUS")
    print("="*80)
    
    positions = kite.positions()
    total_pnl = 0
    for p in positions.get('net', []):
        if p['quantity'] != 0:
            pnl = p['pnl']
            total_pnl += pnl
            emoji = "🟢" if pnl >= 0 else "🔴"
            print(f"{emoji} {p['tradingsymbol']}: Qty {p['quantity']} | P&L ₹{pnl:,.2f}")
    
    print(f"\nTotal P&L: ₹{total_pnl:,.2f}")

if __name__ == "__main__":
    main()
