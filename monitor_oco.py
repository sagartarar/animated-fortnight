#!/usr/bin/env python3
"""
OCO (One-Cancels-Other) Trade Monitor - FIXED VERSION
Uses LTP API for real-time prices
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

# Active trades
TRADES = [
    {
        'symbol': 'TATACONSUM26MARFUT',
        'exchange': 'NFO',
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
        'exchange': 'NFO',
        'qty': 6400,
        'entry_price': 75.43,
        'sl_order_id': '260316190860188',
        'target_order_id': '260316190860210',
        'sl_price': 74.5,
        'target_price': 76.65,
        'type': 'BUY'
    }
]

def log(msg):
    timestamp = datetime.now(IST).strftime('%H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open('trade_monitor.log', 'a') as f:
        f.write(line + '\n')

def get_ltp(exchange, symbol):
    """Get real-time LTP using ltp() API"""
    try:
        key = f"{exchange}:{symbol}"
        data = kite.ltp([key])
        return data[key]['last_price']
    except Exception as e:
        log(f"LTP error for {symbol}: {e}")
        return None

def get_order_status(order_id):
    """Get status of an order"""
    try:
        orders = kite.orders()
        for o in orders:
            if str(o['order_id']) == str(order_id):
                return o['status'], o.get('average_price', 0)
        return 'NOT_FOUND', 0
    except Exception as e:
        log(f"Order status error: {e}")
        return 'ERROR', 0

def cancel_order(order_id):
    """Cancel an open order"""
    try:
        kite.cancel_order(variety=kite.VARIETY_REGULAR, order_id=order_id)
        log(f"✅ Cancelled order {order_id}")
        return True
    except Exception as e:
        log(f"❌ Cancel failed {order_id}: {e}")
        return False

def calculate_pnl(trade, ltp):
    """Calculate current P&L"""
    if trade['type'] == 'BUY':
        return (ltp - trade['entry_price']) * trade['qty']
    else:
        return (trade['entry_price'] - ltp) * trade['qty']

def main():
    # Clear old log
    with open('trade_monitor.log', 'w') as f:
        f.write('')
    
    log("=" * 70)
    log("OCO MONITOR STARTED (Real-time LTP)")
    log("=" * 70)
    
    for t in TRADES:
        log(f"  {t['symbol']}: Entry ₹{t['entry_price']} | SL ₹{t['sl_price']} | TGT ₹{t['target_price']}")
    
    log("-" * 70)
    
    active_trades = TRADES.copy()
    exit_time = dtime(15, 20)
    
    while active_trades:
        now = datetime.now(IST)
        
        if now.time() > exit_time:
            log("⏰ Market closing. Stopping monitor.")
            break
        
        for trade in active_trades[:]:
            symbol = trade['symbol']
            exchange = trade['exchange']
            
            # Get real-time LTP
            ltp = get_ltp(exchange, symbol)
            if ltp is None:
                continue
            
            # Calculate P&L
            pnl = calculate_pnl(trade, ltp)
            pnl_pct = ((ltp - trade['entry_price']) / trade['entry_price']) * 100
            if trade['type'] == 'SHORT':
                pnl_pct = -pnl_pct
            
            # Get order statuses
            sl_status, sl_fill = get_order_status(trade['sl_order_id'])
            tgt_status, tgt_fill = get_order_status(trade['target_order_id'])
            
            # Distance to SL and Target
            if trade['type'] == 'BUY':
                dist_to_sl = ((ltp - trade['sl_price']) / ltp) * 100
                dist_to_tgt = ((trade['target_price'] - ltp) / ltp) * 100
            else:
                dist_to_sl = ((trade['sl_price'] - ltp) / ltp) * 100
                dist_to_tgt = ((ltp - trade['target_price']) / ltp) * 100
            
            # Status emoji
            if pnl >= 0:
                emoji = "🟢"
            else:
                emoji = "🔴"
            
            log(f"{emoji} {symbol}: LTP ₹{ltp:.2f} ({pnl_pct:+.2f}%) | P&L ₹{pnl:,.0f} | SL:{dist_to_sl:.1f}% TGT:{dist_to_tgt:.1f}%")
            
            # Check if SL hit
            if sl_status == 'COMPLETE':
                log(f"❌ SL HIT: {symbol} @ ₹{sl_fill} - Cancelling target order")
                cancel_order(trade['target_order_id'])
                active_trades.remove(trade)
                log(f"   Final P&L: ₹{(sl_fill - trade['entry_price']) * trade['qty']:,.0f}")
                
            # Check if Target hit
            elif tgt_status == 'COMPLETE':
                log(f"✅ TARGET HIT: {symbol} @ ₹{tgt_fill} - Cancelling SL order")
                cancel_order(trade['sl_order_id'])
                active_trades.remove(trade)
                log(f"   Final P&L: ₹{(tgt_fill - trade['entry_price']) * trade['qty']:,.0f}")
            
            # Check if orders are cancelled/rejected
            elif sl_status in ['CANCELLED', 'REJECTED'] and tgt_status in ['CANCELLED', 'REJECTED']:
                log(f"⚠️ Both orders closed for {symbol}")
                active_trades.remove(trade)
        
        log("-" * 70)
        time.sleep(15)  # Check every 15 seconds
    
    # Final summary
    log("=" * 70)
    log("MONITOR ENDED - FINAL STATUS")
    log("=" * 70)
    
    positions = kite.positions()
    total_pnl = 0
    for p in positions.get('net', []):
        if p['quantity'] != 0:
            pnl = p['pnl']
            total_pnl += pnl
            emoji = "🟢" if pnl >= 0 else "🔴"
            log(f"{emoji} {p['tradingsymbol']}: Qty {p['quantity']} | LTP ₹{p['last_price']:.2f} | P&L ₹{pnl:,.2f}")
    
    log(f"TOTAL P&L: ₹{total_pnl:,.2f}")

if __name__ == "__main__":
    main()
