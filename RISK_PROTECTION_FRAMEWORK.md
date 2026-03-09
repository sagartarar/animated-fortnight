# Risk Protection Framework 🛡️

## Lessons from Market Crash (March 9, 2026)
- Nifty: -2.43% (-595 points)
- Bank Nifty: -3.62% (-2,089 points)
- 39 out of 40 stocks down
- Worst hit: Auto (-3.86%), Banking (-3.28%)

---

## 1. PRE-TRADE PROTECTION

### A. Market Health Check (Before Any Trade)
```
✅ Check BEFORE entering any position:

1. Index Trend
   - Nifty above 20-DMA? (bullish)
   - Nifty above 50-DMA? (healthy)
   - Nifty above 200-DMA? (long-term bullish)

2. Market Breadth
   - A/D ratio > 1.0? (healthy)
   - A/D ratio > 1.5? (strong)
   - If A/D < 0.5 → AVOID LONGS

3. Volatility Check
   - India VIX < 15: Low fear, safe to trade
   - India VIX 15-20: Normal, proceed with caution
   - India VIX 20-25: High fear, reduce position size
   - India VIX > 25: STAY OUT or hedge

4. Global Cues
   - US futures (S&P, Nasdaq)
   - Asian markets (SGX Nifty, Hang Seng)
   - Any major news/events?
```

### B. Position Sizing Rules
```
Capital: ₹1,00,000

Normal Conditions (VIX < 20):
- Max risk per trade: ₹2,000 (2%)
- Max position size: 60% of capital (₹60,000)
- Max concurrent trades: 3

High Volatility (VIX 20-25):
- Max risk per trade: ₹1,000 (1%)
- Max position size: 40% of capital (₹40,000)
- Max concurrent trades: 2

Extreme Volatility (VIX > 25):
- NO NEW TRADES
- Close existing positions
- Stay 100% cash
```

---

## 2. DURING-TRADE PROTECTION

### A. Mandatory Stop Loss Rules
```
EVERY trade MUST have:
1. Hard Stop Loss (placed with broker)
2. Maximum 2% risk per trade
3. Never move SL against your position

Stop Loss Placement:
- Intraday: 1-1.5% from entry
- Swing: 2-3% from entry
- Based on ATR: 1.5x ATR
```

### B. Trailing Stop Strategy
```
When position is profitable:

+1% profit → Move SL to breakeven
+2% profit → Trail SL to +1%
+3% profit → Trail SL to +1.5%

This locks in profits and protects from reversals.
```

### C. Time-Based Exits
```
Intraday Trades:
- Exit by 3:15 PM (15 mins before close)
- No overnight positions unless swing trade

Weekly:
- Book profits on Friday if holding uncertain positions
- Weekend gap risk is real (you saved yourself this way!)
```

---

## 3. PORTFOLIO PROTECTION

### A. Hedging Strategies
```
When holding long positions, consider:

1. Put Option Protection
   - Buy OTM put for 1-2% of position value
   - Protects against 5-10% crash
   
2. Index Hedge
   - If holding multiple longs, buy Nifty PUT
   - Ratio: 1 lot Nifty PUT per ₹5-7 lakh exposure

3. Pair Trading
   - Long strong stock + Short weak stock
   - Market neutral, profits from relative strength
```

### B. Diversification Rules
```
Never put all eggs in one basket:

- Max 30% in single sector
- Max 20% in single stock
- Spread across uncorrelated sectors

Today's example:
- Banking: -3.28%
- IT: -0.46%
If you had 50% Banking + 50% IT → Average loss: -1.87% (better than -3.28%)
```

### C. Cash Buffer
```
Always maintain cash buffer:

- Normal markets: 20-30% cash
- Uncertain markets: 40-50% cash
- High VIX (>25): 70-100% cash

Cash = Opportunity to buy at lower levels
```

---

## 4. EARLY WARNING SIGNALS 🚨

### A. Technical Warnings
```
EXIT or REDUCE positions when:

1. Index breaks below 20-DMA
2. Stock breaks below key support
3. RSI divergence (price up, RSI down)
4. Volume spike on down days
5. Failed breakout (bull trap)
```

### B. Sentiment Warnings
```
Be cautious when:

1. India VIX spikes >20%
2. FII selling >₹2000 Cr for 3+ days
3. Global markets crashing
4. Major event risk (elections, budget, Fed)
5. Weekend approaching with uncertainty
```

### C. Friday Rule (Your Success Story!)
```
THE FRIDAY RULE:

Before weekend:
1. Review all open positions
2. Book profits on uncertain trades
3. Keep only high-conviction positions
4. Set tight stop losses

WHY: Weekend gap risk can wipe out weeks of profits
```

---

## 5. KILL SWITCH RULES 🛑

### Automatic Trading Halt When:
```
1. Daily Loss > 5% of capital (₹5,000)
   → STOP trading for the day

2. Weekly Loss > 10% of capital (₹10,000)
   → STOP trading for the week
   → Review what went wrong

3. Monthly Loss > 15% of capital (₹15,000)
   → STOP trading for the month
   → Reassess strategy completely

4. India VIX > 30
   → NO NEW TRADES
   → Exit all speculative positions
```

---

## 6. CRASH PROTECTION CHECKLIST

### Daily Pre-Market (9:00 AM)
```
□ Check global markets (US, Asia)
□ Check SGX Nifty direction
□ Check India VIX level
□ Check any overnight news
□ Check FII/DII data from yesterday
□ Decide: TRADE / REDUCE SIZE / STAY OUT
```

### During Trading Hours
```
□ Monitor index trend (15-min chart)
□ Watch market breadth (advance/decline)
□ Track your positions vs SL levels
□ Be ready to exit if market breaks down
```

### End of Day (3:00 PM)
```
□ Review all open positions
□ Decide: Hold overnight or exit?
□ If Friday: Apply FRIDAY RULE
□ Set alerts for next day
```

---

## 7. RECOVERY STRATEGY (After a Crash)

### Don't Revenge Trade!
```
After a market crash:

Day 1 (Crash Day):
- DO NOT buy the dip immediately
- Market can fall further
- Wait and watch

Day 2-3:
- Look for stabilization
- Wait for VIX to cool down
- Small test positions only

Day 4-5:
- If market holds, gradually add
- Use 50% of normal position size

Week 2:
- If recovery confirmed, return to normal
```

### Best Crash Buying Signals
```
BUY after crash when:
1. VIX drops from peak
2. Index holds key support for 2+ days
3. A/D ratio improves (>1.0)
4. Bullish candle with volume
5. FII turning buyers
```

---

## 8. IMPLEMENTATION IN TRADING ENGINE

Add these checks to `kite_trading_engine.py`:

```python
def should_trade_today():
    """Pre-market check - run before any trading"""
    
    # Get VIX
    vix = get_india_vix()
    if vix > 25:
        return False, "VIX too high - stay out"
    
    # Get market breadth
    ad_ratio = get_advance_decline_ratio()
    if ad_ratio < 0.5:
        return False, "Market breadth weak - avoid longs"
    
    # Check index trend
    nifty_above_20dma = check_nifty_vs_dma(20)
    if not nifty_above_20dma:
        return False, "Nifty below 20-DMA - cautious"
    
    # Check daily loss
    daily_pnl = get_daily_pnl()
    if daily_pnl < -1500:
        return False, "Daily loss limit hit - stop trading"
    
    return True, "All checks passed"
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────┐
│           PROTECTION QUICK REFERENCE                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  VIX < 15  → Trade normally (2% risk)              │
│  VIX 15-20 → Reduce size (1.5% risk)               │
│  VIX 20-25 → Minimal trades (1% risk)              │
│  VIX > 25  → NO TRADES, stay cash                  │
│                                                     │
│  A/D > 1.5 → Strong market, trade freely           │
│  A/D 1.0-1.5 → Normal, proceed                     │
│  A/D 0.5-1.0 → Weak, reduce exposure               │
│  A/D < 0.5 → Very weak, AVOID LONGS                │
│                                                     │
│  Friday → Book uncertain positions                  │
│  Daily loss > 5% → Stop trading                    │
│  Never risk > 2% per trade                         │
│                                                     │
│  REMEMBER: Capital preservation > Profits          │
└─────────────────────────────────────────────────────┘
```

---

*Last Updated: March 9, 2026*
*Triggered by: Market crash -2.43%*
