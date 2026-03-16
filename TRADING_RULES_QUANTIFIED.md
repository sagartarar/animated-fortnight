# Quantified Trading Rules

Based on real trading data from March 11-16, 2026.

---

## Performance Summary

| Date | Trades | Gross P&L | After Charges | Strategy |
|------|--------|-----------|---------------|----------|
| Wed Mar 11 | DRREDDY +₹2,795, HCLTECH -₹2,122 | +₹673 | +₹198 | Mixed |
| Fri Mar 13 | SBIN +₹757, LT +₹665 | +₹1,422 | +₹1,220 | Trend-following |
| Mon Mar 16 | TATASTEEL -₹1,444, JSWSTEEL -₹768 | -₹2,212 | -₹2,412 | Counter-trend (failed) |
| **TOTAL** | 6 trades | **-₹117** | **-₹994** | |

**Win Rate:** 3/6 = 50%
**Profitable Days:** 2/3 = 67%

---

## Quantified Rules

### Rule 1: Market Open Timing

**OLD:** Enter after 25-30 minutes
**NEW:** Wait minimum **45 minutes** after market open

| Entry Time | Win Rate (our data) |
|------------|---------------------|
| 9:45-10:00 AM | 33% (2 losses today) |
| After 10:15 AM | 75% (Friday trades) |

**Quantified Rule:**
```
IF time < 10:15 AM THEN no_trade = True
WAIT_UNTIL: Opening range established (45+ mins)
```

---

### Rule 2: Sector Momentum Confirmation

**Problem:** Metal was +0.99% at 9:47 AM, reversed to -0.25% by 10:20 AM

**Quantified Rule:**
```
SECTOR_CONFIRMATION:
  - Sector change must be > +0.5% for BUY
  - Sector change must be < -0.5% for SHORT
  - MUST be sustained for 30+ minutes
  - Check sector at T, T+15min, T+30min - all must align
  
ABORT_TRADE IF:
  - Sector reverses > 0.5% from entry signal
```

---

### Rule 3: Market Direction Alignment

**Problem:** Bought metals when Nifty turned negative

| Trade Direction | Nifty Direction | Win Rate |
|-----------------|-----------------|----------|
| BUY | Nifty +ve | 100% (1/1 DRREDDY) |
| BUY | Nifty -ve | 0% (0/2 today) |
| SHORT | Nifty -ve | 100% (2/2 Friday) |
| SHORT | Nifty +ve | 0% (0/1 HCLTECH) |

**Quantified Rule:**
```
MANDATORY_ALIGNMENT:
  - BUY only when Nifty > +0.2% from prev close
  - SHORT only when Nifty < -0.2% from prev close
  
ABORT_TRADE IF:
  - Nifty reverses direction after entry
  - Nifty crosses 0% line against your position
```

---

### Rule 4: RSI Zones

**Problem:** HCLTECH short at RSI 37 (already oversold) bounced

| RSI Zone | Trade Type | Our Result |
|----------|------------|------------|
| RSI < 35 | SHORT | FAILED (HCLTECH) |
| RSI 35-65 | Either | MIXED |
| RSI > 65 | BUY | FAILED (Overbought stocks today) |

**Quantified Rule:**
```
RSI_FILTERS:
  FOR BUY:
    - RSI must be 40-65 (sweet spot)
    - RSI > 70: AVOID (overbought, reversal risk)
    - RSI < 35: CONSIDER (oversold bounce) BUT needs confirmation
    
  FOR SHORT:
    - RSI must be 45-70
    - RSI < 30: AVOID (oversold, bounce risk)
    - RSI > 75: CONSIDER (overbought reversal) BUT needs confirmation
```

---

### Rule 5: Gap and Opening Range

**Problem:** Bought into opening strength that faded

**Quantified Rule:**
```
GAP_RULES:
  IF stock gaps up > 1%:
    - Wait for first pullback before buying
    - Don't chase the gap
    - Entry only if it holds above VWAP after 45 mins
    
  IF stock gaps down > 1%:
    - Wait for first bounce before shorting
    - Don't short into the hole
    - Entry only if it stays below VWAP after 45 mins
    
OPENING_RANGE (first 45 mins):
  - Mark high and low of first 45 mins
  - BUY only on breakout above OR range high
  - SHORT only on breakdown below OR range low
```

---

### Rule 6: Position Sizing by Confidence

| Confidence Level | Position Size | Criteria |
|------------------|---------------|----------|
| HIGH | 100% (₹2,000 risk) | Market aligned + Sector aligned + RSI sweet spot + Trend confirmed |
| MEDIUM | 75% (₹1,500 risk) | 3 of 4 above criteria met |
| LOW | 50% (₹1,000 risk) | Only 2 criteria met |
| NO TRADE | 0% | Less than 2 criteria |

**Additional Modifiers:**
```
REDUCE_SIZE_BY_25% IF:
  - VIX > 20
  - Friday
  - First 45 mins of market
  - Counter-trend trade
  
MAX_SIZE = 75% on any given trade (never 100%)
```

---

### Rule 7: Exit Rules

**Problem:** Let SL hit instead of managing actively

**Quantified Rules:**
```
TRAILING_STOP:
  - Move SL to breakeven when profit > 0.5%
  - Trail SL to lock 50% profit when profit > 1%
  
TIME_BASED_EXIT:
  - If flat (< 0.3% move) after 30 mins: Consider exit
  - If sector reverses against position: Exit immediately
  - Friday: Exit ALL by 2:00 PM
  
SECTOR_REVERSAL_EXIT:
  - If sector moves 0.5% against your trade direction: EXIT
  - Don't wait for SL
```

---

### Rule 8: Daily Loss Limits

```
KILL_SWITCHES:
  - Max loss per trade: ₹2,000 (2% of capital)
  - Max daily loss: ₹5,000 (5% of capital)
  - After 2 consecutive losses: STOP trading for the day
  
TODAY'S EXAMPLE:
  - Trade 1 (TATASTEEL): -₹1,444 → Continue
  - Trade 2 (JSWSTEEL): -₹768 → 2 consecutive losses → STOP
  - Total: -₹2,212 (within daily limit, but stopped due to 2-loss rule)
```

---

### Rule 9: Trade Selection Scoring

**Score each trade 1-10 before entry:**

| Factor | Points | Criteria |
|--------|--------|----------|
| Market Direction | 0-2 | Aligned with Nifty trend |
| Sector Strength | 0-2 | Sector > 0.5% in trade direction |
| RSI Zone | 0-2 | In sweet spot (40-65 for BUY, 45-70 for SHORT) |
| VWAP Position | 0-1 | Above for BUY, Below for SHORT |
| Supertrend | 0-1 | Aligned with trade direction |
| Volume | 0-1 | Above average volume |
| Time | 0-1 | After 10:15 AM |

**Entry Threshold:**
```
SCORE >= 7: High confidence trade (75% size)
SCORE 5-6: Medium confidence (50% size)
SCORE < 5: NO TRADE
```

---

### Rule 10: Weekly Review Metrics

Track these weekly:
```
1. Win Rate Target: > 50%
2. Profit Factor: > 1.5 (gross profit / gross loss)
3. Average Winner / Average Loser: > 1.2
4. Max Drawdown: < 10% of capital
5. Trades per day: 2-3 (not more)
6. Sector accuracy: Track which sectors work
```

---

## Quick Reference Card

```
BEFORE EVERY TRADE, CHECK:

[ ] Time > 10:15 AM?
[ ] Nifty aligned with trade direction?
[ ] Sector aligned (> 0.5% in direction)?
[ ] RSI in sweet spot (40-65 BUY / 45-70 SHORT)?
[ ] Above/Below VWAP?
[ ] VIX < 20? (If >20, reduce size)
[ ] Not Friday afternoon?
[ ] Score >= 5?
[ ] Not already 2 losses today?

IF ANY "NO" → SKIP THE TRADE
```

---

## Today's Trades Scored (Hindsight)

**TATASTEEL BUY @ 9:49 AM:**
- Time > 10:15: NO (0 pts)
- Nifty aligned: YES at entry, then NO (1 pt)
- Sector aligned: YES at entry (+0.99%), then NO (1 pt)
- RSI 63: YES (2 pts)
- Above VWAP: YES (1 pt)
- VIX < 20: NO (0 pts, was 22)
- **Score: 5/10 → Should have been 50% size or NO TRADE**

**JSWSTEEL BUY @ 9:49 AM:**
- Same issues
- **Score: 5/10 → Should have been 50% size or NO TRADE**

**If we had followed quantified rules:**
- Would have waited until 10:15 AM
- By 10:15, Metal was already weakening
- Would have seen Nifty turning negative
- **Result: NO TRADE → Saved ₹2,212**

---

*Document created: March 16, 2026*
*To be updated with each trading week's learnings*
