# IB Hybrid Strategy (Tier 2) - Design Document

## Objective
Improve IB Breakout performance by 20%+ using confluence filters and advanced exits.

## Base Strategy Performance (IB Breakout Alone)
- Win Rate: 48.0%
- Profit Factor: 1.06
- Net P&L: +₹12,90,7260 (+645%)
- Max Drawdown: 38.1%
- Time Exit: 77.8%

## Target Improvements
- Win Rate: 55%+
- Profit Factor: 1.30+
- Return Improvement: 20%+
- Max Drawdown: <30%

---

## Hybrid Filters (Entry Confluence)

### 1. VWAP Filter ✅
**Rule:** Only take longs when price > VWAP, shorts when price < VWAP
**Expected Impact:** Win rate +4-8%, fewer false breakouts

### 2. EMA Trend Filter ✅
**Rule:** Only take longs when EMA 9 > EMA 21, shorts when EMA 9 < EMA 21
**Expected Impact:** Win rate +2-6%, avoid counter-trend trades

### 3. ADX Filter ✅
**Rule:** Only trade when ADX > 22 (trending market)
**Expected Impact:** Win rate +2-5%, avoid choppy markets

### 4. IB Size Filter ✅
**Rule:** Only trade narrow/normal IB (IB range < 1.0 × ATR 14)
**Expected Impact:** Win rate +5-10%, better extension probability

### 5. Volume Filter (Enhanced) ✅
**Rule:** Volume > 2.0× 20-period average (up from 1.5×)
**Expected Impact:** Win rate +3-5%, confirm institutional participation

---

## Hybrid Exit Strategy

### Problem with Current Exit
- Target: 1.5× IB range (too far, only 8% hit rate)
- 77.8% of trades hit time exit

### Improved Exit Framework

#### 1. Closer Target ✅
**Change:** 1.5× IB → 1.0× IB range
**Expected:** Target hit rate 11% → 25-30%

#### 2. Trailing Stop ✅
**Rule:** 
- Breakeven when +1.0× ATR profit reached
- Then trail with 2.0× ATR from highest high (longs) / lowest low (shorts)
**Expected:** TIME_EXIT -15-25%, TARGET_HIT +10-15%

#### 3. VWAP Reversal Exit ✅
**Rule:** Exit position if price crosses VWAP in opposite direction (when in profit > 0.5R)
**Expected:** Protect profits on reversals

#### 4. Time-Based Refinement ✅
**Rule:** 
- No new entries after 1:30 PM
- Force exit all positions at 3:15 PM
**Expected:** Avoid late-day chop

---

## Confluence Scoring System

### Points System
```
IB Breakout Signal:        25 pts (required base)
VWAP Alignment:           +20 pts
EMA Trend Alignment:      +15 pts  
ADX > 22:                +10 pts
Volume > 2×:             +10 pts
Narrow IB (<0.5× ATR):    +10 pts
Normal IB (0.5-1× ATR):   +5 pts

MINIMUM TO ENTER: 50 pts (IB + at least 2 filters)
HIGH CONVICTION: 70+ pts (increase size by 50%)
```

### Dynamic Position Sizing
```
Base Risk: 1.5% of capital
Confluence Multiplier: 0.8 + (Score - 50) × 0.01
  - 50 pts → 0.8× (1.2% risk)
  - 60 pts → 0.9× (1.35% risk)  
  - 70 pts → 1.0× (1.5% risk) - normal
  - 80 pts → 1.1× (1.65% risk) - high conviction
  - 90 pts → 1.2× (1.8% risk) - max
```

---

## Expected Performance (Tier 2 Hybrid)

| Metric | IB Alone | IB Hybrid | Improvement |
|--------|----------|-----------|-------------|
| Win Rate | 48.0% | 58-65% | +20-35% |
| Profit Factor | 1.06 | 1.30-1.50 | +23-42% |
| Time Exit % | 77.8% | 40-50% | -35% |
| Target Hit % | 11.4% | 30-40% | +180% |
| Max Drawdown | 38.1% | 28-32% | -15-25% |
| Net Return | +645% | +775-900% | +20-40% |

---

## Implementation Checklist

### Indicators to Calculate
- [x] EMA 9 and EMA 21
- [x] VWAP (daily cumulative)
- [x] ADX (14 period)
- [x] ATR (14 period) for position sizing and trailing stops
- [x] Volume SMA (20 period)

### Entry Logic
- [x] IB Breakout detection (first 30 min)
- [x] VWAP alignment check
- [x] EMA trend alignment check
- [x] ADX > 22 check
- [x] Volume > 2× check
- [x] IB size classification
- [x] Confluence score calculation
- [x] Dynamic position sizing

### Exit Logic
- [x] 1.0× IB range target (reduced from 1.5×)
- [x] Trailing stop (breakeven at +1 ATR, then 2 ATR trail)
- [x] VWAP reversal exit (when profitable)
- [x] Hard stop at opposite IB side
- [x] Time exit at 3:15 PM
- [x] No entries after 1:30 PM

### Risk Management
- [x] Base 1.5% risk per trade
- [x] Confluence-based sizing multiplier
- [x] Max position 20% of capital
- [x] Drawdown ladder (reduce size at 5%, 10%, 15% DD)
- [x] Daily loss limit 3%
- [x] Consecutive loss pause (stop after 3 losses)

---

## Files to Reference
- Base IB logic: `/u/tarar/repos/ib_breakout_strategy/src/main.rs`
- VWAP calculation: `/u/tarar/repos/vwap_bounce_strategy/src/main.rs`
- EMA/ADX calculation: `/u/tarar/repos/ema_crossover_strategy/src/main.rs`
- Confluence ORB: `/u/tarar/repos/confluence_orb/backtest_rust/src/main.rs`

---

## Next Steps
1. Implement hybrid strategy in Rust
2. Run backtest on 196 stocks
3. Compare with base IB results
4. Validate 20%+ improvement target
