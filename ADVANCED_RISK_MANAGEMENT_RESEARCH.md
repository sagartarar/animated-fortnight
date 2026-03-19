# Advanced Risk Management Research for Combined Strategies

**Goal:** Improve returns by 20%+ over current 1.5% static risk per trade setup.

**Current Setup:** 1.5% risk per trade, static position sizing, fixed exits.

---

## Executive Summary

Advanced risk management can improve returns through:
1. **Dynamic position sizing** — Size up on high-conviction setups, size down in adverse conditions
2. **Advanced exits** — Partial exits, trailing stops, time-based refinement
3. **Drawdown protection** — Daily/consecutive loss limits, volatility adjustment
4. **Kelly/Optimal f** — Mathematical optimization of bet size

Expected impact: **15–25% improvement** in profit factor and risk-adjusted returns when combined.

---

## 1. Dynamic Position Sizing

### 1.1 Base Formula (ATR-Based)

Replace fixed risk with volatility-adjusted sizing:

```
Position Size = Risk Amount ÷ Stop Distance
Risk Amount = Equity × Risk% × Size_Multiplier
Stop Distance = ATR × Multiplier × Point_Value
```

**Implementation:**
```python
def atr_position_size(equity, entry, sl, atr, risk_pct=0.015, atr_mult=2.0):
    """ATR-based dynamic sizing - fewer shares when volatility is high"""
    risk_amount = equity * risk_pct
    stop_distance = atr * atr_mult  # or use actual SL distance
    risk_per_share = abs(entry - sl)
    if risk_per_share <= 0:
        return 0
    qty = int(risk_amount / risk_per_share)
    return max(1, qty)
```

**ATR Multiplier Guide:**
- 1.5 ATR: Tighter stops, more whipsaws
- 2.0 ATR: Balanced (recommended)
- 2.5–3.0 ATR: Wider stops, fewer false exits

---

### 1.2 Win Rate / Setup Confidence Multiplier

**Formula:** Scale position size by historical win rate of the specific setup type.

```
Confidence_Mult = 0.7 + 0.6 × (Win_Rate - 0.5)
# Clamp to [0.7, 1.3]
```

| Setup Win Rate | Confidence Mult | Action |
|---------------|-----------------|--------|
| 40% | 0.70 | Reduce size 30% |
| 50% | 1.00 | Base size |
| 60% | 1.10 | Increase 10% |
| 70% | 1.22 | Increase 22% |
| 80% | 1.30 | Cap at +30% |

**Implementation:**
```python
def confidence_mult(win_rate, min_mult=0.7, max_mult=1.3):
    """Higher win rate = larger size (capped)"""
    mult = 0.7 + 0.6 * (win_rate - 0.5)
    return max(min_mult, min(max_mult, mult))
```

**Per-setup tracking:** Maintain rolling win rate for each setup type (e.g., VWAP bounce, IB breakout, EMA crossover) over last 50–100 trades.

---

### 1.3 Volatility (ATR Regime) Adjustment

**Formula:** Reduce size when current ATR exceeds historical average.

```
Vol_Mult = min(1.0, Avg_ATR_20 / Current_ATR)
```

When `Current_ATR > 1.5 × Avg_ATR_20` → reduce size by 50%.

**Implementation (already in your backtest_rust):**
```rust
// ATR > 1.5x average → 50% size
if current_atr > avg_atr * 1.5 {
    multiplier *= 0.5;
}
```

**Refined formula:**
```
Vol_Mult = 1.0 / (1.0 + max(0, (Current_ATR / Avg_ATR - 1.0)))
# Smooth reduction: ATR 1.2x avg → 0.83, 1.5x → 0.67, 2x → 0.5
```

---

### 1.4 Equity Curve Feedback

**Rule:** When equity < N-period MA of equity, reduce position size.

```
Equity_MA = SMA(Equity_Curve, 20)
If Equity < Equity_MA: Size_Mult *= 0.5
```

**Alternative (dual MA crossover):**
- Fast MA (5) < Slow MA (20) → downtrend → reduce size 50%
- Fast MA > Slow MA → uptrend → full size

**Implementation (already in backtest_rust):**
```rust
if current_equity < equity_ma {
    multiplier *= 0.5;  // "BELOW_EQ_MA"
}
```

**Enhancement:** Use graduated reduction instead of binary:
```
Equity_Ratio = Current_Equity / Equity_MA
Size_Mult = 0.5 + 0.5 * min(1.0, Equity_Ratio)
# At 0.8 ratio → 0.9 mult, at 1.0 → 1.0, at 0.6 → 0.8
```

---

### 1.5 Multiple Strategy Confluence Multiplier

**Rule:** More confirming factors = higher conviction = larger size.

Your current score system (0–10+): Use it to scale size.

```
Confluence_Mult = 0.7 + 0.03 × Score  # For score 0–10
# Score 7 (MIN_SCORE) → 0.91
# Score 10 → 1.00
# Score 12 → 1.06 (cap at 1.2)
```

**Refined formula:**
```
Confluence_Mult = min(1.3, 0.6 + 0.05 × (Score - MIN_SCORE))
# Score 7 → 0.6, Score 9 → 0.7, Score 11 → 0.8, Score 15 → 1.0
```

**Implementation:**
```python
def confluence_mult(score, min_score=7, max_mult=1.3):
    return min(max_mult, 0.6 + 0.05 * (score - min_score))
```

---

### 1.6 Combined Dynamic Size Formula

**Final position size multiplier:**

```
Size_Mult = Base_Risk_Mult 
          × Confidence_Mult(Win_Rate) 
          × Vol_Mult(ATR_Ratio) 
          × Equity_Mult(Equity_vs_MA) 
          × Confluence_Mult(Score)
          × Drawdown_Mult(DD_Pct)
          × Consecutive_Loss_Mult
```

Each component is clamped to avoid over-leveraging. Typical range: **0.25 to 1.3**.

---

## 2. Advanced Exit Techniques

### 2.1 Partial Exits (50% at 1R, 50% at 2R)

**Structure:**
- **Target 1 (1R):** Close 50% of position
- **Target 2 (2R):** Close remaining 50%
- **Stop:** Move to breakeven after first partial

**Expectancy comparison (from research):**
- 50/50 at 1R and 3R: ~0.35R per trade, smoother equity
- Single exit at 2R: Similar expectancy but more variance
- Partial exits **increase win rate** (more trades end in profit) while **reducing variance**

**Implementation:**
```python
# On entry: split position into 2 lots
lot1_qty = total_qty // 2
lot2_qty = total_qty - lot1_qty

# Target 1: 1R profit
target1_price = entry + (entry - sl)  # For long; for short: entry - (sl - entry)
# Close lot1 at target1, move stop to breakeven for lot2

# Target 2: 2R profit  
target2_price = entry + 2 * (entry - sl)
# Close lot2 at target2 or trail
```

**Alternative splits (backtest to optimize):**
- 50% at 1R, 50% at 2R
- 50% at 1R, 50% at 3R
- 33% at 1R, 33% at 2R, 34% at 3R

---

### 2.2 Trailing Stops (Break-Even After 1R, Then ATR Trail)

**Stage 1:** Initial stop at entry - 1R (or ATR-based).

**Stage 2:** When price reaches +1R profit:
- Move stop to **breakeven** (entry price)

**Stage 3:** After breakeven, activate **ATR trailing stop:**
- **Long:** Stop = Highest_High_Since_Entry - (ATR × Trail_Mult)
- **Short:** Stop = Lowest_Low_Since_Entry + (ATR × Trail_Mult)

**ATR Trail Multiplier:** 1.5–2.0 typical. Tighter = more exits, wider = let winners run.

**Implementation:**
```python
def trailing_stop(entry, sl, atr, high_since_entry, low_since_entry, direction='long'):
    r = abs(entry - sl)
    # Breakeven trigger: price has exceeded entry + 1R
    if direction == 'long':
        current_stop = sl
        if high_since_entry >= entry + r:
            # Move to breakeven, then trail
            trail_stop = high_since_entry - 1.5 * atr
            current_stop = max(entry, trail_stop)
    else:  # short
        current_stop = sl
        if low_since_entry <= entry - r:
            trail_stop = low_since_entry + 1.5 * atr
            current_stop = min(entry, trail_stop)
    return current_stop
```

---

### 2.3 Time-Based Exit Refinement

**Current:** Exit all at 3:15 PM.

**Refinements:**

| Time | Rule | Rationale |
|------|------|-----------|
| 2:30 PM | Reduce target for new entries (shorter time to target) | Avoid chasing into close |
| 2:45 PM | No new entries | Insufficient time for setup |
| 3:00 PM | If in profit > 0.5R, consider early exit | Lock gains before volatility |
| 3:10 PM | Scale out 50% of any open position | Reduce gap risk |
| 3:15 PM | Mandatory exit remaining | Your current rule |

**Time-adjusted target formula:**
```
Minutes_Remaining = (15*60 + 15)*60 - Current_Time_Seconds  # Seconds to 3:15
Base_Target_R = 2.0
Time_Adj_Target = Base_Target_R * min(1.0, Minutes_Remaining / 120)
# If 60 min left → 1R target, 120 min → 2R
```

---

### 2.4 Multiple Targets Based on Confluence Score

**Rule:** Higher confluence = aim for larger targets (let winners run more).

| Score | Target 1 | Target 2 | % at T1 | % at T2 |
|-------|----------|----------|---------|---------|
| 7–8 | 1R | 1.5R | 50% | 50% |
| 9–10 | 1R | 2R | 50% | 50% |
| 11+ | 1R | 2.5R | 40% | 60% |

**Formula:**
```python
def targets_by_score(score):
    if score <= 8:
        return (1.0, 1.5, 0.5, 0.5)
    elif score <= 10:
        return (1.0, 2.0, 0.5, 0.5)
    else:
        return (1.0, 2.5, 0.4, 0.6)
```

---

## 3. Drawdown Protection

### 3.1 Daily Loss Limits

**Rule:** Stop trading when daily loss exceeds X% of capital.

| Level | % | Action |
|-------|---|--------|
| Conservative | 1–2% | Stop for day |
| Moderate | 3% | **Your current (backtest_rust)** |
| Aggressive | 5% | From RISK_PROTECTION_FRAMEWORK |

**Implementation:**
```python
if daily_pnl < -equity * 0.03:
    halt_trading("DAILY_LIMIT")
```

---

### 3.2 Consecutive Loss Limits

**Rule:** After N consecutive losses, reduce size or stop.

| Option | N | Action |
|--------|---|--------|
| A | 2 | Reduce size 50% |
| B | 3 | Reduce size 50% (**your current**) |
| C | 3 | Stop for day |
| D | 4 | Stop for day |

**Implementation (your current):**
```rust
if consecutive_losses >= 3 {
    multiplier *= 0.5;
}
```

**Enhancement:** After 3 losses, **stop for the day** (not just reduce):
```python
if consecutive_losses >= 3:
    halt_trading("CONSEC_LOSS")
```

---

### 3.3 Volatility-Based Risk Adjustment

**Rule:** When VIX or ATR regime is elevated, reduce risk per trade.

| VIX | ATR vs Avg | Risk % | Position Mult |
|-----|------------|--------|---------------|
| < 15 | < 1.0 | 1.5% | 1.0 |
| 15–20 | 1.0–1.2 | 1.25% | 0.83 |
| 20–25 | 1.2–1.5 | 1.0% | 0.67 |
| > 25 | > 1.5 | 0% | 0 (no trade) |

---

### 3.4 Drawdown Ladder (Already Implemented)

Your backtest_rust has:
- 5% DD → 90% size
- 10% DD → 75% size
- 15% DD → 50% size
- 20% DD → 25% size
- 25% DD → HALT

**Enhancement:** Add 3% tier for earlier response:
- 3% DD → 95% size

---

## 4. Kelly Criterion & Optimal f

### 4.1 Kelly Criterion

**Formula:**
```
f* = (p × b - q) / b
```
Where:
- `f*` = fraction of capital to risk per trade
- `p` = win rate (probability of win)
- `q` = 1 - p (probability of loss)
- `b` = odds = (average win) / (average loss) in R multiples

**Example:** Win rate 55%, avg win 1.5R, avg loss 1R:
- b = 1.5
- p = 0.55, q = 0.45
- f* = (0.55 × 1.5 - 0.45) / 1.5 = (0.825 - 0.45) / 1.5 = **0.25** (25%)

**Warning:** Full Kelly causes 50–70% drawdowns. Use **Half Kelly** or **Quarter Kelly**.

**Half Kelly:** f = 0.25 / 2 = 12.5% risk per trade  
**Quarter Kelly:** f = 0.25 / 4 = 6.25% risk per trade

**Implementation:**
```python
def kelly_fraction(win_rate, avg_win_r, avg_loss_r):
    """Returns optimal fraction of capital to risk. Use half or quarter."""
    b = avg_win_r / avg_loss_r
    p, q = win_rate, 1 - win_rate
    f = (p * b - q) / b
    return max(0.0, min(0.25, f))  # Cap at 25%
```

---

### 4.2 Optimal f (Ralph Vince)

**Concept:** Maximizes geometric growth using full distribution of trade outcomes.

**Terminal Wealth Relative (TWR):**
```
TWR = ∏ (1 + f × Trade_Return_i / |Largest_Loss|)
```

**Process:**
1. Take last 100+ trades
2. Find largest losing trade (L)
3. For each f from 0.01 to 1.0, compute TWR
4. Optimal f = f that maximizes TWR

**Position size from optimal f:**
```
Position_Size = (Account × Optimal_f) / |Largest_Loss_Per_Share|
```

**Warning:** Optimal f often suggests 20–40% risk → severe drawdowns. Use **fractional optimal f** (e.g., 0.25 × optimal f).

**Implementation (simplified):**
```python
def optimal_f(trade_returns):
    """trade_returns: list of P&L per share (or per contract)"""
    largest_loss = abs(min(trade_returns))
    if largest_loss <= 0:
        return 0.0
    best_f, best_twr = 0.0, 0.0
    for f in [x/100 for x in range(1, 25)]:  # 0.01 to 0.24
        twr = 1.0
        for r in trade_returns:
            twr *= (1 + f * r / largest_loss)
        if twr > best_twr:
            best_twr, best_f = twr, f
    return best_f
```

---

### 4.3 Kelly-Lite (Practical Hybrid)

Combine fixed fractional with Kelly-inspired scaling:

```
Base_Risk = 1.5%
Kelly_Mult = min(1.5, Kelly_f / 0.015)  # Scale up if Kelly says you can
Final_Risk = Base_Risk × min(Kelly_Mult, 1.5)
```

Never exceed 2.25% (1.5 × 1.5) risk per trade.

---

## 5. Implementation Priority & Expected Impact

### High Impact, Lower Effort
1. **Partial exits (50% at 1R, 50% at 2R)** — Smooth equity, similar expectancy
2. **Confluence-based target extension** — Let high-score trades run further
3. **Trailing stop after 1R** — Cut losers, protect winners
4. **Consecutive loss = stop for day** — Avoid revenge trading

### High Impact, Medium Effort
5. **ATR-based position sizing** — Already have ATR; add to sizing
6. **Win rate by setup** — Track and size up on best setups
7. **Equity curve filter** — Already in backtest; ensure live use
8. **Time-based exit refinement** — No new entries after 2:45, scale out before 3:15

### Medium Impact, Higher Effort
9. **Kelly/Optimal f** — Requires 100+ trades per setup for stability
10. **Confluence multiplier on size** — Integrate score into position sizing

---

## 6. Specific Formulas Summary

| Component | Formula |
|-----------|---------|
| ATR Position Size | `Qty = (Equity × Risk%) / (ATR × Mult × Point_Value)` |
| Confidence Mult | `0.7 + 0.6 × (WinRate - 0.5)` clamped [0.7, 1.3] |
| Vol Mult | `1 / (1 + max(0, ATR_Ratio - 1))` |
| Confluence Mult | `0.6 + 0.05 × (Score - MIN_SCORE)` capped 1.3 |
| Kelly | `f = (p×b - q) / b`; use Half Kelly |
| Partial Exit | 50% at 1R, 50% at 2R; move SL to BE after 1R |
| Trail Stop | BE at 1R; then `High - 1.5×ATR` (long) |
| Daily Limit | Halt if `daily_loss > 3%` of equity |
| Consecutive | Halt or 50% size after 3 losses |

---

## 7. References

- Multi-Indicator Dynamic Adaptive Position Sizing (Medium)
- Kelly Criterion & Optimal f (ProTraderDashboard, QuantifiedStrategies)
- Equity Curve Position Sizing (QuantifiedStrategies)
- Partial Close Strategies (QuantStrategy.io)
- ATR Trailing Stops (ProRealCode, ForexBee)
- Your existing: `backtest_rust`, `RISK_PROTECTION_FRAMEWORK.md`, `COMPREHENSIVE_STRATEGY_RESEARCH_SYNTHESIS.md`

---

*Last Updated: March 19, 2026*
