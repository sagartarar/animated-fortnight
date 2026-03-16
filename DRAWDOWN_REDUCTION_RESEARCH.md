# Drawdown Reduction Techniques — Research & Implementable Rules

Research summary for reducing maximum drawdown while maintaining profitability. Each section includes **specific, codeable thresholds** for backtesting.

---

## 1. Position Sizing Adjustments During Drawdowns

### Core Concept
Reduce exposure as equity declines from peak. Never increase size during drawdown; only increase when equity reaches new highs.

### Implementable Rules

| Method | Formula / Rule | Thresholds |
|--------|----------------|------------|
| **Fixed Percentage Reduction** | Reduce position size by X% when equity drops Y% from peak | Equity -10% → reduce size 20%; Equity -15% → reduce 40% |
| **Dynamic Equity-Based** | `position_size = base_size × (current_equity / peak_equity)` | Automatic scaling; no manual thresholds |
| **Drawdown Ladder** | Tiered reductions at milestones | 5% DD → -10% size; 10% DD → -25% size; 15% DD → -50% size; 20% DD → halt |
| **Consecutive Loss Rule** | After N consecutive losses, cut risk by 50% | N = 3–4; resume full size only at new equity high |

### Code Pseudocode

```python
# Drawdown ladder
def get_position_multiplier(current_equity, peak_equity):
    dd_pct = (peak_equity - current_equity) / peak_equity * 100
    if dd_pct >= 20: return 0      # Halt
    if dd_pct >= 15: return 0.50   # 50% size
    if dd_pct >= 10: return 0.75   # 75% size
    if dd_pct >= 5:  return 0.90   # 90% size
    return 1.0

# Consecutive loss rule
def get_consecutive_loss_multiplier(consecutive_losses):
    if consecutive_losses >= 4: return 0.5
    if consecutive_losses >= 3: return 0.75
    return 1.0
```

### Key Principle
**Martingale avoidance**: Never increase position size after losses. Only increase when equity makes a new high.

---

## 2. Equity Curve Trading (Pause When Below MA)

### Core Concept
When equity curve crosses below its moving average, pause trading until it crosses back above. Protects during losing streaks.

### Implementable Rules

| Parameter | Typical Range | Recommended for Backtest |
|-----------|---------------|--------------------------|
| MA period | 20–50 bars | 20 (daily) or 30 (intraday sessions) |
| Crossover logic | Equity < EMA(equity) → pause | Use closing equity vs EMA |
| Hysteresis | Optional buffer to avoid whipsaw | ±0.5% buffer zone |

### Code Pseudocode

```python
def should_trade_equity_filter(equity_curve, period=20):
    """Equity curve EMA filter: pause when equity < EMA(equity)"""
    ema = equity_curve.ewm(span=period, adjust=False).mean()
    return equity_curve.iloc[-1] > ema.iloc[-1]

# Alternative: scale size instead of pause
def get_equity_filter_multiplier(equity_curve, period=20):
    if equity_curve.iloc[-1] > ema.iloc[-1]:
        return 1.0
    return 0  # or 0.5 for reduced size
```

### Caveat
Research shows equity-curve MA filters can **degrade** performance in some strategies (lower win rate, lower net profit). Backtest both: full pause vs. reduced size vs. no filter.

---

## 3. Volatility-Based Position Sizing

### Core Concept
Reduce position size when volatility (VIX, ATR, or realized vol) is elevated.

### VIX-Based Framework (US; adapt for India VIX)

| VIX Level | Action | Position Multiplier |
|-----------|--------|---------------------|
| < 15 | Full size | 1.0 |
| 15–25 | Normal | 1.0 |
| 25–35 | Elevated | 0.5–0.75 |
| > 35 | Crisis | 0.25 or halt |

### India VIX (NSE) — Suggested Thresholds

| India VIX | Action | Multiplier |
|-----------|--------|------------|
| < 12 | Full size | 1.0 |
| 12–18 | Normal | 1.0 |
| 18–25 | Elevated | 0.75 |
| 25–30 | High | 0.5 |
| > 30 | Crisis | 0.25 or halt |

### ATR-Based (Asset-Specific)

```python
# Target constant risk; higher ATR = smaller size
target_vol = 0.01  # 1% risk
position_size = (capital * target_vol) / (atr * multiplier)
# Or: scale down when ATR > 1.2 * ATR_20d_avg
def get_vol_multiplier(current_atr, avg_atr_20d):
    if current_atr > 1.5 * avg_atr_20d: return 0.5
    if current_atr > 1.2 * avg_atr_20d: return 0.75
    return 1.0
```

### Code Pseudocode

```python
def get_vix_multiplier(vix):
    if vix > 30: return 0.25
    if vix > 25: return 0.50
    if vix > 18: return 0.75
    return 1.0
```

---

## 4. Daily / Weekly / Monthly Loss Limits

### Core Concept
Hard stops that halt trading when loss limits are hit. Prevents emotional spiral and capital erosion.

### Implementable Thresholds

| Timeframe | % of Equity | Typical Range | Suggested |
|-----------|-------------|---------------|-----------|
| **Daily** | 0.5–3% | 1.5–3% common | 2% (conservative), 3% (aggressive) |
| **Weekly** | 2–6% | 3–5% common | 4–5% |
| **Monthly** | 6–10% | Elder: 6% | 6% (Elder), 8% (firms) |

### R-Multiple Method
If you risk 1% per trade: 3R daily limit ≈ 3% max daily loss.

### Code Pseudocode

```python
DAILY_LOSS_LIMIT_PCT = 0.025   # 2.5%
WEEKLY_LOSS_LIMIT_PCT = 0.05   # 5%
MONTHLY_LOSS_LIMIT_PCT = 0.06  # 6% (Elder)

def check_daily_limit(daily_pnl, starting_equity):
    return (daily_pnl / starting_equity) < -DAILY_LOSS_LIMIT_PCT

def check_weekly_limit(weekly_pnl, week_start_equity):
    return (weekly_pnl / week_start_equity) < -WEEKLY_LOSS_LIMIT_PCT

def check_monthly_limit(monthly_pnl, month_start_equity):
    return (monthly_pnl / month_start_equity) < -MONTHLY_LOSS_LIMIT_PCT
```

### Consecutive Loss Rule (Already in Your Rules)
After 2 consecutive losses → stop for the day. Research supports 3–4; your 2 is conservative (good).

---

## 5. Correlation-Based Position Limits

### Core Concept
Highly correlated positions act as one concentrated bet. Limit effective exposure when correlations rise.

### Key Thresholds

| Correlation | Interpretation | Action |
|-------------|----------------|--------|
| > 0.6 | Diversification eroded | Treat as single exposure; reduce total size |
| 0.2–0.6 | Partial diversification | Scale position by 1 - correlation |
| < 0.2 | Good diversification | Full size |

### Effective N (Number of Independent Bets)

```
Effective N ≈ 1 / (avg_correlation²)
```
Example: 6 positions with 0.82 avg correlation → Effective N ≈ 1.5 (not 6).

### Implementation

```python
def get_correlation_adjusted_size(base_size, correlation_matrix, position_indices):
    """Reduce size when adding correlated positions"""
    avg_corr = correlation_matrix.iloc[position_indices, position_indices].values.mean()
    if avg_corr > 0.6: return base_size * 0.5   # Treat as concentrated
    if avg_corr > 0.4: return base_size * 0.75
    return base_size

# Portfolio heat limit: 5-10% total exposure
MAX_PORTFOLIO_HEAT_PCT = 0.08  # 8%
```

### Rolling Windows
Use 30-, 90-, 252-day rolling correlation for short/medium/long-term regime detection.

---

## 6. Time-Based Filters (Seasonal / Calendar)

### Core Concept
Avoid or reduce size during historically weak periods.

### Months to Avoid / Reduce (US/Global Data)

| Period | Action | Rationale |
|--------|--------|-----------|
| **September** | Reduce 50% or avoid | Historical underperformance |
| **May–August** | Reduce 25% | "Sell in May" / summer doldrums |
| **November–December** | Full size | Santa rally |
| **April** | Full size | Historically strong |

### India-Specific
Validate with Nifty/India data; patterns may differ. Use your own backtest to find significant months.

### Code Pseudocode

```python
# Month filter (1=Jan, 12=Dec)
WEAK_MONTHS = [5, 6, 7, 8, 9]  # May–Sep
AVOID_MONTHS = [9]              # September only

def get_monthly_multiplier(month):
    if month in AVOID_MONTHS: return 0
    if month in WEAK_MONTHS: return 0.75
    return 1.0
```

### Day-of-Week
- Friday: reduce size or exit early (your rule: exit by 2 PM)
- Monday: often volatile; consider reduced size first hour

---

## 7. Dynamic Risk Adjustment Based on Recent Performance

### Core Concept
Adjust risk based on recent win rate, Sharpe, or drawdown. Reduce when performance deteriorates.

### Implementable Rules

| Metric | Window | Threshold | Action |
|--------|--------|-----------|--------|
| Rolling win rate | 20 trades | < 40% | Reduce size 50% |
| Rolling Sharpe | 20 days | < 0 | Reduce size 50% |
| Current drawdown | From peak | > 10% | Use drawdown ladder (Section 1) |
| Recovery factor | Rolling | > 3 | Reduce size (hard to recover) |

### Performance-Based Scaling

```python
def get_performance_multiplier(recent_trades, window=20):
    wins = sum(1 for t in recent_trades[-window:] if t['pnl'] > 0)
    win_rate = wins / min(window, len(recent_trades))
    if win_rate < 0.35: return 0.5
    if win_rate < 0.45: return 0.75
    return 1.0
```

### Conditional Capital Deployment
Only trade when structural conditions support edge (e.g., volatility regime, trend strength). Reduces unnecessary exposure during poor environments.

---

## Combined Implementation Checklist

For backtest integration, apply multipliers **multiplicatively**:

```python
final_multiplier = (
    drawdown_multiplier *
    equity_curve_multiplier *
    vix_multiplier *
    monthly_multiplier *
    performance_multiplier *
    correlation_multiplier
)
position_size = base_size * min(final_multiplier, 1.0)
```

### Suggested Priority Order for Your Strategy

1. **Daily loss limit** (2.5–3%) — already partially in place
2. **Drawdown ladder** (5/10/15/20% tiers)
3. **VIX/India VIX** multiplier (you have VIX>20 reduce)
4. **Consecutive loss** rule (you have 2; consider 3)
5. **Equity curve filter** — backtest; may hurt
6. **Monthly limit** (6%)
7. **Correlation** — if trading multiple stocks/sectors
8. **Seasonal** — validate on India data first

---

## References

- QuantifiedStrategies: Maximum Drawdown Position Sizing
- Quantfish: Reducing Position Sizing During Drawdowns
- P&L Ledger: Daily Loss Limits & Weekly Max Drawdown Rules
- Research Affiliates: Volatility Targeting
- Signal Pilot: Correlation & Position Limits
- QuantStrategy.io: Seasonal Filters

---

*Document created: March 16, 2026*
