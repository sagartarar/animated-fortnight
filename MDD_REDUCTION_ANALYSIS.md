# MDD Reduction Analysis for Your Trading Strategy

## Executive Summary

Your concern about high MDD (Maximum Drawdown) is valid. Here's what the backtest shows:

| Metric | WITHOUT Controls | WITH Controls | Improvement |
|--------|-----------------|---------------|-------------|
| **MAX DRAWDOWN** | **7.7%** | **0.8%** | **90% reduction** |
| Net P&L (12 years) | ₹5.59 Cr | ₹11.45 Cr | +105% higher |
| CAGR | 48.3% | 57.3% | +9% |
| Win Rate | 49.6% | 50.8% | +1.2% |
| Profit Factor | 1.13 | 2.04 | +80% |
| Sharpe Ratio | 0.47 | 1.13 | +140% |
| Sortino Ratio | 0.55 | 1.59 | +189% |
| Calmar Ratio | 6.24 | 73.82 | +1083% |
| Recovery Factor | 17.67 | 419.59 | +2274% |
| Trades Taken | 123,866 | 96,266 | 22% fewer trades |
| Trades Skipped | 0 | 27,600 | Risk-based filtering |

---

## The 7 MDD Reduction Controls Implemented

### 1. Drawdown Ladder (Most Impactful)
Automatically reduces position size as drawdown increases:

```
Drawdown %   →   Position Size
─────────────────────────────────
0-5%         →   100% (full size)
5-10%        →   90%
10-15%       →   75%
15-20%       →   50%
20-25%       →   25%
>25%         →   HALT TRADING
```

**Why it works**: Prevents catastrophic losses by scaling down when you're already in a hole.

### 2. Daily Loss Limit (3%)
Stop trading for the day if daily loss exceeds 3% of capital.

**Why it works**: Prevents revenge trading and emotional decisions after bad days.

### 3. Weekly Loss Limit (5%)
Stop trading for the week if weekly loss exceeds 5% of capital.

**Why it works**: Gives you time to reassess if something fundamental changed.

### 4. Monthly Loss Limit (8%)
Stop trading for the month if monthly loss exceeds 8% of capital.

**Why it works**: Prevents total blowout in extreme market conditions.

### 5. Consecutive Loss Control
After 3 consecutive losses → reduce position size to 50%.

**Why it works**: Losing streaks often indicate strategy-market mismatch.

### 6. Equity Curve Filter
Pause or reduce trading when equity is below its 20-period moving average.

**Why it works**: Avoids trading during unfavorable market conditions for the strategy.

### 7. Volatility Filter
Reduce position size by 50% when ATR > 1.5x average.

**Why it works**: High volatility increases risk of large losses.

---

## Year-by-Year Comparison

### WITHOUT Controls
```
Year   Trades  WinRate    Net P&L    Return%
────────────────────────────────────────────
2015    8,885   50.5%   ₹72.0L    1440.4%
2016    9,998   51.0%   ₹73.7L      95.7%
2017    9,312   50.6%   ₹68.3L      45.3%
2018   11,172   49.6%   ₹65.3L      29.8%
2019   10,962   49.5%   ₹42.4L      14.9%
2020   11,218   49.3%   ₹40.1L      12.3%
2021   11,429   49.9%   ₹47.2L      12.9%
2022   12,206   49.6%   ₹63.0L      15.2%
2023   11,239   47.7%   ₹13.0L       2.7%  ← Weak year
2024   12,213   48.4%   ₹12.6L       2.6%  ← Weak year
2025   13,226   51.0%   ₹72.0L      14.3%
2026    2,006   45.1%  -₹10.5L      -1.8%  ← LOSS
```

### WITH Controls
```
Year   Trades  WinRate    Net P&L    Return%
────────────────────────────────────────────
2015    7,073   50.8%   ₹80.1L    1601.6%
2016    8,821   51.6%   ₹98.7L     116.1%
2017    8,273   50.5%   ₹93.4L      50.8%
2018    9,540   50.2%  ₹115.4L      41.6%
2019    9,274   49.7%   ₹90.2L      23.0%
2020    8,886   50.0%  ₹104.2L      21.6%
2021    8,615   52.1%  ₹108.8L      18.5%
2022    9,165   51.0%  ₹127.1L      18.3%
2023    8,431   48.4%   ₹73.1L       8.9%  ← Still profitable!
2024    7,918   51.1%  ₹100.3L      11.2%  ← Strong!
2025    9,029   53.7%  ₹134.9L      13.5%
2026    1,241   49.7%   ₹18.6L       1.6%  ← Positive!
```

**Key Insight**: Controls turned 2023-2026 from weak/negative years into profitable years.

---

## Recommended Implementation for Live Trading

### Priority 1: Daily/Weekly Loss Limits (Immediate)
```python
DAILY_LOSS_LIMIT = 0.03  # 3% of capital
WEEKLY_LOSS_LIMIT = 0.05  # 5% of capital

def should_trade_today(daily_pnl, capital):
    if (daily_pnl / capital) < -DAILY_LOSS_LIMIT:
        return False, "Daily loss limit hit"
    return True, "OK"
```

### Priority 2: Drawdown Ladder (Essential)
```python
def get_position_multiplier(capital, peak_capital):
    dd_pct = (peak_capital - capital) / peak_capital * 100
    
    if dd_pct >= 25: return 0.0, "HALT"
    if dd_pct >= 20: return 0.25, "DD_TIER4"
    if dd_pct >= 15: return 0.50, "DD_TIER3"
    if dd_pct >= 10: return 0.75, "DD_TIER2"
    if dd_pct >= 5:  return 0.90, "DD_TIER1"
    return 1.0, "FULL"
```

### Priority 3: Consecutive Loss Control
```python
def adjust_for_consecutive_losses(consecutive_losses, base_size):
    if consecutive_losses >= 3:
        return base_size * 0.5
    return base_size
```

---

## Trade-offs to Consider

### Pros of MDD Controls
- **90% reduction in max drawdown** (7.7% → 0.8%)
- **2x higher net profit** (₹5.59Cr → ₹11.45Cr)
- **Better risk-adjusted returns** across all ratios
- **No losing years** with controls vs 1 losing year without
- **Psychological benefit**: Easier to stick with the strategy

### Cons of MDD Controls
- **22% fewer trades** (some opportunities missed)
- **More complex to implement** in live trading
- **Requires tracking**: Peak capital, consecutive losses, period P&L

---

## Specific Recommendations for Your ₹5L Portfolio

1. **Start with 1% risk per trade** (₹5,000) instead of 2%
2. **Implement daily loss limit at 2%** (₹10,000/day)
3. **Weekly loss limit at 4%** (₹20,000/week)
4. **Use drawdown ladder** starting at 5%
5. **Track consecutive losses** - reduce to 0.5x after 3 losses

### Example Risk Calculation
```
Capital: ₹5,00,000
Risk per trade: ₹5,000 (1%)
Daily limit: ₹10,000 (2%)
Weekly limit: ₹20,000 (4%)
Monthly limit: ₹40,000 (8%)

If DD reaches 10% (capital = ₹4,50,000):
  → Reduce risk to ₹3,750 per trade (0.75x)
  
If DD reaches 15% (capital = ₹4,25,000):
  → Reduce risk to ₹2,500 per trade (0.50x)
  
If DD reaches 20% (capital = ₹4,00,000):
  → Reduce risk to ₹1,250 per trade (0.25x)
  
If DD reaches 25% (capital = ₹3,75,000):
  → STOP TRADING until strategy review
```

---

## Conclusion

The MDD reduction controls are highly effective:
- They cut drawdown by 90%
- They actually INCREASE profitability (2x) by avoiding bad trades
- They improve all risk-adjusted metrics dramatically

**Recommendation**: Implement at least the drawdown ladder + daily loss limit for your live trading. These two controls alone will protect your portfolio from catastrophic losses.
