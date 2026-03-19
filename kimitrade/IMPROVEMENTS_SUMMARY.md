# KIMITRADE Backtest Improvements - Agent Analysis Summary

## Critical Bug: RiskState Daily Limits Never Reset

**Issue:** `daily_pnl`, `weekly_pnl`, `monthly_pnl` accumulate over the entire backtest. After ~3 losing trades, the daily limit (3%) is hit and ALL further trades are blocked.

**Impact:** This is the PRIMARY cause of low trade counts (8, 15, 106 trades over 11 years).

**Fix Required:**
- Reset daily P&L when date changes
- Reset weekly P&L when week changes  
- Reset monthly P&L when month changes
- Process stocks by date (all stocks on day 1, then day 2), not stock-by-stock

## Parameter Optimization Recommendations

### Strategy 1: Intraday Momentum
| Parameter | Current | Recommended |
|-----------|---------|-------------|
| MIN_MOMENTUM_PCT | 0.10% | **0.08%** |
| Entry Window | 11:00-14:00 | Keep 11:00-14:00 |
| Bull vs Bear | Same | **Bull: 0.06%, Bear: 0.10%** |

### Strategy 2: VWAP + Ladder
| Parameter | Current | Recommended |
|-----------|---------|-------------|
| ADX Threshold | 20 | Keep 20 |
| VWAP Deviation | 0.3% | **0.35-0.40%** |

### Strategy 3: Regime-Switching
| Parameter | Current | Recommended |
|-----------|---------|-------------|
| Volatility High | 25% | **22%** |
| ADX Trend | >30/<20 | **>28/<18** |
| Unknown Regime | No trade | **Trade at 50% size** |

## Data Processing Bugs

### Bug 1: Date Parsing Incomplete
- Only handled `+05:30`, not negative offsets, ISO format, fractional seconds
- **Fix:** Robust parsing with format normalization

### Bug 2: Candles Not Sorted
- CSV may not be chronological
- **Fix:** Sort candles by datetime before computing indicators

### Bug 3: Missing Trade Exit Fallback
- If loop ends without exit condition, trade is dropped
- **Fix:** Use last candle in window as exit

## Exit Strategy Improvements

### Current Issues
- 62% exit at TIME_EXIT (no directional profit)
- 30-40% hit SL
- Only 5-17% hit target

### Recommended Improvements
1. **ATR Trailing Stop** (highest impact)
   - Replace fixed SL with 2× ATR trailing stop
   - Expected: -15-25% TIME_EXIT, +10-15% TARGET_HIT

2. **Ladder Exits**
   - Exit 25% at 0.5R, 25% at 1R, 25% at 1.5R
   - Trail remaining 25% with 2× ATR

3. **Volume Confirmation**
   - Require 1.5× average volume for entry
   - Expected: +10-15% win rate

## Implementation Priority

### P0 (Critical - Fix First)
1. Fix RiskState daily/weekly/monthly reset
2. Fix date parsing bugs
3. Sort candles by datetime

### P1 (High Impact)
4. Implement ATR trailing stops
5. Relax entry thresholds
6. Add missing exit fallback

### P2 (Optimization)
7. Implement ladder exits
8. Add volume confirmation
9. Optimize regime parameters

## Expected Results After Fixes

| Metric | Current | Expected After Fixes |
|--------|---------|---------------------|
| Total Trades | 129 (11 years) | 5000-10000 |
| Win Rate | 12-45% | 45-55% |
| Profit Factor | 0.16-0.88 | 1.2-1.5 |
| Max DD | 3-7% | 5-10% |
| TIME_EXIT % | 53-65% | 30-40% |

## Next Steps

1. **Immediate:** Fix RiskState bug (will 10x trade count)
2. **Short-term:** Implement ATR trailing stops
3. **Medium-term:** Add ladder exits
4. **Re-run backtest** and compare results
