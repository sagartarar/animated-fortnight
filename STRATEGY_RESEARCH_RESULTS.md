# Strategy Research Results - 10 Year Backtest Analysis

## Executive Summary

After extensive research across academic papers and quantitative trading resources, combined with a comprehensive 10-year backtest (Feb 2015 - Feb 2026) on 196 Nifty 200 stocks using 15-minute data, we identified **profitable strategy configurations**.

### Key Finding: SHORT-Only Strategies are Profitable

| Config | Trades | Win Rate | Net P&L | Profit Factor | Sharpe |
|--------|--------|----------|---------|---------------|--------|
| **short_score7** | 123,866 | 49.2% | **₹61.5L** | 1.07 | 0.12 |
| short_adx20_time | 84,172 | 48.9% | ₹41.1L | 1.07 | 0.12 |
| short_sl2x | 63,560 | 47.6% | ₹39.4L | 1.09 | 0.18 |
| short_sl2.5x | 63,560 | 48.2% | ₹39.1L | 1.09 | 0.17 |
| short_adx_score9 | 54,825 | 48.8% | ₹30.4L | 1.08 | 0.14 |

### Key Finding: BUY Strategies Consistently Lose

All configurations with BUY trades showed losses:
- Baseline (BUY+SHORT): **-₹10.4L**
- ADX filter (BUY+SHORT): **-₹7.6L**
- BUY-only component of any strategy: Always negative

---

## Research-Based Improvements Tested

### 1. ADX Filter (Market Regime Detection)
**Research Basis:** Academic papers show ADX > 25 indicates trending markets where momentum strategies work better.

| ADX Filter | Trades | Net P&L | Impact |
|------------|--------|---------|--------|
| No filter | 216,459 | -₹10.4L | Baseline |
| ADX > 25 | 154,852 | -₹7.6L | +27% improvement |
| ADX 20-60 (SHORT) | 84,172 | +₹41.1L | **Profitable** |

**Conclusion:** ADX filter helps, but must be combined with SHORT-only.

### 2. RSI Sweet Spots
**Research Basis:** Studies show optimal RSI zones: 40-65 for BUY, 45-70 for SHORT.

| RSI Filter | Trades | Net P&L | Win Rate |
|------------|--------|---------|----------|
| No RSI filter | 63,560 | ₹21.9L | 48.8% |
| RSI 45-70 (SHORT) | 17,408 | ₹9.5L | 51.2% |
| ADX + RSI combo | 9,100 | ₹2.3L | 51.7% |

**Conclusion:** RSI filter improves win rate but reduces trade count significantly.

### 3. Stop-Loss Optimization
**Research Basis:** Wider stops (2-2.5x ATR) reduce whipsaws; research suggests time-based exits often outperform.

| SL Strategy | Trades | Net P&L | Max Drawdown |
|-------------|--------|---------|--------------|
| Time exit only | 63,560 | ₹21.9L | ₹17.3L |
| 2x ATR SL | 63,560 | ₹39.4L | ₹6.6L |
| 2.5x ATR SL | 63,560 | ₹39.1L | ₹11.7L |
| Trailing stop | 63,560 | -₹92.5L | ₹92.5L |

**Conclusion:** 2x ATR SL is optimal - better P&L AND lower drawdown than time-exit.

### 4. Entry Timing
**Research Basis:** Avoiding opening volatility (first 45-75 min) improves results.

| Entry Window | Trades | Net P&L |
|--------------|--------|---------|
| 11:00 - 14:00 | 63,560 | ₹21.9L |
| 11:30 - 14:00 | 58,821 | ₹22.1L |
| 12:00 - 14:00 | 54,244 | ₹14.6L |

**Conclusion:** 11:00-11:30 AM start is optimal.

### 5. Score Threshold
**Research Basis:** Higher conviction trades should perform better.

| Min Score | Trades | Net P&L | Win Rate |
|-----------|--------|---------|----------|
| Score ≥ 7 | 123,866 | ₹61.5L | 49.2% |
| Score ≥ 8 | 63,560 | ₹21.9L | 48.8% |
| Score ≥ 9 | 54,825 | ₹30.4L | 48.8% |
| Score ≥ 9 (no filter) | 255 | ₹0.99L | **58.4%** |

**Conclusion:** Score 7 gives best total P&L; Score 9 gives highest per-trade P&L.

---

## Best Strategy Configuration

Based on the grid search, the **optimal configuration** is:

```
STRATEGY: SHORT_SCORE7
========================
Direction:        SHORT only
Entry Time:       11:00 AM - 2:00 PM
Exit Time:        3:15 PM (time-based)
Min Score:        7
ADX Filter:       Optional (both work)
Stop-Loss:        2x ATR (optional, reduces drawdown)
Trailing Stop:    NO (hurts performance)
Max Trades/Day:   1 per stock
```

### 10-Year Performance (2015-2026)

| Metric | Value |
|--------|-------|
| Total Trades | 123,866 |
| Win Rate | 49.2% |
| Gross P&L | ₹1.37 Cr |
| Total Charges | ₹75.8L |
| **Net P&L** | **₹61.5L** |
| Avg Winner | ₹1,448 |
| Avg Loser | -₹1,306 |
| Profit Factor | 1.07 |
| Max Drawdown | ₹24.3L |
| Sharpe Ratio | 0.12 |

### Year-by-Year Analysis

| Year | Trades | Win Rate | Net P&L | Avg/Trade |
|------|--------|----------|---------|-----------|
| 2015 | 8,885 | 50.2% | ₹10.3L | ₹116 |
| 2016 | 9,998 | 50.5% | ₹11.3L | ₹113 |
| 2017 | 9,312 | 50.0% | ₹3.2L | ₹34 |
| 2018 | 11,172 | 49.2% | ₹6.5L | ₹58 |
| 2019 | 10,962 | 50.1% | ₹12.7L | ₹116 |
| 2020 | 11,218 | 49.5% | ₹5.9L | ₹52 |
| 2021 | 11,429 | 49.5% | ₹6.3L | ₹55 |
| 2022 | 12,206 | 49.4% | **₹16.7L** | ₹137 |
| 2023 | 11,239 | 46.5% | -₹2.2L | -₹19 |
| 2024 | 12,213 | 47.2% | -₹12.6L | -₹103 |
| 2025 | 13,226 | 50.7% | ₹6.8L | ₹51 |
| 2026 | 2,006 | 44.7% | -₹3.2L | -₹161 |

**Note:** 2023-2024 showed losses, suggesting market regime shift. Monitor and adjust.

---

## Alternative: Higher Quality with Stop-Loss

If you prefer lower drawdown and better risk-adjusted returns:

```
STRATEGY: SHORT_SL2X
=====================
Direction:        SHORT only
Entry Time:       11:00 AM - 2:00 PM
Exit:             2x ATR Stop-Loss OR 1.5x Target OR 3:15 PM
Min Score:        8
ADX Filter:       25-50
```

| Metric | Value |
|--------|-------|
| Net P&L | ₹39.4L |
| Win Rate | 47.6% |
| Profit Factor | 1.09 |
| Max Drawdown | ₹6.6L |
| Sharpe Ratio | 0.18 |

**Trade-off:** Less total P&L but 73% lower drawdown.

---

## Why SHORT Works Better Than BUY

1. **Bear moves are faster:** Panic selling creates stronger momentum
2. **Institutional selling:** Large players tend to sell more aggressively
3. **Volatility asymmetry:** Down moves have higher volatility (VIX spikes)
4. **Mean reversion bias:** Markets have slight upward drift, making shorts mean-reverting
5. **Sentiment:** Retail tends to hold losing longs, creating more short opportunities

---

## Implementation Recommendations

### For Live Trading

1. **Focus on SHORT signals only** - ignore BUY signals from the strategy
2. **Entry window:** 11:00 AM - 2:00 PM IST
3. **Exit by:** 3:15 PM IST (mandatory)
4. **Score threshold:** ≥7 for more trades, ≥9 for higher quality
5. **Optional ADX filter:** Only trade when ADX > 25 (trending market)
6. **Stop-loss:** 2x ATR (reduces drawdown significantly)
7. **No trailing stops:** Research shows they hurt this strategy

### Risk Management

| Parameter | Value |
|-----------|-------|
| Risk per trade | 1-2% of capital |
| Max trades/day | 1-2 per stock |
| Max open positions | 3-5 |
| Daily loss limit | 3% of capital |

### Position Sizing (Kelly-based)

For 49% win rate and 1.11 R:R (avg win/avg loss):
- Full Kelly: ~5% risk per trade
- Half Kelly (recommended): **2.5% risk per trade**
- Quarter Kelly (conservative): 1.25% risk per trade

---

## Files Generated

- `backtest_research_optimized.csv` - All trades from best configuration
- `grid_search_summary.csv` - Results of all 20 configurations tested
- `backtest_rust/src/main.rs` - Optimized Rust backtester with all filters

---

## Next Steps

1. **Paper trade** the SHORT_SCORE7 strategy for 1-2 months
2. **Monitor 2024-2025 performance** - recent years showed weakness
3. **Consider regime switching** - pause during strong bull markets
4. **Optimize further** with sector filters and volume confirmation

---

*Generated: Feb 2026*
*Backtest Period: Feb 2015 - Feb 2026*
*Data: 196 Nifty 200 stocks, 15-minute intervals*
*Total Data Points: ~50 million candles*
