# Intraday Entry & Exit Strategy Research

## Executive Summary

Research on academic papers and trading strategies to address your current performance gaps:

| Current Issue | % of Trades | Target |
|---------------|-------------|--------|
| TIME_EXIT (3:15 PM) | 62% | 20–30% |
| SL_HIT | 30–40% | 25–35% |
| TARGET_HIT | 5–17% | 25–40% |

**Main finding:** Time-based exits dominate because fixed SL/target are too tight relative to intraday volatility. Dynamic exits (trailing stops, ATR-based, VWAP/ladder) can reduce TIME_EXIT by ~40–50% and improve target hit rate.

---

## 1. Better Exit Strategies

### 1.1 Trailing Stops vs Fixed SL

**Source:** Zarattini, Aziz, Barbon (2024) – SPY intraday momentum with dynamic trailing stops

| Approach | Win Rate | Drawdown | Notes |
|----------|----------|----------|-------|
| Fixed SL | 35–45% | Higher | Frequent stops in normal volatility |
| 2× ATR trailing | 40–50% | ~22% lower | Adapts to volatility |
| 3× ATR trailing | 45–55% | ~32% lower | Fewer false stops |

**Chandelier Exit (ATR-based trailing):**
- Long: Stop = Highest High (n periods) − ATR × Multiplier
- Short: Stop = Lowest Low (n periods) + ATR × Multiplier
- Default: 22-period lookback, 3× ATR multiplier
- Tighter in low volatility, wider in high volatility

**Implementation:** `Highest High since entry − (ATR × 2.0)` for longs; `Lowest Low since entry + (ATR × 2.0)` for shorts.

**Expected impact:** Reduce TIME_EXIT ~15–25%, increase TARGET_HIT ~10–15%.

---

### 1.2 ATR-Based Dynamic Stops

**Source:** TradingSchule.com, ETF Investment For Beginners; StrategyQuant; QuantStock

| Parameter | Calm | Trending | High Volatility |
|-----------|------|----------|-----------------|
| ATR multiplier | 1.5–2× | 2× | 2.5–3.5× |
| SL distance | Tighter | Standard | Wider |

**Multi-step profit taking:**
1. Move stop to breakeven after +1× ATR profit
2. Exit 30–50% at +2× ATR
3. Trail remainder with 2–3× ATR

**ADX-based adjustment:**
- ADX > 25 (trending): use 2× ATR
- ADX < 25 (ranging): use 2.5–3× ATR

**Expected impact:** 3× ATR vs fixed ~15% better performance, ~22% lower drawdown. SL_HIT can drop 5–10% if stops are widened.

---

### 1.3 VWAP + Ladder Exits

**Source:** Maróy (2025), "Improvements to Intraday Momentum Strategies" (SSRN 5095349)

| Exit Type | Sharpe | Annual Return | Drawdown |
|-----------|--------|---------------|----------|
| Time-based | ~1.5 | ~20% | Higher |
| VWAP + Ladder | >3.0 | >50% | <15% |

**Ladder logic:**
- 25% at 0.5R
- 25% at 1.0R
- 25% at 1.5R
- 25% trail with 2× ATR

**VWAP exit:** Exit when price crosses VWAP (momentum reversal).

**Expected impact:** Large reduction in TIME_EXIT (to ~20–30%), higher TARGET_HIT.

---

### 1.4 Support/Resistance Exits

**Source:** PriceActionNinja, QuantStock, Trading Setups Review

**Exit rules:**
- **Bounce:** Exit when price reverses at S/R
- **Breakout:** Exit when price breaks S/R with volume
- **Retest:** Exit after retest of broken level

**Intraday S/R:** Previous day H/L, round numbers, session pivots (9:15–9:45, 11:00–11:30, 2:30–3:15).

**Stop placement:** 15–25 pips beyond S/R level.

**Best window:** 9:30–11:00 AM for intraday S/R.

**Expected impact:** ~5–10% reduction in TIME_EXIT if used as secondary exit.

---

### 1.5 Volatility-Based Exit (Price Jump)

**Source:** Koegelenberg & van Vuuren (2023–2024), "Dynamic Price Jump Exit and Re-Entry Strategy"

- **Trigger:** Value-at-Risk confidence intervals for price jumps
- **Exit:** On extreme volatility
- **Re-entry:** Entropy-based re-entry

**Use case:** Protect against sudden reversals and large gaps.

**Expected impact:** Fewer catastrophic losses; less direct impact on TIME_EXIT %.

---

## 2. Entry Timing Improvements

### 2.1 Multiple Entry Windows vs Single Window

**Source:** Gao et al. (Market Intraday Momentum, JFE); Indian market research

**Current:** Single window (9:15–9:45, enter at 11:00).

**Suggested windows:**

| Window | Time | Rationale |
|--------|------|-----------|
| 1 | 9:15–9:45 → 11:00 | First 30 min predicts last 30 min |
| 2 | 11:00–11:30 | Confirmation of morning trend |
| 3 | 12:30–1:30 | Lunch pullback; avoid 11:30–1:30 low volume |

**Indian market:** Volatility is J-shaped (high early and late).

**Expected impact:** Avoid entries after 1:30 PM; ~5–10% reduction in TIME_EXIT.

---

### 2.2 Volume Confirmation for Entries

**Source:** Trade with the Pros, LuxAlgo, Breakout Strategy Guide

**Rules:**
- Volume ≥ 1.5× average for entry
- ≥ 3× average for strong conviction
- Breakout + volume spike = higher success

**Breakout validation:** Enter only when price closes beyond level with above-average volume.

**Inside bar:** Grade A = high mother bar + decreasing inside volume + expansion on breakout.

**Volume divergence:** Price breakout with decreasing volume → avoid.

**Expected impact:** ~10–15% improvement in win rate by filtering false breakouts.

---

### 2.3 Pullback vs Breakout Entries

**Source:** HeyGoTrade, TradeFundrr, MQL5 Blogs

| Type | Pros | Cons |
|------|------|------|
| **Breakout** | Captures strong moves | Higher false breakouts |
| **Pullback** | Tighter stops, better R:R | May miss moves |

**Pullback:** Enter after retracement in trend direction; ~15–25 pips beyond S/R level.

**Breakout:** Enter at breakout; ~30–40% false breakouts.

**Recommendation:** For intraday, prefer pullbacks; use volume confirmation for breakouts.

**Expected impact:** ~5–10% reduction in SL_HIT if using pullbacks.

---

## 3. Signal Confirmation

### 3.1 Multi-Timeframe Confirmation

**Source:** LuxAlgo, Signal Pilot, X3Algo

**Framework:** 1:4 or 1:5 ratio between timeframes.

| Style | Strategic | Tactical | Execution |
|-------|------------|----------|-----------|
| Day | Daily | 1-hour | 15-min |
| Scalp | 1-hour | 15-min | 5-min |

**Rule:** Trade only when higher timeframe trend aligns with lower timeframe setup.

**Effect:** 15-min EMA crossover: 45% → 62% win rate with 1-hour confirmation.

**Expected impact:** ~15–20% win rate improvement.

---

### 3.2 Index Correlation

**Source:** STRATEGY_CRITIQUE_AND_IMPROVEMENTS.md (internal)

**Rules:**
- Exit SHORT if Nifty turns positive > +0.3%
- Exit LONG if Nifty turns negative < −0.3%
- Use relative strength: stock vs Nifty

**Expected impact:** Fewer large reversal losses.

---

### 3.3 Sector Confirmation

**Source:** STRATEGY_CRITIQUE_AND_IMPROVEMENTS.md

**Rule:** Don’t short sectors outperforming Nifty.

**Expected impact:** ~5–10% fewer false signals.

---

## 4. Risk Management Per Trade

### 4.1 When to Widen vs Tighten SL

**Source:** ETF Investment For Beginners, TradingSchule.com

| Condition | Action |
|-----------|--------|
| ADX > 25 (trending) | Tighten SL (2× ATR) |
| ADX < 25 (ranging) | Widen SL (2.5–3× ATR) |
| High volatility | Widen SL (2.5–3.5× ATR) |
| Low volatility | Tighten SL (1.5–2× ATR) |

**Expected impact:** ~5–10% fewer false SL hits.

---

### 4.2 Position Sizing Based on Volatility

**Source:** QuantStrategy.io, Cliobra, Finaur

**Formula:**
```
Position Size = (Account Risk %) ÷ (ATR × ATR Multiple × Point Value)
```

**ATR multipliers:**
- 1× ATR: Scalping
- 1.5× ATR: Active day trading
- 2× ATR: Intraday (standard)
- 3× ATR: Swing

**Effect:** Same risk per trade across volatility regimes.

**Expected impact:** ~10–15% lower drawdown.

---

### 4.3 Time-Based Position Reduction

**Source:** Alibaba data (2021–2025): 3:00–3:45 PM ET optimal window

**Rules:**
- 3:00–3:15 PM: Best execution quality (99.87% of target, 94.2% fill)
- 2:58 PM: Place limit 0.05% below midpoint
- 3:00:00: Submit order

**Indian market:** 3:00–3:15 PM IST has higher volatility.

**Recommendation:** Reduce size or exit 50% before 3:00 PM for large positions.

**Expected impact:** ~5–10% better execution on TIME_EXIT.

---

## 5. India-Specific Patterns

**Source:** IIMA, SSRN 2255391, Journal of Asset Management

**Volume/volatility:**
- Intraday volatility: J-shaped (high early and late)
- Volume: U-shaped
- First 15 min: Highest volatility after weekends/holidays

**Momentum:** First and last half-hour returns depend on prior day and same-day first half-hour returns. Momentum works best on high volatility, high volume days.

**Overreaction:** Intraday overreaction can predict next-day returns.

---

## 6. Prioritized Implementation Roadmap

### Phase 1: Exit (High Impact)

| Change | Expected Impact | Effort |
|--------|-----------------|--------|
| Replace fixed SL with 2× ATR trailing | TIME_EXIT −15–25%, TARGET_HIT +10–15% | Medium |
| Add ladder exits (0.5R, 1R, 1.5R) | TIME_EXIT −20–30%, Sharpe +1.0+ | High |
| Move stop to breakeven at +1× ATR | SL_HIT −5–10% | Low |

### Phase 2: Entry (Medium Impact)

| Change | Expected Impact | Effort |
|--------|-----------------|--------|
| Volume confirmation (≥1.5× avg) | Win rate +10–15% | Low |
| Multi-timeframe (Daily → 1H → 15m) | Win rate +15–20% | Medium |
| Cut entries after 1:30 PM | TIME_EXIT −5–10% | Low |

### Phase 3: Risk (Medium Impact)

| Change | Expected Impact | Effort |
|--------|-----------------|--------|
| ATR-based position sizing | Drawdown −10–15% | Medium |
| ADX-based SL adjustment | SL_HIT −5–10% | Low |
| Nifty reversal exit (>±0.3%) | Avoid large losses | Low |

---

## 7. Key Papers & Sources

| Source | Type | Key Finding |
|-------|------|--------------|
| SSRN 5095349 (Maróy 2025) | Academic | VWAP + Ladder exits: Sharpe >3, returns >50% |
| SSRN 4374887 (Koegelenberg 2023) | Academic | Dynamic price jump exit for volatility |
| SPY Momentum (Zarattini 2024) | Academic | Trailing stops: 22% lower drawdown |
| Gao et al. (JFE) | Academic | First 30 min predicts last 30 min |
| IIMA (Indian market) | Academic | J-shaped volatility, U-shaped volume |
| TradingSchule.com | Practitioner | 3× ATR trailing: +15% vs fixed, −22% drawdown |
| QuantStrategy.io | Practitioner | ATR position sizing for consistency |

---

## 8. Summary Metrics

| Metric | Current | Target (Phase 1) | Target (Phase 2) |
|--------|---------|------------------|------------------|
| TIME_EXIT % | 62% | 35–45% | 20–30% |
| SL_HIT % | 30–40% | 25–35% | 20–30% |
| TARGET_HIT % | 5–17% | 20–30% | 25–40% |
| Win rate | ~35% | ~45% | ~55% |
| Sharpe | ~1.0 | ~2.0 | ~2.5+ |

**Top 3 changes:**
1. **2× ATR trailing stop** instead of fixed SL
2. **Ladder exits** (0.5R, 1R, 1.5R) + trail remainder
3. **Volume confirmation** (≥1.5× avg) at entry
