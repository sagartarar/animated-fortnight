# Intraday F&O Trading Strategies for Indian Markets — Research Document

**Focus:** Defined-risk strategies suitable for NSE/BSE options  
**Portfolio Reference:** ₹2,00,000 (2 Lakh)  
**Sources:** NSE-focused research, Indian trading platforms, backtest data

---

## Executive Summary

This document consolidates intraday options (F&O) strategies suitable for Indian markets, with emphasis on **defined risk profiles**. Each strategy includes Greeks considerations, entry criteria (IV rank, PCR, etc.), strike selection, adjustment rules, max profit/loss scenarios, and capital allocation for a ₹2L portfolio.

---

## 1. Iron Condor — Intraday Adjustments

### Strategy Overview
Iron Condor is a **neutral, defined-risk** strategy combining a Bull Put Spread and Bear Call Spread. You sell OTM calls and puts while buying further OTM options as hedges. Win rate: **70–80%**.

### Option Greeks Considerations
| Greek | Target / Behavior |
|-------|-------------------|
| **Delta** | Near zero (−0.05 to +0.05) — market-direction neutral |
| **Theta** | Positive (+₹200–500/day per lot) — time decay works in your favor |
| **Vega** | Negative — enter when IV is high, exit when it drops |
| **Gamma** | Negative — risk increases near short strikes; consider early exit |

### Entry Criteria
- **IV Rank:** High (above 50%) — options expensive, favors premium selling
- **Market:** Range-bound for 3–5 days; VIX < 20 ideal
- **Avoid:** Expiry week, strong trends, major news, VIX > 25

### Strike Selection Methodology
- **Nifty:** 300–600 point range between sold strikes; 400–600 typical
- **Bank Nifty:** 600–1,200 point range between sold strikes
- **Delta-based:**
  - Conservative: 0.15–0.20 delta (≈85% win rate, lower premium)
  - Moderate: 0.25–0.30 delta (≈70–75% win rate)
  - Aggressive: 0.35–0.40 delta (≈60–65% win rate, higher premium)
- **Example (Nifty @ 22,000):** Sell 22,200 CE / Buy 22,300 CE; Sell 21,800 PE / Buy 21,700 PE

### Adjustment Rules
1. **Roll the untested side:** Close winning spread, sell new spread at better strike to collect more premium.
2. **Roll tested side out:** Move breached spread to next expiry with wider strikes when expecting reversion.
3. **Convert to Iron Fly:** Bring short strikes together when tested hard (higher risk).
4. **Exit:** When loss ≥ 2× premium collected or strong trend forms.

**Adjustment trigger:** When losing leg premium ≥ 2× winning leg premium.

### Max Profit / Max Loss
- **Max Profit:** Net premium received × lot size
- **Max Loss:** (Spread width − Net credit) × lot size  
  Example: 100-point spread, ₹50 credit, 50 qty → Max loss = (100 − 50) × 50 = **₹2,500**

### Capital for ₹2L Portfolio
- Max risk per trade: 5% = ₹10,000  
- Iron Condor max loss: ~₹2,500 per lot  
- **Max lots:** ₹10,000 / ₹2,500 = **4 lots**  
- Margin ≈ max loss; hedged strategies need ₹40,000–₹60,000 per position.

---

## 2. Straddle / Strangle — Volatility Plays

### 2A. Short Straddle (Sell Volatility)

**Structure:** Sell ATM call + ATM put, same strike and expiry.

| Aspect | Details |
|--------|---------|
| **Greeks** | Delta near 0; Theta positive; Vega negative; Gamma negative (risk near expiry) |
| **Entry** | Low IV or post-event IV crush; range-bound expectation |
| **Strike** | ATM strike closest to spot |
| **Max Profit** | Total premium received |
| **Max Loss** | Unlimited (both sides) |
| **Risk** | **Undefined** — not recommended for strict defined-risk mandate |

### 2B. Long Strangle (Buy Volatility) — Defined Risk

**Structure:** Buy OTM call + OTM put, same expiry, different strikes.

#### Option Greeks Considerations
| Greek | Impact |
|-------|--------|
| **Delta** | Near zero at entry; one leg gains delta on large move |
| **Gamma** | Moderate; accelerates gains as price nears strikes |
| **Theta** | Negative — daily decay; avoid holding to expiry |
| **Vega** | Positive — rising IV boosts both premiums |

#### Entry Criteria
- **IV Rank:** Low (below 50%) — options cheap; buy before IV expansion
- **Events:** Pre-Budget, pre-RBI policy, results season, elections, Supreme Court decisions
- **Technical:** Breakout setups from narrow ranges; Bollinger Band squeeze

#### Strike Selection
- **OTM:** 200–300 points from spot for Nifty (e.g., Nifty 25,000 → 24,800 PE, 25,200 CE)
- **Premium:** Lower than straddle; needs larger move for breakeven
- **Breakeven:** Upper = Call strike + premium; Lower = Put strike − premium

#### Adjustment Rules
- **Exit on move:** Close on large move or target profit; avoid holding to expiry
- **IV crush:** Exit before event if IV likely to collapse
- **Weekly expiry:** Cheaper premium, faster realization

#### Max Profit / Max Loss
- **Max Loss:** Premium paid (defined)
- **Max Profit:** Unlimited in theory
- **Example:** ₹130 debit → Max loss ₹130/lot; breakeven 24,670 / 25,330

#### Capital for ₹2L Portfolio
- Max loss = premium paid; no margin for long options
- **Suggested:** Risk 2–3% per trade = ₹4,000–₹6,000; 1–2 lots typical

### 2C. Short Strangle (Sell Volatility) — Undefined Risk

- **Strikes:** 0.15–0.20 delta (≈85% probability of profit)
- **Premium:** Collect > 1.5% of spot
- **Risk:** Unlimited on both sides — **excluded** from defined-risk mandate

---

## 3. Covered Call — Intraday Writing

### Strategy Overview
Sell call on underlying you own. In India, covered calls are typically **positional** (1 month+), not intraday.

### Limitations for Intraday
- **LEAPS:** Not available on Indian exchanges
- **Holding:** Requires owning underlying; intraday covered call is uncommon
- **Margin:** Lower than naked call; capital tied in stock

### Greeks & Entry
- **Delta:** Long stock + short call; net positive but capped upside
- **Theta:** Positive from short call
- **Entry:** Neutral to bullish; low IV for lower premium
- **Strike:** OTM call (e.g., 1–2% above spot)

### Capital for ₹2L Portfolio
- Needs stock ownership; capital split between stock and margin
- **Practical:** With ₹2L, 1–2 stock positions with covered calls; not ideal for pure intraday

---

## 4. Put-Call Ratio (PCR) Based Entries

### PCR Basics
- **Formula:** PCR = Total Put OI ÷ Total Call OI (OI-based) or Put Volume ÷ Call Volume (volume-based)
- **Interpretation:** Use **seller’s perspective** — high PCR = put sellers confident = bullish; low PCR = call sellers dominant = bearish

### PCR Ranges (Nifty)
| PCR Range | Seller's View | Market Tendency |
|-----------|---------------|-----------------|
| > 1.3 | Very bullish | Often bounces / stays up |
| 1.0–1.3 | Bullish | Bullish bias |
| 0.7–1.0 | Neutral to bearish | Sideways |
| < 0.7 | Bearish | Often stalls / falls |

### Entry Criteria
- **OI PCR:** For positional bias (2–7 days)
- **Volume PCR:** For intraday timing (check every 30–60 min)
- **Extremes:** PCR > 1.5 → contrarian long; PCR < 0.6 → contrarian short
- **Divergence:** Price up + PCR up = caution; Price down + PCR down = hidden bullish

### 4-Checkpoint Intraday Monitoring
1. **9:30 AM** — Baseline (OI + volume PCR)
2. **11:00 AM** — Confirmation of morning bias
3. **1:00 PM** — Afternoon setup
4. **2:30 PM** — Pre-close positioning

**Caution:** Volume PCR unreliable after 1 PM on Tuesday expiry.

### PCR-Based Strategies (Defined Risk)
- **PCR + Long Strangle:** Enter long strangle when PCR extreme suggests big move
- **PCR + Iron Condor:** Enter iron condor when PCR 0.7–1.0 (neutral) and range-bound
- **PCR + Vertical Spreads:** Use PCR for direction; use defined-risk spreads for execution

### Capital for ₹2L Portfolio
PCR is an **entry filter**, not a standalone strategy. Use with defined-risk structures (iron condor, vertical spreads, long strangle).

---

## 5. Option Chain Analysis Strategies

### Key Metrics
| Metric | Use |
|--------|-----|
| **Open Interest (OI)** | High Call OI = resistance; High Put OI = support |
| **PCR** | Sentiment and timing |
| **Max Pain** | Level where most money is positioned |
| **IV** | Option richness; entry for sell vs buy strategies |
| **Greeks** | Delta, Gamma, Theta, Vega for risk and behavior |

### OI-Based Support/Resistance
- High OI at strike = likely reaction level
- Build strikes around these levels for iron condor, strangle, straddle

### Weekly Expiry (Nifty)
- **Expiry:** Tuesday (from Sep 2025)
- **Phases:** Friday positioning → Monday setup → Tuesday execution
- **Theta:** Critical for expiry timing

### Defined-Risk Application
- Use OI to choose strikes for iron condor (avoid max-pain zone if possible)
- Use PCR for entry timing
- Use IV rank for sell vs buy strategy choice

### Capital for ₹2L Portfolio
Option chain analysis is a **tool**, not a standalone strategy. Combine with iron condor, long strangle, or vertical spreads.

---

## 6. Delta Neutral — Intraday Scalping

### Concept
Delta neutral = net delta ≈ 0. Profit from theta decay, vega, or gamma scalping rather than direction.

### Implementation
- **Straddle/Strangle:** Sell both sides to collect premium; delta near 0
- **Iron Condor:** Delta near 0 by design
- **Gamma Scalping:** Rebalance delta by trading underlying as price moves

### Challenges in Indian Context
- **Liquidity:** Frequent rebalancing needs tight spreads
- **Costs:** Brokerage and slippage can erode scalping gains
- **Data:** Limited public backtests for delta-neutral intraday in India

### Greeks Focus
- **Delta:** Keep near 0; rebalance when it drifts
- **Gamma:** Manage acceleration risk near expiry
- **Theta:** Primary profit source for short premium
- **Vega:** Enter when IV high; exit when IV drops

### Practical Approach for ₹2L
- **Iron Condor** and **Short Straddle** are delta-neutral by design
- Avoid complex gamma scalping with ₹2L; focus on simpler premium-selling structures
- **Suggested:** Iron condor as primary delta-neutral strategy

### Capital for ₹2L Portfolio
- Same as Iron Condor: 4 lots max, ~₹2,500 max loss per lot
- Rebalancing adds cost; keep adjustments minimal

---

## Summary: Defined-Risk Strategies for ₹2L Portfolio

| Strategy | Defined Risk? | Max Loss | Capital/Trade | Recommended? |
|----------|---------------|----------|---------------|--------------|
| Iron Condor | Yes | Spread width − premium | ₹40K–₹60K | Yes |
| Long Strangle | Yes | Premium paid | ₹4K–₹6K risk | Yes |
| Short Straddle | No | Unlimited | — | No |
| Short Strangle | No | Unlimited | — | No |
| Covered Call | Yes (capped upside) | Stock decline | Stock + margin | Limited for intraday |
| PCR-Based | N/A (filter) | Depends on structure | — | Use with above |
| Option Chain | N/A (tool) | Depends on structure | — | Use with above |
| Delta Neutral (IC) | Yes | Same as Iron Condor | ₹40K–₹60K | Yes |

---

## Recommended Allocation (₹2L Portfolio)

1. **Primary:** Iron Condor — 2–4 lots, max 5% risk per trade
2. **Volatility events:** Long Strangle — 1–2 lots, 2–3% risk
3. **Filters:** PCR + Option Chain for entry timing and strike selection
4. **Buffer:** Keep 5–10% for brokerage, STT, fees, slippage
5. **Rule:** 50% of margin in cash/liquid; rest in approved collateral (per NSE)

---

## References & Sources

- Zerroday: Iron Condor Strategy Guide (Nifty & Bank Nifty)
- TalkOptions: Long Strangle, IV Rank/Percentile, Short Strangle
- Stoxra: PCR for Nifty Options Trading
- Zerodha Varsity: Strangle Strategy
- Algo Trading Labs, Trading Tick: NSE backtesting
- Samco, Krishnatradingcorner: Capital requirements for options selling
- NiftyBankNifty.com: Strike selection for strangle, iron condor, butterfly

---

*Document generated from web research. Not financial advice. Past performance does not guarantee future results. Trade at your own risk.*
