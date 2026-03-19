# Intraday Order Flow and Footprint Trading Strategies — Research Document

**Focus:** Order flow analysis, footprint charts, DOM trading, and alternatives when only OHLCV data is available  
**Sources:** Web research, trading platforms, order flow manuals, and internal documentation

---

## Executive Summary

Order flow trading examines the continuous stream of buy and sell orders to reveal market microstructure and participant behavior. It goes beyond traditional technical analysis by focusing on **real-time buying and selling pressures** that drive price movements. True order flow analysis requires **tick data** or **Level 2 (L2) order book data**; when only OHLCV is available, several approximation methods exist but with significant limitations.

---

## 1. Bid-Ask Imbalance and Delta Analysis

### Core Concepts

**Delta** measures the net difference between buy volume and sell volume within a period. It tracks **aggressive orders** — those crossing the spread (market orders that lift the ask or hit the bid).

- **Positive delta:** More aggressive buying than selling → buyers lifting the ask
- **Negative delta:** More aggressive selling than buying → sellers hitting the bid
- **Delta = Buy Volume − Sell Volume** (at each price level or aggregated per candle)

### Bid-Ask Imbalance

Bid-ask imbalances occur when one side of the order book dominates at certain levels:

- **Order book imbalance:** Bid volume significantly exceeds ask volume (or vice versa) at a level
- **Footprint imbalance:** At a price tick, one side (bid or ask) overwhelms the other by a threshold (e.g., 3:1 or 300%)
- **Diagonal calculation:** Imbalances compare ask volume at one level with bid volume one tick above (since best bid and ask are always at least one tick apart)

### Practical Application

- **Stacked imbalances:** 3–5 consecutive price levels with imbalances on the same side = strong initiative (buy or sell)
- **Imbalance thresholds:** 200% for backtesting (more setups), 300% for live trading (higher confidence)
- **Delta divergence:** Price rises but delta falls (or vice versa) → hidden weakness/strength, often precedes reversals

### Key Principle

> Price almost always follows the direction of delta in the long run. When they disagree, bet on the order flow — it represents real transactions, while price can be manipulated with smaller orders.

---

## 2. Absorption and Exhaustion Patterns

### Absorption

**Absorption** occurs when aggressive orders hit a price level but price refuses to move accordingly. A large passive participant is quietly absorbing the opposing pressure.

**Example:** Sellers cross the spread, delta goes negative, yet the candle makes no new lows — a large buyer is taking every sell order at that level.

**Critical distinction:** Absorption alone is **not** a trading signal. It only indicates a large player exists at that level. A passive participant can absorb for multiple candles and still lose if the aggressive side eventually overwhelms them.

### When Absorption Becomes Tradeable

Three elements must align (WickLabs, Bookmap):

1. **Location:** The level must have prior structural importance (previous highs/lows, established context)
2. **Large participant presence:** Confirmed via Traders Trying (TT), surge volume, and absorption
3. **Delta confirmation:** Delta flips show the aggressive side has given up and momentum is shifting

**Sequence for a tradeable absorption reversal:**
- Price rallies and creates structure (e.g., HH in a zone)
- Price pulls back and returns to that zone
- Large participants show up (TT, Surge, Absorption)
- Sellers hit aggressively, delta reaches -279 in a surge, price barely moves
- Delta flip: from -79 to +279
- Price moves away from the level

### Exhaustion

**Exhaustion** is a drop-off in follow-through volume from aggressive traders — lack of continuation in buying or selling pressure. It often marks key reversal or continuation zones.

- **Finished auction:** Zero (or near-zero) contracts on one side at the candle extreme → genuine exhaustion, high probability the extreme holds
- **Unfinished auction:** Significant volume on both sides at the extreme → market ran out of time, may return to that level
- **Exhaustion ratio:** Compare volume at last two ticks of a candle high/low. Ratio >10:1 suggests the move ran out of gas

### Distinguishing Absorption vs. Exhaustion

| Concept      | Meaning                                                                 |
|-------------|-------------------------------------------------------------------------|
| Absorption  | Passive interest holding a level; large player taking the other side    |
| Exhaustion  | Aggressive volume failing to continue; effort without price movement    |

Confirmation of reversals: CVD shifts, failed pushes, or aggressive volume flipping direction.

---

## 3. Iceberg Orders and Spoofing Detection

### Iceberg Orders

**Definition:** Large institutional positions split into smaller visible pieces to minimize market impact and avoid revealing full size. The algorithm repeatedly places small visible orders that refill immediately after execution.

**Observable signs:**
- Orders refilling at consistent size and regular intervals
- Price holding steady despite buying/selling pressure
- Orders at psychological levels, round numbers, or key support/resistance
- Repeated order appearances at the same price level

**Detection:** Volume surges past a smart threshold (e.g., volume moving average) while price stalls → suggests a massive hidden limit order absorbing market aggression.

### Spoofing

**Definition:** Placing large, non-bona fide limit orders (often far from best bid/offer) to mislead traders about liquidity depth or direction, then canceling them milliseconds before execution. Spoofers induce directional price moves to execute smaller real orders at favorable prices.

**Detection signals:**
- Massive volume spikes immediately followed by sharp volume drops and price reversals (faked liquidity pulled)
- Orders vanishing just before execution as price approaches (DOM observation)
- Orders that hold or get absorbed as price approaches → more reliable level validity

### Advanced Detection (AI/ML)

- **Level 3 market data** (order-level) for granular analysis
- **Features:** Price-level resilience, volume absorption, refill frequency, order imbalance patterns, trade clustering, queue depletion asymmetry
- **Models:** Logistic regression, random forests, neural networks for probability scores of manipulation

---

## 4. Cumulative Volume Delta (CVD) for Trend Confirmation

### What CVD Is

**Cumulative Volume Delta** = running total of (Buy Volume − Sell Volume) over time.

- **Per-period delta:** Buy Volume − Sell Volume
- **CVD:** Sum of all period deltas from session start (or reset point)

### How CVD Confirms Trends

| Condition                    | Interpretation                                  |
|-----------------------------|--------------------------------------------------|
| Price HH + CVD rising       | Uptrend confirmed; buyers powering the trend    |
| Price LL + CVD falling      | Downtrend confirmed; sellers dominant           |
| Price HH + CVD falling      | Bearish divergence; buying pressure weakening   |
| Price LL + CVD rising       | Bullish divergence; selling pressure losing steam|

### CVD vs. Traditional Volume

Standard volume bars show total activity without distinguishing who initiated trades. CVD separates buying from selling pressure by tracking **aggressive market orders** (those crossing the spread), revealing which side is forcing price movement.

### Divergences

- **Bullish divergence:** Price makes lower lows, CVD makes higher lows → selling pressure losing steam, potential reversal
- **Bearish divergence:** Price makes higher highs, CVD makes lower highs → buying pressure weakening, potential reversal

### Best Practices

- Use **1-minute charts** for most accurate divergence signals
- Works for futures (currencies, gold, oil, indices); **not** standard spot forex (no centralized order flow)
- CVD strategies used in futures, forex, crypto, and prop trading for divergence, absorption, exhaustion, and trend continuation

---

## 5. Volume Clusters at Key Levels

### Definition

**Volume clusters** are high-activity zones where large numbers of trades execute within a tight price range and short timeframe. They represent battles between buyers and sellers at critical turning points or during strong continuation moves.

### High-Volume Nodes (HVN)

- Specific price levels where the most trading occurred within a footprint
- Act as "gravity wells" with strong support/resistance properties
- Price often stalls or rotates around them before moving
- When clusters form at key levels → strong institutional commitment; both sides agreed price was "fair"

### Footprints of Aggressive Order Flow

Clusters leave "footprints" that the market remembers. When price revisits cluster levels:
- Market typically **stalls/bounces** or **breaks through with force**
- Stacked imbalance zones often act as support or resistance when retested

### Point of Control (POC)

- **POC:** Price level with the most volume within a candle (or session)
- **POC near top of bearish candle:** Sellers absorbed buying pressure at high prices
- **POC near bottom of bullish candle:** Buyers accumulated aggressively at lows
- **POC at candle extreme:** One side heavily committed; participants will defend on retest

### Value Area (VA)

- Price range containing ~70% of volume in a candle (or session)
- **VAH / VAL:** Value Area High / Low
- Value area = zone where market considered prices "fair"; can act as support/resistance on retest

---

## 6. Market Depth (DOM) Analysis for Entry Timing

### What is DOM?

**Depth of Market (DOM)** is a real-time, vertical display of the order book showing all pending limit orders at various price levels. Also called the "trading ladder." Displays Level 2 or Level 3 market data.

### DOM Components for Entry Timing

| Component              | Use                                                                 |
|------------------------|---------------------------------------------------------------------|
| **Rate of change**     | Orders appearing/vanishing in milliseconds = algo activity; stable orders = more reliable intent |
| **Behavior near levels** | Large orders hold or disappear as price approaches; vanishing = spoofing; absorbed = level valid |
| **Order imbalances**   | Bid volume >> ask volume = potential buying pressure, favorable long entry |
| **Liquidity assessment** | Deep markets need large orders to move price; shallow markets move easily → affects slippage |
| **Size clusters/walls** | Support/resistance zones; not all large orders are genuine (spoofing) |

### DOM + Times & Sales (Tape)

- **Tape:** Actual executed trades (Time & Sales)
- **DOM:** Declared intentions (pending orders)
- Trades at the ask = aggressive buying; trades at the bid = aggressive selling
- Combining DOM with tape confirms execution flow vs. declared intentions

### Internal Reference: STORM Order Book Adjusted

The firm's **Order Book Adjusted** (Confluence: STORM) provides:
- Size and bid/ask prices across brokers
- Market depth for estimating tradable quantity at different price levels
- Quote rating, cheapness, and transactable quote evaluation
- Use cases: "At what price can I trade X quantity?" and "How much can I trade at Y price?"

---

## 7. Order Flow Reversal Patterns

### Delta Divergence (Primary Pattern)

- **Bullish:** Price makes new lows, delta makes higher lows → buyers absorbing selling pressure
- **Bearish:** Price makes new highs, delta makes lower highs → false breakouts, institutional distribution

### V-Reversal Pattern

- **Setup:** Sharp initiation by sellers (strong negative delta, heavy bid volume) → immediately followed by sharp reversal (strong positive delta, heavy ask volume) in the next candle
- **Interpretation:** Complete shift in control; low of V-reversal becomes significant reference
- **Confirmation:** Retest of that low holding confirms reversal; break suggests buying conviction was weak

### Absorption Reversal (Tradeable Setup)

1. Price pushes into known support/resistance
2. Heavy aggressive selling (at support) or buying (at resistance), but price stalls
3. Delta strongly directional, price refuses to follow
4. Next candle: buying volume on ask (or selling on bid) — delta flip
5. **Entry:** Long when candle closes above absorption zone at support; short when below at resistance
6. **Stop:** Just beyond extreme of absorption candle
7. **Target:** VWAP/session mid-point; prior swing high/structure; typical 2:1 R:R

### Imbalance Breakout

- Price coils in tight range; volume builds
- Breakout candle with stacked imbalances on breakout side; delta spikes
- **Entry:** Close of breakout candle or pullback to breakout base
- **Stop:** Below range (longs), above range (shorts)
- **Note:** When stacked imbalances are strong, market order at close often better than waiting for limit at POC

### Trapped Traders

- Aggressive sellers at support, price refuses to drop → sellers trapped
- When they close shorts, they buy → burst of buying pressure
- **Signal:** Delta flips green at level where absorption was visible = clean entry

---

## 8. Data Requirements and OHLCV Alternatives

### Ideal Data: Tick and Level 2

| Data Type   | Content                                      | Use Case                          |
|-------------|-----------------------------------------------|-----------------------------------|
| **Tick data** | Every trade (price, size, timestamp, aggressor) | Delta, CVD, footprint, tape reading |
| **Level 2** | Full order book (bids/asks, sizes, levels)    | DOM, imbalance, absorption, iceberg/spoofing |
| **Level 3** | Order-level (queue position, order ID)        | Advanced spoofing/iceberg detection |

### Alternatives When Only OHLCV Is Available

True order flow (bid/ask split, delta, footprint) **cannot** be replicated from OHLCV. These are **proxies** that approximate some concepts:

#### 1. Delta Approximation (Price-Weighted)

```
delta_multiplier = (Close - Low) / (High - Low)   // avoid div by zero
Buy_Volume  = Total_Volume × delta_multiplier
Sell_Volume = Total_Volume × (1 - delta_multiplier)
Net_Delta   = Buy_Volume - Sell_Volume
```

**Logic:** If close is near high, more volume attributed to buyers; if close is near low, more to sellers.

#### 2. Polarity-Based Method

- If Open ≠ Close: relative position of close vs. open determines up/down volume
- If Open = Close: relationship between close and high/low assigns volume

#### 3. On Balance Volume (OBV)

- Add volume when close > prior close; subtract when close < prior close
- **Use:** Divergence detection, trend confirmation
- **Limitation:** Binary (up/down only); ignores magnitude of move

#### 4. Effort vs. Result (EvR)

- Compares volume magnitude to actual price movement
- High volume + small move → absorption-like behavior (approximation)

#### 5. Imbalance Ratio (Proxy)

- `(Buy_Vol - Sell_Vol) / Total_Vol` → ranges -1 to +1
- Derived from approximated buy/sell volume

#### 6. CVD (Proxy)

- Running sum of approximated delta over time
- Useful for divergence and trend confirmation even when delta is estimated

### Limitations of OHLCV Proxies

- **No true aggressor side:** Cannot know if trades were at bid or ask
- **No order book:** Cannot see absorption, icebergs, spoofing, or DOM structure
- **Coarse granularity:** Candle-level only; no tick-level precision
- **Best use:** Divergence detection, trend confirmation, rough delta/CVD for backtesting when tick data is unavailable

### Recommendation

For serious order flow strategies (absorption, imbalance, DOM timing), **tick data or L2 is required**. OHLCV proxies are suitable for:
- Backtesting trend/divergence ideas
- Screening when tick data is expensive
- Combining with other indicators (e.g., volume profile, VWAP) for context

---

## 9. Footprint Chart Display Modes

| Mode              | Content                                      | Best For                                      |
|-------------------|----------------------------------------------|-----------------------------------------------|
| **Bid/Ask Split** | Left = bid (sell) vol, Right = ask (buy) vol | Absorption, trapped traders, imbalances       |
| **Delta**         | Net difference per level (green/red)         | Quick scan of pressure across candles         |
| **Volume Profile**| Total volume per level as horizontal bar     | POC, value area, distribution shape           |

---

## 10. Software and Tools

- **Bookmap:** Combines historical TPO with real-time liquidity heatmaps; widely cited for order flow
- **TradingView:** Footprint, CVD, delta volume indicators (some require premium/data)
- **NinjaTrader, Sierra Chart:** Full footprint and DOM for futures
- **Freqtrade:** Order flow module for crypto (advanced)

---

## Summary: Key Takeaways

| Topic                    | Key Point                                                                 |
|--------------------------|----------------------------------------------------------------------------|
| **Delta**                | Net buy vs. sell volume; tracks aggressive orders; price follows delta     |
| **Absorption**           | Not a signal alone; needs location + large participant + delta flip        |
| **Exhaustion**           | Volume drop-off at extremes; finished vs. unfinished auctions             |
| **CVD**                  | Running delta sum; confirms trends; divergences signal reversals           |
| **Volume clusters**      | HVNs and POC act as support/resistance; stacked imbalances = commitment   |
| **DOM**                  | Order book visibility; rate of change, behavior near levels, imbalances  |
| **Reversal patterns**    | Delta divergence, V-reversal, absorption reversal, trapped traders        |
| **OHLCV alternatives**   | Delta proxy, OBV, EvR; useful for divergence/trend, not true order flow    |

---

## References

- GrandAlgo: Footprint Charts, CVD Explained
- WickLabs: Absorption signals, trapped traders
- Bookmap: Absorption/exhaustion, volume clusters
- QuantStrategy.io: Spoofing/iceberg detection, DOM
- TradingView: CVD, delta volume, footprint
- PickMyTrade, LiteFinance, AlgoStorm: Order flow guides
- PinescriptDeveloper: Delta volume approximation
- Internal: STORM Order Book Adjusted (Confluence)

---

*Document generated from web research. Not financial advice. Past performance does not guarantee future results. Trade at your own risk.*
