# Volatility-Based Intraday Trading Strategies — Research Summary

*Compiled from web research. Focus on strategies with proven backtest results.*

---

## 1. ATR Breakout Strategies

### Volatility Measurement Method
- **ATR (Average True Range)** — typically 14–20 period
- **ATR Ratio**: Short-term ATR (e.g., 4-period) / Long-term ATR (e.g., 30-period) to detect compression
- **Compression**: Current ATR < 80% of its rolling mean (e.g., 12-period)
- **Expansion trigger**: Current ATR exceeds rolling mean by configurable multiplier (e.g., 1.5×)

### Entry Criteria Based on Volatility
- **Long**: Price exceeds prior lookback high AND current ATR exceeds rolling mean by multiplier
- **Short**: Price closes below prior lookback low AND same ATR expansion condition
- **Alternative (XRPUSDT example)**: True Range > 1.5× 20-day ATR AND close breaks above previous 20-day high

### Position Sizing Based on Volatility
- Fixed % of capital per trade (e.g., 10%)
- ATR-based sizing: Risk per trade = (Account × Risk%) / (ATR × Multiplier)

### SL and Target Placement (ATR-Based)
- **Initial SL**: 1.0–3.0× ATR from entry
- **Trailing stop**: ATR-multiple trailing stop that widens with rising volatility, tightens with falling volatility
- **Time-based exit**: 7–15 bars (common for intraday)
- **Fixed TP/SL example**: TP +12%, SL −6%, or 25% drawdown cap

### Backtest Results with Different Volatility Regimes
| Source | Asset | Timeframe | Result |
|-------|-------|-----------|--------|
| PyQuantLab | Multi-asset | Various | Volatility filter + ATR trailing stop; rolling performance documented |
| AInvest | XRPUSDT | 6 months | −6.01% return, −8.66% annualized, 1 trade (hit SL) |
| Emma Kirsten | E-mini NQ | 60-min | "Winning" strategy; waits for rare compression, then ATR expansion trigger |

**Caveats**: Whipsaw losses in choppy markets; parameter sensitivity to instrument and timeframe.

### Best Volatility Indicators to Use
- ATR (14–20 period)
- ATR ratio (short/long)
- Rolling ATR mean for compression/expansion detection

---

## 2. Bollinger Band Expansion/Contraction (Squeeze) Trading

### Volatility Measurement Method
- **Bollinger Band width**: Difference between upper and lower bands (2 std dev from 20-period SMA)
- **Squeeze**: Band width contracts (volatility falls)
- **RSI of volatility bands**: 10-bar RSI of band width < 45 (Quantified Strategies variant)

### Entry Criteria Based on Volatility
- Squeeze detected when band width is at recent lows
- **Entry**: Close sets new 5-day high (bullish breakout)
- **Timeframe**: Weekly bars performed better than daily in backtests

### Position Sizing Based on Volatility
- Not explicitly documented in backtests; standard % risk per trade recommended

### SL and Target Placement (ATR-Based)
- Exit after fixed holding period (e.g., 20 weeks in Quantified Strategies test)
- ATR-based stops can be added: typically 2–3× ATR from entry

### Backtest Results with Different Volatility Regimes
| Source | Asset | Result |
|-------|-------|--------|
| Quantified Strategies | Multiple assets | **Poor performance across most assets** |
| Quantified Strategies | Pepsi (PEP) | 12.5% vs 14.8% buy-and-hold; 61% time in market |
| Quantified Strategies | Consumer staples | Best sector; limited success elsewhere |

**Conclusion**: Conceptually sound (contraction → expansion) but **does not reliably beat buy-and-hold** in broad backtests. Use with caution; consider sector-specific optimization.

### Best Volatility Indicators to Use
- Bollinger Band width
- RSI of band width (for squeeze confirmation)
- ATR for stop placement

---

## 3. India VIX Based Strategies (Volatility Forecasting)

### Volatility Measurement Method
- **India VIX**: NSE-derived implied volatility from Nifty 50 index options (30-day forward-looking)
- **Directional change**: ML/DL models forecast daily changes in India VIX
- **Sentiment proxies**: India VIX changes, advance-decline ratios, put-call OI, absolute returns, high-low range

### Entry Criteria Based on Volatility
- **Bullish**: Green candlestick + declining VIX → ~69.6% success rate
- **Bearish**: Red candlestick + rising VIX → ~39.8% success rate (less reliable)
- **15-day holding**: Minimum forecasting error in research
- Use VIX to time entries/exits and adjust equity exposure

### Position Sizing Based on Volatility
- Reduce equity exposure when VIX is high; increase when VIX is low
- Portfolio hedging via options when VIX spikes

### SL and Target Placement (ATR-Based)
- Not explicitly documented; standard ATR-based stops applicable

### Backtest Results with Different Volatility Regimes
| Source | Finding |
|--------|---------|
| IJFMR / Market Insights | Bullish (green candle + falling VIX): 69.6% success |
| IJFMR / Market Insights | Bearish (red candle + rising VIX): 39.8% success |
| IIMB / MDPI | ML/DL forecasting of India VIX direction |
| RePEc | Sentiment + IVIX improves options strategy performance; 15-day hold optimal |

**Note**: India VIX is best used as a **filter/regime indicator**, not a standalone entry signal.

### Best Volatility Indicators to Use
- India VIX (IVIX)
- Put-call OI ratio
- Advance-decline ratio
- High-low range, absolute returns

---

## 4. Volatility Squeeze Setups

### Volatility Measurement Method
- **Bollinger Band width** contraction
- **Keltner Channel** vs Bollinger Band convergence (Bollinger inside Keltner = squeeze)
- **ATR** for stop placement and volatility regime
- **Percent Rank** for squeeze identification

### Entry Criteria Based on Volatility
- Squeeze: BB width at recent lows (or BB inside KC)
- **Entry**: Price breaks above upper band (long) or below lower band (short)
- **ADX > 20** (or 12–25): Trend strength filter to avoid chop
- **OBV / ROC**: Volume or momentum confirmation

### Position Sizing Based on Volatility
- Fixed % (e.g., 10%) or ATR-based risk per trade

### SL and Target Placement (ATR-Based)
- **Initial SL**: 3.0× ATR from entry
- **Trailing stop**: Activated after 1.5× ATR profit; dynamic 1.5–9× ATR trailing
- Fixed TP/SL in points (TradingView variant)

### Backtest Results with Different Volatility Regimes
| Source | Framework | Notes |
|-------|-----------|------|
| PyQuantLab | Backtrader | Squeeze + breakout + ADX + ATR trailing; implementation focus |
| PyQuantLab | Backtrader | Multi-asset with OBV; 250+ strategy variations |
| PyQuantLab | Rolling backtest | ATR-based initial and trailing stops |

**Note**: Specific return/win-rate metrics not published; strategy design and implementation documented.

### Best Volatility Indicators to Use
- Bollinger Bands (squeeze)
- Keltner Channels (squeeze confirmation)
- ATR (stops, regime)
- ADX (trend filter)
- OBV, ROC (momentum)

---

## 5. Range Expansion After Contraction

### Volatility Measurement Method
- **ATR Ratio**: 4-period ATR / 30-period ATR
- **Signal**: Ratio < 0.7 = contraction (current volatility low vs historical)
- **NR4**: Narrow Range 4 — today’s range is smallest of last 4 days (Connors & Raschke)
- **NR7**: Narrow Range 7 — smallest range in 7 days

### Entry Criteria Based on Volatility
- Contraction: ATR ratio < 0.7 or NR4/NR7 pattern
- **Entry**: Price moves 0.7× 8-day ATR in trend direction; or next day breaks above NR4 high (long) / below NR4 low (short)
- **Filters**: MACD/ADX for trend; stochastic for overbought/oversold

### Position Sizing Based on Volatility
- ATR-based: Risk = (Account × Risk%) / (ATR × Multiplier)

### SL and Target Placement (ATR-Based)
- **SL**: Opposite side of NR4/NR7 bar, or 1–2× ATR
- **Trailing stop**: ATR-multiple trailing
- Stops widen in high volatility

### Backtest Results with Different Volatility Regimes
| Source | Strategy | Result |
|--------|----------|--------|
| CryptoDataDownload | NR4 breakout (BTCUSDT) | Python implementation; hypothetical results |
| TradingMarkets | ATR ratio contraction | "Range contraction is the interface to range expansion" |
| Quantified Strategies | Range Expansion Index (REI) | See REI section below |

**NR4/NR7**: Well-known from Connors & Raschke; best on daily; requires parameter tuning per market.

### Best Volatility Indicators to Use
- ATR (short/long ratio)
- NR4/NR7 patterns
- REI (Range Expansion Index)
- MACD, ADX (trend)

---

## 6. Keltner Channel Volatility Strategies

### Volatility Measurement Method
- **Keltner Channels**: Middle = 20-period EMA (or typical price); bands = ±2× ATR
- **Typical price**: (H + L + C) / 3
- Band width reflects ATR-based volatility

### Entry Criteria Based on Volatility
- **Mean reversion**: Buy when close < lower band; sell when close > center
- **Momentum**: Buy when close > upper band; sell when close < center
- **Intraday filters**: ATR above threshold (avoid flat markets); bar opens below upper band for longs

### Position Sizing Based on Volatility
- Channel width vs recent price → volatility; adjust size via ATR
- Fixed 10% in some backtests

### SL and Target Placement (ATR-Based)
- Exit at center line (typical price) or opposite band
- ATR-based stops: 1–2× ATR from entry

### Backtest Results with Different Volatility Regimes
| Source | Strategy | Asset | Result |
|--------|----------|-------|--------|
| Quantified Strategies | Mean reversion | SPY | **77% win rate**, 288 trades, 6.3% CAGR, 15% time in market, PF 2 |
| Quantified Strategies | Mean reversion | SPY | 6-day, 1.3 ATR: **80% win rate**; performance weaker after 2016 |
| Quantified Strategies | Momentum | SPY | 4.7% CAGR, 158 trades; 30-day, 1.3 ATR best |
| Quantified Strategies | Momentum | GLD | ~1% per trade; low win rate, few large winners (trend-following profile) |

**Optimal parameters**: 6–10 day period, 1–1.5 ATR for mean reversion; ~30 day, 1.3 ATR for momentum.

### Best Volatility Indicators to Use
- Keltner Channels (ATR-based)
- ATR (stops, sizing)
- RSI, MACD (confirmation)

---

## 7. Range Expansion Index (REI) — Bonus Strategy

### Volatility Measurement Method
- **REI**: Momentum oscillator from true high/low over N periods; range −100 to +100
- Overbought > +60, oversold < −60
- Duration: >6 bars in extreme zone = strong trend (avoid reversal)

### Entry Criteria
- Buy: Oversold (REI < −60) with duration ≤6 bars; combine with trend (e.g., MA)
- Sell: After N days (7 optimal for QQQ) or center-line exit

### Backtest Results (QQQ)
- **7-day hold**: 0.99% avg gain/trade, 62% win rate
- Avg winner 3.4%, avg loser 3.1%
- Max drawdown 49%
- 348 trades; works on QQQ, not on TLT/GLD

---

## Summary: Best Volatility Indicators by Use Case

| Use Case | Primary | Secondary |
|----------|--------|-----------|
| Breakout/expansion | ATR, ATR ratio | Bollinger width |
| Squeeze detection | Bollinger width, KC vs BB | ATR |
| Stop placement | ATR (1.5–3×) | — |
| Regime filter | India VIX, ATR | ADX |
| Mean reversion | Keltner Channels | REI, RSI |
| Range contraction | ATR ratio, NR4/NR7 | REI |

---

## References

- Quantified Strategies: Bollinger Squeeze, Keltner Channel, REI
- PyQuantLab (Medium): ATR breakout, volatility squeeze, Keltner strategies
- Emma Kirsten (Medium): ATR Volatility Compression
- Market Insights India, MDPI, IIMB, RePEc: India VIX
- CryptoDataDownload: NR4 strategy
- Connors & Raschke: Street Smarts (NR4/NR7)
- Thomas DeMark: Range Expansion Index
