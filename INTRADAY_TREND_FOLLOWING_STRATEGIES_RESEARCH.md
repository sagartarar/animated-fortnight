# Proven Trend Following Strategies for Intraday Trading

**Research compiled from verifiable backtest sources**  
*Focus: ADX, Supertrend, EMA Crossover, Parabolic SAR, Ichimoku Cloud, Donchian Channel*

---

## 1. ADX Trend Following Strategy

### Trend Detection Rules
- **ADX > 20** (or 25 for stronger filter): Indicates trending market; below 20 = no-trade zone
- **Bullish pattern**: ADX crosses above 20 from below + **+DI > -DI** and diverging (green line rising, red line falling)
- **Bearish pattern**: ADX crosses above 20 from below + **-DI > +DI** and diverging (red line rising, green line falling)
- ADX sloping upward = trend strengthening; sloping down = trend weakening (not reversal)

### Entry Trigger Conditions
- **Long**: ADX crosses above 20, +DI above -DI, +DI rising while -DI falling, price above 50-day MA
- **Short**: ADX crosses above 20, -DI above +DI, -DI rising while +DI falling, price below 50-day MA
- **Filters**: Avoid when RSI < 30 (oversold) or RSI > 70 (overbought); avoid earnings within 4 weeks

### Exit Rules
- **Long exit**: +DI crosses below -DI (while ADX > 25), or ADX crosses below 25
- **Short exit**: +DI crosses above -DI (while ADX > 25), or ADX crosses below 25
- **Take profit**: 50% of initial premium (when using options); timed exit at 1 month

### Recommended Timeframes
- Daily bars (14-period ADX default)
- Intraday: 15-min, 30-min with ADX 14

### Backtested Results
| Source | Win Rate | Notes |
|--------|----------|-------|
| Options Trading IQ (Dow stocks 2018–2020) | **65–67%** | Bullish pattern: 14/21 higher 1 month later; bearish similar |
| Options IQ with filters (50 MA, RSI, no earnings) | **80%** (8/10 bearish), **100%** (12/12 bullish) | Credit spreads; $844–$1,531 profit |
| PyQuantLab enhancement (Bollinger + trailing stops) | **36–182%** profit improvement | vs. basic ADX strategy |

### Optimal Parameters
- **ADX period**: 14 (default)
- **Threshold**: 20 (earlier entries) or 25 (stronger trend filter)
- **Filters**: 50-day MA, RSI 30/70, avoid earnings

---

## 2. Supertrend Indicator Strategy

### Trend Detection Rules
- **Price above Supertrend line** → Uptrend (green)
- **Price below Supertrend line** → Downtrend (red)
- **Reversal**: Price close crosses previous period’s Supertrend value → trend change
- Supertrend = ATR-based bands; uses Median Price ± (Multiplier × ATR)

### Entry Trigger Conditions
- **Long**: Price closes above Supertrend line (flip from red to green)
- **Short**: Price closes below Supertrend line (flip from green to red)

### Exit Rules
- **Trailing stop**: Supertrend line acts as dynamic stop
- **Long exit**: Price closes below Supertrend
- **Short exit**: Price closes above Supertrend
- Exit on opposite signal (trend reversal)

### Recommended Timeframes
- **Weekly** (best for S&P 500 in long-term backtest)
- **Intraday**: 5-min, 15-min with Period 10, Multiplier 3
- Scalping: 1-min, 5-min with Period 7, Multiplier 2.5

### Backtested Results
| Source | Asset | Period | Win Rate | Avg Win | Max DD | Profit Factor |
|--------|-------|--------|----------|---------|--------|---------------|
| QuantifiedStrategies | S&P 500 | 60 years (weekly) | **65.79%** | **11.07%**/trade | 24.6% | — |
| QuantifiedStrategies | S&P 500 | 60 years | — | — | 24.6% vs 56.24% B&H | Risk-adj return 9.44% |
| Share.Market (200k+ trades) | Nifty 500 | 2012–2025 | **40–43%** | 8.54%–25.53% | — | EV varies by params |
| QuantConnect | — | 3,609 days | — | — | **26.3%** | CAGR 6.3%, Sharpe 0.3 |

### Optimal Parameters
| Use Case | Period | Multiplier | Timeframe |
|----------|--------|------------|-----------|
| **Intraday (default)** | 10 | 3 | 5-min, 15-min |
| **Scalping** | 7 | 2.5 | 1-min, 5-min |
| **Swing / Best EV** | 3 | 3 | Daily (hold ~69 days) |
| **Conservative** | 14 | 4 | General |
| **Original (Seban)** | 10 | 3 | Weekly |

**Key insight (Share.Market)**: Win rate is consistently 40–43%; edge comes from **larger average wins than losses**. Higher multiplier (3.0) = fewer trades, much larger avg win (25%+ vs 8%).

---

## 3. EMA Crossover (9/21, 20/50) Intraday

### Trend Detection Rules
- **9/21**: Fast (9) EMA vs slow (21) EMA
- **20/50**: Medium (20) vs slower (50) — more swing-oriented
- **Bullish**: Fast EMA above slow EMA
- **Bearish**: Fast EMA below slow EMA

### Entry Trigger Conditions
- **9/21 Long**: 9 EMA crosses above 21 EMA
- **9/21 Short**: 9 EMA crosses below 21 EMA
- **20/50**: Golden cross (bullish) / death cross (bearish) — typically daily/swing

### Exit Rules
- **9/21**: Exit long when 9 crosses below 21; exit short when 9 crosses above 21
- **Trailing**: None in basic system; optional ATR-based stop

### Recommended Timeframes (9/21)
| Timeframe | Profit Factor (SPY) |
|-----------|---------------------|
| **15-min** | **1.262** (best) |
| **5-min** | **1.208** |
| 1-min | 1.082 |
| 30-min | 1.116 |
| 1-hour | 1.069 |
| 4-hour | 0.886 |
| 1-day | 0.947 |

### Backtested Results
- **Source**: Backtestx.com, SPY
- **Best intraday**: 15-min (PF 1.262), 5-min (PF 1.208)
- **20/50**: No specific intraday backtest data found; typically used on daily for swing

### Optimal Parameters
- **Intraday**: 9/21 EMA on 5-min or 15-min
- **20/50**: Daily bars for swing; not optimized for intraday in available data

---

## 4. Parabolic SAR Trend Following

### Trend Detection Rules
- **Bullish**: Dots below price
- **Bearish**: Dots above price
- **Reversal**: Dots flip from one side to the other

### Entry Trigger Conditions
- **Long**: Parabolic SAR dots flip from above price to below (close crosses above SAR)
- **Short**: Parabolic SAR dots flip from below price to above (close crosses below SAR)

### Exit Rules
- **Trailing stop**: Place stop just beyond most recent SAR dot
- **Long exit**: Dots flip above price
- **Short exit**: Dots flip below price

### Recommended Timeframes
- Works best in **strong trending** markets
- **Not effective** in sideways/choppy conditions
- Default: AF 0.02, max 0.20

### Backtested Results
| Source | Asset | Trades | Win Rate | Avg Gain | Max DD | Profit Factor |
|--------|-------|-------|----------|----------|--------|---------------|
| QuantifiedStrategies (Strategy 1) | SPY | 368 | **73%** | 0.56% | **41%** | 1.6 |
| QuantifiedStrategies (Strategy 2, flipped) | SPY | — | Poor | — | — | — |
| QuantifiedStrategies | Multiple assets | — | — | — | — | **Not profitable** on others |

**Caveat**: Standalone PSAR is **not consistently profitable** across assets. Best used with filters (RSI, ADX, MA). RSI filter: RSI < 70 for longs, RSI > 30 for shorts.

### Optimal Parameters
- **Default**: AF start 0.02, max 0.20
- **Enhancement**: Combine with RSI, ADX, or MA for better reliability

---

## 5. Ichimoku Cloud Intraday

### Components (Standard Settings)
- **Tenkan-sen (Conversion)**: 9 periods — (High+Low)/2
- **Kijun-sen (Base)**: 26 periods
- **Senkou Span A**: (Tenkan + Kijun)/2, shifted +26
- **Senkou Span B**: 52-period midpoint, shifted +26
- **Chikou Span**: Close shifted -26

### Trend Detection Rules
- **Uptrend**: Price above cloud; Senkou A above Senkou B
- **Downtrend**: Price below cloud; Senkou A below Senkou B
- **Tenkan/Kijun cross**: Bullish when Tenkan crosses above Kijun; bearish opposite

### Entry Trigger Conditions
- **Long**: Price breaks above cloud + Tenkan above Kijun; optional ADX > 20
- **Short**: Price breaks below cloud + Tenkan below Kijun; optional ADX > 20
- **5-min scalping**: Conversion 9 (or 7 for faster), Base 26, Leading B 52

### Exit Rules
- Next Tenkan/Kijun crossover
- ATR-based or swing high/low stop
- Take profit: fixed ticks (e.g., 200) or percentage

### Recommended Timeframes
- **Daily** (QuantifiedStrategies: best risk-adjusted)
- **Intraday**: 15-min, 30-min; 5-min for scalping with adjusted Conversion (7–9)

### Backtested Results
| Asset | CAGR (Strategy) | CAGR (B&H) | Time in Market | Max DD |
|-------|-----------------|------------|----------------|--------|
| QQQ | 7.7% | 7.92% | 63% | Lower |
| MDY | 6.5% | 11.45% | ~65% | **< half of B&H** |
| Bitcoin | **78.05%** | 59.8% | — | — |
| S&P 500 | 5.2% | 6.9% | — | — |
| Gold, EURUSD | Underperform | — | — | — |

**Conclusion**: Reduces drawdowns but **often underperforms buy-and-hold** on most assets. Best on Bitcoin in this sample.

### Optimal Parameters
- **Standard**: 9, 26, 52
- **5-min scalping**: Conversion 9 (or 7), Base 26, Leading B 52
- **Filter**: ADX > 20 for entry confirmation

---

## 6. Donchian Channel Breakout Strategy

### Trend Detection Rules
- **Upper band**: Highest high over N periods
- **Lower band**: Lowest low over N periods
- **Middle**: (Upper + Lower) / 2
- **Breakout up**: Price above upper band = bullish breakout
- **Breakout down**: Price below lower band = bearish breakout

### Entry Trigger Conditions
- **Long**: Price **closes above** upper band
- **Short**: Price **closes below** lower band
- **Filters**: Volume above average; momentum (RSI, MACD) in same direction; avoid low-volatility consolidation

### Exit Rules
- **Long exit**: Price closes below lower band, or moves back inside channel
- **Short exit**: Price closes above upper band (or middle)
- **Trailing stop**: Along the channel bands
- **Stop-loss**: Below recent low (long), above recent high (short); 1–2% risk per trade

### Recommended Timeframes
- **20-period**: ~1 month on daily; common for intermediate breakouts
- **Intraday**: 5-min, 15-min, 4-hour (TradeSearcher examples)
- **Daily**: NASDAQ 100, Gold Futures (35-year backtest)

### Backtested Results
| Source | Asset | Timeframe | Trades | Risk/Reward | ROI |
|--------|-------|-----------|--------|-------------|-----|
| TradeSearcher (112 backtests) | Various | 5m–Daily | 19–445 | 1.04–2.47 | 12–329% |
| Algomatic Trading | NASDAQ 100, Gold | Daily (1990–2025) | — | — | Compelling results |
| FMZ/Medium | — | — | — | — | Strategy documented |

**Note**: Win rate not standardized; performance varies by asset and timeframe. 20-period is standard.

### Optimal Parameters
- **Period**: 20 (default for ~1-month lookback)
- **Alternative**: 10 (faster), 55 (Turtle-style)
- **Risk**: 1–2% per trade

---

## Summary: Best Intraday Picks by Verifiable Data

| Strategy | Best Timeframe | Win Rate | Key Strength | Key Weakness |
|----------|----------------|----------|--------------|--------------|
| **9/21 EMA** | 15-min, 5-min | — | PF 1.26 on 15m SPY | Simple; many false signals in chop |
| **Supertrend** | 5m, 15m (10,3) | 40–67%* | Clear stops; 11% avg win (weekly) | Low win rate intraday; needs patience |
| **ADX + DI** | Daily, 15m | 65–80% | Good filters; options enhancement | Needs +DI/-DI + filters |
| **Donchian** | 5m–Daily | Varies | Clear breakout rules | Asset-dependent |
| **Ichimoku** | Daily, 30m | — | Lower drawdown | Often underperforms B&H |
| **Parabolic SAR** | Trending only | 73% (SPY) | Simple reversals | 41% DD; fails on other assets |

*Supertrend win rate: 65% on weekly S&P 60yr; 40–43% on Nifty 500 daily.

---

## Sources

1. **QuantifiedStrategies.com** – Supertrend, Parabolic SAR, Ichimoku backtests  
2. **Backtestx.com** – 9/21 EMA crossover profit factors by timeframe  
3. **Share.Market** – Supertrend 200k+ trades, Nifty 500 (2012–2025)  
4. **Options Trading IQ** – ADX strategy with DI, 50 MA, RSI filters  
5. **TradeSearcher** – Donchian breakout 112 backtests  
6. **PyQuantLab** – ADX enhancement, Parabolic SAR + RSI  
7. **FMZ/Research360** – Ichimoku scalping, Donchian strategies  
8. **Algomatic Trading** – Donchian 35-year backtest  

---

*Document generated from web research. Backtest results are from cited sources; performance may vary by instrument, timeframe, and implementation.*
