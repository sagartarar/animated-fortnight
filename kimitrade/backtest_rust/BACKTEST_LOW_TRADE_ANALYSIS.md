# Backtester Low Trade Count Analysis

**Current Results:** Momentum: 8 trades | VWAP: 106 trades | Regime: 15 trades over 11 years (196 stocks, ~200K candles/stock)

---

## 1. CRITICAL: RiskState Daily/Weekly/Monthly Limits Never Reset

**Location:** Lines 464-481, 505-518

**Problem:** `daily_pnl`, `weekly_pnl`, and `monthly_pnl` in `RiskState` are **never reset** when crossing day/week/month boundaries. They accumulate over the entire backtest. The "daily loss limit" check at line 505-507 compares against this cumulative value:

```rust
let daily_loss = (-self.daily_pnl / capital) * 100.0;
if daily_loss >= DAILY_LOSS_LIMIT_PCT {
    return (0.0, format!("DAILY_LIMIT ({:.1}%)", daily_loss));
}
```

**Impact:** After ~3 losing trades (each ~1% of capital), `daily_loss` exceeds 3% and **all subsequent trading halts** (`size_mult = 0`). With stocks processed sequentially (stock 1 fully, then stock 2, etc.), the first few stocks may get trades; all later stocks get `size_mult = 0` and never trade.

**Fix:** Reset `daily_pnl`/`weekly_pnl`/`monthly_pnl` when crossing calendar boundaries. This requires either:
- Processing all stocks in **global date order** (one date across all symbols, then next date), or
- Passing current date into `get_position_multiplier` and `update`, and resetting when date changes.

---

## 2. Simulation Order: Stock-by-Stock vs. Date-by-Date

**Location:** Lines 627-630 (main loop)

**Problem:** The main loop processes each stock completely before moving to the next:

```rust
for (symbol, candles) in &all_stock_data {
    let trades = simulate_strategy(strategy, candles, symbol, &mut capital, &mut risk_state, base_capital);
    ...
}
```

Combined with the non-resetting daily limit, the first stock(s) consume the "daily" budget, and later stocks never get to trade. Even with a fix to reset limits, this order means capital/risk state evolves per-stock rather than per-calendar-day, which can distort results.

**Fix:** Restructure to process by date: for each trading day, iterate over all stocks, then advance to the next day. This aligns with how real trading works and allows proper daily/weekly/monthly resets.

---

## 3. Strategy 1 (Momentum): Overly Strict Conditions

### 3a. First 30-Min Return Threshold (0.10%)

**Location:** Lines 251-254

```rust
if momentum.abs() < 0.10 {
    return None;
}
```

**Problem:** 0.10% in 30 minutes is a meaningful move, but with 196 stocks × ~252 days × 11 years, only 8 trades suggests either:
- The threshold is still too high for the data, or
- Other filters (e.g., risk limits) are blocking most trades.

**Fix:** Try lowering to 0.05% or 0.07% for more signals, then evaluate quality.

### 3b. First 30-Min Window Logic

**Location:** Lines 224-227

```rust
let first_30: Vec<_> = day_candles.iter()
    .filter(|c| c.time >= NaiveTime::from_hms_opt(9, 15, 0).unwrap() && c.time <= NaiveTime::from_hms_opt(9, 45, 0).unwrap())
    .take(3)
    .collect();
```

**Problem:** If data has only 2 candles in 9:15–9:45 (e.g., 9:30 and 9:45), the return is over 15 minutes, not 30. A 0.10% threshold calibrated for 30 minutes may be too strict for 15 minutes.

**Fix:** If `first_30.len() == 2`, consider using a lower threshold (e.g., 0.05%) or explicitly require 3 candles for the 30-min return.

### 3c. Entry Window (11:00–14:00)

**Location:** Lines 243-246

Only 12 candles per day (15-min bars) are eligible. This is intentional but further restricts opportunities.

---

## 4. Strategy 2 (VWAP): ADX Filter

**Location:** Lines 274-277

```rust
if candle.adx.is_nan() || candle.adx < 20.0 {
    return None;
}
```

**Problem:** ADX on 15-minute data is often in the 15–25 range. Requiring ADX ≥ 20 filters out many candles. VWAP has 106 trades vs. 8 for Momentum, so it is less restrictive, but ADX still removes a large share of bars.

**Fix:** Lower ADX threshold to 15, or make it configurable. Consider relaxing or removing it for a VWAP mean-reversion strategy.

---

## 5. Strategy 3 (Regime): Multiple Filters

### 5a. Volatile Regime Excluded

**Location:** Lines 379-382

```rust
if regime == MarketRegime::Volatile {
    return None;
}
```

**Problem:** No trades in Volatile regime. If volatility is often high, many bars are skipped.

### 5b. Volatility Annualization

**Location:** Lines 318-324

```rust
let vol = if !returns.is_empty() {
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    variance.sqrt() * (252.0_f64).sqrt() * 100.0  // Annualized %
} else {
    0.0
};
```

**Problem:** Returns are 15-minute returns over 20 bars (~5 hours). The annualization uses `sqrt(252)`, which is for daily returns. For 15-minute data, a more appropriate factor is `sqrt(252 * 26)` (≈26 bars per day). Using `sqrt(252)` understates volatility by ~5×, so `vol > 25` is rarely true and Volatile regime is under-detected. Range/Unknown regimes may dominate.

### 5c. Regime Classification Logic

**Location:** Lines 345-364

Conditions are:

- Bull/Bear: `adx > 30` (and vol/trend checks)
- Volatile: `vol > 25` and `adx <= 30`
- Range: `adx < 20` and `vol < 15`
- Unknown: everything else

With the wrong vol annualization, most bars fall into Range or Unknown. Bull/Bear require ADX > 30, which is strict for 15-min data.

**Fix:** Correct volatility annualization for 15-minute data. Consider lowering ADX thresholds for regime detection.

---

## 6. Data Processing: Date/Time Parsing

**Location:** Lines 559-564

```rust
let date_str = candle.date.split('+').next().unwrap_or(&candle.date);
let date_str = date_str.trim();
let datetime = NaiveDateTime::parse_from_str(date_str, "%Y-%m-%d %H:%M:%S").ok()?;
```

**Problems:**
1. **UTC timestamps:** If data is stored in UTC (e.g., `03:45` for 9:15 IST), times will not match 9:15–9:45 or 11:00–14:00 windows. First-30-min and entry logic would fail.
2. **ISO format:** `"2015-02-02T09:15:00"` (with `T`) does not match `"%Y-%m-%d %H:%M:%S"` and would cause parse failure and candle drop.
3. **Negative offset:** `"2015-02-02 09:15:00-05:00"` is not handled; `split('+')` leaves the suffix, and parse fails.

**Fix:**
- Normalize timezone: if data is UTC, convert to IST before applying time filters.
- Support `T` in the format or normalize before parsing.
- Strip timezone suffixes like `-05:00`, `Z`, etc., before parsing.

---

## 7. Simulation Loop: Additional Filters

**Location:** Lines 612-631

- **Loop range:** `50..(candles.len().saturating_sub(8))` — first 50 bars and last 8 are skipped (indicator warmup and exit simulation). Acceptable.
- **Max 2 trades/day/stock:** Line 454 — limits to 2 trades per stock per day.
- **Capital floor:** Line 458 — `capital < 50_000` skips trading.
- **ATR check:** Lines 478-480 — `atr.is_nan() || atr <= 0.0` skips. ATR(14) is usually valid after ~15 bars.

---

## 8. Summary of Recommended Fixes (Priority Order)

| Priority | Issue | Location | Suggested Fix |
|----------|------|----------|----------------|
| **P0** | Daily/weekly/monthly PnL never reset | 464-481, 505-518 | Reset when date/week/month changes; process in date order |
| **P0** | Simulation order | Main loop 627-630 | Process by date across all stocks |
| **P1** | Momentum threshold 0.10% | 251-254 | Lower to 0.05–0.07% |
| **P1** | VWAP ADX ≥ 20 | 274-277 | Lower to 15 or make configurable |
| **P2** | Regime vol annualization | 318-324 | Use `sqrt(252 * 26)` for 15-min data |
| **P2** | Date/time parsing | 559-564 | Handle UTC, ISO, and negative timezone offsets |
| **P3** | Regime ADX thresholds | 345-364 | Consider lowering ADX for Bull/Bear detection |

---

## 9. Quick Validation

To confirm the RiskState bug:

1. Add logging when `get_position_multiplier` returns `(0.0, "DAILY_LIMIT")`.
2. Log `daily_pnl` and `daily_loss` at that moment.
3. Check how many candles/stocks are processed before the first DAILY_LIMIT.

If DAILY_LIMIT triggers after a small number of trades, that explains the low trade counts.
