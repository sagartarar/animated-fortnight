use chrono::{NaiveDateTime, NaiveTime, Datelike, Weekday};
use csv::ReaderBuilder;
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

// ============== CAPITAL CONFIGURATION ==============
const STARTING_CAPITAL: f64 = 1000000.0;  // ₹10 Lakh
const LEVERAGE: f64 = 5.0;
const RISK_PER_TRADE_PCT: f64 = 2.0;  // 2% of capital per trade
const MAX_EXPOSURE_PCT: f64 = 80.0;
const MAX_CONCURRENT_TRADES: usize = 2;
const COMPOUND_MONTHLY: bool = true;  // Compound monthly, not per-trade
const MAX_BASE_CAPITAL: f64 = 10000000.0;  // Cap base capital at ₹1 Cr for realistic sizing

// ============== MDD REDUCTION CONTROLS ==============
// 1. Drawdown Ladder - reduce position size as drawdown increases
const USE_DRAWDOWN_LADDER: bool = true;
const DD_TIER_1: f64 = 5.0;   // At 5% DD -> 90% position size
const DD_TIER_2: f64 = 10.0;  // At 10% DD -> 75% position size
const DD_TIER_3: f64 = 15.0;  // At 15% DD -> 50% position size
const DD_TIER_4: f64 = 20.0;  // At 20% DD -> 25% position size
const DD_HALT: f64 = 25.0;    // At 25% DD -> STOP trading

// 2. Daily Loss Limit
const USE_DAILY_LOSS_LIMIT: bool = true;
const DAILY_LOSS_LIMIT_PCT: f64 = 3.0;  // Stop trading if daily loss > 3%

// 3. Weekly Loss Limit
const USE_WEEKLY_LOSS_LIMIT: bool = true;
const WEEKLY_LOSS_LIMIT_PCT: f64 = 5.0;  // Stop trading if weekly loss > 5%

// 4. Monthly Loss Limit
const USE_MONTHLY_LOSS_LIMIT: bool = true;
const MONTHLY_LOSS_LIMIT_PCT: f64 = 8.0;  // Stop trading if monthly loss > 8%

// 5. Consecutive Loss Control
const USE_CONSECUTIVE_LOSS_CONTROL: bool = true;
const MAX_CONSECUTIVE_LOSSES: usize = 3;  // After 3 losses, reduce size by 50%
const CONSECUTIVE_LOSS_SIZE_MULT: f64 = 0.5;

// 6. Equity Curve Filter
const USE_EQUITY_CURVE_FILTER: bool = true;
const EQUITY_MA_PERIOD: usize = 20;  // Pause when equity < 20-period MA

// 7. Volatility Filter (simulated via ATR regime)
const USE_VOLATILITY_FILTER: bool = true;
const HIGH_VOL_ATR_MULT: f64 = 1.5;  // If ATR > 1.5x avg, reduce size

// Strategy parameters
const ENTRY_START_HOUR: u32 = 11;
const ENTRY_START_MIN: u32 = 0;
const ENTRY_END_HOUR: u32 = 14;
const ENTRY_END_MIN: u32 = 0;
const EXIT_HOUR: u32 = 15;
const EXIT_MIN: u32 = 15;

const MIN_SCORE: i32 = 7;
const ATR_MULTIPLIER: f64 = 2.0;
const RR_RATIO: f64 = 1.5;
const MAX_TRADES_PER_STOCK_PER_DAY: usize = 1;

const USE_SL: bool = true;
const USE_ADX_FILTER: bool = true;
const MIN_ADX: f64 = 25.0;
const MAX_ADX: f64 = 50.0;

const TRADE_DIRECTION: &str = "SHORT_ONLY";
const DEFAULT_LOT_SIZE: i32 = 50;

// ============== DATA STRUCTURES ==============

#[derive(Debug, Deserialize, Clone)]
struct Candle {
    date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

#[derive(Debug, Clone)]
struct ProcessedCandle {
    datetime: NaiveDateTime,
    date_only: String,
    time: NaiveTime,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    rsi: f64,
    ema9: f64,
    ema21: f64,
    vwap: f64,
    supertrend: i32,
    atr: f64,
    adx: f64,
    plus_di: f64,
    minus_di: f64,
    day_change: f64,
}

#[derive(Debug, Clone)]
struct Trade {
    date: String,
    year: i32,
    month: u32,
    symbol: String,
    trade_type: String,
    score: i32,
    entry_price: f64,
    exit_price: f64,
    exit_reason: String,
    lot_size: i32,
    position_value: f64,
    margin_used: f64,
    gross_pnl: f64,
    charges: f64,
    net_pnl: f64,
    capital_before: f64,
    capital_after: f64,
    return_pct: f64,
    position_size_mult: f64,  // Position size multiplier applied
    dd_at_entry: f64,         // Drawdown at entry
}

#[derive(Debug, Clone)]
struct RiskState {
    peak_capital: f64,
    current_dd_pct: f64,
    consecutive_losses: usize,
    daily_pnl: f64,
    weekly_pnl: f64,
    monthly_pnl: f64,
    equity_history: Vec<f64>,
    recent_atr_values: Vec<f64>,
    is_trading_halted: bool,
    halt_reason: String,
}

impl RiskState {
    fn new(starting_capital: f64) -> Self {
        Self {
            peak_capital: starting_capital,
            current_dd_pct: 0.0,
            consecutive_losses: 0,
            daily_pnl: 0.0,
            weekly_pnl: 0.0,
            monthly_pnl: 0.0,
            equity_history: vec![starting_capital],
            recent_atr_values: Vec::new(),
            is_trading_halted: false,
            halt_reason: String::new(),
        }
    }
    
    fn update_after_trade(&mut self, capital: f64, pnl: f64) {
        // Update peak and drawdown
        if capital > self.peak_capital {
            self.peak_capital = capital;
        }
        self.current_dd_pct = ((self.peak_capital - capital) / self.peak_capital) * 100.0;
        
        // Update consecutive losses
        if pnl < 0.0 {
            self.consecutive_losses += 1;
        } else {
            self.consecutive_losses = 0;
        }
        
        // Update period P&L
        self.daily_pnl += pnl;
        self.weekly_pnl += pnl;
        self.monthly_pnl += pnl;
        
        // Update equity history
        self.equity_history.push(capital);
    }
    
    fn reset_daily(&mut self) {
        self.daily_pnl = 0.0;
    }
    
    fn reset_weekly(&mut self) {
        self.weekly_pnl = 0.0;
    }
    
    fn reset_monthly(&mut self) {
        self.monthly_pnl = 0.0;
    }
    
    fn get_position_size_multiplier(&self, capital: f64, current_atr: f64) -> (f64, String) {
        let mut multiplier = 1.0;
        let mut reason = String::new();
        
        // 1. Drawdown Ladder
        if USE_DRAWDOWN_LADDER {
            if self.current_dd_pct >= DD_HALT {
                return (0.0, format!("DD_HALT ({:.1}% DD)", self.current_dd_pct));
            } else if self.current_dd_pct >= DD_TIER_4 {
                multiplier *= 0.25;
                reason = format!("DD_T4({:.1}%)", self.current_dd_pct);
            } else if self.current_dd_pct >= DD_TIER_3 {
                multiplier *= 0.50;
                reason = format!("DD_T3({:.1}%)", self.current_dd_pct);
            } else if self.current_dd_pct >= DD_TIER_2 {
                multiplier *= 0.75;
                reason = format!("DD_T2({:.1}%)", self.current_dd_pct);
            } else if self.current_dd_pct >= DD_TIER_1 {
                multiplier *= 0.90;
                reason = format!("DD_T1({:.1}%)", self.current_dd_pct);
            }
        }
        
        // 2. Daily Loss Limit
        if USE_DAILY_LOSS_LIMIT {
            let daily_loss_pct = (-self.daily_pnl / capital) * 100.0;
            if daily_loss_pct >= DAILY_LOSS_LIMIT_PCT {
                return (0.0, format!("DAILY_LIMIT ({:.1}%)", daily_loss_pct));
            }
        }
        
        // 3. Weekly Loss Limit
        if USE_WEEKLY_LOSS_LIMIT {
            let weekly_loss_pct = (-self.weekly_pnl / capital) * 100.0;
            if weekly_loss_pct >= WEEKLY_LOSS_LIMIT_PCT {
                return (0.0, format!("WEEKLY_LIMIT ({:.1}%)", weekly_loss_pct));
            }
        }
        
        // 4. Monthly Loss Limit
        if USE_MONTHLY_LOSS_LIMIT {
            let monthly_loss_pct = (-self.monthly_pnl / capital) * 100.0;
            if monthly_loss_pct >= MONTHLY_LOSS_LIMIT_PCT {
                return (0.0, format!("MONTHLY_LIMIT ({:.1}%)", monthly_loss_pct));
            }
        }
        
        // 5. Consecutive Loss Control
        if USE_CONSECUTIVE_LOSS_CONTROL && self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES {
            multiplier *= CONSECUTIVE_LOSS_SIZE_MULT;
            if reason.is_empty() {
                reason = format!("CONSEC_LOSS({})", self.consecutive_losses);
            }
        }
        
        // 6. Equity Curve Filter
        if USE_EQUITY_CURVE_FILTER && self.equity_history.len() >= EQUITY_MA_PERIOD {
            let recent: Vec<f64> = self.equity_history.iter()
                .rev()
                .take(EQUITY_MA_PERIOD)
                .cloned()
                .collect();
            let equity_ma = recent.iter().sum::<f64>() / recent.len() as f64;
            let current_equity = *self.equity_history.last().unwrap_or(&capital);
            
            if current_equity < equity_ma {
                multiplier *= 0.5;
                if reason.is_empty() {
                    reason = "BELOW_EQ_MA".to_string();
                }
            }
        }
        
        // 7. Volatility Filter
        if USE_VOLATILITY_FILTER && self.recent_atr_values.len() >= 20 {
            let avg_atr: f64 = self.recent_atr_values.iter().sum::<f64>() / self.recent_atr_values.len() as f64;
            if current_atr > avg_atr * HIGH_VOL_ATR_MULT {
                multiplier *= 0.5;
                if reason.is_empty() {
                    reason = "HIGH_VOL".to_string();
                }
            }
        }
        
        if reason.is_empty() {
            reason = "FULL_SIZE".to_string();
        }
        
        (multiplier, reason)
    }
}

// ============== INDICATORS ==============

fn calculate_rsi(closes: &[f64], period: usize) -> Vec<f64> {
    let mut rsi = vec![f64::NAN; closes.len()];
    if closes.len() < period + 1 { return rsi; }

    let mut gains = vec![0.0; closes.len()];
    let mut losses = vec![0.0; closes.len()];

    for i in 1..closes.len() {
        let change = closes[i] - closes[i - 1];
        if change > 0.0 { gains[i] = change; } else { losses[i] = -change; }
    }

    let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

    if avg_loss > 0.0 {
        rsi[period] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
    } else {
        rsi[period] = 100.0;
    }

    for i in (period + 1)..closes.len() {
        avg_gain = (avg_gain * (period as f64 - 1.0) + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + losses[i]) / period as f64;
        if avg_loss > 0.0 {
            rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
        } else {
            rsi[i] = 100.0;
        }
    }
    rsi
}

fn calculate_ema(data: &[f64], period: usize) -> Vec<f64> {
    let mut ema = vec![f64::NAN; data.len()];
    if data.len() < period { return ema; }
    let multiplier = 2.0 / (period as f64 + 1.0);
    let sma: f64 = data[0..period].iter().sum::<f64>() / period as f64;
    ema[period - 1] = sma;
    for i in period..data.len() {
        ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
    }
    ema
}

fn calculate_atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let mut atr = vec![f64::NAN; closes.len()];
    if closes.len() < period + 1 { return atr; }
    let mut tr = vec![0.0; closes.len()];
    for i in 1..closes.len() {
        let hl = highs[i] - lows[i];
        let hc = (highs[i] - closes[i - 1]).abs();
        let lc = (lows[i] - closes[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }
    atr[period] = tr[1..=period].iter().sum::<f64>() / period as f64;
    for i in (period + 1)..closes.len() {
        atr[i] = (atr[i - 1] * (period as f64 - 1.0) + tr[i]) / period as f64;
    }
    atr
}

fn calculate_adx(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let len = closes.len();
    let mut adx = vec![f64::NAN; len];
    let mut plus_di = vec![f64::NAN; len];
    let mut minus_di = vec![f64::NAN; len];
    if len < period * 2 { return (adx, plus_di, minus_di); }
    
    let mut tr_sum = 0.0;
    let mut plus_dm_sum = 0.0;
    let mut minus_dm_sum = 0.0;
    
    for i in 1..=period {
        let tr = (highs[i] - lows[i]).max((highs[i] - closes[i-1]).abs()).max((lows[i] - closes[i-1]).abs());
        tr_sum += tr;
        let up_move = highs[i] - highs[i-1];
        let down_move = lows[i-1] - lows[i];
        if up_move > down_move && up_move > 0.0 { plus_dm_sum += up_move; }
        if down_move > up_move && down_move > 0.0 { minus_dm_sum += down_move; }
    }
    
    let mut smoothed_tr = tr_sum;
    let mut smoothed_plus_dm = plus_dm_sum;
    let mut smoothed_minus_dm = minus_dm_sum;
    let mut dx_values = Vec::new();
    
    for i in period..len {
        if i > period {
            let tr = (highs[i] - lows[i]).max((highs[i] - closes[i-1]).abs()).max((lows[i] - closes[i-1]).abs());
            let up_move = highs[i] - highs[i-1];
            let down_move = lows[i-1] - lows[i];
            let plus_dm = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
            let minus_dm = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };
            smoothed_tr = smoothed_tr - (smoothed_tr / period as f64) + tr;
            smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period as f64) + plus_dm;
            smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period as f64) + minus_dm;
        }
        if smoothed_tr > 0.0 {
            plus_di[i] = 100.0 * smoothed_plus_dm / smoothed_tr;
            minus_di[i] = 100.0 * smoothed_minus_dm / smoothed_tr;
            let di_sum = plus_di[i] + minus_di[i];
            let di_diff = (plus_di[i] - minus_di[i]).abs();
            if di_sum > 0.0 {
                let dx = 100.0 * di_diff / di_sum;
                dx_values.push(dx);
                if dx_values.len() >= period {
                    if dx_values.len() == period {
                        adx[i] = dx_values.iter().sum::<f64>() / period as f64;
                    } else {
                        adx[i] = (adx[i-1] * (period as f64 - 1.0) + dx) / period as f64;
                    }
                }
            }
        }
    }
    (adx, plus_di, minus_di)
}

fn calculate_supertrend(highs: &[f64], lows: &[f64], closes: &[f64], atr: &[f64], multiplier: f64) -> Vec<i32> {
    let mut direction = vec![0; closes.len()];
    let mut supertrend = vec![0.0; closes.len()];
    for i in 0..closes.len() {
        if atr[i].is_nan() { direction[i] = -1; continue; }
        let hl2 = (highs[i] + lows[i]) / 2.0;
        let upper = hl2 + multiplier * atr[i];
        let lower = hl2 - multiplier * atr[i];
        if i == 0 {
            supertrend[i] = upper;
            direction[i] = -1;
        } else {
            if closes[i] > supertrend[i - 1] { supertrend[i] = lower; direction[i] = 1; }
            else { supertrend[i] = upper; direction[i] = -1; }
        }
    }
    direction
}

fn calculate_vwap(candles: &[ProcessedCandle], highs: &[f64], lows: &[f64], closes: &[f64], volumes: &[f64]) -> Vec<f64> {
    let mut vwap = vec![f64::NAN; closes.len()];
    let mut cum_tp_vol = 0.0;
    let mut cum_vol = 0.0;
    let mut current_date = String::new();
    for i in 0..closes.len() {
        if candles[i].date_only != current_date {
            current_date = candles[i].date_only.clone();
            cum_tp_vol = 0.0;
            cum_vol = 0.0;
        }
        let tp = (highs[i] + lows[i] + closes[i]) / 3.0;
        cum_tp_vol += tp * volumes[i];
        cum_vol += volumes[i];
        if cum_vol > 0.0 { vwap[i] = cum_tp_vol / cum_vol; }
    }
    vwap
}

// ============== SCORING ==============

fn score_short_trade(candle: &ProcessedCandle, nifty_change: f64) -> i32 {
    let mut score = 0;
    if nifty_change >= 0.0 { return 0; }
    if USE_ADX_FILTER {
        if candle.adx.is_nan() || candle.adx < MIN_ADX || candle.adx > MAX_ADX { return 0; }
        if candle.minus_di > candle.plus_di { score += 1; }
    }
    if nifty_change < -0.5 { score += 3; }
    else if nifty_change < -0.3 { score += 2; }
    else if nifty_change < -0.1 { score += 1; }
    if !candle.rsi.is_nan() {
        if candle.rsi >= 60.0 && candle.rsi <= 75.0 { score += 2; }
        else if candle.rsi > 75.0 { score += 1; }
    }
    if !candle.vwap.is_nan() && candle.close < candle.vwap { score += 1; }
    if candle.supertrend == -1 { score += 2; }
    if !candle.ema9.is_nan() && !candle.ema21.is_nan() {
        if candle.ema9 < candle.ema21 { score += 1; }
        if candle.close < candle.ema9 { score += 1; }
    }
    score
}

fn score_buy_trade(candle: &ProcessedCandle, nifty_change: f64) -> i32 {
    let mut score = 0;
    if nifty_change <= 0.0 { return 0; }
    if USE_ADX_FILTER {
        if candle.adx.is_nan() || candle.adx < MIN_ADX || candle.adx > MAX_ADX { return 0; }
        if candle.plus_di > candle.minus_di { score += 1; }
    }
    if nifty_change > 0.5 { score += 3; }
    else if nifty_change > 0.3 { score += 2; }
    else if nifty_change > 0.1 { score += 1; }
    if !candle.rsi.is_nan() {
        if candle.rsi >= 35.0 && candle.rsi <= 50.0 { score += 2; }
        else if candle.rsi >= 30.0 && candle.rsi < 35.0 { score += 1; }
    }
    if !candle.vwap.is_nan() && candle.close > candle.vwap { score += 1; }
    if candle.supertrend == 1 { score += 2; }
    if !candle.ema9.is_nan() && !candle.ema21.is_nan() {
        if candle.ema9 > candle.ema21 { score += 1; }
        if candle.close > candle.ema9 { score += 1; }
    }
    score
}

// ============== CHARGES ==============

fn calculate_fno_charges(buy_value: f64, sell_value: f64) -> f64 {
    let total_turnover = buy_value + sell_value;
    let brokerage = (20.0_f64).min(buy_value * 0.0003) + (20.0_f64).min(sell_value * 0.0003);
    let stt = sell_value * 0.000125;
    let exchange = total_turnover * 0.0000173;
    let sebi = total_turnover * 0.000001;
    let gst = (brokerage + exchange + sebi) * 0.18;
    let stamp = buy_value * 0.00002;
    brokerage + stt + exchange + sebi + gst + stamp
}

// ============== POSITION SIZING ==============

fn calculate_fno_position(base_capital: f64, entry_price: f64, sl_price: f64, size_multiplier: f64) -> (i32, f64, f64) {
    let available_margin = base_capital * LEVERAGE;
    let max_position = available_margin * (MAX_EXPOSURE_PCT / 100.0);
    
    // Risk amount based on percentage of BASE capital (monthly compounded), adjusted by MDD multiplier
    let risk_amount = base_capital * (RISK_PER_TRADE_PCT / 100.0) * size_multiplier;
    let risk_per_share = (entry_price - sl_price).abs();
    
    if risk_per_share <= 0.0 || risk_amount <= 0.0 { return (0, 0.0, 0.0); }
    
    let lots_by_risk = (risk_amount / (risk_per_share * DEFAULT_LOT_SIZE as f64)).floor() as i32;
    let position_value_per_lot = entry_price * DEFAULT_LOT_SIZE as f64;
    let lots_by_position = (max_position / position_value_per_lot).floor() as i32;
    
    let num_lots = lots_by_risk.min(lots_by_position).max(1);
    
    // Apply size multiplier to lot count
    let adjusted_lots = ((num_lots as f64) * size_multiplier).ceil() as i32;
    if adjusted_lots < 1 { return (0, 0.0, 0.0); }
    
    let total_qty = adjusted_lots * DEFAULT_LOT_SIZE;
    let position_value = total_qty as f64 * entry_price;
    let margin_required = position_value / LEVERAGE;
    
    if margin_required > base_capital * 0.9 {
        let affordable_lots = ((base_capital * 0.9 * LEVERAGE) / position_value_per_lot).floor() as i32;
        if affordable_lots < 1 { return (0, 0.0, 0.0); }
        let adj_qty = affordable_lots * DEFAULT_LOT_SIZE;
        let adj_value = adj_qty as f64 * entry_price;
        let adj_margin = adj_value / LEVERAGE;
        return (adj_qty, adj_value, adj_margin);
    }
    
    (total_qty, position_value, margin_required)
}

// ============== LOAD DATA ==============

fn load_stock_data(file_path: &Path) -> Option<(String, Vec<ProcessedCandle>)> {
    let symbol = file_path.file_stem()?.to_str()?.replace("_15min", "");
    if symbol.contains("NIFTY") || symbol.contains("BANKNIFTY") { return None; }
    
    let file = File::open(file_path).ok()?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let candles: Vec<Candle> = reader.deserialize().filter_map(|r| r.ok()).collect();
    if candles.len() < 100 { return None; }

    let mut processed: Vec<ProcessedCandle> = Vec::with_capacity(candles.len());
    for candle in &candles {
        // Robust date parsing: handle +05:30, -05:30, T format, fractional seconds, trim
        let mut date_str = candle.date.split('+').next().unwrap_or(&candle.date).trim().to_string();
        if date_str.len() > 19 {
            let after_time = &date_str[19..];
            if after_time.starts_with('-') || after_time.starts_with('Z') {
                date_str = date_str[..19].to_string();
            }
        }
        date_str = date_str.replace('T', " ");
        if let Some(dot) = date_str.find('.') {
            date_str = date_str[..dot].to_string();
        }
        let datetime = NaiveDateTime::parse_from_str(date_str.trim(), "%Y-%m-%d %H:%M:%S").ok()?;
        processed.push(ProcessedCandle {
            datetime, date_only: datetime.date().to_string(), time: datetime.time(),
            open: candle.open, high: candle.high, low: candle.low, close: candle.close, volume: candle.volume,
            rsi: f64::NAN, ema9: f64::NAN, ema21: f64::NAN, vwap: f64::NAN, supertrend: -1,
            atr: f64::NAN, adx: f64::NAN, plus_di: f64::NAN, minus_di: f64::NAN, day_change: 0.0,
        });
    }
    if processed.len() < 100 { return None; }

    // Ensure chronological order for correct day_opens and indicators
    processed.sort_by(|a, b| a.datetime.cmp(&b.datetime));

    let closes: Vec<f64> = processed.iter().map(|c| c.close).collect();
    let highs: Vec<f64> = processed.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = processed.iter().map(|c| c.low).collect();
    let volumes: Vec<f64> = processed.iter().map(|c| c.volume).collect();

    let rsi = calculate_rsi(&closes, 14);
    let ema9 = calculate_ema(&closes, 9);
    let ema21 = calculate_ema(&closes, 21);
    let atr = calculate_atr(&highs, &lows, &closes, 10);
    let supertrend = calculate_supertrend(&highs, &lows, &closes, &atr, 3.0);
    let vwap = calculate_vwap(&processed, &highs, &lows, &closes, &volumes);
    let (adx, plus_di, minus_di) = calculate_adx(&highs, &lows, &closes, 14);

    let mut day_opens: HashMap<String, f64> = HashMap::new();
    for candle in &processed { day_opens.entry(candle.date_only.clone()).or_insert(candle.open); }

    for i in 0..processed.len() {
        processed[i].rsi = rsi[i];
        processed[i].ema9 = ema9[i];
        processed[i].ema21 = ema21[i];
        processed[i].atr = atr[i];
        processed[i].supertrend = supertrend[i];
        processed[i].vwap = vwap[i];
        processed[i].adx = adx[i];
        processed[i].plus_di = plus_di[i];
        processed[i].minus_di = minus_di[i];
        if let Some(&day_open) = day_opens.get(&processed[i].date_only) {
            if day_open > 0.0 { processed[i].day_change = ((processed[i].close - day_open) / day_open) * 100.0; }
        }
    }
    Some((symbol, processed))
}

// ============== MAIN ==============

fn main() {
    println!("================================================================");
    println!("F&O BACKTEST WITH MDD REDUCTION CONTROLS");
    println!("================================================================");
    println!();
    println!("CAPITAL: ₹{:.0} | Leverage: {}x", STARTING_CAPITAL, LEVERAGE);
    println!();
    println!("MDD REDUCTION CONTROLS:");
    println!("├─ Drawdown Ladder: {} (5%→90%, 10%→75%, 15%→50%, 20%→25%, 25%→HALT)", if USE_DRAWDOWN_LADDER {"ON"} else {"OFF"});
    println!("├─ Daily Loss Limit: {} ({}%)", if USE_DAILY_LOSS_LIMIT {"ON"} else {"OFF"}, DAILY_LOSS_LIMIT_PCT);
    println!("├─ Weekly Loss Limit: {} ({}%)", if USE_WEEKLY_LOSS_LIMIT {"ON"} else {"OFF"}, WEEKLY_LOSS_LIMIT_PCT);
    println!("├─ Monthly Loss Limit: {} ({}%)", if USE_MONTHLY_LOSS_LIMIT {"ON"} else {"OFF"}, MONTHLY_LOSS_LIMIT_PCT);
    println!("├─ Consecutive Loss Control: {} ({}→{}x size)", if USE_CONSECUTIVE_LOSS_CONTROL {"ON"} else {"OFF"}, MAX_CONSECUTIVE_LOSSES, CONSECUTIVE_LOSS_SIZE_MULT);
    println!("├─ Equity Curve Filter: {} ({}-period MA)", if USE_EQUITY_CURVE_FILTER {"ON"} else {"OFF"}, EQUITY_MA_PERIOD);
    println!("└─ Volatility Filter: {} (>{}x ATR→50%)", if USE_VOLATILITY_FILTER {"ON"} else {"OFF"}, HIGH_VOL_ATR_MULT);
    println!();

    let data_dir = "../trading_data_repo/data/nifty_200_15min";
    let pattern = format!("{}/*_15min.csv", data_dir);
    let files: Vec<_> = glob(&pattern).expect("Failed to read glob pattern").filter_map(|e| e.ok()).collect();

    println!("Loading {} stock files...", files.len());
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(ProgressStyle::default_bar().template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}").unwrap().progress_chars("#>-"));

    let mut all_stock_data: Vec<(String, Vec<ProcessedCandle>)> = Vec::new();
    for file in &files {
        if let Some(data) = load_stock_data(file) { all_stock_data.push(data); }
        pb.inc(1);
    }
    pb.finish_with_message("Loaded!");
    println!("Loaded {} stocks\n", all_stock_data.len());

    // Collect opportunities
    let entry_start = NaiveTime::from_hms_opt(ENTRY_START_HOUR, ENTRY_START_MIN, 0).unwrap();
    let entry_end = NaiveTime::from_hms_opt(ENTRY_END_HOUR, ENTRY_END_MIN, 0).unwrap();
    let exit_time = NaiveTime::from_hms_opt(EXIT_HOUR, EXIT_MIN, 0).unwrap();

    #[derive(Clone)]
    struct TradeOpp {
        symbol: String, candle_idx: usize, datetime: NaiveDateTime, date_only: String,
        year: i32, month: u32, week: u32, trade_type: String, score: i32,
        entry_price: f64, sl_price: f64, target_price: f64, atr: f64,
    }

    let mut opportunities: Vec<TradeOpp> = Vec::new();
    for (symbol, candles) in &all_stock_data {
        let mut daily_trades: HashMap<String, usize> = HashMap::new();
        for i in 50..(candles.len().saturating_sub(8)) {
            let candle = &candles[i];
            if candle.time < entry_start || candle.time > entry_end { continue; }
            let day_count = daily_trades.entry(candle.date_only.clone()).or_insert(0);
            if *day_count >= MAX_TRADES_PER_STOCK_PER_DAY { continue; }
            if candle.rsi.is_nan() || candle.vwap.is_nan() || candle.atr.is_nan() { continue; }

            let nifty_change = candle.day_change;
            let (trade_type, score) = match TRADE_DIRECTION {
                "SHORT_ONLY" => { let s = score_short_trade(candle, nifty_change); if s >= MIN_SCORE { ("SHORT", s) } else { continue; } }
                "BUY_ONLY" => { let s = score_buy_trade(candle, nifty_change); if s >= MIN_SCORE { ("BUY", s) } else { continue; } }
                _ => {
                    let buy_score = score_buy_trade(candle, nifty_change);
                    let short_score = score_short_trade(candle, nifty_change);
                    if buy_score >= MIN_SCORE && buy_score > short_score { ("BUY", buy_score) }
                    else if short_score >= MIN_SCORE { ("SHORT", short_score) }
                    else { continue; }
                }
            };

            let entry_price = candle.close;
            let sl_distance = candle.atr * ATR_MULTIPLIER;
            let (sl_price, target_price) = if trade_type == "BUY" {
                (entry_price - sl_distance, entry_price + sl_distance * RR_RATIO)
            } else {
                (entry_price + sl_distance, entry_price - sl_distance * RR_RATIO)
            };

            let week = candle.datetime.iso_week().week();
            opportunities.push(TradeOpp {
                symbol: symbol.clone(), candle_idx: i, datetime: candle.datetime, date_only: candle.date_only.clone(),
                year: candle.datetime.year(), month: candle.datetime.month(), week,
                trade_type: trade_type.to_string(), score, entry_price, sl_price, target_price, atr: candle.atr,
            });
            *day_count += 1;
        }
    }
    opportunities.sort_by(|a, b| a.datetime.cmp(&b.datetime));
    println!("Found {} trading opportunities\n", opportunities.len());

    // Simulate with risk controls
    println!("Simulating with MDD controls...");
    let mut capital = STARTING_CAPITAL;
    let mut base_capital = STARTING_CAPITAL;  // For monthly compounding
    let mut risk_state = RiskState::new(STARTING_CAPITAL);
    let mut trades: Vec<Trade> = Vec::new();
    let mut skipped_trades = 0usize;
    let mut current_date = String::new();
    let mut current_week = 0u32;
    let mut current_month = 0u32;
    let mut current_year = 0i32;

    let stock_data_map: HashMap<String, &Vec<ProcessedCandle>> = all_stock_data.iter().map(|(s, c)| (s.clone(), c)).collect();

    let pb = ProgressBar::new(opportunities.len() as u64);
    pb.set_style(ProgressStyle::default_bar().template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}").unwrap().progress_chars("#>-"));

    for opp in &opportunities {
        // Reset periods
        if opp.date_only != current_date {
            current_date = opp.date_only.clone();
            risk_state.reset_daily();
        }
        if opp.week != current_week || opp.year != current_year {
            current_week = opp.week;
            risk_state.reset_weekly();
        }
        if opp.month != current_month || opp.year != current_year {
            current_month = opp.month;
            current_year = opp.year;
            risk_state.reset_monthly();
            // Monthly compounding: update base capital (capped for realistic sizing)
            if COMPOUND_MONTHLY {
                base_capital = capital.min(MAX_BASE_CAPITAL);
            }
        }

        // Check risk controls
        let (size_mult, size_reason) = risk_state.get_position_size_multiplier(capital, opp.atr);
        
        if size_mult <= 0.0 {
            skipped_trades += 1;
            pb.inc(1);
            continue;
        }

        if capital < 50000.0 {
            skipped_trades += 1;
            pb.inc(1);
            continue;
        }

        let candles = match stock_data_map.get(&opp.symbol) { Some(c) => c, None => { pb.inc(1); continue; } };
        let (quantity, position_value, margin_used) = calculate_fno_position(base_capital, opp.entry_price, opp.sl_price, size_mult);
        if quantity == 0 { skipped_trades += 1; pb.inc(1); continue; }

        // Simulate trade
        let mut exit_price = None;
        let mut exit_reason = String::new();

        for j in (opp.candle_idx + 1)..std::cmp::min(opp.candle_idx + 20, candles.len()) {
            let future = &candles[j];
            if future.date_only != opp.date_only {
                exit_price = Some(candles[j - 1].close);
                exit_reason = "EOD".to_string();
                break;
            }
            if future.time >= exit_time {
                exit_price = Some(future.close);
                exit_reason = "TIME_EXIT".to_string();
                break;
            }
            if USE_SL {
                if opp.trade_type == "BUY" {
                    if future.low <= opp.sl_price { exit_price = Some(opp.sl_price); exit_reason = "SL".to_string(); break; }
                    if future.high >= opp.target_price { exit_price = Some(opp.target_price); exit_reason = "TARGET".to_string(); break; }
                } else {
                    if future.high >= opp.sl_price { exit_price = Some(opp.sl_price); exit_reason = "SL".to_string(); break; }
                    if future.low <= opp.target_price { exit_price = Some(opp.target_price); exit_reason = "TARGET".to_string(); break; }
            }
        }
    }

        // Fallback: if loop completed without exit (e.g. data gap, last candle before 15:15), use last candle
        if exit_price.is_none() && opp.candle_idx + 1 < candles.len() {
            let last_idx = (opp.candle_idx + 19).min(candles.len() - 1);
            exit_price = Some(candles[last_idx].close);
            exit_reason = "WINDOW_END".to_string();
        }

        if let Some(exit_px) = exit_price {
            let gross_pnl = if opp.trade_type == "BUY" { (exit_px - opp.entry_price) * quantity as f64 }
                           else { (opp.entry_price - exit_px) * quantity as f64 };
            let exit_value = exit_px * quantity as f64;
            let charges = calculate_fno_charges(position_value, exit_value);
            let net_pnl = gross_pnl - charges;

            let capital_before = capital;
            let dd_at_entry = risk_state.current_dd_pct;
            capital += net_pnl;
            risk_state.update_after_trade(capital, net_pnl);
            risk_state.recent_atr_values.push(opp.atr);
            if risk_state.recent_atr_values.len() > 50 { risk_state.recent_atr_values.remove(0); }

            let return_pct = (net_pnl / capital_before) * 100.0;

            trades.push(Trade {
                date: opp.date_only.clone(), year: opp.year, month: opp.month,
                symbol: opp.symbol.clone(), trade_type: opp.trade_type.clone(), score: opp.score,
                entry_price: opp.entry_price, exit_price: exit_px, exit_reason,
                lot_size: quantity, position_value, margin_used, gross_pnl, charges, net_pnl,
                capital_before, capital_after: capital, return_pct,
                position_size_mult: size_mult, dd_at_entry,
            });
        }
        pb.inc(1);
    }
    pb.finish_with_message("Done!");
    println!();

    // Calculate statistics
    let total_trades = trades.len();
    let winners: usize = trades.iter().filter(|t| t.net_pnl > 0.0).count();
    let losers = total_trades - winners;
    let win_rate = if total_trades > 0 { (winners as f64 / total_trades as f64) * 100.0 } else { 0.0 };

    let gross_pnl: f64 = trades.iter().map(|t| t.gross_pnl).sum();
    let total_charges: f64 = trades.iter().map(|t| t.charges).sum();
    let net_pnl: f64 = trades.iter().map(|t| t.net_pnl).sum();

    let avg_winner: f64 = if winners > 0 { trades.iter().filter(|t| t.net_pnl > 0.0).map(|t| t.net_pnl).sum::<f64>() / winners as f64 } else { 0.0 };
    let avg_loser: f64 = if losers > 0 { trades.iter().filter(|t| t.net_pnl <= 0.0).map(|t| t.net_pnl).sum::<f64>() / losers as f64 } else { 0.0 };

    let total_wins: f64 = trades.iter().filter(|t| t.net_pnl > 0.0).map(|t| t.net_pnl).sum();
    let total_losses: f64 = trades.iter().filter(|t| t.net_pnl <= 0.0).map(|t| t.net_pnl.abs()).sum();
    let profit_factor = if total_losses > 0.0 { total_wins / total_losses } else { 0.0 };

    let max_drawdown = risk_state.peak_capital - capital.min(STARTING_CAPITAL);
    let max_drawdown_pct = if risk_state.peak_capital > 0.0 { ((risk_state.peak_capital - trades.iter().map(|t| t.capital_after).fold(f64::MAX, f64::min)) / risk_state.peak_capital) * 100.0 } else { 0.0 };

    // Calculate actual max DD from equity curve
    let mut peak = STARTING_CAPITAL;
    let mut max_dd_actual = 0.0;
    let mut max_dd_pct_actual = 0.0;
    for trade in &trades {
        if trade.capital_after > peak { peak = trade.capital_after; }
        let dd = peak - trade.capital_after;
        let dd_pct = (dd / peak) * 100.0;
        if dd > max_dd_actual { max_dd_actual = dd; max_dd_pct_actual = dd_pct; }
    }

    let daily_returns: Vec<f64> = trades.iter().map(|t| t.return_pct).collect();
    let mean_return = if !daily_returns.is_empty() { daily_returns.iter().sum::<f64>() / daily_returns.len() as f64 } else { 0.0 };
    let variance: f64 = daily_returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / daily_returns.len().max(1) as f64;
    let std_dev = variance.sqrt();
    let sharpe = if std_dev > 0.0 { mean_return / std_dev * (252.0_f64).sqrt() } else { 0.0 };

    let negative_returns: Vec<f64> = daily_returns.iter().filter(|&&r| r < 0.0).cloned().collect();
    let downside_variance: f64 = negative_returns.iter().map(|r| r.powi(2)).sum::<f64>() / negative_returns.len().max(1) as f64;
    let downside_dev = downside_variance.sqrt();
    let sortino = if downside_dev > 0.0 { mean_return / downside_dev * (252.0_f64).sqrt() } else { 0.0 };

    let years: std::collections::HashSet<_> = trades.iter().map(|t| t.year).collect();
    let num_years = years.len() as f64;
    let total_return = ((capital / STARTING_CAPITAL) - 1.0) * 100.0;
    let cagr = if num_years > 0.0 && capital > 0.0 { ((capital / STARTING_CAPITAL).powf(1.0 / num_years) - 1.0) * 100.0 } else { 0.0 };
    let calmar = if max_dd_pct_actual > 0.0 { cagr / max_dd_pct_actual } else { 0.0 };
    let recovery_factor = if max_dd_actual > 0.0 { net_pnl / max_dd_actual } else { 0.0 };

    println!("================================================================");
    println!("RESULTS WITH MDD REDUCTION CONTROLS");
    println!("================================================================");
    println!();
    println!("CAPITAL SUMMARY:");
    println!("├─ Starting Capital:    ₹{:>12.0}", STARTING_CAPITAL);
    println!("├─ Final Capital:       ₹{:>12.0}", capital);
    println!("├─ Net P&L:             ₹{:>12.0}", net_pnl);
    println!("├─ Total Return:        {:>12.1}%", total_return);
    println!("├─ CAGR:                {:>12.1}%", cagr);
    println!("├─ Peak Capital:        ₹{:>12.0}", risk_state.peak_capital);
    println!("├─ MAX DRAWDOWN:        ₹{:>12.0} ({:.1}%)", max_dd_actual, max_dd_pct_actual);
    println!("└─ Years:               {:>12}", num_years as i32);
    println!();
    println!("TRADE STATISTICS:");
    println!("├─ Total Trades:        {:>12}", total_trades);
    println!("├─ Skipped (Risk):      {:>12}", skipped_trades);
    println!("├─ Winners:             {:>12}", winners);
    println!("├─ Losers:              {:>12}", losers);
    println!("├─ Win Rate:            {:>11.1}%", win_rate);
    println!("│");
    println!("├─ Gross P&L:           ₹{:>12.0}", gross_pnl);
    println!("├─ Total Charges:       ₹{:>12.0}", total_charges);
    println!("├─ Net P&L:             ₹{:>12.0}", net_pnl);
    println!("│");
    println!("├─ Avg Winner:          ₹{:>12.0}", avg_winner);
    println!("├─ Avg Loser:           ₹{:>12.0}", avg_loser);
    println!("├─ Avg Trade:           ₹{:>12.0}", net_pnl / total_trades.max(1) as f64);
    println!();
    println!("RISK-ADJUSTED RATIOS:");
    println!("├─ Profit Factor:       {:>12.2}", profit_factor);
    println!("├─ Sharpe Ratio:        {:>12.2}", sharpe);
    println!("├─ Sortino Ratio:       {:>12.2}", sortino);
    println!("├─ Calmar Ratio:        {:>12.2}", calmar);
    println!("├─ Recovery Factor:     {:>12.2}", recovery_factor);
    println!("└─ Expectancy:          ₹{:>12.0}/trade", net_pnl / total_trades.max(1) as f64);
    println!();

    // Year-wise
    println!("================================================================");
    println!("YEAR-WISE PERFORMANCE");
    println!("================================================================");
    let mut year_stats: Vec<_> = years.iter().collect();
    year_stats.sort();
    println!("{:<6} {:>8} {:>8} {:>12} {:>10} {:>14}", "Year", "Trades", "WinRate", "Net P&L", "Return%", "Capital");
    println!("{}", "-".repeat(65));
    for year in year_stats {
        let year_trades: Vec<_> = trades.iter().filter(|t| t.year == *year).collect();
        let yr_total = year_trades.len();
        let yr_winners = year_trades.iter().filter(|t| t.net_pnl > 0.0).count();
        let yr_wr = if yr_total > 0 { (yr_winners as f64 / yr_total as f64) * 100.0 } else { 0.0 };
        let yr_net: f64 = year_trades.iter().map(|t| t.net_pnl).sum();
        let yr_end_cap = year_trades.last().map(|t| t.capital_after).unwrap_or(STARTING_CAPITAL);
        let yr_start_cap = year_trades.first().map(|t| t.capital_before).unwrap_or(STARTING_CAPITAL);
        let yr_return = if yr_start_cap > 0.0 { (yr_net / yr_start_cap) * 100.0 } else { 0.0 };
        println!("{:<6} {:>8} {:>7.1}% {:>12.0} {:>9.1}% {:>14.0}", year, yr_total, yr_wr, yr_net, yr_return, yr_end_cap);
    }

    // Position size multiplier analysis
    println!();
    println!("================================================================");
    println!("POSITION SIZE ADJUSTMENTS (MDD Controls Impact)");
    println!("================================================================");
    let full_size = trades.iter().filter(|t| t.position_size_mult >= 0.99).count();
    let reduced_size = trades.iter().filter(|t| t.position_size_mult < 0.99).count();
    println!("├─ Full Size Trades:    {:>12}", full_size);
    println!("├─ Reduced Size Trades: {:>12}", reduced_size);
    println!("└─ Skipped Trades:      {:>12}", skipped_trades);

    // Export
    let capital_label = if STARTING_CAPITAL >= 1000000.0 { "10L" } else { "5L" };
    let output_file = format!("../backtest_fno_mdd_{}.csv", capital_label);
    let mut wtr = csv::Writer::from_path(&output_file).expect("Failed to create output file");
    wtr.write_record(&["date", "year", "symbol", "type", "score", "entry", "exit", "exit_reason", "lot_size", "position_value", "gross_pnl", "charges", "net_pnl", "capital_after", "return_pct", "size_mult", "dd_at_entry"]).unwrap();
    for trade in &trades {
        wtr.write_record(&[
            &trade.date, &trade.year.to_string(), &trade.symbol, &trade.trade_type, &trade.score.to_string(),
            &format!("{:.2}", trade.entry_price), &format!("{:.2}", trade.exit_price), &trade.exit_reason,
            &trade.lot_size.to_string(), &format!("{:.0}", trade.position_value),
            &format!("{:.2}", trade.gross_pnl), &format!("{:.2}", trade.charges), &format!("{:.2}", trade.net_pnl),
            &format!("{:.0}", trade.capital_after), &format!("{:.4}", trade.return_pct),
            &format!("{:.2}", trade.position_size_mult), &format!("{:.2}", trade.dd_at_entry),
        ]).unwrap();
    }
    wtr.flush().unwrap();
    println!();
    println!("✅ Results saved to: {}", output_file);
}
