use chrono::{NaiveDateTime, NaiveTime, Datelike, Weekday, Duration};
use csv::ReaderBuilder;
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

// ============== CAPITAL CONFIGURATION ==============
const STARTING_CAPITAL: f64 = 200000.0;  // ₹2 Lakh
const LEVERAGE: f64 = 5.0;
const RISK_PER_TRADE_PCT: f64 = 1.0;  // 1% risk per trade (conservative)
const MAX_EXPOSURE_PCT: f64 = 80.0;
const COMPOUND_MONTHLY: bool = true;
const MAX_BASE_CAPITAL: f64 = 5000000.0;  // Cap at ₹50L for realistic sizing

// ============== MDD REDUCTION CONTROLS ==============
const USE_DRAWDOWN_LADDER: bool = true;
const DD_TIER_1: f64 = 5.0;
const DD_TIER_2: f64 = 10.0;
const DD_TIER_3: f64 = 15.0;
const DD_TIER_4: f64 = 20.0;
const DD_HALT: f64 = 25.0;

const USE_DAILY_LOSS_LIMIT: bool = true;
const DAILY_LOSS_LIMIT_PCT: f64 = 3.0;

const USE_WEEKLY_LOSS_LIMIT: bool = true;
const WEEKLY_LOSS_LIMIT_PCT: f64 = 5.0;

const USE_MONTHLY_LOSS_LIMIT: bool = true;
const MONTHLY_LOSS_LIMIT_PCT: f64 = 8.0;

const USE_CONSECUTIVE_LOSS_CONTROL: bool = true;
const MAX_CONSECUTIVE_LOSSES: usize = 3;
const CONSECUTIVE_LOSS_SIZE_MULT: f64 = 0.5;

const USE_EQUITY_CURVE_FILTER: bool = true;
const EQUITY_MA_PERIOD: usize = 20;

// F&O settings
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
    // Indicators
    ema9: f64,
    ema21: f64,
    vwap: f64,
    atr: f64,
    adx: f64,
    plus_di: f64,
    minus_di: f64,
    supertrend: i32,
}

#[derive(Debug, Clone)]
struct Trade {
    date: String,
    year: i32,
    month: u32,
    symbol: String,
    strategy: String,
    trade_type: String,
    entry_price: f64,
    exit_price: f64,
    exit_reason: String,
    quantity: i32,
    position_value: f64,
    gross_pnl: f64,
    charges: f64,
    net_pnl: f64,
    capital_before: f64,
    capital_after: f64,
    return_pct: f64,
    dd_at_entry: f64,
    // Strategy-specific
    regime: Option<String>,
    momentum: Option<f64>,
    vwap_deviation: Option<f64>,
}

// ============== INDICATORS ==============

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

// ============== STRATEGY 1: INTRADAY MOMENTUM ==============

fn calculate_first_30min_return(candles: &[ProcessedCandle], date: &str) -> Option<f64> {
    // Find all candles for this date
    let day_candles: Vec<_> = candles.iter()
        .filter(|c| c.date_only == date)
        .collect();
    
    if day_candles.len() < 3 {
        return None;
    }
    
    // Get first 3 candles (9:15, 9:30, 9:45)
    let first_30: Vec<_> = day_candles.iter()
        .filter(|c| c.time >= NaiveTime::from_hms_opt(9, 15, 0).unwrap() && c.time <= NaiveTime::from_hms_opt(9, 45, 0).unwrap())
        .take(3)
        .collect();
    
    if first_30.len() < 2 {
        return None;
    }
    
    let first_price = first_30[0].open;
    let last_price = first_30.last().unwrap().close;
    
    Some(((last_price - first_price) / first_price) * 100.0)
}

fn generate_momentum_signal(candles: &[ProcessedCandle], idx: usize, date: &str) -> Option<(String, f64)> {
    let candle = &candles[idx];
    
    // Entry window: 11:00 AM - 2:00 PM (relaxed)
    if candle.time < NaiveTime::from_hms_opt(11, 0, 0).unwrap() || 
       candle.time > NaiveTime::from_hms_opt(14, 0, 0).unwrap() {
        return None;
    }
    
    // Calculate momentum
    let momentum = calculate_first_30min_return(candles, date)?;
    
    // Min momentum threshold (relaxed to 0.08% as per research)
    if momentum.abs() < 0.08 {
        return None;
    }
    
    // Determine direction
    let trade_type = if momentum > 0.0 { "LONG" } else { "SHORT" };
    
    Some((trade_type.to_string(), momentum))
}

// ============== STRATEGY 2: VWAP + LADDER ==============

fn generate_vwap_signal(candles: &[ProcessedCandle], idx: usize) -> Option<(String, f64)> {
    let candle = &candles[idx];
    
    // Entry window: 11:00 AM - 2:00 PM (relaxed)
    if candle.time < NaiveTime::from_hms_opt(11, 0, 0).unwrap() || 
       candle.time > NaiveTime::from_hms_opt(14, 0, 0).unwrap() {
        return None;
    }
    
    // VWAP deviation check (allow even without ADX filter for more signals)
    if candle.vwap.is_nan() {
        return None;
    }
    
    let deviation = ((candle.close - candle.vwap) / candle.vwap) * 100.0;
    
    // Need significant deviation (0.35% as per research)
    if deviation.abs() < 0.35 {
        return None;
    }
    
    // Check ADX if available (relaxed to 18)
    if !candle.adx.is_nan() && candle.adx >= 18.0 {
        // ADX confirms trend - follow trend direction
        if candle.plus_di > candle.minus_di && deviation > 0.0 {
            // Bullish trend above VWAP - continue LONG
            return Some(("LONG".to_string(), deviation));
        } else if candle.minus_di > candle.plus_di && deviation < 0.0 {
            // Bearish trend below VWAP - continue SHORT
            return Some(("SHORT".to_string(), deviation));
        }
        // Otherwise mean reversion
    }
    
    // Default: mean reversion (above VWAP = SHORT, below = LONG)
    let trade_type = if deviation > 0.0 { "SHORT" } else { "LONG" };
    
    Some((trade_type.to_string(), deviation))
}

// ============== STRATEGY 3: REGIME SWITCHING ==============

#[derive(Debug, Clone, PartialEq)]
enum MarketRegime {
    Bull,
    Bear,
    Range,
    Volatile,
    Unknown,
}

fn detect_regime(candles: &[ProcessedCandle], idx: usize) -> MarketRegime {
    if idx < 20 {
        return MarketRegime::Unknown;
    }
    
    let candle = &candles[idx];
    
    // Get recent candles for trend/volatility
    let recent: Vec<_> = candles.iter().take(idx + 1).rev().take(20).collect();
    
    // Calculate volatility (using closes)
    let closes: Vec<f64> = recent.iter().map(|c| c.close).collect();
    let returns: Vec<f64> = closes.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
    let vol = if !returns.is_empty() {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        variance.sqrt() * (252.0_f64).sqrt() * 100.0  // Annualized %
    } else {
        0.0
    };
    
    // Trend check
    let price = candle.close;
    let ema20 = if idx >= 20 {
        let ema_data: Vec<f64> = candles.iter().take(idx + 1).map(|c| c.close).collect();
        let ema = calculate_ema(&ema_data, 20);
        ema[idx]
    } else {
        f64::NAN
    };
    
    let trend = if !ema20.is_nan() {
        if price > ema20 { "up" } else { "down" }
    } else {
        "unknown"
    };
    
    let adx = candle.adx;
    
    // Regime classification
    if vol > 25.0 {
        if adx > 30.0 {
            match trend {
                "up" => MarketRegime::Bull,
                "down" => MarketRegime::Bear,
                _ => MarketRegime::Volatile,
            }
        } else {
            MarketRegime::Volatile
        }
    } else if adx > 30.0 {
        match trend {
            "up" => MarketRegime::Bull,
            "down" => MarketRegime::Bear,
            _ => MarketRegime::Unknown,
        }
    } else if adx < 20.0 && vol < 15.0 {
        MarketRegime::Range
    } else {
        MarketRegime::Unknown
    }
}

fn generate_regime_signal(candles: &[ProcessedCandle], idx: usize) -> Option<(String, MarketRegime)> {
    let candle = &candles[idx];
    
    // Entry window: 11:00 AM - 2:00 PM (relaxed)
    if candle.time < NaiveTime::from_hms_opt(11, 0, 0).unwrap() || 
       candle.time > NaiveTime::from_hms_opt(14, 0, 0).unwrap() {
        return None;
    }
    
    let regime = detect_regime(candles, idx);
    
    // Don't trade in volatile
    if regime == MarketRegime::Volatile {
        return None;
    }
    
    let ema9 = candle.ema9;
    let ema21 = candle.ema21;
    let close = candle.close;
    let vwap = candle.vwap;
    
    match regime {
        MarketRegime::Bull => {
            // LONG on pullback to EMA21 OR trend following
            if !ema21.is_nan() && close > ema21 {
                Some(("LONG".to_string(), regime))
            } else {
                None
            }
        },
        MarketRegime::Bear => {
            // SHORT on rally OR trend following
            if !ema21.is_nan() && close < ema21 {
                Some(("SHORT".to_string(), regime))
            } else {
                None
            }
        },
        MarketRegime::Range | MarketRegime::Unknown => {
            // VWAP mean reversion (relaxed threshold from 1.0 to 0.5)
            if !vwap.is_nan() {
                let dev = ((close - vwap) / vwap) * 100.0;
                if dev > 0.5 {
                    Some(("SHORT".to_string(), MarketRegime::Range))
                } else if dev < -0.5 {
                    Some(("LONG".to_string(), MarketRegime::Range))
                } else {
                    None
                }
            } else {
                None
            }
        },
        _ => None,
    }
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

// ============== RISK MANAGER ==============

struct RiskState {
    peak_capital: f64,
    current_dd_pct: f64,
    consecutive_losses: usize,
    daily_pnl: f64,
    weekly_pnl: f64,
    monthly_pnl: f64,
    equity_history: Vec<f64>,
    // Track current periods for reset
    current_day: String,
    current_week: u32,
    current_month: u32,
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
            current_day: String::new(),
            current_week: 0,
            current_month: 0,
        }
    }
    
    fn check_and_reset_periods(&mut self, date: &str, week: u32, month: u32) {
        // Reset daily P&L if day changed
        if date != self.current_day {
            self.daily_pnl = 0.0;
            self.current_day = date.to_string();
        }
        
        // Reset weekly P&L if week changed
        if week != self.current_week {
            self.weekly_pnl = 0.0;
            self.current_week = week;
        }
        
        // Reset monthly P&L if month changed
        if month != self.current_month {
            self.monthly_pnl = 0.0;
            self.current_month = month;
        }
    }
    
    fn update(&mut self, capital: f64, pnl: f64) {
        if capital > self.peak_capital {
            self.peak_capital = capital;
        }
        self.current_dd_pct = ((self.peak_capital - capital) / self.peak_capital) * 100.0;
        
        if pnl < 0.0 {
            self.consecutive_losses += 1;
        } else {
            self.consecutive_losses = 0;
        }
        
        self.daily_pnl += pnl;
        self.weekly_pnl += pnl;
        self.monthly_pnl += pnl;
        self.equity_history.push(capital);
    }
    
    fn get_position_multiplier(&self, capital: f64) -> (f64, String) {
        let mut mult = 1.0;
        let mut reason = String::new();
        
        // Drawdown ladder
        if self.current_dd_pct >= DD_HALT {
            return (0.0, format!("HALT ({:.1}% DD)", self.current_dd_pct));
        } else if self.current_dd_pct >= DD_TIER_4 {
            mult *= 0.25;
            reason = format!("DD_T4({:.1}%)", self.current_dd_pct);
        } else if self.current_dd_pct >= DD_TIER_3 {
            mult *= 0.50;
            reason = format!("DD_T3({:.1}%)", self.current_dd_pct);
        } else if self.current_dd_pct >= DD_TIER_2 {
            mult *= 0.75;
            reason = format!("DD_T2({:.1}%)", self.current_dd_pct);
        } else if self.current_dd_pct >= DD_TIER_1 {
            mult *= 0.90;
            reason = format!("DD_T1({:.1}%)", self.current_dd_pct);
        }
        
        // Loss limits
        let daily_loss = (-self.daily_pnl / capital) * 100.0;
        if daily_loss >= DAILY_LOSS_LIMIT_PCT {
            return (0.0, format!("DAILY_LIMIT ({:.1}%)", daily_loss));
        }
        
        let weekly_loss = (-self.weekly_pnl / capital) * 100.0;
        if weekly_loss >= WEEKLY_LOSS_LIMIT_PCT {
            return (0.0, format!("WEEKLY_LIMIT ({:.1}%)", weekly_loss));
        }
        
        let monthly_loss = (-self.monthly_pnl / capital) * 100.0;
        if monthly_loss >= MONTHLY_LOSS_LIMIT_PCT {
            return (0.0, format!("MONTHLY_LIMIT ({:.1}%)", monthly_loss));
        }
        
        // Consecutive losses
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES {
            mult *= CONSECUTIVE_LOSS_SIZE_MULT;
            if reason.is_empty() {
                reason = format!("CONSEC({})", self.consecutive_losses);
            }
        }
        
        // Equity curve
        if self.equity_history.len() >= EQUITY_MA_PERIOD {
            let recent: Vec<f64> = self.equity_history.iter().rev().take(EQUITY_MA_PERIOD).cloned().collect();
            let ma = recent.iter().sum::<f64>() / recent.len() as f64;
            if capital < ma {
                mult *= 0.5;
                if reason.is_empty() {
                    reason = "EQ_MA".to_string();
                }
            }
        }
        
        if reason.is_empty() {
            reason = "FULL".to_string();
        }
        
        (mult, reason)
    }
}

// ============== LOAD DATA ==============

fn load_stock_data(file_path: &Path) -> Option<(String, Vec<ProcessedCandle>)> {
    let symbol = file_path.file_stem()?.to_str()?.replace("_15min", "");
    
    let file = File::open(file_path).ok()?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let candles: Vec<Candle> = reader.deserialize().filter_map(|r| r.ok()).collect();
    if candles.len() < 100 { 
        eprintln!("Skipping {}: only {} candles", symbol, candles.len());
        return None; 
    }

    let mut processed: Vec<ProcessedCandle> = Vec::with_capacity(candles.len());
    for candle in &candles {
        // Robust date parsing for various formats
        let mut date_str = candle.date.clone();
        
        // Trim whitespace
        date_str = date_str.trim().to_string();
        
        // Replace 'T' with space for ISO format
        date_str = date_str.replace('T', " ");
        
        // Handle timezone by truncating after position 19 (YYYY-MM-DD HH:MM:SS)
        // This handles +05:30, -05:30, etc.
        if date_str.len() > 19 {
            // Check if position 19 is where time ends
            if date_str.chars().nth(19).map_or(false, |c| c == '+' || c == '-' || c == '.') {
                date_str = date_str[..19].to_string();
            }
        }
        
        let datetime = match NaiveDateTime::parse_from_str(&date_str, "%Y-%m-%d %H:%M:%S") {
            Ok(dt) => dt,
            Err(e) => {
                eprintln!("Date parse error for {}: {} (input: {})", symbol, e, candle.date);
                continue;
            }
        };
        
        processed.push(ProcessedCandle {
            datetime, date_only: datetime.date().to_string(), time: datetime.time(),
            open: candle.open, high: candle.high, low: candle.low, close: candle.close, volume: candle.volume,
            ema9: f64::NAN, ema21: f64::NAN, vwap: f64::NAN, atr: f64::NAN, 
            adx: f64::NAN, plus_di: f64::NAN, minus_di: f64::NAN, supertrend: -1,
        });
    }
    
    if processed.len() < 100 { 
        eprintln!("Skipping {}: only {} valid candles after parsing", symbol, processed.len());
        return None; 
    }

    // Sort by datetime to ensure chronological order
    processed.sort_by(|a, b| a.datetime.cmp(&b.datetime));

    let closes: Vec<f64> = processed.iter().map(|c| c.close).collect();
    let highs: Vec<f64> = processed.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = processed.iter().map(|c| c.low).collect();
    let volumes: Vec<f64> = processed.iter().map(|c| c.volume).collect();

    let ema9 = calculate_ema(&closes, 9);
    let ema21 = calculate_ema(&closes, 21);
    let atr = calculate_atr(&highs, &lows, &closes, 14);
    let vwap = calculate_vwap(&processed, &highs, &lows, &closes, &volumes);
    let (adx, plus_di, minus_di) = calculate_adx(&highs, &lows, &closes, 14);

    for i in 0..processed.len() {
        processed[i].ema9 = ema9[i];
        processed[i].ema21 = ema21[i];
        processed[i].atr = atr[i];
        processed[i].vwap = vwap[i];
        processed[i].adx = adx[i];
        processed[i].plus_di = plus_di[i];
        processed[i].minus_di = minus_di[i];
    }
    
    Some((symbol, processed))
}

// ============== SIMULATION ==============

fn simulate_strategy(strategy_name: &str, candles: &[ProcessedCandle], symbol: &str,
                     capital: &mut f64, risk_state: &mut RiskState, base_capital: f64) -> Vec<Trade> {
    let mut trades = Vec::new();
    let sl_atr_mult = 2.0;
    let rr_ratio = 1.5;
    let exit_time = NaiveTime::from_hms_opt(15, 15, 0).unwrap();
    
    // Track daily trades
    let mut last_date = String::new();
    let mut daily_trade_count = 0;
    
    for i in 50..(candles.len().saturating_sub(8)) {
        let candle = &candles[i];
        let date = &candle.date_only;
        
        // Reset daily count
        if date != &last_date {
            last_date = date.clone();
            daily_trade_count = 0;
        }
        
        // Max 2 trades per day per stock
        if daily_trade_count >= 2 {
            continue;
        }
        
        // Check and reset risk state periods
        let week = candle.datetime.iso_week().week();
        let month = candle.datetime.month();
        risk_state.check_and_reset_periods(date, week, month);
        
        // Check if we can trade
        let (size_mult, reason) = risk_state.get_position_multiplier(*capital);
        if size_mult <= 0.0 || *capital < 50000.0 {
            // Log first time we hit a limit for debugging
            if !reason.is_empty() && !reason.contains("FULL") {
                // Only log periodically to avoid spam
            }
            continue;
        }
        
        // Generate signal based on strategy
        let signal_opt = match strategy_name {
            "MOMENTUM" => generate_momentum_signal(candles, i, date),
            "VWAP" => generate_vwap_signal(candles, i),
            "REGIME" => generate_regime_signal(candles, i).map(|(t, r)| (t, match r {
                MarketRegime::Bull => 1.0,
                MarketRegime::Bear => -1.0,
                MarketRegime::Range => 0.0,
                _ => 0.0,
            })),
            _ => None,
        };
        
        let (trade_type, signal_value) = match signal_opt {
            Some(s) => s,
            None => continue,
        };
        
        // Skip if indicators not ready
        if candle.atr.is_nan() || candle.atr <= 0.0 {
            continue;
        }
        
        let entry_price = candle.close;
        let sl_distance = candle.atr * sl_atr_mult;
        
        let (sl_price, target_price) = if trade_type == "LONG" {
            (entry_price - sl_distance, entry_price + (sl_distance * rr_ratio))
        } else {
            (entry_price + sl_distance, entry_price - (sl_distance * rr_ratio))
        };
        
        // Calculate position size
        let risk_amount = base_capital * (RISK_PER_TRADE_PCT / 100.0) * size_mult;
        let risk_per_share = sl_distance;
        
        if risk_per_share <= 0.0 || risk_amount <= 0.0 {
            continue;
        }
        
        let num_lots = ((risk_amount / (risk_per_share * DEFAULT_LOT_SIZE as f64)).floor() as i32).max(1);
        let total_qty = num_lots * DEFAULT_LOT_SIZE;
        let position_value = total_qty as f64 * entry_price;
        
        // Simulate trade
        let mut exit_price = None;
        let mut exit_reason = String::new();
        
        for j in (i + 1)..std::cmp::min(i + 30, candles.len()) {
            let future = &candles[j];
            
            // Check for new day
            if future.date_only != *date {
                exit_price = Some(candles[j - 1].close);
                exit_reason = "EOD".to_string();
                break;
            }
            
            // Check time exit
            if future.time >= exit_time {
                exit_price = Some(future.close);
                exit_reason = "TIME_EXIT".to_string();
                break;
            }
            
            // Check SL
            if trade_type == "LONG" {
                if future.low <= sl_price {
                    exit_price = Some(sl_price);
                    exit_reason = "SL".to_string();
                    break;
                }
                if future.high >= target_price {
                    exit_price = Some(target_price);
                    exit_reason = "TARGET".to_string();
                    break;
                }
            } else {
                if future.high >= sl_price {
                    exit_price = Some(sl_price);
                    exit_reason = "SL".to_string();
                    break;
                }
                if future.low <= target_price {
                    exit_price = Some(target_price);
                    exit_reason = "TARGET".to_string();
                    break;
                }
            }
        }
        
        // Fallback: if no exit condition hit, use last candle in window
        if exit_price.is_none() && (i + 1) < candles.len() {
            let last_idx = std::cmp::min(i + 29, candles.len() - 1);
            // Only use if same day
            if candles[last_idx].date_only == *date {
                exit_price = Some(candles[last_idx].close);
                exit_reason = "WINDOW_END".to_string();
            }
        }
        
        if let Some(exit_px) = exit_price {
            let gross_pnl = if trade_type == "LONG" {
                (exit_px - entry_price) * total_qty as f64
            } else {
                (entry_price - exit_px) * total_qty as f64
            };
            
            let exit_value = exit_px * total_qty as f64;
            let charges = calculate_fno_charges(position_value, exit_value);
            let net_pnl = gross_pnl - charges;
            
            let capital_before = *capital;
            *capital += net_pnl;
            risk_state.update(*capital, net_pnl);
            
            let return_pct = (net_pnl / capital_before) * 100.0;
            
            let regime_str = if strategy_name == "REGIME" {
                if signal_value > 0.0 { Some("BULL".to_string()) }
                else if signal_value < 0.0 { Some("BEAR".to_string()) }
                else { Some("RANGE".to_string()) }
            } else {
                None
            };
            
            trades.push(Trade {
                date: date.clone(),
                year: candle.datetime.year(),
                month: candle.datetime.month(),
                symbol: symbol.to_string(),
                strategy: strategy_name.to_string(),
                trade_type: trade_type.clone(),
                entry_price,
                exit_price: exit_px,
                exit_reason,
                quantity: total_qty,
                position_value,
                gross_pnl,
                charges,
                net_pnl,
                capital_before,
                capital_after: *capital,
                return_pct,
                dd_at_entry: risk_state.current_dd_pct,
                regime: regime_str,
                momentum: if strategy_name == "MOMENTUM" { Some(signal_value) } else { None },
                vwap_deviation: if strategy_name == "VWAP" { Some(signal_value) } else { None },
            });
            
            daily_trade_count += 1;
        }
    }
    
    trades
}

// ============== MAIN ==============

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     KIMITRADE BACKTESTER - 3 Strategies (₹2L Capital)          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    
    let data_dir = "../../trading_data_repo/data/nifty_200_15min";
    let pattern = format!("{}/*_15min.csv", data_dir);
    let files: Vec<_> = glob(&pattern).expect("Failed to read glob").filter_map(|e| e.ok()).collect();
    
    println!("Loading {} stock files...", files.len());
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(ProgressStyle::default_bar().template("[{elapsed_precise}] [{bar:40}] {pos}/{len}").unwrap());
    
    let mut all_stock_data: Vec<(String, Vec<ProcessedCandle>)> = Vec::new();
    for file in &files {
        if let Some(data) = load_stock_data(file) { all_stock_data.push(data); }
        pb.inc(1);
    }
    pb.finish_with_message("Done!");
    println!("Loaded {} stocks\n", all_stock_data.len());
    
    // Run all three strategies
    let strategies = vec!["MOMENTUM", "VWAP", "REGIME"];
    
    for strategy in &strategies {
        println!("\n════════════════════════════════════════════════════════════════════");
        println!("STRATEGY: {}", match *strategy {
            "MOMENTUM" => "1. Intraday Momentum (Gao et al., JFE)",
            "VWAP" => "2. VWAP + Ladder Exit (SSRN 5095349)",
            "REGIME" => "3. Regime-Switching (HMM-based)",
            _ => strategy,
        });
        println!("════════════════════════════════════════════════════════════════════\n");
        
        let mut capital = STARTING_CAPITAL;
        let mut base_capital = STARTING_CAPITAL;
        let mut risk_state = RiskState::new(STARTING_CAPITAL);
        let mut all_trades: Vec<Trade> = Vec::new();
        
        println!("Backtesting on {} stocks...", all_stock_data.len());
        let pb = ProgressBar::new(all_stock_data.len() as u64);
        pb.set_style(ProgressStyle::default_bar().template("[{elapsed_precise}] [{bar:40}] {pos}/{len}").unwrap());
        
        for (symbol, candles) in &all_stock_data {
            let trades = simulate_strategy(strategy, candles, symbol, &mut capital, &mut risk_state, base_capital);
            all_trades.extend(trades);
            pb.inc(1);
        }
        pb.finish_with_message("Done!");
        
        // Statistics
        let total_trades = all_trades.len();
        let winners: usize = all_trades.iter().filter(|t| t.net_pnl > 0.0).count();
        let losers = total_trades - winners;
        let win_rate = if total_trades > 0 { (winners as f64 / total_trades as f64) * 100.0 } else { 0.0 };
        
        let gross_pnl: f64 = all_trades.iter().map(|t| t.gross_pnl).sum();
        let total_charges: f64 = all_trades.iter().map(|t| t.charges).sum();
        let net_pnl: f64 = all_trades.iter().map(|t| t.net_pnl).sum();
        
        let avg_winner = if winners > 0 { all_trades.iter().filter(|t| t.net_pnl > 0.0).map(|t| t.net_pnl).sum::<f64>() / winners as f64 } else { 0.0 };
        let avg_loser = if losers > 0 { all_trades.iter().filter(|t| t.net_pnl <= 0.0).map(|t| t.net_pnl).sum::<f64>() / losers as f64 } else { 0.0 };
        
        let total_wins: f64 = all_trades.iter().filter(|t| t.net_pnl > 0.0).map(|t| t.net_pnl).sum();
        let total_losses: f64 = all_trades.iter().filter(|t| t.net_pnl <= 0.0).map(|t| t.net_pnl.abs()).sum();
        let profit_factor = if total_losses > 0.0 { total_wins / total_losses } else { 0.0 };
        
        // Calculate max drawdown
        let mut peak = STARTING_CAPITAL;
        let mut max_dd = 0.0;
        let mut max_dd_pct = 0.0;
        for trade in &all_trades {
            if trade.capital_after > peak { peak = trade.capital_after; }
            let dd = peak - trade.capital_after;
            let dd_pct = (dd / peak) * 100.0;
            if dd > max_dd { max_dd = dd; max_dd_pct = dd_pct; }
        }
        
        // Years
        let years: std::collections::HashSet<_> = all_trades.iter().map(|t| t.year).collect();
        let num_years = years.len().max(1) as f64;
        let total_return = ((capital / STARTING_CAPITAL) - 1.0) * 100.0;
        let cagr = if num_years > 0.0 && capital > 0.0 { ((capital / STARTING_CAPITAL).powf(1.0 / num_years) - 1.0) * 100.0 } else { 0.0 };
        let calmar = if max_dd_pct > 0.0 { cagr / max_dd_pct } else { 0.0 };
        
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║                      BACKTEST RESULTS                            ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!("\nCAPITAL SUMMARY:");
        println!("├─ Starting Capital:    ₹{:>12.0}", STARTING_CAPITAL);
        println!("├─ Final Capital:       ₹{:>12.0}", capital);
        println!("├─ Net P&L:             ₹{:>12.0}", net_pnl);
        println!("├─ Total Return:        {:>11.1}%", total_return);
        println!("├─ CAGR:                {:>11.1}%", cagr);
        println!("├─ MAX DRAWDOWN:        ₹{:>12.0} ({:.1}%)", max_dd, max_dd_pct);
        println!("└─ Calmar Ratio:        {:>11.2}", calmar);
        println!();
        println!("TRADE STATISTICS:");
        println!("├─ Total Trades:        {:>12}", total_trades);
        println!("├─ Winners:             {:>12}", winners);
        println!("├─ Losers:              {:>12}", losers);
        println!("├─ Win Rate:            {:>11.1}%", win_rate);
        println!("├─ Profit Factor:         {:>11.2}", profit_factor);
        println!("├─ Avg Winner:          ₹{:>12.0}", avg_winner);
        println!("├─ Avg Loser:           ₹{:>12.0}", avg_loser);
        println!("└─ Total Charges:       ₹{:>12.0}", total_charges);
        
        // Exit reasons breakdown
        if !all_trades.is_empty() {
            println!();
            println!("EXIT REASONS:");
            let mut reasons: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for trade in &all_trades {
                *reasons.entry(trade.exit_reason.clone()).or_insert(0) += 1;
            }
            for (reason, count) in reasons.iter() {
                let pct = (*count as f64 / total_trades as f64) * 100.0;
                println!("├─ {}: {} ({:.1}%)", reason, count, pct);
            }
        }
        
        // Year-wise
        if !all_trades.is_empty() {
            println!();
            println!("YEAR-WISE PERFORMANCE:");
            let mut year_stats: Vec<_> = years.iter().collect();
            year_stats.sort();
            println!("{:<6} {:>8} {:>8} {:>12} {:>10}", "Year", "Trades", "WinRate", "Net P&L", "Return%");
            println!("{}", "-".repeat(55));
            for year in year_stats {
                let year_trades: Vec<_> = all_trades.iter().filter(|t| t.year == *year).collect();
                let yr_total = year_trades.len();
                let yr_winners = year_trades.iter().filter(|t| t.net_pnl > 0.0).count();
                let yr_wr = if yr_total > 0 { (yr_winners as f64 / yr_total as f64) * 100.0 } else { 0.0 };
                let yr_net: f64 = year_trades.iter().map(|t| t.net_pnl).sum();
                let yr_start = year_trades.first().map(|t| t.capital_before).unwrap_or(STARTING_CAPITAL);
                let yr_return = if yr_start > 0.0 { (yr_net / yr_start) * 100.0 } else { 0.0 };
                println!("{:<6} {:>8} {:>7.1}% {:>12.0} {:>9.1}%", year, yr_total, yr_wr, yr_net, yr_return);
            }
        }
        
        // Export
        let output_file = format!("../backtest_{}_2L.csv", strategy.to_lowercase());
        let mut wtr = csv::Writer::from_path(&output_file).expect("Failed to create output");
        wtr.write_record(&["date", "year", "symbol", "strategy", "type", "entry", "exit", "exit_reason", 
                           "qty", "position_value", "gross_pnl", "charges", "net_pnl", "capital_after"]).unwrap();
        for trade in &all_trades {
            wtr.write_record(&[
                &trade.date, &trade.year.to_string(), &trade.symbol, &trade.strategy, &trade.trade_type,
                &format!("{:.2}", trade.entry_price), &format!("{:.2}", trade.exit_price), &trade.exit_reason,
                &trade.quantity.to_string(), &format!("{:.0}", trade.position_value),
                &format!("{:.2}", trade.gross_pnl), &format!("{:.2}", trade.charges),
                &format!("{:.2}", trade.net_pnl), &format!("{:.0}", trade.capital_after),
            ]).unwrap();
        }
        wtr.flush().unwrap();
        println!("\n✅ Results saved to: {}", output_file);
    }
    
    println!("\n════════════════════════════════════════════════════════════════════");
    println!("                    ALL STRATEGIES COMPLETED                      ");
    println!("════════════════════════════════════════════════════════════════════");
}
