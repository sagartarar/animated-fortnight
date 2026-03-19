use chrono::{NaiveDateTime, Timelike};
use csv::ReaderBuilder;
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;

// ============== CONFIGURATION ==============
const START_CAPITAL: f64 = 200000.0;
const BASE_RISK_PCT: f64 = 1.5;
const LOT_SIZE: i32 = 50;
const MAX_POS_PCT: f64 = 0.20;

// Filter thresholds
const VWAP_THRESHOLD: f64 = 0.005;  // 0.5% deviation allowed
const EMA_FAST: usize = 9;
const EMA_SLOW: usize = 21;
const ADX_PERIOD: usize = 14;
const ADX_MIN: f64 = 22.0;
const ATR_PERIOD: usize = 14;
const VOL_PERIOD: usize = 20;
const VOL_MULTIPLIER: f64 = 2.0;  // Enhanced from 1.5

// Entry/Exit parameters
const IB_MINUTES: u32 = 30;  // First 30 min for IB
const TARGET_MULTIPLIER: f64 = 1.0;  // Reduced from 1.5
const ENTRY_DEADLINE: u32 = 1330;  // 1:30 PM - no new entries after
const EXIT_TIME: u32 = 1515;  // 3:15 PM
const TRAIL_ATR_MULT: f64 = 2.0;
const BREAKEVEN_ATR_MULT: f64 = 1.0;

// IB size limits (as multiplier of ATR)
const IB_MAX_ATR_MULT: f64 = 1.0;  // Skip extreme IB > 1× ATR
const IB_NARROW_ATR_MULT: f64 = 0.5;  // Narrow IB < 0.5× ATR (bonus points)

#[derive(Debug, Deserialize, Clone)]
struct Candle {
    #[serde(rename = "date")]
    datetime: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

#[derive(Debug, Clone)]
struct Trade {
    date: String,
    symbol: String,
    direction: String,
    entry: f64,
    exit: f64,
    qty: i32,
    pnl: f64,
    charges: f64,
    net_pnl: f64,
    capital_before: f64,
    capital_after: f64,
    exit_reason: String,
    confluence_score: i32,
    risk_pct: f64,
}

// ============== INDICATORS ==============

fn calculate_ema(data: &[f64], period: usize) -> Vec<f64> {
    let mut ema = vec![f64::NAN; data.len()];
    if data.len() < period { return ema; }
    let multiplier = 2.0 / (period as f64 + 1.0);
    let sma: f64 = data.iter().take(period).sum::<f64>() / period as f64;
    ema[period - 1] = sma;
    for i in period..data.len() {
        ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1];
    }
    ema
}

fn calculate_atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let len = closes.len();
    let mut atr = vec![f64::NAN; len];
    if len < period + 1 { return atr; }
    
    let mut tr_sum: f64 = 0.0;
    for i in 1..=period {
        let tr1: f64 = highs[i] - lows[i];
        let tr2: f64 = (highs[i] - closes[i-1]).abs();
        let tr3: f64 = (lows[i] - closes[i-1]).abs();
        tr_sum += f64::max(f64::max(tr1, tr2), tr3);
    }
    atr[period] = tr_sum / period as f64;
    
    for i in (period+1)..len {
        let tr1: f64 = highs[i] - lows[i];
        let tr2: f64 = (highs[i] - closes[i-1]).abs();
        let tr3: f64 = (lows[i] - closes[i-1]).abs();
        let tr: f64 = f64::max(f64::max(tr1, tr2), tr3);
        atr[i] = (atr[i-1] * (period as f64 - 1.0) + tr) / period as f64;
    }
    atr
}

fn calculate_adx(highs: &[f64], lows: &[f64], period: usize) -> Vec<f64> {
    let len = highs.len();
    let mut adx = vec![f64::NAN; len];
    if len < period * 2 { return adx; }
    
    let mut plus_dm = vec![0.0; len];
    let mut minus_dm = vec![0.0; len];
    
    for i in 1..len {
        let up = highs[i] - highs[i-1];
        let down = lows[i-1] - lows[i];
        plus_dm[i] = if up > down && up > 0.0 { up } else { 0.0 };
        minus_dm[i] = if down > up && down > 0.0 { down } else { 0.0 };
    }
    
    // Simplified ATR
    let mut tr_values = Vec::new();
    for i in 1..len {
        let tr1: f64 = highs[i] - lows[i];
        let tr2: f64 = (highs[i] - highs[i-1]).abs();
        let tr3: f64 = (lows[i] - lows[i-1]).abs();
        tr_values.push(f64::max(f64::max(tr1, tr2), tr3));
    }
    
    let mut tr_sum: f64 = tr_values.iter().take(period).sum();
    let mut atr: f64 = tr_sum / period as f64;
    
    let mut plus_di_sum: f64 = plus_dm.iter().skip(1).take(period).sum();
    let mut minus_di_sum: f64 = minus_dm.iter().skip(1).take(period).sum();
    
    if atr > 0.0 {
        let plus_di: f64 = 100.0 * plus_di_sum / atr;
        let minus_di: f64 = 100.0 * minus_di_sum / atr;
        let dx: f64 = if (plus_di + minus_di) > 0.0 {
            100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        } else { 0.0 };
        adx[period] = dx;
    }
    
    for i in (period+1)..len {
        atr = (atr * (period as f64 - 1.0) + tr_values[i-1]) / period as f64;
        plus_di_sum = plus_di_sum - (plus_di_sum / period as f64) + plus_dm[i];
        minus_di_sum = minus_di_sum - (minus_di_sum / period as f64) + minus_dm[i];
        
        if atr > 0.0 {
            let plus_di: f64 = 100.0 * plus_di_sum / atr;
            let minus_di: f64 = 100.0 * minus_di_sum / atr;
            let dx: f64 = if (plus_di + minus_di) > 0.0 {
                100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            } else { 0.0 };
            adx[i] = (adx[i-1] * (period as f64 - 1.0) + dx) / period as f64;
        }
    }
    adx
}

fn calculate_charges(buy: f64, sell: f64) -> f64 {
    let turnover = buy + sell;
    let brokerage = (buy * 0.0003).min(20.0) + (sell * 0.0003).min(20.0);
    let stt = sell * 0.000125;
    let exchange = turnover * 0.0000173;
    let sebi = turnover * 0.000001;
    let gst = (brokerage + exchange + sebi) * 0.18;
    let stamp = buy * 0.00002;
    brokerage + stt + exchange + sebi + gst + stamp
}

fn load_data(path: &str) -> Vec<(String, Vec<(NaiveDateTime, Candle)>)> {
    let pattern = format!("{}/*_15min.csv", path);
    let files: Vec<_> = glob(&pattern).unwrap().filter_map(|e| e.ok()).collect();
    println!("Found {} files", files.len());
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(ProgressStyle::default_bar().template("[{bar:40}] {pos}/{len}").unwrap());
    
    let mut all_data = Vec::new();
    for file in &files {
        let symbol = file.file_stem().unwrap().to_str().unwrap().replace("_15min", "");
        let f = File::open(file).unwrap();
        let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(f);
        let candles: Vec<Candle> = rdr.deserialize().filter_map(|r| r.ok()).collect();
        
        let mut processed = Vec::new();
        for candle in &candles {
            let dt = NaiveDateTime::parse_from_str(
                &candle.datetime.replace("T", " ")[..19], 
                "%Y-%m-%d %H:%M:%S"
            ).unwrap();
            processed.push((dt, candle.clone()));
        }
        processed.sort_by(|a, b| a.0.cmp(&b.0));
        
        if processed.len() > 100 {
            all_data.push((symbol, processed));
        }
        pb.inc(1);
    }
    pb.finish();
    all_data
}

// ============== CONFLUENCE SCORING ==============

fn calculate_confluence_score(
    direction: &str,
    price: f64,
    vwap: f64,
    ema_fast: f64,
    ema_slow: f64,
    adx: f64,
    volume: f64,
    vol_sma: f64,
    ib_range: f64,
    atr: f64,
) -> i32 {
    let mut score: i32 = 25;  // Base IB breakout signal
    
    // VWAP alignment (20 pts)
    let vwap_ok = if direction == "LONG" {
        price > vwap
    } else {
        price < vwap
    };
    if vwap_ok { score += 20; }
    
    // EMA trend alignment (15 pts)
    let ema_ok = if direction == "LONG" {
        ema_fast > ema_slow
    } else {
        ema_fast < ema_slow
    };
    if ema_ok { score += 15; }
    
    // ADX trending (10 pts)
    if adx > ADX_MIN { score += 10; }
    
    // Volume confirmation (10 pts)
    if volume > vol_sma * VOL_MULTIPLIER { score += 10; }
    
    // IB size bonus (5-10 pts)
    let ib_atr_ratio = ib_range / atr;
    if ib_atr_ratio < IB_NARROW_ATR_MULT {
        score += 10;  // Narrow IB
    } else if ib_atr_ratio < IB_MAX_ATR_MULT {
        score += 5;   // Normal IB
    }
    
    score
}

fn calculate_position_size(
    capital: f64,
    entry: f64,
    sl: f64,
    confluence_score: i32,
) -> (i32, f64) {
    // Base risk amount
    let base_risk_amt = capital * BASE_RISK_PCT / 100.0;
    
    // Confluence multiplier: 0.8 at 50 pts, 1.0 at 70 pts, 1.2 at 90 pts
    let raw_adj: f64 = (confluence_score - 50) as f64 * 0.01;
    let score_adj: f64 = if raw_adj > 0.0 { raw_adj } else { 0.0 };
    let confluence_mult: f64 = 0.8 + if score_adj < 0.4 { score_adj } else { 0.4 };
    let risk_amt = base_risk_amt * confluence_mult;
    let actual_risk_pct = BASE_RISK_PCT * confluence_mult;
    
    let sl_distance = (entry - sl).abs();
    if sl_distance <= 0.0 { return (0, 0.0); }
    
    let shares = ((risk_amt / sl_distance) as i32).min((capital * MAX_POS_PCT / entry) as i32);
    let qty = (shares / LOT_SIZE) * LOT_SIZE;
    
    if qty < LOT_SIZE { return (0, 0.0); }
    
    (qty, actual_risk_pct)
}

fn run_backtest() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║     IB HYBRID STRATEGY (Tier 2) - ₹2L Capital              ║");
    println!("║     VWAP + EMA + ADX Filters | Trailing Stops              ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    
    let data = load_data("../trading_data_repo/data/nifty_200_15min");
    println!("Loaded {} stocks\n", data.len());
    
    let mut capital = START_CAPITAL;
    let mut trades: Vec<Trade> = Vec::new();
    let mut daily_loss_count: i32 = 0;
    let mut consecutive_losses: i32 = 0;
    
    let pb = ProgressBar::new(data.len() as u64);
    pb.set_style(ProgressStyle::default_bar().template("[{bar:40}] {pos}/{len}").unwrap());
    
    for (symbol, candles) in &data {
        let closes: Vec<f64> = candles.iter().map(|(_, c)| c.close).collect();
        let highs: Vec<f64> = candles.iter().map(|(_, c)| c.high).collect();
        let lows: Vec<f64> = candles.iter().map(|(_, c)| c.low).collect();
        let volumes: Vec<f64> = candles.iter().map(|(_, c)| c.volume).collect();
        
        let ema9 = calculate_ema(&closes, EMA_FAST);
        let ema21 = calculate_ema(&closes, EMA_SLOW);
        let adx = calculate_adx(&highs, &lows, ADX_PERIOD);
        let atr = calculate_atr(&highs, &lows, &closes, ATR_PERIOD);
        let vol_sma = calculate_ema(&volumes, VOL_PERIOD);
        
        let mut last_date = String::new();
        let mut daily_pnl: f64 = 0.0;
        let mut in_trade = false;
        let mut direction = String::new();
        let mut entry_price: f64 = 0.0;
        let mut qty: i32 = 0;
        let mut sl: f64 = 0.0;
        let mut target: f64 = 0.0;
        let mut highest_price: f64 = 0.0;
        let mut lowest_price: f64 = 0.0;
        let mut confluence_score: i32 = 0;
        let mut risk_pct: f64 = 0.0;
        let mut vwap_sum_pv: f64 = 0.0;
        let mut vwap_sum_v: f64 = 0.0;
        let mut ib_high: f64 = 0.0;
        let mut ib_low: f64 = 0.0;
        let mut ib_close: f64 = 0.0;
        let mut ib_set: bool = false;
        
        for i in 50..candles.len() {
            let (dt, candle) = &candles[i];
            let date = dt.date().to_string();
            let time = dt.time();
            let hhmm = time.hour() * 100 + time.minute();
            
            // New day - reset everything
            if date != last_date {
                // Close any open trade at EOD
                if in_trade {
                    let exit = candle.close;
                    let pnl = if direction == "LONG" {
                        (exit - entry_price) * qty as f64
                    } else {
                        (entry_price - exit) * qty as f64
                    };
                    let pos_val = entry_price * qty as f64;
                    let exit_val = exit * qty as f64;
                    let charges = calculate_charges(pos_val, exit_val);
                    let net_pnl = pnl - charges;
                    
                    if capital + net_pnl > 10000.0 {
                        let cb = capital;
                        capital += net_pnl;
                        daily_pnl += net_pnl;
                        
                        if net_pnl > 0.0 {
                            consecutive_losses = 0;
                        } else {
                            consecutive_losses += 1;
                        }
                        
                        trades.push(Trade {
                            date: last_date.clone(),
                            symbol: symbol.clone(),
                            direction: direction.clone(),
                            entry: entry_price,
                            exit,
                            qty,
                            pnl,
                            charges,
                            net_pnl,
                            capital_before: cb,
                            capital_after: capital,
                            exit_reason: "EOD".to_string(),
                            confluence_score,
                            risk_pct,
                        });
                    }
                }
                
                last_date = date;
                daily_pnl = 0.0;
                in_trade = false;
                ib_set = false;
                vwap_sum_pv = 0.0;
                vwap_sum_v = 0.0;
                consecutive_losses = 0;
            }
            
            // Update VWAP calculation
            let tp = (candle.high + candle.low + candle.close) / 3.0;
            vwap_sum_pv += tp * candle.volume;
            vwap_sum_v += candle.volume;
            let vwap = if vwap_sum_v > 0.0 { vwap_sum_pv / vwap_sum_v } else { f64::NAN };
            
            // Set Initial Balance (9:15-9:45)
            if hhmm >= 915 && hhmm <= 945 && !ib_set {
                if candle.high > ib_high { ib_high = candle.high; }
                if candle.low < ib_low || ib_low == 0.0 { ib_low = candle.low; }
                if hhmm == 945 || (ib_close == 0.0 && hhmm > 945) {
                    ib_close = candle.close;
                    ib_set = true;
                }
            }
            
            // Skip if indicators not ready
            if vwap.is_nan() || ema9[i].is_nan() || ema21[i].is_nan() || 
               adx[i].is_nan() || atr[i].is_nan() || vol_sma[i].is_nan() || !ib_set {
                continue;
            }
            
            // Check for exit if in trade
            if in_trade {
                // Update highest/lowest for trailing stop
                if candle.high > highest_price { highest_price = candle.high; }
                if candle.low < lowest_price { lowest_price = candle.low; }
                
                // Calculate current profit in ATR terms
                let current_profit = if direction == "LONG" {
                    candle.close - entry_price
                } else {
                    entry_price - candle.close
                };
                let profit_atr = current_profit / atr[i];
                
                // Check exits
                let mut exit_price: f64 = 0.0;
                let mut exit_reason: String = String::new();
                
                // Target hit
                let target_hit = if direction == "LONG" {
                    candle.high >= target
                } else {
                    candle.low <= target
                };
                if target_hit {
                    exit_price = target;
                    exit_reason = "TARGET".to_string();
                }
                // Stop loss hit
                else if (direction == "LONG" && candle.low <= sl) || 
                        (direction == "SHORT" && candle.high >= sl) {
                    exit_price = sl;
                    exit_reason = "STOP_LOSS".to_string();
                }
                // Trailing stop (after breakeven)
                else if profit_atr >= BREAKEVEN_ATR_MULT {
                    let trail_stop = if direction == "LONG" {
                        highest_price - atr[i] * TRAIL_ATR_MULT
                    } else {
                        lowest_price + atr[i] * TRAIL_ATR_MULT
                    };
                    
                    if (direction == "LONG" && candle.low <= trail_stop) ||
                       (direction == "SHORT" && candle.high >= trail_stop) {
                        exit_price = if trail_stop > sl { trail_stop } else { sl };  // Don't go below original SL
                        exit_reason = "TRAIL_STOP".to_string();
                    }
                }
                // VWAP reversal exit (when in profit)
                else if profit_atr > 0.5 {
                    let vwap_cross = if direction == "LONG" {
                        candle.close < vwap
                    } else {
                        candle.close > vwap
                    };
                    if vwap_cross {
                        exit_price = candle.close;
                        exit_reason = "VWAP_REVERSAL".to_string();
                    }
                }
                // Time exit
                else if hhmm >= EXIT_TIME {
                    exit_price = candle.close;
                    exit_reason = "TIME".to_string();
                }
                
                if !exit_reason.is_empty() {
                    let pnl = if direction == "LONG" {
                        (exit_price - entry_price) * qty as f64
                    } else {
                        (entry_price - exit_price) * qty as f64
                    };
                    let pos_val = entry_price * qty as f64;
                    let exit_val = exit_price * qty as f64;
                    let charges = calculate_charges(pos_val, exit_val);
                    let net_pnl = pnl - charges;
                    
                    // Check daily loss limit
                    if daily_pnl + net_pnl < -capital * 0.03 {
                        // Skip this trade, daily limit hit
                        in_trade = false;
                        continue;
                    }
                    
                    if capital + net_pnl > 10000.0 {
                        let cb = capital;
                        capital += net_pnl;
                        daily_pnl += net_pnl;
                        
                        if net_pnl > 0.0 {
                            consecutive_losses = 0;
                        } else {
                            consecutive_losses += 1;
                        }
                        
                        trades.push(Trade {
                            date: last_date.clone(),
                            symbol: symbol.clone(),
                            direction: direction.clone(),
                            entry: entry_price,
                            exit: exit_price,
                            qty,
                            pnl,
                            charges,
                            net_pnl,
                            capital_before: cb,
                            capital_after: capital,
                            exit_reason: exit_reason.clone(),
                            confluence_score,
                            risk_pct,
                        });
                    }
                    
                    in_trade = false;
                }
                continue;
            }
            
            // Entry check - only if within trading window and no daily limit
            if !in_trade && capital > 50000.0 && hhmm >= 945 && hhmm <= ENTRY_DEADLINE {
                // Skip after 3 consecutive losses
                if consecutive_losses >= 3 { continue; }
                
                // Skip if daily loss limit hit
                if daily_pnl < -capital * 0.03 { continue; }
                
                let ib_range = ib_high - ib_low;
                let ib_atr_ratio = ib_range / atr[i];
                
                // Skip extreme IB (> 1× ATR)
                if ib_atr_ratio > IB_MAX_ATR_MULT { continue; }
                
                // Check IB breakout
                let break_long = candle.close > ib_high && candle.open <= ib_high;
                let break_short = candle.close < ib_low && candle.open >= ib_low;
                
                if !break_long && !break_short { continue; }
                
                // Calculate confluence score
                let dir = if break_long { "LONG" } else { "SHORT" };
                let score = calculate_confluence_score(
                    dir, candle.close, vwap, ema9[i], ema21[i], 
                    adx[i], candle.volume, vol_sma[i], ib_range, atr[i]
                );
                
                // Minimum score to enter (50 points)
                if score < 50 { continue; }
                
                // Calculate position size
                let (sl_price, tgt_price) = if break_long {
                    (ib_low, entry_price + ib_range * TARGET_MULTIPLIER)
                } else {
                    (ib_high, entry_price - ib_range * TARGET_MULTIPLIER)
                };
                
                let (position_qty, actual_risk) = calculate_position_size(
                    capital, candle.close, sl_price, score
                );
                
                if position_qty >= LOT_SIZE {
                    in_trade = true;
                    direction = dir.to_string();
                    entry_price = candle.close;
                    qty = position_qty;
                    sl = sl_price;
                    target = tgt_price;
                    highest_price = candle.close;
                    lowest_price = candle.close;
                    confluence_score = score;
                    risk_pct = actual_risk;
                }
            }
        }
        pb.inc(1);
    }
    pb.finish();
    
    print_results(&trades, capital);
}

fn print_results(trades: &[Trade], final_capital: f64) {
    if trades.is_empty() {
        println!("\nNo trades executed.");
        return;
    }
    
    let total = trades.len();
    let wins = trades.iter().filter(|t| t.net_pnl > 0.0).count();
    let losses = total - wins;
    let win_pct = (wins as f64 / total as f64) * 100.0;
    
    let gross_profit: f64 = trades.iter().filter(|t| t.net_pnl > 0.0).map(|t| t.net_pnl).sum();
    let gross_loss: f64 = trades.iter().filter(|t| t.net_pnl <= 0.0).map(|t| t.net_pnl.abs()).sum();
    let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else { 0.0 };
    
    let net_pnl = final_capital - START_CAPITAL;
    let total_return = (net_pnl / START_CAPITAL) * 100.0;
    
    let mut max_dd: f64 = 0.0;
    let mut peak: f64 = START_CAPITAL;
    for t in trades {
        if t.capital_after > peak { peak = t.capital_after; }
        let dd = peak - t.capital_after;
        if dd > max_dd { max_dd = dd; }
    }
    let dd_pct = (max_dd / START_CAPITAL) * 100.0;
    
    let total_charges: f64 = trades.iter().map(|t| t.charges).sum();
    let avg_score: f64 = trades.iter().map(|t| t.confluence_score as f64).sum::<f64>() / total as f64;
    
    let mut exit_counts: HashMap<String, usize> = HashMap::new();
    for t in trades {
        *exit_counts.entry(t.exit_reason.clone()).or_insert(0) += 1;
    }
    
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                   BACKTEST RESULTS                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("Capital:     ₹{:>12.0} → ₹{:>12.0}", START_CAPITAL, final_capital);
    println!("Net P&L:     ₹{:>12.0} ({:.1}%)", net_pnl, total_return);
    println!("Max DD:      ₹{:>12.0} ({:.1}%)", max_dd, dd_pct);
    println!("Trades:      {:>12} ({} wins, {} losses)", total, wins, losses);
    println!("Win Rate:    {:>11.1}%", win_pct);
    println!("Profit Factor: {:>9.2}", profit_factor);
    println!("Avg Winner:  ₹{:>12.0}", if wins > 0 { gross_profit / wins as f64 } else { 0.0 });
    println!("Avg Loser:   ₹{:>12.0}", if losses > 0 { -gross_loss / losses as f64 } else { 0.0 });
    println!("Total Charges: ₹{:>10.0}", total_charges);
    println!("Avg Confluence: {:>8.1}", avg_score);
    
    println!("\nExit Reasons:");
    for (reason, count) in exit_counts.iter() {
        println!("  {}: {} ({:.1}%)", reason, count, (*count as f64 / total as f64) * 100.0);
    }
    
    // Confluence score analysis
    println!("\nPerformance by Confluence Score:");
    let mut score_buckets: HashMap<i32, (usize, f64)> = HashMap::new();
    for t in trades {
        let bucket = (t.confluence_score / 10) * 10;  // 50-59, 60-69, etc.
        let entry = score_buckets.entry(bucket).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += t.net_pnl;
    }
    
    let mut sorted_buckets: Vec<_> = score_buckets.iter().collect();
    sorted_buckets.sort_by(|a, b| a.0.cmp(b.0));
    for (score, (count, pnl)) in sorted_buckets {
        let avg = pnl / *count as f64;
        println!("  Score {}-{}: {} trades, avg ₹{:.0}", score, score+9, count, avg);
    }
    
    // Save CSV
    let mut wtr = csv::Writer::from_path("trades.csv").unwrap();
    wtr.write_record(&["date", "symbol", "direction", "entry", "exit", "qty", "pnl", "charges", "net_pnl", "capital_after", "exit_reason", "confluence", "risk_pct"]).unwrap();
    for t in trades {
        wtr.write_record(&[
            &t.date, &t.symbol, &t.direction,
            &format!("{:.2}", t.entry), &format!("{:.2}", t.exit),
            &format!("{}", t.qty), &format!("{:.2}", t.pnl),
            &format!("{:.2}", t.charges), &format!("{:.2}", t.net_pnl),
            &format!("{:.0}", t.capital_after), &t.exit_reason,
            &format!("{}", t.confluence_score), &format!("{:.2}", t.risk_pct),
        ]).unwrap();
    }
    wtr.flush().unwrap();
    println!("\n✅ Results saved to: trades.csv");
}

fn main() {
    run_backtest();
}
