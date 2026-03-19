use chrono::{NaiveDateTime, NaiveTime, Timelike};
use csv::ReaderBuilder;
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

// ============== CONFIGURATION ==============
const STARTING_CAPITAL: f64 = 200000.0;
const RISK_PER_TRADE_PCT: f64 = 1.5;

// Simplified - single exit at 3:15 PM or when 1R profit is reached
const TARGET_R: f64 = 1.0;  // Exit full position at 1R
const STOP_ATR_MULT: f64 = 2.0;
const ENTRY_START: u32 = 945;  // 9:45 AM (as HHMM)
const ENTRY_END: u32 = 1030;   // 10:30 AM
const EXIT_TIME: u32 = 1515;   // 3:15 PM

const RSI_PERIOD: usize = 2;
const ATR_PERIOD: usize = 14;
const ADX_PERIOD: usize = 14;
const VOLUME_PERIOD: usize = 20;

const LOT_SIZE: i32 = 50;
const MAX_POSITION_PCT: f64 = 0.20;  // Max 20% of capital per trade

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
    date: String,
    time: u32,  // HHMM format
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    rsi: f64,
    atr: f64,
    adx: f64,
    plus_di: f64,
    minus_di: f64,
    ema21: f64,
    ema50: f64,
    volume_sma: f64,
}

#[derive(Debug, Clone)]
struct Trade {
    id: String,
    date: String,
    symbol: String,
    trade_type: String,
    entry_price: f64,
    entry_time: u32,
    exit_price: f64,
    exit_time: u32,
    exit_reason: String,
    quantity: i32,
    position_value: f64,
    gross_pnl: f64,
    charges: f64,
    net_pnl: f64,
    capital_before: f64,
    capital_after: f64,
}

// ============== INDICATORS ==============

fn calculate_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let mut rsi = vec![f64::NAN; prices.len()];
    if prices.len() < period + 1 { return rsi; }
    
    let mut gains = 0.0;
    let mut losses = 0.0;
    
    for i in 1..=period {
        let change = prices[i] - prices[i-1];
        if change > 0.0 { gains += change; } else { losses += change.abs(); }
    }
    
    let mut avg_gain = gains / period as f64;
    let mut avg_loss = losses / period as f64;
    
    for i in period..prices.len() {
        let change = prices[i] - prices[i-1];
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { change.abs() } else { 0.0 };
        
        avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;
        
        if avg_loss > 0.0 {
            let rs = avg_gain / avg_loss;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        } else {
            rsi[i] = 100.0;
        }
    }
    rsi
}

fn calculate_atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let len = closes.len();
    let mut atr = vec![f64::NAN; len];
    if len < period + 1 { return atr; }
    
    let mut tr_sum = 0.0;
    for i in 1..=period {
        let tr = (highs[i] - lows[i])
            .max((highs[i] - closes[i-1]).abs())
            .max((lows[i] - closes[i-1]).abs());
        tr_sum += tr;
    }
    
    atr[period] = tr_sum / period as f64;
    
    for i in (period+1)..len {
        let tr = (highs[i] - lows[i])
            .max((highs[i] - closes[i-1]).abs())
            .max((lows[i] - closes[i-1]).abs());
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
    
    let atr = calculate_atr(highs, lows, highs, period);  // Use highs as placeholder for closes
    
    let mut plus_di_sum: f64 = plus_dm.iter().skip(1).take(period).sum();
    let mut minus_di_sum: f64 = minus_dm.iter().skip(1).take(period).sum();
    let atr_val = atr[period];
    
    if atr_val > 0.0 {
        let plus_di = 100.0 * plus_di_sum / atr_val;
        let minus_di = 100.0 * minus_di_sum / atr_val;
        let dx = if (plus_di + minus_di) > 0.0 {
            100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        } else { 0.0 };
        adx[period] = dx;
    }
    
    for i in (period+1)..len {
        if !atr[i].is_nan() && atr[i] > 0.0 {
            plus_di_sum = plus_di_sum - (plus_di_sum / period as f64) + plus_dm[i];
            minus_di_sum = minus_di_sum - (minus_di_sum / period as f64) + minus_dm[i];
            
            let plus_di = 100.0 * plus_di_sum / atr[i];
            let minus_di = 100.0 * minus_di_sum / atr[i];
            let dx = if (plus_di + minus_di) > 0.0 {
                100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            } else { 0.0 };
            
            adx[i] = (adx[i-1] * (period as f64 - 1.0) + dx) / period as f64;
        }
    }
    adx
}

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

fn time_to_hhmm(time: NaiveTime) -> u32 {
    time.hour() * 100 + time.minute()
}

// ============== LOAD DATA ==============

fn load_stock_data(file_path: &Path) -> Option<(String, Vec<ProcessedCandle>)> {
    let symbol = file_path.file_stem()?.to_str()?.replace("_15min", "");
    
    let file = File::open(file_path).ok()?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let candles: Vec<Candle> = reader.deserialize().filter_map(|r| r.ok()).collect();
    
    if candles.len() < 100 { return None; }
    
    let mut processed = Vec::with_capacity(candles.len());
    
    for candle in &candles {
        let mut date_str = candle.date.clone();
        date_str = date_str.replace('T', " ");
        date_str = date_str.trim().to_string();
        
        if date_str.len() > 19 {
            if let Some(pos) = date_str.find('+') {
                date_str = date_str[..pos].to_string();
            } else if let Some(pos) = date_str.find('-') {
                if pos > 10 { date_str = date_str[..pos].to_string(); }
            }
        }
        
        let datetime = NaiveDateTime::parse_from_str(&date_str, "%Y-%m-%d %H:%M:%S").ok()?;
        
        processed.push(ProcessedCandle {
            datetime,
            date: datetime.date().to_string(),
            time: time_to_hhmm(datetime.time()),
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
            volume: candle.volume,
            rsi: f64::NAN,
            atr: f64::NAN,
            adx: f64::NAN,
            plus_di: f64::NAN,
            minus_di: f64::NAN,
            ema21: f64::NAN,
            ema50: f64::NAN,
            volume_sma: f64::NAN,
        });
    }
    
    processed.sort_by(|a, b| a.datetime.cmp(&b.datetime));
    
    let closes: Vec<f64> = processed.iter().map(|c| c.close).collect();
    let highs: Vec<f64> = processed.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = processed.iter().map(|c| c.low).collect();
    let volumes: Vec<f64> = processed.iter().map(|c| c.volume).collect();
    
    let rsi = calculate_rsi(&closes, RSI_PERIOD);
    let atr = calculate_atr(&highs, &lows, &closes, ATR_PERIOD);
    let adx = calculate_adx(&highs, &lows, ADX_PERIOD);
    let ema21 = calculate_ema(&closes, 21);
    let ema50 = calculate_ema(&closes, 50);
    let volume_sma = calculate_ema(&volumes, VOLUME_PERIOD);
    
    for i in 0..processed.len() {
        processed[i].rsi = rsi[i];
        processed[i].atr = atr[i];
        processed[i].adx = adx[i];
        processed[i].ema21 = ema21[i];
        processed[i].ema50 = ema50[i];
        processed[i].volume_sma = volume_sma[i];
    }
    
    Some((symbol, processed))
}

// ============== STRATEGY LOGIC ==============

fn calculate_position_size(capital: f64, entry: f64, sl: f64) -> i32 {
    let risk_amount = capital * (RISK_PER_TRADE_PCT / 100.0);
    let sl_distance = (entry - sl).abs();
    
    if sl_distance < 0.5 || sl_distance > entry * 0.05 {
        return 0;  // Skip if SL too tight or too wide
    }
    
    let shares = (risk_amount / sl_distance) as i32;
    let max_shares = ((capital * MAX_POSITION_PCT) / entry) as i32;
    let final_shares = shares.min(max_shares);
    
    if final_shares < LOT_SIZE { return 0; }
    
    let lots = final_shares / LOT_SIZE;
    lots * LOT_SIZE
}

fn calculate_charges(buy_value: f64, sell_value: f64) -> f64 {
    let turnover = buy_value + sell_value;
    let brokerage = (buy_value * 0.0003).min(20.0) + (sell_value * 0.0003).min(20.0);
    let stt = sell_value * 0.000125;
    let exchange = turnover * 0.0000173;
    let sebi = turnover * 0.000001;
    let gst = (brokerage + exchange + sebi) * 0.18;
    let stamp = buy_value * 0.00002;
    brokerage + stt + exchange + sebi + gst + stamp
}

fn check_entry_signal(candles: &[ProcessedCandle], idx: usize, 
                      orb_high: f64, orb_low: f64, orb_close: f64) -> Option<String> {
    let candle = &candles[idx];
    let prev = &candles[idx-1];
    
    // Time check: 9:45 - 10:30 AM
    if candle.time < ENTRY_START || candle.time > ENTRY_END {
        return None;
    }
    
    // Check indicators are valid
    if candle.rsi.is_nan() || candle.atr.is_nan() || candle.adx.is_nan() {
        return None;
    }
    
    // ORB Direction
    let orb_range = orb_high - orb_low;
    let orb_mid = orb_low + orb_range / 2.0;
    
    // RSI Filter
    let rsi_ok = candle.rsi > 15.0 && candle.rsi < 85.0;
    if !rsi_ok { return None; }
    
    // Volume check
    let volume_ok = !candle.volume_sma.is_nan() && candle.volume > candle.volume_sma * 1.2;
    if !volume_ok { return None; }
    
    // ADX check
    let adx_ok = candle.adx > 20.0;
    if !adx_ok { return None; }
    
    // EMA trend alignment
    let trend_ok = !candle.ema21.is_nan() && !candle.ema50.is_nan();
    if !trend_ok { return None; }
    
    // Entry logic
    if orb_close > orb_mid && candle.close > orb_high && candle.close > candle.ema21 {
        // LONG: Above ORB, above EMA
        if candle.plus_di > candle.minus_di || candle.plus_di.is_nan() {
            return Some("LONG".to_string());
        }
    } else if orb_close < orb_mid && candle.close < orb_low && candle.close < candle.ema21 {
        // SHORT: Below ORB, below EMA
        if candle.minus_di > candle.plus_di || candle.minus_di.is_nan() {
            return Some("SHORT".to_string());
        }
    }
    
    None
}

// ============== BACKTEST ==============

fn run_backtest() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     CONFLUENCE-ORB BACKTEST - ₹2L Capital                      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");
    
    let data_dir = "../../trading_data_repo/data/nifty_200_15min";
    let pattern = format!("{}/*_15min.csv", data_dir);
    let files: Vec<_> = glob(&pattern).expect("Failed to read glob").filter_map(|e| e.ok()).collect();
    
    println!("Found {} stock files", files.len());
    println!("Loading and processing data...\n");
    
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(ProgressStyle::default_bar().template("[{elapsed_precise}] [{bar:40}] {pos}/{len}").unwrap());
    
    let mut all_data: Vec<(String, Vec<ProcessedCandle>)> = Vec::new();
    for file in &files {
        if let Some(data) = load_stock_data(file) {
            all_data.push(data);
        }
        pb.inc(1);
    }
    pb.finish_with_message("Done!");
    
    println!("Loaded {} stocks\n", all_data.len());
    
    // Run simulation
    let mut capital = STARTING_CAPITAL;
    let mut trades: Vec<Trade> = Vec::new();
    let mut trade_id = 0;
    
    println!("Running backtest...");
    let pb = ProgressBar::new(all_data.len() as u64);
    pb.set_style(ProgressStyle::default_bar().template("[{elapsed_precise}] [{bar:40}] {pos}/{len}").unwrap());
    
    for (symbol, candles) in &all_data {
        let mut current_date = String::new();
        let mut orb_high: f64 = 0.0;
        let mut orb_low: f64 = 0.0;
        let mut orb_close: f64 = 0.0;
        let mut orb_set = false;
        let mut in_position: Option<(String, f64, f64, i32, f64, u32)> = None; // (type, entry, sl, qty, target, entry_idx)
        let mut daily_trade_done = false;
        
        for i in 50..candles.len() {
            let candle = &candles[i];
            
            // New day - reset ORB and daily state
            if candle.date != current_date {
                // Close any overnight position (shouldn't happen for intraday)
                if let Some((pos_type, entry, _sl, qty, _target, _idx)) = in_position.take() {
                    let gross_pnl = if pos_type == "LONG" {
                        (candle.close - entry) * qty as f64
                    } else {
                        (entry - candle.close) * qty as f64
                    };
                    let pos_value = entry * qty as f64;
                    let exit_value = candle.close * qty as f64;
                    let charges = calculate_charges(pos_value, exit_value);
                    let net_pnl = gross_pnl - charges;
                    
                    if capital + net_pnl > 10000.0 {
                        let capital_before = capital;
                        capital += net_pnl;
                        trade_id += 1;
                        trades.push(Trade {
                            id: format!("{}", trade_id),
                            date: current_date.clone(),
                            symbol: symbol.clone(),
                            trade_type: pos_type.clone(),
                            entry_price: entry,
                            entry_time: 0,
                            exit_price: candle.close,
                            exit_time: candle.time,
                            exit_reason: "OVERNIGHT".to_string(),
                            quantity: qty,
                            position_value: pos_value,
                            gross_pnl,
                            charges,
                            net_pnl,
                            capital_before,
                            capital_after: capital,
                        });
                    }
                }
                
                current_date = candle.date.clone();
                orb_high = 0.0;
                orb_low = f64::MAX;
                orb_close = 0.0;
                orb_set = false;
                daily_trade_done = false;
            }
            
            // Set ORB (9:15-9:45)
            if candle.time >= 915 && candle.time <= 945 {
                if candle.high > orb_high { orb_high = candle.high; }
                if candle.low < orb_low { orb_low = candle.low; }
                if candle.time == 945 || (orb_close == 0.0 && candle.time > 945) {
                    orb_close = candle.close;
                    orb_set = true;
                }
            }
            
            // Skip if no ORB set yet or already traded today
            if !orb_set || daily_trade_done {
                continue;
            }
            
            // Check for exit if in position
            if let Some((ref pos_type, entry, sl, qty, target, _entry_idx)) = in_position {
                // Check target hit
                let target_hit = if pos_type == "LONG" {
                    candle.high >= target
                } else {
                    candle.low <= target
                };
                
                // Check stop loss
                let sl_hit = if pos_type == "LONG" {
                    candle.low <= sl
                } else {
                    candle.high >= sl
                };
                
                // Hard exit at 3:15 PM
                let time_exit = candle.time >= EXIT_TIME;
                
                let (exit_price, exit_reason) = if target_hit {
                    (target, "TARGET")
                } else if sl_hit {
                    (sl, "STOP_LOSS")
                } else if time_exit {
                    (candle.close, "TIME_EXIT")
                } else {
                    continue;  // Still in position
                };
                
                let gross_pnl = if pos_type == "LONG" {
                    (exit_price - entry) * qty as f64
                } else {
                    (entry - exit_price) * qty as f64
                };
                
                let pos_value = entry * qty as f64;
                let exit_value = exit_price * qty as f64;
                let charges = calculate_charges(pos_value, exit_value);
                let net_pnl = gross_pnl - charges;
                
                // Capital protection
                if capital + net_pnl < 10000.0 {
                    in_position = None;
                    continue;
                }
                
                let capital_before = capital;
                capital += net_pnl;
                
                trade_id += 1;
                trades.push(Trade {
                    id: format!("{}", trade_id),
                    date: candle.date.clone(),
                    symbol: symbol.clone(),
                    trade_type: pos_type.clone(),
                    entry_price: entry,
                    entry_time: candles[_entry_idx as usize].time,
                    exit_price,
                    exit_time: candle.time,
                    exit_reason: exit_reason.to_string(),
                    quantity: qty,
                    position_value: pos_value,
                    gross_pnl,
                    charges,
                    net_pnl,
                    capital_before,
                    capital_after: capital,
                });
                
                in_position = None;
                daily_trade_done = true;
                continue;
            }
            
            // Check for entry (only if not in position)
            if in_position.is_none() && capital > 50000.0 {
                if let Some(signal) = check_entry_signal(candles, i, orb_high, orb_low, orb_close) {
                    let entry = candle.close;
                    let atr = if candle.atr.is_nan() { 5.0 } else { candle.atr };
                    let sl_distance = atr * STOP_ATR_MULT;
                    
                    let (sl, target) = if signal == "LONG" {
                        (entry - sl_distance, entry + sl_distance * TARGET_R)
                    } else {
                        (entry + sl_distance, entry - sl_distance * TARGET_R)
                    };
                    
                    let qty = calculate_position_size(capital, entry, sl);
                    
                    if qty > 0 {
                        in_position = Some((signal, entry, sl, qty, target, i as u32));
                    }
                }
            }
        }
        
        pb.inc(1);
    }
    pb.finish_with_message("Done!");
    
    // Calculate and display results
    println!("\n");
    print_results(&trades, capital);
}

fn print_results(trades: &[Trade], final_capital: f64) {
    if trades.is_empty() {
        println!("No trades executed.");
        return;
    }
    
    let total_trades = trades.len();
    let winners = trades.iter().filter(|t| t.net_pnl > 0.0).count();
    let losers = total_trades - winners;
    let win_rate = (winners as f64 / total_trades as f64) * 100.0;
    
    let gross_pnl: f64 = trades.iter().map(|t| t.gross_pnl).sum();
    let total_charges: f64 = trades.iter().map(|t| t.charges).sum();
    let net_pnl: f64 = trades.iter().map(|t| t.net_pnl).sum();
    
    let total_wins: f64 = trades.iter().filter(|t| t.net_pnl > 0.0).map(|t| t.net_pnl).sum();
    let total_losses: f64 = trades.iter().filter(|t| t.net_pnl <= 0.0).map(|t| t.net_pnl.abs()).sum();
    let profit_factor = if total_losses > 0.0 { total_wins / total_losses } else { 0.0 };
    
    let avg_winner = if winners > 0 { total_wins / winners as f64 } else { 0.0 };
    let avg_loser = if losers > 0 { -total_losses / losers as f64 } else { 0.0 };
    
    // Calculate max drawdown
    let mut peak = STARTING_CAPITAL;
    let mut max_dd = 0.0;
    for trade in trades {
        if trade.capital_after > peak { peak = trade.capital_after; }
        let dd = peak - trade.capital_after;
        if dd > max_dd { max_dd = dd; }
    }
    let max_dd_pct = (max_dd / STARTING_CAPITAL) * 100.0;
    
    // Years
    let years: std::collections::HashSet<_> = trades.iter().map(|t| {
        t.date.split('-').next().unwrap_or("2020").to_string()
    }).collect();
    let num_years = years.len().max(1) as f64;
    let cagr = if final_capital > 0.0 {
        ((final_capital / STARTING_CAPITAL).powf(1.0 / num_years) - 1.0) * 100.0
    } else { 0.0 };
    
    // Exit reasons
    let mut exit_counts: HashMap<String, usize> = HashMap::new();
    for trade in trades {
        *exit_counts.entry(trade.exit_reason.clone()).or_insert(0) += 1;
    }
    
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                      BACKTEST RESULTS                            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    
    println!("\nCAPITAL SUMMARY:");
    println!("├─ Starting Capital:    ₹{:>12.0}", STARTING_CAPITAL);
    println!("├─ Final Capital:       ₹{:>12.0}", final_capital);
    println!("├─ Net P&L:             ₹{:>12.0}", net_pnl);
    println!("├─ Total Return:        {:>11.1}%", ((final_capital / STARTING_CAPITAL) - 1.0) * 100.0);
    println!("├─ CAGR:                {:>11.1}%", cagr);
    println!("├─ Max Drawdown:        ₹{:>12.0} ({:.1}%)", max_dd, max_dd_pct);
    println!("└─ Calmar Ratio:        {:>11.2}", if max_dd_pct > 0.0 { cagr / max_dd_pct } else { 0.0 });
    
    println!("\nTRADE STATISTICS:");
    println!("├─ Total Trades:        {:>12}", total_trades);
    println!("├─ Winners:             {:>12}", winners);
    println!("├─ Losers:              {:>12}", losers);
    println!("├─ Win Rate:            {:>11.1}%", win_rate);
    println!("├─ Profit Factor:       {:>11.2}", profit_factor);
    println!("├─ Avg Winner:          ₹{:>12.0}", avg_winner);
    println!("├─ Avg Loser:           ₹{:>12.0}", avg_loser);
    println!("├─ Expectancy:          ₹{:>12.0}", net_pnl / total_trades as f64);
    println!("└─ Total Charges:       ₹{:>12.0}", total_charges);
    
    println!("\nEXIT REASONS:");
    let mut sorted_exits: Vec<_> = exit_counts.iter().collect();
    sorted_exits.sort_by(|a, b| b.1.cmp(a.1));
    for (reason, count) in sorted_exits {
        let pct = (*count as f64 / total_trades as f64) * 100.0;
        println!("├─ {}: {} ({:.1}%)", reason, count, pct);
    }
    
    // Export to CSV
    let mut wtr = csv::Writer::from_path("confluence_orb_backtest_2L.csv").expect("Failed to create CSV");
    wtr.write_record(&["id", "date", "symbol", "type", "entry", "entry_time", "exit", "exit_time", "exit_reason", "qty", "position_value", "gross_pnl", "charges", "net_pnl", "capital_after"]).unwrap();
    
    for trade in trades {
        wtr.write_record(&[
            &trade.id, &trade.date, &trade.symbol, &trade.trade_type,
            &format!("{:.2}", trade.entry_price), &format!("{}", trade.entry_time),
            &format!("{:.2}", trade.exit_price), &format!("{}", trade.exit_time),
            &trade.exit_reason, &format!("{}", trade.quantity),
            &format!("{:.0}", trade.position_value), &format!("{:.2}", trade.gross_pnl),
            &format!("{:.2}", trade.charges), &format!("{:.2}", trade.net_pnl),
            &format!("{:.0}", trade.capital_after),
        ]).unwrap();
    }
    wtr.flush().unwrap();
    println!("\n✅ Results saved to: confluence_orb_backtest_2L.csv");
}

fn main() {
    run_backtest();
}
