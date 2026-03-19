use chrono::{NaiveDateTime, Timelike};
use csv::ReaderBuilder;
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;

const START_CAPITAL: f64 = 200000.0;
const RISK_PCT: f64 = 1.0;
const LOT_SIZE: i32 = 50;
const MAX_POS_PCT: f64 = 0.20;
const EMA_FAST: usize = 9;
const EMA_SLOW: usize = 15;
const ADX_PERIOD: usize = 14;

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

fn calculate_adx(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
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
    
    // Simplified ATR for ADX
    let mut tr_values = Vec::new();
    for i in 1..len {
        let tr1 = highs[i] - lows[i];
        let tr2 = (highs[i] - closes[i-1]).abs();
        let tr3 = (lows[i] - closes[i-1]).abs();
        let tr = tr1.max(tr2).max(tr3);
        tr_values.push(tr);
    }
    
    let mut tr_sum: f64 = tr_values.iter().take(period).sum();
    let mut atr = tr_sum / period as f64;
    
    let mut plus_di_sum: f64 = plus_dm.iter().skip(1).take(period).sum();
    let mut minus_di_sum: f64 = minus_dm.iter().skip(1).take(period).sum();
    
    if atr > 0.0 {
        let plus_di = 100.0 * plus_di_sum / atr;
        let minus_di = 100.0 * minus_di_sum / atr;
        let dx = if (plus_di + minus_di) > 0.0 {
            100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        } else { 0.0 };
        adx[period] = dx;
    }
    
    for i in (period+1)..len {
        atr = (atr * (period as f64 - 1.0) + tr_values[i-1]) / period as f64;
        plus_di_sum = plus_di_sum - (plus_di_sum / period as f64) + plus_dm[i];
        minus_di_sum = minus_di_sum - (minus_di_sum / period as f64) + minus_dm[i];
        
        if atr > 0.0 {
            let plus_di = 100.0 * plus_di_sum / atr;
            let minus_di = 100.0 * minus_di_sum / atr;
            let dx = if (plus_di + minus_di) > 0.0 {
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

fn run_backtest() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║     EMA 9/15 CROSSOVER STRATEGY - ₹2L Capital            ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    
    let data = load_data("../trading_data_repo/data/nifty_200_15min");
    println!("Loaded {} stocks\n", data.len());
    
    let mut capital = START_CAPITAL;
    let mut trades: Vec<Trade> = Vec::new();
    
    let pb = ProgressBar::new(data.len() as u64);
    pb.set_style(ProgressStyle::default_bar().template("[{bar:40}] {pos}/{len}").unwrap());
    
    for (symbol, candles) in &data {
        let closes: Vec<f64> = candles.iter().map(|(_, c)| c.close).collect();
        let highs: Vec<f64> = candles.iter().map(|(_, c)| c.high).collect();
        let lows: Vec<f64> = candles.iter().map(|(_, c)| c.low).collect();
        let volumes: Vec<f64> = candles.iter().map(|(_, c)| c.volume).collect();
        
        let ema9 = calculate_ema(&closes, EMA_FAST);
        let ema15 = calculate_ema(&closes, EMA_SLOW);
        let adx = calculate_adx(&highs, &lows, &closes, ADX_PERIOD);
        let vol_sma = calculate_ema(&volumes, 20);
        
        let mut last_date = String::new();
        let mut in_trade = false;
        let mut entry_price = 0.0;
        let mut direction = String::new();
        let mut qty = 0;
        let mut sl = 0.0;
        let mut target = 0.0;
        let mut entry_idx = 0;
        
        for i in 50..candles.len() {
            let (dt, candle) = &candles[i];
            let date = dt.date().to_string();
            let time = dt.time();
            let hhmm = time.hour() * 100 + time.minute();
            
            // New day
            if date != last_date {
                if in_trade {
                    // Force close at EOD
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
                        });
                    }
                }
                last_date = date;
                in_trade = false;
            }
            
            // Skip if no indicators
            if ema9[i].is_nan() || ema15[i].is_nan() || adx[i].is_nan() || vol_sma[i].is_nan() {
                continue;
            }
            
            // Check for exit if in trade
            if in_trade {
                // Opposite crossover exit
                let cross_exit = if direction == "LONG" {
                    ema9[i] < ema15[i] && ema9[i-1] >= ema15[i-1]
                } else {
                    ema9[i] > ema15[i] && ema9[i-1] <= ema15[i-1]
                };
                
                // SL hit
                let sl_hit = if direction == "LONG" {
                    candle.low <= sl
                } else {
                    candle.high >= sl
                };
                
                // Target hit
                let tgt_hit = if direction == "LONG" {
                    candle.high >= target
                } else {
                    candle.low <= target
                };
                
                // Time exit
                let time_exit = hhmm >= 1515;
                
                if cross_exit || sl_hit || tgt_hit || time_exit {
                    let (exit, reason) = if tgt_hit {
                        (target, "TARGET")
                    } else if sl_hit {
                        (sl, "STOP_LOSS")
                    } else if time_exit {
                        (candle.close, "TIME")
                    } else {
                        (candle.close, "CROSSOVER")
                    };
                    
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
                            exit_reason: reason.to_string(),
                        });
                    }
                    
                    in_trade = false;
                }
                continue;
            }
            
            // Entry check - only first 2.5 hours (9:15-11:45)
            if !in_trade && capital > 50000.0 && hhmm >= 915 && hhmm <= 1145 {
                // Volume filter
                let vol_ok = candle.volume > vol_sma[i] * 1.2;
                if !vol_ok { continue; }
                
                // ADX filter - must be trending
                let adx_ok = adx[i] > 25.0;
                if !adx_ok { continue; }
                
                // EMA Crossover
                let cross_up = ema9[i] > ema15[i] && ema9[i-1] <= ema15[i-1];
                let cross_down = ema9[i] < ema15[i] && ema9[i-1] >= ema15[i-1];
                
                if cross_up {
                    direction = "LONG".to_string();
                    entry_price = candle.close;
                    
                    // Find recent swing low for SL
                    let mut swing_low = f64::MAX;
                    for j in (i.saturating_sub(5)..i).rev() {
                        swing_low = swing_low.min(candles[j].1.low);
                    }
                    sl = swing_low * 0.995;  // Slightly below
                    let r_dist = entry_price - sl;
                    if r_dist <= 0.0 { continue; }
                    
                    target = entry_price + r_dist * 2.0;  // 1:2 RR
                    
                    let risk_amt = capital * RISK_PCT / 100.0;
                    let shares = ((risk_amt / r_dist) as i32).min((capital * MAX_POS_PCT / entry_price) as i32);
                    qty = (shares / LOT_SIZE) * LOT_SIZE;
                    
                    if qty >= LOT_SIZE {
                        in_trade = true;
                        entry_idx = i;
                    }
                } else if cross_down {
                    direction = "SHORT".to_string();
                    entry_price = candle.close;
                    
                    // Find recent swing high for SL
                    let mut swing_high: f64 = 0.0;
                    for j in (i.saturating_sub(5)..i).rev() {
                        if candles[j].1.high > swing_high {
                            swing_high = candles[j].1.high;
                        }
                    }
                    sl = swing_high * 1.005;  // Slightly above
                    let r_dist = sl - entry_price;
                    if r_dist <= 0.0 { continue; }
                    
                    target = entry_price - r_dist * 2.0;  // 1:2 RR
                    
                    let risk_amt = capital * RISK_PCT / 100.0;
                    let shares = ((risk_amt / r_dist) as i32).min((capital * MAX_POS_PCT / entry_price) as i32);
                    qty = (shares / LOT_SIZE) * LOT_SIZE;
                    
                    if qty >= LOT_SIZE {
                        in_trade = true;
                        entry_idx = i;
                    }
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
    
    println!("\nExit Reasons:");
    for (reason, count) in exit_counts.iter() {
        println!("  {}: {} ({:.1}%)", reason, count, (*count as f64 / total as f64) * 100.0);
    }
    
    // Save CSV
    let mut wtr = csv::Writer::from_path("trades.csv").unwrap();
    wtr.write_record(&["date", "symbol", "direction", "entry", "exit", "qty", "pnl", "charges", "net_pnl", "capital_after", "exit_reason"]).unwrap();
    for t in trades {
        wtr.write_record(&[
            &t.date, &t.symbol, &t.direction,
            &format!("{:.2}", t.entry), &format!("{:.2}", t.exit),
            &format!("{}", t.qty), &format!("{:.2}", t.pnl),
            &format!("{:.2}", t.charges), &format!("{:.2}", t.net_pnl),
            &format!("{:.0}", t.capital_after), &t.exit_reason,
        ]).unwrap();
    }
    wtr.flush().unwrap();
    println!("\n✅ Results saved to: trades.csv");
}

fn main() {
    run_backtest();
}
