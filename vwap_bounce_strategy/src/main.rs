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
    println!("║     VWAP BOUNCE STRATEGY - ₹2L Capital                     ║");
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
        
        let ema9 = calculate_ema(&closes, 9);
        let ema21 = calculate_ema(&closes, 21);
        let atr = calculate_atr(&highs, &lows, &closes, 14);
        
        let mut last_date = String::new();
        let mut trades_today = 0;
        let mut in_trade = false;
        let mut entry_price = 0.0;
        let mut direction = String::new();
        let mut qty = 0;
        let mut sl = 0.0;
        let mut target = 0.0;
        let mut vwap_sum_pv = 0.0;
        let mut vwap_sum_v = 0.0;
        
        for i in 50..candles.len() {
            let (dt, candle) = &candles[i];
            let date = dt.date().to_string();
            let time = dt.time();
            let hhmm = time.hour() * 100 + time.minute();
            
            // New day - reset VWAP
            if date != last_date {
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
                trades_today = 0;
                in_trade = false;
                vwap_sum_pv = 0.0;
                vwap_sum_v = 0.0;
            }
            
            // Update VWAP calculation
            let tp = (candle.high + candle.low + candle.close) / 3.0;
            vwap_sum_pv += tp * candle.volume;
            vwap_sum_v += candle.volume;
            let vwap = if vwap_sum_v > 0.0 { vwap_sum_pv / vwap_sum_v } else { f64::NAN };
            
            // Skip if no indicators
            if vwap.is_nan() || ema9[i].is_nan() || ema21[i].is_nan() || atr[i].is_nan() {
                continue;
            }
            
            // Check for exit
            if in_trade {
                let sl_hit = if direction == "LONG" { candle.low <= sl } else { candle.high >= sl };
                let tgt_hit = if direction == "LONG" { candle.high >= target } else { candle.low <= target };
                let time_exit = hhmm >= 1515;
                
                if sl_hit || tgt_hit || time_exit {
                    let (exit, reason) = if tgt_hit {
                        (target, "TARGET")
                    } else if sl_hit {
                        (sl, "STOP_LOSS")
                    } else {
                        (candle.close, "TIME")
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
                    trades_today += 1;
                }
                continue;
            }
            
            // Entry check - VWAP Bounce
            if !in_trade && trades_today < 2 && capital > 50000.0 && hhmm >= 945 && hhmm <= 1430 {
                let trend_up = ema9[i] > ema21[i];
                let trend_down = ema9[i] < ema21[i];
                let vwap_dev = ((candle.close - vwap) / vwap).abs();
                
                // Long: Price near VWAP in uptrend (within 1.5%)
                if trend_up && vwap_dev <= 0.015 && candle.close <= vwap * 1.01 {
                    direction = "LONG".to_string();
                    entry_price = candle.close;
                    let atr_val = atr[i];
                    sl = vwap - atr_val * 0.5;
                    let r_dist = entry_price - sl;
                    if r_dist <= 0.0 { continue; }
                    target = entry_price + r_dist * 2.0;
                    
                    let risk_amt = capital * RISK_PCT / 100.0;
                    let shares = ((risk_amt / r_dist) as i32).min((capital * MAX_POS_PCT / entry_price) as i32);
                    qty = (shares / LOT_SIZE) * LOT_SIZE;
                    
                    if qty >= LOT_SIZE {
                        in_trade = true;
                    }
                }
                // Short: Price near VWAP in downtrend
                else if trend_down && vwap_dev <= 0.015 && candle.close >= vwap * 0.99 {
                    direction = "SHORT".to_string();
                    entry_price = candle.close;
                    let atr_val = atr[i];
                    sl = vwap + atr_val * 0.5;
                    let r_dist = sl - entry_price;
                    if r_dist <= 0.0 { continue; }
                    target = entry_price - r_dist * 2.0;
                    
                    let risk_amt = capital * RISK_PCT / 100.0;
                    let shares = ((risk_amt / r_dist) as i32).min((capital * MAX_POS_PCT / entry_price) as i32);
                    qty = (shares / LOT_SIZE) * LOT_SIZE;
                    
                    if qty >= LOT_SIZE {
                        in_trade = true;
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
    
    let mut max_dd = 0.0;
    let mut peak = START_CAPITAL;
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
