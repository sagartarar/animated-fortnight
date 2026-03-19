//! Initial Balance Breakout Strategy Backtest
//!
//! Strategy Rules:
//! - IB: First 2 candles (9:15-9:45 AM)
//! - Entry: Close above IB High (long) or below IB Low (short)
//! - Volume filter: Volume > 1.5x 20-period average
//! - Stop: Opposite side of IB range
//! - Target: 1.5x IB range from breakout
//! - Time exit: 3:15 PM
//! - 1 trade per stock per day, 1.5% risk per trade

use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use csv::ReaderBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const STARTING_CAPITAL: f64 = 200_000.0;
const LOT_SIZE: u32 = 50;
const RISK_PCT: f64 = 0.015;
const VOLUME_MULTIPLIER: f64 = 1.5;
const TARGET_MULTIPLIER: f64 = 1.5;
const VOL_AVG_PERIOD: usize = 20;

#[derive(Debug, Clone, Deserialize)]
struct Candle {
    date: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

#[derive(Debug, Clone)]
struct ParsedCandle {
    datetime: NaiveDateTime,
    date: NaiveDate,
    #[allow(dead_code)]
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Trade {
    symbol: String,
    date: String,
    side: String,
    entry_time: String,
    entry_price: f64,
    exit_time: String,
    exit_price: f64,
    quantity: u32,
    pnl: f64,
    pnl_pct: f64,
    exit_reason: String,
}

fn parse_datetime(s: &str) -> Option<NaiveDateTime> {
    // Format: 2015-02-02 09:15:00+05:30 (IST)
    let s = s.trim();
    if s.len() < 19 {
        return None;
    }
    let date_part = &s[..10];
    let time_part = &s[11..19];
    let date = NaiveDate::parse_from_str(date_part, "%Y-%m-%d").ok()?;
    let time = NaiveTime::parse_from_str(time_part, "%H:%M:%S").ok()?;
    Some(NaiveDateTime::new(date, time))
}

fn load_stock_data(path: &Path) -> Vec<ParsedCandle> {
    let mut candles = Vec::new();
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", path, e));

    for result in reader.deserialize() {
        let rec: Candle = match result {
            Ok(r) => r,
            Err(_) => continue,
        };
        let datetime = match parse_datetime(&rec.date) {
            Some(dt) => dt,
            None => continue,
        };
        let date = datetime.date();
        candles.push(ParsedCandle {
            datetime,
            date,
            open: rec.open,
            high: rec.high,
            low: rec.low,
            close: rec.close,
            volume: rec.volume,
        });
    }
    candles
}

fn load_all_data(data_dir: &Path) -> HashMap<String, Vec<ParsedCandle>> {
    let mut data = HashMap::new();
    let entries = fs::read_dir(data_dir).expect("Failed to read data directory");

    for entry in entries {
        let entry = entry.expect("Failed to read dir entry");
        let path = entry.path();
        if path.extension().map(|e| e == "csv").unwrap_or(false) {
            let fname = path.file_stem().unwrap().to_string_lossy();
            let symbol = fname
                .strip_suffix("_15min")
                .unwrap_or(&fname)
                .to_string();
            let candles = load_stock_data(&path);
            if candles.len() >= 25 {
                data.insert(symbol, candles);
            }
        }
    }
    data
}

fn compute_vol_avg(candles: &[ParsedCandle], idx: usize) -> f64 {
    if idx < VOL_AVG_PERIOD {
        return 0.0;
    }
    let start = idx - VOL_AVG_PERIOD;
    let sum: f64 = candles[start..idx].iter().map(|c| c.volume).sum();
    sum / VOL_AVG_PERIOD as f64
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PositionSide {
    Long,
    Short,
}

fn run_backtest(data: &HashMap<String, Vec<ParsedCandle>>) -> (Vec<Trade>, f64) {
    let mut trades = Vec::new();
    let mut capital = STARTING_CAPITAL;
    let mut equity_curve = vec![capital];
    let mut traded_today: HashMap<NaiveDate, Vec<String>> = HashMap::new();

    // Collect all unique dates across all stocks
    let mut all_dates: Vec<NaiveDate> = data
        .values()
        .flat_map(|candles| candles.iter().map(|c| c.date))
        .collect();
    all_dates.sort();
    all_dates.dedup();

    for &trade_date in &all_dates {
        // Reset traded_today for this date
        traded_today.insert(trade_date, Vec::new());

        for (symbol, candles) in data.iter() {
            if traded_today.get(&trade_date).map(|v| v.contains(symbol)).unwrap_or(false) {
                continue;
            }

            let day_candles: Vec<_> = candles
                .iter()
                .filter(|c| c.date == trade_date)
                .collect();

            if day_candles.len() < 3 {
                continue;
            }

            // First 2 candles = 9:15 and 9:30 (IB period)
            let c0 = day_candles[0];
            let c1 = day_candles[1];
            let ib_high = c0.high.max(c1.high);
            let ib_low = c0.low.min(c1.low);
            let ib_range = ib_high - ib_low;

            if ib_range <= 0.0 {
                continue;
            }

            // Find index of first candle in full series for volume avg
            let first_idx = candles.iter().position(|c| c.date == trade_date).unwrap_or(0);

            for (i, &c) in day_candles.iter().enumerate() {
                if i < 2 {
                    continue;
                }
                let time_str = c.datetime.format("%H:%M").to_string();
                if time_str == "15:15" {
                    break;
                }

                let idx = first_idx + i;
                let vol_avg = compute_vol_avg(candles, idx);
                if vol_avg <= 0.0 || c.volume <= vol_avg * VOLUME_MULTIPLIER {
                    continue;
                }

                let mut position_side = None;
                let mut entry_price = 0.0;
                let mut stop = 0.0;
                let mut target = 0.0;

                if c.close > ib_high {
                    position_side = Some(PositionSide::Long);
                    entry_price = c.close;
                    stop = ib_low;
                    target = entry_price + ib_range * TARGET_MULTIPLIER;
                } else if c.close < ib_low {
                    position_side = Some(PositionSide::Short);
                    entry_price = c.close;
                    stop = ib_high;
                    target = entry_price - ib_range * TARGET_MULTIPLIER;
                }

                if let Some(side) = position_side {
                    let risk_amount = STARTING_CAPITAL * RISK_PCT;
                    let risk_per_share = match side {
                        PositionSide::Long => entry_price - stop,
                        PositionSide::Short => stop - entry_price,
                    };

                    if risk_per_share <= 0.0 {
                        continue;
                    }

                    let raw_qty = (risk_amount / risk_per_share) as u32;
                    let quantity = ((raw_qty / LOT_SIZE) * LOT_SIZE).max(LOT_SIZE);

                    let mut exit_price = 0.0;
                    let mut exit_reason = String::new();
                    let mut exit_time = String::new();

                    for &c2 in day_candles.iter().skip(i + 1) {
                        let t = c2.datetime.format("%H:%M").to_string();
                        let hit_stop = match side {
                            PositionSide::Long => c2.low <= stop,
                            PositionSide::Short => c2.high >= stop,
                        };
                        let hit_target = match side {
                            PositionSide::Long => c2.high >= target,
                            PositionSide::Short => c2.low <= target,
                        };

                        if hit_stop && hit_target {
                            if side == PositionSide::Long {
                                if c2.low <= stop {
                                    exit_price = stop;
                                    exit_reason = "stop_loss".to_string();
                                } else {
                                    exit_price = target;
                                    exit_reason = "target".to_string();
                                }
                            } else {
                                if c2.high >= stop {
                                    exit_price = stop;
                                    exit_reason = "stop_loss".to_string();
                                } else {
                                    exit_price = target;
                                    exit_reason = "target".to_string();
                                }
                            }
                            exit_time = t;
                            break;
                        } else if hit_stop {
                            exit_price = stop;
                            exit_reason = "stop_loss".to_string();
                            exit_time = t;
                            break;
                        } else if hit_target {
                            exit_price = target;
                            exit_reason = "target".to_string();
                            exit_time = t;
                            break;
                        } else if t == "15:15" {
                            exit_price = c2.close;
                            exit_reason = "time_exit".to_string();
                            exit_time = t;
                            break;
                        }
                    }

                    if exit_price == 0.0 {
                        if let Some(&last) = day_candles.last() {
                            exit_price = last.close;
                            exit_reason = "time_exit".to_string();
                            exit_time = last.datetime.format("%H:%M").to_string();
                        } else {
                            continue;
                        }
                    }

                    let pnl = match side {
                        PositionSide::Long => (exit_price - entry_price) * quantity as f64,
                        PositionSide::Short => (entry_price - exit_price) * quantity as f64,
                    };
                    let pnl_pct = (pnl / (entry_price * quantity as f64)) * 100.0;

                    capital += pnl;
                    equity_curve.push(capital);

                    trades.push(Trade {
                        symbol: symbol.clone(),
                        date: trade_date.format("%Y-%m-%d").to_string(),
                        side: match side {
                            PositionSide::Long => "LONG".to_string(),
                            PositionSide::Short => "SHORT".to_string(),
                        },
                        entry_time: c.datetime.format("%H:%M").to_string(),
                        entry_price,
                        exit_time,
                        exit_price,
                        quantity,
                        pnl,
                        pnl_pct,
                        exit_reason,
                    });

                    traded_today
                        .get_mut(&trade_date)
                        .unwrap()
                        .push(symbol.clone());
                    break;
                }
            }
        }
    }

    (trades, *equity_curve.last().unwrap_or(&STARTING_CAPITAL))
}

fn main() {
    let data_dir = Path::new("/u/tarar/repos/trading_data_repo/data/nifty_200_15min");
    println!("Loading data from {:?}...", data_dir);

    let data = load_all_data(data_dir);
    println!("Loaded {} stocks", data.len());

    let (trades, final_capital) = run_backtest(&data);

    let total_trades = trades.len();
    let winners = trades.iter().filter(|t| t.pnl > 0.0).count();
    let win_rate = if total_trades > 0 {
        (winners as f64 / total_trades as f64) * 100.0
    } else {
        0.0
    };
    let net_pnl = final_capital - STARTING_CAPITAL;
    let gross_profit: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
    let gross_loss: f64 = trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl.abs()).sum();
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    let mut peak = STARTING_CAPITAL;
    let mut max_dd = 0.0;
    let mut eq = STARTING_CAPITAL;
    for t in &trades {
        eq += t.pnl;
        if eq > peak {
            peak = eq;
        }
        let dd = (peak - eq) / peak * 100.0;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    println!("\n========== BACKTEST RESULTS ==========");
    println!("Total Trades:     {}", total_trades);
    println!("Win Rate:         {:.2}%", win_rate);
    println!("Net P&L:         ₹{:.2}", net_pnl);
    println!("Final Capital:   ₹{:.2}", final_capital);
    println!("Max Drawdown:    {:.2}%", max_dd);
    println!("Profit Factor:   {:.2}", profit_factor);
    println!("=====================================\n");

    let csv_path = "/u/tarar/repos/ib_breakout_strategy/trades.csv";
    let mut wtr = csv::Writer::from_path(csv_path).expect("Failed to create CSV");
    for t in &trades {
        wtr.serialize(t).unwrap();
    }
    wtr.flush().unwrap();
    println!("Trades written to {}", csv_path);
}
