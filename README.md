# Trading System - Animated Fortnight 🚀

A semi-automated trading system for Indian markets (NSE) using Zerodha Kite Connect API.

## Features

- **Semi-Auto Mode**: System suggests trades, you approve before execution
- **Multiple Strategies**: VWAP+RSI, EMA Crossover, Supertrend, ORB, Aura V14
- **Risk Management**: Fixed risk per trade (₹600), SL/Target auto-placement
- **Live Monitoring**: Continuous position tracking with SL/Target hit detection
- **Local Data Bank**: Historical data stored locally for fast backtesting
- **Comprehensive Logging**: All trades, signals, and API calls logged

## Architecture

```
├── kite_login.py           # Kite Connect authentication
├── kite_trading_engine.py  # Main trading engine (semi-auto mode)
├── kite_monitor.py         # Position monitoring script
├── backtest_vwap_rsi.py    # VWAP+RSI strategy backtest
├── backtest_aura_v14.py    # Aura V14 strategy backtest
├── create_data_bank.py     # Historical data fetcher & storage
├── TRADING_RULEBOOK.md     # Trading rules and guidelines
└── data_bank/              # Local historical data (not in git)
    ├── day/                # Daily OHLCV (5.5 years)
    └── 60minute/           # Hourly OHLCV (13 months)
```

## Setup

### 1. Install Dependencies

```bash
pip install --user kiteconnect pandas numpy pytz
```

### 2. Configure Kite Connect Credentials

Create `.kite_creds.json`:
```json
{
  "api_key": "YOUR_API_KEY",
  "api_secret": "YOUR_API_SECRET"
}
```

### 3. Login to Kite

```bash
python3 kite_login.py
```

### 4. Run Trading Engine

```bash
python3 kite_trading_engine.py
```

## Trading Strategies

### 1. VWAP + RSI Reversal
- **BUY**: Price < VWAP × 0.995 AND RSI < 30
- **SELL**: Price > VWAP × 1.005 AND RSI > 70
- **Best on**: Daily timeframe

### 2. EMA 9/21 Crossover
- **BUY**: EMA9 crosses above EMA21
- **SELL**: EMA9 crosses below EMA21

### 3. Supertrend Scalp
- Trend-following with ATR-based bands

### 4. Opening Range Breakout (ORB)
- First 15-minute high/low breakout

### 5. Aura V14 (Advanced)
Components:
- Alpha Trend (MFI-based)
- Magic Trend (CCI-based)
- UT Bot (ATR trailing stop)
- Consensus Engine (RSI, MFI, ADX, EMA50)
- Volume Filter

## Risk Management Rules

| Parameter | Value |
|-----------|-------|
| Capital | ₹30,000 |
| Max Risk per Trade | ₹600 (2%) |
| Max Daily Loss | ₹1,500 (5%) |
| Min Risk:Reward | 1:2 |
| Max Concurrent Trades | 3 |

## Data Bank

Local storage of historical data to avoid repeated API calls:

| Timeframe | Stocks | Candles | Period |
|-----------|--------|---------|--------|
| Daily | 192 | 259,615 | 5.5 years |
| Hourly | 192 | 364,416 | 13 months |

### Usage

```python
import pandas as pd

# Read from local data bank (no API calls!)
df = pd.read_parquet('data_bank/day/RELIANCE.parquet')
df = pd.read_parquet('data_bank/60minute/TCS.parquet')
```

## Backtest Results

### VWAP+RSI Strategy (1 Year, 35 F&O Stocks)

| Timeframe | Trades | Win% | Net P&L | ROI |
|-----------|--------|------|---------|-----|
| 5 Min | 14,070 | 37.7% | -₹3.31L | -1103% |
| 15 Min | 7,802 | 37.6% | -₹2.11L | -702% |
| 1 Hour | 3,345 | 41.7% | -₹76K | -253% |
| **Daily** | 382 | 41.4% | **+₹39.5K** | **+131%** |

**Conclusion**: VWAP+RSI only profitable on Daily timeframe.

## Logs

All activity is logged to `/logs/`:
- `trading_YYYY-MM-DD.log` - Trade execution
- `api_YYYY-MM-DD.log` - API calls
- `signals_YYYY-MM-DD.log` - Signal generation
- `orders_YYYY-MM-DD.log` - Order management
- `login_YYYY-MM-DD.log` - Authentication

## Safety Features

1. **Mandatory SL Orders**: Every trade has SL placed with broker
2. **Emergency Exit**: If SL placement fails, position is closed immediately
3. **Margin Check**: Available funds verified before suggesting trades
4. **OCO Logic**: When SL/Target hit, opposite order is cancelled
5. **Daily Loss Limit**: Trading stops if daily loss exceeds 5%

## Disclaimer

This is for educational purposes only. Trading involves significant risk of loss. Past performance does not guarantee future results. Always consult a financial advisor before trading.

## License

MIT License
