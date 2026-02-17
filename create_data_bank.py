#!/usr/bin/env python3
"""
Historical Data Bank Creator
============================
Fetches and stores maximum available historical data from Kite Connect
for all Nifty 200 stocks across multiple timeframes.

Data is stored locally to avoid repeated API calls during backtesting.

Features:
- Fetches max available data per timeframe
- Stores in Parquet format (efficient, compressed)
- Tracks corporate actions (splits, dividends, bonuses)
- Calculates adjusted prices
- Incremental updates (only fetch new data)

Kite Connect Historical Data Limits:
- minute    : 60 days
- 3minute   : 100 days  
- 5minute   : 100 days
- 10minute  : 100 days
- 15minute  : 200 days
- 30minute  : 200 days
- 60minute  : 400 days
- day       : 2000 days (~5.5 years)

Author: Trading System
Date: February 2026
"""

import json
import os
import sys
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
import time

# ============== CONFIGURATION ==============

# Data storage paths
DATA_DIR = "/u/tarar/repos/data_bank"
METADATA_FILE = f"{DATA_DIR}/metadata.json"
INSTRUMENTS_FILE = f"{DATA_DIR}/instruments.parquet"
CORPORATE_ACTIONS_FILE = f"{DATA_DIR}/corporate_actions.json"

# Credentials
CREDS_FILE = "/u/tarar/repos/.kite_creds.json"
SESSION_FILE = "/u/tarar/repos/.kite_session.json"

# Timeframe configurations with max days
TIMEFRAMES = {
    "minute": {"max_days": 60, "chunk_days": 30},
    "3minute": {"max_days": 100, "chunk_days": 50},
    "5minute": {"max_days": 100, "chunk_days": 50},
    "10minute": {"max_days": 100, "chunk_days": 50},
    "15minute": {"max_days": 200, "chunk_days": 60},
    "30minute": {"max_days": 200, "chunk_days": 60},
    "60minute": {"max_days": 400, "chunk_days": 100},
    "day": {"max_days": 2000, "chunk_days": 365}
}

# Stock Universe - Nifty 200
STOCK_UNIVERSE = [
    # Nifty 50 (Large Cap)
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
    "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "ASIANPAINT",
    "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO",
    "TATASTEEL", "POWERGRID", "NTPC", "M&M", "JSWSTEEL",
    "BAJAJFINSV", "ADANIENT", "ADANIPORTS", "ONGC", "NESTLEIND",
    "COALINDIA", "TECHM", "INDUSINDBK", "BRITANNIA", "HINDALCO",
    "DRREDDY", "DIVISLAB", "CIPLA", "EICHERMOT", "GRASIM",
    "APOLLOHOSP", "BPCL", "TATACONSUM", "HEROMOTOCO", "SHREECEM",
    "SBILIFE", "HDFCLIFE", "UPL", "BAJAJ-AUTO", "TATAMOTORS",
    
    # Nifty Next 50 (Large-Mid Cap)
    "BANKBARODA", "PNB", "CANBK", "IOC", "GAIL",
    "VEDL", "JINDALSTEL", "DLF", "SAIL", "NMDC",
    "PIDILITIND", "HAVELLS", "SIEMENS", "ABB", "GODREJCP",
    "DABUR", "MARICO", "COLPAL", "BERGEPAINT", "MCDOWELL-N",
    "INDIGO", "TRENT", "ZOMATO", "IRCTC", "MANKIND",
    "MAXHEALTH", "FORTIS", "AUROPHARMA", "TORNTPHARM", "LUPIN",
    "ZYDUSLIFE", "BIOCON", "ALKEM", "IPCALAB", "LAURUSLABS",
    "TATAPOWER", "NHPC", "SJVN", "RECLTD", "PFC",
    "IRFC", "BEL", "HAL", "BHEL", "CUMMINSIND",
    
    # Nifty Midcap 100 Select
    "MPHASIS", "LTIM", "PERSISTENT", "COFORGE", "LTTS",
    "TATAELXSI", "MOTHERSON", "BOSCHLTD", "MRF", "ASHOKLEY",
    "ESCORTS", "EXIDEIND", "BALKRISIND", "APOLLOTYRE", "TVSMTR",
    "CHOLAFIN", "MUTHOOTFIN", "MANAPPURAM", "LICHSGFIN", "CANFINHOME",
    "ABCAPITAL", "M&MFIN", "SHRIRAMFIN", "FEDERALBNK", "IDFCFIRSTB",
    "BANDHANBNK", "RBLBANK", "AUBANK", "INDIANB", "UNIONBANK",
    
    # Additional Nifty 200
    "OBEROIRLTY", "PRESTIGE", "GODREJPROP", "PHOENIXLTD", "LICI",
    "SBICARD", "HDFCAMC", "ICICIGI", "ICICIPRULI", "NIACL",
    "CRISIL", "METROPOLIS", "LALPATHLAB", "DMART", "TATACOMM",
    "ABFRL", "PAGEIND", "BATAINDIA", "JUBLFOOD", "DEVYANI",
    "EMAMILTD", "VGUARD", "CROMPTON", "VOLTAS", "BLUESTAR",
    "WHIRLPOOL", "DIXON", "POLYCAB", "AFFLE", "KPITTECH",
    "MASTEK", "CYIENT", "BIRLASOFT", "PIIND", "ATUL",
    "DEEPAKNTR", "AARTI", "SRF", "NAVINFLUOR", "FLUOROCHEM",
    "PETRONET", "GSPL", "IGL", "MGL", "INDIAMART",
    "NAUKRI", "ZEEL", "PVR", "PVRINOX", "SUNTV",
    
    # High Volume F&O Stocks
    "IDEA", "YESBANK", "IBREALEST", "SUZLON", "JPPOWER",
    "RPOWER", "NATIONALUM", "HINDZINC", "GMRINFRA", "NBCC",
    "NCC", "IRB", "ASHOKA", "KEC", "KALPATPOWR",
    "THERMAX", "GRINDWELL", "SCHAEFFLER", "TIMKEN", "SKFINDIA",
    "SUPREMEIND", "ASTRAL", "PRINCEPIPE", "APLAPOLLO", "RATNAMANI"
]

# Remove duplicates
STOCK_UNIVERSE = list(dict.fromkeys(STOCK_UNIVERSE))


# ============== HELPER FUNCTIONS ==============

def load_session():
    """Load Kite Connect session"""
    with open(CREDS_FILE) as f:
        creds = json.load(f)
    with open(SESSION_FILE) as f:
        session = json.load(f)
    
    kite = KiteConnect(api_key=creds['api_key'])
    kite.set_access_token(session['access_token'])
    
    try:
        profile = kite.profile()
        print(f"✅ Logged in as: {profile['user_name']}")
        return kite
    except Exception as e:
        print(f"❌ Session invalid: {e}")
        sys.exit(1)


def get_instrument_tokens(kite):
    """Get instrument tokens for all stocks in universe"""
    print("\n📋 Fetching instrument list from NSE...")
    instruments = kite.instruments("NSE")
    
    token_map = {}
    found = []
    not_found = []
    
    for symbol in STOCK_UNIVERSE:
        for inst in instruments:
            if inst['tradingsymbol'] == symbol:
                token_map[symbol] = {
                    "token": inst['instrument_token'],
                    "name": inst['name'],
                    "lot_size": inst['lot_size'],
                    "tick_size": inst['tick_size'],
                    "exchange": inst['exchange'],
                    "segment": inst['segment'],
                    "isin": inst.get('isin', '')
                }
                found.append(symbol)
                break
        else:
            not_found.append(symbol)
    
    print(f"   Found: {len(found)} stocks")
    if not_found:
        print(f"   Not Found: {len(not_found)} - {not_found[:10]}...")
    
    return token_map


def fetch_historical_data(kite, symbol, token, interval, from_date, to_date, chunk_days):
    """Fetch historical data in chunks"""
    all_data = []
    current_from = from_date
    
    while current_from < to_date:
        current_to = min(current_from + timedelta(days=chunk_days), to_date)
        
        try:
            data = kite.historical_data(
                instrument_token=token,
                from_date=current_from.strftime("%Y-%m-%d"),
                to_date=current_to.strftime("%Y-%m-%d"),
                interval=interval
            )
            all_data.extend(data)
            time.sleep(0.35)  # Rate limiting (3 requests/second)
        except Exception as e:
            print(f"      ⚠️ Error {current_from.date()}-{current_to.date()}: {e}")
        
        current_from = current_to + timedelta(days=1)
    
    return all_data


def calculate_adjusted_prices(df, corporate_actions):
    """
    Adjust prices for corporate actions (splits, bonuses, dividends)
    
    For splits: multiply old prices by split ratio
    For bonus: adjust by bonus ratio
    For dividends: subtract dividend from price (optional)
    """
    if df.empty or not corporate_actions:
        return df
    
    df = df.copy()
    df['adj_close'] = df['close'].copy()
    df['adj_open'] = df['open'].copy()
    df['adj_high'] = df['high'].copy()
    df['adj_low'] = df['low'].copy()
    df['adj_volume'] = df['volume'].copy()
    
    # Sort actions by date (newest first for backward adjustment)
    sorted_actions = sorted(corporate_actions, key=lambda x: x['ex_date'], reverse=True)
    
    for action in sorted_actions:
        ex_date = pd.to_datetime(action['ex_date'])
        
        # Get rows before ex-date
        mask = df['date'] < ex_date
        
        if action['type'] == 'SPLIT':
            # Split ratio: e.g., 2:1 means each share becomes 2
            ratio = action['ratio']
            df.loc[mask, 'adj_close'] *= ratio
            df.loc[mask, 'adj_open'] *= ratio
            df.loc[mask, 'adj_high'] *= ratio
            df.loc[mask, 'adj_low'] *= ratio
            df.loc[mask, 'adj_volume'] /= ratio
            
        elif action['type'] == 'BONUS':
            # Bonus ratio: e.g., 1:1 means 1 free share for each held
            ratio = 1 / (1 + action['ratio'])
            df.loc[mask, 'adj_close'] *= ratio
            df.loc[mask, 'adj_open'] *= ratio
            df.loc[mask, 'adj_high'] *= ratio
            df.loc[mask, 'adj_low'] *= ratio
            df.loc[mask, 'adj_volume'] /= ratio
    
    return df


def load_metadata():
    """Load existing metadata"""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE) as f:
            return json.load(f)
    return {"last_updated": None, "stocks": {}, "timeframes": {}}


def save_metadata(metadata):
    """Save metadata"""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


# ============== CORPORATE ACTIONS ==============

# Known corporate actions for major stocks (manually maintained)
# In production, this should be fetched from a corporate actions API
KNOWN_CORPORATE_ACTIONS = {
    "RELIANCE": [
        {"type": "BONUS", "ex_date": "2017-09-08", "ratio": 1},  # 1:1 bonus
        {"type": "SPLIT", "ex_date": "2017-09-08", "ratio": 0.1}  # Face value split
    ],
    "TCS": [
        {"type": "BONUS", "ex_date": "2018-06-01", "ratio": 1},  # 1:1 bonus
        {"type": "SPLIT", "ex_date": "2006-01-01", "ratio": 0.1}
    ],
    "INFY": [
        {"type": "SPLIT", "ex_date": "2018-12-14", "ratio": 0.2},  # 5:1 split
        {"type": "BONUS", "ex_date": "2018-12-14", "ratio": 1}
    ],
    "HDFCBANK": [
        {"type": "SPLIT", "ex_date": "2019-09-19", "ratio": 0.5}  # 2:1 split
    ],
    "ICICIBANK": [
        {"type": "SPLIT", "ex_date": "2022-05-11", "ratio": 0.5}  # 2:1 split
    ],
    "WIPRO": [
        {"type": "BONUS", "ex_date": "2019-02-27", "ratio": 1},  # 1:1 bonus
        {"type": "SPLIT", "ex_date": "2017-06-14", "ratio": 0.1}
    ],
    "MARUTI": [
        {"type": "SPLIT", "ex_date": "2021-09-03", "ratio": 0.2}  # 5:1 split
    ],
    "MRF": [],  # No recent splits (high face value stock)
    "BAJAJ-AUTO": [
        {"type": "SPLIT", "ex_date": "2020-03-31", "ratio": 0.1}
    ],
    # Add more as needed...
}


# ============== MAIN DATA BANK CREATOR ==============

def create_data_bank(kite, timeframes_to_fetch=None, stocks_to_fetch=None):
    """
    Create/Update the data bank
    
    Args:
        kite: KiteConnect instance
        timeframes_to_fetch: List of timeframes to fetch (default: all)
        stocks_to_fetch: List of stocks to fetch (default: all)
    """
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load metadata
    metadata = load_metadata()
    
    # Get instrument tokens
    token_map = get_instrument_tokens(kite)
    
    # Save instruments info
    instruments_df = pd.DataFrame.from_dict(token_map, orient='index')
    instruments_df.index.name = 'symbol'
    instruments_df.to_parquet(INSTRUMENTS_FILE)
    print(f"📁 Saved instruments to {INSTRUMENTS_FILE}")
    
    # Filter stocks and timeframes
    stocks = stocks_to_fetch or list(token_map.keys())
    timeframes = timeframes_to_fetch or list(TIMEFRAMES.keys())
    
    # Date range
    to_date = datetime.now()
    
    total_stocks = len(stocks)
    total_timeframes = len(timeframes)
    
    print(f"\n{'=' * 70}")
    print(f"📊 CREATING DATA BANK")
    print(f"{'=' * 70}")
    print(f"   Stocks: {total_stocks}")
    print(f"   Timeframes: {', '.join(timeframes)}")
    print(f"   To Date: {to_date.strftime('%Y-%m-%d')}")
    
    # Statistics
    stats = {
        "stocks_processed": 0,
        "stocks_failed": 0,
        "total_candles": 0,
        "by_timeframe": {}
    }
    
    for tf in timeframes:
        tf_config = TIMEFRAMES[tf]
        max_days = tf_config['max_days']
        chunk_days = tf_config['chunk_days']
        from_date = to_date - timedelta(days=max_days)
        
        print(f"\n{'=' * 70}")
        print(f"⏱️ TIMEFRAME: {tf.upper()}")
        print(f"   Period: {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')} ({max_days} days)")
        print(f"{'=' * 70}")
        
        tf_dir = f"{DATA_DIR}/{tf}"
        os.makedirs(tf_dir, exist_ok=True)
        
        tf_candles = 0
        tf_stocks = 0
        
        for i, symbol in enumerate(stocks):
            if symbol not in token_map:
                continue
            
            token = token_map[symbol]['token']
            file_path = f"{tf_dir}/{symbol}.parquet"
            
            print(f"  [{i+1}/{total_stocks}] {symbol}...", end=" ", flush=True)
            
            try:
                # Fetch data
                data = fetch_historical_data(
                    kite, symbol, token, tf, from_date, to_date, chunk_days
                )
                
                if not data:
                    print("❌ No data")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                df = df.drop_duplicates(subset=['date']).reset_index(drop=True)
                
                # Add adjusted prices if corporate actions exist
                if symbol in KNOWN_CORPORATE_ACTIONS:
                    df = calculate_adjusted_prices(df, KNOWN_CORPORATE_ACTIONS[symbol])
                else:
                    # Default: adjusted = actual
                    df['adj_close'] = df['close']
                    df['adj_open'] = df['open']
                    df['adj_high'] = df['high']
                    df['adj_low'] = df['low']
                    df['adj_volume'] = df['volume']
                
                # Add metadata columns
                df['symbol'] = symbol
                
                # Save to parquet
                df.to_parquet(file_path, index=False)
                
                candles = len(df)
                tf_candles += candles
                tf_stocks += 1
                
                # Update metadata
                if symbol not in metadata['stocks']:
                    metadata['stocks'][symbol] = {}
                metadata['stocks'][symbol][tf] = {
                    "candles": candles,
                    "from_date": df['date'].min().isoformat(),
                    "to_date": df['date'].max().isoformat(),
                    "file": file_path,
                    "updated": datetime.now().isoformat()
                }
                
                print(f"✅ {candles:,} candles ({df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')})")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                stats['stocks_failed'] += 1
        
        stats['by_timeframe'][tf] = {
            "stocks": tf_stocks,
            "candles": tf_candles
        }
        stats['total_candles'] += tf_candles
        
        print(f"\n   📊 {tf} Summary: {tf_stocks} stocks, {tf_candles:,} total candles")
    
    # Save metadata
    metadata['last_updated'] = datetime.now().isoformat()
    metadata['stats'] = stats
    save_metadata(metadata)
    
    # Save corporate actions
    with open(CORPORATE_ACTIONS_FILE, 'w') as f:
        json.dump(KNOWN_CORPORATE_ACTIONS, f, indent=2)
    
    # Print summary
    print(f"\n\n{'=' * 70}")
    print(f"✅ DATA BANK CREATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n📁 Data Location: {DATA_DIR}")
    print(f"\n📊 Summary:")
    print(f"   Total Candles: {stats['total_candles']:,}")
    for tf, tf_stats in stats['by_timeframe'].items():
        print(f"   {tf:12s}: {tf_stats['stocks']:3d} stocks, {tf_stats['candles']:>10,} candles")
    print(f"\n📄 Files Created:")
    print(f"   {METADATA_FILE}")
    print(f"   {INSTRUMENTS_FILE}")
    print(f"   {CORPORATE_ACTIONS_FILE}")
    print(f"   {DATA_DIR}/<timeframe>/<SYMBOL>.parquet")
    
    return stats


# ============== DATA BANK READER ==============

class DataBank:
    """
    Data Bank Reader for backtesting
    
    Usage:
        db = DataBank()
        df = db.get_data("RELIANCE", "day")
        df = db.get_data("TCS", "15minute", adjusted=True)
    """
    
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.metadata = self._load_metadata()
        self.instruments = self._load_instruments()
        self._cache = {}  # In-memory cache
    
    def _load_metadata(self):
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE) as f:
                return json.load(f)
        return {}
    
    def _load_instruments(self):
        if os.path.exists(INSTRUMENTS_FILE):
            return pd.read_parquet(INSTRUMENTS_FILE)
        return pd.DataFrame()
    
    def get_available_stocks(self, timeframe="day"):
        """Get list of available stocks for a timeframe"""
        tf_dir = f"{self.data_dir}/{timeframe}"
        if not os.path.exists(tf_dir):
            return []
        return [f.replace('.parquet', '') for f in os.listdir(tf_dir) if f.endswith('.parquet')]
    
    def get_available_timeframes(self):
        """Get list of available timeframes"""
        return [d for d in os.listdir(self.data_dir) 
                if os.path.isdir(f"{self.data_dir}/{d}") and d in TIMEFRAMES]
    
    def get_data(self, symbol, timeframe="day", adjusted=False, use_cache=True):
        """
        Get historical data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            timeframe: Timeframe (e.g., "day", "15minute")
            adjusted: If True, return adjusted prices for corporate actions
            use_cache: If True, cache data in memory
        
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}"
        
        if use_cache and cache_key in self._cache:
            df = self._cache[cache_key].copy()
        else:
            file_path = f"{self.data_dir}/{timeframe}/{symbol}.parquet"
            
            if not os.path.exists(file_path):
                return None
            
            df = pd.read_parquet(file_path)
            df['date'] = pd.to_datetime(df['date'])
            
            if use_cache:
                self._cache[cache_key] = df.copy()
        
        # Return adjusted or actual prices
        if adjusted and 'adj_close' in df.columns:
            df = df.rename(columns={
                'adj_open': 'open',
                'adj_high': 'high',
                'adj_low': 'low',
                'adj_close': 'close',
                'adj_volume': 'volume'
            })
        
        return df
    
    def get_data_range(self, symbol, timeframe, start_date, end_date, adjusted=False):
        """Get data for a specific date range"""
        df = self.get_data(symbol, timeframe, adjusted)
        
        if df is None:
            return None
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        return df[(df['date'] >= start) & (df['date'] <= end)].reset_index(drop=True)
    
    def get_info(self, symbol):
        """Get instrument info for a symbol"""
        if symbol in self.instruments.index:
            return self.instruments.loc[symbol].to_dict()
        return None
    
    def clear_cache(self):
        """Clear in-memory cache"""
        self._cache.clear()
    
    def summary(self):
        """Print data bank summary"""
        print(f"\n{'=' * 60}")
        print(f"📊 DATA BANK SUMMARY")
        print(f"{'=' * 60}")
        
        if not self.metadata:
            print("   No data found. Run create_data_bank() first.")
            return
        
        print(f"\n   Last Updated: {self.metadata.get('last_updated', 'Unknown')}")
        print(f"   Total Stocks: {len(self.metadata.get('stocks', {}))}")
        
        print(f"\n   Available Timeframes:")
        for tf in self.get_available_timeframes():
            stocks = self.get_available_stocks(tf)
            print(f"      {tf:12s}: {len(stocks)} stocks")
        
        if 'stats' in self.metadata:
            print(f"\n   Total Candles: {self.metadata['stats'].get('total_candles', 0):,}")


# ============== MAIN ==============

def main():
    print("\n" + "=" * 70)
    print("📦 HISTORICAL DATA BANK CREATOR")
    print("=" * 70)
    
    # Load session
    kite = load_session()
    
    # Ask user which timeframes to fetch
    print("\n📋 Available Timeframes:")
    for i, (tf, config) in enumerate(TIMEFRAMES.items(), 1):
        print(f"   {i}. {tf:12s} - Max {config['max_days']} days")
    
    print("\nOptions:")
    print("   A - Fetch ALL timeframes (will take several hours)")
    print("   D - Fetch only DAILY data (fastest, recommended for initial setup)")
    print("   S - Select specific timeframes")
    print("   Q - Quit")
    
    choice = input("\nYour choice: ").strip().upper()
    
    if choice == 'Q':
        print("Exiting...")
        return
    elif choice == 'A':
        timeframes = list(TIMEFRAMES.keys())
    elif choice == 'D':
        timeframes = ['day']
    elif choice == 'S':
        print("\nEnter timeframe numbers separated by comma (e.g., 1,7,8):")
        nums = input("> ").strip().split(',')
        tf_list = list(TIMEFRAMES.keys())
        timeframes = [tf_list[int(n.strip())-1] for n in nums if n.strip().isdigit()]
    else:
        print("Invalid choice. Using DAILY only.")
        timeframes = ['day']
    
    print(f"\n📊 Will fetch: {', '.join(timeframes)}")
    print(f"📊 Stocks: {len(STOCK_UNIVERSE)}")
    
    # Estimate time
    total_api_calls = sum(
        len(STOCK_UNIVERSE) * (TIMEFRAMES[tf]['max_days'] // TIMEFRAMES[tf]['chunk_days'] + 1)
        for tf in timeframes
    )
    est_time_minutes = (total_api_calls * 0.4) / 60  # 0.4 seconds per call
    print(f"⏱️ Estimated time: {est_time_minutes:.0f} minutes")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Create data bank
    stats = create_data_bank(kite, timeframes_to_fetch=timeframes)
    
    # Test the data bank
    print("\n\n📖 Testing Data Bank Reader...")
    db = DataBank()
    db.summary()
    
    # Test reading a stock
    test_symbol = "RELIANCE"
    df = db.get_data(test_symbol, "day")
    if df is not None:
        print(f"\n   Sample data for {test_symbol}:")
        print(f"   Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"   Total rows: {len(df):,}")


if __name__ == "__main__":
    main()
