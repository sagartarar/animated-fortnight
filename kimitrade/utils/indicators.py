"""
Technical Indicators Module
All indicators needed for the three alternative strategies
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return prices.ewm(span=period, adjust=False).mean()

def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return prices.rolling(window=period).mean()

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price (daily reset)"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_volume = typical_price * df['volume']
    
    # Group by date and calculate cumulative
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    df_copy['date_only'] = df_copy['date'].dt.date
    df_copy['tp_volume'] = tp_volume
    
    cum_tp_vol = df_copy.groupby('date_only')['tp_volume'].cumsum()
    cum_vol = df_copy.groupby('date_only')['volume'].cumsum()
    
    return cum_tp_vol / cum_vol

def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index
    Returns: (ADX, +DI, -DI)
    """
    df_copy = df.copy()
    
    # True Range
    high_low = df_copy['high'] - df_copy['low']
    high_close = abs(df_copy['high'] - df_copy['close'].shift())
    low_close = abs(df_copy['low'] - df_copy['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = df_copy['high'].diff()
    minus_dm = df_copy['low'].diff().multiply(-1)
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smooth TR and DM
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).sum() / (atr * period))
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / (atr * period))
    
    # Directional Index
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    Supertrend Indicator
    Returns: (Direction, ATR)
    """
    atr = calculate_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = -1
        else:
            if df['close'].iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lowerband.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upperband.iloc[i]
                direction.iloc[i] = -1
    
    return direction, atr

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands
    Returns: (Upper, Middle, Lower)
    """
    middle = calculate_sma(prices, period)
    std = prices.rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return upper, middle, lower

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD Indicator
    Returns: (MACD, Signal, Histogram)
    """
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    histogram = macd - signal_line
    
    return macd, signal_line, histogram

def calculate_volume_profile(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """
    Calculate Volume Profile (price levels with most volume)
    Returns DataFrame with price levels and volume
    """
    price_min = df['low'].min()
    price_max = df['high'].max()
    
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    
    profile = []
    for i in range(bins):
        mask = (df['close'] >= bin_edges[i]) & (df['close'] < bin_edges[i+1])
        volume = df[mask]['volume'].sum()
        profile.append({
            'price_level': (bin_edges[i] + bin_edges[i+1]) / 2,
            'volume': volume
        })
    
    return pd.DataFrame(profile)

def calculate_first_30min_momentum(df: pd.DataFrame) -> float:
    """
    Calculate first 30-minute return (9:15-9:45 AM)
    For Intraday Momentum Strategy
    """
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    df_copy['time'] = df_copy['date'].dt.time
    df_copy['date_only'] = df_copy['date'].dt.date
    
    today = df_copy['date_only'].iloc[-1]
    today_data = df_copy[df_copy['date_only'] == today]
    
    # Find first 30 minutes (9:15 - 9:45)
    first_30 = today_data[
        (today_data['time'] >= pd.Timestamp('09:15').time()) &
        (today_data['time'] <= pd.Timestamp('09:45').time())
    ]
    
    if len(first_30) < 2:
        return 0.0
    
    first_price = first_30.iloc[0]['open']
    last_price = first_30.iloc[-1]['close']
    
    return ((last_price - first_price) / first_price) * 100

def calculate_vwap_deviation(df: pd.DataFrame) -> pd.Series:
    """
    Calculate price deviation from VWAP as percentage
    """
    vwap = calculate_vwap(df)
    return ((df['close'] - vwap) / vwap) * 100

def calculate_regime_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate indicators for regime detection
    Returns dict with various metrics
    """
    close = df['close']
    
    # Trend
    sma_20 = calculate_sma(close, 20).iloc[-1]
    sma_50 = calculate_sma(close, 50).iloc[-1]
    
    # Volatility
    returns = close.pct_change()
    volatility = returns.std() * np.sqrt(252)  # Annualized
    
    # ADX for trend strength
    adx, plus_di, minus_di = calculate_adx(df)
    
    # RSI
    rsi = calculate_rsi(close).iloc[-1]
    
    return {
        'price': close.iloc[-1],
        'sma_20': sma_20,
        'sma_50': sma_50,
        'trend': 'up' if close.iloc[-1] > sma_20 > sma_50 else 'down' if close.iloc[-1] < sma_20 < sma_50 else 'mixed',
        'volatility': volatility,
        'adx': adx.iloc[-1],
        'plus_di': plus_di.iloc[-1],
        'minus_di': minus_di.iloc[-1],
        'rsi': rsi,
        'trend_strength': adx.iloc[-1]
    }

# Import Dict type
from typing import Dict
