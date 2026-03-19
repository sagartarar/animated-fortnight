#!/usr/bin/env python3
"""
KIMITRADE - Strategy Runner
Run all three alternative strategies for paper trading

Usage:
    python run_strategies.py --strategy momentum --mode paper
    python run_strategies.py --strategy vwap --mode backtest --data data.csv
    python run_strategies.py --strategy regime --mode live
"""

import argparse
import json
from datetime import datetime
import pandas as pd

# Import strategies
from strategies.intraday_momentum import IntradayMomentumStrategy, IntradayMomentumBacktester
from strategies.vwap_ladder import VWAPLadderStrategy, VWAPLadderBacktester
from strategies.regime_switching import RegimeSwitchingStrategy, RegimeSwitchingBacktester


def run_momentum_strategy(mode: str, capital: float = 500000, data_file: str = None):
    """Run intraday momentum strategy"""
    print("="*70)
    print("STRATEGY 1: Intraday Momentum")
    print("Based on: Gao et al., Journal of Financial Economics")
    print("="*70)
    print()
    
    if mode == "paper":
        strategy = IntradayMomentumStrategy(capital)
        print(f"Paper trading initialized with ₹{capital:,.0f}")
        print(f"Entry: 11:00 AM | Exit: 3:15 PM or SL")
        print(f"Risk per trade: {strategy.config.RISK_PER_TRADE_PCT}%")
        print()
        print("Status:", json.dumps(strategy.get_status(), indent=2))
        
    elif mode == "backtest":
        if not data_file:
            print("Error: --data required for backtest mode")
            return
        
        print(f"Backtesting with data: {data_file}")
        # Load data and run backtest
        # Implementation depends on data format
        
    elif mode == "live":
        print("Live trading mode - requires Kite connection")
        # Connect to Kite and run


def run_vwap_strategy(mode: str, capital: float = 500000, data_file: str = None):
    """Run VWAP + Ladder strategy"""
    print("="*70)
    print("STRATEGY 2: VWAP + Ladder Exit")
    print("Based on: SSRN 5095349 - Sharpe >3.0, Returns >50%")
    print("="*70)
    print()
    
    if mode == "paper":
        strategy = VWAPLadderStrategy(capital)
        print(f"Paper trading initialized with ₹{capital:,.0f}")
        print(f"Entry: VWAP deviation | Ladder exits at 0.5R, 1.0R, 1.5R")
        print(f"Trailing stop on remainder")
        print()
        print("Status:", json.dumps(strategy.get_status(), indent=2))


def run_regime_strategy(mode: str, capital: float = 500000, data_file: str = None):
    """Run Regime-Switching strategy"""
    print("="*70)
    print("STRATEGY 3: Regime-Switching")
    print("Based on: Hidden Markov Models for regime detection")
    print("="*70)
    print()
    
    if mode == "paper":
        strategy = RegimeSwitchingStrategy(capital)
        print(f"Paper trading initialized with ₹{capital:,.0f}")
        print(f"Regimes: BULL (long pullbacks) | BEAR (short rallies) | RANGE (VWAP)")
        print(f"No trading in VOLATILE regime")
        print()
        print("Status:", json.dumps(strategy.get_status(), indent=2))


def compare_strategies():
    """Compare all three strategies"""
    print("="*70)
    print("STRATEGY COMPARISON")
    print("="*70)
    print()
    
    comparison = """
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Strategy 1          │ Strategy 2          │ Strategy 3          │
│ Intraday Momentum   │ VWAP + Ladder       │ Regime-Switching    │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Based on:           │ Based on:           │ Based on:           │
│ Gao et al. (JFE)    │ SSRN 5095349        │ HMM Research        │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Logic:              │ Logic:              │ Logic:              │
│ First 30-min        │ VWAP deviation +    │ Detect regime       │
│ predicts direction  │ Ladder exits        │ Adapt strategy      │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Direction:          │ Direction:          │ Direction:          │
│ Both Long & Short   │ Both (VWAP based)   │ Regime-dependent    │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Exit:               │ Exit:               │ Exit:               │
│ 3:15 PM or SL       │ Ladder + Trail      │ SL/Target/Regime    │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Performance:        │ Performance:        │ Performance:        │
│ 6.3% annual return  │ Sharpe >3.0         │ +5-8% alpha         │
│ Works in both       │ 50%+ returns        │ -30-40% drawdown    │
│ bull & bear         │ <15% drawdown       │ reduction           │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Best for:           │ Best for:           │ Best for:           │
│ Trending markets    │ High volatility     │ Changing markets    │
│ All market types    │ Risk-adjusted       │ Regime shifts       │
└─────────────────────┴─────────────────────┴─────────────────────┘
    """
    print(comparison)
    
    print()
    print("RECOMMENDATION:")
    print("1. Start with Strategy 1 (Momentum) - simplest, works in all markets")
    print("2. Add Strategy 2 (VWAP) for higher risk-adjusted returns")
    print("3. Use Strategy 3 (Regime) to avoid bad periods")
    print()
    print("Paper trade all three for 1 month, then deploy the best performer.")


def main():
    parser = argparse.ArgumentParser(description='KIMITRADE - Strategy Runner')
    parser.add_argument('--strategy', choices=['momentum', 'vwap', 'regime', 'all'],
                       default='all', help='Strategy to run')
    parser.add_argument('--mode', choices=['paper', 'backtest', 'live'],
                       default='paper', help='Trading mode')
    parser.add_argument('--capital', type=float, default=500000,
                       help='Starting capital (default: 500000)')
    parser.add_argument('--data', type=str, help='Data file for backtest')
    parser.add_argument('--compare', action='store_true', help='Compare strategies')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_strategies()
        return
    
    if args.strategy == 'all':
        print()
        print("Running all three strategies for comparison...")
        print()
        run_momentum_strategy(args.mode, args.capital, args.data)
        print()
        run_vwap_strategy(args.mode, args.capital, args.data)
        print()
        run_regime_strategy(args.mode, args.capital, args.data)
    elif args.strategy == 'momentum':
        run_momentum_strategy(args.mode, args.capital, args.data)
    elif args.strategy == 'vwap':
        run_vwap_strategy(args.mode, args.capital, args.data)
    elif args.strategy == 'regime':
        run_regime_strategy(args.mode, args.capital, args.data)


if __name__ == "__main__":
    main()
