#!/bin/bash

# Run all 4 scenarios: 5L/10L with and without MDD controls

cd /u/tarar/repos/backtest_rust

echo "Building..."
cargo build --release 2>&1 | tail -1

# Function to update config and run
run_scenario() {
    local capital=$1
    local controls=$2
    local label=$3
    
    # Update STARTING_CAPITAL
    if [ "$capital" == "5L" ]; then
        sed -i 's/STARTING_CAPITAL: f64 = [0-9]*\.0/STARTING_CAPITAL: f64 = 500000.0/' src/main.rs
    else
        sed -i 's/STARTING_CAPITAL: f64 = [0-9]*\.0/STARTING_CAPITAL: f64 = 1000000.0/' src/main.rs
    fi
    
    # Update MDD controls
    if [ "$controls" == "ON" ]; then
        sed -i 's/USE_DRAWDOWN_LADDER: bool = false/USE_DRAWDOWN_LADDER: bool = true/' src/main.rs
        sed -i 's/USE_DAILY_LOSS_LIMIT: bool = false/USE_DAILY_LOSS_LIMIT: bool = true/' src/main.rs
        sed -i 's/USE_WEEKLY_LOSS_LIMIT: bool = false/USE_WEEKLY_LOSS_LIMIT: bool = true/' src/main.rs
        sed -i 's/USE_MONTHLY_LOSS_LIMIT: bool = false/USE_MONTHLY_LOSS_LIMIT: bool = true/' src/main.rs
        sed -i 's/USE_CONSECUTIVE_LOSS_CONTROL: bool = false/USE_CONSECUTIVE_LOSS_CONTROL: bool = true/' src/main.rs
        sed -i 's/USE_EQUITY_CURVE_FILTER: bool = false/USE_EQUITY_CURVE_FILTER: bool = true/' src/main.rs
        sed -i 's/USE_VOLATILITY_FILTER: bool = false/USE_VOLATILITY_FILTER: bool = true/' src/main.rs
    else
        sed -i 's/USE_DRAWDOWN_LADDER: bool = true/USE_DRAWDOWN_LADDER: bool = false/' src/main.rs
        sed -i 's/USE_DAILY_LOSS_LIMIT: bool = true/USE_DAILY_LOSS_LIMIT: bool = false/' src/main.rs
        sed -i 's/USE_WEEKLY_LOSS_LIMIT: bool = true/USE_WEEKLY_LOSS_LIMIT: bool = false/' src/main.rs
        sed -i 's/USE_MONTHLY_LOSS_LIMIT: bool = true/USE_MONTHLY_LOSS_LIMIT: bool = false/' src/main.rs
        sed -i 's/USE_CONSECUTIVE_LOSS_CONTROL: bool = true/USE_CONSECUTIVE_LOSS_CONTROL: bool = false/' src/main.rs
        sed -i 's/USE_EQUITY_CURVE_FILTER: bool = true/USE_EQUITY_CURVE_FILTER: bool = false/' src/main.rs
        sed -i 's/USE_VOLATILITY_FILTER: bool = true/USE_VOLATILITY_FILTER: bool = false/' src/main.rs
    fi
    
    cargo build --release 2>&1 | tail -1
    
    echo ""
    echo "============================================================"
    echo "$label"
    echo "============================================================"
    ./target/release/backtest 2>&1
}

# Run all 4 scenarios
run_scenario "5L" "OFF" "SCENARIO 1: ₹5 LAKH - NO MDD CONTROLS"
run_scenario "5L" "ON" "SCENARIO 2: ₹5 LAKH - WITH MDD CONTROLS"
run_scenario "10L" "OFF" "SCENARIO 3: ₹10 LAKH - NO MDD CONTROLS"
run_scenario "10L" "ON" "SCENARIO 4: ₹10 LAKH - WITH MDD CONTROLS"

echo ""
echo "============================================================"
echo "ALL SCENARIOS COMPLETED"
echo "============================================================"
