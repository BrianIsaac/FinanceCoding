#!/usr/bin/env python3
"""
Prepare data from archived files for backtest execution.
"""
import pandas as pd
from pathlib import Path

# Create required directories
Path("data/final_new_pipeline").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Load archived price data
print("Loading archived price data...")
prices = pd.read_parquet("archived/legacy_scripts/data/merged/prices.parquet")
print(f"  Loaded prices: {prices.shape} from {prices.index[0]} to {prices.index[-1]}")

# Calculate daily returns (pct_change)
print("Calculating daily returns...")
returns = prices.pct_change()
print(f"  Returns shape: {returns.shape}")

# Save returns to required location
returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")
returns.to_parquet(returns_path)
print(f"  Saved returns to: {returns_path}")

# Load membership data and convert to parquet
print("\nLoading membership data...")
membership = pd.read_csv("archived/legacy_scripts/data/processed/universe_membership_clean.csv")
print(f"  Loaded membership: {membership.shape}")
print(f"  Columns: {list(membership.columns)}")

# Convert dates to datetime
membership['start'] = pd.to_datetime(membership['start'])
membership['end'] = pd.to_datetime(membership['end'])

# Save to required location
membership_path = Path("data/processed/universe_calendar_midcap400.parquet")
membership.to_parquet(membership_path, index=False)
print(f"  Saved membership to: {membership_path}")

print("\n✓ Data preparation complete!")
print(f"  - Returns: {returns_path}")
print(f"  - Membership: {membership_path}")
