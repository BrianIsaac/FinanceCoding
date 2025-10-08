#!/usr/bin/env python3
"""
Fix Universe Membership Data Schema

Converts the existing universe membership CSV from start/end format to daily format
required by UniverseManager for proper dynamic membership filtering.

Current format: ticker,start,end,index_name
Required format: date,ticker,in_universe
"""

import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_universe_membership():
    """Convert universe membership from start/end format to daily format."""

    # File paths
    input_file = "data/processed/universe_membership_clean.csv"
    output_file = "data/processed/universe_membership_daily.csv"

    logger.info(f"Reading universe membership data from {input_file}")

    # Read the existing data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} membership records")

    # Convert date columns
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])

    # Determine date range for daily data
    min_date = df['start'].min()
    max_date = df['end'].max()

    # Extend range to cover training period (2016-2024)
    start_date = pd.to_datetime('2016-01-01')
    end_date = pd.to_datetime('2024-12-31')

    logger.info(f"Generating daily membership data from {start_date} to {end_date}")

    # Create daily date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Get all unique tickers
    all_tickers = df['ticker'].unique()
    logger.info(f"Processing {len(all_tickers)} unique tickers")

    # Create daily membership records
    daily_records = []

    for ticker in all_tickers:
        ticker_records = df[df['ticker'] == ticker]

        for date in date_range:
            # Check if ticker is in universe on this date
            in_universe = 0
            for _, record in ticker_records.iterrows():
                if record['start'] <= date <= record['end']:
                    in_universe = 1
                    break

            daily_records.append({
                'date': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'in_universe': in_universe
            })

    # Create daily DataFrame
    daily_df = pd.DataFrame(daily_records)

    logger.info(f"Generated {len(daily_df)} daily records")

    # Save to file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(output_file, index=False)

    logger.info(f"Saved daily universe membership to {output_file}")

    # Print summary statistics
    total_ticker_days = len(daily_df)
    active_ticker_days = len(daily_df[daily_df['in_universe'] == 1])

    logger.info(f"Summary:")
    logger.info(f"  Total ticker-days: {total_ticker_days:,}")
    logger.info(f"  Active ticker-days: {active_ticker_days:,}")
    logger.info(f"  Coverage: {active_ticker_days/total_ticker_days:.1%}")

    # Sample verification
    sample_date = '2020-01-01'
    sample_data = daily_df[daily_df['date'] == sample_date]
    active_on_date = len(sample_data[sample_data['in_universe'] == 1])

    logger.info(f"  Active tickers on {sample_date}: {active_on_date}")

    return output_file


if __name__ == "__main__":
    output_file = convert_universe_membership()
    print(f"\nUniverse membership data conversion completed!")
    print(f"Output file: {output_file}")
    print("\nYou can now update the UniverseManager to use this file.")