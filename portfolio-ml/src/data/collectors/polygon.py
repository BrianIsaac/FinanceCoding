"""Polygon.io data collector for recent historical OHLCV data."""

import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

from src.config_schemas import CollectorConfig

load_dotenv()


class PolygonCollector:
    """
    Collector for Polygon.io financial data API.

    Polygon provides institutional-grade OHLCV data with 2 years history on free tier.
    Free tier: 5 API calls/minute.

    Args:
        config: Collector configuration with API key

    Example:
        >>> from dotenv import load_dotenv
        >>> load_dotenv()  # Loads POLYGON_API_KEY from .env
        >>> config = CollectorConfig(
        ...     source_name='polygon',
        ...     rate_limit=12.0  # 5 calls/min = 12 sec between calls
        ... )
        >>> collector = PolygonCollector(config)
        >>> ticker_periods = [
        ...     {'ticker': 'AAPL', 'start': '2023-01-01', 'end': '2024-12-31'}
        ... ]
        >>> data = collector.collect_ohlcv_data(ticker_periods)
    """

    def __init__(self, config: CollectorConfig):
        """
        Initialise Polygon collector.

        Args:
            config: Collector configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.api_key = os.getenv("POLYGON_API_KEY")

    def collect_ohlcv_data(
        self,
        ticker_periods: list[dict[str, str]],
    ) -> dict[str, pd.DataFrame]:
        """
        Collect OHLCV data for tickers with individual date ranges.

        Args:
            ticker_periods: List of dicts with keys:
                - ticker: Symbol to collect
                - start: Start date (YYYY-MM-DD) or None
                - end: End date (YYYY-MM-DD) or None

        Returns:
            Dictionary with keys ['open', 'high', 'low', 'close', 'volume'],
            each containing a DataFrame with shape (Dates × Tickers)

        Note:
            Rate limit: 5 calls/minute on free tier
            Free tier limited to 2 years historical data
        """
        if not self.api_key:
            raise ValueError("Polygon API key not set. Set collector.api_key before collection.")

        client = RESTClient(api_key=self.api_key)

        all_data = {field: {} for field in ["open", "high", "low", "close", "volume"]}
        successful = []
        failed = []

        two_years_ago = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

        self.logger.info(
            f"Collecting data for {len(ticker_periods)} ticker-period combinations via Polygon"
        )
        self.logger.info(f"Free tier limitation: Data only available from {two_years_ago} onwards")

        for i, period in enumerate(ticker_periods, 1):
            ticker = period["ticker"]
            start_date = period.get("start")
            end_date = period.get("end")

            if start_date and start_date < two_years_ago:
                self.logger.debug(
                    f"{ticker}: Start date {start_date} exceeds free tier limit, adjusting to {two_years_ago}"
                )
                start_date = two_years_ago

            if i % 25 == 0 or i == len(ticker_periods):
                self.logger.info(
                    f"Progress: {i}/{len(ticker_periods)} ({i/len(ticker_periods)*100:.1f}%)"
                )

            try:
                time.sleep(self.config.rate_limit)

                aggs = client.get_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan='day',
                    from_=start_date,
                    to=end_date,
                    adjusted=True,
                    sort='asc'
                )

                if not aggs:
                    self.logger.debug(f"{ticker}: No data returned")
                    failed.append(ticker)
                    continue

                records = []
                for agg in aggs:
                    records.append({
                        'date': pd.to_datetime(agg.timestamp, unit='ms'),
                        'open': agg.open,
                        'high': agg.high,
                        'low': agg.low,
                        'close': agg.close,
                        'volume': agg.volume
                    })

                df = pd.DataFrame(records).set_index('date')

                if df.empty:
                    self.logger.debug(f"{ticker}: Empty DataFrame")
                    failed.append(ticker)
                    continue

                for field in ["open", "high", "low", "close", "volume"]:
                    all_data[field][ticker] = df[field]

                successful.append(ticker)

            except Exception as e:
                self.logger.warning(f"{ticker}: Collection failed - {e}")
                failed.append(ticker)

        self.logger.info(
            f"Polygon collection complete: {len(successful)} successful, {len(failed)} failed"
        )

        result = {}
        for field in ["open", "high", "low", "close", "volume"]:
            if all_data[field]:
                result[field] = pd.DataFrame(all_data[field])
            else:
                result[field] = pd.DataFrame()

        return result
