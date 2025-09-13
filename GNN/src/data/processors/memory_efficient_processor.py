"""
Memory-efficient data processing strategies for large-scale pipeline operations.

This module provides chunked processing, memory monitoring, and optimization
techniques for handling 8+ years of data across 400+ securities efficiently.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    process_mb: float


@dataclass
class ProcessingChunk:
    """Data processing chunk with metadata."""

    data: pd.DataFrame
    chunk_id: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    tickers: list[str]
    memory_mb: float


class MemoryEfficientProcessor:
    """
    Memory-efficient processor for large-scale data operations.

    Provides chunked processing, memory monitoring, and optimization
    techniques to handle large datasets within memory constraints.
    """

    def __init__(
        self,
        max_memory_mb: float = 16000,  # 16GB default limit
        chunk_size_months: int = 6,
        ticker_batch_size: int = 100,
        enable_monitoring: bool = True,
    ):
        """
        Initialize memory-efficient processor.

        Args:
            max_memory_mb: Maximum memory usage limit in MB
            chunk_size_months: Temporal chunk size in months
            ticker_batch_size: Number of tickers to process per batch
            enable_monitoring: Enable memory monitoring
        """
        self.max_memory_mb = max_memory_mb
        self.chunk_size_months = chunk_size_months
        self.ticker_batch_size = ticker_batch_size
        self.enable_monitoring = enable_monitoring

        # Initialize process monitor
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_stats()

        logger.info(f"MemoryEfficientProcessor initialized with {max_memory_mb}MB limit")

    def get_memory_stats(self) -> MemoryStats:
        """Get current system and process memory statistics."""
        # System memory
        system_memory = psutil.virtual_memory()

        # Process memory
        process_memory = self.process.memory_info()

        return MemoryStats(
            total_mb=system_memory.total / (1024 * 1024),
            available_mb=system_memory.available / (1024 * 1024),
            used_mb=system_memory.used / (1024 * 1024),
            percent_used=system_memory.percent,
            process_mb=process_memory.rss / (1024 * 1024),
        )

    @contextmanager
    def memory_monitor(self, operation_name: str = "operation"):
        """Context manager for monitoring memory usage during operations."""
        if not self.enable_monitoring:
            yield
            return

        start_stats = self.get_memory_stats()
        start_process_mb = start_stats.process_mb

        try:
            logger.debug(f"Starting {operation_name} - Process memory: {start_process_mb:.1f}MB")
            yield

        finally:
            end_stats = self.get_memory_stats()
            end_process_mb = end_stats.process_mb
            memory_delta = end_process_mb - start_process_mb

            logger.debug(
                f"Completed {operation_name} - "
                f"Process memory: {end_process_mb:.1f}MB "
                f"(Δ{memory_delta:+.1f}MB)"
            )

            # Check memory limits
            if end_stats.process_mb > self.max_memory_mb:
                logger.warning(
                    f"Process memory ({end_process_mb:.1f}MB) exceeds limit ({self.max_memory_mb}MB)"
                )

    def create_temporal_chunks(
        self,
        start_date: str,
        end_date: str,
        chunk_size_months: int = None,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Create temporal chunks for processing large date ranges.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            chunk_size_months: Override default chunk size

        Returns:
            List of (start, end) timestamp tuples for each chunk
        """
        chunk_size = chunk_size_months or self.chunk_size_months
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)

        chunks = []
        current_start = start_ts

        while current_start < end_ts:
            # Calculate chunk end (beginning of next period)
            current_end = min(
                current_start + pd.DateOffset(months=chunk_size),
                end_ts
            )

            chunks.append((current_start, current_end))
            current_start = current_end

        logger.info(f"Created {len(chunks)} temporal chunks of ~{chunk_size} months each")
        return chunks

    def create_ticker_batches(self, tickers: list[str], batch_size: int = None) -> list[list[str]]:
        """
        Create ticker batches for processing large universe.

        Args:
            tickers: List of ticker symbols
            batch_size: Override default batch size

        Returns:
            List of ticker batches
        """
        batch_size = batch_size or self.ticker_batch_size
        batches = []

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            batches.append(batch)

        logger.info(f"Created {len(batches)} ticker batches of ~{batch_size} tickers each")
        return batches

    def process_data_in_chunks(
        self,
        data_loader_func: callable,
        processor_func: callable,
        start_date: str,
        end_date: str,
        tickers: list[str],
        output_dir: Path = None,
        **kwargs: Any,
    ) -> Generator[ProcessingChunk, None, None]:
        """
        Process data in memory-efficient chunks.

        Args:
            data_loader_func: Function to load data for a chunk
            processor_func: Function to process loaded data
            start_date: Start date for processing
            end_date: End date for processing
            tickers: List of ticker symbols
            output_dir: Optional output directory for intermediate results
            **kwargs: Additional arguments for processing functions

        Yields:
            ProcessingChunk: Processed data chunks with metadata
        """
        temporal_chunks = self.create_temporal_chunks(start_date, end_date)
        ticker_batches = self.create_ticker_batches(tickers)

        chunk_id = 0

        for chunk_start, chunk_end in temporal_chunks:
            for ticker_batch in ticker_batches:
                with self.memory_monitor(f"chunk_{chunk_id}"):
                    try:
                        # Load data for current chunk
                        chunk_data = data_loader_func(
                            start_date=chunk_start,
                            end_date=chunk_end,
                            tickers=ticker_batch,
                            **kwargs
                        )

                        if chunk_data is None or chunk_data.empty:
                            logger.debug(f"Skipping empty chunk {chunk_id}")
                            continue

                        # Process loaded data
                        processed_data = processor_func(chunk_data, **kwargs)

                        # Calculate memory usage
                        memory_mb = self._estimate_dataframe_memory(processed_data)

                        # Create processing chunk
                        processing_chunk = ProcessingChunk(
                            data=processed_data,
                            chunk_id=chunk_id,
                            start_date=chunk_start,
                            end_date=chunk_end,
                            tickers=ticker_batch,
                            memory_mb=memory_mb,
                        )

                        # Save intermediate results if requested
                        if output_dir:
                            self._save_intermediate_chunk(processing_chunk, output_dir)

                        logger.debug(
                            f"Processed chunk {chunk_id}: "
                            f"{chunk_start.date()} to {chunk_end.date()}, "
                            f"{len(ticker_batch)} tickers, "
                            f"{memory_mb:.1f}MB"
                        )

                        yield processing_chunk

                        # Clean up chunk data to free memory
                        del chunk_data, processed_data

                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
                        raise

                    finally:
                        # Force garbage collection
                        gc.collect()
                        chunk_id += 1

    def process_parquet_efficiently(
        self,
        input_parquet_path: Path,
        output_parquet_path: Path,
        processor_func: callable,
        **kwargs: Any,
    ) -> None:
        """
        Process large parquet files efficiently with chunked reading.

        Args:
            input_parquet_path: Path to input parquet file
            output_parquet_path: Path to output parquet file
            processor_func: Function to process each chunk
            **kwargs: Additional arguments for processor function
        """
        output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

        # Read parquet file metadata to determine chunking strategy
        parquet_file = pd.read_parquet(input_parquet_path, nrows=1)
        total_size_mb = input_parquet_path.stat().st_size / (1024 * 1024)

        logger.info(f"Processing parquet file: {total_size_mb:.1f}MB")

        # Determine appropriate chunk size based on available memory
        available_memory = self.get_memory_stats().available_mb
        chunk_size_mb = min(available_memory * 0.3, self.max_memory_mb * 0.5)  # Use 30% of available or 50% of limit

        # Estimate rows per chunk
        if hasattr(parquet_file, "memory_usage"):
            row_memory_kb = parquet_file.memory_usage(deep=True).sum() / 1024
            rows_per_chunk = int((chunk_size_mb * 1024) / row_memory_kb)
        else:
            rows_per_chunk = 10000  # Conservative default

        logger.info(f"Using chunk size: ~{rows_per_chunk} rows (~{chunk_size_mb:.1f}MB)")

        # Process in chunks
        processed_chunks = []

        try:
            parquet_reader = pd.read_parquet(input_parquet_path, chunksize=rows_per_chunk)

            for i, chunk in enumerate(parquet_reader):
                with self.memory_monitor(f"parquet_chunk_{i}"):
                    processed_chunk = processor_func(chunk, **kwargs)
                    processed_chunks.append(processed_chunk)

                    # Periodically write and clear chunks to manage memory
                    if len(processed_chunks) >= 10:  # Write every 10 chunks
                        self._write_chunk_batch(processed_chunks, output_parquet_path, append=i > 0)
                        processed_chunks.clear()
                        gc.collect()

        except Exception as e:
            # Handle cases where chunksize is not supported
            logger.warning(f"Chunked reading not supported, processing full file: {e}")

            with self.memory_monitor("full_parquet_processing"):
                full_data = pd.read_parquet(input_parquet_path)
                processed_data = processor_func(full_data, **kwargs)
                processed_data.to_parquet(output_parquet_path)
                return

        # Write remaining chunks
        if processed_chunks:
            self._write_chunk_batch(processed_chunks, output_parquet_path, append=True)

        logger.info(f"Parquet processing complete: {output_parquet_path}")

    def optimize_dataframe_memory(self, df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types.

        Args:
            df: DataFrame to optimize
            aggressive: Use more aggressive optimization (may lose precision)

        Returns:
            Memory-optimized DataFrame
        """
        with self.memory_monitor("memory_optimization"):
            optimized_df = df.copy()
            initial_memory = self._estimate_dataframe_memory(df)

            # Optimize numeric columns
            for col in optimized_df.select_dtypes(include=[np.number]):
                col_data = optimized_df[col]

                # Skip if column has NaN values and aggressive mode is off
                if not aggressive and col_data.isna().any():
                    continue

                # Try to downcast integers
                if col_data.dtype in ['int64', 'int32']:
                    try:
                        optimized_df[col] = pd.to_numeric(col_data, downcast='integer')
                    except Exception:
                        pass

                # Try to downcast floats
                elif col_data.dtype in ['float64', 'float32']:
                    try:
                        if aggressive:
                            # More aggressive downcasting
                            optimized_df[col] = pd.to_numeric(col_data, downcast='float')
                        else:
                            # Conservative downcasting
                            if col_data.max() < 3.4e38 and col_data.min() > -3.4e38:
                                optimized_df[col] = col_data.astype('float32')
                    except Exception:
                        pass

            # Optimize object columns (strings)
            for col in optimized_df.select_dtypes(include=['object']):
                try:
                    # Try to convert to category if cardinality is low
                    unique_ratio = optimized_df[col].nunique() / len(optimized_df[col])
                    if unique_ratio < 0.5:  # Less than 50% unique values
                        optimized_df[col] = optimized_df[col].astype('category')
                except Exception:
                    pass

            final_memory = self._estimate_dataframe_memory(optimized_df)
            memory_reduction = ((initial_memory - final_memory) / initial_memory) * 100

            logger.info(
                f"Memory optimization: {initial_memory:.1f}MB -> {final_memory:.1f}MB "
                f"({memory_reduction:.1f}% reduction)"
            )

            return optimized_df

    def _estimate_dataframe_memory(self, df: pd.DataFrame) -> float:
        """Estimate DataFrame memory usage in MB."""
        if hasattr(df, "memory_usage"):
            return df.memory_usage(deep=True).sum() / (1024 * 1024)
        else:
            # Fallback estimation
            return len(df) * len(df.columns) * 8 / (1024 * 1024)  # Assume 8 bytes per value

    def _save_intermediate_chunk(self, chunk: ProcessingChunk, output_dir: Path) -> None:
        """Save intermediate processing chunk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = output_dir / f"chunk_{chunk.chunk_id:04d}.parquet"
        chunk.data.to_parquet(chunk_path)

        # Save chunk metadata
        metadata_path = output_dir / f"chunk_{chunk.chunk_id:04d}_metadata.json"
        metadata = {
            "chunk_id": chunk.chunk_id,
            "start_date": chunk.start_date.isoformat(),
            "end_date": chunk.end_date.isoformat(),
            "tickers": chunk.tickers,
            "memory_mb": chunk.memory_mb,
            "data_shape": chunk.data.shape,
        }

        import json
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _write_chunk_batch(
        self, chunks: list[pd.DataFrame], output_path: Path, append: bool = False
    ) -> None:
        """Write batch of chunks to parquet file."""
        combined_data = pd.concat(chunks, ignore_index=True)

        if append and output_path.exists():
            # Append to existing file
            existing_data = pd.read_parquet(output_path)
            combined_data = pd.concat([existing_data, combined_data], ignore_index=True)

        combined_data.to_parquet(output_path)

    def get_memory_recommendations(self, data_size_mb: float) -> dict[str, Any]:
        """Get memory optimization recommendations based on data size."""
        available_memory = self.get_memory_stats().available_mb

        recommendations = {
            "data_size_mb": data_size_mb,
            "available_memory_mb": available_memory,
            "memory_ratio": data_size_mb / available_memory,
            "recommendations": []
        }

        if data_size_mb > available_memory * 0.8:
            recommendations["recommendations"].extend([
                "Data size exceeds 80% of available memory",
                "Use chunked processing with smaller chunk sizes",
                "Consider processing on machine with more RAM"
            ])

        if data_size_mb > self.max_memory_mb:
            recommendations["recommendations"].extend([
                "Data size exceeds configured memory limit",
                "Reduce chunk size or ticker batch size",
                "Enable aggressive memory optimization"
            ])

        if data_size_mb < available_memory * 0.3:
            recommendations["recommendations"].append(
                "Data size is small relative to available memory - chunking may not be necessary"
            )

        return recommendations


class ChunkedDataFrameWriter:
    """Utility for writing large DataFrames efficiently."""

    def __init__(self, output_path: Path, chunk_size: int = 50000):
        """Initialize chunked writer."""
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_dataframe(self, df: pd.DataFrame, compression: str = "gzip") -> None:
        """Write DataFrame in chunks to manage memory."""
        if len(df) <= self.chunk_size:
            # Small DataFrame, write directly
            df.to_parquet(self.output_path, compression=compression)
            return

        # Write in chunks
        temp_files = []

        try:
            for i in range(0, len(df), self.chunk_size):
                chunk = df.iloc[i:i + self.chunk_size]
                temp_path = self.output_path.parent / f"temp_chunk_{i}.parquet"
                chunk.to_parquet(temp_path, compression=compression)
                temp_files.append(temp_path)

            # Combine temp files
            combined_data = pd.concat([pd.read_parquet(temp_file) for temp_file in temp_files])
            combined_data.to_parquet(self.output_path, compression=compression)

        finally:
            # Clean up temp files
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()


@contextmanager
def memory_limit_context(max_memory_mb: float):
    """Context manager to enforce memory limits during processing."""
    psutil.virtual_memory()
    psutil.Process().memory_info().rss / (1024 * 1024)

    try:
        yield
    finally:
        final_process = psutil.Process().memory_info().rss / (1024 * 1024)
        if final_process > max_memory_mb:
            logger.warning(
                f"Memory usage ({final_process:.1f}MB) exceeded limit ({max_memory_mb}MB)"
            )
