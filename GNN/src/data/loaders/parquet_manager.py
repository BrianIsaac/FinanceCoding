"""Parquet storage manager for efficient financial data I/O.

This module provides optimized parquet storage with monthly partitioning,
compression strategies, and schema evolution support for ML pipeline consumption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import compute as pc

from src.config.data import ValidationConfig


class ParquetManager:
    """
    High-performance parquet storage manager for financial time series data.

    Provides optimized I/O operations, monthly partitioning strategies,
    and schema evolution support for analytical workloads.
    """

    def __init__(
        self,
        base_path: str,
        compression: str = "snappy",
        row_group_size: int = 50000,
        validation_config: ValidationConfig | None = None,
    ):
        """
        Initialize parquet manager.

        Args:
            base_path: Base directory for parquet storage
            compression: Compression algorithm ('snappy', 'gzip', 'lz4', 'brotli')
            row_group_size: Target row group size for optimization
            validation_config: Optional validation configuration
        """
        self.base_path = Path(base_path)
        self.compression = compression
        self.row_group_size = row_group_size
        self.config = validation_config

        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Standard schemas for financial data
        self.schemas = self._initialize_schemas()

    def _initialize_schemas(self) -> dict[str, pa.Schema]:
        """Initialize standard schemas for financial data types.

        Returns:
            Dictionary of PyArrow schemas
        """
        # Price/returns schema - Date index with ticker columns as float64
        price_schema = pa.schema(
            [
                pa.field("date", pa.timestamp("ns")),
            ]
        )

        # Volume schema - similar to price but may have larger integers
        volume_schema = pa.schema(
            [
                pa.field("date", pa.timestamp("ns")),
            ]
        )

        # Universe schema for membership data
        universe_schema = pa.schema(
            [
                pa.field("date", pa.timestamp("ns")),
                pa.field("ticker", pa.string()),
                pa.field("action", pa.string()),  # 'added', 'removed', 'continued'
                pa.field("index_name", pa.string()),
            ]
        )

        # Quality metrics schema
        quality_schema = pa.schema(
            [
                pa.field("date", pa.timestamp("ns")),
                pa.field("ticker", pa.string()),
                pa.field("completeness_score", pa.float64()),
                pa.field("consistency_score", pa.float64()),
                pa.field("outlier_score", pa.float64()),
                pa.field("overall_score", pa.float64()),
            ]
        )

        return {
            "prices": price_schema,
            "volume": volume_schema,
            "returns": price_schema,  # Same as prices
            "universe": universe_schema,
            "quality": quality_schema,
        }

    def save_dataframe(
        self,
        df: pd.DataFrame,
        data_type: str,
        partition_strategy: str = "monthly",
        overwrite: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save DataFrame with optimized parquet storage.

        Args:
            df: DataFrame to save
            data_type: Data type ('prices', 'volume', 'returns', etc.)
            partition_strategy: Partitioning strategy ('monthly', 'yearly', 'none')
            overwrite: Whether to overwrite existing files
            metadata: Optional metadata to store with file

        Returns:
            Path to saved file(s)
        """
        if df.empty:
            raise ValueError("Cannot save empty DataFrame")

        # Ensure date index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        # Create output directory
        output_dir = self.base_path / data_type
        output_dir.mkdir(parents=True, exist_ok=True)

        if partition_strategy == "none":
            # Single file storage
            file_path = output_dir / f"{data_type}.parquet"
            return self._save_single_file(df, file_path, data_type, overwrite, metadata)

        elif partition_strategy == "monthly":
            # Monthly partitioned storage
            return self._save_monthly_partitioned(df, output_dir, data_type, overwrite, metadata)

        elif partition_strategy == "yearly":
            # Yearly partitioned storage
            return self._save_yearly_partitioned(df, output_dir, data_type, overwrite, metadata)

        else:
            raise ValueError(f"Unknown partition strategy: {partition_strategy}")

    def _save_single_file(
        self,
        df: pd.DataFrame,
        file_path: Path,
        data_type: str,
        overwrite: bool,
        metadata: dict[str, Any] | None,
    ) -> str:
        """Save DataFrame as single parquet file.

        Args:
            df: DataFrame to save
            file_path: Output file path
            data_type: Data type for schema selection
            overwrite: Whether to overwrite existing file
            metadata: Optional metadata

        Returns:
            Saved file path
        """
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"File exists and overwrite=False: {file_path}")

        # Reset index to make date a column for consistency
        df_to_save = df.reset_index()

        # Create PyArrow table
        table = pa.Table.from_pandas(df_to_save)

        # Add metadata if provided
        if metadata:
            existing_metadata = table.schema.metadata or {}
            # Convert all metadata values to strings to avoid PyArrow type errors
            string_metadata = {str(key): str(value) for key, value in metadata.items()}
            updated_metadata = {**existing_metadata, **string_metadata}
            table = table.replace_schema_metadata(updated_metadata)

        # Write with optimization settings
        pq.write_table(
            table,
            file_path,
            compression=self.compression,
            row_group_size=self.row_group_size,
            use_dictionary=True,  # Optimize for repeated strings
            write_statistics=True,  # Enable column statistics
        )

        return str(file_path)

    def _save_monthly_partitioned(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        data_type: str,
        overwrite: bool,
        metadata: dict[str, Any] | None,
    ) -> str:
        """Save DataFrame with monthly partitioning.

        Args:
            df: DataFrame to save
            output_dir: Output directory
            data_type: Data type for schema selection
            overwrite: Whether to overwrite existing files
            metadata: Optional metadata

        Returns:
            Output directory path
        """
        # Group by year-month
        df_copy = df.copy()
        df_copy["year_month"] = df_copy.index.to_period("M")

        saved_files = []

        for period, group_df in df_copy.groupby("year_month"):
            # Remove the grouping column
            group_df = group_df.drop("year_month", axis=1)

            # Create monthly file path
            year_month_str = str(period)  # Format: 2024-01
            file_path = output_dir / f"{data_type}_{year_month_str}.parquet"

            if file_path.exists() and not overwrite:
                continue

            # Save monthly partition
            monthly_metadata = metadata.copy() if metadata else {}
            monthly_metadata.update(
                {"partition_period": year_month_str, "partition_strategy": "monthly"}
            )

            saved_path = self._save_single_file(
                group_df, file_path, data_type, overwrite, monthly_metadata
            )
            saved_files.append(saved_path)

        return str(output_dir)

    def _save_yearly_partitioned(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        data_type: str,
        overwrite: bool,
        metadata: dict[str, Any] | None,
    ) -> str:
        """Save DataFrame with yearly partitioning.

        Args:
            df: DataFrame to save
            output_dir: Output directory
            data_type: Data type for schema selection
            overwrite: Whether to overwrite existing files
            metadata: Optional metadata

        Returns:
            Output directory path
        """
        # Group by year
        df_copy = df.copy()
        df_copy["year"] = df_copy.index.year

        saved_files = []

        for year, group_df in df_copy.groupby("year"):
            # Remove the grouping column
            group_df = group_df.drop("year", axis=1)

            # Create yearly file path
            file_path = output_dir / f"{data_type}_{year}.parquet"

            if file_path.exists() and not overwrite:
                continue

            # Save yearly partition
            yearly_metadata = metadata.copy() if metadata else {}
            yearly_metadata.update({"partition_year": str(year), "partition_strategy": "yearly"})

            saved_path = self._save_single_file(
                group_df, file_path, data_type, overwrite, yearly_metadata
            )
            saved_files.append(saved_path)

        return str(output_dir)

    def load_dataframe(
        self,
        data_type: str,
        start_date: str | None = None,
        end_date: str | None = None,
        columns: list[str] | None = None,
        partition_filter: str | None = None,
    ) -> pd.DataFrame:
        """Load DataFrame with optimized filtering.

        Args:
            data_type: Data type to load
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            columns: Optional list of columns to load
            partition_filter: Optional partition filter pattern

        Returns:
            Loaded DataFrame with DatetimeIndex
        """
        data_dir = self.base_path / data_type

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Find parquet files
        parquet_files = list(data_dir.glob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        # Apply partition filtering
        if partition_filter:
            parquet_files = [f for f in parquet_files if partition_filter in f.name]

        # Load files with date filtering
        dataframes = []

        for file_path in parquet_files:
            try:
                # Use PyArrow for efficient filtering
                table = pq.read_table(str(file_path))

                # Apply date filters if specified
                if start_date or end_date:
                    # Find the date column - it could be "date" or "index"
                    date_col_name = None
                    if "date" in table.schema.names:
                        date_col_name = "date"
                    elif "index" in table.schema.names:
                        date_col_name = "index"

                    if date_col_name:
                        date_column = table.column(date_col_name)
                        filters = []

                        if start_date:
                            start_ts = pd.to_datetime(start_date)
                            filters.append(pc.greater_equal(date_column, pa.scalar(start_ts)))

                        if end_date:
                            end_ts = pd.to_datetime(end_date)
                            filters.append(pc.less_equal(date_column, pa.scalar(end_ts)))

                        if filters:
                            combined_filter = filters[0]
                            for f in filters[1:]:
                                combined_filter = pc.and_(combined_filter, f)
                            table = table.filter(combined_filter)

                # Apply column selection
                if columns:
                    # Include the date/index column plus requested columns
                    date_cols = [col for col in ["date", "index"] if col in table.schema.names]
                    available_columns = [
                        col for col in date_cols + columns if col in table.schema.names
                    ]
                    table = table.select(available_columns)

                # Convert to pandas
                df = table.to_pandas()

                if not df.empty:
                    dataframes.append(df)

            except Exception:
                continue

        if not dataframes:
            return pd.DataFrame()

        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Set date as index and sort
        date_col = None
        if "date" in combined_df.columns:
            date_col = "date"
        elif "index" in combined_df.columns:
            date_col = "index"

        if date_col:
            combined_df = combined_df.set_index(date_col)
            combined_df.index = pd.to_datetime(combined_df.index)
            # Reset index name to None for consistency with original data
            combined_df.index.name = None
            combined_df = combined_df.sort_index()

        return combined_df

    def get_data_info(self, data_type: str) -> dict[str, Any]:
        """Get information about stored data.

        Args:
            data_type: Data type to analyze

        Returns:
            Dictionary with data information
        """
        data_dir = self.base_path / data_type
        info: dict[str, Any] = {
            "data_type": data_type,
            "exists": data_dir.exists(),
            "files": [],
            "total_size_mb": 0.0,
            "date_range": (None, None),
            "total_rows": 0,
            "partition_strategy": "unknown",
        }

        if not data_dir.exists():
            return info

        parquet_files = list(data_dir.glob("*.parquet"))

        min_date = None
        max_date = None

        for file_path in parquet_files:
            try:
                # Get file info
                file_size_mb = file_path.stat().st_size / (1024 * 1024)

                # Read metadata only
                parquet_file = pq.ParquetFile(str(file_path))
                num_rows = parquet_file.metadata.num_rows

                # Try to get date range from filename or metadata
                if "_" in file_path.stem:
                    # Partitioned file
                    if len(file_path.stem.split("_")) >= 2:
                        date_part = file_path.stem.split("_")[-1]
                        try:
                            if "-" in date_part and len(date_part.split("-")) == 2:
                                # Monthly partition (YYYY-MM)
                                partition_date = pd.to_datetime(date_part + "-01")
                                info["partition_strategy"] = "monthly"
                            elif date_part.isdigit() and len(date_part) == 4:
                                # Yearly partition (YYYY)
                                partition_date = pd.to_datetime(date_part + "-01-01")
                                info["partition_strategy"] = "yearly"
                            else:
                                partition_date = None

                            if partition_date:
                                min_date = (
                                    partition_date
                                    if min_date is None
                                    else min(min_date, partition_date)
                                )
                                max_date = (
                                    partition_date
                                    if max_date is None
                                    else max(max_date, partition_date)
                                )
                        except Exception:
                            pass
                else:
                    # Single file
                    info["partition_strategy"] = "none"

                info["files"].append(
                    {"name": file_path.name, "size_mb": round(file_size_mb, 2), "rows": num_rows}
                )

                info["total_size_mb"] = float(info["total_size_mb"]) + file_size_mb
                info["total_rows"] = int(info["total_rows"]) + num_rows

            except Exception:
                pass

        info["total_size_mb"] = round(float(info["total_size_mb"]), 2)
        info["date_range"] = (min_date, max_date)
        info["num_files"] = len(info["files"])

        return info

    def optimize_storage(
        self, data_type: str, target_row_group_size: int | None = None
    ) -> dict[str, Any]:
        """Optimize existing parquet storage.

        Args:
            data_type: Data type to optimize
            target_row_group_size: Target row group size for rewrite

        Returns:
            Optimization results
        """
        data_dir = self.base_path / data_type

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        parquet_files = list(data_dir.glob("*.parquet"))
        optimization_results = {
            "files_processed": 0,
            "size_before_mb": 0,
            "size_after_mb": 0,
            "compression_ratio": 0.0,
        }

        row_group_size = target_row_group_size or self.row_group_size

        for file_path in parquet_files:
            try:
                # Get original size
                original_size = file_path.stat().st_size

                # Load and rewrite with optimized settings
                df = pd.read_parquet(file_path)

                # Backup original
                backup_path = file_path.with_suffix(".parquet.backup")
                file_path.rename(backup_path)

                # Rewrite optimized
                table = pa.Table.from_pandas(df)
                pq.write_table(
                    table,
                    file_path,
                    compression=self.compression,
                    row_group_size=row_group_size,
                    use_dictionary=True,
                    write_statistics=True,
                    compression_level=6 if self.compression == "gzip" else None,
                )

                # Get new size
                new_size = file_path.stat().st_size

                # Remove backup if successful
                backup_path.unlink()

                optimization_results["files_processed"] += 1
                optimization_results["size_before_mb"] += original_size / (1024 * 1024)
                optimization_results["size_after_mb"] += new_size / (1024 * 1024)

            except Exception:
                # Restore backup if it exists
                backup_path = file_path.with_suffix(".parquet.backup")
                if backup_path.exists():
                    backup_path.rename(file_path)

        if optimization_results["size_before_mb"] > 0:
            optimization_results["compression_ratio"] = (
                optimization_results["size_after_mb"] / optimization_results["size_before_mb"]
            )

        return optimization_results

    def create_data_loading_interface(self) -> DataLoader:
        """Create optimized data loading interface for ML pipeline consumption.

        Returns:
            DataLoader instance for efficient data access
        """
        return DataLoader(self)


class DataLoader:
    """Optimized data loading interface for ML pipeline consumption."""

    def __init__(self, parquet_manager: ParquetManager):
        """Initialize data loader.

        Args:
            parquet_manager: ParquetManager instance
        """
        self.manager = parquet_manager

    def load_training_data(
        self,
        data_types: list[str],
        start_date: str,
        end_date: str,
        tickers: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load training data efficiently.

        Args:
            data_types: List of data types to load
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            tickers: Optional ticker filter

        Returns:
            Dictionary of loaded DataFrames
        """
        data = {}

        for data_type in data_types:
            try:
                df = self.manager.load_dataframe(
                    data_type=data_type, start_date=start_date, end_date=end_date, columns=tickers
                )
                data[data_type] = df
            except Exception as e:
                # Log the specific error for debugging but continue with empty DataFrame
                import logging

                logging.warning(f"Failed to load data for {data_type}: {e}")
                data[data_type] = pd.DataFrame()

        return data

    def get_aligned_datasets(
        self,
        primary_data_type: str = "close",
        start_date: str = "2016-01-01",
        end_date: str = "2024-12-31",
        required_data_types: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Get aligned datasets for ML training.

        Args:
            primary_data_type: Primary data type for alignment
            start_date: Start date
            end_date: End date
            required_data_types: Required data types

        Returns:
            Dictionary of aligned DataFrames
        """
        if required_data_types is None:
            required_data_types = ["close", "volume", "returns"]

        # Load primary dataset first
        primary_data = self.manager.load_dataframe(
            data_type=primary_data_type, start_date=start_date, end_date=end_date
        )

        if primary_data.empty:
            return {dt: pd.DataFrame() for dt in required_data_types}

        # Get common index and columns from primary data
        common_index = primary_data.index
        common_columns = primary_data.columns

        aligned_data = {primary_data_type: primary_data}

        # Load and align other data types
        for data_type in required_data_types:
            if data_type == primary_data_type:
                continue

            df = self.manager.load_dataframe(
                data_type=data_type, start_date=start_date, end_date=end_date
            )

            if not df.empty:
                # Align to common index and columns
                aligned_df = df.reindex(index=common_index, columns=common_columns)
                aligned_data[data_type] = aligned_df
            else:
                aligned_data[data_type] = pd.DataFrame(
                    index=common_index, columns=common_columns, dtype=float
                )

        return aligned_data
