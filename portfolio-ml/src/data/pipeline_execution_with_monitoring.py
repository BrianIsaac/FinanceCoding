"""
Enhanced data pipeline execution with comprehensive temporal integrity monitoring.

This module integrates automated look-ahead bias detection, memory-efficient processing,
and system resource monitoring for robust large-scale data pipeline execution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.processors.memory_efficient_processor import MemoryEfficientProcessor
from src.data.processors.temporal_bias_detector import DataPipelineBiasDetector
from src.monitoring.resource_monitor import SystemResourceMonitor

logger = logging.getLogger(__name__)


class EnhancedDataPipelineExecutor:
    """
    Enhanced data pipeline executor with integrated monitoring and validation.

    Combines temporal integrity validation, memory-efficient processing, and
    comprehensive resource monitoring for robust pipeline execution.
    """

    def __init__(
        self,
        max_memory_mb: float = 16000,
        chunk_size_months: int = 6,
        ticker_batch_size: int = 100,
        enable_strict_temporal_validation: bool = True,
        monitoring_interval: float = 10.0,
        auto_generate_reports: bool = True,
        output_dir: str = "data/processed",
    ):
        """
        Initialize enhanced pipeline executor.

        Args:
            max_memory_mb: Maximum memory usage limit
            chunk_size_months: Temporal chunk size for processing
            ticker_batch_size: Batch size for ticker processing
            enable_strict_temporal_validation: Enable strict temporal validation
            monitoring_interval: Resource monitoring interval in seconds
            auto_generate_reports: Auto-generate monitoring reports
            output_dir: Output directory for processed data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.memory_processor = MemoryEfficientProcessor(
            max_memory_mb=max_memory_mb,
            chunk_size_months=chunk_size_months,
            ticker_batch_size=ticker_batch_size,
            enable_monitoring=True,
        )

        self.bias_detector = DataPipelineBiasDetector(
            strict_mode=enable_strict_temporal_validation,
            auto_report=auto_generate_reports,
            report_dir=str(self.output_dir / "quality_reports"),
        )

        self.resource_monitor = SystemResourceMonitor(
            monitoring_interval=monitoring_interval,
            auto_export_reports=auto_generate_reports,
            reports_dir=str(self.output_dir / "monitoring_reports"),
        )

        logger.info("Enhanced data pipeline executor initialized")

    def execute_full_pipeline(
        self,
        start_date: str,
        end_date: str,
        universe_tickers: list[str],
        data_sources: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """
        Execute complete data pipeline with monitoring and validation.

        Args:
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
            universe_tickers: List of ticker symbols to process
            data_sources: Configuration for data sources

        Returns:
            Pipeline execution summary with all validation and monitoring results
        """
        pipeline_start = pd.Timestamp.now()

        # Start system resource monitoring
        self.resource_monitor.start_monitoring("full_data_pipeline")

        try:
            logger.info(
                f"Starting enhanced pipeline execution: {start_date} to {end_date}, "
                f"{len(universe_tickers)} tickers"
            )

            execution_summary = {
                "pipeline_start": pipeline_start.isoformat(),
                "parameters": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "universe_size": len(universe_tickers),
                    "data_sources": data_sources or {},
                },
                "stages": {},
                "final_status": "unknown",
            }

            # Stage 1: Universe Construction with Temporal Validation
            execution_summary["stages"]["universe_construction"] = self._execute_universe_construction(
                start_date, end_date, universe_tickers
            )

            # Stage 2: Multi-Source Data Collection with Validation
            execution_summary["stages"]["data_collection"] = self._execute_data_collection(
                start_date, end_date, universe_tickers, data_sources
            )

            # Stage 3: Gap Filling with Temporal Integrity Checks
            execution_summary["stages"]["gap_filling"] = self._execute_gap_filling(
                start_date, end_date
            )

            # Stage 4: Final Dataset Generation with Validation
            execution_summary["stages"]["dataset_generation"] = self._execute_dataset_generation(
                start_date, end_date
            )

            # Determine final pipeline status
            execution_summary["final_status"] = self._determine_pipeline_status(execution_summary)
            execution_summary["pipeline_end"] = pd.Timestamp.now().isoformat()
            execution_summary["total_duration"] = str(pd.Timestamp.now() - pipeline_start)

            logger.info(f"Pipeline execution completed: {execution_summary['final_status']}")
            return execution_summary

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            execution_summary["final_status"] = "failed"
            execution_summary["error"] = str(e)
            return execution_summary

        finally:
            # Stop resource monitoring and get session data
            monitoring_session = self.resource_monitor.stop_monitoring()
            if monitoring_session:
                execution_summary["resource_monitoring"] = self.resource_monitor.get_session_summary(
                    monitoring_session
                )

            # Get comprehensive bias detection summary
            execution_summary["temporal_integrity"] = self.bias_detector.get_pipeline_integrity_summary()

    def _execute_universe_construction(
        self, start_date: str, end_date: str, universe_tickers: list[str]
    ) -> dict[str, Any]:
        """Execute universe construction stage with temporal validation."""
        stage_start = pd.Timestamp.now()

        with self.memory_processor.memory_monitor("universe_construction"):
            logger.info("Stage 1: Universe construction with temporal validation")

            # Simulate universe construction (in real implementation, this would use UniverseBuilder)
            universe_calendar = pd.DataFrame({
                "ticker": universe_tickers,
                "start": [pd.to_datetime(start_date) for _ in universe_tickers],
                "end": [pd.to_datetime(end_date) for _ in universe_tickers],
            })

            # Validate universe construction temporal integrity
            construction_result = self.bias_detector.validate_universe_construction_integrity(
                membership_data=universe_calendar,
                construction_date=pd.Timestamp.now(),
                universe_type="midcap400"
            )

            # Save universe calendar
            universe_path = self.output_dir / "universe_calendar.parquet"
            universe_calendar.to_parquet(universe_path)

            return {
                "status": "completed" if construction_result.clean else "completed_with_warnings",
                "duration": str(pd.Timestamp.now() - stage_start),
                "universe_size": len(universe_tickers),
                "temporal_validation": {
                    "clean": construction_result.clean,
                    "violations": construction_result.violations_detected,
                    "critical_violations": construction_result.critical_violations,
                },
                "output_path": str(universe_path),
            }

    def _execute_data_collection(
        self, start_date: str, end_date: str, universe_tickers: list[str], data_sources: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute data collection stage with memory-efficient processing."""
        stage_start = pd.Timestamp.now()

        logger.info("Stage 2: Multi-source data collection with memory-efficient processing")

        # Data collection results
        collection_results = []

        # Simulate chunked data collection
        def mock_data_loader(start_date, end_date, tickers, **kwargs):
            """Mock data loader for demonstration."""
            dates = pd.date_range(start_date, end_date, freq="D")
            data = pd.DataFrame(
                {ticker: [100 + i * 0.1 for i in range(len(dates))] for ticker in tickers},
                index=dates
            )
            return data

        def mock_data_processor(data, **kwargs):
            """Mock data processor for demonstration."""
            # Simple processing: calculate returns
            return data.pct_change().dropna()

        # Process data in chunks
        total_chunks = 0
        for chunk in self.memory_processor.process_data_in_chunks(
            data_loader_func=mock_data_loader,
            processor_func=mock_data_processor,
            start_date=start_date,
            end_date=end_date,
            tickers=universe_tickers,
            output_dir=self.output_dir / "chunks",
        ):
            # Validate each chunk for temporal integrity
            validation_result = self.bias_detector.validate_data_collection_integrity(
                collected_data=chunk.data,
                collection_date=pd.Timestamp.now(),
                source_name="stooq_chunk"
            )

            collection_results.append({
                "chunk_id": chunk.chunk_id,
                "temporal_validation": validation_result.clean,
                "memory_mb": chunk.memory_mb,
                "data_shape": chunk.data.shape,
            })

            total_chunks += 1

        return {
            "status": "completed",
            "duration": str(pd.Timestamp.now() - stage_start),
            "total_chunks_processed": total_chunks,
            "chunk_results": collection_results,
            "temporal_violations": sum(1 for r in collection_results if not r["temporal_validation"]),
        }

    def _execute_gap_filling(self, start_date: str, end_date: str) -> dict[str, Any]:
        """Execute gap filling stage with temporal integrity validation."""
        stage_start = pd.Timestamp.now()

        with self.memory_processor.memory_monitor("gap_filling"):
            logger.info("Stage 3: Gap filling with temporal integrity validation")

            # Simulate gap filling process
            sample_data_path = self.output_dir / "chunks" / "chunk_0000.parquet"

            if sample_data_path.exists():
                # Load sample data for gap filling validation
                original_data = pd.read_parquet(sample_data_path)

                # Simulate gap filling (forward fill)
                filled_data = original_data.ffill()

                # Validate gap filling temporal integrity
                gap_fill_result = self.bias_detector.validate_gap_filling_integrity(
                    original_data=original_data,
                    filled_data=filled_data,
                    fill_date=pd.Timestamp.now(),
                    fill_method="forward_fill"
                )

                return {
                    "status": "completed" if gap_fill_result.clean else "completed_with_warnings",
                    "duration": str(pd.Timestamp.now() - stage_start),
                    "temporal_validation": {
                        "clean": gap_fill_result.clean,
                        "violations": gap_fill_result.violations_detected,
                    },
                }
            else:
                return {
                    "status": "skipped",
                    "reason": "No data chunks available for gap filling",
                    "duration": str(pd.Timestamp.now() - stage_start),
                }

    def _execute_dataset_generation(self, start_date: str, end_date: str) -> dict[str, Any]:
        """Execute final dataset generation with validation."""
        stage_start = pd.Timestamp.now()

        logger.info("Stage 4: Final dataset generation with validation")

        # Generate final datasets
        datasets_generated = []

        for data_type in ["prices", "returns", "volume"]:
            # Simulate dataset generation
            dates = pd.date_range(start_date, end_date, freq="D")
            sample_data = pd.DataFrame(
                {"AAPL": range(len(dates)), "MSFT": range(len(dates))},
                index=dates
            )

            # Save dataset
            dataset_path = self.output_dir / f"{data_type}.parquet"
            sample_data.to_parquet(dataset_path)

            # Validate parquet generation
            parquet_result = self.bias_detector.validate_parquet_generation_integrity(
                source_data=sample_data,
                generated_parquet_path=dataset_path,
                generation_date=pd.Timestamp.now(),
                data_type=data_type
            )

            datasets_generated.append({
                "data_type": data_type,
                "path": str(dataset_path),
                "temporal_validation": parquet_result.clean,
                "violations": parquet_result.violations_detected,
            })

        return {
            "status": "completed",
            "duration": str(pd.Timestamp.now() - stage_start),
            "datasets_generated": datasets_generated,
            "total_violations": sum(d["violations"] for d in datasets_generated),
        }

    def _determine_pipeline_status(self, execution_summary: dict[str, Any]) -> str:
        """Determine overall pipeline execution status."""
        # Check if any stage failed
        for _stage_name, stage_result in execution_summary["stages"].items():
            if stage_result.get("status") == "failed":
                return "failed"

        # Check for critical temporal violations
        temporal_summary = self.bias_detector.get_pipeline_integrity_summary()
        if temporal_summary.get("critical_violations", 0) > 0:
            return "completed_with_critical_issues"

        # Check for warnings
        has_warnings = False
        for _stage_name, stage_result in execution_summary["stages"].items():
            if "completed_with_warnings" in stage_result.get("status", ""):
                has_warnings = True
                break

        return "completed_with_warnings" if has_warnings else "completed_successfully"

    def get_comprehensive_health_report(self) -> dict[str, Any]:
        """Get comprehensive health report for pipeline systems."""
        system_health = self.resource_monitor.get_system_health_check()
        pipeline_integrity = self.bias_detector.get_pipeline_integrity_summary()
        memory_stats = self.memory_processor.get_memory_stats()

        return {
            "report_timestamp": pd.Timestamp.now().isoformat(),
            "system_health": system_health,
            "temporal_integrity": pipeline_integrity,
            "memory_status": {
                "total_mb": memory_stats.total_mb,
                "available_mb": memory_stats.available_mb,
                "process_mb": memory_stats.process_mb,
                "percent_used": memory_stats.percent_used,
            },
            "recommendations": self._generate_health_recommendations(
                system_health, pipeline_integrity, memory_stats
            ),
        }

    def _generate_health_recommendations(
        self, system_health: dict, pipeline_integrity: dict, memory_stats: Any
    ) -> list[str]:
        """Generate health recommendations based on current status."""
        recommendations = []

        # System health recommendations
        if system_health["health_status"] != "healthy":
            recommendations.extend(system_health.get("recommendations", []))

        # Temporal integrity recommendations
        if pipeline_integrity.get("critical_violations", 0) > 0:
            recommendations.append("CRITICAL: Address temporal integrity violations before proceeding")
            recommendations.append("Review bias detection reports and fix look-ahead bias issues")

        # Memory recommendations
        if memory_stats.process_mb > 8000:  # 8GB
            recommendations.append("High memory usage detected - consider reducing chunk sizes")

        if memory_stats.percent_used > 85:
            recommendations.append("System memory usage high - close unnecessary applications")

        return recommendations

    def cleanup_intermediate_files(self, keep_reports: bool = True) -> dict[str, Any]:
        """Clean up intermediate files generated during pipeline execution."""
        cleanup_summary = {"files_removed": 0, "space_freed_mb": 0, "errors": []}

        try:
            # Clean up chunk files
            chunks_dir = self.output_dir / "chunks"
            if chunks_dir.exists():
                for chunk_file in chunks_dir.glob("chunk_*.parquet"):
                    file_size_mb = chunk_file.stat().st_size / (1024 * 1024)
                    chunk_file.unlink()
                    cleanup_summary["files_removed"] += 1
                    cleanup_summary["space_freed_mb"] += file_size_mb

            # Clean up old reports if not keeping them
            if not keep_reports:
                self.resource_monitor.cleanup_old_reports(days_to_keep=0)

            logger.info(
                f"Cleanup completed: {cleanup_summary['files_removed']} files removed, "
                f"{cleanup_summary['space_freed_mb']:.1f}MB freed"
            )

        except Exception as e:
            cleanup_summary["errors"].append(str(e))
            logger.error(f"Error during cleanup: {e}")

        return cleanup_summary


# Example usage function
def demonstrate_enhanced_pipeline():
    """Demonstrate the enhanced pipeline with all monitoring and validation features."""

    # Initialize enhanced pipeline executor
    executor = EnhancedDataPipelineExecutor(
        max_memory_mb=8000,  # 8GB limit
        chunk_size_months=3,  # 3-month chunks
        ticker_batch_size=50,  # 50 tickers per batch
        enable_strict_temporal_validation=True,
        monitoring_interval=5.0,  # 5-second monitoring interval
        auto_generate_reports=True,
    )

    # Example execution parameters
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    universe_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # Small example

    try:
        # Execute full pipeline
        logger.info("Starting enhanced pipeline demonstration")

        execution_summary = executor.execute_full_pipeline(
            start_date=start_date,
            end_date=end_date,
            universe_tickers=universe_tickers,
            data_sources={"primary": "stooq", "fallback": "yahoo_finance"}
        )

        # Print execution summary
        logger.info(f"Pipeline execution status: {execution_summary['final_status']}")
        logger.info(f"Total duration: {execution_summary['total_duration']}")

        # Get health report
        health_report = executor.get_comprehensive_health_report()
        logger.info(f"System health status: {health_report['system_health']['health_status']}")

        if health_report["recommendations"]:
            logger.info("Health recommendations:")
            for rec in health_report["recommendations"]:
                logger.info(f"  - {rec}")

        return execution_summary

    except Exception as e:
        logger.error(f"Pipeline demonstration failed: {str(e)}")
        raise

    finally:
        # Cleanup intermediate files
        cleanup_result = executor.cleanup_intermediate_files(keep_reports=True)
        logger.info(f"Cleanup: {cleanup_result['files_removed']} files removed")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run demonstration
    demonstrate_enhanced_pipeline()
