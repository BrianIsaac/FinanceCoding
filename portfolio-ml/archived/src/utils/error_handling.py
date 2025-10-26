"""
Enhanced error handling and logging utilities.

This module provides comprehensive error handling, logging, and recovery
mechanisms for robust model training and backtesting operations.
"""

from __future__ import annotations

import functools
import logging
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ErrorContext:
    """Contextual information about an error."""

    error_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = ""
    operation: str = ""
    error_type: str = ""
    error_message: str = ""
    stack_trace: str = ""
    context_data: dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    user_guidance: str = ""


@dataclass
class ErrorRecoveryStrategy:
    """Strategy for error recovery."""

    strategy_name: str
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    fallback_action: Optional[Callable] = None
    cleanup_action: Optional[Callable] = None


class EnhancedErrorHandler:
    """
    Comprehensive error handling system.

    Provides structured error logging, recovery strategies, and user guidance
    for robust operation in production environments.
    """

    def __init__(
        self,
        component_name: str,
        log_file: Optional[Path] = None,
        enable_detailed_logging: bool = True
    ):
        """
        Initialise error handler.

        Args:
            component_name: Name of the component using this handler
            log_file: Optional file to write error logs
            enable_detailed_logging: Whether to include stack traces
        """
        self.component_name = component_name
        self.log_file = log_file
        self.enable_detailed_logging = enable_detailed_logging

        # Error tracking
        self.error_history: list[ErrorContext] = []
        self.recovery_strategies: dict[str, ErrorRecoveryStrategy] = {}

        # Configure logging
        self._setup_logging()

        # Register common recovery strategies
        self._register_default_strategies()

    def _setup_logging(self) -> None:
        """Configure enhanced logging."""
        # Create formatter with detailed information
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )

        # Add file handler if specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.setLevel(logging.DEBUG)

    def _register_default_strategies(self) -> None:
        """Register common error recovery strategies."""
        # Memory cleanup strategy
        self.register_recovery_strategy(
            "memory_cleanup",
            ErrorRecoveryStrategy(
                strategy_name="memory_cleanup",
                max_retries=2,
                cleanup_action=self._cleanup_memory
            )
        )

        # GPU memory recovery
        self.register_recovery_strategy(
            "gpu_memory_recovery",
            ErrorRecoveryStrategy(
                strategy_name="gpu_memory_recovery",
                max_retries=3,
                retry_delay_seconds=2.0,
                cleanup_action=self._cleanup_gpu_memory
            )
        )

        # Data validation recovery
        self.register_recovery_strategy(
            "data_validation",
            ErrorRecoveryStrategy(
                strategy_name="data_validation",
                max_retries=1,
                fallback_action=self._fallback_data_validation
            )
        )

    def register_recovery_strategy(
        self,
        strategy_key: str,
        strategy: ErrorRecoveryStrategy
    ) -> None:
        """Register a recovery strategy."""
        self.recovery_strategies[strategy_key] = strategy
        logger.debug(f"Registered recovery strategy: {strategy_key}")

    def handle_error(
        self,
        error: Exception,
        operation: str = "",
        context_data: Optional[dict[str, Any]] = None,
        recovery_strategy: Optional[str] = None,
        critical: bool = False
    ) -> ErrorContext:
        """
        Handle an error with comprehensive logging and optional recovery.

        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            context_data: Additional context information
            recovery_strategy: Name of recovery strategy to attempt
            critical: Whether this is a critical error that should stop execution

        Returns:
            ErrorContext with details about the error and recovery attempt
        """
        # Generate unique error ID
        error_id = f"{self.component_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.error_history)}"

        # Create error context
        error_context = ErrorContext(
            error_id=error_id,
            component=self.component_name,
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc() if self.enable_detailed_logging else "",
            context_data=context_data or {}
        )

        # Add error to history
        self.error_history.append(error_context)

        # Log the error
        log_message = (
            f"Error in {self.component_name}.{operation}: {error_context.error_type}: {error_context.error_message}"
        )

        if critical:
            logger.critical(log_message)
            if self.enable_detailed_logging:
                logger.critical(f"Stack trace:\n{error_context.stack_trace}")
        else:
            logger.error(log_message)
            if self.enable_detailed_logging:
                logger.debug(f"Stack trace:\n{error_context.stack_trace}")

        # Log context data
        if context_data:
            logger.info(f"Error context: {context_data}")

        # Attempt recovery if strategy is specified
        if recovery_strategy and recovery_strategy in self.recovery_strategies:
            error_context.recovery_attempted = True
            success = self._attempt_recovery(error_context, recovery_strategy)
            error_context.recovery_successful = success

            if success:
                logger.info(f"Successfully recovered from error {error_id} using strategy {recovery_strategy}")
            else:
                logger.warning(f"Failed to recover from error {error_id} using strategy {recovery_strategy}")

        # Add user guidance
        error_context.user_guidance = self._generate_user_guidance(error_context)

        return error_context

    def _attempt_recovery(self, error_context: ErrorContext, strategy_key: str) -> bool:
        """Attempt to recover from an error using the specified strategy."""
        strategy = self.recovery_strategies[strategy_key]

        for attempt in range(strategy.max_retries):
            try:
                logger.info(f"Recovery attempt {attempt + 1}/{strategy.max_retries} for error {error_context.error_id}")

                # Execute cleanup action if available
                if strategy.cleanup_action:
                    strategy.cleanup_action()

                # Execute fallback action if available
                if strategy.fallback_action:
                    strategy.fallback_action()

                # Wait before next attempt
                if attempt < strategy.max_retries - 1:
                    import time
                    time.sleep(strategy.retry_delay_seconds)

                logger.info(f"Recovery attempt {attempt + 1} completed successfully")
                return True

            except Exception as recovery_error:
                logger.warning(f"Recovery attempt {attempt + 1} failed: {recovery_error}")
                if attempt == strategy.max_retries - 1:
                    logger.error(f"All recovery attempts failed for error {error_context.error_id}")

        return False

    def _generate_user_guidance(self, error_context: ErrorContext) -> str:
        """Generate helpful guidance for the user based on the error."""
        error_type = error_context.error_type
        error_message = error_context.error_message.lower()

        # Common error patterns and guidance
        if "out of memory" in error_message or "cuda" in error_message:
            return (
                "GPU memory error detected. Try reducing batch size, enabling mixed precision, "
                "or closing other GPU-intensive applications."
            )

        elif "dimension" in error_message or "shape" in error_message:
            return (
                "Data dimension mismatch detected. Check that input data shapes match model expectations. "
                "Verify universe sizes and feature dimensions."
            )

        elif "file not found" in error_message or "filenotfound" in error_type.lower():
            return (
                "Required file missing. Ensure all data files exist and paths are correct. "
                "Check data processing pipeline completion."
            )

        elif "nan" in error_message or "inf" in error_message:
            return (
                "Invalid numerical values detected. Check data quality, ensure proper handling "
                "of missing values, and validate calculation results."
            )

        elif "timeout" in error_message or "connection" in error_message:
            return (
                "Network or connection issue detected. Check internet connectivity and retry. "
                "Consider implementing retry logic for data downloads."
            )

        else:
            return (
                f"Unexpected {error_type} occurred. Check logs for details and verify system configuration. "
                "Consider filing an issue if problem persists."
            )

    def _cleanup_memory(self) -> None:
        """Clean up system memory."""
        import gc
        gc.collect()
        logger.debug("System memory cleanup completed")

    def _cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("GPU memory cleanup completed")
        except ImportError:
            logger.warning("PyTorch not available for GPU memory cleanup")

    def _fallback_data_validation(self) -> None:
        """Fallback action for data validation errors."""
        logger.info("Attempting data validation fallback - using relaxed validation criteria")

    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of all errors encountered."""
        if not self.error_history:
            return {"total_errors": 0, "error_types": {}, "recovery_success_rate": 0.0}

        error_types = {}
        recovery_attempts = 0
        successful_recoveries = 0

        for error in self.error_history:
            error_type = error.error_type
            error_types[error_type] = error_types.get(error_type, 0) + 1

            if error.recovery_attempted:
                recovery_attempts += 1
                if error.recovery_successful:
                    successful_recoveries += 1

        recovery_rate = (successful_recoveries / recovery_attempts * 100) if recovery_attempts > 0 else 0.0

        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "recovery_attempts": recovery_attempts,
            "successful_recoveries": successful_recoveries,
            "recovery_success_rate": recovery_rate,
            "recent_errors": [
                {
                    "error_id": error.error_id,
                    "timestamp": error.timestamp.isoformat(),
                    "error_type": error.error_type,
                    "operation": error.operation,
                    "recovered": error.recovery_successful
                }
                for error in self.error_history[-5:]  # Last 5 errors
            ]
        }

    def save_error_report(self, output_path: Path) -> None:
        """Save detailed error report to file."""
        report = {
            "component": self.component_name,
            "report_timestamp": datetime.now().isoformat(),
            "summary": self.get_error_summary(),
            "detailed_errors": [
                {
                    "error_id": error.error_id,
                    "timestamp": error.timestamp.isoformat(),
                    "component": error.component,
                    "operation": error.operation,
                    "error_type": error.error_type,
                    "error_message": error.error_message,
                    "context_data": error.context_data,
                    "recovery_attempted": error.recovery_attempted,
                    "recovery_successful": error.recovery_successful,
                    "user_guidance": error.user_guidance
                }
                for error in self.error_history
            ]
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([report]).to_json(output_path, orient="records", indent=2)
        logger.info(f"Error report saved to {output_path}")


def error_handler(
    component: str = "",
    operation: str = "",
    recovery_strategy: Optional[str] = None,
    critical: bool = False,
    reraise: bool = True
):
    """
    Decorator for automatic error handling.

    Args:
        component: Component name for error tracking
        operation: Operation description
        recovery_strategy: Recovery strategy to attempt
        critical: Whether error is critical
        reraise: Whether to re-raise the exception after handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = EnhancedErrorHandler(component or func.__module__)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = handler.handle_error(
                    error=e,
                    operation=operation or func.__name__,
                    recovery_strategy=recovery_strategy,
                    critical=critical
                )

                if reraise and not error_context.recovery_successful:
                    raise

                return None

        return wrapper
    return decorator


# Global error handler instance
global_error_handler = EnhancedErrorHandler("global", enable_detailed_logging=True)