"""
Base portfolio model interfaces and constraint system.

This module provides the foundational interfaces and constraint enforcement
that ensure consistent APIs and risk management across all portfolio models.
"""

from .constraint_engine import UnifiedConstraintEngine
from .constraints import (
    ConstraintEngine,
    ConstraintViolation,
    ConstraintViolationType,
    PortfolioConstraints,
    ViolationSeverity,
)
from .portfolio_model import PortfolioModel
from .violation_handler import RemediationStrategy, ViolationHandler

__all__ = [
    "PortfolioModel",
    "PortfolioConstraints",
    "ConstraintEngine",
    "UnifiedConstraintEngine",
    "ViolationHandler",
    "ConstraintViolation",
    "ConstraintViolationType",
    "ViolationSeverity",
    "RemediationStrategy",
]
