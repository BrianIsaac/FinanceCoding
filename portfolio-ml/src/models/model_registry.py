"""
Model registry for portfolio optimization models.

This module provides a centralized registry for all portfolio models,
enabling dynamic model loading and integration with the backtesting framework.
"""

from __future__ import annotations

from typing import Any

from .base.portfolio_model import PortfolioModel
from .hrp.model import HRPModel
from .lstm.model import LSTMPortfolioModel


class ModelRegistry:
    """Registry for portfolio optimization models."""

    _models: dict[str, type[PortfolioModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: type[PortfolioModel]) -> None:
        """
        Register a portfolio model.

        Args:
            name: Model identifier
            model_class: Model class implementing PortfolioModel interface
        """
        cls._models[name] = model_class

    @classmethod
    def get_model(cls, name: str) -> type[PortfolioModel]:
        """
        Get a registered model class.

        Args:
            name: Model identifier

        Returns:
            Model class

        Raises:
            ValueError: If model is not registered
        """
        if name not in cls._models:
            raise ValueError(
                f"Model '{name}' not registered. Available models: {list(cls._models.keys())}"
            )

        return cls._models[name]

    @classmethod
    def list_models(cls) -> list[str]:
        """
        List all registered models.

        Returns:
            List of model names
        """
        return list(cls._models.keys())

    @classmethod
    def create_model(cls, name: str, **kwargs: Any) -> PortfolioModel:
        """
        Create a model instance.

        Args:
            name: Model identifier
            **kwargs: Model initialization parameters

        Returns:
            Initialized model instance
        """
        model_class = cls.get_model(name)
        return model_class(**kwargs)


# Register available models
ModelRegistry.register("hrp", HRPModel)
ModelRegistry.register("lstm", LSTMPortfolioModel)

# Note: GAT model will be registered when available
try:
    from .gat.model import GATModel

    ModelRegistry.register("gat", GATModel)
except ImportError:
    # GAT model not yet implemented
    pass
