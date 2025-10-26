"""Environment helpers (loads .env from project root even under Hydra)."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from hydra.utils import to_absolute_path


def load_project_dotenv(filename: str = ".env", override: bool = False) -> None:
    """Load environment variables from a .env at the project root.

    Hydra changes cwd to the run directory, so we resolve the path
    relative to the original project root.

    Args:
        filename: Name of the .env file at the project root.
        override: Whether variables in .env should override existing env vars.
    """
    path = to_absolute_path(filename)
    load_dotenv(dotenv_path=path, override=override)


def require_env(var_name: str) -> str:
    """Return the value of an environment variable or raise a helpful error.

    Args:
        var_name: Environment variable name to fetch.

    Returns:
        The variable's value.

    Raises:
        RuntimeError: If the variable is not set after loading .env.
    """
    val = os.getenv(var_name)
    if not val:
        raise RuntimeError(
            f"Missing environment variable '{var_name}'. "
            "Add it to your .env at the project root or export it in your shell."
        )
    return val
