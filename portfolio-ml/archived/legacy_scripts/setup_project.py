#!/usr/bin/env python3
"""
Project setup and verification script.

This script initializes the development environment and verifies all tools
are properly configured.
"""

import subprocess
import sys
from pathlib import Path

import torch


def check_command(command: str) -> bool:
    """Check if a command is available."""
    try:
        subprocess.run([command, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_python_version() -> bool:
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        return True
    else:
        return False


def check_uv() -> bool:
    """Check uv package manager."""
    if check_command("uv"):
        return True
    else:
        return False


def check_gpu() -> bool:
    """Check GPU availability."""
    if torch.cuda.is_available():
        torch.cuda.get_device_name(0)
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return True
    else:
        return False


def check_dependencies() -> bool:
    """Check critical dependencies."""
    try:
        import black
        import flake8
        import isort
        import mypy
        import numpy
        import pandas
        import pytest
        import torch
        import torch_geometric

        return True
    except ImportError:
        return False


def check_project_structure() -> bool:
    """Check project directory structure."""
    required_dirs = [
        "src/config",
        "src/data",
        "src/models",
        "src/evaluation",
        "src/utils",
        "tests/unit",
        "tests/integration",
        "tests/fixtures",
        "configs",
        "scripts",
        "notebooks",
        "data",
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)

    if missing_dirs:
        return False
    else:
        return True


def run_setup_checks() -> bool:
    """Run all setup verification checks."""

    checks = [
        ("Python Version", check_python_version),
        ("UV Package Manager", check_uv),
        ("GPU Support", check_gpu),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
    ]

    all_passed = True
    for _check_name, check_func in checks:
        if not check_func():
            all_passed = False

    if all_passed:
        pass
    else:
        pass

    return all_passed


if __name__ == "__main__":
    success = run_setup_checks()
    sys.exit(0 if success else 1)
