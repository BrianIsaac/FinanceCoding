"""
Comprehensive gradient flow diagnostic utilities for LSTM debugging.

This module provides tools to diagnose zero gradient issues in PyTorch models,
specifically tailored for LSTM architectures with packed sequences.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)


def plot_grad_flow(named_parameters: list[tuple[str, torch.nn.Parameter]]) -> None:
    """
    Visualise gradient flow through model layers.

    Plots average and maximum gradient magnitudes per layer to identify
    vanishing or exploding gradients.

    Args:
        named_parameters: List of (name, parameter) tuples from model.named_parameters()
    """
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().item())
            max_grads.append(p.grad.abs().max().cpu().item())

    if not ave_grads:
        logger.warning("No gradients found to plot!")
        return

    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="c", label="max-gradient")
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.3, lw=1, color="b", label="mean-gradient")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads)), layers, rotation=90)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001)
    plt.xlabel("Layers")
    plt.ylabel("Gradient magnitude")
    plt.title("Gradient Flow Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("gradient_flow_analysis.png", dpi=150)
    logger.info("Gradient flow plot saved to gradient_flow_analysis.png")
    plt.close()


def compute_gradient_norm(model: nn.Module) -> dict[str, float]:
    """
    Compute gradient norms for all parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with total gradient norm and per-parameter norms
    """
    total_norm = 0.0
    param_norms = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            param_norms[name] = param_norm

    total_norm = total_norm ** 0.5

    return {
        "total_norm": total_norm,
        "param_norms": param_norms
    }


def diagnose_gradient_flow(
    model: nn.Module,
    loss: torch.Tensor,
    save_plot: bool = True
) -> dict[str, Any]:
    """
    Comprehensive gradient flow diagnostics.

    Args:
        model: PyTorch model
        loss: Computed loss tensor (before backward())
        save_plot: Whether to save gradient flow visualization

    Returns:
        Diagnostic report dictionary
    """
    logger.info("\n" + "="*80)
    logger.info("GRADIENT FLOW DIAGNOSTIC ANALYSIS")
    logger.info("="*80)

    # Check if loss requires gradients
    if not loss.requires_grad:
        logger.error("❌ CRITICAL: Loss does not require gradients!")
        logger.error("   This means gradients will NOT flow through the model.")
        return {"status": "FAILED", "reason": "loss.requires_grad=False"}

    # Check for NaN/Inf in loss
    if not torch.isfinite(loss):
        logger.error(f"❌ CRITICAL: Loss is not finite: {loss.item()}")
        return {"status": "FAILED", "reason": "non-finite loss"}

    # Perform backward pass with anomaly detection
    logger.info("Performing backward pass with anomaly detection...")
    try:
        with torch.autograd.detect_anomaly():
            loss.backward()
    except RuntimeError as e:
        logger.error(f"❌ CRITICAL: Backward pass failed with error: {e}")
        return {"status": "FAILED", "reason": f"backward error: {e}"}

    logger.info("✓ Backward pass completed successfully\n")

    # Analyse gradients
    logger.info("Analysing gradients per parameter:")
    logger.info("-" * 80)

    grad_report = {
        "status": "SUCCESS",
        "no_grad_params": [],
        "zero_grad_params": [],
        "small_grad_params": [],
        "healthy_grad_params": [],
        "large_grad_params": []
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            logger.warning(f"⚠️  {name}: requires_grad=False (frozen parameter)")
            grad_report["no_grad_params"].append(name)
            continue

        if param.grad is None:
            logger.error(f"❌ {name}: NO GRADIENT (None)")
            grad_report["no_grad_params"].append(name)
            continue

        grad_norm = param.grad.norm().item()

        if grad_norm == 0.0:
            logger.error(f"❌ {name}: ZERO GRADIENT")
            grad_report["zero_grad_params"].append(name)
        elif grad_norm < 1e-6:
            logger.warning(f"⚠️  {name}: Very small gradient ({grad_norm:.2e})")
            grad_report["small_grad_params"].append(name)
        elif grad_norm > 100:
            logger.warning(f"⚠️  {name}: Large gradient ({grad_norm:.2e})")
            grad_report["large_grad_params"].append(name)
        else:
            logger.info(f"✓ {name}: {grad_norm:.6f}")
            grad_report["healthy_grad_params"].append(name)

    # Compute total gradient norm
    grad_norms = compute_gradient_norm(model)
    total_norm = grad_norms["total_norm"]

    logger.info("-" * 80)
    logger.info(f"\nTotal Gradient Norm: {total_norm:.6f}")

    # Diagnostic summary
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("="*80)
    logger.info(f"Parameters with NO gradients: {len(grad_report['no_grad_params'])}")
    logger.info(f"Parameters with ZERO gradients: {len(grad_report['zero_grad_params'])}")
    logger.info(f"Parameters with small gradients (<1e-6): {len(grad_report['small_grad_params'])}")
    logger.info(f"Parameters with healthy gradients: {len(grad_report['healthy_grad_params'])}")
    logger.info(f"Parameters with large gradients (>100): {len(grad_report['large_grad_params'])}")

    # Determine overall status
    if len(grad_report['zero_grad_params']) > 0:
        logger.error("\n❌ ISSUE DETECTED: Zero gradients found!")
        grad_report["status"] = "ZERO_GRADIENTS"
        logger.error(f"   Affected parameters: {grad_report['zero_grad_params']}")
    elif total_norm < 1e-6:
        logger.error(f"\n❌ ISSUE DETECTED: Total gradient norm too small ({total_norm:.2e})")
        grad_report["status"] = "VANISHING_GRADIENTS"
    elif total_norm > 1000:
        logger.warning(f"\n⚠️  WARNING: Total gradient norm very large ({total_norm:.2e})")
        grad_report["status"] = "EXPLODING_GRADIENTS"
    else:
        logger.info("\n✓ Gradient flow appears healthy")
        grad_report["status"] = "HEALTHY"

    logger.info("="*80 + "\n")

    # Plot gradient flow
    if save_plot:
        plot_grad_flow(model.named_parameters())

    grad_report["total_norm"] = total_norm
    grad_report["param_norms"] = grad_norms["param_norms"]

    return grad_report


def check_hidden_state_detach(model_code_path: str = "src/models/lstm") -> None:
    """
    Check for improper hidden state detach() calls in LSTM code.

    Args:
        model_code_path: Path to model code directory
    """
    import subprocess

    logger.info("\n" + "="*80)
    logger.info("CHECKING FOR IMPROPER DETACH() CALLS")
    logger.info("="*80)

    # Search for detach() calls
    try:
        result = subprocess.run(
            ["grep", "-rn", "detach()", model_code_path],
            capture_output=True,
            text=True
        )

        if result.stdout:
            logger.info("Found detach() calls:")
            logger.info(result.stdout)
            logger.warning("\n⚠️  Review these calls to ensure they don't occur before loss.backward()")
        else:
            logger.info("✓ No detach() calls found")
    except Exception as e:
        logger.warning(f"Could not check for detach() calls: {e}")

    logger.info("="*80 + "\n")


def check_inplace_operations(model: nn.Module) -> None:
    """
    Check model for in-place operations that could break gradients.

    Args:
        model: PyTorch model
    """
    logger.info("\n" + "="*80)
    logger.info("CHECKING FOR IN-PLACE OPERATIONS")
    logger.info("="*80)

    inplace_ops = []

    for name, module in model.named_modules():
        # Check for ReLU with inplace=True
        if isinstance(module, nn.ReLU) and module.inplace:
            inplace_ops.append(f"{name}: ReLU(inplace=True)")

        # Check for other activation functions with inplace
        if isinstance(module, (nn.LeakyReLU, nn.ELU, nn.SELU)) and hasattr(module, 'inplace') and module.inplace:
            inplace_ops.append(f"{name}: {module.__class__.__name__}(inplace=True)")

    if inplace_ops:
        logger.warning("⚠️  Found in-place operations:")
        for op in inplace_ops:
            logger.warning(f"   {op}")
        logger.warning("   These may cause gradient issues if used incorrectly")
    else:
        logger.info("✓ No problematic in-place operations found")

    logger.info("="*80 + "\n")


def register_gradient_hooks(model: nn.Module) -> list[Any]:
    """
    Register backward hooks on LSTM layers to monitor gradient flow.

    Args:
        model: PyTorch model

    Returns:
        List of hook handles (for cleanup)
    """
    hooks = []

    def make_hook(name):
        def hook(grad):
            if grad is not None:
                grad_norm = grad.norm().item()
                logger.debug(f"Gradient hook [{name}]: norm={grad_norm:.6f}")
                if grad_norm == 0.0:
                    logger.error(f"❌ Zero gradient detected in {name}")
                elif grad_norm < 1e-6:
                    logger.warning(f"⚠️  Very small gradient in {name}: {grad_norm:.2e}")
            else:
                logger.error(f"❌ None gradient in {name}")
        return hook

    for name, param in model.named_parameters():
        if param.requires_grad:
            handle = param.register_hook(make_hook(name))
            hooks.append(handle)

    logger.info(f"Registered {len(hooks)} gradient hooks")
    return hooks


def remove_gradient_hooks(hooks: list[Any]) -> None:
    """
    Remove registered gradient hooks.

    Args:
        hooks: List of hook handles from register_gradient_hooks()
    """
    for hook in hooks:
        hook.remove()
    logger.info(f"Removed {len(hooks)} gradient hooks")
