"""
Comprehensive training flow debugging toolkit.

This module provides instrumentation to track the actual execution flow during
LSTM training, specifically focusing on gradient accumulation, optimizer steps,
and parameter updates.
"""

from __future__ import annotations

import logging
from typing import Any
from collections import defaultdict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TrainingFlowDebugger:
    """
    Comprehensive training flow debugger to track optimizer steps,
    gradient accumulation, and parameter updates.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """
        Initialise training flow debugger.

        Args:
            model: PyTorch model to debug
            optimizer: Optimizer to monitor
        """
        self.model = model
        self.optimizer = optimizer

        # Tracking state
        self.epoch_stats = defaultdict(lambda: {
            'batch_indices': [],
            'optimizer_steps': 0,
            'backward_calls': 0,
            'clip_calls': 0,
            'accumulation_triggered': [],
            'gradient_norms': [],
        })

        self.current_epoch = 0

        # Store initial parameters for change detection
        self.initial_params = {}
        self.last_epoch_params = {}
        self._store_parameters('initial')

        # Hook handles
        self.hooks = []

    def _store_parameters(self, label: str) -> None:
        """Store current parameter values for comparison."""
        param_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_dict[name] = param.data.clone().cpu()

        if label == 'initial':
            self.initial_params = param_dict
        else:
            self.last_epoch_params = param_dict

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for tracking."""
        # Store parameters from last epoch
        if epoch > 0:
            self._store_parameters(f'epoch_{epoch-1}')

        self.current_epoch = epoch
        logger.info(f"\n{'='*80}")
        logger.info(f"DEBUGGER: Starting Epoch {epoch}")
        logger.info(f"{'='*80}")

    def log_batch_start(self, batch_idx: int) -> None:
        """Log when a batch starts."""
        self.epoch_stats[self.current_epoch]['batch_indices'].append(batch_idx)
        logger.debug(f"DEBUGGER: Epoch {self.current_epoch}, Batch {batch_idx} started")

    def log_backward_call(self, batch_idx: int, loss_value: float) -> None:
        """Log when backward() is called."""
        self.epoch_stats[self.current_epoch]['backward_calls'] += 1
        logger.debug(
            f"DEBUGGER: Backward called at epoch {self.current_epoch}, "
            f"batch {batch_idx}, loss={loss_value:.6f}"
        )

    def log_gradient_clip(self, batch_idx: int, grad_norm: float, clip_value: float) -> None:
        """Log when gradient clipping occurs."""
        self.epoch_stats[self.current_epoch]['clip_calls'] += 1
        self.epoch_stats[self.current_epoch]['gradient_norms'].append(grad_norm)
        logger.info(
            f"DEBUGGER: ✓ Gradient clipped at epoch {self.current_epoch}, "
            f"batch {batch_idx}, norm={grad_norm:.2f}, clip_value={clip_value}"
        )

    def log_optimizer_step(self, batch_idx: int) -> None:
        """Log when optimizer.step() is called."""
        self.epoch_stats[self.current_epoch]['optimizer_steps'] += 1
        logger.info(
            f"DEBUGGER: ✓ Optimizer step #{self.epoch_stats[self.current_epoch]['optimizer_steps']} "
            f"at epoch {self.current_epoch}, batch {batch_idx}"
        )

    def log_accumulation_check(
        self,
        batch_idx: int,
        accumulation_steps: int,
        condition_result: bool
    ) -> None:
        """Log gradient accumulation condition checks."""
        self.epoch_stats[self.current_epoch]['accumulation_triggered'].append({
            'batch_idx': batch_idx,
            'accumulation_steps': accumulation_steps,
            'condition': f"({batch_idx} + 1) % {accumulation_steps}",
            'result': f"{(batch_idx + 1) % accumulation_steps}",
            'triggered': condition_result
        })

        logger.debug(
            f"DEBUGGER: Accumulation check: "
            f"({batch_idx} + 1) % {accumulation_steps} = {(batch_idx + 1) % accumulation_steps} "
            f"({'TRIGGERED' if condition_result else 'SKIPPED'})"
        )

    def check_parameter_changes(self, epoch: int) -> dict[str, Any]:
        """
        Check if parameters have changed since last epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with parameter change statistics
        """
        if not self.last_epoch_params:
            return {'status': 'no_previous_epoch', 'changed': False}

        changes = {}
        total_change = 0.0
        max_change = 0.0
        changed_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.last_epoch_params:
                old_param = self.last_epoch_params[name]
                new_param = param.data.cpu()

                # Calculate L2 norm of change
                change = (new_param - old_param).norm().item()
                changes[name] = change
                total_change += change
                max_change = max(max_change, change)

                if change > 1e-8:
                    changed_params.append(name)

        result = {
            'status': 'checked',
            'changed': len(changed_params) > 0,
            'num_changed': len(changed_params),
            'total_params': len(changes),
            'total_change': total_change,
            'max_change': max_change,
            'changed_params': changed_params[:5],  # First 5 for brevity
        }

        if result['changed']:
            logger.info(f"\n{'='*80}")
            logger.info(f"DEBUGGER: Parameter Changes After Epoch {epoch}")
            logger.info(f"{'='*80}")
            logger.info(f"Changed parameters: {result['num_changed']}/{result['total_params']}")
            logger.info(f"Total change (L2): {result['total_change']:.6f}")
            logger.info(f"Max change: {result['max_change']:.6f}")
            logger.info(f"Changed params (sample): {result['changed_params']}")
        else:
            logger.warning(f"\n{'='*80}")
            logger.warning(f"DEBUGGER: NO PARAMETER CHANGES DETECTED AFTER EPOCH {epoch}")
            logger.warning(f"{'='*80}")

        return result

    def print_epoch_summary(self, epoch: int) -> None:
        """Print comprehensive summary for an epoch."""
        stats = self.epoch_stats[epoch]

        logger.info(f"\n{'='*80}")
        logger.info(f"DEBUGGER: Epoch {epoch} Summary")
        logger.info(f"{'='*80}")
        logger.info(f"Batches processed: {len(stats['batch_indices'])}")
        logger.info(f"Batch indices: {stats['batch_indices']}")
        logger.info(f"Backward calls: {stats['backward_calls']}")
        logger.info(f"Gradient clip calls: {stats['clip_calls']}")
        logger.info(f"Optimizer steps: {stats['optimizer_steps']}")

        if stats['gradient_norms']:
            logger.info(f"Gradient norms: min={min(stats['gradient_norms']):.2f}, "
                       f"max={max(stats['gradient_norms']):.2f}, "
                       f"mean={sum(stats['gradient_norms'])/len(stats['gradient_norms']):.2f}")

        logger.info(f"\nAccumulation checks:")
        for check in stats['accumulation_triggered']:
            logger.info(
                f"  Batch {check['batch_idx']}: {check['condition']} = {check['result']} "
                f"({'✓ TRIGGERED' if check['triggered'] else '✗ SKIPPED'})"
            )

        # Check parameter changes
        self.check_parameter_changes(epoch)

        logger.info(f"{'='*80}\n")

    def print_final_report(self) -> None:
        """Print final comprehensive report."""
        logger.info(f"\n{'#'*80}")
        logger.info(f"DEBUGGER: FINAL TRAINING FLOW REPORT")
        logger.info(f"{'#'*80}\n")

        total_epochs = len(self.epoch_stats)
        total_batches = sum(len(stats['batch_indices']) for stats in self.epoch_stats.values())
        total_optimizer_steps = sum(stats['optimizer_steps'] for stats in self.epoch_stats.values())
        total_backward_calls = sum(stats['backward_calls'] for stats in self.epoch_stats.values())
        total_clip_calls = sum(stats['clip_calls'] for stats in self.epoch_stats.values())

        logger.info(f"Training Statistics:")
        logger.info(f"  Total epochs: {total_epochs}")
        logger.info(f"  Total batches: {total_batches}")
        logger.info(f"  Total backward calls: {total_backward_calls}")
        logger.info(f"  Total gradient clip calls: {total_clip_calls}")
        logger.info(f"  Total optimizer steps: {total_optimizer_steps}")
        logger.info(f"  Batches per epoch: {total_batches / total_epochs if total_epochs > 0 else 0:.1f}")
        logger.info(f"  Optimizer steps per epoch: {total_optimizer_steps / total_epochs if total_epochs > 0 else 0:.1f}")

        # Critical findings
        logger.info(f"\nCritical Findings:")

        if total_optimizer_steps == 0:
            logger.error(f"  ❌ CRITICAL: Optimizer NEVER stepped - model did not learn!")
        elif total_optimizer_steps < total_epochs:
            logger.warning(f"  ⚠️  WARNING: Optimizer steps ({total_optimizer_steps}) < epochs ({total_epochs})")
        else:
            logger.info(f"  ✓ Optimizer stepped at least once per epoch")

        if total_clip_calls == 0:
            logger.error(f"  ❌ CRITICAL: Gradient clipping NEVER occurred!")
        elif total_clip_calls < total_epochs:
            logger.warning(f"  ⚠️  WARNING: Gradient clipping ({total_clip_calls}) < epochs ({total_epochs})")
        else:
            logger.info(f"  ✓ Gradient clipping occurred at least once per epoch")

        if total_clip_calls != total_optimizer_steps:
            logger.warning(
                f"  ⚠️  WARNING: Clip calls ({total_clip_calls}) != optimizer steps ({total_optimizer_steps})"
            )

        # Per-epoch breakdown
        logger.info(f"\nPer-Epoch Breakdown:")
        for epoch in sorted(self.epoch_stats.keys()):
            stats = self.epoch_stats[epoch]
            logger.info(
                f"  Epoch {epoch}: {len(stats['batch_indices'])} batches, "
                f"{stats['backward_calls']} backwards, "
                f"{stats['clip_calls']} clips, "
                f"{stats['optimizer_steps']} steps"
            )

        logger.info(f"\n{'#'*80}\n")


def instrument_trainer(trainer: Any, debugger: TrainingFlowDebugger) -> None:
    """
    Instrument an LSTMTrainer instance with debugging hooks.

    Args:
        trainer: LSTMTrainer instance
        debugger: TrainingFlowDebugger instance
    """
    # Store original methods
    original_backward_mixed = trainer._backward_pass_mixed_precision
    original_backward_std = trainer._backward_pass_standard
    original_safe_training_step = trainer._safe_training_step

    # Wrap backward pass (mixed precision)
    def instrumented_backward_mixed(loss: torch.Tensor, batch_idx: int) -> None:
        debugger.log_backward_call(batch_idx, loss.item())

        # Store original clip_grad_norm_
        original_clip = torch.nn.utils.clip_grad_norm_

        def instrumented_clip(parameters, max_norm, *args, **kwargs):
            grad_norm = original_clip(parameters, max_norm, *args, **kwargs)
            debugger.log_gradient_clip(batch_idx, grad_norm, max_norm)
            return grad_norm

        # Temporarily replace clip function
        torch.nn.utils.clip_grad_norm_ = instrumented_clip

        # Check accumulation condition
        accumulation_steps = trainer.config.gradient_accumulation_steps
        condition_met = (batch_idx + 1) % accumulation_steps == 0
        debugger.log_accumulation_check(batch_idx, accumulation_steps, condition_met)

        # Call original
        result = original_backward_mixed(loss, batch_idx)

        # Restore original
        torch.nn.utils.clip_grad_norm_ = original_clip

        # Check if optimizer stepped
        if condition_met:
            debugger.log_optimizer_step(batch_idx)

        return result

    # Wrap backward pass (standard precision)
    def instrumented_backward_std(loss: torch.Tensor, batch_idx: int) -> None:
        debugger.log_backward_call(batch_idx, loss.item())

        # Store original clip_grad_norm_
        original_clip = torch.nn.utils.clip_grad_norm_

        def instrumented_clip(parameters, max_norm, *args, **kwargs):
            grad_norm = original_clip(parameters, max_norm, *args, **kwargs)
            debugger.log_gradient_clip(batch_idx, grad_norm, max_norm)
            return grad_norm

        # Temporarily replace clip function
        torch.nn.utils.clip_grad_norm_ = instrumented_clip

        # Check accumulation condition
        accumulation_steps = trainer.config.gradient_accumulation_steps
        condition_met = (batch_idx + 1) % accumulation_steps == 0
        debugger.log_accumulation_check(batch_idx, accumulation_steps, condition_met)

        # Call original
        result = original_backward_std(loss, batch_idx)

        # Restore original
        torch.nn.utils.clip_grad_norm_ = original_clip

        # Check if optimizer stepped
        if condition_met:
            debugger.log_optimizer_step(batch_idx)

        return result

    # Wrap safe_training_step to track batches
    def instrumented_safe_training_step(
        sequences: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor,
        batch_idx: int
    ) -> float:
        debugger.log_batch_start(batch_idx)
        return original_safe_training_step(sequences, targets, lengths, batch_idx)

    # Replace methods
    trainer._backward_pass_mixed_precision = instrumented_backward_mixed
    trainer._backward_pass_standard = instrumented_backward_std
    trainer._safe_training_step = instrumented_safe_training_step

    logger.info("✓ Trainer instrumented with debugging hooks")
