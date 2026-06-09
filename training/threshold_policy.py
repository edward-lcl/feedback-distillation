"""
Adaptive EMA-based threshold policy for gating knowledge distillation steps.

The policy tracks multi-metric loss vectors and decides whether the student
has improved sufficiently to skip (or stop) further distillation.
"""

from dataclasses import dataclass
from collections import deque
import hashlib
import numpy as np
import torch


def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().numpy().mean())
    return float(x)


def recalibrate_baselines(baseline_losses, current_losses, minimum_relative_gain: float, verbose=True):
    baseline_losses = [_to_float(v) for v in baseline_losses]
    current_losses = [_to_float(v) for v in current_losses]
    new_baselines = []
    for i, (pb, cl) in enumerate(zip(baseline_losses, current_losses)):
        min_allowed = cl * (1 - minimum_relative_gain / 100.0)
        if pb > cl and pb <= (1 + minimum_relative_gain / 100.0) * cl:
            status, new_val = "KEEP", pb
        elif min_allowed <= pb < cl:
            status, new_val = "KEEP", pb
        else:
            status = f"UPDATE (pb={pb:.4f}, cl={cl:.4f} -> min={min_allowed:.4f})"
            new_val = min_allowed
        new_baselines.append(new_val)
        if verbose:
            print(f"  Loss {i}: PrevBaseline={pb:.4f}, CurrLoss={cl:.4f}, MinAllowed={min_allowed:.4f} -> {new_val:.4f} [{status}]")
    return new_baselines


@dataclass
class ThresholdConfig:
    strategy: str = "relative"
    ema_alpha: float = 0.1
    minimum_relative_gain: float = 10.0
    relative_gain_stop_threshold: float = 90.0


class AdaptiveLossWeighting:
    def __init__(self, num_losses: int, beta: float = 1.0, device: str = "cuda"):
        self.num_losses = num_losses
        self.beta = beta
        self.previous_losses = None
        self.device = device

    def compute_weights(self, current_losses, in_kd: bool = False):
        curr = torch.tensor(current_losses, device=self.device, dtype=torch.float32)
        if self.previous_losses is None:
            weights = torch.ones_like(curr) / self.num_losses
        else:
            deltas = curr - self.previous_losses
            weights = torch.softmax(self.beta * deltas, dim=0)
        if not in_kd:
            self.previous_losses = curr.detach()
        return weights


class AdaptiveWeightedKDPolicyEMA:
    """
    EMA-smoothed adaptive weighting + threshold gating for the KD loop.

    At each batch:
    - Recomputes baseline losses with EMA smoothing.
    - Computes per-loss relative improvements.
    - Assigns higher weight to lagging losses (adaptive weighting).
    - Returns (weights, skip_kd, freeze_student).
    """

    def __init__(
        self,
        num_losses: int,
        initial_losses: list | None = None,
        device: str = "cuda",
        config: ThresholdConfig | None = None,
    ):
        self.config = config or ThresholdConfig()
        self.device = device
        self.num_losses = num_losses
        self.baseline_losses = initial_losses[:] if initial_losses else None
        self.ema_losses = initial_losses[:] if initial_losses else None
        self.adaptive_weighting = AdaptiveLossWeighting(num_losses, device=device)
        self.student_performance: list[dict] = []
        self._last_hash: str | None = None

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def should_skip_kd(self, loss_vector, kd: bool = True, update_baseline: bool = True):
        losses = [_to_float(v) for v in loss_vector]
        weights = self.adaptive_weighting.compute_weights(losses, in_kd=kd).tolist()

        if self.baseline_losses is None:
            self.baseline_losses = losses[:]
            return False, {"adaptive_weights": weights, "relative_gains": [0.0] * self.num_losses}

        relative_gains = [
            max(0.0, (b - c) / (b + 1e-8) * 100)
            for b, c in zip(self.baseline_losses, losses)
        ]
        skip = all(g >= self.config.relative_gain_stop_threshold for g in relative_gains)

        if update_baseline and not skip:
            self.baseline_losses = recalibrate_baselines(
                self.baseline_losses, losses, self.config.minimum_relative_gain, verbose=False
            )

        return skip, {"adaptive_weights": weights, "relative_gains": relative_gains}

    def update_weights(self, current_losses, last_qna_in_batch: str = "", in_KD: bool = True):
        losses = [_to_float(v) for v in current_losses]

        # EMA update
        if self.ema_losses is None:
            self.ema_losses = losses[:]
        else:
            alpha = self.config.ema_alpha
            self.ema_losses = [alpha * c + (1 - alpha) * e for c, e in zip(losses, self.ema_losses)]

        new_hash = self._hash(last_qna_in_batch)
        if self._last_hash is None:
            print("Hash val is initialized")
        self._last_hash = new_hash

        # Baseline & skip logic
        skip_kd = False
        freeze_student = False
        if self.baseline_losses is None:
            self.baseline_losses = self.ema_losses[:]
            weights = [1.0 / self.num_losses] * self.num_losses
        else:
            updated = recalibrate_baselines(self.baseline_losses, self.ema_losses, self.config.minimum_relative_gain)
            relative_gains = [
                max(0.0, (b - c) / (b + 1e-8) * 100)
                for b, c in zip(self.baseline_losses, self.ema_losses)
            ]
            skip_kd = all(g >= self.config.relative_gain_stop_threshold for g in relative_gains)
            freeze_student = skip_kd

            # Adaptive: upweight lagging losses
            max_gain = max(relative_gains) if relative_gains else 1.0
            lag = [max(0.0, max_gain - g) + 1e-6 for g in relative_gains]
            total = sum(lag)
            weights = [l / total for l in lag]
            self.baseline_losses = updated

        self.student_performance.append({"ema_losses": self.ema_losses[:], "weights": weights})
        return weights, skip_kd, freeze_student
