from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _subsample(x: torch.Tensor, n: int) -> torch.Tensor:
    if len(x) <= n:
        return x
    # Evenly spaced selection is deterministic and does not consume the run RNG.
    indices = torch.linspace(0, len(x) - 1, n, device=x.device).long()
    return x[indices]


def multiscale_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    max_samples: int = 256,
    scales: tuple[float, ...] | list[float] = (0.5, 1.0, 2.0),
) -> float:
    """Biased, non-negative multi-kernel RBF MMD estimate."""
    x = _subsample(x.detach().float(), max_samples)
    y = _subsample(y.detach().float(), max_samples)
    joined = torch.cat([x, y], dim=0)
    distances = torch.cdist(joined, joined).square()
    positive = distances[distances > 0]
    bandwidth = torch.median(positive) if positive.numel() else torch.tensor(1.0, device=x.device)
    bandwidth = bandwidth.clamp_min(1e-8)
    total = torch.zeros_like(distances)
    for scale in scales:
        total += torch.exp(-distances / (2.0 * bandwidth * float(scale) ** 2))
    kernel = total / max(1, len(scales))
    n = len(x)
    k_xx = kernel[:n, :n].mean()
    k_yy = kernel[n:, n:].mean()
    k_xy = kernel[:n, n:].mean()
    return float(torch.clamp(k_xx + k_yy - 2.0 * k_xy, min=0.0).item())


@dataclass
class DriftDecision:
    distances: dict[str, float]
    z_scores: dict[str, float]
    requirements: dict[str, float]
    severity: float
    triggered: bool
    emergency: bool = False


class LayerDriftMonitor:
    def __init__(
        self,
        layer_names: list[str],
        kappa: float,
        z_max: float,
        threshold: float,
        persistence: int,
        top_k: int,
        max_samples: int,
        kernel_scales: list[float],
        emergency_threshold: float | None = None,
    ):
        self.layer_names = layer_names
        self.kappa = kappa
        self.z_max = z_max
        self.threshold = threshold
        self.persistence = persistence
        self.top_k = top_k
        self.max_samples = max_samples
        self.kernel_scales = kernel_scales
        self.emergency_threshold = emergency_threshold
        self.medians: dict[str, float] = {}
        self.mads: dict[str, float] = {}
        self._consecutive = 0

    def reset_episode(self) -> None:
        self._consecutive = 0

    def calibrate(
        self,
        reference: dict[str, torch.Tensor],
        calibration: list[dict[str, torch.Tensor]],
    ) -> None:
        if not calibration:
            raise ValueError("At least one calibration representation is required")
        for name in self.layer_names:
            values = np.array([
                multiscale_mmd(reference[name], item[name], self.max_samples, self.kernel_scales)
                for item in calibration
            ])
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            self.medians[name] = median
            self.mads[name] = max(mad, 1e-6)

    def evaluate(
        self,
        reference: dict[str, torch.Tensor],
        current: dict[str, torch.Tensor],
    ) -> DriftDecision:
        distances: dict[str, float] = {}
        z_scores: dict[str, float] = {}
        requirements: dict[str, float] = {}
        denominator = max(self.z_max - self.kappa, 1e-8)
        for name in self.layer_names:
            distance = multiscale_mmd(
                reference[name], current[name], self.max_samples, self.kernel_scales
            )
            z = max(0.0, (distance - self.medians[name]) / (1.4826 * self.mads[name] + 1e-8))
            requirement = float(np.clip((z - self.kappa) / denominator, 0.0, 1.0))
            distances[name] = distance
            z_scores[name] = z
            requirements[name] = requirement
        top = sorted(requirements.values(), reverse=True)[: max(1, min(self.top_k, len(requirements)))]
        severity = float(np.mean(top))
        self._consecutive = self._consecutive + 1 if severity > self.threshold else 0
        emergency = bool(
            self.emergency_threshold is not None
            and severity >= self.emergency_threshold
        )
        return DriftDecision(
            distances=distances,
            z_scores=z_scores,
            requirements=requirements,
            severity=severity,
            triggered=emergency or self._consecutive >= self.persistence,
            emergency=emergency,
        )
