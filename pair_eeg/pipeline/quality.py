"""Epoch quality gating.

Runs before any feature work, because there is no point spending DSP on an
epoch that is mostly jaw clench. Dry electrodes normally lose 20-40% of
epochs; that is the expected operating point, not a fault condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ringbuffer import Window

# Per-channel verdicts, worst first.
FLAT = "flat"
NOISY = "noisy"
OK = "ok"


@dataclass(frozen=True)
class QualityVerdict:
    accepted: bool
    channels: dict[str, str]
    fill_ratio: float
    reason: str = ""
    rms_uv: dict[str, float] = field(default_factory=dict)

    @property
    def n_ok(self) -> int:
        return sum(1 for v in self.channels.values() if v == OK)


class QualityGate:
    """Amplitude and contact gating on a single epoch.

    Thresholds follow the existing capture client's RMS bands rather than
    inventing new ones. Motion gating from the IMU is a separate concern and
    is applied by the caller, which is the only place that has the IMU window
    to hand.
    """

    def __init__(
        self,
        channels: tuple[str, ...],
        flat_below_uv: float = 2.0,
        noisy_above_uv: float = 80.0,
        reject_peak_uv: float = 150.0,
        min_ok_channels: int = 2,
        min_fill: float = 0.9,
    ):
        self.channels = channels
        self.flat_below_uv = flat_below_uv
        self.noisy_above_uv = noisy_above_uv
        self.reject_peak_uv = reject_peak_uv
        self.min_ok_channels = min_ok_channels
        self.min_fill = min_fill

    def check(self, window: Window) -> QualityVerdict:
        fill = window.fill_ratio
        samples = window.samples

        if fill < self.min_fill:
            return QualityVerdict(
                accepted=False,
                channels={c: FLAT for c in self.channels},
                fill_ratio=fill,
                reason=f"incomplete window ({fill:.0%} filled)",
            )

        verdicts: dict[str, str] = {}
        rms: dict[str, float] = {}

        for i, name in enumerate(self.channels):
            col = samples[:, i]
            col = col[~np.isnan(col)]
            if col.size == 0:
                verdicts[name] = FLAT
                rms[name] = 0.0
                continue

            centred = col - col.mean()
            r = float(np.sqrt(np.mean(centred**2)))
            rms[name] = r

            if r < self.flat_below_uv:
                verdicts[name] = FLAT
            elif r > self.noisy_above_uv:
                verdicts[name] = NOISY
            elif float(np.max(np.abs(centred))) > self.reject_peak_uv:
                # A transient the RMS did not catch — usually a blink.
                verdicts[name] = NOISY
            else:
                verdicts[name] = OK

        n_ok = sum(1 for v in verdicts.values() if v == OK)
        accepted = n_ok >= self.min_ok_channels
        reason = "" if accepted else f"only {n_ok}/{len(self.channels)} channels usable"

        return QualityVerdict(
            accepted=accepted,
            channels=verdicts,
            fill_ratio=fill,
            reason=reason,
            rms_uv=rms,
        )


class AcceptRate:
    """Trailing accept-rate tracker, used to decide the degraded state."""

    def __init__(self, window_s: float, hop_s: float):
        self.capacity = max(1, int(round(window_s / hop_s)))
        self._history: list[bool] = []

    def record(self, accepted: bool) -> None:
        self._history.append(accepted)
        if len(self._history) > self.capacity:
            del self._history[: len(self._history) - self.capacity]

    @property
    def rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    @property
    def n(self) -> int:
        return len(self._history)
