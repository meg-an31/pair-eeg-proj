"""Affect seam.  ***NOT IMPLEMENTED — DELIBERATELY BLANK.***

Turns processed features into the 0-1 values the front end animates.

Everything leaving here is bounded [0, 1] by contract, so the front end never
has to know a band power from a z-score. The convention is:

    0.5  = the wearer's own resting baseline
    > 0.5 = above their baseline
    < 0.5 = below it

which means these numbers are only meaningful once a baseline exists. Before
that the session reports `calibrated: false` and the front end should treat
the values as a liveness indicator, not a reading.

Why several axes rather than one emotion word: valence and arousal alone
cannot separate anger from fear — both are negative and high-arousal. What
separates them is motivational direction (approach vs withdrawal), which is
a third quantity. Collapsing to a label on the server throws that away and
makes the guess unauditable. So the server ships coordinates; the front end
renders them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .processing import ProcessedFeatures

# The axes the front end animates. All values are 0-1.
AXES: tuple[str, ...] = (
    "valence",     # unpleasant -> pleasant        (corrugator EMG)
    "arousal",     # calm -> activated             (beta/alpha)
    "direction",   # withdrawal -> approach        (frontal alpha asymmetry)
    "engagement",  # disengaged -> absorbed        (beta/(alpha+theta))
    "autonomic",   # slow-lane arousal             (HRV, 60 s window)
)

NEUTRAL = 0.5


@dataclass(frozen=True)
class AffectValues:
    """0-1 values plus honesty about where they came from.

    `confidence` is per-axis and also 0-1. `calibrated` is false until a
    baseline has been frozen; `implemented` is false while the null mapper
    is in place.
    """

    axes: dict[str, float]
    confidence: dict[str, float] = field(default_factory=dict)
    calibrated: bool = False
    implemented: bool = True
    source: str = ""

    def __post_init__(self) -> None:
        for name, value in self.axes.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"axis {name!r} out of range: {value}")


@runtime_checkable
class AffectMapper(Protocol):
    """Processed features -> 0-1 axes."""

    name: str

    def map(self, features: ProcessedFeatures, calibrated: bool) -> AffectValues: ...


class NullAffectMapper:
    """Placeholder. Every axis sits at neutral with zero confidence.

    Deliberately does not fake movement. A mapper that returned drifting
    values would look like it worked, and the front end would be animating
    noise. Flat 0.5 with `implemented=False` is unambiguous.
    """

    name = "null_v0"

    def map(self, features: ProcessedFeatures, calibrated: bool) -> AffectValues:
        return AffectValues(
            axes={axis: NEUTRAL for axis in AXES},
            confidence={axis: 0.0 for axis in AXES},
            calibrated=calibrated,
            implemented=False,
            source=self.name,
        )


class Smoother:
    """Exponential smoothing over the axes.

    Per-window estimates are jittery. This costs roughly
    `smoothing_windows * hop` of extra lag, which is why the unsmoothed
    values are shipped alongside rather than discarded.
    """

    def __init__(self, alpha: float):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: dict[str, float] = {}

    def update(self, axes: dict[str, float]) -> dict[str, float]:
        for name, value in axes.items():
            prev = self._state.get(name)
            self._state[name] = value if prev is None else self.alpha * value + (1 - self.alpha) * prev
        return dict(self._state)

    def reset(self) -> None:
        self._state.clear()
