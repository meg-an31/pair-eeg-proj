"""Affect seam.  ***NOT IMPLEMENTED — DELIBERATELY BLANK.***

Turns processed features into the 0-1 values the front end animates.

Everything leaving here is bounded [0, 1] by contract, so the front end never
has to know a band power from a z-score. The convention is:

    0.5  = the wearer's own resting baseline
    > 0.5 = above their baseline
    < 0.5 = below it

which means these numbers are only meaningful once a resting block exists.
Before that the session reports `calibrated: false` and the front end should
treat the values as a liveness indicator, not a reading.

The resting block — two minutes of staring at a wall — is passed in on every
call. It is what 0.5 is defined against.

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
from .resting import RestingBaseline

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
        for name, value in self.confidence.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"confidence {name!r} out of range: {value}")


@runtime_checkable
class AffectMapper(Protocol):
    """Processed features -> 0-1 axes.

    `resting` is the same wall-staring block the processor received. It is
    passed here too because the mapping from a feature to a 0-1 position is
    itself relative: 0.5 means the wearer's own resting level, and the spread
    either side has to come from the distribution of that recording rather
    than from a constant somebody picked.
    """

    name: str

    def map(
        self,
        features: ProcessedFeatures,
        calibrated: bool,
        resting: RestingBaseline | None = None,
    ) -> AffectValues: ...


class NullAffectMapper:
    """Placeholder. Every axis sits at neutral with zero confidence.

    Deliberately does not fake movement. A mapper that returned drifting
    values would look like it worked, and the front end would be animating
    noise. Flat 0.5 with `implemented=False` is unambiguous.
    """

    name = "null_v0"

    def map(
        self,
        features: ProcessedFeatures,
        calibrated: bool,
        resting: RestingBaseline | None = None,
    ) -> AffectValues:
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
        """Smooth and return exactly the axes passed in.

        Returning the whole retained state would leak axes a later mapper
        stopped producing, so `axes` and `axes_raw` could disagree on which
        keys exist.
        """
        out: dict[str, float] = {}
        for name, value in axes.items():
            prev = self._state.get(name)
            smoothed = value if prev is None else self.alpha * value + (1 - self.alpha) * prev
            self._state[name] = smoothed
            out[name] = smoothed
        return out

    def reset(self) -> None:
        self._state.clear()


# --------------------------------------------------------------------------
# The real affect stage: the ~/projects/muse scorer, unaltered
# --------------------------------------------------------------------------
#
# muse's `score.score_epoch` is used as-is. It takes the epoch's per-window
# feature rows plus a resting baseline and returns valence, arousal and
# tension, each in -1..+1, with a per-axis confidence. This class does three
# things around it, and no scoring of its own.
#
# **1. It builds the baseline from the resting block.** `RestingBaseline`
# carries the raw resting samples, which is exactly what muse's
# `compute_baseline` wants — it is handed a `preprocess_eeg` output. Derived
# once and cached, because it re-runs the whole preprocessing chain over two
# minutes of samples and could not do that every hop.
#
# **2. It converts -1..+1 into the 0-1 axis contract.** `(x + 1) / 2`, so
# muse's 0 — the wearer's own resting level — lands on 0.5, which is what this
# module's contract says 0.5 means. The two conventions agree exactly; only
# the arithmetic differs.
#
# **3. It unfolds two axes muse already computes but keeps folded in.**
# `direction` is frontal alpha asymmetry and `engagement` is beta/(alpha+theta):
# muse computes both per window and then feeds them into valence and arousal as
# voters. The front end wants them separately — valence and arousal alone
# cannot separate anger from fear, and motivational direction is what does — so
# they are read from the same sign-aligned z-scores the scorer produced, which
# keeps them consistent with the axes they also contribute to rather than
# recomputed by a second route.
#
# `autonomic` is muse's heart and breathing evidence. Nothing supplies those
# streams yet (see MuseProcessor._autonomic), so it reports NEUTRAL with zero
# confidence rather than a value — Invariant 6.

import numpy as np                                            # noqa: E402

from .muse.baseline import compute_baseline                    # noqa: E402
from .muse.preprocess import preprocess_eeg                    # noqa: E402
from .muse.score import score_epoch                            # noqa: E402

# Which sign-aligned z-score each unfolded axis is read from, and the scale
# that turns it into a 0-1 position. tanh(z/2) is muse's own squash — the same
# one score_epoch applies to a combined axis — so a z of 2 sits at the same
# distance from centre here as it would inside valence or arousal.
UNFOLDED = {"direction": "z_faa", "engagement": "z_engagement"}

# Confidence for an unfolded single-index axis. Lower than a combined axis can
# reach, because muse's confidence is *agreement between indices* and a lone
# index has nothing to agree with — score_epoch says the same thing when it
# assigns a single-index axis a mediocre spread of 1.0.
SINGLE_INDEX_CONFIDENCE = 0.5


def to_unit(value: float) -> float:
    """muse's -1..+1 onto this module's 0-1, with 0 -> 0.5 (= resting).

    Clipped rather than trusted: the axis contract is enforced by
    AffectValues, and a tanh that returns exactly +-1.0 in float arithmetic
    would otherwise sit precisely on the boundary.
    """
    if value is None or not np.isfinite(value):
        return NEUTRAL
    return float(np.clip((float(value) + 1.0) / 2.0, 0.0, 1.0))


class MuseAffectMapper:
    """muse's valence/arousal scorer behind the five-axis contract.

    Constructed with the `MuseProcessor` it runs behind, because muse scores
    from per-window feature rows and `ProcessedFeatures.extras` is a flat dict
    of floats that cannot carry them. The session calls `process()` then
    `map()` on the same tick, so the rows are always the ones belonging to the
    features passed in.
    """

    name = "muse_v1"

    def __init__(self, processor):
        self.processor = processor
        self._cache_key: tuple | None = None
        self._baseline: dict | None = None

    def map(
        self,
        features: ProcessedFeatures,
        calibrated: bool,
        resting: RestingBaseline | None = None,
    ) -> AffectValues:
        baseline = self._resting_baseline(resting) if calibrated else None
        if baseline is None:
            # Every muse index is a z-score against the resting distribution.
            # With no baseline there is no scale, and a moving needle would
            # read as a working one.
            return self._flat(calibrated, "no resting baseline")

        rows = self.processor.rows
        if not rows or not any(r["good"] for r in rows):
            return self._flat(calibrated, "no clean windows")

        score = score_epoch(rows, self.processor.autonomic_row, baseline)
        return AffectValues(
            axes=self._axes(score),
            confidence=self._confidence(score),
            calibrated=True,
            implemented=True,
            source=self.name,
        )

    # ------------------------------------------------------------- internals

    def _axes(self, score: dict) -> dict[str, float]:
        axes = {
            "valence": to_unit(score["valence"]),
            "arousal": to_unit(score["arousal"]),
            "autonomic": NEUTRAL,
            "direction": NEUTRAL,
            "engagement": NEUTRAL,
        }
        for axis, key in UNFOLDED.items():
            z = score.get(key)
            if z is not None and np.isfinite(z):
                axes[axis] = to_unit(np.tanh(float(z) / 2.0))
        return axes

    def _confidence(self, score: dict) -> dict[str, float]:
        combined = {
            "valence": float(score["confidence_valence"]),
            "arousal": float(score["confidence_arousal"]),
        }
        out = {axis: 0.0 for axis in AXES}
        for axis, value in combined.items():
            out[axis] = float(np.clip(value, 0.0, 1.0))
        for axis, key in UNFOLDED.items():
            z = score.get(key)
            if z is not None and np.isfinite(z):
                out[axis] = SINGLE_INDEX_CONFIDENCE
        # autonomic stays 0.0: nothing supplies heart or breathing yet.
        return out

    def _flat(self, calibrated: bool, why: str) -> AffectValues:
        return AffectValues(
            axes={axis: NEUTRAL for axis in AXES},
            confidence={axis: 0.0 for axis in AXES},
            calibrated=calibrated,
            implemented=True,
            source=f"{self.name} ({why})",
        )

    def _resting_baseline(self, resting: RestingBaseline | None) -> dict | None:
        """muse's baseline.json equivalent, derived from the resting samples.

        Cached on the recording's identity: this re-runs filtering and feature
        extraction over the whole resting block, far too slow to repeat per
        hop, and the key includes the timestamp so a mid-session re-baseline
        is picked up rather than ignored.
        """
        if resting is None:
            return None
        key = (resting.wearer, resting.recorded_at, resting.n_samples)
        if key != self._cache_key:
            pre = preprocess_eeg(resting.eeg.T, fs=resting.fs)
            if not any(pre["good"]):
                return None
            # No rr_ms or breathing: an EEG-only rest recording gives an
            # EEG-only baseline, and the scorer skips the indices it has no
            # statistics for.
            self._baseline = compute_baseline(pre)
            self._cache_key = key
        return self._baseline
