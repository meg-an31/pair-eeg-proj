"""Signal processing seam.  ***NOT IMPLEMENTED — DELIBERATELY BLANK.***

This is where raw samples become spectra and band powers. The interface is
fixed so the rest of the system can be built and tested; the implementation
is a null object that returns correctly shaped zeros.

To fill it in, write a class satisfying `Processor` and hand it to the
Session. Nothing else changes.

Two inputs arrive on every call:

  * `epoch`   — the newest window of samples (see `EpochWindow`)
  * `resting` — two minutes of the wearer staring at a wall, raw, or None

The resting block is the normalisation reference. Absolute band powers are
close to meaningless across sessions — they shift with skull, hair, sweat and
how the band sat that morning — so almost every useful number here is a ratio
against, or a distance from, that recording.

Target shapes, agreed with the front end:

    spectrum : (n_channels, 128) float32   0-127 Hz at 1 Hz spacing
    bands    : (n_channels, 5)   float32   delta theta alpha beta gamma

Bin spacing follows from the hardware: the Muse samples at 256 Hz and a
256-sample Welch segment gives 1 Hz spacing. Running Welch with nperseg=256
across a longer window keeps that spacing while averaging several segments,
which cuts variance without changing resolution.

Mind the bin count. `scipy.signal.welch(fs=256, nperseg=256)` returns **129**
bins covering 0-128 Hz inclusive. We ship 128, covering 0-127 Hz — Nyquist
itself is dropped. The implementation must discard that last bin explicitly;
`check_shapes` below will catch it if you forget.

Two things the real implementation must get right, both of which the existing
offline pipeline does not:

  * Filtering must be causal (`sosfilt` with retained state), not zero-phase
    `filtfilt`, which needs the whole recording and cannot stream.
  * Band integration must be defined once. The current repo has two
    implementations that disagree (half-open trapezoid vs inclusive
    rectangular sum) and two different relative-power denominators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import StreamSpec
from .resting import RestingBaseline

# Canonical band edges, half-open [lo, hi). Matches the existing repo.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_NAMES: tuple[str, ...] = tuple(BANDS)

N_BINS = 128
BIN_HZ = 1.0
FREQS: np.ndarray = np.arange(N_BINS, dtype=np.float32) * BIN_HZ


@dataclass(frozen=True)
class EpochWindow:
    """One epoch, post-quality-gate, handed to the processor.

    `eeg` is (n_samples, n_channels) in microvolts, NaN where samples were
    never received. `imu` and `ppg` are the co-timed windows where available.
    """

    t: float
    counter: int
    fs: float
    channels: tuple[str, ...]
    eeg: np.ndarray
    ppg: np.ndarray | None = None
    imu: np.ndarray | None = None

    @property
    def n_channels(self) -> int:
        return len(self.channels)


@dataclass(frozen=True)
class ProcessedFeatures:
    """Output of the processing stage.

    `extras` carries whatever a real implementation wants to pass through to
    the affect stage (FAA, corrugator envelope, blink rate, HRV, coherence)
    without changing this dataclass every time.
    """

    spectrum: np.ndarray            # (n_channels, N_BINS)
    bands: np.ndarray               # (n_channels, len(BAND_NAMES))
    channels: tuple[str, ...]
    freqs: np.ndarray = field(default_factory=lambda: FREQS.copy())
    band_names: tuple[str, ...] = BAND_NAMES
    extras: dict[str, float] = field(default_factory=dict)
    implemented: bool = True

    def band(self, channel: str, band: str) -> float:
        return float(self.bands[self.channels.index(channel), BAND_NAMES.index(band)])

    def check_shapes(self) -> None:
        """Fail loudly on a processor that returns the wrong shape.

        The commonest mistake is shipping scipy's natural 129 bins instead of
        128. Unchecked, that reaches the front end as an off-by-one frequency
        axis, which looks like a calibration problem rather than a bug.
        """
        n_ch = len(self.channels)
        if self.spectrum.shape != (n_ch, N_BINS):
            raise ValueError(
                f"spectrum must be {(n_ch, N_BINS)}, got {self.spectrum.shape}"
                + (" — drop the Nyquist bin" if self.spectrum.shape[-1] == N_BINS + 1 else "")
            )
        if self.bands.shape != (n_ch, len(BAND_NAMES)):
            raise ValueError(
                f"bands must be {(n_ch, len(BAND_NAMES))}, got {self.bands.shape}"
            )


@runtime_checkable
class Processor(Protocol):
    """Raw epoch -> spectra and band powers.

    `resting` is two minutes of the wearer staring at a wall, as raw samples.
    It is None only when no resting block has been recorded or restored for
    this wearer — a processor that needs it should say so rather than silently
    returning uncalibrated numbers.

    It is passed on every call rather than at construction because it can
    arrive mid-session: a wearer may go live uncalibrated and record the
    resting block afterwards.
    """

    name: str

    def process(
        self, epoch: EpochWindow, resting: RestingBaseline | None = None
    ) -> ProcessedFeatures: ...


class NullProcessor:
    """Placeholder. Returns correct shapes, all zeros, `implemented=False`.

    Present so the transport, session lifecycle and payload can be exercised
    end to end before any DSP exists. It does no filtering, no FFT and no
    band integration — it is not a degraded processor, it is an absent one.
    """

    name = "null_v0"

    def process(
        self, epoch: EpochWindow, resting: RestingBaseline | None = None
    ) -> ProcessedFeatures:
        n_ch = epoch.n_channels
        return ProcessedFeatures(
            spectrum=np.zeros((n_ch, N_BINS), dtype=np.float32),
            bands=np.zeros((n_ch, len(BAND_NAMES)), dtype=np.float32),
            channels=epoch.channels,
            implemented=False,
        )
