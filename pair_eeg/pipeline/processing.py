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
from scipy.signal import welch

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


# --------------------------------------------------------------------------
# The real processing stage: the ~/projects/muse pipeline, unaltered
# --------------------------------------------------------------------------
#
# `pipeline/muse/` is a verbatim copy of that project's `pipeline/` package —
# preprocess, features_eeg, features_autonomic, baseline, score. It is
# vendored as a package rather than pasted into this file because its own
# module boundaries are load-bearing: features_eeg and features_autonomic each
# define `FEATURE_COLUMNS`, `write_features_csv` and `_fmt`, and its
# `features_eeg.BANDS` would collide with the canonical `BANDS` above. Pasting
# them together would have meant renaming, and the point was to copy the code
# without touching it.
#
# What this class is, then, is purely an adapter. It owns three things the
# muse code cannot know about, and no analysis of its own:
#
#   1. orientation — muse works in (channels, samples), an EpochWindow is
#      (samples, channels)
#   2. gaps — a NaN anywhere in an IIR-filtered array propagates across the
#      whole array, so gaps are filled before filtering and every window that
#      touched one is then force-rejected. Nothing interpolated ever reaches a
#      feature; the fill exists only to stop one dropped packet destroying the
#      other 29 seconds.
#   3. the seam's two mandatory outputs, `spectrum` and `bands`, which the
#      muse pipeline does not produce — see `_spectrum_and_bands`.
#
# The per-window feature rows are kept on the instance. The affect stage needs
# them (muse's scorer takes window rows, not summaries — that is how the
# tonic/phasic muscle split and the good-window medians are defined), and
# `extras` is a flat dict of floats that cannot carry them, so
# `MuseAffectMapper` reads them from the processor it is constructed with.
# That couples the two stages, which the architecture otherwise avoids; the
# alternative was altering muse's `score_epoch` signature.

from .muse.features_eeg import (                              # noqa: E402
    BANDS as MUSE_BANDS,
    eeg_features,
    emg_tonic_phasic,
    good_median,
)
from .muse.features_autonomic import epoch_features            # noqa: E402
from .muse.preprocess import (                                # noqa: E402
    AMPLITUDE_LIMIT_UV,
    preprocess_eeg,
    split_bands,
)

# Extra bands the front end's five slots require and muse does not compute.
# No index reads these; they exist so `bands` has the shape the wire format
# promises. theta/alpha/beta come from MUSE_BANDS, so the displayed spectrum
# is divided the same way the axes are actually computed — muse's alpha is
# 8-12 and its beta 12-28, which is NOT what this module's canonical `BANDS`
# says. Deliberate: the axes are calibrated against those edges, and drawing
# different ones would be showing the wearer a band that no number uses.
DISPLAY_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": MUSE_BANDS["theta"],
    "alpha": MUSE_BANDS["alpha"],
    "beta": MUSE_BANDS["beta"],
    "gamma": (30.0, 45.0),
}


def fill_gaps(eeg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate NaN holes. Returns (filled, was_missing).

    Not a licence to analyse invented samples — `MuseProcessor` rejects every
    window that overlaps a filled position. This exists because the muse
    preprocessor is IIR and zero-phase: `sosfiltfilt` on an array containing a
    single NaN returns an array that is entirely NaN, so without this one
    dropped packet would cost the whole 30 s epoch instead of the two seconds
    it actually damaged. Interpolating rather than zero-filling keeps the
    filter from ringing on a step edge at the hole, which would contaminate
    the neighbouring windows that are otherwise fine.
    """
    eeg = np.array(eeg, dtype=float, copy=True)
    missing = ~np.isfinite(eeg)
    if not missing.any():
        return eeg, missing
    index = np.arange(eeg.shape[0])
    for c in range(eeg.shape[1]):
        bad = missing[:, c]
        if not bad.any():
            continue
        if bad.all():
            eeg[:, c] = 0.0          # nothing to interpolate from
            continue
        eeg[bad, c] = np.interp(index[bad], index[~bad], eeg[~bad, c])
    return eeg, missing


class MuseProcessor:
    """Muse affect pipeline, steps 2-4: filtering, windowing, EEG features.

    One call = one epoch. With the streaming config set to muse's native
    geometry (30 s window, 5 s hop) an epoch yields 29 overlapping 2 s windows,
    which is what the scorer expects.
    """

    name = "muse_v1"

    def __init__(self, limit_uv: float = AMPLITUDE_LIMIT_UV):
        self.limit_uv = float(limit_uv)
        # Newest epoch's per-window rows, read by MuseAffectMapper.
        self.rows: list[dict] = []
        self.autonomic_row: dict | None = None
        # Optional heart and breathing history, on the sample clock. Empty
        # unless something feeds them; see add_beats / add_breathing.
        self._rr: list[tuple[float, float]] = []
        self._breath: list[tuple[float, float]] = []
        self._counter0: int | None = None

    # -------------------------------------------------- optional slow streams

    def add_beats(self, intervals_ms) -> None:
        """Append newly detected beat-to-beat intervals, in milliseconds.

        Optional. Nothing in the transport produces these yet — the wire
        format carries raw PPG counts, and beat detection is not part of the
        muse pipeline — so unless a capture integration calls this, the heart
        indices are simply absent from every score and the remaining ones are
        scored without them.

        Beat times are accumulated from the intervals themselves, exactly as
        muse's own realtime path does (`stream.py:add_rr`): the first beat
        lands one interval after the stream started, so the timestamps stay on
        the sample clock without the caller having to supply them.
        """
        last_t = self._rr[-1][0] if self._rr else 0.0
        for interval in np.atleast_1d(np.asarray(intervals_ms, dtype=float)):
            last_t += float(interval) / 1000.0
            self._rr.append((last_t, float(interval)))

    def add_breathing(self, t_s: float, breaths_per_min: float) -> None:
        """Append one breathing-rate reading at sample-clock time `t_s`.

        Optional in the same way, and more so: the Muse has no respiration
        sensor at all, so this exists for a rig that adds one (a chest belt, or
        a rate derived from the PPG envelope). `t_s` must be seconds since the
        first EEG sample of the session — the sample counter is the clock
        (Invariant 1), and mixing in wall time would drift against it.
        """
        self._breath.append((float(t_s), float(breaths_per_min)))

    def process(
        self, epoch: EpochWindow, resting: RestingBaseline | None = None
    ) -> ProcessedFeatures:
        if self._counter0 is None:
            self._counter0 = int(epoch.counter)
        filled, missing = fill_gaps(epoch.eeg)
        pre = preprocess_eeg(filled.T, fs=epoch.fs, limit_uv=self.limit_uv)

        # Force-reject any window that overlapped a gap. muse's own mask
        # already rejects a window holding a NaN (|NaN| < limit is False), but
        # it cannot see holes we have just filled in.
        if missing.any():
            bad_samples = missing.any(axis=1)
            pre["good"] = np.array([
                bool(g and not bad_samples[s:e].any())
                for g, (s, e) in zip(pre["good"], pre["windows"])
            ])

        self.rows = eeg_features(pre)
        self.autonomic_row = self._autonomic(epoch)

        spectrum, bands = self._spectrum_and_bands(pre["cortical"], epoch.fs)
        return ProcessedFeatures(
            spectrum=spectrum,
            bands=bands,
            channels=epoch.channels,
            extras=self._extras(epoch, pre),
        )

    # ---------------------------------------------------------------- extras

    def _extras(self, epoch: EpochWindow, pre: dict) -> dict[str, float]:
        """Flat float summary of the epoch, for the resting baseline and the
        payload. A feature whose windows were all rejected is OMITTED, never
        emitted as NaN: these values are summed into the session baseline and
        persisted, so one NaN would poison every statistic derived from the
        recording, and a zero would be indistinguishable from a measurement.
        """
        good = [r for r in self.rows if r["good"]]
        extras: dict[str, float] = {
            "n_windows": float(len(self.rows)),
            "n_good_windows": float(len(good)),
            "window_s": float(epoch.eeg.shape[0]) / float(epoch.fs),
            "fill": float(np.isfinite(epoch.eeg).mean()),
        }
        if not good:
            return extras
        for index in ("beta_alpha_ratio", "engagement", "faa", "emg_envelope"):
            value = good_median(self.rows, index)
            if np.isfinite(value):
                extras[index] = float(value)
        for key, value in emg_tonic_phasic(self.rows).items():
            if np.isfinite(value):
                extras[key] = float(value)
        if self.autonomic_row and self.autonomic_row.get("good"):
            for index in ("mean_hr_bpm", "rmssd_ms", "breathing_bpm"):
                if index in self.autonomic_row:
                    extras[index] = float(self.autonomic_row[index])
        return extras

    def _autonomic(self, epoch: EpochWindow) -> dict | None:
        """Heart and breathing features for this epoch, from whatever exists.

        Returns None when neither stream has been fed, which is the default
        state — see add_beats / add_breathing. The two are independent: with
        beats and no breathing the score uses heart rate and RMSSD; with
        breathing and no beats it uses the breathing rate; with neither the
        axes are computed from the EEG indices alone.

        Spans are taken on the SAMPLE clock, anchored to the first counter
        this processor saw, because the device counter is the clock and it does
        not start at zero on real hardware. muse's `epoch_features` derives
        beat times from the intervals it is handed, so the selection is
        re-anchored to the epoch before the call, as muse's own realtime path
        does (`stream.py:_score_tick`).
        """
        if not self._rr and not self._breath:
            return None

        window_s = float(epoch.eeg.shape[0]) / float(epoch.fs)
        end_s = (int(epoch.counter) - (self._counter0 or 0)) / float(epoch.fs) + window_s
        start_s = end_s - window_s

        rr = [iv for (bt, iv) in self._rr if start_s < bt <= end_s] or None
        breath = [(bt - start_s, v) for (bt, v) in self._breath
                  if start_s <= bt < end_s]
        b_t = [bt for bt, _ in breath] or None
        b_v = [v for _, v in breath] or None

        self._trim(start_s)
        return epoch_features(rr, b_t, b_v, 0.0, window_s)

    def _trim(self, keep_from_s: float) -> None:
        """Drop history older than the epoch needs, so memory stays flat
        however long the headband is worn."""
        self._rr = [x for x in self._rr if x[0] > keep_from_s]
        self._breath = [x for x in self._breath if x[0] >= keep_from_s]

    # ------------------------------------------------- the seam's two outputs

    def _spectrum_and_bands(self, cortical: np.ndarray, fs: float):
        """The `(4, 128)` spectrum and `(4, 5)` band table the wire format
        requires, which the muse pipeline does not produce.

        It comes closer than it looks. muse's `features_eeg.band_powers`
        already runs `welch(nperseg=int(fs))` on every window and then throws
        the spectrum away after integrating three bands — and at 256 Hz that
        call returns exactly this grid: 129 bins at 1 Hz spacing, 0-128 Hz,
        which is the required 128 bins plus the Nyquist bin that Invariant 5
        says to drop. So this is the same estimate over the whole epoch rather
        than a new kind of analysis, and it is computed here rather than
        harvested from the per-window call only because doing that would have
        meant changing `band_powers`.
        """
        freqs, psd = welch(cortical, fs=fs, nperseg=int(fs), axis=-1)
        spectrum = np.ascontiguousarray(psd[:, :N_BINS], dtype=np.float32)

        bands = np.zeros((psd.shape[0], len(BAND_NAMES)), dtype=np.float32)
        for j, name in enumerate(BAND_NAMES):
            lo, hi = DISPLAY_BANDS[name]
            sel = (freqs >= lo) & (freqs < hi)
            if sel.any():
                # Mean power in the band, matching band_powers' convention
                # rather than the integral, so a number shown next to the
                # spectrum means the same thing as the one inside an index.
                bands[:, j] = psd[:, sel].mean(axis=-1)
        return spectrum, bands
