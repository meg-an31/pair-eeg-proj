"""The muse stages behind the seams: the adapters, and optional slow streams.

The muse pipeline itself is tested by that project's own step demos. What is
tested here is everything the adapters own — orientation, gaps, the two
mandatory wire outputs, and whether the optional heart and breathing streams
are really optional.
"""

import time

import numpy as np
import pytest

from pair_eeg.config import EEG
from pair_eeg.pipeline.affect import MuseAffectMapper, to_unit
from pair_eeg.pipeline.processing import (
    BAND_NAMES,
    DISPLAY_BANDS,
    N_BINS,
    EpochWindow,
    MuseProcessor,
    fill_gaps,
)
from pair_eeg.pipeline.resting import RestingBaseline
from tests.muse_signals import synth

FS = EEG.rate_hz


def epoch(eeg_ch_first, counter=0, ppg=None):
    """muse orientation in, EpochWindow orientation out."""
    return EpochWindow(t=30.0, counter=counter, fs=FS, channels=EEG.channels,
                       eeg=np.ascontiguousarray(eeg_ch_first.T, dtype=np.float32),
                       ppg=ppg)


def resting(dur=120.0, **kw):
    return RestingBaseline(wearer="w", recorded_at=time.time(), fs=FS,
                           channels=EEG.channels,
                           eeg=np.ascontiguousarray(synth(dur, seed=7, **kw).T,
                                                    dtype=np.float32))


# --- the adapter's own responsibilities ------------------------------------

def test_epoch_becomes_the_29_windows_muse_expects():
    proc = MuseProcessor()
    proc.process(epoch(synth(30.0)))
    assert len(proc.rows) == 29
    assert all(r["good"] for r in proc.rows)


def test_orientation_is_transposed_not_reinterpreted():
    """muse works in (channels, samples) and an EpochWindow is (samples,
    channels). Reading one as the other would silently mix the four
    electrodes, and asymmetry is the measure that would notice."""
    proc = MuseProcessor()
    proc.process(epoch(synth(30.0, asym=+0.35)))
    faa = np.median([r["faa"] for r in proc.rows if r["good"]])
    proc.process(epoch(synth(30.0, asym=-0.35)))
    faa_flipped = np.median([r["faa"] for r in proc.rows if r["good"]])
    assert faa > 0.1 and faa_flipped < -0.1


def test_shapes_satisfy_the_seam():
    features = MuseProcessor().process(epoch(synth(30.0)))
    features.check_shapes()
    assert features.spectrum.shape == (4, N_BINS)
    assert features.bands.shape == (4, len(BAND_NAMES))
    assert features.implemented is True


def test_reported_bands_are_the_edges_the_indices_actually_use():
    """muse's alpha is 8-12 and its beta 12-28, not the canonical 8-13/13-30.
    The displayed spectrum has to be divided the way the axes are computed, or
    the front end shows a band no number uses."""
    assert DISPLAY_BANDS["alpha"] == (8.0, 12.0)
    assert DISPLAY_BANDS["beta"] == (12.0, 28.0)


def test_alpha_bump_lands_in_the_alpha_band():
    features = MuseProcessor().process(epoch(synth(30.0, alpha=12.0, beta=0.0)))
    alpha = features.bands[:, BAND_NAMES.index("alpha")]
    for other in ("delta", "theta", "beta", "gamma"):
        assert (alpha > 3 * features.bands[:, BAND_NAMES.index(other)]).all(), other


# --- gaps ------------------------------------------------------------------

def test_a_gap_costs_the_windows_that_touched_it_not_the_epoch():
    """One NaN through a zero-phase IIR filter returns an all-NaN array, so
    without the fill a single dropped packet would cost all 30 seconds."""
    eeg = synth(30.0)
    proc = MuseProcessor()
    features = proc.process(epoch(eeg))
    clean_good = sum(r["good"] for r in proc.rows)

    gappy = eeg.copy()
    gappy[:, 5000:5050] = np.nan
    features = proc.process(epoch(gappy))
    gappy_good = sum(r["good"] for r in proc.rows)

    assert 0 < gappy_good < clean_good
    assert all(np.isfinite(v) for v in features.extras.values())
    assert np.isfinite(features.spectrum).all()


def test_interpolated_samples_never_reach_a_feature():
    eeg = synth(30.0)
    eeg[:, 5000:5050] = np.nan
    proc = MuseProcessor()
    proc.process(epoch(eeg))
    touched = [r for r in proc.rows
               if r["win_start_s"] <= 5025 / FS < r["win_end_s"]]
    assert touched and not any(r["good"] for r in touched)


def test_fill_gaps_interpolates_rather_than_zeroing():
    """A zero-filled hole is a step edge, and the filter rings on it into the
    neighbouring windows that are otherwise fine."""
    x = np.linspace(0.0, 10.0, 100).reshape(-1, 1)
    holed = x.copy()
    holed[40:50] = np.nan
    filled, missing = fill_gaps(holed)
    assert missing[40:50, 0].all()
    assert np.allclose(filled, x, atol=1e-6)


# --- optional heart and breathing ------------------------------------------

def beats(n=40, bpm=68.0):
    return [60_000.0 / bpm] * n


def test_neither_stream_means_no_autonomic_row():
    proc = MuseProcessor()
    features = proc.process(epoch(synth(30.0)))
    assert proc.autonomic_row is None
    assert not {"mean_hr_bpm", "rmssd_ms", "breathing_bpm"} & set(features.extras)


def test_beats_alone_contribute_heart_features_only():
    proc = MuseProcessor()
    proc.add_beats(beats())
    features = proc.process(epoch(synth(30.0)))
    assert proc.autonomic_row["good"] == 1
    assert "mean_hr_bpm" in features.extras and "rmssd_ms" in features.extras
    assert "breathing_bpm" not in features.extras
    assert features.extras["mean_hr_bpm"] == pytest.approx(68.0, abs=1.0)


def test_breathing_alone_contributes_the_breathing_feature_only():
    proc = MuseProcessor()
    for t in np.arange(0.0, 30.0, 1.0):
        proc.add_breathing(float(t), 16.0)
    features = proc.process(epoch(synth(30.0)))
    assert proc.autonomic_row["good"] == 1
    assert features.extras["breathing_bpm"] == pytest.approx(16.0, abs=0.5)
    assert "mean_hr_bpm" not in features.extras


def test_both_streams_contribute_all_three():
    proc = MuseProcessor()
    proc.add_beats(beats())
    for t in np.arange(0.0, 30.0, 1.0):
        proc.add_breathing(float(t), 16.0)
    features = proc.process(epoch(synth(30.0)))
    assert {"mean_hr_bpm", "rmssd_ms", "breathing_bpm"} <= set(features.extras)


def test_too_few_beats_drops_the_heart_features_not_the_breathing():
    """The old all-or-nothing gate discarded breathing too when the beats were
    unusable, costing three of arousal's voters instead of one."""
    proc = MuseProcessor()
    proc.add_beats(beats(n=4))
    for t in np.arange(0.0, 30.0, 1.0):
        proc.add_breathing(float(t), 16.0)
    features = proc.process(epoch(synth(30.0)))
    assert "mean_hr_bpm" not in features.extras
    assert "breathing_bpm" in features.extras


def test_history_is_trimmed_so_memory_stays_flat():
    proc = MuseProcessor()
    for k in range(6):
        proc.add_beats(beats(n=10))
        for t in np.arange(k * 5.0, (k + 1) * 5.0, 1.0):
            proc.add_breathing(float(t), 16.0)
        proc.process(epoch(synth(30.0), counter=int(k * 5 * FS)))
    assert len(proc._rr) < 60 and len(proc._breath) < 60


def test_partial_autonomic_reaches_the_axes_without_crashing():
    """The reason score.py now reads only the keys that are present: with
    `good` meaning 'at least one measurable', indexing all three raised
    KeyError on a breathing-only rig."""
    rest = resting()
    proc = MuseProcessor()
    mapper = MuseAffectMapper(proc)
    for t in np.arange(0.0, 30.0, 1.0):
        proc.add_breathing(float(t), 24.0)          # fast breathing
    features = proc.process(epoch(synth(30.0)), resting=rest)
    values = mapper.map(features, calibrated=True, resting=rest)
    assert 0.0 <= values.axes["arousal"] <= 1.0
    assert values.confidence["arousal"] > 0.0


# --- the axis contract -----------------------------------------------------

def test_unit_conversion_puts_resting_at_the_middle():
    assert to_unit(0.0) == 0.5
    assert to_unit(-1.0) == 0.0
    assert to_unit(1.0) == 1.0
    assert to_unit(float("nan")) == 0.5


def test_uncalibrated_axes_stay_neutral():
    proc = MuseProcessor()
    features = proc.process(epoch(synth(30.0)))
    values = MuseAffectMapper(proc).map(features, calibrated=False, resting=None)
    assert all(v == 0.5 for v in values.axes.values())
    assert all(c == 0.0 for c in values.confidence.values())
    assert values.implemented is True


def test_direction_and_engagement_are_unfolded_from_the_same_z_scores():
    rest = resting()
    proc = MuseProcessor()
    mapper = MuseAffectMapper(proc)

    away = mapper.map(proc.process(epoch(synth(30.0, asym=-0.35)), resting=rest),
                      calibrated=True, resting=rest)
    toward = mapper.map(proc.process(epoch(synth(30.0, asym=+0.35)), resting=rest),
                        calibrated=True, resting=rest)
    assert toward.axes["direction"] > away.axes["direction"] + 0.2
    assert toward.confidence["direction"] > 0.0


def test_resting_baseline_is_derived_once_per_recording():
    rest = resting()
    proc = MuseProcessor()
    mapper = MuseAffectMapper(proc)
    features = proc.process(epoch(synth(30.0)), resting=rest)
    mapper.map(features, calibrated=True, resting=rest)
    first = mapper._baseline
    mapper.map(features, calibrated=True, resting=rest)
    assert mapper._baseline is first, "baseline was re-derived"
