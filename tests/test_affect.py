import numpy as np
import pytest

from pair_eeg.pipeline.affect import AXES, AffectValues, NullAffectMapper, Smoother
from pair_eeg.pipeline.processing import (
    BAND_NAMES,
    N_BINS,
    EpochWindow,
    NullProcessor,
)
from pair_eeg.config import EEG


def make_epoch(n=1024):
    return EpochWindow(
        t=0.0,
        counter=0,
        fs=EEG.rate_hz,
        channels=EEG.channels,
        eeg=np.zeros((n, EEG.n_channels), dtype=np.float32),
    )


def test_bins_follow_from_sample_rate():
    # 256 Hz sampling -> 128 Hz Nyquist -> 128 bins at 1 Hz spacing.
    assert N_BINS == 128
    assert EEG.rate_hz / 2 == N_BINS


def test_null_processor_shapes():
    features = NullProcessor().process(make_epoch())
    assert features.spectrum.shape == (EEG.n_channels, N_BINS)
    assert features.bands.shape == (EEG.n_channels, len(BAND_NAMES))
    assert features.implemented is False


def test_null_affect_is_neutral_and_flagged():
    features = NullProcessor().process(make_epoch())
    values = NullAffectMapper().map(features, calibrated=False)
    assert set(values.axes) == set(AXES)
    assert all(v == 0.5 for v in values.axes.values())
    assert values.implemented is False


def test_axes_must_be_bounded():
    with pytest.raises(ValueError):
        AffectValues(axes={"valence": 1.4})
    with pytest.raises(ValueError):
        AffectValues(axes={"valence": -0.1})


def test_smoother_stays_in_range():
    sm = Smoother(alpha=0.33)
    rng = np.random.default_rng(0)
    for _ in range(200):
        out = sm.update({a: float(rng.random()) for a in AXES})
        assert all(0.0 <= v <= 1.0 for v in out.values())


def test_smoother_converges_to_constant():
    sm = Smoother(alpha=0.5)
    for _ in range(60):
        out = sm.update({"valence": 0.8})
    assert out["valence"] == pytest.approx(0.8, abs=1e-6)


def test_smoother_first_value_passes_through():
    sm = Smoother(alpha=0.1)
    assert sm.update({"arousal": 0.9})["arousal"] == 0.9


def test_smoother_rejects_bad_alpha():
    with pytest.raises(ValueError):
        Smoother(alpha=0.0)
