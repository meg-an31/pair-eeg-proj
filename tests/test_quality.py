import numpy as np

from pair_eeg.config import EEG
from pair_eeg.pipeline.quality import FLAT, NOISY, OK, AcceptRate, QualityGate
from pair_eeg.pipeline.ringbuffer import Window


def window_of(values):
    arr = np.asarray(values, dtype=np.float32)
    return Window(start=0, samples=arr, n_missing=0)


def test_good_signal_accepted():
    rng = np.random.default_rng(1)
    arr = rng.normal(0, 15, (1024, 4)).astype(np.float32)
    verdict = QualityGate(EEG.channels).check(window_of(arr))
    assert verdict.accepted
    assert all(v == OK for v in verdict.channels.values())


def test_flat_channel_detected():
    arr = np.zeros((1024, 4), dtype=np.float32)
    verdict = QualityGate(EEG.channels).check(window_of(arr))
    assert not verdict.accepted
    assert all(v == FLAT for v in verdict.channels.values())


def test_noisy_channel_detected():
    rng = np.random.default_rng(2)
    arr = rng.normal(0, 15, (1024, 4)).astype(np.float32)
    arr[:, 1] = rng.normal(0, 200, 1024)
    verdict = QualityGate(EEG.channels).check(window_of(arr))
    assert verdict.channels["AF7"] == NOISY
    assert verdict.accepted  # three good channels is still usable


def test_incomplete_window_rejected():
    arr = np.full((1024, 4), np.nan, dtype=np.float32)
    verdict = QualityGate(EEG.channels).check(Window(0, arr, n_missing=900))
    assert not verdict.accepted
    assert "incomplete" in verdict.reason


def test_accept_rate_windowing():
    rate = AcceptRate(window_s=10.0, hop_s=1.0)
    for _ in range(10):
        rate.record(True)
    assert rate.rate == 1.0
    for _ in range(5):
        rate.record(False)
    assert rate.rate == 0.5
    assert rate.n == 10  # capped at the window
