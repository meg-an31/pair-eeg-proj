import asyncio
import time

import numpy as np
import pytest

from pair_eeg.config import EEG, IMU, PPG, Config
from pair_eeg.pipeline.affect import AffectValues, Smoother
from pair_eeg.pipeline.processing import EpochWindow, ProcessedFeatures, N_BINS, BAND_NAMES
from pair_eeg.pipeline.session import Baseline, Session, SessionState
from pair_eeg.transport.protocol import DataFrame

CFG = Config(window_s=4.0, hop_s=1.0, baseline_s=5.0, sessions_dir="/tmp/none")


def mk(cfg=CFG, **kw):
    out = []

    async def emit(p):
        out.append(p)

    return Session("s", "w", cfg, emit, **kw), out


def eeg_frame(counter, n, amp=15.0, seed=0):
    rng = np.random.default_rng(seed)
    return DataFrame(EEG, counter, 0.0, (rng.standard_normal((n, 4)) * amp).astype(np.float32))


# --- shapes ---------------------------------------------------------------

async def test_null_processor_shapes_through_the_pipe():
    s, out = mk()
    s.skip_baseline()
    for i in range(0, 1024, 128):
        s.ingest(eeg_frame(i, 128, seed=i))
    await s._tick(0, 1024)
    p = out[-1]
    assert np.array(p["spectrum"]).shape == (4, N_BINS)
    assert np.array(p["bands"]).shape == (4, len(BAND_NAMES))
    assert p["freqs_hz"] == {"n": 128, "spacing": 1.0}


# --- payload contract -----------------------------------------------------

FRONTEND_KEYS = {"type", "session", "seq", "t", "state", "quality",
                 "channels", "freqs_hz", "spectrum", "bands", "band_names",
                 "processing", "axes", "axes_raw", "confidence", "affect"}


def test_freeze_with_no_features_still_claims_calibrated():
    """NullProcessor returns extras={}, so observe() bumps n without recording
    anything; freeze() then sets frozen=True on an empty baseline."""
    b = Baseline()
    for _ in range(10):
        b.observe({})
    b.freeze()
    assert b.n == 10
    assert b.mean == {} and b.sd == {}
    assert not b.frozen, "an empty baseline must not report itself as frozen/calibrated"


def test_skip_baseline_is_unguarded():
    s, _ = mk()
    s.state = SessionState.ENDED
    s.skip_baseline()
    assert s.state is SessionState.ENDED, "skip_baseline resurrected an ended session"


def test_begin_baseline_from_live_does_not_reset_the_smoother():
    s, _ = mk()
    s.smoother.update({"valence": 1.0})
    s.state = SessionState.LIVE
    s.begin_baseline()
    assert s.smoother._state == {}, "stale smoothed axes survive re-baselining"


@pytest.mark.xfail(strict=True, reason=
    "BUG: _update_state only runs from _tick, so a stalled stream leaves the state at LIVE forever (session.py:259).")
async def test_state_goes_stale_when_data_stops():
    """DEGRADED exists to distinguish 'calm' from 'not measuring', but the
    state machine only runs from _tick, which needs data."""
    s, out = mk()
    for i in range(0, 1024, 128):
        s.ingest(eeg_frame(i, 128, seed=i))
    s.skip_baseline()
    await s._tick(0, 1024)
    assert s.state is SessionState.LIVE
    await s.start()
    await asyncio.sleep(3.0)          # data has stopped
    await s.stop()
    assert s.state is SessionState.DEGRADED, (
        f"3s with no data left the session in {s.state.value} and it emitted "
        f"{len(out)} messages"
    )


# --- co-window ------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=
    "BUG: _co_window scales the EEG counter directly, assuming all streams share a counter origin (session.py:251-257).")
def test_co_window_assumes_shared_counter_origins():
    """PPG/IMU counters are independent of the EEG counter on real hardware."""
    s, _ = mk()
    s.ingest(eeg_frame(100_000, 1024))
    rng = np.random.default_rng(0)
    # PPG stream that started at its own counter 0, co-timed with the EEG above
    s.ingest(DataFrame(PPG, 0, 0.0, rng.standard_normal((256, 3)).astype(np.float32)))
    w = s._co_window(PPG, 100_000, 1024)
    assert w is not None
    assert not np.all(np.isnan(w)), "co-window came back entirely NaN"


def test_co_window_lengths():
    s, _ = mk()
    s.ingest(eeg_frame(0, 1024))
    s.ingest(DataFrame(PPG, 0, 0.0, np.zeros((256, 3), np.float32)))
    s.ingest(DataFrame(IMU, 0, 0.0, np.zeros((208, 6), np.float32)))
    assert s._co_window(PPG, 0, 1024).shape == (256, 3)   # 4.0 s at 64 Hz
    assert s._co_window(IMU, 0, 1024).shape == (208, 6)   # 4.0 s at 52 Hz


def test_co_window_hides_a_completely_missing_stream():
    s, _ = mk()
    s.ingest(eeg_frame(0, 1024))
    s.ingest(DataFrame(IMU, 0, 0.0, np.zeros((1, 6), np.float32)))
    w = s._co_window(IMU, 500_000, 1024)
    assert w is None or not np.all(np.isnan(w)), (
        "an all-NaN array is returned indistinguishably from real data"
    )


# --- smoother -------------------------------------------------------------

def test_smoother_stays_in_range():
    sm = Smoother(0.333)
    rng = np.random.default_rng(0)
    for _ in range(500):
        v = sm.update({a: float(x) for a, x in zip("abc", rng.random(3))})
        assert all(0.0 <= x <= 1.0 for x in v.values())


def test_smoother_returns_axes_the_mapper_did_not_produce():
    sm = Smoother(0.5)
    sm.update({"valence": 0.9, "arousal": 0.1})
    out = sm.update({"valence": 0.9})
    assert set(out) == {"valence"}, f"stale axes leaked through: {out}"


def test_affect_values_validates_confidence():
    with pytest.raises(ValueError, match="confidence"):
        AffectValues(axes={"valence": 0.5}, confidence={"valence": 7.0})
