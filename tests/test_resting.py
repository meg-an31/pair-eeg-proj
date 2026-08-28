"""The wall-staring block: collect, persist, restore, reuse."""

import time

import numpy as np
import pytest

from pair_eeg.config import EEG
from pair_eeg.pipeline.resting import (
    LIVE,
    RESTORED,
    RestingBaseline,
    RestingCollector,
    RestingStore,
)


def make(n=1024, wearer="meg"):
    rng = np.random.default_rng(3)
    return RestingBaseline(
        wearer=wearer,
        recorded_at=time.time(),
        fs=EEG.rate_hz,
        channels=EEG.channels,
        eeg=rng.normal(0, 15, (n, 4)).astype(np.float32),
    )


def test_round_trip_preserves_samples(tmp_path):
    store = RestingStore(tmp_path)
    original = make()
    store.save(original)

    restored = store.load("meg")
    assert restored is not None
    np.testing.assert_array_equal(restored.eeg, original.eeg)
    assert restored.channels == EEG.channels
    assert restored.fs == EEG.rate_hz
    assert restored.source == RESTORED
    assert original.source == LIVE


def test_missing_wearer_is_a_miss(tmp_path):
    assert RestingStore(tmp_path).load("nobody") is None
    assert not RestingStore(tmp_path).exists("nobody")


def test_corrupt_cache_is_a_miss_not_a_crash(tmp_path):
    store = RestingStore(tmp_path)
    store.save(make())
    (tmp_path / "meg" / "resting.npz").write_bytes(b"not an npz")
    assert store.load("meg") is None


def test_save_overwrites(tmp_path):
    store = RestingStore(tmp_path)
    store.save(make(n=512))
    store.save(make(n=2048))
    restored = store.load("meg")
    assert restored.n_samples == 2048


def test_clear(tmp_path):
    store = RestingStore(tmp_path)
    store.save(make())
    assert store.exists("meg")
    store.clear("meg")
    assert not store.exists("meg")


def test_duration_and_age():
    b = make(n=int(EEG.rate_hz * 120))
    assert b.duration_s == pytest.approx(120.0)
    assert b.age_s < 1.0


def test_channel_lookup_by_name():
    b = make()
    np.testing.assert_array_equal(b.channel("AF7"), b.eeg[:, 1])


def test_collector_takes_only_the_hop_not_the_window():
    """Overlapping windows must not count the same samples repeatedly."""
    c = RestingCollector(EEG, target_s=4.0)
    window_n, hop_n = 1024, 256
    for i in range(4):
        window = np.full((window_n, 4), float(i), dtype=np.float32)
        c.observe(i * hop_n, window, hop_n)

    assert c.seconds == pytest.approx(4 * hop_n / EEG.rate_hz)
    built = c.build("meg")
    assert built.n_samples == 4 * hop_n, "window overlap was double-counted"


def test_collector_progress_and_completion():
    c = RestingCollector(EEG, target_s=2.0)
    hop_n = 256
    assert not c.complete
    for i in range(int(2.0 * EEG.rate_hz / hop_n)):
        c.observe(i * hop_n, np.zeros((1024, 4), dtype=np.float32), hop_n)
    assert c.complete
    assert c.progress == 1.0


def test_collector_with_no_epochs_builds_nothing():
    assert RestingCollector(EEG, 2.0).build("meg") is None


def test_reset_clears():
    c = RestingCollector(EEG, 2.0)
    c.observe(0, np.zeros((1024, 4), dtype=np.float32), 256)
    c.reset()
    assert c.seconds == 0.0
    assert c.build("meg") is None


def test_session_restores_saved_resting(tmp_path):
    """A wearer who has stared at a wall before should not have to again."""
    from dataclasses import replace
    from pair_eeg.config import DEFAULT
    from pair_eeg.pipeline.session import Session, SessionState

    store = RestingStore(tmp_path)
    store.save(make(n=int(EEG.rate_hz * 120)))

    async def emit(_):
        pass

    s = Session(
        session_id="s_test",
        wearer="meg",
        config=replace(DEFAULT, sessions_dir=str(tmp_path)),
        emit=emit,
        resting_store=store,
    )
    assert s.resting is not None
    assert s.calibrated is True
    assert s.resting.source == RESTORED

    assert s.use_saved_resting() is True
    assert s.state is SessionState.LIVE


def test_session_without_saved_resting_is_uncalibrated(tmp_path):
    from dataclasses import replace
    from pair_eeg.config import DEFAULT
    from pair_eeg.pipeline.session import Session

    async def emit(_):
        pass

    s = Session(
        session_id="s_test",
        wearer="nobody",
        config=replace(DEFAULT, sessions_dir=str(tmp_path)),
        emit=emit,
        resting_store=RestingStore(tmp_path),
    )
    assert s.resting is None
    assert s.calibrated is False
    assert s.use_saved_resting() is False


def test_processor_and_mapper_accept_resting():
    """Both blank seams must take the resting block."""
    from pair_eeg.pipeline.processing import EpochWindow, NullProcessor
    from pair_eeg.pipeline.affect import NullAffectMapper

    epoch = EpochWindow(
        t=0.0, counter=0, fs=EEG.rate_hz, channels=EEG.channels,
        eeg=np.zeros((1024, 4), dtype=np.float32),
    )
    resting = make()
    features = NullProcessor().process(epoch, resting=resting)
    values = NullAffectMapper().map(features, calibrated=True, resting=resting)
    assert set(values.axes)


# --- a short block must not replace a good one ----------------------------

def test_a_too_short_block_is_refused_and_leaves_the_stored_one_alone(tmp_path):
    """A baseline that ends early still builds a technically valid block. If
    that gets saved it replaces a good recording with one that cannot support a
    single statistic, and every axis afterwards reads 0.5 with no confidence —
    indistinguishable from an unimplemented stage."""
    import asyncio

    from pair_eeg.config import EEG, Config
    from pair_eeg.pipeline.session import MIN_RESTING_S, Session

    store = RestingStore(tmp_path)
    good = RestingBaseline(
        wearer="w", recorded_at=time.time(), fs=EEG.rate_hz,
        channels=EEG.channels,
        eeg=np.zeros((int(EEG.rate_hz * 60), EEG.n_channels), dtype=np.float32),
    )
    store.save(good)

    async def emit(_):
        pass

    s = Session("s", "w", Config(sessions_dir=str(tmp_path)), emit,
                resting_store=store)
    assert s.resting is not None and s.resting.duration_s == pytest.approx(60.0)

    # one second of "rest" — far below the floor
    s._collector.observe(0, np.zeros((256, EEG.n_channels), dtype=np.float32), 256)
    s.baseline.observe({"x": 1.0})
    s.baseline.freeze()
    s._finish_resting()

    assert s.resting.duration_s == pytest.approx(60.0), "short block was adopted"
    assert store.load("w").duration_s == pytest.approx(60.0), "stored block overwritten"
    assert MIN_RESTING_S > 1.0
