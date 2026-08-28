"""The whole pipe with the muse stages in it.

`tests/test_muse_stages.py` checks the adapters in isolation. This one drives a
Session the way the server does — frames in, payloads out — at muse's native
30 s / 5 s geometry.
"""

import time

import numpy as np
import pytest

from pair_eeg.config import EEG, Config
from pair_eeg.pipeline.affect import MuseAffectMapper
from pair_eeg.pipeline.processing import MuseProcessor
from pair_eeg.pipeline.session import Session, SessionState
from pair_eeg.transport.protocol import DataFrame
from tests.muse_signals import synth

CFG = Config(window_s=30.0, hop_s=5.0, baseline_s=1e9, sessions_dir="/tmp/none",
             smoothing_windows=1.0)


class Rig:
    """Drives a session in hop-sized steps without waiting for wall time."""

    def __init__(self):
        self.out: list[dict] = []
        self.processor = MuseProcessor()
        self.session = Session("s", "w", CFG, self._emit,
                               processor=self.processor,
                               affect=MuseAffectMapper(self.processor))
        self.hop_n = int(EEG.rate_hz * CFG.hop_s)
        self.window_n = int(EEG.rate_hz * CFG.window_s)
        self.counter = 0

    async def _emit(self, payload):
        self.out.append(payload)

    def push_hop(self, **kw):
        eeg = synth(CFG.hop_s, seed=self.counter, **kw).T      # (samples, ch)
        self.session.ingest(DataFrame(EEG, self.counter, 0.0,
                                      np.ascontiguousarray(eeg, dtype=np.float32)))
        self.counter += self.hop_n

    async def tick(self):
        await self.session._tick(self.counter - self.window_n, self.window_n)
        return self.out[-1]

    async def run(self, seconds, **kw):
        for _ in range(int(seconds / CFG.hop_s)):
            self.push_hop(**kw)
            if self.counter >= self.window_n:
                await self.tick()
        return self.out[-1]

    def start_baseline(self):
        """One frame first: begin_baseline() is ignored while CONNECTING."""
        self.push_hop()
        self.session.begin_baseline()

    def finish_baseline(self):
        """The baseline block is timed by wall clock rather than by accepted
        data (a known bug). Rewinding the start is how a test triggers the
        freeze without sitting still for two minutes."""
        self.session._baseline_started = time.time() - CFG.baseline_s - 1.0


async def test_muse_stages_produce_calibrated_axes():
    rig = Rig()
    rig.start_baseline()
    await rig.run(150)
    assert rig.session.state is SessionState.BASELINE
    assert rig.out[-1]["axes"] is None, "no axes are published during baseline"

    rig.finish_baseline()
    await rig.run(CFG.hop_s)
    assert rig.session.state is SessionState.LIVE
    assert rig.session.calibrated
    assert rig.session.resting.duration_s >= 30.0

    payload = await rig.run(30)
    assert payload["affect"]["implemented"] is True
    assert payload["affect"]["calibrated"] is True
    assert payload["affect"]["name"] == "muse_v1"
    assert payload["processing"]["name"] == "muse_v1"
    assert set(payload["axes"]) == {"valence", "arousal", "direction",
                                    "engagement", "autonomic"}
    assert all(0.0 <= v <= 1.0 for v in payload["axes"].values())
    assert np.array(payload["spectrum"]).shape == (4, 128)
    assert np.array(payload["bands"]).shape == (4, 5)


async def test_epoch_is_cut_into_the_29_windows_muse_expects():
    rig = Rig()
    rig.session.skip_baseline()
    await rig.run(35)
    assert len(rig.processor.rows) == 29
    assert rig.out[-1]["axes"] is not None


async def test_uncalibrated_axes_do_not_move():
    """Every muse index is a z-score against the resting distribution, so with
    no baseline there is no scale — Invariant 6."""
    rig = Rig()
    rig.session.skip_baseline()
    payload = await rig.run(35)
    assert all(v == 0.5 for v in payload["axes"].values())
    assert all(c == 0.0 for c in payload["confidence"].values())
    assert payload["affect"]["calibrated"] is False


async def test_arousal_follows_the_signal_after_calibration():
    rig = Rig()
    rig.start_baseline()
    await rig.run(150, alpha=12.0, beta=4.0)
    rig.finish_baseline()
    await rig.run(CFG.hop_s)

    calm = (await rig.run(35, alpha=12.0, beta=2.0))["axes"]["arousal"]
    hot = (await rig.run(35, alpha=5.0, beta=14.0))["axes"]["arousal"]
    assert hot > calm + 0.1, f"arousal did not move: {calm:.2f} -> {hot:.2f}"


async def test_brief_frowns_lower_valence_after_calibration():
    """Valence is driven by the PHASIC frown-muscle component, so the signal
    it responds to is brief bursts, not a held clench."""
    rig = Rig()
    rig.start_baseline()
    await rig.run(150)
    rig.finish_baseline()
    await rig.run(CFG.hop_s)

    quiet = (await rig.run(35))["axes"]["valence"]
    frowning = (await rig.run(
        35, emg_bursts=((1.0, 3.0, 14.0), (3.5, 3.0, 14.0))))["axes"]["valence"]
    assert frowning < quiet - 0.1, f"valence did not move: {quiet:.2f} -> {frowning:.2f}"


async def test_a_held_clench_is_not_read_as_unpleasantness():
    """The other half of the same design, carried over deliberately: sustained
    muscle is tension, not valence. Worth pinning down, because it is the
    behaviour that looks like a broken valence axis if you do not expect it."""
    rig = Rig()
    rig.start_baseline()
    await rig.run(150)
    rig.finish_baseline()
    await rig.run(CFG.hop_s)

    quiet = (await rig.run(35))["axes"]["valence"]
    clenched = (await rig.run(35, emg=15.0))["axes"]["valence"]
    assert abs(clenched - quiet) < 0.1, f"{quiet:.2f} -> {clenched:.2f}"


async def test_payload_keys_never_vary_by_variant():
    """Invariant 11 — a rejected epoch must not change the message shape."""
    rig = Rig()
    rig.session.skip_baseline()
    await rig.run(35)
    live = rig.out[-1]

    rig.push_hop()
    rig.session.buffers[EEG.name].write(
        rig.counter - rig.window_n,
        np.zeros((rig.window_n, 4), dtype=np.float32),          # flat = rejected
    )
    rejected = await rig.tick()
    assert rejected["quality"]["accepted"] is False
    assert set(rejected) == set(live)


async def test_nothing_reaching_the_payload_is_nan():
    """Invariant 10 — NaN serialises as a token that makes JSON.parse discard
    the whole message, not just the bad field."""
    rig = Rig()
    rig.start_baseline()
    await rig.run(150)
    rig.finish_baseline()
    payload = await rig.run(35)

    def check(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                check(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                check(v, f"{path}[{i}]")
        elif isinstance(node, float):
            assert np.isfinite(node), f"non-finite at {path}"

    check(payload)
