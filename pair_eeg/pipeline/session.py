"""Session lifecycle and the epoch loop.

    connecting -> fit_check -> baseline -> live
                                   |         |
                                   +--- degraded <--+

Session state is part of the protocol, not just internal bookkeeping: nothing
downstream is interpretable before a baseline exists, so the front end has to
know which phase it is in to render honestly.

`degraded` exists because losing 20-40% of epochs is the normal operating
point for dry electrodes. A single rejected epoch is not a state change — it
just produces no estimate that tick. Sustained rejection is, and the interface
must be able to tell "calm" from "not currently measuring".
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

import numpy as np

from ..config import EEG, IMU, PPG, THERM, Config
from ..transport.protocol import DataFrame
from .affect import AffectMapper, AffectValues, NullAffectMapper, Smoother
from .processing import (
    BIN_HZ,
    N_BINS,
    EpochWindow,
    NullProcessor,
    ProcessedFeatures,
    Processor,
)
from .quality import AcceptRate, QualityGate, QualityVerdict
from .resting import RestingBaseline, RestingCollector, RestingStore
from .rawlog import RawLog
from .ringbuffer import RingBuffer


log = logging.getLogger("pair_eeg.session")


class SessionState(str, Enum):
    CONNECTING = "connecting"
    FIT_CHECK = "fit_check"
    BASELINE = "baseline"
    LIVE = "live"
    DEGRADED = "degraded"
    ENDED = "ended"


@dataclass
class Baseline:
    """Running feature statistics, frozen at the end of the baseline phase.

    Absolute band powers shift with skull, hair, sweat and how the band sat
    that morning, so every downstream number is relative to this. Persisted
    the moment it freezes — losing it to a restart costs the wearer another
    two minutes of sitting still.
    """

    n: int = 0
    _sum: dict[str, float] = field(default_factory=dict)
    _sumsq: dict[str, float] = field(default_factory=dict)
    frozen: bool = False
    mean: dict[str, float] = field(default_factory=dict)
    sd: dict[str, float] = field(default_factory=dict)

    def observe(self, features: dict[str, float]) -> None:
        if self.frozen:
            return
        for key, value in features.items():
            self._sum[key] = self._sum.get(key, 0.0) + value
            self._sumsq[key] = self._sumsq.get(key, 0.0) + value * value
        self.n += 1

    def freeze(self) -> None:
        # No observed features means no calibration, whatever the tick count
        # says. Claiming otherwise would put `calibrated: true` on a payload
        # normalised against nothing.
        if self.frozen or self.n == 0 or not self._sum:
            return
        for key, total in self._sum.items():
            mean = total / self.n
            var = max(self._sumsq[key] / self.n - mean * mean, 0.0)
            self.mean[key] = mean
            self.sd[key] = float(np.sqrt(var)) or 1.0
        self.frozen = True

    def to_dict(self) -> dict:
        return {"n": self.n, "frozen": self.frozen, "mean": self.mean, "sd": self.sd}

    @classmethod
    def from_dict(cls, d: dict) -> "Baseline":
        b = cls(n=int(d.get("n", 0)), frozen=bool(d.get("frozen", False)))
        b.mean = dict(d.get("mean", {}))
        b.sd = dict(d.get("sd", {}))
        return b


# Emitted on every tick. The consumer decides what to draw.
Emit = Callable[[dict], Awaitable[None]]


class Session:
    """One wearer, one recording, one pipeline instance.

    MAX_TICK_ERRORS bounds how long a broken processor is tolerated before
    the session gives up; a handful of transient failures should not end a
    recording, but a persistently raising stage should not be papered over.

    Owns the ring buffers, the state machine and the epoch loop. Frames go in
    via `ingest`; results come out via the `emit` callback.
    """

    MAX_TICK_ERRORS = 20

    def __init__(
        self,
        session_id: str,
        wearer: str,
        config: Config,
        emit: Emit,
        processor: Processor | None = None,
        affect: AffectMapper | None = None,
        raw_log: RawLog | None = None,
        session_dir: Path | None = None,
        resting_store: RestingStore | None = None,
    ):
        self.id = session_id
        self.wearer = wearer
        self.cfg = config
        self.emit = emit
        self.processor: Processor = processor or NullProcessor()
        self.affect: AffectMapper = affect or NullAffectMapper()
        self.raw_log = raw_log
        self.session_dir = session_dir

        self.state = SessionState.CONNECTING
        self.baseline = Baseline()

        # Two minutes of staring at a wall. Restored from the last recording
        # for this wearer so nobody has to sit still twice.
        self.resting_store = resting_store
        self.resting: RestingBaseline | None = (
            resting_store.load(wearer) if resting_store else None
        )
        self._collector = RestingCollector(EEG, config.baseline_s)
        self.started_at = time.time()
        self._state_since = time.time()
        self._baseline_started: float | None = None

        self.buffers = {
            s.name: RingBuffer(s, int(s.rate_hz * config.buffer_s))
            for s in (EEG, PPG, IMU, THERM)
        }
        self.gate = QualityGate(EEG.channels)
        self.accept_rate = AcceptRate(config.accept_rate_window_s, config.hop_s)
        self.smoother = Smoother(config.smoothing_alpha)

        self._seq = 0
        self._next_epoch_end: int | None = None
        self._task: asyncio.Task | None = None
        self._degraded_since: float | None = None
        self._frames_in = 0
        self._discontinuities = 0
        self._last_ingest: float | None = None
        self._tick_errors = 0

        # Most recent result, held so a poller has something to read. Always
        # overwritten, never queued: a slow reader should skip values, not
        # fall progressively further behind a growing backlog.
        self.latest: dict | None = None
        self.latest_axes: dict | None = None

    # ------------------------------------------------------------------ io

    def ingest(self, frame: DataFrame) -> None:
        """Accept one data frame. Raw goes to disk before anything reads it."""
        if self.raw_log is not None:
            self.raw_log.submit(frame)

        buf = self.buffers[frame.stream.name]
        if not buf.empty and frame.counter > buf.head:
            self._discontinuities += 1
        buf.write(frame.counter, frame.samples)
        self._frames_in += 1
        self._last_ingest = time.time()

        if self.state is SessionState.CONNECTING:
            self._transition(SessionState.FIT_CHECK)

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name=f"session-{self.id}")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._transition(SessionState.ENDED)

    # -------------------------------------------------------------- control

    @property
    def calibrated(self) -> bool:
        """True only with a resting block to normalise against."""
        return self.resting is not None

    def use_saved_resting(self) -> bool:
        """Skip the wall-staring block and reuse the stored one."""
        if self.resting is None and self.resting_store is not None:
            self.resting = self.resting_store.load(self.wearer)
        if self.resting is not None:
            self._transition(SessionState.LIVE)
            return True
        return False

    def begin_baseline(self) -> None:
        """Start the eyes-open rest block. Idempotent."""
        if self.state in (
            SessionState.FIT_CHECK,
            SessionState.LIVE,
            SessionState.DEGRADED,
            SessionState.BASELINE,
        ):
            self.baseline = Baseline()
            self._collector.reset()
            self._baseline_started = time.time()
            # Old smoother state was normalised against the previous
            # baseline; carrying it across would blend two calibrations.
            self.smoother.reset()
            self._transition(SessionState.BASELINE)

    def skip_baseline(self) -> None:
        """Go live uncalibrated. Values will be flagged `calibrated: false`."""
        if self.state in (SessionState.ENDED, SessionState.CONNECTING):
            return
        self._transition(SessionState.LIVE)

    def _transition(self, state: SessionState) -> None:
        if state is self.state:
            return
        self.state = state
        self._state_since = time.time()

    # ------------------------------------------------------------- the loop

    async def _loop(self) -> None:
        window_n = int(self.cfg.window_s * EEG.rate_hz)
        hop_n = int(self.cfg.hop_s * EEG.rate_hz)
        eeg = self.buffers[EEG.name]

        if window_n <= 0:
            raise ValueError(f"window_s={self.cfg.window_s} is shorter than one sample")
        if hop_n <= 0:
            # A zero hop never advances _next_epoch_end and spins forever,
            # starving the event loop with no way back.
            raise ValueError(f"hop_s={self.cfg.hop_s} is shorter than one sample")

        while True:
            await asyncio.sleep(self.cfg.hop_s)

            if await self._check_stalled():
                continue

            if eeg.empty or eeg.available() < window_n:
                continue

            # Align the first epoch to whatever the device was at, then step
            # by fixed hops so window boundaries stay on the sample grid.
            if self._next_epoch_end is None:
                self._next_epoch_end = eeg.head

            while self._next_epoch_end <= eeg.head:
                start = self._next_epoch_end - window_n
                if start < eeg.tail:
                    self._next_epoch_end = eeg.head
                    break
                try:
                    await self._tick(start, window_n)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Without this the task dies, the socket stays open and
                    # the front end simply goes quiet with nothing logged.
                    self._tick_errors += 1
                    log.exception(
                        "session %s: epoch tick failed (%d so far)", self.id, self._tick_errors
                    )
                    if self._tick_errors >= self.MAX_TICK_ERRORS:
                        log.error("session %s: too many tick failures, stopping", self.id)
                        raise
                self._next_epoch_end += hop_n

    async def _check_stalled(self) -> bool:
        """Report a stream that has stopped arriving.

        Everything else runs off `_tick`, which only fires when there is data
        to epoch — so without this a dead capture client would leave the state
        at LIVE indefinitely and viewers would see a stale reading as current.
        """
        if self._last_ingest is None or self.state in (
            SessionState.ENDED,
            SessionState.CONNECTING,
        ):
            return False

        quiet = time.time() - self._last_ingest
        if quiet < max(3.0 * self.cfg.hop_s, 2.0):
            return False

        if self.state is not SessionState.DEGRADED:
            self._transition(SessionState.DEGRADED)

        self._seq += 1
        await self.emit(
            {
                **self._envelope(
                    QualityVerdict(
                        accepted=False,
                        channels={c: "flat" for c in EEG.channels},
                        fill_ratio=0.0,
                        reason=f"no data for {quiet:.1f}s",
                    )
                ),
                **self._empty_fields(),
            }
        )
        return True

    async def _tick(self, start: int, window_n: int) -> None:
        eeg_window = self.buffers[EEG.name].read(start, window_n)
        verdict = self.gate.check(eeg_window)
        self.accept_rate.record(verdict.accepted)
        self._seq += 1

        self._update_state(verdict)

        if not verdict.accepted:
            await self.emit(self._quality_only_payload(verdict))
            return

        epoch = EpochWindow(
            t=time.time() - self.started_at,
            counter=start,
            fs=EEG.rate_hz,
            channels=EEG.channels,
            eeg=eeg_window.samples,
            ppg=self._co_window(PPG, start, window_n),
            imu=self._co_window(IMU, start, window_n),
        )

        features = self.processor.process(epoch, resting=self.resting)
        features.check_shapes()

        if self.state is SessionState.BASELINE:
            self.baseline.observe(features.extras)
            self._collector.observe(
                start, eeg_window.samples, int(self.cfg.hop_s * EEG.rate_hz)
            )
            await self.emit(self._baseline_payload(verdict, features))
            return

        values = self.affect.map(
            features, calibrated=self.calibrated, resting=self.resting
        )
        smoothed = self.smoother.update(values.axes)
        await self.emit(self._payload(verdict, features, values, smoothed))

    def _co_window(self, stream, eeg_start: int, eeg_n: int) -> np.ndarray | None:
        """The same span of wall time on a stream with a different rate."""
        buf = self.buffers[stream.name]
        if buf.empty:
            return None
        # NOTE: assumes this stream shares a counter origin with EEG. That
        # holds for the synthetic client and for BrainFlow, but the browser
        # decodes a separate counter per characteristic, so PPG and IMU need
        # time-based alignment before they can be trusted here.
        ratio = stream.rate_hz / EEG.rate_hz
        window = buf.read(int(eeg_start * ratio), max(1, int(eeg_n * ratio)))
        if window.n_missing == window.n_samples:
            return None          # entirely absent, not merely gappy
        return window.samples

    def _update_state(self, verdict: QualityVerdict) -> None:
        now = time.time()
        rate = self.accept_rate.rate

        if self.state is SessionState.FIT_CHECK:
            settled = now - self._state_since >= self.cfg.fit_check_settle_s
            if settled and rate >= self.cfg.fit_check_min_accept:
                self.begin_baseline()
            return

        if self.state is SessionState.BASELINE:
            elapsed = now - (self._baseline_started or now)
            if elapsed >= self.cfg.baseline_s:
                self.baseline.freeze()
                self._finish_resting()
                self._persist_baseline()
                self._transition(SessionState.LIVE)
            return

        if self.state in (SessionState.LIVE, SessionState.DEGRADED):
            poor = rate < self.cfg.degraded_below_accept and self.accept_rate.n > 3
            if poor:
                self._degraded_since = self._degraded_since or now
                if now - self._degraded_since >= self.cfg.degraded_after_s:
                    self._transition(SessionState.DEGRADED)
            else:
                self._degraded_since = None
                self._transition(SessionState.LIVE)

    def _finish_resting(self) -> None:
        """Build the resting block and cache it for next time."""
        built = self._collector.build(self.wearer)
        if built is None:
            log.warning(
                "session %s: baseline block ended with no accepted epochs", self.id
            )
            return
        built = replace(built, features=dict(self.baseline.mean))
        self.resting = built
        if self.resting_store is not None:
            try:
                path = self.resting_store.save(built)
                log.info(
                    "session %s: resting block saved (%.0fs) -> %s",
                    self.id,
                    built.duration_s,
                    path,
                )
            except OSError:
                log.exception("session %s: could not save resting block", self.id)

    def _persist_baseline(self) -> None:
        if self.session_dir is None:
            return
        import json

        (self.session_dir / "baseline.json").write_text(
            json.dumps(self.baseline.to_dict(), indent=2)
        )

    # ------------------------------------------------------------- payloads

    def _remember(self, payload: dict) -> dict:
        """Hold the newest payload for polling. Overwrite, never queue."""
        self.latest = payload
        self.latest_axes = {
            "type": "axes",
            "session": self.id,
            "seq": payload["seq"],
            "t": payload["t"],
            "state": payload["state"],
            "axes": payload.get("axes"),
            "confidence": payload.get("confidence"),
            "calibrated": (payload.get("affect") or {}).get("calibrated", False),
            "implemented": (payload.get("affect") or {}).get("implemented", False),
            "accept_rate": (payload.get("quality") or {}).get("accept_rate"),
            "accepted": (payload.get("quality") or {}).get("accepted"),
        }
        return payload

    def _envelope(self, verdict: QualityVerdict) -> dict:
        return {
            "type": "estimate",
            "session": self.id,
            "seq": self._seq,
            "t": round(time.time() - self.started_at, 3),
            "state": self.state.value,
            "quality": {
                "accepted": verdict.accepted,
                "channels": verdict.channels,
                "accept_rate": round(self.accept_rate.rate, 3),
                "fill": round(verdict.fill_ratio, 3),
                "reason": verdict.reason,
                "rms_uv": {k: round(v, 2) for k, v in verdict.rms_uv.items()},
            },
        }

    def _empty_fields(self) -> dict:
        """Every optional field, so the key set never varies by variant."""
        return {
            "channels": list(EEG.channels),
            "freqs_hz": None,
            "spectrum": None,
            "bands": None,
            "band_names": None,
            "processing": None,
            "baseline": None,
            "axes": None,
            "axes_raw": None,
            "confidence": None,
            "affect": None,
        }

    def _quality_only_payload(self, verdict: QualityVerdict) -> dict:
        return self._remember({**self._envelope(verdict), **self._empty_fields()})

    def _baseline_payload(self, verdict: QualityVerdict, features: ProcessedFeatures) -> dict:
        elapsed = time.time() - (self._baseline_started or time.time())
        return self._remember({
            **self._envelope(verdict),
            **self._empty_fields(),
            "baseline": {
                "elapsed_s": round(elapsed, 1),
                "total_s": self.cfg.baseline_s,
                "progress": round(min(elapsed / self.cfg.baseline_s, 1.0), 3),
                "n": self.baseline.n,
            },
            **self._spectral(features),
        })

    def _spectral(self, features: ProcessedFeatures) -> dict:
        return {
            "channels": list(features.channels),
            "freqs_hz": {"n": len(features.freqs), "spacing": BIN_HZ},
            "spectrum": features.spectrum.round(4).tolist(),
            "bands": features.bands.round(4).tolist(),
            "band_names": list(features.band_names),
            "processing": {
                "implemented": features.implemented,
                "name": getattr(self.processor, "name", "unknown"),
            },
        }

    def _payload(
        self,
        verdict: QualityVerdict,
        features: ProcessedFeatures,
        values: AffectValues,
        smoothed: dict[str, float],
    ) -> dict:
        return self._remember({
            **self._envelope(verdict),
            **self._empty_fields(),
            **self._spectral(features),
            "axes": {k: round(v, 4) for k, v in smoothed.items()},
            "axes_raw": {k: round(v, 4) for k, v in values.axes.items()},
            "confidence": {k: round(v, 3) for k, v in values.confidence.items()},
            "affect": {
                "implemented": values.implemented,
                "calibrated": values.calibrated,
                "name": values.source,
            },
        })

    def snapshot(self) -> dict:
        """Sent to a subscriber on connect so a reload costs nothing."""
        return {
            "type": "snapshot",
            "session": self.id,
            "wearer": self.wearer,
            "state": self.state.value,
            "seq": self._seq,
            "uptime_s": round(time.time() - self.started_at, 1),
            "frames_in": self._frames_in,
            "discontinuities": self._discontinuities,
            "accept_rate": round(self.accept_rate.rate, 3),
            "calibrated": self.calibrated,
            "resting": self.resting.summary() if self.resting else None,
            "resting_available": self.resting is not None,
            "config": {
                "window_s": self.cfg.window_s,
                "hop_s": self.cfg.hop_s,
                "baseline_s": self.cfg.baseline_s,
                "n_bins": N_BINS,
                "bin_hz": BIN_HZ,
            },
            "stages": {
                "processing": getattr(self.processor, "name", "unknown"),
                "affect": getattr(self.affect, "name", "unknown"),
            },
        }
