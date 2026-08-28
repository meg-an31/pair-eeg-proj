"""The wearer staring at a wall.

Two minutes of doing nothing, kept as raw samples rather than as summary
statistics, because what a processor wants to do with it is not knowable in
advance. One implementation wants per-band means to z-score against; another
wants the resting spectrum itself as a divisor; another wants to fit a noise
floor. Storing derived numbers would pick for them.

Recorded once and reused. Absolute band powers shift with skull, hair, sweat
and how the band sat that morning, so a resting recording is only strictly
valid for its own session — but making someone sit still for two minutes
every single time is a good way to ensure nobody uses the thing. So it is
persisted per wearer, restored automatically, and stamped with its age so a
processor can decide for itself how much to trust a stale one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import StreamSpec

LIVE = "live"
RESTORED = "restored"


@dataclass(frozen=True)
class RestingBaseline:
    """Raw resting EEG plus whatever has been derived from it so far.

    `eeg` is (n_samples, n_channels) in microvolts, NaN where a sample was
    never received — same convention as an epoch window. Rejected epochs are
    excluded, so it is not necessarily contiguous; `counters` records where
    each retained block came from.
    """

    wearer: str
    recorded_at: float
    fs: float
    channels: tuple[str, ...]
    eeg: np.ndarray
    source: str = LIVE
    counters: np.ndarray | None = None
    features: dict[str, float] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(self.eeg.shape[0]) / self.fs

    @property
    def age_s(self) -> float:
        return max(time.time() - self.recorded_at, 0.0)

    @property
    def n_samples(self) -> int:
        return int(self.eeg.shape[0])

    def channel(self, name: str) -> np.ndarray:
        """One channel as a 1-D array, by name."""
        return self.eeg[:, self.channels.index(name)]

    def summary(self) -> dict:
        return {
            "wearer": self.wearer,
            "recorded_at": self.recorded_at,
            "age_s": round(self.age_s, 1),
            "duration_s": round(self.duration_s, 1),
            "fs": self.fs,
            "channels": list(self.channels),
            "n_samples": self.n_samples,
            "source": self.source,
            "n_features": len(self.features),
        }


class RestingCollector:
    """Accumulates accepted epochs during the baseline block."""

    def __init__(self, stream: StreamSpec, target_s: float):
        self.stream = stream
        self.target_s = target_s
        self._blocks: list[np.ndarray] = []
        self._counters: list[tuple[int, int]] = []
        self._n = 0

    @property
    def seconds(self) -> float:
        return self._n / self.stream.rate_hz

    @property
    def progress(self) -> float:
        return min(self.seconds / self.target_s, 1.0) if self.target_s else 1.0

    @property
    def complete(self) -> bool:
        return self.seconds >= self.target_s

    def observe(self, counter: int, samples: np.ndarray, hop_samples: int) -> None:
        """Take the newest `hop_samples` of an accepted epoch.

        Windows overlap, so taking the whole window each time would count most
        samples several times over and weight the middle of the block far more
        than its edges.
        """
        take = min(hop_samples, samples.shape[0])
        block = samples[-take:]
        self._blocks.append(np.asarray(block, dtype=np.float32))
        self._counters.append((counter + samples.shape[0] - take, take))
        self._n += take

    def build(self, wearer: str) -> RestingBaseline | None:
        if not self._blocks:
            return None
        return RestingBaseline(
            wearer=wearer,
            recorded_at=time.time(),
            fs=self.stream.rate_hz,
            channels=self.stream.channels,
            eeg=np.concatenate(self._blocks, axis=0),
            source=LIVE,
            counters=np.asarray(self._counters, dtype=np.int64),
        )

    def reset(self) -> None:
        self._blocks.clear()
        self._counters.clear()
        self._n = 0


class RestingStore:
    """Persists one resting recording per wearer, overwritten each time.

    Deliberately keeps only the most recent. A history of resting blocks is a
    research artefact, and the raw sessions on disk already are that history;
    this is a cache so the wearer does not have to sit still again.
    """

    FILENAME = "resting.npz"
    METANAME = "resting.json"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _dir(self, wearer: str) -> Path:
        return self.root / wearer

    def save(self, baseline: RestingBaseline) -> Path:
        d = self._dir(baseline.wearer)
        d.mkdir(parents=True, exist_ok=True)
        path = d / self.FILENAME

        # Write beside and rename, so an interrupted save cannot leave a
        # half-written file that looks valid on the next restore. The handle
        # is passed rather than the path because savez appends ".npz" to any
        # name that lacks it, which would defeat the rename.
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                eeg=baseline.eeg,
                counters=baseline.counters
                if baseline.counters is not None
                else np.zeros((0, 2), dtype=np.int64),
            )
        tmp.replace(path)

        (d / self.METANAME).write_text(
            json.dumps(
                {
                    "wearer": baseline.wearer,
                    "recorded_at": baseline.recorded_at,
                    "fs": baseline.fs,
                    "channels": list(baseline.channels),
                    "duration_s": baseline.duration_s,
                    "features": baseline.features,
                },
                indent=2,
            )
        )
        return path

    def load(self, wearer: str) -> RestingBaseline | None:
        d = self._dir(wearer)
        path, meta_path = d / self.FILENAME, d / self.METANAME
        if not path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
            with np.load(path) as data:
                eeg = data["eeg"]
                counters = data["counters"]
        except (OSError, ValueError, KeyError):
            return None       # corrupt cache is a miss, not a crash

        return RestingBaseline(
            wearer=wearer,
            recorded_at=float(meta["recorded_at"]),
            fs=float(meta["fs"]),
            channels=tuple(meta["channels"]),
            eeg=eeg,
            source=RESTORED,
            counters=counters if counters.size else None,
            features=dict(meta.get("features", {})),
        )

    def exists(self, wearer: str) -> bool:
        return (self._dir(wearer) / self.FILENAME).exists()

    def clear(self, wearer: str) -> None:
        for name in (self.FILENAME, self.METANAME):
            (self._dir(wearer) / name).unlink(missing_ok=True)
