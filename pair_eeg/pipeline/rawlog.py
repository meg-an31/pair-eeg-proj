"""Append-only raw capture.

Features are re-derivable; samples are not. Raw goes to disk before anything
else looks at it, and logging must never be able to stall the live path — so
writes are queued and drained by a background task. A slow disk degrades the
recording, not the estimate.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import BinaryIO

import numpy as np

from ..config import STREAMS, StreamSpec
from ..transport.protocol import DataFrame


class RawLog:
    """One directory per session, one flat float32 file per stream.

    Sample counters are recorded in an index sidecar so gaps stay visible
    when the recording is replayed.
    """

    def __init__(self, session_dir: Path, queue_max: int = 4096):
        self.dir = Path(session_dir)
        self.raw_dir = self.dir / "raw"
        self._files: dict[str, BinaryIO] = {}
        self._index: dict[str, BinaryIO] = {}
        self._queue: asyncio.Queue[DataFrame | None] = asyncio.Queue(maxsize=queue_max)
        self._task: asyncio.Task | None = None
        self._dropped = 0
        self._written = 0

    @property
    def dropped(self) -> int:
        """Frames discarded because the writer could not keep up."""
        return self._dropped

    @property
    def written(self) -> int:
        return self._written

    async def start(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        for name in STREAMS:
            self._files[name] = open(self.raw_dir / f"{name}.f32", "ab")
            self._index[name] = open(self.raw_dir / f"{name}.idx", "ab")
        self._task = asyncio.create_task(self._drain(), name="rawlog-drain")

    def submit(self, frame: DataFrame) -> None:
        """Non-blocking. Drops the frame rather than stalling ingest."""
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            self._dropped += 1

    async def _drain(self) -> None:
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            try:
                # Off the event loop for real. Doing this inline would block
                # every session on the slowest disk write, which is exactly
                # what the queue exists to prevent.
                await asyncio.to_thread(self._write, frame)
                self._written += 1
            except Exception:  # a broken log must not take the session down
                self._dropped += 1

    def _write(self, frame: DataFrame) -> None:
        name = frame.stream.name
        self._files[name].write(
            np.ascontiguousarray(frame.samples, dtype=np.float32).tobytes()
        )
        # (first counter, sample count) per frame, so gaps survive replay
        self._index[name].write(
            np.array([frame.counter, frame.n_samples], dtype=np.uint32).tobytes()
        )

    def write_meta(self, meta: dict) -> None:
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def append_event(self, event: dict) -> None:
        with open(self.dir / "events.jsonl", "a") as fh:
            fh.write(json.dumps(event, separators=(",", ":")) + "\n")

    async def close(self) -> None:
        if self._task is not None:
            await self._queue.put(None)
            await self._task
            self._task = None
        for fh in (*self._files.values(), *self._index.values()):
            fh.flush()
            fh.close()
        self._files.clear()
        self._index.clear()


def read_stream(session_dir: Path, stream: StreamSpec) -> tuple[np.ndarray, np.ndarray]:
    """Read a logged stream back as (samples, per-frame index).

    The index is (n_frames, 2) of (first_counter, n_samples) — enough to
    reconstruct where the dropouts were.
    """
    raw = Path(session_dir) / "raw"
    samples = np.fromfile(raw / f"{stream.name}.f32", dtype=np.float32)
    samples = samples.reshape(-1, stream.n_channels)
    index = np.fromfile(raw / f"{stream.name}.idx", dtype=np.uint32).reshape(-1, 2)
    return samples, index
