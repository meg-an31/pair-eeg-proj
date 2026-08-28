"""Counter-indexed ring buffer that refuses to hide gaps.

A dropout that gets silently concatenated produces a step edge, and a step
edge is broadband power — it corrupts every band estimate that touches the
window. So the buffer tracks which sample positions were actually written
and reports gaps rather than papering over them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import StreamSpec


@dataclass(frozen=True)
class Window:
    """A contiguous span of sample positions pulled from the buffer."""

    start: int
    samples: np.ndarray  # (n_samples, n_channels), NaN where never written
    n_missing: int

    @property
    def end(self) -> int:
        return self.start + int(self.samples.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def complete(self) -> bool:
        return self.n_missing == 0

    @property
    def fill_ratio(self) -> float:
        total = self.n_samples
        return 1.0 if total == 0 else (total - self.n_missing) / total


class RingBuffer:
    """Fixed-capacity buffer addressed by absolute device sample counter.

    Writes may arrive out of order or with holes; both are handled. Writes
    older than the retained span are dropped, and a write far beyond the
    current head resets the buffer rather than spinning through the gap.
    """

    def __init__(self, stream: StreamSpec, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.stream = stream
        self.capacity = int(capacity)
        self._data = np.full((self.capacity, stream.n_channels), np.nan, dtype=np.float32)
        self._written = np.zeros(self.capacity, dtype=bool)
        self._head = 0          # one past the highest counter written
        self._origin: int | None = None

    # -- state ---------------------------------------------------------

    @property
    def head(self) -> int:
        return self._head

    @property
    def tail(self) -> int:
        """Oldest counter still retained."""
        return max(self._head - self.capacity, self._origin or 0)

    @property
    def empty(self) -> bool:
        return self._origin is None

    def available(self) -> int:
        return 0 if self.empty else self._head - self.tail

    # -- writing --------------------------------------------------------

    def write(self, counter: int, samples: np.ndarray) -> int:
        """Insert samples at absolute position `counter`.

        Returns the number of samples actually stored (0 if the whole write
        fell off the back of the buffer).
        """
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != self.stream.n_channels:
            raise ValueError(
                f"{self.stream.name}: expected (n, {self.stream.n_channels}), got {samples.shape}"
            )
        n = samples.shape[0]
        if n == 0:
            return 0

        if self._origin is None:
            self._origin = counter

        # A jump larger than the buffer means everything retained is stale.
        if counter >= self._head + self.capacity:
            self._data.fill(np.nan)
            self._written.fill(False)
            self._origin = counter

        end = counter + n
        # Advancing the head invalidates the slots the wrap is about to reuse.
        if end > self._head:
            self._invalidate(self._head, end)
            self._head = end

        # Trim anything already older than the retained span.
        keep_from = max(counter, self._head - self.capacity)
        if keep_from >= end:
            return 0
        offset = keep_from - counter
        n_kept = end - keep_from

        idx = np.arange(keep_from, end) % self.capacity
        self._data[idx] = samples[offset:]
        self._written[idx] = True
        return int(n_kept)

    def _invalidate(self, frm: int, to: int) -> None:
        """Mark slots between the old and new head as unwritten."""
        span = min(to - frm, self.capacity)
        if span <= 0:
            return
        idx = np.arange(to - span, to) % self.capacity
        self._written[idx] = False
        self._data[idx] = np.nan

    # -- reading ---------------------------------------------------------

    def read(self, start: int, n: int) -> Window:
        """Read `n` samples from absolute position `start`.

        Positions never written come back as NaN and are counted in
        `n_missing` — the caller decides whether that is tolerable.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        out = np.full((n, self.stream.n_channels), np.nan, dtype=np.float32)
        missing = n

        if not self.empty:
            lo = max(start, self.tail)
            hi = min(start + n, self._head)
            if hi > lo:
                idx = np.arange(lo, hi) % self.capacity
                valid = self._written[idx]
                dst = np.arange(lo - start, hi - start)
                out[dst[valid]] = self._data[idx[valid]]
                missing = n - int(valid.sum())

        return Window(start=start, samples=out, n_missing=missing)

    def latest(self, n: int) -> Window:
        """Read the most recent `n` samples."""
        return self.read(max(self._head - n, 0), n)
