import numpy as np
import pytest

from pair_eeg.config import EEG, StreamSpec
from pair_eeg.pipeline.ringbuffer import RingBuffer

S = StreamSpec("t", 9, 256.0, ("a", "b"))


def mk(cap=100):
    return RingBuffer(S, cap)


def blk(start, n):
    """Samples whose value encodes the absolute counter."""
    c = np.arange(start, start + n, dtype=np.float32)
    return np.stack([c, c + 0.5], axis=1)


def check_identity(win):
    """Every non-NaN row must equal its own absolute counter."""
    for i in range(win.n_samples):
        v = win.samples[i, 0]
        if not np.isnan(v):
            assert v == win.start + i, f"row {i}: stale {v} at counter {win.start+i}"


# --- basic ----------------------------------------------------------------

def test_in_order():
    b = mk()
    for c in range(0, 90, 10):
        assert b.write(c, blk(c, 10)) == 10
    assert b.head == 90 and b.tail == 0 and b.available() == 90
    w = b.read(0, 90)
    assert w.n_missing == 0 and w.fill_ratio == 1.0
    check_identity(w)


def test_out_of_order():
    b = mk()
    b.write(0, blk(0, 10))       # sets _origin = 0
    b.write(20, blk(20, 10))
    b.write(10, blk(10, 10))     # backfill into the hole
    assert b.head == 30
    w = b.read(0, 30)
    assert w.n_missing == 0
    check_identity(w)


@pytest.mark.xfail(strict=True, reason=
    "BUG: RingBuffer.tail floors at _origin, so a frame arriving below the first counter ever written is stored but unreadable (ringbuffer.py:71).")
def test_backfill_below_origin_is_lost():
    """BUG: `tail` floors at `_origin`, the first counter ever written, so a
    frame that arrives out of order *below* it is stored but unreadable."""
    b = mk()
    b.write(20, blk(20, 10))     # _origin = 20
    stored = b.write(0, blk(0, 10))
    assert stored == 10          # write claims success
    assert b._written[0:10].all()  # and the bytes really are in the array
    w = b.read(0, 30)
    assert w.n_missing == 10, "10..20 is the only real hole"


def test_gap():
    b = mk()
    b.write(0, blk(0, 10))
    b.write(50, blk(50, 10))
    w = b.read(0, 60)
    assert w.n_missing == 40
    check_identity(w)


def test_jump_beyond_capacity_resets():
    b = mk(100)
    b.write(0, blk(0, 50))
    b.write(1000, blk(1000, 10))
    assert b.head == 1010
    assert b.read(0, 50).n_missing == 50           # old data gone
    w = b.read(1000, 10)
    assert w.n_missing == 0
    check_identity(w)


def test_wraparound():
    b = mk(100)
    for c in range(0, 500, 10):
        b.write(c, blk(c, 10))
    assert b.head == 500 and b.tail == 400
    w = b.read(400, 100)
    assert w.n_missing == 0
    check_identity(w)
    # everything before the tail must read as missing, never stale
    old = b.read(300, 100)
    assert old.n_missing == 100
    assert np.all(np.isnan(old.samples))


def test_read_straddling_tail():
    b = mk(100)
    for c in range(0, 300, 10):
        b.write(c, blk(c, 10))
    w = b.read(150, 100)          # 150..250, tail is 200
    check_identity(w)
    assert w.n_missing == 50


def test_read_beyond_head():
    b = mk()
    b.write(0, blk(0, 10))
    w = b.read(5, 20)             # 5..25, head is 10
    check_identity(w)
    assert w.n_missing == 15


def test_read_entirely_before_tail():
    b = mk(100)
    for c in range(0, 400, 10):
        b.write(c, blk(c, 10))
    w = b.read(0, 50)
    assert w.n_missing == 50 and np.all(np.isnan(w.samples))


def test_write_entirely_before_tail_dropped():
    b = mk(100)
    for c in range(0, 300, 10):
        b.write(c, blk(c, 10))
    assert b.write(0, blk(0, 10)) == 0
    check_identity(b.read(200, 100))


def test_write_partially_before_tail():
    b = mk(100)
    for c in range(0, 300, 10):
        b.write(c, blk(c, 10))
    assert b.write(190, blk(190, 20)) == 10       # only 200..210 kept
    check_identity(b.read(200, 100))


def test_invalidate_no_stale_after_head_advance():
    """Advance the head over old slots without writing them; the aliased
    slots must not report the previous generation's data."""
    b = mk(100)
    b.write(0, blk(0, 100))
    b.write(150, blk(150, 10))    # head 160, slots 100..150 aliased to 0..50
    w = b.read(60, 100)           # 60..160, tail=60
    check_identity(w)
    # 60..100 survive (still inside the retained span), 150..160 are new
    assert w.n_missing == 50


def test_exact_capacity_boundary_jump():
    b = mk(100)
    b.write(0, blk(0, 10))
    b.write(110, blk(110, 5))     # 110 >= head(10)+100 -> reset
    check_identity(b.read(105, 20))
    assert b.read(0, 10).n_missing == 10


def test_one_below_reset_threshold():
    b = mk(100)
    b.write(0, blk(0, 10))
    b.write(109, blk(109, 5))     # 109 < 10+100 -> no reset
    assert b.head == 114
    check_identity(b.read(14, 100))
    assert b.read(0, 10).n_missing == 10   # trimmed by the tail


def test_write_longer_than_capacity():
    b = mk(100)
    assert b.write(0, blk(0, 250)) == 100
    assert b.head == 250 and b.tail == 150
    w = b.read(150, 100)
    assert w.n_missing == 0
    check_identity(w)


def test_latest():
    b = mk(100)
    b.write(0, blk(0, 60))
    w = b.latest(20)
    assert w.start == 40 and w.n_missing == 0
    check_identity(w)


def test_latest_shorter_than_head():
    b = mk(100)
    b.write(0, blk(0, 5))
    w = b.latest(20)
    assert w.start == 0 and w.n_samples == 20 and w.n_missing == 15


def test_fill_ratio():
    b = mk()
    b.write(0, blk(0, 10))
    b.write(20, blk(20, 10))
    w = b.read(0, 40)
    assert w.n_missing == 20
    assert w.fill_ratio == pytest.approx(0.5)


def test_wrong_shape_rejected():
    b = mk()
    with pytest.raises(ValueError):
        b.write(0, np.zeros((10, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        b.write(0, np.zeros(10, dtype=np.float32))


def test_zero_samples():
    b = mk()
    assert b.write(0, np.zeros((0, 2), np.float32)) == 0
    assert b.empty


def test_read_zero_raises():
    b = mk()
    with pytest.raises(ValueError):
        b.read(0, 0)


def test_nonzero_origin():
    b = mk(100)
    b.write(1_000_000, blk(1_000_000, 10))
    assert b.tail == 1_000_000 and b.available() == 10
    check_identity(b.read(1_000_000, 10))


# --- adversarial ----------------------------------------------------------

def test_counter_restart_backwards():
    """Device reconnects and its counter restarts at 0.

    Recovery costs the first few blocks: one write far behind the head is a
    late frame and must be dropped, so a restart is only recognised once it
    persists. Losing a little data to that ambiguity is the right trade —
    treating every late frame as a restart would let a straggler wipe the
    buffer. What matters is that the stream recovers and the loss shows up
    as a gap rather than as silently wrong samples.
    """
    b = mk(100)
    for c in range(0, 300, 10):
        b.write(c, blk(c, 10))

    for c in range(0, 100, 10):
        b.write(c, blk(c, 10))

    assert b.head <= 100, f"never followed the restart (head={b.head})"

    # Everything after the confirmation delay is intact.
    tail = b.read(50, 50)
    assert tail.n_missing == 0, "did not recover after the restart was confirmed"


def test_randomised_no_stale_reads():
    rng = np.random.default_rng(0)
    b = mk(64)
    c = 0
    for _ in range(2000):
        n = int(rng.integers(1, 20))
        jitter = int(rng.integers(-30, 5))
        pos = max(0, c + jitter)
        b.write(pos, blk(pos, n))
        c = max(c, pos + n)
        lo = max(0, c - int(rng.integers(1, 120)))
        check_identity(b.read(lo, int(rng.integers(1, 100))))


def test_counter_restart_recovers_after_a_run():
    """A reconnected device restarts its counter; the buffer must follow.

    One far-behind write is a late frame and is dropped. A sustained run of
    them is a restart, and continuing to drop would stall the stream forever.
    """
    from pair_eeg.config import EEG
    from pair_eeg.pipeline.ringbuffer import RingBuffer
    import numpy as np

    buf = RingBuffer(EEG, 1024)
    buf.write(500_000, np.ones((256, 4), dtype=np.float32))
    assert buf.head == 500_256

    # Device reconnects and starts again from zero.
    for i in range(RingBuffer.RESTART_AFTER):
        buf.write(i * 256, np.full((256, 4), 7.0, dtype=np.float32))

    assert buf.head <= 1024, "buffer should have followed the restart"
    window = buf.latest(256)
    assert window.n_missing == 0
    assert float(np.nanmax(window.samples)) == 7.0
