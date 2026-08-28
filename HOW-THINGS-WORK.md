# How things work

Written for whoever extends this next, human or model. It covers the shape of
the system, the decisions that are load-bearing, and the traps that are not
obvious from reading the code.

If you only read one section, read **Invariants**. Most of them exist because
breaking them produces numbers that look plausible and are wrong, which is
worse than a crash.

---

## The one-paragraph model

A Muse headband streams four EEG channels over Bluetooth to a browser. The
browser decodes packets and forwards **raw samples** to a Python server. The
server buffers them, cuts overlapping windows, and passes each window through
two swappable stages — **processing** (samples → spectra and band powers) and
**affect** (features → five 0–1 numbers). Results go out over a WebSocket and
are also parked in a slot that `GET /latest` reads.

Both stages are now filled: `spectral.py` computes Welch spectra, band powers
and four EEG features, `pulse.py` tracks heart rate and HRV from PPG on its own
slower clock, and `mapper.py` turns all of that into distance-from-resting
axes. The null objects are still there and still the right choice when
debugging transport rather than DSP — `python -m pair_eeg --null`.

```
Muse ──BLE──> browser ──wss──> server ──> [processing] ──> [affect] ──> browser
                                  │                                        ▲
                                  └── raw log to disk                      │
                                  └── latest slot ──── GET /latest ────────┘
```

---

## File map

| File | What it owns |
|---|---|
| `pair_eeg/config.py` | Stream specs (rates, channel names), tunables |
| `pair_eeg/transport/protocol.py` | Wire format: 20-byte binary header, JSON control |
| `pair_eeg/transport/server.py` | WebSocket hub, capture gate, HTTP polling routes |
| `pair_eeg/transport/raw_api.py` | `/raw/*` inspection routes + its own entry point |
| `pair_eeg/pipeline/ringbuffer.py` | Counter-indexed, gap-aware buffer |
| `pair_eeg/pipeline/quality.py` | Amplitude/contact gating, accept-rate tracking |
| `pair_eeg/pipeline/processing.py` | **SEAM 1** interface + `EpochWindow`, `ProcessedFeatures` |
| `pair_eeg/pipeline/affect.py` | **SEAM 2** interface + `AffectValues`, `Smoother` |
| `pair_eeg/pipeline/spectral.py` | Fills seam 1: gap-tolerant Welch, band integration, EEG features |
| `pair_eeg/pipeline/pulse.py` | Heart rate and HRV from PPG — the slow lane, own rolling state |
| `pair_eeg/pipeline/mapper.py` | Fills seam 2: features → five 0–1 axes relative to resting |
| `pair_eeg/pipeline/resting.py` | The two-minute wall-staring block: collect, persist, restore |
| `pair_eeg/pipeline/session.py` | State machine, epoch loop, payload construction |
| `pair_eeg/pipeline/rawlog.py` | Append-only capture to disk |
| `pair_eeg/pipeline/store.py` | Session directories, SQLite index |
| `web/muse.js` | BLE: connect, decode, reconnect, watchdog |
| `web/uplink.js` | Batching, spooling, framing to the server |
| `web/index.html` | Capture page · `web/view.html` viewer · `web/raw.html` inspector |
| `clients/synthetic.py` | Fake headband — develop without hardware |

---

## The pipeline, in order

**1. Browser decodes BLE.** Each EEG notification is a 16-bit packet counter
plus twelve 12-bit samples, on four separate characteristics. `muse.js` unwraps
the counter past its ~51-minute rollover, aligns the four channels into rows by
absolute sample index, and sample-and-holds a channel that falls more than 96
samples behind.

**2. Uplink batches and sends.** ~100 ms per frame, binary, with the sample
counter of the first sample in the header. While the socket is down, frames
spool in memory and replay on reconnect **with their original counters**.

**3. Server ingests.** `Session.ingest()` writes to the raw log *first*, then
into a per-stream `RingBuffer` keyed by absolute counter.

**4. Epoch loop.** Every `hop_s` (1 s), cut the newest `window_s` (4 s) of EEG.
Windows overlap 75%.

**5. Quality gate.** Per-channel RMS and peak thresholds. Rejected epochs emit
a quality-only payload and no estimate. Losing 20–40% is the normal operating
point for dry electrodes, not a fault.

**6. Processing stage.** `process(epoch, resting) -> ProcessedFeatures`.

**7. Affect stage.** `map(features, calibrated, resting) -> AffectValues`.

**8. Smoothing, payload, fan-out.** EMA over ~5 windows, then the payload is
stored in `session.latest` and broadcast to the capture client and all viewers.

---

## The two seams

Everything you plug in goes through one of these. Neither requires touching
anything else. Both currently hold a real implementation; the interfaces below
are what a replacement has to satisfy, and `spectral.py` / `mapper.py` are
worked examples of satisfying them.

### Processing — `pair_eeg/pipeline/processing.py`

```python
def process(self, epoch: EpochWindow, resting: RestingBaseline | None = None
            ) -> ProcessedFeatures: ...
```

`EpochWindow` gives you:

| Field | Shape / type | Notes |
|---|---|---|
| `eeg` | `(1024, 4)` float32 | **microvolts**, unfiltered, `NaN` where a sample never arrived |
| `ppg` | `(256, 3)` or `None` | ambient / IR / red, raw counts |
| `imu` | `(208, 6)` or `None` | ax ay az gx gy gz |
| `fs` | `256.0` | |
| `channels` | `("TP9","AF7","AF8","TP10")` | column order of `eeg` |
| `counter` | int | device sample index of the first row |
| `t` | float | seconds since session start |

You must return `spectrum` of `(4, 128)` and `bands` of `(4, 5)`.
`check_shapes()` runs automatically and rejects the wrong size.

Anything else you compute goes in `extras: dict[str, float]` — that dict is
what travels to the affect stage and what the baseline accumulates statistics
over. Adding a feature means adding a key, not changing a dataclass.

### Affect — `pair_eeg/pipeline/affect.py`

```python
def map(self, features: ProcessedFeatures, calibrated: bool,
        resting: RestingBaseline | None = None) -> AffectValues: ...
```

Return five axes — `valence`, `arousal`, `direction`, `engagement`,
`autonomic` — every one in `[0, 1]`. `AffectValues.__post_init__` enforces the
range on both `axes` and `confidence` and will raise if you exceed it.

### Wiring one in

```python
from pair_eeg.transport import raw_api
raw_api.run(config, processor=MyProcessor(), affect=MyMapper())
```

`Processor` and `AffectMapper` are `typing.Protocol` — no base class to
inherit, just match the signature and expose a `name` attribute.

---

## Invariants

Break these and things go wrong quietly.

**1. The device sample counter is the clock. Wall time is not.**
Every window boundary and every gap is computed from counters. `t_client` in
the frame header exists only for drift estimation. BLE delivers in jittery
bursts, so arrival time tells you almost nothing about when a sample was taken.

**2. Never concatenate across a gap.**
A spliced dropout is a step edge, and an FFT reads a step edge as energy at
*every* frequency — one dropped packet manufactures a convincing burst of
gamma. `RingBuffer` returns `NaN` for positions never written and reports
`n_missing`. Decide explicitly what to do with holes; do not zero-fill.

**3. Raw goes to disk before anything reads it.**
Features are re-derivable, samples are not. `RawLog.submit()` is called at the
top of `ingest()`, and it queues rather than writing inline so a slow disk
degrades the recording, not the estimate.

**4. Filtering must be causal.**
`filtfilt` / `sosfiltfilt` need the whole recording and cannot stream. Use
`sosfilt` with retained `zi` state. (The offline pipeline on the
`muse-open-source-pipeline` branch uses `filtfilt`, so live and offline numbers
will not match exactly. That is expected — do not "fix" it by going zero-phase.)

There is a second half to this that is easy to get wrong. Retained filter state
is only correct for a stage that sees each sample **once**, and the epoch loop
hands out windows overlapping by 75%: a processor that kept `zi` across calls
would push every sample through the filter four times and its state would mean
nothing. So `spectral.py` does no time-domain filtering at all — a periodogram
over a finite window is causal by construction — and `pulse.py`, which genuinely
needs the time domain for peak detection, keeps `zi` and is fed only each hop's
new samples. Either discipline is fine. Mixing them is not.

**5. 128 bins means 0–127 Hz, and Nyquist is dropped.**
`scipy.signal.welch(fs=256, nperseg=256)` returns **129** bins spanning
0–128 Hz. We ship 128. Discard the last bin explicitly. `check_shapes()`
catches this and says so.

**6. Never fake movement.**
`NullAffectMapper` returns flat 0.5 with `implemented=False` rather than
drifting values, because a mapper that moved would look like it worked and the
front end would be animating noise. If your stage cannot produce a real number,
say so in the flag rather than inventing one.

**7. All axes are 0–1, and 0.5 is the wearer's own resting level.**
Not a global constant, not a population norm. Absolute band powers shift with
skull, hair, sweat and how the band sat that morning, so almost every useful
number is a ratio against — or distance from — the resting block.

**8. `calibrated` means a resting block exists.** Nothing else. It used to mean
something weaker and that was a bug.

**9. The latest slot is overwritten, never queued.**
If you queue results per reader, a slow poller falls progressively further
behind forever. Overwriting means it skips values and always sees now. Missing
intermediate values is correct here; lagging is not.

**10. NaN and Infinity must not reach the wire.**
They serialise as bare `NaN` / `Infinity` tokens, which are invalid JSON —
`JSON.parse` throws away the *whole message*, not the bad field. `encode_payload`
sanitises to `null`. Holes in `/raw/*` are `null`, never `0`: a gap and a
genuine zero are different measurements.

**11. Every payload variant carries the same keys.**
Quality-only, baseline and live payloads all have the full key set with `null`
for what does not apply, so a front end can rely on field presence. Rejected
epochs are 20–40% of them; a client should not have to branch on which shape
arrived.

**12. Do not block the event loop.**
File and SQLite writes go through `asyncio.to_thread` or a queue. The epoch
loop and the fan-out share a loop with ingest.

**13. One capture client, many viewers.**
The headband allows one BLE connection, so a second streamer can only be a
mistake. Viewers are read-only: they cannot send sensor data and cannot control
the session.

**14. Overlapping windows double-count.**
Windows overlap 75%. `RestingCollector.observe()` takes only the newest *hop*
of each epoch — taking the whole window would count most samples four times and
weight the middle of a block far above its edges. Any future accumulator over
epochs has the same trap.

**15. Resting statistics belong to a window length.**
A feature's spread depends on how much data went into it, so statistics
gathered over 4 s windows do not describe the same feature measured over 8 s
ones — the axis would be scaled by a factor nobody chose. `window_s` therefore
travels in `extras` and `mapper` derives its resting statistics over windows of
the same length, re-deriving them if it changes. Any future normalisation over
the resting block inherits this.

---

## Worked example: heart rate and HRV from PPG

This was the most likely next extension and it is now built — `pulse.py`, wired
in as a collaborator of `SpectralProcessor`. Kept here because both traps it
walks into are still live for anything else that touches PPG or IMU, and
because the third one has never been closed.

### Trap one: PPG was not aligned to EEG (fixed)

`Session._co_window()` used to build the `ppg` array by scaling the EEG counter:

```python
ratio = stream.rate_hz / EEG.rate_hz
window = buf.read(int(eeg_start * ratio), ...)
```

which assumes all streams share a counter origin. They do not: `muse.js` gives
EEG and PPG independent `Counter` instances seeded from independent 16-bit
hardware counters, so an EEG counter of 100000 and a PPG counter of 0 can be
simultaneous. The symptom was an all-`NaN` PPG array and nothing looking broken.

Now translated through each buffer's own origin, with `RingBuffer.origin`
exposed for it:

```python
start = ppg_buf.origin + (eeg_start - eeg_buf.origin) * ratio
```

The `xfail` that recorded the bug is now
`test_co_window_translates_through_stream_origins`.

**What this assumes, and what is left.** That the streams *started* together,
which the capture client guarantees by subscribing to everything before it
sends anything. Two residual drifts: a stream that reconnects and restarts its
counter mid-session (the offset is recomputed every call, so it self-corrects
at the next read), and the difference between a stream's declared rate and its
true one, which walks the window off by a sample every ten seconds per 0.1 Hz
of error — about forty minutes before a 4 s window misses entirely. If PPG goes
quiet on a long recording, suspect this before the electrodes.

`clients/synthetic.py` now streams PPG on a **deliberately different** counter
origin, so this stays exercised with no hardware in the room. Fixing it also
turned up that the client was truncating its batch sizes — `int(256 * 0.1)` is
25, not 25.6 — so each stream's counter advanced at its own wrong rate and the
two drifted apart in seconds. `batch_size()` differences a running total
instead.

### Trap two: HRV needs a longer window than an epoch (handled by state)

RMSSD needs **at least 30 s** of beat intervals to mean anything; LF/HF wants
closer to two minutes. A 4-second epoch is not a noisy version of the right
answer — it is not the quantity at all.

Of the two options — widen `window_s` for everyone, or keep rolling state in
the processor — `pulse.py` takes the second, so the EEG axes stay responsive.
It buffers the trailing 60 s of filtered PPG, re-detects beats across the whole
buffer on each refresh (~5 s), and publishes `hr_bpm`, `hrv_rmssd_ms`,
`hrv_window_s`, `hrv_age_s` and `n_beats` in `extras`. Re-detecting rather than
tracking peaks incrementally sidesteps the chunk-boundary case where a peak's
neighbourhood is split across two calls and is either missed or counted twice.

Three things there are load-bearing:

- **It is fed each hop's new samples only.** Windows overlap 75%, so feeding
  whole windows would present every heartbeat four times (Invariant 14). The
  new span is inferred from how far the epoch counter moved.
- **RMSSD is withheld until 20 intervals spanning 30 s exist.** Reporting it
  early would produce a number the affect stage cannot distinguish from a
  settled one.
- **The reading carries its own age.** It describes the last minute, refreshed
  every few seconds. A front end animating it at the rate of the fast axes is
  lying about how quickly it can move.

### Trap three: the browser may not be sending PPG at all (still open)

`muse.js` subscribes to PPG characteristics `273e000f/0010/0011` inside a
`.catch(() => {})`, so a rejected preset or a wrong UUID fails **silently** —
no PPG, no error. The byte-level PPG decode (24-bit big-endian triplets) and
the IMU scale factors have **never been verified against real hardware**, and
nothing added here changes that: everything above was built and tested against
the synthetic client, which speaks the wire protocol but cannot vouch for the
browser's decode.

**So this is still the first thing to check with a headband on.** With one
connected:

```
GET /raw/ppg          # should show three channels of moving values
GET /raw/stats        # streams.ppg.available should be > 0
```

If `/raw/eeg` flows and `/raw/ppg` stays empty, fix the browser before
anything else — the server side is ready and will simply report `autonomic` as
unavailable, which looks identical to a wearer with a calm heart.

### Where the resting block fits

`RestingBaseline` still stores **EEG only** (`resting.eeg`), so there is no raw
resting PPG to derive a spread from. The two axes are normalised differently
because of it, and this is the sharpest remaining rough edge:

- **EEG features** are z-scored against statistics recomputed from the resting
  block's raw samples (`spectral.resting_extras`) — a median and a MAD.
- **HRV** has only the *mean* resting RMSSD, which arrives free: the processor
  puts `hrv_rmssd_ms` in `extras`, `Baseline.observe` accumulates it, and
  `_finish_resting` copies the means into `resting.features`, which is
  persisted in `resting.json` and restored across sessions. With no spread
  available, `mapper._autonomic` uses a log-ratio against that mean with a
  documented guessed scale rather than pretending to a z-score.

Giving HRV a real spread means extending `RestingBaseline` to carry PPG — the
collector, the `npz` save and the loader. Keep the save format
backward-compatible: `RestingStore.load()` treats a missing key as a cache
miss, not a crash.

One consequence of optional features worth knowing: **a resting mean is only as
good as the number of ticks that reported it.** `hrv_rmssd_ms` appears late in
the baseline block, so a baseline shorter than ~90 s may freeze without it at
all, and the `autonomic` axis then correctly reports itself unavailable. This
is also what surfaced the `Baseline` bug fixed below.

## Known bugs and limitations

Recorded so you do not rediscover them. The `xfail` tests are the live record.

| Issue | Where | Impact |
|---|---|---|
| Baseline block timed by wall clock, not accepted data | `session.py:_update_state` | 120 s of sitting still yields ~72–96 s of samples. `RestingCollector.complete` exists and is never read. Tests rewind `_baseline_started` rather than sit through it. |
| Backfill below the first counter ever seen is unreadable | `ringbuffer.py` | `write()` reports success for data no reader can see. Rare. |
| PPG/IMU decode unverified against hardware | `muse.js` | Silent failure — wrong UUID or preset yields nothing, no error. Server side is ready and waiting on this. |
| `DEGRADED` entry requires 20 s uninterrupted poor ticks | `session.py` | A rate oscillating around the threshold never enters it. |
| No authentication | everywhere | Anyone with the URL can claim the capture slot. |
| `RestingBaseline` holds EEG only | `resting.py` | No resting *spread* for HRV, so `autonomic` uses a log-ratio against the resting mean with a guessed scale. |
| Co-window alignment drifts on declared-vs-true rate error | `session.py:_co_window` | ~40 min before a 4 s PPG window misses entirely. Looks like an unplugged sensor. |
| IMU is co-windowed but nothing consumes it | `session.py` | No motion gating yet; `epoch.imu` is there for whoever adds it. |

Fixed since this document was first written, kept here because the reasoning
still matters:

| Was | Where | Now |
|---|---|---|
| `_co_window` assumed a shared counter origin, so PPG/IMU arrived as all-NaN | `session.py` | Translated through each buffer's own `origin`. `xfail` flipped to `test_co_window_translates_through_stream_origins`. |
| `Baseline` divided every feature's sum by the tick count | `session.py:Baseline.freeze` | Divides by the ticks that actually reported that feature. Features are optional by design — a stage omits one rather than inventing it — so a key present in 3 of 60 ticks was coming out at a twentieth of its value, and anything normalised against it was confidently wrong. |
| `clients/synthetic.py` truncated batch sizes, so each stream's counter ran at its own wrong rate | `clients/synthetic.py` | `batch_size()` differences a running total; the average matches the declared rate exactly. |
| A baseline block that ended early still overwrote a good stored resting recording | `session.py:_finish_resting` | Refuses to adopt or save a block under `MIN_RESTING_S` (12 s — three 4 s windows, the least that can describe a spread). The old behaviour was silent and permanent: every axis afterwards read 0.5 with zero confidence, which is indistinguishable from an unimplemented stage. |

---

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/python -m pair_eeg                            # real stages
.venv/bin/python -m pair_eeg --null                     # placeholder stages
.venv/bin/python -m pair_eeg.transport.raw_api          # + /raw routes
PYTHONPATH=. .venv/bin/python -m clients.synthetic --baseline   # fake headband
.venv/bin/python -m pytest tests/ -q
```

scipy is a dependency now (Welch, the pulse filter, peak detection).

`clients/synthetic.py` speaks the real wire protocol, so if it works and the
browser does not, the fault is in the browser. It streams EEG and PPG on
independent counter origins, as the hardware does. `--drop-at 20 --drop-for 5`
simulates a dropout and is the fastest way to exercise gap handling; `--hr 95`
sets the fake heart rate and `--no-ppg` streams EEG alone.

A note on watching it work: with `--baseline`, the axes stay at 0.5 with zero
confidence until the resting block freezes, because there is nothing to be
relative to before then (Invariant 6, and `mapper.py`). `autonomic` needs
longer still — a baseline of at least ~90 s before a resting RMSSD exists to
compare against. Both are correct behaviour, and both look like a broken
pipeline if you do not expect them.

With hardware: `cd web && python3 -m http.server 8000`, then
<http://localhost:8000>. Web Bluetooth needs a secure context (localhost or
https) and works only in Chrome, Edge and Opera. **Close the Muse phone app and
unpair the headband in system Bluetooth settings first** — it allows one
connection and the app will hold it.

Live deployment: see `DEPLOY.md`.

---

## Conventions

- **Comments explain why, not what.** If a line needs explaining, it is usually
  because a plausible alternative is wrong; say which and why.
- **Failures are loud or flagged, never silent.** A stage that cannot produce a
  number sets `implemented=False`. An exception in the epoch loop is caught,
  logged and counted, not swallowed.
- **Tests that document bugs use `xfail(strict=True)`** with a `file:line`
  reason, so they flip to failures the moment someone fixes the bug.
- **New capability, new file.** The `/raw` API composes the existing hub rather
  than editing it. Prefer that when several people are working at once.
