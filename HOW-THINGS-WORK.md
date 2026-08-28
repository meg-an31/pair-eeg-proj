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
are also parked in a slot that `GET /latest` reads. Both stages are currently
null objects that return correctly shaped placeholders.

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
| `pair_eeg/pipeline/processing.py` | **BLANK SEAM 1** + `EpochWindow`, `ProcessedFeatures` |
| `pair_eeg/pipeline/affect.py` | **BLANK SEAM 2** + `AffectValues`, `Smoother` |
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
anything else.

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

---

## Worked example: adding heart rate and HRV from PPG

This is the most likely next extension, and it walks straight into the one
serious trap in the codebase. Read the whole section before writing code.

### The trap: PPG is not aligned to EEG

`Session._co_window()` produces the `ppg` array by scaling the EEG counter:

```python
ratio = stream.rate_hz / EEG.rate_hz
window = buf.read(int(eeg_start * ratio), ...)
```

That assumes **all streams share a counter origin**. They do not. `muse.js`
gives EEG and PPG independent `Counter` instances seeded from independent
16-bit hardware counters, so an EEG counter of 100000 and a PPG counter of 0
can be simultaneous. The current code will hand you an **all-`NaN` PPG array**
and nothing will look broken.

There is an `xfail` test recording this: `test_co_window_assumes_shared_counter_origins`.

**The fix** is to translate through each buffer's own origin rather than
scaling absolutely:

```python
ppg_start = ppg_buf.origin + (eeg_start - eeg_buf.origin) * ratio
```

`RingBuffer` tracks `_origin` privately; exposing it as a property is the
minimal change. Do that in `ringbuffer.py` and `session.py` rather than working
around it in your own stage — otherwise the next person hits it too.

### The second trap: HRV needs a longer window than an epoch

RMSSD needs **at least 30 s** of beat intervals to mean anything; LF/HF wants
closer to two minutes. A 4-second epoch is not a noisy version of the right
answer — it is not the quantity at all.

So do **not** compute HRV from `epoch.ppg`. Two options:

- **Widen the window.** `window_s` is config. A 30 s window gives you HRV in
  the same call as everything else, at the cost of every estimate describing
  the last 30 seconds.
- **Keep your own state.** Your processor is a long-lived object. Maintain a
  rolling inter-beat-interval deque across calls and recompute HRV every ~5 s,
  reporting it with its own window length and age. This is the design the
  architecture assumed, and it keeps the EEG axes responsive.

Prefer the second unless you have a reason not to.

### The third trap: the browser may not be sending PPG at all

`muse.js` subscribes to PPG characteristics `273e000f/0010/0011` inside a
`.catch(() => {})`, so a rejected preset or a wrong UUID fails **silently** —
no PPG, no error. The original implementation on the other branch requests
preset `p50` and then never subscribes at all.

The byte-level PPG decode in `muse.js` (24-bit big-endian triplets) and the IMU
scale factors have **never been verified against real hardware**. The EEG path
has a working precedent; these do not.

**Check this first, before writing any DSP.** With a headband connected:

```
GET /raw/ppg          # should show three channels of moving values
GET /raw/stats        # streams.ppg.available should be > 0
```

If `/raw/eeg` flows and `/raw/ppg` stays empty, fix the browser before
anything else. Debugging a beat detector against an empty array wastes a day.

### Suggested shape

1. Verify PPG actually arrives (`/raw/ppg`).
2. Fix `_co_window` origin translation.
3. Write `HeartRateProcessor` implementing `Processor`, keeping a rolling IBI
   buffer as instance state.
4. Bandpass PPG 0.7–3.5 Hz (42–210 bpm), causally. `scipy.signal.find_peaks`
   with `distance=int(0.4*fs)` matches the offline pipeline's parameters —
   reuse them so live and offline agree.
5. Put results in `extras`: `hr_bpm`, `hrv_rmssd_ms`, `hrv_window_s`,
   `hrv_age_s`, `n_beats`. Do not add dataclass fields.
6. In the affect stage, map RMSSD onto the `autonomic` axis, normalised against
   the resting block. Report low `confidence` when `n_beats` is small.
7. The front end should render `autonomic` differently from the fast axes —
   it updates on a different clock and pretending otherwise misleads.

### Where the resting block fits

`RestingBaseline` currently stores **EEG only** (`resting.eeg`). If you want a
resting HRV reference you will need to extend it to carry PPG, which means
touching `resting.py` (the collector, the `npz` save, the loader) — a real
change, so do it deliberately and keep the save format backward-compatible:
`RestingStore.load()` treats a missing key as a cache miss, not a crash.

---

## Known bugs and limitations

Recorded so you do not rediscover them. The `xfail` tests are the live record.

| Issue | Where | Impact |
|---|---|---|
| `_co_window` assumes shared counter origins | `session.py` | **PPG/IMU arrive as all-NaN.** Blocks HR work. |
| Baseline block timed by wall clock, not accepted data | `session.py:_update_state` | 120 s of sitting still yields ~72–96 s of samples. `RestingCollector.complete` exists and is never read. |
| Backfill below the first counter ever seen is unreadable | `ringbuffer.py` | `write()` reports success for data no reader can see. Rare. |
| PPG/IMU decode unverified against hardware | `muse.js` | Silent failure — wrong UUID or preset yields nothing, no error. |
| `DEGRADED` entry requires 20 s uninterrupted poor ticks | `session.py` | A rate oscillating around the threshold never enters it. |
| No authentication | everywhere | Anyone with the URL can claim the capture slot. |
| `RestingBaseline` holds EEG only | `resting.py` | No resting reference for PPG/HRV. |

---

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/python -m pair_eeg.transport.raw_api          # server + /raw routes
PYTHONPATH=. .venv/bin/python -m clients.synthetic --baseline   # fake headband
.venv/bin/python -m pytest tests/ -q
```

`clients/synthetic.py` speaks the real wire protocol, so if it works and the
browser does not, the fault is in the browser. `--drop-at 20 --drop-for 5`
simulates a dropout and is the fastest way to exercise gap handling.

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
