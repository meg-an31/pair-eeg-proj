# pair-eeg — streaming backend

    Muse 2 --BLE--> browser --wss--> server --> [processing] --> [affect] --> browser

The whole chain exists, both bracketed stages included: Welch spectra and band
powers into five 0-1 axes measured against the wearer's own resting recording,
plus heart rate and HRV from PPG on a slower clock. The original null objects
are still shipped and still useful — `--null` runs them, producing correctly
shaped output with `implemented: false`, which is what you want when debugging
transport rather than DSP.

## Run it without a headband

    python -m venv .venv && .venv/bin/pip install -e ".[dev]"
    .venv/bin/python -m pair_eeg                       # terminal 1
    PYTHONPATH=. .venv/bin/python -m clients.synthetic --baseline   # terminal 2

`clients/synthetic.py` speaks the same wire protocol the browser does, so if
it works and the browser does not, the fault is in the browser. It streams EEG
and PPG on independent counter origins, as the hardware does. It can simulate a
dropout (`--drop-at 20 --drop-for 5`) to exercise gap handling, and `--hr 95`
sets the fake heart rate.

Expect the axes to sit at 0.5 with zero confidence until the resting block
finishes: 0.5 *means* "this wearer at rest", so there is nothing to report
before one exists. `autonomic` needs a baseline of ~90 s or more, since a
resting HRV has to accumulate first.

## Run it with a headband

    cd web && python3 -m http.server 8000

Then open <http://localhost:8000>. Web Bluetooth needs a secure context, so
`localhost` or https — `file://` will not work. Chrome, Edge or Opera only;
Safari and Firefox do not implement it.

**The Muse allows one connection at a time.** Close the Muse phone app, unpair
the headband in system Bluetooth settings, and close other tabs holding it.

## The two stages

| Stage | Interface | Implementation |
|---|---|---|
| Processing | `pipeline/processing.py` — `EpochWindow` → `ProcessedFeatures` | `pipeline/spectral.py` (+ `pipeline/pulse.py`) |
| Affect | `pipeline/affect.py` — `ProcessedFeatures` → `AffectValues` | `pipeline/mapper.py` |

To replace either, write a class satisfying the `Processor` / `AffectMapper`
protocol and pass it to `transport.server.run()`. Nothing else changes.

The five axes, and what each is read from:

| Axis | From |
|---|---|
| `valence` | frown-muscle amplitude, 55-110 Hz on the frontal pair (negated) |
| `arousal` | frontal beta/alpha |
| `direction` | frontal alpha asymmetry — withdrawal to approach |
| `engagement` | beta/(alpha+theta), whole head |
| `autonomic` | HRV (RMSSD) over the trailing 60 s, from PPG |

Every one is a distance from the wearer's resting distribution squashed into
0-1, so 0.5 is that wearer at rest and nothing else. Where a reading cannot be
placed — no resting block, a detached electrode, too few heartbeats — the axis
stays at 0.5 and its confidence goes to 0 rather than the needle moving.

Output shapes, fixed by the hardware:

    spectrum  (n_channels, 128)   0-127 Hz at 1 Hz spacing
    bands     (n_channels, 5)     delta theta alpha beta gamma
    axes      5 values, all 0-1

128 bins at 1 Hz follows from the sample rate. The Muse streams EEG at 256 Hz,
so Nyquist is 128 Hz, and a 256-sample Welch segment gives 1 Hz spacing.
Running Welch with `nperseg=256` across a longer window keeps the 128 bins
while averaging several segments — variance drops, resolution stays put.

Two things the real processor must do that the existing offline pipeline does
not: filter **causally** (`sosfilt` with retained state, not `filtfilt`, which
needs the whole recording), and define band integration **once** — there are
currently two implementations that disagree.

## Why the server sends numbers, not an emotion word

Valence and arousal alone cannot separate anger from fear; both are negative
and high-arousal. What separates them is motivational direction, approach vs
withdrawal, which is a third quantity. Collapsing to a label server-side
discards it and makes the guess unauditable. So the server ships coordinates
and the front end renders them.

All axes are bounded 0-1 by contract (`AffectValues.__post_init__` enforces
it), with **0.5 meaning the wearer's own resting baseline**. That convention is
why the numbers mean little until a baseline exists — before then the payload
carries `calibrated: false`.

## Session lifecycle

    connecting -> fit_check -> baseline (120 s) -> live
                                     |               |
                                     +--- degraded <-+

State is part of the protocol, not internal bookkeeping: nothing is
interpretable before a baseline, so the front end must know which phase it is
in to render honestly.

`degraded` exists because losing 20-40% of epochs is the normal operating point
for dry electrodes. One rejected epoch is not a state change — it just produces
no estimate that tick. Sustained rejection is, and the interface has to tell
"calm" from "not currently measuring".

## Recording layout

    sessions/<wearer>/<session_id>/
      raw/{eeg,ppg,imu,therm}.f32   precious
      raw/*.idx                     frame counters, so gaps survive replay
      meta.json
      events.jsonl                  markers, stimuli, self-report
      baseline.json                 written the moment stats freeze
      derived/                      cache, safe to delete

Raw is logged before anything reads it, because features are re-derivable and
samples are not. Writes are queued and drained off the ingest path, so a slow
disk degrades the recording rather than the estimate.

## Gap handling

The device sample counter is the clock, not wall time. Both the browser and the
server place samples by counter and mark discontinuities rather than
concatenating across them — a spliced dropout is a step edge, and an FFT reads a
step edge as energy at every frequency, which manufactures a convincing burst of
gamma out of nothing.

The browser spools frames in memory while the socket is down and replays them
with their original counters, so a network dropout becomes a marked gap.

## Layout

    pair_eeg/
      config.py               stream specs, tunables
      transport/protocol.py   20-byte binary header + JSON control
      transport/server.py     websocket hub, session fan-out
      pipeline/ringbuffer.py  counter-indexed, gap-aware
      pipeline/quality.py     amplitude and contact gating
      pipeline/processing.py  seam 1 interface
      pipeline/spectral.py    seam 1: Welch, bands, EEG features
      pipeline/pulse.py       heart rate + HRV from PPG (slow lane)
      pipeline/affect.py      seam 2 interface
      pipeline/mapper.py      seam 2: features -> 0-1 axes vs resting
      pipeline/session.py     state machine + epoch loop
      pipeline/rawlog.py      append-only capture
      pipeline/store.py       session dirs + sqlite index
    web/
      muse.js                 BLE, reconnection, packet decoding
      uplink.js               batching, spooling, framing
      index.html              capture page
    clients/synthetic.py      fake headband
