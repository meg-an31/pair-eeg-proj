# pair-eeg — streaming backend

    Muse 2 --BLE--> browser --wss--> server --> [processing] --> [affect] --> browser

Everything in that chain exists except the two bracketed stages, which are
deliberately empty. They ship as null objects producing correctly shaped
output, so the transport, session lifecycle, recording and front end can all
be built and tested before any DSP is written.

## Run it without a headband

    python -m venv .venv && .venv/bin/pip install -e ".[dev]"
    .venv/bin/python -m pair_eeg                       # terminal 1
    PYTHONPATH=. .venv/bin/python -m clients.synthetic --baseline   # terminal 2

`clients/synthetic.py` speaks the same wire protocol the browser does, so if
it works and the browser does not, the fault is in the browser. It can
simulate a dropout (`--drop-at 20 --drop-for 5`) to exercise gap handling.

## Run it with a headband

    cd web && python3 -m http.server 8000

Then open <http://localhost:8000>. Web Bluetooth needs a secure context, so
`localhost` or https — `file://` will not work. Chrome, Edge or Opera only;
Safari and Firefox do not implement it.

**The Muse allows one connection at a time.** Close the Muse phone app, unpair
the headband in system Bluetooth settings, and close other tabs holding it.

## The two blank stages

| Stage | File | Contract |
|---|---|---|
| Processing | `pair_eeg/pipeline/processing.py` | `EpochWindow` → `ProcessedFeatures` |
| Affect | `pair_eeg/pipeline/affect.py` | `ProcessedFeatures` → `AffectValues` |

Write a class satisfying the `Processor` / `AffectMapper` protocol and pass it
to `transport.server.run()`. Nothing else changes.

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
      pipeline/processing.py  BLANK SEAM
      pipeline/affect.py      BLANK SEAM
      pipeline/session.py     state machine + epoch loop
      pipeline/rawlog.py      append-only capture
      pipeline/store.py       session dirs + sqlite index
    web/
      muse.js                 BLE, reconnection, packet decoding
      uplink.js               batching, spooling, framing
      index.html              capture page
    clients/synthetic.py      fake headband
