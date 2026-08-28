# Which library, and why

Four open-source options were considered for talking to the Muse 2. All of them
speak the same underlying BLE protocol; they differ in maintenance and in what
they need installed.

## Chosen

**BrainFlow** — `python/capture.py`. Actively maintained, on PyPI, native BLE via
SimpleBLE on Windows 10.0.19041+, macOS 10.15+ and Linux. Gives EEG + PPG + IMU
through one API with no LSL layer. The BLED112 dongle it once required is now
deprecated.

**Web Bluetooth** — `web/index.html`. Zero install, works from any Chromium
browser, does its own device picking so no OS pairing is involved. EEG only, and
the protocol is implemented directly in the page (~250 lines), so there is no
dependency to rot.

## Considered and set aside

**muse-lsl** (`muselsl`) — the traditional research route, streams over Lab
Streaming Layer. Workable, but it drags in `pylsl` and the native `liblsl`
binary, is stale on Python 3.12+, and needs `--backend bleak` on current
versions because the default still expects a BLED112 dongle. Fine if you already
have an LSL setup; otherwise it is extra failure surface for no gain here.

**muse-js** — the original Web Bluetooth implementation, and the reference every
later port learned from. Published on npm, still functional, but unmaintained,
and all of its hosted demo pages are now 404. Worth reading as documentation of
the protocol.

**web-muse** — advertises itself as the maintained successor to muse-js. The BLE
protocol implementation is sound and was a useful reference, but the repo is thin
(single author, ~20 stars) and its README documents things that do not exist: no
build script, no `dist`, no tests, no npm publish, and no package `exports`
despite documenting `import ... from "web-muse/react"`. Not something to hang a
graded project on.

## Protocol notes

Useful if you want to implement this yourself — `web/index.html` is the worked
example.

- Service `0xfe8d`.
- Control characteristic `273e0001-4c4d-454d-96be-f03bac821358`. Commands are
  ASCII with a length prefix: `"X" + cmd + "\n"`, then byte 0 is overwritten with
  `length - 1`. Startup sequence: `h` (halt), `p50` (preset: 4×EEG + PPG + IMU),
  `s` (status), `d` (resume).
- EEG characteristics `273e0003`…`273e0006` = TP9, AF7, AF8, TP10 (`0007` is
  AUX). Each notification is a `uint16` packet counter followed by twelve 12-bit
  unsigned samples, packed three bytes per two samples.
- Scale to microvolts with `0.48828125 * (x - 0x800)`.
- 256 Hz per channel, so ~21.3 notifications per second per channel.
- The four channels arrive as separate notifications. Key rows by absolute sample
  number (`packetCounter * 12 + i`) rather than by arrival order, or a dropped
  packet on one channel silently shifts the others out of alignment.
- The packet counter is 16-bit and wraps every ~51 minutes; unwrap it if you
  record longer than that.
