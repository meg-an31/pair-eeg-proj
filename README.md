# Muse 2 EEG — open-source capture and analysis

Getting signals off a Muse 2 headband **without the Muse app**, using only
open-source tooling. Two independent capture paths, one shared analysis pipeline.

The headband is not locked down: it advertises a plain Bluetooth LE GATT service
(`0xfe8d`) with notify characteristics for EEG, PPG and IMU, plus a control
channel that takes short ASCII commands. Nothing about reading it requires the
vendor app or a subscription.

```
                 ┌─ web/index.html ──── browser, Web Bluetooth ─┐
Muse 2 ──BLE──┤                                              ├── eeg.csv + meta.json ──> python/analyze.py
                 └─ python/capture.py ── BrainFlow, native BLE ─┘
```

---

## Path A — browser (no install)

Fastest way to see live signal. EEG only.

1. Open `web/index.html` in **Chrome, Edge or Opera** (desktop or Android).
   Either serve it locally:
   ```bash
   cd web && python3 -m http.server 8000
   # then open http://localhost:8000
   ```
   or use the hosted copy: <https://app.62-238-126-163.sslip.io>
2. Charge the headband, **quit the Muse phone app and turn the phone's Bluetooth
   off**, and leave the headband **unpaired** in your OS Bluetooth settings.
3. Click **Connect headset** and pick `Muse-XXXX` in the browser's device picker.
   That picker *is* the connection — there is no OS-level pairing step.
4. Check the quality table: RMS roughly 5–45 µV per channel. Blink hard; AF7/AF8
   should spike. Damp the behind-ear sensors and clear hair if a channel reads
   *no contact*.
5. **Guided eyes open/closed** runs alternating blocks and writes condition
   markers into the data. Then **Download eeg.csv** and **Download meta.json**
   into `data/<label>/`.

Web Bluetooth needs a secure context, so `http://localhost` or `https://` — a
file:// path will not work. No Safari, no Firefox, nothing on iOS.

`?demo=1` in the URL starts a synthetic 10 Hz stream so you can demo the
interface with no hardware. `?demo=1&rec=60&protocol=eyes&dump=1` runs a
scripted 60 s capture and dumps the CSV into the page — used for the headless
test below.

## Path B — Python / BrainFlow

Use this when you need PPG (heart rate) or the accelerometer too, or want
scripted protocols. Runs on the laptop that has the Bluetooth radio.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r python/requirements.txt

python python/capture.py --check --line-hz 50          # 15 s quality check
python python/capture.py --protocol eyes --label s01    # guided recording
```

`--line-hz 50` in Europe, `60` in North America. `--mac AA:BB:...` speeds up
discovery. No BLED112 dongle is needed on Windows 10.0.19041+ — BrainFlow uses
native BLE via SimpleBLE, and the old dongle path is deprecated.

## Analysis (either path)

```bash
python python/analyze.py data/s01
```

Filtering → artifact rejection → epoching by condition → Welch PSD → band powers
→ eyes-closed vs eyes-open statistics → figures + `results.json` +
`band_powers.csv`. See `examples/synthetic_run/` for the output shape.

## No headband yet?

```bash
python python/synth.py --outdir data --label synthetic
python python/analyze.py data/synthetic
```

`synth.py` bakes in known ground truth: 10 Hz alpha ~3.5× stronger in amplitude
at TP9/TP10 during eyes-closed blocks, so ~10× in power. If `analyze.py` does
not recover a posterior alpha power ratio near 10, the pipeline is wrong. Use it
before trusting any real recording.

---

## Layout

| Path | What |
|---|---|
| `web/index.html` | Self-contained Web Bluetooth EEG client — live scope, quality table, band powers, guided protocol, CSV export. No dependencies. |
| `python/capture.py` | BrainFlow capture: EEG + PPG + IMU, guided or continuous protocols, signal-quality check. |
| `python/analyze.py` | Shared analysis pipeline. Consumes either path's output. |
| `python/synth.py` | Synthetic known-truth generator for validating the pipeline. |
| `examples/synthetic_run/` | Reference output of `analyze.py` on synthetic data. |
| `docs/SETUP.md` | Step-by-step install, and the problems we actually hit. |
| `docs/TOOLING.md` | Why these two libraries and not the other four. |

## Data format

`eeg.csv` — 256 Hz, microvolts, one row per sample:

```csv
time_s,TP9,AF7,AF8,TP10,marker
0.000000,-50.859539,0.496682,35.328564,9.251408,1.000000
0.003906,-51.323713,-19.759685,29.002132,18.649346,0.000000
```

`marker` is 0 except at condition boundaries: `1` eyes-open start, `2`
eyes-closed start, `9` run end. `meta.json` carries the sample rate, channel
names and the protocol timeline; `analyze.py` needs both files.

Both capture paths emit this identical format, which is the point — the browser
and the Python route are interchangeable as far as analysis is concerned.

## Verification

The browser path was tested end to end headlessly before use: driving
`?demo=1&rec=60&protocol=eyes` in headless Chromium produced 15,780 samples at
256 Hz, and `analyze.py` ingested the export unmodified — 57 epochs kept, 0
rejected, figures and summary files written. The alpha ratio in that test is
1.00 by construction (the demo generator emits constant alpha and does not
modulate it by block); that number only becomes meaningful with a real head in
the headband.

## Note on the data

EEG is person-identifiable. `data/` is gitignored on purpose — share real
recordings among the group directly rather than committing them.
