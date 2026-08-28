# Setup, and the problems we hit

## Windows + Python

The plain `pip install` fails on a system-wide Python:

```
ERROR: Could not install packages due to an OSError: [WinError 5]
Access is denied: 'C:\Python311\share'
```

`brainflow` and `pylsl` both write data files into `<prefix>\share\`, which a
non-admin account cannot touch. A virtual environment fixes it — do **not** run
pip as administrator.

```powershell
mkdir $HOME\muse-project; cd $HOME\muse-project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r python\requirements.txt
```

If PowerShell blocks the activation script, once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

`pip install --user` also gets past WinError 5, but it puts console scripts
outside `PATH` and produces a second confusing failure. Use the venv.

## Headband prep — this is where most failures happen

- Charge above ~30%. A low headband advertises intermittently or not at all.
- **Quit the Muse phone app and turn the phone's Bluetooth off.** The app holds
  an exclusive BLE connection; nothing else can see the headband while it does.
- **Do not pair the Muse in your OS Bluetooth settings.** Both capture paths
  connect over GATT themselves, and an OS pairing competes with them. If it is
  already paired, remove it (`bluetoothctl remove <MAC>` on Linux, "Forget This
  Device" on macOS, "Remove device" on Windows).
- Power on — the LED should blink, not sit solid.

## Sensor contact

Four electrodes: TP9 (left ear), AF7 (left forehead), AF8 (right forehead),
TP10 (right ear). Aim for RMS roughly 5–45 µV per channel.

- Dry skin is the usual culprit. Slightly damp the behind-ear sensors.
- Clear hair from under the ear sensors.
- Wipe the forehead for AF7/AF8.
- Blink hard as a sanity check — AF7/AF8 should spike. If nothing moves, you are
  looking at noise, not EEG.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Device never appears | phone app holds the connection | quit the app, phone Bluetooth off |
| Device appears then drops instantly | firmware preset | `muselsl stream --preset p21`; BrainFlow handles this itself |
| Nothing found at all, Linux | adapter down | `systemctl start bluetooth`, `bluetoothctl power on`, check `rfkill` |
| Nothing found at all, VM/WSL/container | no BT passthrough | run on the host OS |
| `liblsl not found` | `pylsl` needs the native library | `conda install -c conda-forge liblsl`, or the liblsl .deb — or skip LSL and use BrainFlow |
| Install fails on Python 3.12+ | `muse-lsl` is stale | use Python 3.10/3.11 |
| Browser: no Connect button | no Web Bluetooth | Chrome/Edge/Opera; never Safari/Firefox/iOS |
| Browser: page loads but cannot connect | insecure context | serve over `http://localhost` or `https://`, not `file://` |
| One flat channel | sensor contact | see above |

## PPG / heart rate

The preset code that enables PPG differs by model: **`p50`/`p51` on Muse 2**,
**`p61`/`p50` on Muse S**. A Muse 2 *accepts* the Muse S code `p61` without any
error and then streams no PPG at all — `ppg.csv` comes out with 0 samples while
EEG and IMU look fine. `capture.py` now tries the model's codes in order and
checks that ancillary samples actually arrive before trusting the result;
`--ppg-preset p51` forces a specific one.

**A rate around 45-50 bpm is the classic drift artefact.** The cardiac band
starts at 0.7 Hz = 42 bpm, and baseline wander is 1/f-shaped, so the largest raw
bin in that band sits at the low edge whatever your pulse is doing. Both the
Python analysis and the web page now whiten the spectrum against its local
median and score candidates across their harmonics (a real pulse puts energy at
2f and 3f; a drift shoulder does not), which recovers the true rate from a trace
where the naive method read 47 bpm for a true 78. If you are on an older copy,
update. And check it against your wrist: a fit person at rest genuinely can sit
at 50.

**A rate flipping rapidly between two very different values** (say 80 and 170)
is not your heart - especially when the high value is about double the low one.
Two mechanisms cause it, and both are now handled: a movement artifact can
dominate one PPG channel (the estimate the most channels agree on wins), and
the pulse's own 2nd harmonic can outweigh its fundamental when forehead
perfusion fades in and out (an octave guard takes the fundamental whenever half
the candidate frequency also carries a real peak, and the live page tracks the
rate over time so one bad window cannot flip the readout). A genuinely fast
pulse has nothing at half its rate, so real high rates are not clamped. The
readout says "unstable" with the spread rather than showing a number it cannot
stand behind.

If PPG streams but the rate looks wrong — a large `bpm_sd`, an implausible
RMSSD, or far fewer beats than the rate implies — run
`python python/ppg_check.py data/<run>` and look at `ppg_check.png`. A clear
periodic wave means the detector; no wave means contact.

The other requirement is physical: the pulse sensor is in the **centre of the
forehead band** and needs firmer, flatter contact than the EEG electrodes do.
Sit still — PPG is much more motion-sensitive than EEG.

## A note on cloud machines

A cloud VM has no Bluetooth radio — no adapter, no BlueZ, no `bluetooth.service`.
`muselsl` and `brainflow` install fine there and then find nothing, forever.
Capture on a laptop; analysis can run anywhere.
