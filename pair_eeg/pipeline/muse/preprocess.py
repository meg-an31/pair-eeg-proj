"""Step 2: preprocessing — from raw EEG to clean, analysis-ready windows.

Three jobs, done in this order:

  1. Split the raw signal into two paths:
       - cortical path: 1-40 Hz band-pass + mains notch. Brain rhythms
         (alpha, beta) live here. Everything above is discarded FROM THIS
         PATH ONLY.
       - EMG path: 55-110 Hz band-pass. Facial muscle activity lives here.
         Most pipelines throw this away; we keep it because on the Muse's
         forehead electrodes it carries the frown-muscle valence signal.
  2. Cut the recording into short overlapping windows (2 s, 50% overlap).
     All features later are computed per-window.
  3. Mark bad windows: any window where the cortical signal exceeds
     ±100 µV on any channel is rejected — that amplitude is not brain,
     it's blinks, motion, or a loose electrode. Losing 20-40% of windows
     on a real dry-electrode headband is normal.

Filtering uses forward-backward ("zero-phase") filters so nothing gets
shifted in time relative to the other signals.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

DEFAULT_FS = 256.0        # Muse EEG sample rate

CORTICAL_LO, CORTICAL_HI = 1.0, 40.0
EMG_LO, EMG_HI = 55.0, 110.0   # starts at 55, not 50, to skirt mains hum
NOTCH_HZ = 50.0                # UK/EU mains frequency; use 60.0 in the US

AMPLITUDE_LIMIT_UV = 100.0     # cortical signal beyond this is artifact


def _bandpass(data, fs, lo_hz, hi_hz, order=4):
    """Zero-phase Butterworth band-pass along the last axis."""
    sos = butter(order, [lo_hz, hi_hz], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def _notch(data, fs, freq_hz=NOTCH_HZ, quality=30.0):
    """Zero-phase narrow notch to remove mains hum."""
    b, a = iirnotch(freq_hz, quality, fs=fs)
    return filtfilt(b, a, data, axis=-1)


def split_bands(eeg, fs=DEFAULT_FS):
    """Fork the raw EEG into (cortical, emg), both same shape as the input.

    The notch on the cortical path is belt-and-braces (the 1-40 Hz band-pass
    already attenuates 50 Hz) but real amplifiers leak hum below the cutoff.
    """
    cortical = _notch(_bandpass(eeg, fs, CORTICAL_LO, CORTICAL_HI), fs)
    emg = _bandpass(eeg, fs, EMG_LO, EMG_HI)
    return cortical, emg


def window_indices(n_samples, fs=DEFAULT_FS, win_s=2.0, overlap=0.5):
    """Start/stop sample indices of overlapping windows covering a recording.

    2 s windows with 50% overlap -> a new window starts every second.
    A 30 s recording gives 29 of them.
    """
    win = int(round(win_s * fs))
    step = max(int(round(win * (1.0 - overlap))), 1)
    return [(s, s + win) for s in range(0, n_samples - win + 1, step)]


def good_window_mask(cortical, windows, limit_uv=AMPLITUDE_LIMIT_UV):
    """True for each window that stays inside ±limit_uv on every channel.

    Checked on the cortical path: blinks and motion are low-frequency, so
    they survive the 1-40 Hz filter and are exactly what this catches.
    """
    return np.array([
        bool(np.all(np.abs(cortical[:, s:e]) < limit_uv)) for s, e in windows
    ])


def preprocess_eeg(eeg, fs=DEFAULT_FS, win_s=2.0, overlap=0.5,
                   limit_uv=AMPLITUDE_LIMIT_UV):
    """The whole step-2 chain in one call.

    Takes raw EEG shaped (channels, samples) in µV. Returns a dict:
      cortical  filtered brain-rhythm signal, same shape as input
      emg       filtered muscle signal, same shape as input
      windows   list of (start, stop) sample indices
      good      boolean per window; False = rejected as artifact
      fs        sample rate, passed through for later stages
    """
    cortical, emg = split_bands(eeg, fs)
    windows = window_indices(eeg.shape[1], fs, win_s, overlap)
    good = good_window_mask(cortical, windows, limit_uv)
    return {"cortical": cortical, "emg": emg, "windows": windows,
            "good": good, "fs": fs}
