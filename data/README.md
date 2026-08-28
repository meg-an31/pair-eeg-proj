Recordings live here, one folder per run:

    data/<label>/eeg.csv      time_s, TP9, AF7, AF8, TP10, marker
    data/<label>/meta.json    sample rate, channels, protocol timeline
    data/<label>/ppg.csv      (capture.py only) time_s, PPG0..2
    data/<label>/imu.csv      (capture.py only) time_s, ACC_X..Z, GYR_X..Z

Everything in here is gitignored. EEG is person-identifiable - keep real
recordings out of the repo and share them among the group directly.

See ../examples/synthetic_run/ for what analyze.py produces.
