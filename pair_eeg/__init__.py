"""Streaming backend for Muse EEG affect estimation.

    Muse --BLE--> browser --wss--> server --> [processing] --> [affect] --> browser

The two bracketed stages are deliberately empty. `pipeline.processing` turns
raw samples into spectra and band powers; `pipeline.affect` turns those into
0-1 values. Both ship as null implementations that produce correctly shaped
output so the rest of the system can be built and tested against them.
"""

__version__ = "0.1.0"
