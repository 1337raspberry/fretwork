"""
CURVES - smoothes density.py's windowed NPS/VPS arrays for visualization

The raw arrays are converted to rates (notes/sec, lane-changes/sec) before smoothing by an EMA
"""

import math

import numpy as np

from functions import density

# Shared smoothing time constant for curves
TAU_MS = 2000.0 

ZERO_PHASE = True # fix for peaks lagging by tau from single pass version

# Sequential single-pole accumulator
def _ema_forward(samples, decay):
    gain = 1.0 - decay
    acc = 0.0
    out = []
    for raw in samples.tolist():
        acc = acc * decay + raw * gain
        out.append(acc)
    return np.array(out, dtype=np.float64)

# Single-pole low-pass over the uniform sample array
def _ema_curve(samples, step_ms, tau_ms, zero_phase=ZERO_PHASE):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        return samples
    if tau_ms <= 0:
        return samples.copy()

    decay = math.exp(-step_ms / tau_ms)

    forward = _ema_forward(samples, decay)
    if not zero_phase:
        return forward

    backward = _ema_forward(forward[::-1], decay)
    return backward[::-1]

# initial smoothing function
def smooth_curves(windows, window_ms, step_ms,
                            tau_ms=TAU_MS, zero_phase=ZERO_PHASE):

    if windows is None or len(windows['time_ms']) == 0:
        return None

    window_s = window_ms / 1000.0
    nps_rate = windows['raw_nps_samples'] / window_s
    vps_rate = windows['raw_vps_samples'] / window_s

    nps_curve = _ema_curve(nps_rate, step_ms, tau_ms, zero_phase)
    vps_curve = _ema_curve(vps_rate, step_ms, tau_ms, zero_phase)

    return {
        'time_ms': windows['time_ms'],
        'nps': nps_curve,
        'vps': vps_curve,
        'd_raw': np.sqrt(nps_curve * vps_curve),
    }

# final curves for render
def calc_curves(notes,
                   window_ms=density.WINDOW_MS, step_ms=density.STEP_MS,
                   tau_ms=TAU_MS, zero_phase=ZERO_PHASE):
    
    windows = density.window_arrays(notes, window_ms, step_ms)
    return smooth_curves(windows, window_ms, step_ms,
                                   tau_ms, zero_phase)
