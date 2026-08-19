"""
TIMING - Shared tick/time helpers for the chart and midi parsers

Both formats have a similar tempo model - ticks with bpm stamps
.chart has a SyncTrack section
.mid has set_tempo meta messages
outputs from both parsers are the same so these functions are shared
"""

import bisect

# tempo mapping
def map_cum(tempos, tick_res):
    sorted_ticks = sorted(tempos.keys())
    cumulative_ms = {}
    running_ms = 0.0

    for i, t in enumerate(sorted_ticks):
        cumulative_ms[t] = running_ms
        if i + 1 < len(sorted_ticks):
            next_t = sorted_ticks[i + 1]
            bpm = tempos[t]
            running_ms += ((next_t - t) / tick_res) * (60000.0 / bpm)

    return sorted_ticks, cumulative_ms

# convert ticks to ms
def tick_to_ms(tick, tick_res, tempos, sorted_ticks, cum_ms):
    tick = int(tick)
    idx = max(bisect.bisect_right(sorted_ticks, tick) - 1, 0)
    ref_tick = sorted_ticks[idx]
    return (
        cum_ms[ref_tick]
        + ((tick - ref_tick) / tick_res) * (60000.0 / tempos[ref_tick])
    )
