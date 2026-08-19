"""
CHART_PARSER - Parses notes.chart files into the same note stream shape produced by mid_parser:
    {
        'song_path': str,
        'source_format': 'chart',
        'resolution': int,
        'notes': {
            'time_ms': np.ndarray,   # sorted, one entry per tick
            'lanes':   np.ndarray uint8,   # bitmask, bit N = lane N
        },
        'spans': {
            'star_power': [(start_ms, end_ms), ...],
            'solo':       [(start_ms, end_ms), ...],
        },
        'dropped': {counter_name: int, ...},
    }

Scope is Expert difficulty only, matching mid_parser    

Fret ENCODING
    One uint8 per tick. Bit N set means fret N is played: 
    bits 0-4 are GRBYO, bit 7 is open
    Bits 5-6 are unused - in .chart those are the tap and force-flip modifiers

    A bitmask rather than a frozenset to support small cache size

NOTE STATE IS NOT PARSED - strum/tap/hopo are not used in the calcs and are discarded
"""

import pathlib

import numpy as np
import tqdm

from parsers.timing import map_cum, tick_to_ms

# ------------------------
# Chart-specific constants
# ------------------------
OPEN_NOTE = 7
NOTE_FRETS = {0, 1, 2, 3, 4, OPEN_NOTE}

SP_ID = 2 # 'S 2 <length>' in the difficulty section

X_GUITAR = 'ExpertSingle'

SOLO = 'solo'
SOLO_END = 'soloend'


# ---------------------------------------------------------------------
# Raw section parsing (chart's [Section] / key = value text format)
# ---------------------------------------------------------------------

def parse_chart(chart_source):
    c_dict = {}
    c_sect = None

    with open(chart_source, encoding='utf-8-sig') as c_data:
        for line in c_data:
            line = line.strip()

            if line.startswith('[') and line.endswith(']'):
                c_sect = line[1:-1]
                c_dict[c_sect] = {}

            elif ' = ' in line and c_sect is not None:
                key, value = line.split(' = ', 1)
                key = key.strip()
                value = value.strip()

                if key in c_dict[c_sect]:
                    if not isinstance(c_dict[c_sect][key], list):
                        c_dict[c_sect][key] = [c_dict[c_sect][key]]
                    c_dict[c_sect][key].append(value)
                else:
                    c_dict[c_sect][key] = value

    # Global [Events] holds section names / lyrics - nothing used from this section
    c_dict.pop('Events', None)
    return c_dict

# SyncTrack 'B <bpm*1000>' markers -> ({tick: bpm}, sorted, cumulative)
def build_tempo_map(sync_track, tick_res):
    tempos = {}
    for tick, markers in sync_track.items():
        tick = int(tick)
        markers = markers if isinstance(markers, list) else [markers]
        for marker in markers:
            if marker.startswith('B'):
                tempos[tick] = int(marker.split()[1]) / 1000

    if not tempos:
        tempos[0] = 120.0

    sorted_ticks, cum_ms = map_cum(tempos, tick_res)
    return tempos, sorted_ticks, cum_ms


# ----------------------
# Note-stream extraction
# ----------------------

# Solo check
def _event_text(event):
    parts = event.split(None, 1)
    if len(parts) < 2:
        return ''
    return parts[1].strip().strip('"').strip().lower()


def chart_notes(chart_source):
    c_dict = parse_chart(chart_source)
    for required in ('Song', 'SyncTrack', X_GUITAR):
        if required not in c_dict:
            raise ValueError(f"Missing required section '{required}' in {chart_source}")

    tick_res = int(c_dict['Song']['Resolution'])
    tempos, sorted_ticks, cum_ms = build_tempo_map(c_dict['SyncTrack'], tick_res)

    def to_ms(tick):
        return tick_to_ms(tick, tick_res, tempos, sorted_ticks, cum_ms)

    # Bucket lanes by tick. 
    # parse_chart collapses repeated keys to a chord
    masks_by_tick = {}
    sp = []
    solos = []

    dropped = {
        'unclosed_solo': 0,
        'malformed_star_power': 0,
    }

    solo_open_tick = None

    # Tick order matters for solo start/end pairing, so walk sorted.
    for tick_str in sorted(c_dict[X_GUITAR].keys(), key=int):
        events = c_dict[X_GUITAR][tick_str]
        tick = int(tick_str)
        events = events if isinstance(events, list) else [events]

        mask = 0

        for event in events:
            parts = event.split()
            if not parts:
                continue

            kind = parts[0]

            if kind == 'N' and len(parts) >= 2:
                n_val = int(parts[1])
                if n_val in NOTE_FRETS:
                    mask |= 1 << n_val

            elif kind == 'S' and len(parts) >= 3:
                if int(parts[1]) == SP_ID:
                    length = int(parts[2])
                    if length > 0:
                        sp.append((to_ms(tick), to_ms(tick + length)))
                    else:
                        dropped['malformed_star_power'] += 1

            elif kind == 'E':
                text = _event_text(event)
                if text == SOLO:
                    if solo_open_tick is not None:
                        dropped['unclosed_solo'] += 1
                    solo_open_tick = tick
                elif text == SOLO_END:
                    if solo_open_tick is not None:
                        solos.append((to_ms(solo_open_tick), to_ms(tick)))
                        solo_open_tick = None

        if mask:  # skip ticks that only carried modifiers or phrases
            masks_by_tick[tick] = mask

    # A solo left open at end of track never closes - dropped, counted.
    if solo_open_tick is not None:
        dropped['unclosed_solo'] += 1

    if not masks_by_tick:
        raise ValueError(f"No note events found in {X_GUITAR}")

    ordered_ticks = sorted(masks_by_tick.keys())

    return {
        'song_path': str(pathlib.Path(chart_source).parent.resolve()),
        'source_format': 'chart',
        'resolution': tick_res,
        'notes': {
            'time_ms': np.array([to_ms(t) for t in ordered_ticks]),
            'lanes': np.array([masks_by_tick[t] for t in ordered_ticks], dtype=np.uint8),
        },
        'spans': {
            'star_power': sp,
            'solo': solos,
        },
        'dropped': dropped,
    }


# -----------
# Search loop
# -----------

# loops through path and reports errors for unparesable files
def chart_loop(search_path, errors=None):
    chart_out = {}

    search = pathlib.Path(search_path)
    files = list(search.rglob("notes.chart"))

    for file in tqdm.tqdm(files, desc="Parsing charts", unit="file"):
        try:
            stream = chart_notes(file)
            chart_out[stream['song_path']] = stream
        except Exception as exc:
            if errors is not None:
                errors.append((str(file), type(exc).__name__, str(exc)))
            continue

    return chart_out
