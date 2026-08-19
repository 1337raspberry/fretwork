"""
MID_PARSER - Parses notes.mid files into the same note stream shape produced by chart_parser:
    {
        'song_path': str,
        'source_format': 'mid',
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

Scope is Expert difficulty only, matching chart_parser

NOTE STATE IS NOT PARSED - strum/tap/hopo are not used in the calcs and are discarded

STAR POWER / SOLO
    Modern charts: pitch 116 = star power, pitch 103 = solo.
    Older charts:  pitch 103 = star power, no solo track.
    song.ini's multiplier_note / star_power_note tag is passed from build

SysEx-based open note (0x01) events are not implemented
"""

import pathlib

import mido
import numpy as np
import tqdm

from parsers.timing import map_cum, tick_to_ms

# ---------------------------------------------------------------------
# Mid-specific constants (Expert only)
# ---------------------------------------------------------------------
GUITAR_TRACK_NAMES = ['PART GUITAR', 'T1 GEMS']

OPEN_MID = 7
X_FRETS = {96: 0, 97: 1, 98: 2, 99: 3, 100: 4}  # pitch -> lane number
X_OPEN_PIT = 95        # note-based open, requires ENHANCED_OPENS

SP_PIT = 116             # modern star power phrase
SOLO_PIT = 103                   # solo phrase, unless it IS star power
LEGACY_SP = 103      # older charts, per multiplier_note tag

# Legacy GH1/2-style open note encoding
M_OPEN_PIT = 0
M_OPEN_CNL = 5

ENH_OPEN = 'ENHANCED_OPENS'


# ----------
# Tempo map
# ----------

def map_mid_tempo(mid):
    tempos = {}

    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                bpm = 60_000_000 / msg.tempo
                tempos[abs_tick] = bpm  # last writer wins on tie

    if not tempos:
        tempos[0] = 120.0  # MIDI default

    sorted_ticks, cum_ms = map_cum(tempos, mid.ticks_per_beat)
    return tempos, sorted_ticks, cum_ms


# -----------------------
# Note-stream extraction
# -----------------------

def _is_note_off(msg):
    return msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)


def mid_notes(mid_source, multiplier_note=None):
    mid = mido.MidiFile(str(mid_source), clip=True)
    tick_res = mid.ticks_per_beat

    tempos, sorted_ticks, cum_ms = map_mid_tempo(mid)

    def to_ms(tick):
        return tick_to_ms(tick, tick_res, tempos, sorted_ticks, cum_ms)

    # locate guitar track
    track_map = {t.name.strip(): t for t in mid.tracks if t.name}
    guitar_track = None
    for name in GUITAR_TRACK_NAMES:
        if name in track_map:
            guitar_track = track_map[name]
            break

    if guitar_track is None:
        raise ValueError(f"No guitar track found. Available tracks: {list(track_map.keys())}")

    # check for [ENHANCED_OPENS] text event anywhere in the track
    enhanced_opens = any(
        msg.type == 'text' and ENH_OPEN in msg.text.upper()
        for msg in guitar_track
    )

    # star power / solo pitch assignment
    legacy_sp = (multiplier_note == LEGACY_SP)
    sp_pitch = LEGACY_SP if legacy_sp else SP_PIT
    solo_pitch = None if legacy_sp else SOLO_PIT

    phrase_pitches = {sp_pitch}
    if solo_pitch is not None:
        phrase_pitches.add(solo_pitch)

    dropped = {
        'unclosed_star_power': 0,
        'unclosed_solo': 0,
    }

    masks_by_tick = {}
    star_power = []
    solos = []

    open_starts = {}
    abs_tick = 0

    for msg in guitar_track:
        abs_tick += msg.time

        if msg.type == 'note_on' and msg.velocity > 0:
            pitch = msg.note

            if pitch in X_FRETS:
                lane = X_FRETS[pitch]
            elif pitch == M_OPEN_PIT and msg.channel == M_OPEN_CNL:
                lane = OPEN_MID                      # legacy open
            elif pitch == X_OPEN_PIT and enhanced_opens:
                lane = OPEN_MID                      # note-based open
            else:
                lane = None
                if pitch in phrase_pitches:
                    open_starts.setdefault(pitch, []).append(abs_tick)

            if lane is not None:
                masks_by_tick[abs_tick] = masks_by_tick.get(abs_tick, 0) | (1 << lane)

        elif _is_note_off(msg):
            pitch = msg.note
            if pitch in phrase_pitches:
                starts = open_starts.get(pitch)
                if starts:
                    start_tick = starts.pop(0)
                    if pitch == sp_pitch:
                        star_power.append((to_ms(start_tick), to_ms(abs_tick)))
                    elif pitch == solo_pitch:
                        solos.append((to_ms(start_tick), to_ms(abs_tick)))

    # Anything still open at end of track never closes. Dropped, but counted for errors
    for pitch, starts in open_starts.items():
        if not starts:
            continue
        if pitch == sp_pitch:
            dropped['unclosed_star_power'] += len(starts)
        elif pitch == solo_pitch:
            dropped['unclosed_solo'] += len(starts)

    if not masks_by_tick:
        raise ValueError("No valid Expert note events found in guitar track")

    ordered_ticks = sorted(masks_by_tick.keys())

    return {
        'song_path': str(pathlib.Path(mid_source).parent.resolve()),
        'source_format': 'mid',
        'resolution': tick_res,
        'notes': {
            'time_ms': np.array([to_ms(t) for t in ordered_ticks]),
            'lanes': np.array([masks_by_tick[t] for t in ordered_ticks], dtype=np.uint8),
        },
        'spans': {
            'star_power': star_power,
            'solo': solos,
        },
        'dropped': dropped,
    }


# -----------
# Search loop
# -----------

# loops through search path, retrieving errors to provide along with cache
def mid_loop(search_path, multiplier_notes=None, errors=None):
    multiplier_notes = multiplier_notes or {}
    mid_out = {}

    search = pathlib.Path(search_path)
    files = list(search.rglob("notes.mid"))

    for file in tqdm.tqdm(files, desc="Parsing midis", unit="file"):
        try:
            song_path = str(pathlib.Path(file).parent.resolve())
            stream = mid_notes(file, multiplier_notes.get(song_path))
            mid_out[stream['song_path']] = stream
        except Exception as exc:
            if errors is not None:
                errors.append((str(file), type(exc).__name__, str(exc)))
            continue

    return mid_out
