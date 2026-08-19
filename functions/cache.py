"""
CACHE - The cache save/load pieces and retrieval code generation
BUILD writes a cache of all the song data needed from the search_path in config.py
ANALYZE and RENDER can read from caches to generate metrics/visuals

Shape:

    {
        'generated_at': str,
        'search_path':  str,
        'codes':        {code: song_path},
        'songs': {
            song_path: {
                'code':          str,
                'song_path':     str,
                'meta':          {...},   # trimmed ini row, CSV columns only
                'source_format': 'chart' | 'mid',
                'notes': {
                    'time_ms': ndarray,   # sorted
                    'lanes':   ndarray uint8,  # bitmask, bit N = lane N
                },
                'spans': {'star_power': [(ms, ms)...], 'solo': [...]},
            },
            ...
        },
        'dropped':  {counter_name: int, ...},
    }

Caches should be managed based on timestamp / generation time & date

When generated with errors, a CSV is produced alongside the cache with details
"""

import hashlib
import pickle
from datetime import datetime

# Hash-derived retrieval codes digit length
CODE_LEN = 8

def gen_ts():
    return datetime.now().strftime("%m%d%Y-%H%M")

# Retrieval codes
def _hash_code(song_path, digits):
    digest = hashlib.sha1(song_path.encode('utf-8')).hexdigest()
    return int(digest, 16) % (10 ** digits)

# assigns the code used to select for visualization
def assign_codes(song_paths, digits=None):
    digits = digits or CODE_LEN
    span = 10 ** digits

    if len(song_paths) > span // 2:
        raise ValueError(
            f"{len(song_paths)} songs is too many for {digits}-digit codes; "
            f"raise cache.CODE_LEN"
        )

    taken = {}
    for song_path in sorted(song_paths):
        code_int = _hash_code(song_path, digits)
        while code_int in taken:
            code_int = (code_int + 1) % span
        taken[code_int] = song_path

    codes = {path: str(code).zfill(digits) for code, path in taken.items()}

    assert len(set(codes.values())) == len(codes), "code collision survived probing"
    return codes

# Persistence
def save(cache, cache_path):
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cache_path

def load(cache_path):
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

# Lookup retrieval codes - str or int w/ zero padding so '421' and '00000421' both work
def entries_by_code(cache, codes):
    entries = []
    missing = []

    for raw in codes:
        code = str(raw).strip().zfill(CODE_LEN)
        song_path = cache['codes'].get(code)
        if song_path is None:
            missing.append(code)
            continue
        entries.append(cache['songs'][song_path])

    return entries, missing
