"""
INI_PARSER - Parses song.ini files for metadata (name/artist/charter/difficulty/release) 
also captures star-power pitch disambiguation for .mid: (multiplier_note / star_power_note).

Source tables (gh/rb/ch) and html-tag cleanup regex live here
"""

import configparser
import pathlib
import re

import pandas as pd
import tqdm

# --------------
# Source tables
# --------------
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
SOURCES_DIR = BASE_DIR / "sources"

SOURCE_FILES = [
    ("gh.txt", True),          # Guitar Hero officials
    ("rb.txt", True),          # Rock Band officials
    ("sources.txt", False),    # Clone Hero community icons list
]

_RELEASE_SOURCES = None

def load_sources(path):
    table = {}
    with open(path, encoding='utf-8', errors='replace') as data:
        for line in data:
            line = line.strip()
            if ' = ' in line:
                key, value = line.split(' = ', 1)
                table[key] = value
    return table


def release_sources():
    global _RELEASE_SOURCES
    if _RELEASE_SOURCES is None:
        _RELEASE_SOURCES = []
        for filename, is_official in SOURCE_FILES:
            path = SOURCES_DIR / filename
            if path.exists():
                _RELEASE_SOURCES.append((load_sources(path), is_official))
            else:
                print(f"  [warn] source table missing, Release will be incomplete: {path}")
    return _RELEASE_SOURCES


# regex to strip html tags from ini metadata fields
DETAG = re.compile(r"<.*?>")


# --------
# Parsing
# --------

def _parse_int_tag(raw):
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def ini_parse(file):
    config = configparser.ConfigParser(strict=False)
    try:
        config.read(file, encoding='utf-8-sig')
    except UnicodeDecodeError:
        config.read(file, encoding='cp1252')

    section = 'song' if config.has_section('song') else (config.sections()[0] if config.sections() else None)
    if section is None:
        raise ValueError(f"No sections found in {file}")

    ini = config[section]

    # clean up tags & fix missing data, hard codes for malformed or missing
    name = DETAG.sub("", ini.get('name', 'unk'))
    artist = DETAG.sub("", ini.get('artist', 'unk'))
    charter = DETAG.sub("", ini.get('charter', 'unk'))
    diff = ini.get('diff_guitar', '-1')
    icon = ini.get('icon', '')

    # determine source and mark as official or not
    release, official = "Custom", False
    for source_dict, is_official in release_sources():
        if icon in source_dict:
            release = source_dict[icon]
            official = is_official
            break

    # multiplier_note / star_power_note: valid values are 103 or 116 only.
    # mid_parser to decide whether 103 means star power or solo, then dropped by build.py
    mult_note = _parse_int_tag(ini.get('multiplier_note', ini.get('star_power_note', None)))

    # Song folder identity - full resolved path to account for duplicate songs across different sources
    song_path = str(file.parent.resolve())

    return {
        'SongPath': song_path,
        'Name': name,
        'Artist': artist,
        'Charter': charter,
        'Difficulty': diff,
        'Release': release,
        'Official': official,
        'MultiplierNote': mult_note,
    }

# -----------
# Search loop
# -----------

# loops through search_path and provides errors to output along with cache
def ini_loop(search_path, errors=None):
    ini_out = []
    search = pathlib.Path(search_path)
    files = list(search.rglob("song.ini"))

    for file in tqdm.tqdm(files, desc="Gathering ini data", unit="file"):
        try:
            ini_out.append(ini_parse(file))
        except Exception as exc:
            if errors is not None:
                errors.append((str(file), type(exc).__name__, str(exc)))
            continue

    return pd.DataFrame(ini_out)
