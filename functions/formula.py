"""
FORMULA - Per-song difficulty scalar, computed by functions.density.compute_density_metrics

D = N * V * COV
    
    epsN = pNPS * 0.05
    N = ((medNPS + epsN) * aNPS * pNPS) ** (1 / 3)
    cvN = (stdNPS / aNPS + medNPS)

    epsV = pVPS * 0.05
    V = ((medVPS + epsV) * aVPS * pVPS) ** (1 / 3)
    cvV = (stdVPS / aVPS + medVPS)

    COV = 1 + (cvN * cvV) ** 0.5

N & V balance peak segment impact against average and median
COV is the interaction that accounts for spikes of difficulty - more variable songs >1, less variable -> 1
epsilon prevents median values of 0 from collapsing D while still being derived from song data
"""

import math

# RB-style 0-6 remap on raw D w/ integer cutoffs
# based on general distribution from official library across tiers by percentage
"""
Tier    Official	remap
0	    4.3%	    4.3%
1	    10.6%	    11.2%
2	    24.0%	    22.8%
3	    25.1%	    25.9%
4	    17.7%	    17.2%
5	    12.0%	    12.3%
6	    6.2%	    6.4%
"""
BIN_EDGES = [0, 8, 14, 21, 29, 38, 55, math.inf]
DIFF_LABELS = [0, 1, 2, 3, 4, 5, 6]

# Continuous tier remap - calculated, while generally fitting the manual remap
# ~One tier per LN_INC of log(D / BASE_D).
BASE_D = 7.6
LN_INC = 0.44

# RB manual 0-6 fit
def remap_diff(D):
    lower = BIN_EDGES[0]
    for label, upper in zip(DIFF_LABELS, BIN_EDGES[1:]):
        if lower < D <= upper:
            return label
        lower = upper
    return None

# log spaced tier calculation
def calc_tier(D):
    if D < BASE_D:
        return 0
    return int(math.floor(math.log(D / BASE_D) / LN_INC) + 1)

# D Formula
def calc_diff(metrics):
    pNPS, medNPS, aNPS, stdNPS = metrics['pNPS'], metrics['medNPS'], metrics['aNPS'], metrics['stdNPS']
    pVPS, medVPS, aVPS, stdVPS = metrics['pVPS'], metrics['medVPS'], metrics['aVPS'], metrics['stdVPS']

    # NPS combo
    epsN = pNPS * 0.05
    N = ((medNPS + epsN) * aNPS * pNPS) ** (1 / 3)
    cvN = (stdNPS / (aNPS + medNPS))

    # VPS combo
    epsV = pVPS * 0.05
    V = ((medVPS + epsV) * aVPS * pVPS) ** (1 / 3)
    cvV = (stdVPS / (aVPS + medVPS))

    # CoV interaction across NPS & VPS
    COV = 1 + (cvN * cvV) ** 0.5

    # base scalar difficulty
    D = N * V * COV

    return {
        'N': N,
        'V': V,
        'COV': COV,
        'D': D,
        'RemapDiff': remap_diff(D),
        'CalcTier': calc_tier(D),
    }
