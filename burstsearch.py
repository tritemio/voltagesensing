#
# Copyright 2015 Antonino Ingargiola <tritemio@gmail.com>
#
"""
This module defines functions to detect bursts of signal.
"""

from __future__ import division
import numpy as np


def burstsearch_py(signal, m, threshold, debug=False):
    """Sliding window burst search. Pure python version.

    Returns:
        Record array of burst data, one element per burst.
        Each element is a composite data type containing
        burst start, burst stop and burst score (the
        integrated signal over the burst duration).
    """
    bursts = []
    in_burst = False
    score = signal[:m].mean()
    deltasignal = (signal[m:] - signal[:-m])/m
    if debug:
        score_list = [score]
    for i, delta in enumerate(deltasignal):
        if np.abs(score)**2 > threshold:
            if not in_burst:
                # index of first cycle in burst
                start = i
                in_burst = True
        elif in_burst:
            # index of last cycle in burst
            stop = i + m - 2
            totalscore = signal[start:stop + 1].sum()
            bursts.append((start, stop, totalscore))
            in_burst = False
        score += delta
        if debug:
            score_list.append(score)

    # Create numpy recarray
    dt = np.dtype([('start','int32'), ('stop','int32'), ('score', 'float64')])
    bursts = np.array(bursts, dtype=dt).view(np.recarray)
    if debug:
        return bursts, np.array(score_list)
    else:
        return bursts


