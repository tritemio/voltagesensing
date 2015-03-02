#
# Copyright 2015 Antonino Ingargiola <tritemio@gmail.com>
#
"""
This module defines functions to extract timetraces from video ROIs,
filter the timetraces (detrend, smoothing), identify blinking and
compute the signal alternation in phase with the patch-clamp signal.
"""
from __future__ import division

import numpy as np
import scipy.ndimage as ndi


##
#  Timetrace extraction and processing: detrend and blinking
#

def get_roi_square(point, pad=2):
    """Return a square selection of pixels around `point`.
    """
    col, row = point
    mask = (slice(None), slice(row-pad, row+pad+1), slice(col-pad, col+pad+1))
    return mask

def get_timetrace_square(video, point, pad=2):
    """Returna a timetrace from `video` by averaging a square pixel selection.
    """
    mask = get_roi_square(point, pad)
    timetrace = video[mask].mean(1).mean(1)
    return timetrace


def get_roi_circle(point, clip_radius, shape2d):
    """Return a circular selection of pixels around `point`.
    """
    # Get integer and fractional part of the point coordinate
    xfrac, col = np.modf(point[0])
    yfrac, row = np.modf(point[1])
    assert col + xfrac == point[0]
    assert row + yfrac == point[1]
    iclip = np.ceil(clip_radius)

    Y, X = np.mgrid[-iclip - 2:iclip + 3, -iclip - 2: iclip + 3]
    R = np.sqrt((X - xfrac)**2 + (Y - yfrac)**2)
    local_mask = R <= clip_radius + 0.5
    total_mask = np.zeros(shape2d, dtype=bool)
    total_mask[-iclip - 2 + row: iclip + 3 + row,
               -iclip - 2 + col: iclip + 3 + col] = local_mask
    imask = np.nonzero(total_mask)
    return imask

def get_timetrace_circle(video, point, clip_radius=2):
    """Returna a timetrace from `video` by averaging a "circular" region.
    """
    imask = get_roi_circle(point, clip_radius, video.shape[1:])
    timetrace = np.zeros(video.shape[0])
    for i in range(timetrace.size):
        timetrace[i] = video[i, imask[0], imask[1]].mean()
    return timetrace


def get_timetrace(video, point, clip_radius=1.5, detrend_sigma=250):
    """Returna a processed timetrace from a "circular" region.

    The timetrace processing removes the mean and applies a detrend filter
    that is a time-domain Gaussian filter.
    """
    timetrace = get_timetrace_circle(video, point, clip_radius=clip_radius)
    timetrace -= timetrace.mean()
    # Detrend very slow variations
    if detrend_sigma is not None and detrend_sigma > 0:
        timetrace -= ndi.filters.gaussian_filter1d(timetrace, detrend_sigma)
    return timetrace

def get_on_periods_slices(timetrace, threshold, lowpass_sigma=15, align=4):
    """Returns a list of slices selecting the on-periods during blinking.
    """
    lp_timetrace = ndi.filters.gaussian_filter1d(timetrace, lowpass_sigma)
    on_mask = lp_timetrace >= threshold

    on_periods = []
    on = False
    for i in range(0, timetrace.size, align):
        if on_mask[i:i+align].all():
            if not on:
                # ON period started
                on = True
                start = i
        elif on:
            # ON period ended
            on = False
            stop = i  # last sample is (i-1)
            on_periods.append(slice(start, stop))
    if on:
        on_periods.append(slice(start, timetrace.size))
    return on_periods

def get_on_periods_timetrace(timetrace, threshold, lowpass_sigma=15, align=4):
    """Compute a timetrace by stitching together on-periods of blinking.

    Returns:
        A tuple of arrays (time, trace).
    """
    on_slices = get_on_periods_slices(timetrace, threshold,
                                      lowpass_sigma=lowpass_sigma, align=align)
    time = np.arange(timetrace.size)
    time_list, trace_list = [], []

    time_list = (time[on_slice] for on_slice in on_slices)
    trace_list = (timetrace[on_slice] for on_slice in on_slices)
    return np.hstack(time_list), np.hstack(trace_list)


##
#  Averaging functions
#
def block_average(samples, num_samples):
    """Return an array of non-overlapping n-elements averages.
    """
    assert num_samples > 0
    if num_samples == 1: return samples

    avg_size = samples.size // num_samples
    avg_data = samples[:avg_size*num_samples]
    return avg_data.reshape(avg_size, num_samples).mean(axis=1)

def running_average(samples, num_samples):
    """Return an array of n-elements running average.
    """
    assert num_samples > 0
    if num_samples == 1: return samples

    weigths = np.repeat(1.0, num_samples)/num_samples
    return np.convolve(samples, weigths, mode='valid')

def avg_nsamples(samples, num_samples, running_avg=True, offset=0):
    """Return an array with `samples` averaged every `num_samples`.

    Arguments:
        running_avg (bool): if True, computes the average sliding the window
            by one sample a time (running average). Else computes the average
            in "blocks", i.e. on each non-overlapping adjacent windows.
        offset (int): number of initial samples to drop in the input array.
    """
    samples = samples[offset:]
    if running_avg:
        avg_func = running_average
    else:
        avg_func = block_average
    return avg_func(samples, num_samples=num_samples)


##
#  Alternation detection
#
def edge_diff_sum(timetrace, offset=0):
    """Return an array of rising/falling edges differences.

    This function takes the full 4-frame per period timetrace,
    applies a 2-sample average, and compute the rising/falling edge
    differences. The offset is applied before the 2-frame averaging.

    After the 2-sample average we compute:

    0   2   4   6
    *   *   *   *
     \ / \ / \ /            (t[0] - t[1]) + (t[2] - t[1])
      *   *   *  (*)
      1   3   5   7

    the sum of all the consecutive falling and rising edges marked in
    figure. Note that the last sample (i.e. 7) is always discarded.
    """
    avg_timetrace = block_average(timetrace[offset:], num_samples=2)
    return avg_timetrace[:-2:2] - 2*avg_timetrace[1:-1:2] + avg_timetrace[2::2]


def edge_diff_1pair(timetrace, offset=0, first_pair=True):
    """Return an array of non-overlapping pairs differences.

    This function takes the full 4-frame per period timetrace,
    applies a 2-sample average, and compute the rising or falling edge
    differences. The offset is applied before the 2-frame averaging.

    After the 2-sample average we compute:

    0   2   4   6
    *   *   *   *     t[0] - t[1], t[2] - t[3], ...  if first_pair == True
     \ / \ / \ /      t[2] - t[1], t[4] - t[3], ...  if first_pair == False
      *   *   *  (*)
      1   3   5   7

    the marked falling (first_pair=True) or rising (first_pair=False)
    edges. Note that the last sample (i.e. 7) is always discarded.
    """
    avg_timetrace = block_average(timetrace[offset:], num_samples=2)
    if first_pair:
        first_term = avg_timetrace[:-2:2]
    else:
        first_term = avg_timetrace[2::2]
    return first_term - avg_timetrace[1:-1:2]

def edge_diff_all(timetrace, offset=0):
    """Return an array of all the differences with alternating sign.

    This function takes the full 4-frame per period timetrace,
    applies a 2-sample average, and compute the rising or falling edge
    differences. The offset is applied before the 2-frame averaging.

    After the 2-sample average, we compute:

    0   2   4   6
    *   *   *   *     (t[0] - t[1]),
     \ / \ / \ / \    (t[2] - t[1]),
      *   *   *   *   (t[2] - t[3]), ...
      1   3   5   7

    that is the array of differences with alternating sign.
    """
    avg_timetrace = block_average(timetrace[offset:], num_samples=2)
    res = np.diff(avg_timetrace)
    res[::2] *= -1
    return res

##
#  Tests
#
def test_edge_diff_all(timetrace, offset=0):
    diff_all = edge_diff_all(timetrace, offset=offset)
    even_size = (diff_all.size // 2)*2
    diff1 = edge_diff_1pair(timetrace, offset=offset, first_pair=True)
    diff2 = edge_diff_1pair(timetrace, offset=offset, first_pair=False)
    msg =  'The two arrays differ.'
    assert np.allclose(diff_all[:even_size:2], diff1), msg
    assert np.allclose(diff_all[1:even_size:2], diff2), msg

def test_edge_diff_sum(timetrace, offset=0):
    diff1 = edge_diff_sum(timetrace, offset=offset)
    diff2 = edge_diff_1pair(timetrace, offset=offset, first_pair=True) + \
            edge_diff_1pair(timetrace, offset=offset, first_pair=False)
    assert np.allclose(diff1, diff2), 'The two arrays differ.'
