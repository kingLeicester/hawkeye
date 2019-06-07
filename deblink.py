# coding: utf8

# Written by Nathan Vack <njvack@wisc.edu> at the Waisman Laborotory
# for Brain Imaging and Behavior, University of Wisconsin - Madison.

import numpy as np


def find_zero_areas(array):
    """
    Return a list of (start, end length) tuples for each run of zeros in
    the input array
    """
    # First, mark the ends of the ends of the array with 1 in case the
    # array starts or ends with 0
    padded_array = np.hstack((1, array, 1)).astype(bool)
    inverted = np.invert(padded_array).astype(int)
    deriv = np.diff(inverted)
    starts = np.where(deriv > 0)[0]
    ends = np.where(deriv < 0)[0]
    lengths = ends - starts
    return zip(starts, ends, lengths)
