# coding: utf8
# Part of the gazehound package for analzying eyetracking data
#
# Copyright (c) 2010 Board of Regents of the University of Wisconsin System
#
# Written by Nathan Vack <njvack@wisc.edu> at the Waisman Laborotory
# for Brain Imaging and Behavior, University of Wisconsin - Madison.

# Edited by David Lee, 2019

import numpy as np
from scipy.ndimage import maximum_filter1d as max_flt
from scipy.stats import scoreatpercentile
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

class AdaptiveDetector(object):
    """
    A saccade detector employing an adaptive velocity threshold.

    References:

    [1] M. Nyström, K. Holmqvist, An adaptive algorithm for fixation,
    saccade, and glissade detection in eyetracking data. Behavior Research
    Methods, 2010. doi:10.3758/BRM.42.1.188
    """

    def __init__(
            self, point_array, samples_per_second=120,
            clip_speed_percent=99.5, minimum_fixation_ms=60,
            threshold_start_percent=99.5, threshold_sd_scale=2.5,
            threshold_min_change=0.001, threshold_max_iters=10000):

        self._p_arr = point_array
        self.samples_per_second = samples_per_second
        self.minimum_fixation_ms = minimum_fixation_ms
        self.clip_speed_percent = clip_speed_percent
        self.threshold_start_percent = threshold_start_percent
        self.threshold_sd_scale = threshold_sd_scale
        self.threshold_min_change = threshold_min_change
        self.threshold_max_iters = threshold_max_iters

        self.__minimum_fixation_width = int(np.round(
            (self.samples_per_second/1000.0) * minimum_fixation_ms
        ))
        self._sg_filter_width = self.__minimum_fixation_width
        # This must be an odd number.
        if (self._sg_filter_width % 2) == 0:
            self._sg_filter_width += 1
        #self._compute_saccades()


    def _find_threshold(self, speeds, start_percentile, min_change_frac,
            sd_threshold_scale, max_iters):
        # Speed is *normalized
        # Find a starting value
        considered = speeds

        	#considered[index,1] = 0
        # remove 0's when computing threshold
        considered = considered[considered != 0]
        #print (considered)
        thresh = scoreatpercentile(considered, start_percentile)
        current_change_frac = 1
        self._thresh_iters = 0
        while (current_change_frac > min_change_frac and
                self._thresh_iters < max_iters):
            self._thresh_iters += 1
            # Anything lower than 95% percentile of speed, and avergage + SD of choice. 
            considered = considered[considered < thresh]
            new_thresh = (np.mean(considered) +
                            sd_threshold_scale*np.std(considered))
            current_change_frac = abs((new_thresh-thresh)/thresh)
            thresh = new_thresh

            #print ("thresh:" + str(thresh))
            #print((self._thresh_iters, thresh, current_change_frac))

        if self._thresh_iters == max_iters: thresh = None

        return thresh


    # Basically removes weird outliers above and beyond 99.5 percentile
    def _clamp_to_percentile(self, arr, percentiles):
        return arr.clip(
            scoreatpercentile(arr, percentiles[0]),
            scoreatpercentile(arr, percentiles[1]))

    def compute_drop_index(self):
    	#print (self._p_arr)
        # extract indices 0,0 coordinates
        missing_data_index = np.where(self._p_arr == 0)
        #missing_data_index = [x-1 for x in missing_data_index]
        # remove repeated indices
        missing_data_index = set(missing_data_index[0].tolist())
        #print (len(missing_data_index))
        import more_itertools as mit
        # create a list of consecutive indices
        missing_data_group = [list(group) for group in mit.consecutive_groups(missing_data_index)]
        # extract the first index of all groups
        #drop_index_list = [group[0] for group in missing_data_group]

        drop_index_list = []
        for group in missing_data_group:
        	drop_index_list.append(group[0])
        	drop_index_list.append(group[-1])

        return (drop_index_list)
 

    def compute_speed(self):
        self._p_diffs = np.apply_along_axis(
            sgolay, 0, self._p_arr, self._sg_filter_width, 2, 1)
        #speed_X = self._p_diffs[:,0]
        #speed_Y = self._p_diffs[:,1]
        csp = self.clip_speed_percent
        clamped = self._clamp_to_percentile(
            self._p_diffs, (100 - csp, csp))

        def normalize(arr):
            return arr / np.max(np.abs(arr))
        self._normed = np.apply_along_axis(normalize, 0, clamped)
        
        speed_X = self._normed[:,0]
        speed_Y = self._normed[:,1]

        # find velocity peak (all postivie values)
        self._speeds = np.sqrt(np.apply_along_axis(np.sum, 1, self._normed**2))

        #drop_index_list = self.compute_drop_index()
        #for index in drop_index_list:
        	#print (self._speeds[index-2:index+5])
        	#self._speeds[index-2:index+5] = 0
            #print (self._speeds[index-2:index+5])
            #self._speeds[index] = 0
  

        speed_combined = self._speeds
        #pos_speeds = np.abs(normed)
        #averages = np.apply_along_axis(np.mean, 1, pos_speeds)
        # Find the peaks
        max_filtered = max_flt(self._speeds, self.__minimum_fixation_width)
     
        # This returns boolean array
        self._peak_mask = (max_filtered == self._speeds)

        # This turns True into Value and False into 0
        self._peaks = self._speeds*self._peak_mask


        # remove false saccades at 0 index
        drop_index_list = self.compute_drop_index()

        # remove saccades caused by missing data
        for index in drop_index_list:
            self._peaks[index-1: index+2] = 0
            self._peaks[0] = 0
    
        self._threshold = self._find_threshold(
            self._speeds,
            self.threshold_start_percent,
            self.threshold_min_change,
            self.threshold_sd_scale,
            self.threshold_max_iters)

        return (speed_X, speed_Y, speed_combined, self._peaks, self._threshold, self._p_diffs)

    def _compute_saccades(self):
        self._p_diffs = np.apply_along_axis(
            sgolay, 0, self._p_arr, self._sg_filter_width, 2, 1)

   #      drop_index_list = self.compute_drop_index()
   #      for index in drop_index_list:
   #      	#print (self._p_diffs[339])
   #      	#exit()
   #      	self._p_diffs[index,0] = 0
   #      	self._p_diffs[index,1] = 0
			# #(105, 339)
   #      print (self._p_diffs[339])
        #speed_X = self._p_diffs[:,0]
        #speed_Y = self._p_diffs[:,1]
        csp = self.clip_speed_percent
        clamped = self._clamp_to_percentile(
            self._p_diffs, (100 - csp, csp))

        def normalize(arr):
            return arr / np.max(np.abs(arr))
        self._normed = np.apply_along_axis(normalize, 0, clamped)
        # combine both X and Y speed , but why? should I treat is separately?
        self._speeds = np.sqrt(np.apply_along_axis(np.sum, 1, self._normed**2))

        #drop_index_list = self.compute_drop_index()
        #for index in drop_index_list:
        	# print (self._speeds[index-2:index+5])
        	# self._speeds[index-2:index+5] = 0
        	# print (self._speeds[index-2:index+5])
            #self._speeds[index] = 0

        #pos_speeds = np.abs(normed)
        #averages = np.apply_along_axis(np.mean, 1, pos_speeds)
        # Find the peaks
        #print ("minimum fixation width: " + str(self.__minimum_fixation_width))
        max_filtered = max_flt(self._speeds, self.__minimum_fixation_width)
        self._peak_mask = (max_filtered == self._speeds)
        self._peaks = self._speeds*self._peak_mask

        # remove false saccades at 0 index
        drop_index_list = self.compute_drop_index()

        # remove saccades caused by missing data
        for index in drop_index_list:
            self._peaks[index-1: index+2] = 0
            self._peaks[0] = 0
    
        self._threshold = self._find_threshold(
            self._speeds,
            self.threshold_start_percent,
            self.threshold_min_change,
            self.threshold_sd_scale,
            self.threshold_max_iters)
        #print ("final threshold:" + str(self._threshold))
        self._candidates = self._peaks >= self._threshold

def sgolay(y, window_size, order, deriv=0):
    #y = y[y != "missing"]
    """
    Implementation of the Savitzky-Golay filter -- taken from:
    http://www.scipy.org/Cookbook/SavitzkyGolay

    Note that this seems to be implemented in modern versions of scipy:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.savgol_filter.html#scipy.signal.savgol_filter

    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-5, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv]


    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])


    # Add all the arrays together
    y = np.concatenate((firstvals, y, lastvals))


    # returns positions where two arrays completely overlap
    return np.convolve( m, y, mode='valid')
