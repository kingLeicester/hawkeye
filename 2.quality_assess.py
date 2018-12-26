#!/usr/bin/env python3

# Very much in the works - David Lee
# Please do not use it to analyze eyetracking just yet! (10/16/2018)

import pandas as pd
import os
import sys
import re
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from scipy import signal
import deblink
import nystrom_saccade_detector

from hawkeye import GazeReader
from hawkeye import EPrimeReader
from hawkeye import GazeDenoisor

subject_number = sys.argv[1]


#--------------------Gaze Data--------------------
# Read in gaze data 
gaze_reader = GazeReader(subject_number)
gaze = gaze_reader.read_gaze_data()

#gaze.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/MIDUSref_startle_order1_FINAL_VERSION-%s-%s.csv"%(subject_number, subject_number))

#--------------------E-prime Data--------------------
# Convert Eprime file in to tsv
eprime_reader = EPrimeReader(subject_number)
e_prime = eprime_reader.read_eprime_data()

#--------------------Gaze and E-prime Data Merged--------------------
data_merged = pd.merge(gaze, e_prime, on='image')

#--------------------Sampling Rate--------------------
gaze_denoisor = GazeDenoisor()

sample_per_second = gaze_denoisor.compute_sampling_rate(data_merged)
print ("Sampling rate for %s: %s/s"%(subject_number, sample_per_second))

one_sample_time = round((1/sample_per_second) * 1000, 1)

#--------------------Denoising1: Remove 6 Practice Picture, Pause, and Fixation Cross (~1000ms) Trials (Applies Universally)--------------------

data_merged = gaze_denoisor.denoise_practice_and_pause(data_merged)

#### Total number of raw trials (after denoising 1 though)
raw_gaze_count = len(data_merged.index)
print ("raw_trials: " + str(raw_gaze_count))

data_merged = gaze_denoisor.denoise_fixation_cross(data_merged)

### Total number of trials after Denoising #1
pre_denoise_gaze_count = len(data_merged.index)
print ("pre_denoise_trials: " + str(pre_denoise_gaze_count))

### Total number of stims (IAPS) after Denoising #1
preDenoise_imageList = data_merged['image'].unique()
preDenoise_stim_count = str(len(preDenoise_imageList))
print ("pre_denoise_stim_count: " + preDenoise_stim_count)

### Figure out indexing before further denoising (later used in constructing plots in all 4000ms)
indexLengthDict = {}
indexListDict = {}
for image in preDenoise_imageList:
	sample_image = data_merged.loc[data_merged['image'] == image]
	minIndex = (min(sample_image.index))
	maxIndex = (max(sample_image.index))
	#indexRange= range(minIndex, maxIndex-1)
	# Range Max is exclusive (add 1 to the max to make it inclusive)
	indexRange= range(minIndex, maxIndex + 1)
	indexLengthDict[image] = len(indexRange)
	indexListDict[image] = indexRange

#--------------------Denoising2: Filter by Validity (Applies Differently by Subject)--------------------

##### Filter data by validity
data_denoised = gaze_denoisor.denoise_invalid(data_merged)

# Total number of trials after Denoising #2
post_denoise_gaze_count = len(data_denoised.index)
print ("post_denoise_trials: " + str(post_denoise_gaze_count))

# Total number of stim count after Denoising #2
postDenoise_imageList = data_denoised['image'].unique()
postDenoise_stim_count = str(len(postDenoise_imageList))
print ("post_denoise_stim_count: " + postDenoise_stim_count)


# Figure out which Stim has been removed due to Denoising #2
missingIAPSList = list(set(preDenoise_imageList) - set(postDenoise_imageList))
#print (missingIAPSList)

# Compare missingIAPSList to the Original, figure out which Nth element is missing
missing_stim_number_list = [] 
for index, stim in enumerate(preDenoise_imageList):
	for missingIAPS in missingIAPSList:
		if missingIAPS == stim:
			stim_number = "stim_" + str(index + 1)
			missing_stim_number_list.append(stim_number)
#rint (missing_stim_number_list)

# Total valid data after Denoising #2
percent_good_data_subject = round((post_denoise_gaze_count/pre_denoise_gaze_count) * 100, 2)
print ("Percent Good Data for subject {}: {}%".format(subject_number, percent_good_data_subject))

#--------------------Percent Good Data--------------------
IAPSList = [] + missingIAPSList
GoodPercentList = [0] * len(missing_stim_number_list)

# Create a list with stim1~stim90
default_stim_list = []
for i in range(1,91):
	stim_name = "stim_" + str(i)
	default_stim_list.append(stim_name)

# Combine stimlist with preDenoise image list in a dataframe (in order)
if len(preDenoise_imageList) == 90 & len(default_stim_list) == 90:
	IAPS_stim_df = pd.DataFrame({"IAPS":preDenoise_imageList,
		"STIM":default_stim_list})

# Check if there are 90 IAPS
if len(IAPS_stim_df.index) == 90:
	print ("There are 90 IAPS, good to go")
else:
	print ("There needs to be exactly 90 IAPS images presented, please check preprocessing")

# Compute good percent data 
for image in postDenoise_imageList:

	# Work with data relavant to single IAPS image at a time
	single_image_df = data_denoised.loc[data_denoised['image'] == image]

	# Number of good raw trials (before interpolation)
	raw_trials = len(single_image_df)

	# Figure out missing values due to previous denoising and fill in with "NaN"
	indexList = indexListDict[image]
	indexLength = indexLengthDict[image]
	#print ("Original Index Range: {}".format(indexList))
	#print ("Original Index Length: {}".format(indexLength))

	single_image_df = single_image_df.reindex(indexList, fill_value=np.nan)

	# Re-set the index from 0 
	# "drop = True" drops old indices
	# "inplace = True" modifies the DataFrame in place (do not create a new object)
	# Here both are used to preserve "nan" values
	single_image_df.reset_index(drop=True, inplace=True)

	# Number of total trials (after interpolation)
	re_indexed_trials = len(single_image_df)

	index_verfication = re_indexed_trials/indexLength
	if index_verfication == 1:
		print ("Index Length verification successful")

	else:
		print ("Index Length verification failure, recheck index values")

	# Percent raw data for single IAPS
	try:
		percent_good_data_IAPS = round((raw_trials/re_indexed_trials) * 100, 2)

	except ZeroDivisionError:
		percent_good_data_IAPS = 0

	# Append to lists to be used in later QA steps
	IAPSList.append(image)
	GoodPercentList.append(percent_good_data_IAPS)

# Create a dataframe with IAPS and Good Percentage
IAPS_good_percent_df = pd.DataFrame({"IAPS": IAPSList,
	subject_number: GoodPercentList})

# Merge IAPS, Good Percentage, and Stim #
QA_df = pd.merge(IAPS_stim_df, IAPS_good_percent_df, how='left', left_on=['IAPS'], right_on=['IAPS'])

# Separate into stim good data and IAPS good data
good_data_IAPS_df = QA_df[['IAPS', subject_number]]
good_data_IAPS_df_transposed = good_data_IAPS_df.T
good_data_IAPS_df_transposed.insert(loc=0, column="TotalValid", value=["TotalValid", percent_good_data_subject])
good_data_IAPS_df_transposed.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/%s/%s_valid_IAPS.csv"%(subject_number, subject_number), header=False)

good_data_stim_df = QA_df[['STIM', subject_number]]
good_data_stim_df_transposed = good_data_stim_df.T
good_data_stim_df_transposed.insert(loc=0, column="TotalValid", value=["TotalValid", percent_good_data_subject])
good_data_stim_df_transposed.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/%s/%s_valid_STIM.csv"%(subject_number, subject_number), header=False)

