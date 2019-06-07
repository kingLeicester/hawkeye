#!/usr/bin/env python3

### Reminder Notes before Processing
# MERGE Eprime and Eye-Tracking data before applying any filtering
# Sampling rate varies by subject (last 28 subjects of MIDSREF has 60 sample/S) - always compute sampling rate in the beggning 
# Resample AOIs 

import pandas as pd
import os
import sys
import re
import math
import numpy as np
import matplotlib
#matplotlib.use('Agg') # Comment this out if graphing is not working
import matplotlib.pyplot as plt
import glob
from scipy import signal

from hawkeye import GazeReader
from hawkeye import EPrimeReader
from hawkeye import GazeDenoisor
from hawkeye import AOIReader
from hawkeye import AOIScalar
from hawkeye import SignalDenoisor
from hawkeye import SaccadeDetector
from hawkeye import FixationDetector
from hawkeye import GazeCompiler
from hawkeye import Processor

# These are Nate Vack's Work. Should be included in the package with Nate Vack's ownership
import deblink
import nystrom_saccade_detector

subject_number = sys.argv[1]

coordinate_limits = (0, 1280)
sample_limits = (0, 500)
median_width_max = 1.0 / 20
max_blink_sec = 0.4
minimum_fixation_duration = 60
maximum_gap_duration = 75

processor = Processor()
postDenoise_imageList, data_denoised, indexListDict, indexLengthDict, sample_per_second, maximum_gap_threshold, percent_good_data_subject, one_sample_time = processor.process(subject_number, maximum_gap_duration)

# #--------------------Gaze Data--------------------
# # Read in gaze data 
# gaze_reader = GazeReader(subject_number)
# gaze = gaze_reader.read_gaze_data()

# #gaze.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/MIDUSref_startle_order1_FINAL_VERSION-%s-%s.csv"%(subject_number, subject_number))

# #--------------------E-prime Data--------------------
# # Convert and read-in Eprime file in to tsv
# eprime_reader = EPrimeReader(subject_number)
# e_prime = eprime_reader.read_eprime_data()

# #--------------------Gaze and E-prime Data Merged--------------------
# data_merged = pd.merge(gaze, e_prime, on='image')

# #--------------------Sampling Rate--------------------
# gaze_denoisor = GazeDenoisor()

# sample_per_second = gaze_denoisor.compute_sampling_rate(data_merged)
# print ("-Sampling rate for %s: %s/s"%(subject_number, sample_per_second))

# one_sample_time = round((1/sample_per_second) * 1000, 1)
# print (f"-One sample time: {one_sample_time}ms")

# #--------------------Interpolation Threshold--------------------
# # This is the maximum number of conseuctive missing data that will be interpolated. Anything more than 9 trials missing in a row, leave it NaN (do NOT interpolate)
# if sample_per_second == 120.0:
# 	maximum_gap_threshold = 9
# 	print (f"-threshold for interpolating: {maximum_gap_threshold} samples")
# else:
# 	#compute new thershold to nearest whole number
# 	maximum_gap_threshold = round(maximum_gap_duration/one_sample_time)
# 	print (f"-new threshold for interpolating: {maximum_gap_threshold} samples")

# #--------------------Denoising1: Remove 6 Practice Picture, Pause, and Fixation Cross (~1000ms) Trials (Applies Universally)--------------------
# data_merged = gaze_denoisor.denoise_practice_and_pause(data_merged)
# # Comment out to include the fixation cross time
# #data_merged = gaze_denoisor.denoise_fixation_cross(data_merged)

# ### Total number of smples after Denoising #1
# raw_gaze_count = len(data_merged.index)
# print ("-raw_sample_count: " + str(raw_gaze_count))

# ### Total number of stims (IAPS) after Denoising #1
# raw_image_list = data_merged['image'].unique()
# raw_stim_count = str(len(raw_image_list))
# print ("-raw_stim_count: " + raw_stim_count)

# ### Figure out indexing before further denoising (later used in constructing plots in all 4000ms)
# indexLengthDict = {}
# indexListDict = {}
# for image in raw_image_list:
# 	sample_image = data_merged.loc[data_merged['image'] == image]
# 	minIndex = (min(sample_image.index))
# 	maxIndex = (max(sample_image.index))
# 	indexRange= range(minIndex, maxIndex+1)
# 	indexLengthDict[image] = len(indexRange)
# 	indexListDict[image] = indexRange

# #--------------------Denoising2: Filter by Validity (Applies Differently by Subject)--------------------

# ##### Filter data by validity
# data_denoised = gaze_denoisor.denoise_invalid(data_merged)

# # Total number of trials after Denoising #2
# post_denoise_gaze_count = len(data_denoised.index)
# print ("-post_denoise_sample_count: " + str(post_denoise_gaze_count))

# # Total number of stim count after Denoising #2
# postDenoise_imageList = data_denoised['image'].unique()
# postDenoise_stim_count = str(len(postDenoise_imageList))
# print ("-post_denoise_stim_count: " + postDenoise_stim_count)

# # Figure out which Stim has been removed due to Denoising #2
# missingIAPSList = list(set(raw_image_list) - set(postDenoise_imageList))
# print ("-missing_IAPS", missingIAPSList)

# # Compare missingIAPSList to the Original, figure out which Nth element is missing
# missing_stim_number_list = [] 
# for index, stim in enumerate(raw_image_list):
# 	for missingIAPS in missingIAPSList:
# 		if missingIAPS == stim:
# 			stim_number = "stim_" + str(index + 1)
# 			missing_stim_number_list.append(stim_number)
# print ("-missing_stim_number", missing_stim_number_list)

# # Total valid data after Denoising #2
# percent_good_data_subject = round((post_denoise_gaze_count/raw_gaze_count) * 100, 2)
# print ("=====Percent Good Data for subject {}: {}% (Out of 4s picture onset time)=====".format(subject_number, percent_good_data_subject))

# # #--------------------AOI data--------------------
# # aoi_reader = AOIReader()

# # aoi = aoi_reader.read_in_AOI(data_denoised)

# # rectangle_aoi_df = aoi[(aoi['AOItype'] == 'Rectangle')]
# # rectangle_aoi_df = rectangle_aoi_df.reset_index(drop=True)

# # ellipse_aoi_df = aoi[(aoi['AOItype'] == 'Ellipse')]
# # ellipse_aoi_df = ellipse_aoi_df.reset_index(drop=True)

# # aoi_scalar = AOIScalar()

# # ### Refine coordinates for Rectangles
# # rectangle_aoi_data = aoi_scalar.scale_rectangle_aoi(rectangle_aoi_df)

# # ### Refine coordinates for Ellipses
# # ellipse_aoi_data = aoi_scalar.scale_ellipse_aoi(ellipse_aoi_df)


#--------------------Raw vs. Interpolated Data--------------------
for image in postDenoise_imageList:
#image = 2190

	# Work with data relavant to single IAPS image at a time
	single_image_df = data_denoised.loc[data_denoised['image'] == image]

	# # Number of good raw trials (before interpolation)
	# raw_trials = len(single_image_df)

	# Figure out missing values due to previous denoising and fill in with "NaN"
	indexList = indexListDict[image]

	single_image_df = single_image_df.reindex(indexList, fill_value=np.nan)

	# Re-set the index from 0 
	# "drop = True" drops old indices
	# "inplace = True" modifies the data_frame in place (do not create a new object)
	# Here both are used to preserve "nan" values
	single_image_df.reset_index(drop=True, inplace=True)

	# Used in denoising 3 step
	new_index_list = range(min(single_image_df.index), max(single_image_df.index)+1)
	# Fill in the empty coordinate columns with 0's
	#single_image_df['CursorX'] = single_image_df['CursorX'].fillna(0)
	#single_image_df['CursorY'] = single_image_df['CursorY'].fillna(0)

	# Plot Raw Data
	#coordinate_limits = (0, 1280)
	fig = plt.figure(figsize=(14, 4))
	plt.ylim(coordinate_limits)
	plt.xlim(sample_limits)
	plt.xticks(np.arange(0, 500, 50))
	fig.suptitle('subject%s %s Raw Data'%(subject_number, image))
	plt.ylabel("Coordinates")
	plt.xlabel("# Samples")
	plt.plot(single_image_df['CursorX'], 'b', alpha=0.8)
	plt.plot(single_image_df['CursorY'], 'r', alpha=0.8)
	plt.legend(['raw_X','raw_Y'], loc='upper left')
	#plt.show()

	os.makedirs('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/'.format(subject_number), exist_ok = True)
	print ("creating raw data plot for {}".format(image))
	fig.savefig('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/{}_{}_0.raw_data.png'.format(subject_number, subject_number, image))

	#--------------------Denoising 3: Median Filtering per each IAPS--------------------
	signal_denoisor = SignalDenoisor(median_width_max, max_blink_sec, sample_per_second, maximum_gap_duration, maximum_gap_threshold)
	# Handles short (1-sample) dropouts and x & y values surrounding blinks
	median_filtered_df = signal_denoisor.meidan_filter(single_image_df)

	median_filtered_df = median_filtered_df.reindex(new_index_list, fill_value=np.nan)

	# Create Offset values for QA purposes
	median_filtered_df['raw_x_offset_column'] = median_filtered_df['CursorX'] + 30
	median_filtered_df['raw_y_offset_column'] = median_filtered_df['CursorY'] + 30

	#Plot median filterded
	#coordinate_limits = (0, 1280)
	fig = plt.figure(figsize=(14, 4))
	plt.ylim(coordinate_limits)
	plt.xlim(sample_limits)
	plt.xticks(np.arange(0, 500, 50))
	fig.suptitle('subject%s %s Denoise 1: Median Filtered'%(subject_number, image))
	plt.ylabel("Coordinates")
	plt.xlabel("# Samples")
	plt.plot(median_filtered_df['raw_x_offset_column'], 'c', alpha=0.5)
	plt.plot(median_filtered_df['x_filtered'], 'b', alpha=0.8)
	plt.plot(median_filtered_df['raw_y_offset_column'], 'm', alpha=0.5)
	plt.plot(median_filtered_df['y_filtered'], 'r', alpha=0.8)
	plt.legend(['raw_X', 'filtered_X', 'raw_Y', 'filtered_Y'], loc='upper left')
	#plt.show()

	os.makedirs('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/'.format(subject_number), exist_ok = True)
	print ("creating median filtered plot for {}".format(image))
	fig.savefig('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/{}_{}_1.median_filtered.png'.format(subject_number, subject_number, image))

	#--------------------Denoising4: Remove Blinks--------------------
	# Blink range == 50 ~ 400ms
	# Currently cut-off long end of blinks
	# Inerpolate these cuts
	# Simple forward-fill (alternative would be linear interpolation)

	deblinked_df = signal_denoisor.remove_blinks(median_filtered_df)

	# Create Offset values for QA purposes
	deblinked_df['filtered_x_offset_column'] = deblinked_df['x_filtered'] + 30
	deblinked_df['filtered_y_offset_column'] = deblinked_df['y_filtered'] + 30

	# Plot deblinked
	fig = plt.figure(figsize=(14, 4))
	plt.ylim(coordinate_limits)
	plt.xlim(sample_limits)
	plt.xticks(np.arange(0, 500, 50))
	fig.suptitle('subject%s %s Denoise 2: Deblinked'%(subject_number, image))
	plt.ylabel("Coordinates")
	plt.xlabel("# Samples")
	plt.plot(deblinked_df['filtered_x_offset_column'], color='c', alpha=0.5)
	plt.plot(deblinked_df['x_deblinked'], color='b', alpha=0.8)
	plt.plot(deblinked_df['filtered_y_offset_column'], color='m', alpha=0.5)
	plt.plot(deblinked_df['y_deblinked'], color='r', alpha=0.8)
	plt.legend(['filtered_X', 'deblinked_X', 'filtered_Y', 'deblinked_Y'], loc='upper left')
	#plt.show()

	os.makedirs('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}'.format(subject_number), exist_ok = True)
	print ("creating deblinked plot for {}".format(image))
	fig.savefig('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/{}_{}_2.deblinked.png'.format(subject_number, subject_number, image))

	#--------------------Denoising5: Interpolate--------------------

	interpolated_df = signal_denoisor.interpolate(deblinked_df)
	signal_denoisor.compute_interpolation_ratio(interpolated_df)

	interpolated_df['deblinked_x_offset_column'] = interpolated_df['x_deblinked'] + 30
	interpolated_df['deblinked_y_offset_column'] = interpolated_df['y_deblinked'] + 30

	# Plot deblinked
	fig = plt.figure(figsize=(14, 4))
	plt.ylim(coordinate_limits)
	plt.xlim(sample_limits)
	plt.xticks(np.arange(0, 500, 50))
	fig.suptitle('subject%s %s Denoise 3: Interpolated'%(subject_number, image))
	plt.ylabel("Coordinates")
	plt.xlabel("# Samples")
	#plt.plot(interpolated_df['deblinked_x_offset_column'], color='c', alpha=0.5)
	plt.plot(interpolated_df['x_interpolated'], color='b', alpha=0.8)
	#plt.plot(interpolated_df['deblinked_y_offset_column'], color='m', alpha=0.5)
	plt.plot(interpolated_df['y_interpolated'], color='r', alpha=0.8)
	plt.legend(['deblinked_X', 'interpolated_X', 'deblinked_Y', 'interpolated_Y'], loc='upper left')
	plt.legend(['interpolated_X', 'interpolated_Y'], loc='upper left')
	#plt.show()

	os.makedirs('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}'.format(subject_number), exist_ok = True)
	print ("creating interpolated plot for {}".format(image))
	fig.savefig('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/{}_{}_3.interpolated.png'.format(subject_number, subject_number, image))

	#--------------------Detect Saccades--------------------
	saccade_detector = SaccadeDetector(sample_per_second)
	saccade_df, speed_X, speed_Y, speed_combined, filtered_peaks, threshold, speed_original = saccade_detector.detect_saccade(interpolated_df)


	# Get indices that are saccades 
	candidate_t = (saccade_df[saccade_df['saccade_candidate'] == True]).index

	number_peaks = len(candidate_t)

	#fig, ax1 = plt.figure(figsize=(14, 4))
	fig, ax1 = plt.subplots(figsize=(14, 4))
	plt.ylim(coordinate_limits)
	color = 'tab:gray'
	ax1.set_xlabel('Sample #')
	ax1.set_ylabel('Coordinates', color=color)
	plt.xticks(np.arange(0, 600, 50))


	#iloc[0:100].plot()
	#ax1.plot(fixation_included_df['CursorX'][:120], linestyle='--', color='b', linewidth=1, alpha=0.8)#, linewidth=7.0)
	ax1.plot(saccade_df['x_interpolated'], color='b', linewidth=1, alpha=0.8)#, linewidth=7.0)

	#ax1.plot(fixation_included_df['CursorY'][:120], linestyle='--', color='r', linewidth=1, alpha=0.8)#, linewidth=7.0)
	ax1.plot(saccade_df['y_interpolated'], color='r', linewidth=1, alpha=0.8)#, linewidth=7.0)
	plt.legend(['X', 'Y'], loc='upper left')

	# Place horizontal line at fixation cross location
	plt.axhline(640, 0, 0.01, linewidth=0.5, color='b', alpha = 0.5)
	plt.axhline(512, 0, 0.01, linewidth=0.5, color='r', alpha = 0.5)

	ax2 = ax1.twinx()
	color = 'tab:gray'
	plt.ylim(0,0.3)
	ax2.set_ylabel('Normalized Velocity', color = color)
	#ax2.plot(speed_combined, linewidth=0.5, color='b', alpha= 1.0)
	#ax2.plot(speed_Y, linewidth=0.5, color='r', alpha= 1.0)
	ax2.plot(filtered_peaks, linewidth=0.5, color='g', alpha= 0.5)

	#ax2.plot(speed_Y , color='c', alpha=0.8)
	#print (threshold)
	plt.axhline(threshold, 0, 1, linewidth=1, color='c', alpha = 1.0)
	#plt.axhline(0, 0, 1, linewidth=0.5, color='k', alpha = 1.0)

	
	# Place vertical line at picture onset time
	#plt.axvline(120, 0, 1, linewidth = 3, color='m', alpha = 1.0)

	for t in candidate_t:
		plt.axvline(t, 0, 1, linewidth=0.5, color='k', alpha = 1.0)


	fig.suptitle('subject%s %s Analysis 1: Saccades Detected'%(subject_number, image))
	#plt.ylabel("Coordinates")
	#plt.xlabel("# Samples")
	plt.legend(['Velocity', f'Threshold:{round(threshold, 5)}', f'Saccades:{number_peaks}'], loc='upper right')
	#plt.legend(['Velocity', f'Threshold:{round(threshold, 5)}', f'Picture Onset', f'Saccades:{number_peaks}'], loc='upper right')
	# Plot vertical lines at saccades
	# fig = plt.figure(figsize=(14, 4))
	# plt.ylim(coordinate_limits)
	# plt.xlim(sample_limits)
	# plt.xticks(np.arange(0, 500, 25)) 
	# plt.plot(speed_X, color='c', alpha=0.8)
	# plt.plot(saccade_df['x_interpolated'], color='b', alpha=0.8)
	# plt.plot(speed_Y , color='m', alpha=0.8)
	# plt.plot(saccade_df['y_interpolated'], color='r', alpha=0.8)
	# for t in candidate_t:
	#     plt.axvline(t, 0, 1, color='k')
	# fig.suptitle('subject%s %s Analysis 1: Saccades Detected'%(subject_number, image))
	# plt.ylabel("Coordinates")
	# plt.xlabel("# Samples")
	# plt.legend(['speed_X', 'X', 'speed_Y', 'Y'], loc='upper left')
	# exit()
	#plt.show()

	#saccade_df.to_csv('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/saccade.csv')

	# Create Plotsplt.suptitle('subject%s %s'%(subject_number, image))
	os.makedirs('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}'.format(subject_number), exist_ok = True)
	print ("creating saccade plot for {}".format(image))
	fig.savefig('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/{}_{}_4.saccade.png'.format(subject_number, subject_number, image))

	#--------------------Detect Fixations--------------------

	# Get indices that are not saccades (fixations)
	candidate_t = (saccade_df[saccade_df['saccade_candidate'] == False]).index

	# Create data_frame of all fixations
	fixation_df = saccade_df.loc[saccade_df['saccade_candidate'] == False]


print ("processing for %s complete without error"%(subject_number))
#single_image_df.to_csv("/home/slee/Desktop/eye_sample.csv")
#print (single_image_df)