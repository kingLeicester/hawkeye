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
import more_itertools as mit

from hawkeye import GazeReader
from hawkeye import EPrimeReader
from hawkeye import GazeDenoisor
from hawkeye import AOIReader
from hawkeye import AOIScalar
from hawkeye import SignalDenoisor
from hawkeye import SaccadeDetector
from hawkeye import FixationDetector
from hawkeye import GazeCompiler

subject_number = sys.argv[1]

coordinate_limits = (0, 1280)
median_with_max = 1.0 / 20
max_blink_sec = 0.4
minimum_fixation_duration = 60
maximum_gap_duration = 75

#--------------------Gaze Data--------------------
# Read in gaze data 
gaze_reader = GazeReader(subject_number)
gaze = gaze_reader.read_gaze_data()

#gaze.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/MIDUSref_startle_order1_FINAL_VERSION-%s-%s.csv"%(subject_number, subject_number))

#--------------------E-prime Data--------------------
# Convert and read-in Eprime file in to tsv
eprime_reader = EPrimeReader(subject_number)
e_prime = eprime_reader.read_eprime_data()

#--------------------Gaze and E-prime Data Merged--------------------
data_merged = pd.merge(gaze, e_prime, on='image')

#--------------------Sampling Rate--------------------
gaze_denoisor = GazeDenoisor()

sample_per_second = gaze_denoisor.compute_sampling_rate(data_merged)
print ("-Sampling rate for %s: %s/s"%(subject_number, sample_per_second))

one_sample_time = round((1/sample_per_second) * 1000, 1)
print (f"-One sample time: {one_sample_time}ms")

#--------------------Interpolation Threshold--------------------
# This is the maximum number of conseuctive missing data that will be interpolated. Anything more than 9 trials missing in a row, leave it NaN (do NOT interpolate)
if sample_per_second == 120.0:
	maximum_gap_threshold = 9
	print (f"-threshold for interpolating: {maximum_gap_threshold} samples")
else:
	#compute new thershold to nearest whole number
	maximum_gap_threshold = round(maximum_gap_duration/one_sample_time)
	print (f"-new threshold for interpolating: {maximum_gap_threshold} samples")

#--------------------Denoising1: Remove 6 Practice Picture, Pause, and Fixation Cross (~1000ms) Trials (Applies Universally)--------------------
data_merged = gaze_denoisor.denoise_practice_and_pause(data_merged)
data_merged = gaze_denoisor.denoise_fixation_cross(data_merged)

### Total number of smples after Denoising #1
raw_gaze_count = len(data_merged.index)
print ("-raw_sample_count: " + str(raw_gaze_count))

### Total number of stims (IAPS) after Denoising #1
raw_image_list = data_merged['image'].unique()
raw_stim_count = str(len(raw_image_list))
print ("-raw_stim_count: " + raw_stim_count)

### Figure out indexing before further denoising (later used in constructing plots in all 4000ms)
indexLengthDict = {}
indexListDict = {}
for image in raw_image_list:
	sample_image = data_merged.loc[data_merged['image'] == image]
	minIndex = (min(sample_image.index))
	maxIndex = (max(sample_image.index))
	indexRange= range(minIndex, maxIndex+1)
	indexLengthDict[image] = len(indexRange)
	indexListDict[image] = indexRange

#--------------------Denoising2: Filter by Validity (Applies Differently by Subject)--------------------

##### Filter data by validity
data_denoised = gaze_denoisor.denoise_invalid(data_merged)

# Total number of trials after Denoising #2
post_denoise_gaze_count = len(data_denoised.index)
print ("-post_denoise_sample_count: " + str(post_denoise_gaze_count))

# Total number of stim count after Denoising #2
postDenoise_imageList = data_denoised['image'].unique()
postDenoise_stim_count = str(len(postDenoise_imageList))
print ("-post_denoise_stim_count: " + postDenoise_stim_count)

# Figure out which Stim has been removed due to Denoising #2
missingIAPSList = list(set(raw_image_list) - set(postDenoise_imageList))
print ("-missing_IAPS", missingIAPSList)

# Compare missingIAPSList to the Original, figure out which Nth element is missing
missing_stim_number_list = [] 
for index, stim in enumerate(raw_image_list):
	for missingIAPS in missingIAPSList:
		if missingIAPS == stim:
			stim_number = "stim_" + str(index + 1)
			missing_stim_number_list.append(stim_number)
print ("-missing_stim_number", missing_stim_number_list)

# Total valid data after Denoising #2
percent_good_data_subject = round((post_denoise_gaze_count/raw_gaze_count) * 100, 2)
print ("=====Percent Good Data for subject {}: {}% (Out of 4s picture onset time)=====".format(subject_number, percent_good_data_subject))

#--------------------AOI data--------------------
# aoi_reader = AOIReader()

# aoi = aoi_reader.read_in_AOI(data_denoised)

# rectangle_aoi_df = aoi[(aoi['AOItype'] == 'Rectangle')]
# rectangle_aoi_df = rectangle_aoi_df.reset_index(drop=True)

# ellipse_aoi_df = aoi[(aoi['AOItype'] == 'Ellipse')]
# ellipse_aoi_df = ellipse_aoi_df.reset_index(drop=True)

# aoi_scalar = AOIScalar()

# ### Refine coordinates for Rectangles
# rectangle_aoi_data = aoi_scalar.scale_rectangle_aoi(rectangle_aoi_df)

# ### Refine coordinates for Ellipses
# ellipse_aoi_data = aoi_scalar.scale_ellipse_aoi(ellipse_aoi_df)

# Create a empty list for data to compile
percent_good_data_list = []
sampling_rate_list = []

subject_list = []
iaps_list = []
total_list = []
number_interpolated_list = []
number_original_list = []
number_missing_list = []
ratio_interpolated_list = []
ratio_original_list = []
ratio_missing_list = []

subject_number_list = []
valence_list = []
image_number_list = []
total_saccade_duration_list = []
aoi_type_list = []
object_number_list = []
time_to_first_fixation_list = []
first_fixation_duration_list = []
number_fixations_list = []
total_in_AOI_list = []
total_out_AOI_list = []
number_fixations_before_AOI_list = []

rectangle_aoi_subject_list = []
ellipse_aoi_subject_list = []
rectangle_aoi_list = []
ellipse_aoi_list = []
rectangle_aoi_iaps_list = []
ellipse_aoi_iaps_list = []

#--------------------Raw vs. Interpolated Data--------------------


for image in postDenoise_imageList:


# Work with data relavant to single IAPS image at a time
#image = postDenoise_imageList[4]

	print (image)
	single_image_df = data_denoised.loc[data_denoised['image'] == image]

	# # Number of good raw trials (before interpolation)
	# raw_trials = len(single_image_df)

	# Figure out missing values due to previous denoising and fill in with "NaN"
	indexList = indexListDict[image]

	single_image_df = single_image_df.reindex(indexList, fill_value=np.nan)

	# Re-set the index from 0 
	# "drop = True" drops old indices
	# "inplace = True" modifies the DataFrame in place (do not create a new object)
	# Here both are used to preserve "nan" values
	single_image_df.reset_index(drop=True, inplace=True)

	# Used in denoising 3 step
	new_index_list = range(min(single_image_df.index), max(single_image_df.index)+1)
	# Fill in the empty coordinate columns with 0's
	#single_image_df['CursorX'] = single_image_df['CursorX'].fillna(0)
	#single_image_df['CursorY'] = single_image_df['CursorY'].fillna(0)

	#--------------------Denoising 3: Median Filtering per each IAPS--------------------
	signal_denoisor = SignalDenoisor(median_with_max, max_blink_sec, sample_per_second, maximum_gap_duration, maximum_gap_threshold)

	# Handles short (1-sample) dropouts and x & y values surrounding blinks
	median_filtered_df = signal_denoisor.meidan_filter(single_image_df)

	median_filtered_df = median_filtered_df.reindex(new_index_list, fill_value=np.nan)

	# # Create Offset values for QA purposes
	# median_filtered_df['raw_x_offset_column'] = median_filtered_df['CursorX'] + 30
	# median_filtered_df['raw_y_offset_column'] = median_filtered_df['CursorY'] + 30

	#--------------------Denoising4: Remove Blinks--------------------
	# Blink range == 50 ~ 400ms
	# Currently cut-off long end of blinks
	# Inerpolate these cuts
	# Simple forward-fill (alternative would be linear interpolation)

	deblinked_df = signal_denoisor.remove_blinks(median_filtered_df)

	# # Create Offset values for QA purposes
	# deblinked_df['filtered_x_offset_column'] = deblinked_df['x_filtered'] + 30
	# deblinked_df['filtered_y_offset_column'] = deblinked_df['y_filtered'] + 30

	#--------------------Denoising5: Interpolate--------------------

	interpolated_df = signal_denoisor.interpolate(deblinked_df)


	total_number_samples, number_interpolated, number_original, number_missing = signal_denoisor.compute_interpolation_ratio(interpolated_df)
	if number_interpolated + number_original + number_missing == total_number_samples:
			print ("number of data checks out")
			missing_data_ratio = round((number_missing/total_number_samples) * 100, 2)
			interpolated_data_ratio = round((number_interpolated/total_number_samples) * 100, 2)
			original_data_ratio = round((number_original/total_number_samples) * 100, 2)
			print (f"percent missing data :{missing_data_ratio}% ({number_missing}/{total_number_samples})")
			print (f"percent interpolated data :{interpolated_data_ratio}% ({number_interpolated}/{total_number_samples})")
			print (f"percent origianl data :{original_data_ratio}% ({number_original}/{total_number_samples})")
			
			subject_list.append(subject_number)
			percent_good_data_list.append(percent_good_data_subject)
			sampling_rate_list.append(sample_per_second)
			iaps_list.append(image)
			total_list.append(total_number_samples)
			number_interpolated_list.append(number_interpolated)
			number_original_list.append(number_original)
			number_missing_list.append(number_missing) 
			ratio_interpolated_list.append(interpolated_data_ratio)
			ratio_original_list.append(original_data_ratio)
			ratio_missing_list.append(missing_data_ratio)

	else:
		print ("number of data does NOT check out, check you math!")

	#--------------------Detect Saccades--------------------
	saccade_detector = SaccadeDetector(sample_per_second)

	# Create a column with the points when saccades occur
	saccade_detected_df, speed_X, speed_Y, speed_combined, peak, threshold = saccade_detector.detect_saccade(interpolated_df)

	# Compute total saccade duration
	saccade_candidate_t = (saccade_detected_df[saccade_detected_df['saccade_candidate'] == True]).index
	total_saccade_duration = len(list(saccade_candidate_t)) * one_sample_time

	# Create a column with saccade intervals
	saccade_df = saccade_detector.compute_saccade_interval(saccade_detected_df)

	saccade_df = saccade_detector.detect_missing_data(saccade_df)

	saccade_df.to_csv(f"/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{subject_number}/saccade_data_{image}_{subject_number}.csv")

	#--------------------Valence Data--------------------
	valence_image_set = set(saccade_df['valence_x'].dropna())

	if len(valence_image_set) == 1:
		for valence in valence_image_set:
			valence = valence

	#--------------------AOI data--------------------
	aoi_reader = AOIReader()

	aoi = aoi_reader.read_in_AOI(image)


	if aoi.empty:
		rectangle_aoi_data = aoi
		ellipse_aoi_data = aoi

	else:
		rectangle_aoi_df = aoi[(aoi['AOItype'] == 'Rectangle')]
		rectangle_aoi_df = rectangle_aoi_df.reset_index(drop=True)

		ellipse_aoi_df = aoi[(aoi['AOItype'] == 'Ellipse')]
		ellipse_aoi_df = ellipse_aoi_df.reset_index(drop=True)

		aoi_scalar = AOIScalar()

		### Refine coordinates for Rectangles
		rectangle_aoi_data = aoi_scalar.scale_rectangle_aoi(rectangle_aoi_df)

		### Refine coordinates for Ellipses
		ellipse_aoi_data = aoi_scalar.scale_ellipse_aoi(ellipse_aoi_df)

	#--------------------Detect Fixations--------------------
	# Create dataFrame of only fixations
	fixation_df = saccade_df.loc[saccade_df['final_gaze_type'] == "fixation"]

	# --------------------Detect Fixations in IAPS--------------------
	fixation_detector = FixationDetector(one_sample_time, minimum_fixation_duration, image)

	# list of indices that are consdiered fixations
	fixation_in_IAPS_list = fixation_detector.detect_fixation(fixation_df)

	# subset to true fixations longer than minimum threshold 60ms
	total_number_fixations_in_IAPS, true_fixation_in_IAPS_list = fixation_detector.detect_true_fixation(fixation_in_IAPS_list)


	# create  dataframe of only true fixations
	true_fixation_labeld_df = fixation_detector.compute_final_data_type(true_fixation_in_IAPS_list, saccade_df)
	true_fixation_labeld_df.to_csv(f"/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{subject_number}/true_fixation_{image}_{subject_number}.csv")

	true_fixation_df = true_fixation_labeld_df.loc[true_fixation_labeld_df['final_data_type'] == "true_fixation"]

	# duration of all fixations
	total_duration_IAPS = fixation_detector.compute_total_duration_fixation(true_fixation_in_IAPS_list)

	# time to first fixation on IAPS
	if len(true_fixation_in_IAPS_list) >= 1:
		first_fixation_in_IAPS_index = true_fixation_in_IAPS_list[0]
		time_at_first_fixation_in_IAPS = round(first_fixation_in_IAPS_index[0] * one_sample_time, 2)
		first_fixation_in_IAPS_duration = round(len(first_fixation_in_IAPS_index) * one_sample_time, 2)

	else:
		first_fixation_in_IAPS_index = "N/A"
		time_at_first_fixation_in_IAPS = "N/A"
		first_fixation_in_IAPS_duration = "N/A"


	print ("==========Fixation on IAPS=========")
	print ("first fixating on IAPS at {}ms".format(time_at_first_fixation_in_IAPS))
	print ("total number of fixations in IAPS {} that last more than 60ms : {}".format(image, total_number_fixations_in_IAPS))
	print ("total duration of fixations in IAPS {} : {}ms".format(image, total_duration_IAPS))
	print ("")

	subject_number_list.append(subject_number)
	valence_list.append(valence)
	image_number_list.append(image)
	total_saccade_duration_list.append(total_saccade_duration)
	aoi_type_list.append("IAPS")
	object_number_list.append("Object01")
	time_to_first_fixation_list.append(time_at_first_fixation_in_IAPS)
	first_fixation_duration_list.append(first_fixation_in_IAPS_duration)
	number_fixations_list.append(total_number_fixations_in_IAPS)
	total_in_AOI_list.append(total_duration_IAPS)
	total_out_AOI_list.append("N/A")
	number_fixations_before_AOI_list.append("N/A")

	# --------------------Detect Fixations in AOI--------------------
	gaze_compiler = GazeCompiler()

	# Create dataFrame of Rectangle AOIs
	single_rectangle_aoi_df = rectangle_aoi_data.loc[rectangle_aoi_data['image'] == image]
	single_rectangle_aoi_df['subject_number']=subject_number

	# print last 4 columns (coordinates)
	rectangle_coordinates = single_rectangle_aoi_df[single_rectangle_aoi_df.columns[-4:]]
	print (rectangle_coordinates)
	#rectangle_aoi_subject_list.append(subject_number)
	#rectangle_aoi_list.append(single_rectangle_aoi_df)
	#rectangle_aoi_iaps_list.append(image)
	single_rectangle_aoi_df.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/{}_{}_rectangle_aoi.csv".format(subject_number, subject_number, image), index=False)


	if single_rectangle_aoi_df.empty:
		print ('No Rectangle AOI for {}'.format(image))
		print ("")

		# append only the image number
		subject_number_list.append(subject_number)
		valence_list.append(valence)
		image_number_list.append(image)
		total_saccade_duration_list.append(total_saccade_duration)
		aoi_type_list.append("N/A")
		object_number_list.append("N/A")
		time_to_first_fixation_list.append("N/A")
		first_fixation_duration_list.append("N/A")
		number_fixations_list.append("N/A")
		total_in_AOI_list.append("N/A")
		total_out_AOI_list.append("N/A")
		number_fixations_before_AOI_list.append("N/A")

	else:
		# Count the rows (how many AOIs for each IAPS?)
		number_aoi = len(single_rectangle_aoi_df.index)
		print ("==========Fixation on Rectangle AOI==========")
		print ("total number of Rectangle AOIs for IAPS {} : {}".format(image, number_aoi))
		print ("")

		# Start the AOI counter
		aoi_counter = 0
		
		# Start processing if there truly are more than 1 AOIs
		if number_aoi >= 1:
			while aoi_counter < number_aoi:
				aoi_number = single_rectangle_aoi_df.iloc[aoi_counter]['objectNumber']
				print ("=====AOI {}====".format(aoi_number))

				#Combine fixation data with AOI data
				#merged = fixation_df.merge(single_rectangle_aoi_df.iloc[[aoi_counter]], how='left').set_index(fixation_df.index)
				merged = true_fixation_df.merge(single_rectangle_aoi_df.iloc[[aoi_counter]], on=['image'])

				# Clean X,Y gaze and coordinates
				cleand_fixation_df = gaze_compiler.rectangle_clean_gaze_and_coordinate(merged)

				# Compute if a gaze is in the AOI grid or not
				fixation_in_aoi_df = gaze_compiler.rectangle_compute_gaze_in_AOI(cleand_fixation_df)

				# sometimes there are no fixations in AOIs
				if fixation_in_aoi_df.empty:
					print ('No Fixation for {} {}'.format(image, aoi_number))
					print ("")
					subject_number_list.append(subject_number)
					valence_list.append(valence)
					image_number_list.append(image)
					total_saccade_duration_list.append(total_saccade_duration)
					aoi_type_list.append("rectangle")
					object_number_list.append(aoi_number)
					time_to_first_fixation_list.append("N/A")
					first_fixation_duration_list.append("N/A")
					number_fixations_list.append("N/A")
					total_in_AOI_list.append("N/A")
					total_out_AOI_list.append("N/A")
					number_fixations_before_AOI_list.append("N/A")

					aoi_counter += 1

				# if there are at least one fixation in AOIs
				else:
					fixation_in_AOI_list = fixation_detector.detect_fixation(fixation_in_aoi_df)
					total_number_fixations_in_AOI, true_fixation_in_AOI_list = fixation_detector.detect_true_fixation(fixation_in_AOI_list)
					total_duration_AOI = fixation_detector.compute_total_duration_fixation(true_fixation_in_AOI_list)

					subject_number_list.append(subject_number)
					valence_list.append(valence)
					image_number_list.append(image)
					total_saccade_duration_list.append(total_saccade_duration)
					aoi_type_list.append("rectangle")
					object_number_list.append(aoi_number)
					number_fixations_list.append(total_number_fixations_in_AOI)

					print ("total number of fixations in Rectangle AOI that last more than 60ms : {}".format(total_number_fixations_in_AOI))
					print ("total duration of fixations in Rectangle AOI {} : {}ms".format(aoi_number, total_duration_AOI))
					print ("")

					if len(true_fixation_in_AOI_list) > 0:
						first_fixation_index = true_fixation_in_AOI_list[0]

						# Account for when missing data in the beggning?
						# ====================a.time to first fixation in AOI====================
						time_at_first_fixation_in_AOI = round(first_fixation_index[0] * one_sample_time, 2)
						print ("-----a. Initial fixation on AOI at {}ms-----".format(time_at_first_fixation_in_AOI))

						# ====================b.fixation duration of first fixation in AOI====================
						first_fixation_in_AOI_duration = round(len(first_fixation_index) * one_sample_time, 2)
						print ("-----b. fixation duration of first fixation in Rectangle AOI: {}ms-----".format(first_fixation_in_AOI_duration))
					
						time_to_first_fixation_list.append(time_at_first_fixation_in_AOI)
						first_fixation_duration_list.append(first_fixation_in_AOI_duration)

						# ====================c.total time fixating in AOIs compared to elsewhere in the picture====================
						percent_fixation_in_AOI = round((float(total_duration_AOI)/float(total_duration_IAPS)) * 100, 2)
						print ("-----c1. total duration of fixations in AOI: {}ms ({}%)-----".format(str(total_duration_AOI), str(percent_fixation_in_AOI)))
		
						total_in_AOI_list.append(str(total_duration_AOI))

						total_fixation_duration_outside_AOI = round(float(total_duration_IAPS) - float(total_duration_AOI), 2)
						percent_fixation_out_AOI = round((float(total_fixation_duration_outside_AOI)/float(total_duration_IAPS)) * 100, 2)
						print ("-----c2. total duration of fixations outside AOI: {}ms({}%)-----".format(str(total_fixation_duration_outside_AOI), str(percent_fixation_out_AOI)))
						print ("")
						
						total_out_AOI_list.append(str(total_fixation_duration_outside_AOI))
							
						# ====================d.number of fixations before fixating on AOIs====================
						if first_fixation_index[0] == 0:
							print ("{} started fixating on AOI as soon as IAPS was presented".format(subject_number))
							print ("-----d.number of fixations before fixation on AOI: 0-----")
							print ("")
							number_fixations_before_AOI_list.append("0")

						else: 
							if time_at_first_fixation_in_IAPS == time_at_first_fixation_in_AOI:
								print ("Initial fixation on IAPS and AOI match, good!")
								print ("-----d.number of fixations before fixation on AOI: 0-----")
								print ("")
								number_fixations_before_AOI_list.append("0")

							elif time_at_first_fixation_in_IAPS > time_at_first_fixation_in_AOI:
								print ("Initial Fixation on AOI, good!")
								print ("-----d.number of fixations before fixation on AOI: 0-----")
								print ("")
								number_fixations_before_AOI_list.append("0")

							elif time_at_first_fixation_in_IAPS < time_at_first_fixation_in_AOI:
								print ("Initial Fixation on IAPS, compute how many fixations before hitting AOI")
								
								# If any of fixations in IAPS occur before first fixating on AOI
								# They are already filtred for at minimum fixation duration (60ms)
								fixation_counter = 0
								for fixation in true_fixation_in_IAPS_list:
									if max(fixation) < first_fixation_index[0]:
										fixation_counter += 1
								
								number_fixations_before_AOI = fixation_counter
								number_fixations_before_AOI_list.append(fixation_counter)
								print ("-----d.number of fixations before fixation on AOI: {}-----".format(number_fixations_before_AOI))
								print ("")
							

					else:
						print ("no fixation longer than 60ms")

						time_to_first_fixation_list.append("N/A")
						first_fixation_duration_list.append("N/A")
						total_in_AOI_list.append("N/A")
						total_out_AOI_list.append("N/A")
						number_fixations_before_AOI_list.append("N/A")

					aoi_counter += 1
		print ("")

	### For Ellipse AOIs
	single_ellipse_aoi_df = ellipse_aoi_data.loc[ellipse_aoi_data['image'] == image]
	single_ellipse_aoi_df['subject_number']=subject_number

	# print last 4 columns (coordinates)
	ellipse_coordinates = single_ellipse_aoi_df[single_ellipse_aoi_df.columns[-4:]]
	print (ellipse_coordinates)
	#ellipse_aoi_subject_list.append(subject_number)
	#ellipse_aoi_list.append(single_ellipse_aoi_df)
	#ellipse_aoi_iaps_list.append(image)
	single_ellipse_aoi_df .to_csv("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/{}_{}_ellipse_aoi.csv".format(subject_number, subject_number, image), index=False)

	if single_ellipse_aoi_df.empty:
		print ('No Ellipse AOI for {}'.format(image))
		print ("")

		# append only the image number
		subject_number_list.append(subject_number)
		valence_list.append(valence)
		image_number_list.append(image)
		total_saccade_duration_list.append(total_saccade_duration)
		aoi_type_list.append("N/A")
		object_number_list.append("N/A")
		time_to_first_fixation_list.append("N/A")
		first_fixation_duration_list.append("N/A")
		number_fixations_list.append("N/A")
		total_in_AOI_list.append("N/A")
		total_out_AOI_list.append("N/A")
		number_fixations_before_AOI_list.append("N/A")

	else:
		# Count the rows (how many AOIs for each IAPS?)
		number_aoi = len(single_ellipse_aoi_df.index)
		print ("==========Fixation on Ellipse AOI==========")
		print ("total number of Ellipse AOIs for IAPS {} : {}".format(image, number_aoi))
		print ("")

		# Start the AOI counter
		aoi_counter = 0

		# Start processing if there truly are more than 1 AOIs
		if number_aoi >= 1:
			while aoi_counter < number_aoi:
				aoi_number = single_ellipse_aoi_df.iloc[aoi_counter]['objectNumber']

				print ("=====AOI {}====".format(aoi_number))

				#Combine fixation data with AOI data
				#merged = pd.merge(fixation_df, single_ellipse_aoi_df.iloc[[aoi_counter]], on='image')
				merged = true_fixation_df.merge(single_ellipse_aoi_df.iloc[[aoi_counter]], on=['image'])#.set_index(fixation_df.index)

				# Clean X,Y gaze and coordinates
				cleaned_fixation_df = gaze_compiler.ellipse_clean_gaze_and_coordinate(merged)
				
				#print (row['x_deblinked'], row['y_deblinked'], row['Xcenter'], row['Ycenter'], row['Width'], row['Height'])
				fixation_in_aoi_df = gaze_compiler.ellipse_compute_gaze_in_AOI(cleaned_fixation_df)

				
				#print (merged.dtypes)
				#merged = merged[(merged['x_deblinked'] > merged['Xmin']) & (merged['x_deblinked'] < merged['Xmax']) & (merged['y_deblinked'] > merged['Ymin']) & (merged['y_deblinked'] < merged['Ymax'])]
				fixation_in_aoi_df = fixation_in_aoi_df[(fixation_in_aoi_df['ellipse_value'] <= 1)]

				# sometimes there are no fixations in AOIs
				if fixation_in_aoi_df.empty:
					print ('No Fixation for {} {}'.format(image, single_ellipse_aoi_df.iloc[aoi_counter]['objectNumber']))
					print ("")
					subject_number_list.append(subject_number)
					valence_list.append(valence)
					image_number_list.append(image)
					total_saccade_duration_list.append(total_saccade_duration)
					aoi_type_list.append("ellipse")
					object_number_list.append(single_ellipse_aoi_df.iloc[aoi_counter]['objectNumber'])
					time_to_first_fixation_list.append("N/A")
					first_fixation_duration_list.append("N/A")
					number_fixations_list.append("N/A")
					total_in_AOI_list.append("N/A")
					total_out_AOI_list.append("N/A")
					number_fixations_before_AOI_list.append("N/A")

					aoi_counter += 1

				# if there are at least one fixation in AOIs
				else:
					fixation_in_AOI_list = fixation_detector.detect_fixation(fixation_in_aoi_df)
					total_number_fixations_in_AOI, true_fixation_in_AOI_list = fixation_detector.detect_true_fixation(fixation_in_AOI_list)
					total_duration_AOI = fixation_detector.compute_total_duration_fixation(true_fixation_in_AOI_list)

					subject_number_list.append(subject_number)
					valence_list.append(valence)
					image_number_list.append(image)
					total_saccade_duration_list.append(total_saccade_duration)
					aoi_type_list.append("ellipse")
					object_number_list.append(single_ellipse_aoi_df.iloc[aoi_counter]['objectNumber'])
					number_fixations_list.append(total_number_fixations_in_AOI)

					print ("total number of fixations in Elllipse AOI that last more than 60ms : {}".format(total_number_fixations_in_AOI))
					print ("total duration of fixations in Elllipse AOI {} : {}ms".format(aoi_number, total_duration_AOI))
					print ("")

					if len(true_fixation_in_AOI_list) > 0:
						first_fixation_index = true_fixation_in_AOI_list[0]

						# Account for when missing data in the beggning?
						# ====================a.time to first fixation in AOI====================
						time_at_first_fixation_in_AOI = round(first_fixation_index[0] * one_sample_time, 2)
						print ("-----a. Initial fixation on AOI at {}ms-----".format(time_at_first_fixation_in_AOI))

						# ====================b.fixation duration of first fixation in AOI====================
						first_fixation_in_AOI_duration = round(len(first_fixation_index) * one_sample_time, 2)
						print ("-----b. fixation duration of first fixation in Ellipse AOI: {}ms-----".format(first_fixation_in_AOI_duration))
					
						time_to_first_fixation_list.append(time_at_first_fixation_in_AOI)
						first_fixation_duration_list.append(first_fixation_in_AOI_duration)

						# ====================c.total time fixating in AOIs compared to elsewhere in the picture====================
						percent_fixation_in_AOI = round((float(total_duration_AOI)/float(total_duration_IAPS)) * 100, 2)
						print ("-----c1. total duration of fixations in AOI: {}ms ({}%)-----".format(str(total_duration_AOI), str(percent_fixation_in_AOI)))
		
						total_in_AOI_list.append(str(total_duration_AOI))

						total_fixation_duration_outside_AOI = round(float(total_duration_IAPS) - float(total_duration_AOI), 2)
						percent_fixation_out_AOI = round((float(total_fixation_duration_outside_AOI)/float(total_duration_IAPS)) * 100, 2)
						print ("-----c2. total duration of fixations outside AOI: {}ms({}%)-----".format(str(total_fixation_duration_outside_AOI), str(percent_fixation_out_AOI)))
						print ("")

						total_out_AOI_list.append(str(total_fixation_duration_outside_AOI))

							
						# ====================d.number of fixations before fixating on AOIs====================
						if first_fixation_index[0] == 0:
							print ("{} started fixating on AOI as soon as IAPS was presented".format(subject_number))
							print ("-----d.number of fixations before fixation on AOI: 0-----")
							print ("")
							number_fixations_before_AOI_list.append("0")

						else: 
							if time_at_first_fixation_in_IAPS == time_at_first_fixation_in_AOI:
								print ("Initial fixation on IAPS and AOI match, good!")
								print ("-----d.number of fixations before fixation on AOI: 0-----")
								print ("")
								number_fixations_before_AOI_list.append("0")

							elif time_at_first_fixation_in_IAPS > time_at_first_fixation_in_AOI:
								print ("Initial Fixation on AOI, good!")
								print ("-----d.number of fixations before fixation on AOI: 0-----")
								print ("")
								number_fixations_before_AOI_list.append("0")

							elif time_at_first_fixation_in_IAPS < time_at_first_fixation_in_AOI:
								print ("Initial Fixation on IAPS, compute how many before hitting AOI")

								# If any of fixations in IAPS occur before first fixating on AOI
								# They are already filtred for at minimum fixation duration (60ms)
								fixation_counter = 0
								for fixation in true_fixation_in_IAPS_list:
									if max(fixation) < first_fixation_index[0]:
										fixation_counter += 1
									

								number_fixations_before_AOI = fixation_counter
								number_fixations_before_AOI_list.append(fixation_counter)
								print ("-----d.number of fixations before fixation on AOI: {}-----".format(number_fixations_before_AOI))
								print ("")


					else:
						print ("no fixation longer than 60ms")

						time_to_first_fixation_list.append("N/A")
						first_fixation_duration_list.append("N/A")
						total_in_AOI_list.append("N/A")
						total_out_AOI_list.append("N/A")
						number_fixations_before_AOI_list.append("N/A")


					aoi_counter += 1
		print ("")

	print (len(subject_number_list))
	print (len(valence_list))
	print (len(image_number_list))
	print (len(total_saccade_duration_list))
	print (len(aoi_type_list))
	print (len(object_number_list))
	print (len(time_to_first_fixation_list))
	print (len(first_fixation_duration_list))
	print (len(number_fixations_list))
	print (len(total_in_AOI_list))
	print (len(total_out_AOI_list))
	print (len(number_fixations_before_AOI_list))
	
	print (len(rectangle_aoi_list))
	print (len(ellipse_aoi_list))
	


	# rectangle_aoi_df = pd.DataFrame(
	# 	{'subject_number':rectangle_aoi_subject_list,
	# 	'iaps_number': rectangle_aoi_iaps_list,
	# 	'rectangle_coordinates':rectangle_aoi_list})

	# ellipse_aoi_df = pd.DataFrame(
	# 	{'subject_number':ellipse_aoi_subject_list,
	# 	'iaps_number': ellipse_aoi_iaps_list,
	# 	'ellipse_coordinates':ellipse_aoi_list})

	# rectangle_aoi_df.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/{}_rectangle_aoi_.csv".format(subject_number, subject_number), index=False)

	# ellipse_aoi_df.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/{}_ellipse_aoi_.csv".format(subject_number, subject_number), index=False)


	analysis_df = pd.DataFrame(
		{'subject_number':subject_number_list,
		'iaps_number':image_number_list,
		'valence':valence_list,
		'total_saccade_duration': total_saccade_duration_list,
		'aoi_type':aoi_type_list,
		'object_number':object_number_list,
		'time_to_first_fixation':time_to_first_fixation_list,
		'first_fixation_duration':first_fixation_duration_list,
		'number_fixations':number_fixations_list,
		'total_fixation_duration_in_AOI': total_in_AOI_list,
		'total_fixation_duration_out_AOI': total_out_AOI_list,
		'number_fixations_before_AOI': number_fixations_before_AOI_list})

	# Remove unnecessary AOI rows and only keep the ones that actaully exist
	analysis_df_cleaned = analysis_df[analysis_df.aoi_type != "N/A"]

	#print (analysis_df)
	analysis_df_cleaned.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/{}_fixation_compiled.csv".format(subject_number, subject_number), index=False)


	print ("compiling")

	data_type_df = pd.DataFrame(
		{'subject_number':subject_list,
		'sampling_rate': sampling_rate_list,
		'percent_good_data': percent_good_data_list,
		'iaps_number': iaps_list,
		'total_samples':total_list,
		'number_interpolated':number_interpolated_list,
		'number_original': number_original_list,
		'number_missing':number_missing_list,
		'percent_interpolated':ratio_interpolated_list,
		'percent_original': ratio_original_list,
		'percent_missing':ratio_missing_list})

	data_type_df.to_csv(f"/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{subject_number}/{subject_number}_data_type_compiled.csv", index=False)
	# print ("processing for %s complete without error"%(subject_number))
	# #single_image_df.to_csv("/home/slee/Desktop/eye_sample.csv")
	# #print (single_image_df)


