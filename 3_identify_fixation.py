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
from hawkeye import GazeCompiler

subject_number = sys.argv[1]

coordinate_limits = (0, 1280)
median_with_max = 1.0 / 20
max_blink_sec = 0.4
minimum_fixation_duration = 60

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
	indexRange= range(minIndex, maxIndex+1)
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
print (missingIAPSList)

# Compare missingIAPSList to the Original, figure out which Nth element is missing
missing_stim_number_list = [] 
for index, stim in enumerate(preDenoise_imageList):
	for missingIAPS in missingIAPSList:
		if missingIAPS == stim:
			stim_number = "stim_" + str(index + 1)
			missing_stim_number_list.append(stim_number)
print (missing_stim_number_list)

# Total valid data after Denoising #2
percent_good_data_subject = round((post_denoise_gaze_count/pre_denoise_gaze_count) * 100, 2)
print ("Percent Good Data for subject {}: {}%".format(subject_number, percent_good_data_subject))

#--------------------AOI data--------------------
aoi_reader = AOIReader()

aoi = aoi_reader.read_in_AOI(data_denoised)

rectangle_aoi_df = aoi[(aoi['AOItype'] == 'Rectangle')]
rectangle_aoi_df = rectangle_aoi_df.reset_index(drop=True)

ellipse_aoi_df = aoi[(aoi['AOItype'] == 'Ellipse')]
ellipse_aoi_df = ellipse_aoi_df.reset_index(drop=True)

aoi_scalar = AOIScalar()

### Refine coordinates for Rectangles
rectangle_aoi_data = aoi_scalar.scale_rectangle_aoi(rectangle_aoi_df)

### Refine coordinates for Ellipses
ellipse_aoi_data = aoi_scalar.scale_ellipse_aoi(ellipse_aoi_df)

# Create a empty list for data to compile
image_number_list = []
aoi_type_list = []
object_number_list = []
time_to_first_fixation_list = []
first_fixation_duration_list = []
number_fixations_list = []

#--------------------Raw vs. Interpolated Data--------------------


#for image in postDenoise_imageList:

# Work with data relavant to single IAPS image at a time
image = postDenoise_imageList[33]
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

# Fill in the empty coordinate columns with 0's
single_image_df['CursorX'] = single_image_df['CursorX'].fillna(0)
single_image_df['CursorY'] = single_image_df['CursorY'].fillna(0)

#--------------------Denoising 3: Median Filtering per each IAPS--------------------

signal_denoisor = SignalDenoisor(median_with_max, max_blink_sec, sample_per_second)

# Handles short (1-sample) dropouts and x & y values surrounding blinks
median_filtered_df = signal_denoisor.meidan_filter(single_image_df)

# Create Offset values for QA purposes
median_filtered_df['raw_x_offset_column'] = median_filtered_df['CursorX'] + 30
median_filtered_df['raw_y_offset_column'] = median_filtered_df['CursorY'] + 30


#--------------------Denoising4: Remove Blinks--------------------
# Blink range == 50 ~ 400ms
# Currently cut-off long end of blinks
# Inerpolate these cuts
# Simple forward-fill (alternative would be linear interpolation)

deblinked_df = signal_denoisor.remove_blinks(median_filtered_df)

# Create Offset values for QA purposes
deblinked_df['filtered_x_offset_column'] = deblinked_df['x_to_deblink'] + 30
deblinked_df['filtered_y_offset_column'] = deblinked_df['y_to_deblink'] + 30

#--------------------Detect Saccades--------------------
saccade_detector = SaccadeDetector(sample_per_second)

saccade_df = saccade_detector.detect_saccade(deblinked_df)

# Get indices that are saccades 
candidate_t = (saccade_df[saccade_df['saccade_candidate'] == True]).index

#--------------------Detect Fixations--------------------

# Get indices that are not saccades (fixations)
candidate_t = (saccade_df[saccade_df['saccade_candidate'] == False]).index


print (candidate_t)

exit()
# 이게 틀렸음!
# Create dataFrame of all fixations
fixation_df = saccade_df.loc[saccade_df['saccade_candidate'] == False]


# --------------------Detect Fixations in AOI--------------------
gaze_compiler = GazeCompiler()

# Create dataFrame of Rectangle AOIs
single_rectangle_aoi_df = rectangle_aoi_data.loc[rectangle_aoi_data['image'] == image]


if single_rectangle_aoi_df.empty:
	print ('No Rectangle AOI for {}'.format(image))
	print ("")

	# add image number
	image_number_list.append(image)
	aoi_type_list.append("N/A")
	object_number_list.append("N/A")
	time_to_first_fixation_list.append("N/A")
	first_fixation_duration_list.append("N/A")
	number_fixations_list.append("N/A")

else:
	# Count the rows (how many AOIs for each IAPS?)
	number_aoi = len(single_rectangle_aoi_df.index)
	print ("total number of Rectangle AOIs for IAPS {} : {}".format(image, number_aoi))

	# Start the AOI counter
	aoi_counter = 0

	# Start processing if there truly are more than 1 AOIs
	if number_aoi >= 1:
		while aoi_counter < number_aoi:
			#Combine fixation data with AOI data
			merged = fixation_df.merge(single_rectangle_aoi_df.iloc[[aoi_counter]], how='left').set_index(fixation_df.index)

			# Clean X,Y gaze and coordinates
			cleand_fixation_df = gaze_compiler.rectangle_clean_gaze_and_coordinate(merged)

			# Compute if a gaze is in the AOI grid or not
			fixation_in_aoi_df = gaze_compiler.rectangle_compute_gaze_in_AOI(cleand_fixation_df)

			# sometimes there are no fixations in AOIs
			if fixation_in_aoi_df.empty:
				print ('No Fixation for {} {}'.format(image, single_rectangle_aoi_df.iloc[aoi_counter]['objectNumber']))
				print ("")
				image_number_list.append(image)
				aoi_type_list.append("rectangle")
				object_number_list.append(single_rectangle_aoi_df.iloc[aoi_counter]['objectNumber'])
				time_to_first_fixation_list.append("N/A")
				first_fixation_duration_list.append("N/A")
				number_fixations_list.append("N/A")

				aoi_counter += 1

			else:
				# Create a list of all indices that are fixations in AOI
				fixation_in_aoi_indices = list(fixation_in_aoi_df.index)

				# Group continous indicies (to figure out how many distinct fixations are there)
				all_fixations_list = [list(group) for group in mit.consecutive_groups(fixation_in_aoi_indices)]

				# total number of fixations
				total_number_fixations = len(all_fixations_list)
				print ('Fixation exist for {} {}'.format(image, single_rectangle_aoi_df.iloc[aoi_counter]['objectNumber']))
				print ("total number of fixations in Rectangle AOI : {}".format(total_number_fixations))
				
				image_number_list.append(image)
				aoi_type_list.append("rectangle")
				object_number_list.append(single_rectangle_aoi_df.iloc[aoi_counter]['objectNumber'])
				number_fixations_list.append(total_number_fixations)

				# Subset fixations that are longer than threshold
				true_fixation_list = []

				for fixations in all_fixations_list:
					start_fixation = min(fixations)
					end_fixation = max(fixations)
					index_fixation = end_fixation - start_fixation
					time_fixation_ms = index_fixation * ONE_SAMPLE_TIME
					
					if time_fixation_ms > MINIMUM_FIXATION_DURATION:
						true_fixation_list.append(fixations)

			
				print ("total number of fixations in Rectangle AOI that last more than 60ms : {}".format(str(len(true_fixation_list))))
				

				if len(true_fixation_list) > 0:
					first_fixation_index = true_fixation_list[0]

					# fixation duration of first fixation in AOI
					first_fixation_duration = round(len(first_fixation_index) * ONE_SAMPLE_TIME, 2)
					print ("fixation duration of first fixation in Rectangle AOI: {}ms".format(first_fixation_duration))
					
					# time to first fixation in AOI
					time_to_first_fixation = min(first_fixation_index) * ONE_SAMPLE_TIME
					print ("time to first fixation in Rectangle AOI: {}ms".format(time_to_first_fixation))

					time_to_first_fixation_list.append(time_to_first_fixation)
					first_fixation_duration_list.append(first_fixation_duration)

				else:
					print ("no fixation longer than 60ms")

					time_to_first_fixation_list.append("N/A")
					first_fixation_duration_list.append("N/A")

				aoi_counter += 1
	print ("")


### For Ellipse AOIs
single_ellipse_aoi_df = ellipse_aoi_data.loc[ellipse_aoi_data['image'] == image]

if single_ellipse_aoi_df.empty:
	print ('No Ellipse AOI for {}'.format(image))
	print ("")
	image_number_list.append(image)
	aoi_type_list.append("N/A")
	object_number_list.append("N/A")
	time_to_first_fixation_list.append("N/A")
	first_fixation_duration_list.append("N/A")
	number_fixations_list.append("N/A")

else:
	#print (single_ellipse_aoi_df)
	
	
	# Count the rows
	number_aoi = len(single_ellipse_aoi_df.index)
	print ("total number of Ellipse AOIs for IAPS {} : {}".format(image, number_aoi))

	aoi_counter = 0
	if number_aoi >= 1:
		while aoi_counter < number_aoi:
			#try:
				
			merged = pd.merge(fixation_df, single_ellipse_aoi_df.iloc[[aoi_counter]], on='image')
			
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
				image_number_list.append(image)
				aoi_type_list.append("ellipse")
				object_number_list.append(single_ellipse_aoi_df.iloc[aoi_counter]['objectNumber'])
				time_to_first_fixation_list.append("N/A")
				first_fixation_duration_list.append("N/A")
				number_fixations_list.append("N/A")
				aoi_counter += 1

			else:

				# Create a list of all indices that are fixations in AOI
				fixation_in_aoi_indices = list(fixation_in_aoi_df.index)

				# Group continous indicies (to figure out how many distinct fixations are there)
				all_fixations_list = [list(group) for group in mit.consecutive_groups(fixation_in_aoi_indices)]

				# total number of fixations
				total_number_fixations = len(all_fixations_list)
				print ('Fixation exist for {} {}'.format(image, single_ellipse_aoi_df.iloc[aoi_counter]['objectNumber']))
				print ("total number of fixations in Ellipse AOI : {}".format(total_number_fixations))

				image_number_list.append(image)
				aoi_type_list.append("ellipse")
				object_number_list.append(single_ellipse_aoi_df.iloc[aoi_counter]['objectNumber'])
				number_fixations_list.append(total_number_fixations)

				# Subset fixations that are longer than threshold
				true_fixation_list = []

				for fixations in all_fixations_list:
					start_fixation = min(fixations)
					end_fixation = max(fixations)
					index_fixation = end_fixation - start_fixation
					time_fixation_ms = index_fixation * ONE_SAMPLE_TIME
					
					if time_fixation_ms > MINIMUM_FIXATION_DURATION:
						true_fixation_list.append(fixations)

			
				print ("total number of fixations in Ellipse AOI that last more than 60ms : {}".format(str(len(true_fixation_list))))
				

				if len(true_fixation_list) > 0:
					first_fixation_index = true_fixation_list[0]

					# fixation duration of first fixation in AOI
					first_fixation_duration = round(len(first_fixation_index) * ONE_SAMPLE_TIME, 2)
					print ("fixation duration of first fixation in Ellipse AOI: {}ms".format(first_fixation_duration))
					
					# time to first fixation in AOI
					time_to_first_fixation = round(min(first_fixation_index) * ONE_SAMPLE_TIME, 2)
					print ("time to first fixation in Ellipse AOI: {}ms".format(time_to_first_fixation))

					time_to_first_fixation_list.append(time_to_first_fixation)
					first_fixation_duration_list.append(first_fixation_duration)

				else:
					print ("no fixation longer than 60ms")

					time_to_first_fixation_list.append("N/A")
					first_fixation_duration_list.append("N/A")

				aoi_counter += 1
	print ("")


print (len(image_number_list))
print (len(aoi_type_list))
print (len(object_number_list))
print (len(time_to_first_fixation_list))
print (len(first_fixation_duration_list))
print (len(number_fixations_list))


analysis_df = pd.DataFrame(
	{'IAPS_number':image_number_list,
	'aoi_type':aoi_type_list,
	'object_number':object_number_list,
	'time_to_first_fixation':time_to_first_fixation_list,
	'first_fixation_duration':first_fixation_duration_list,
	'number_fixations':number_fixations_list})

# Remove unnecessary AOI rows and only keep the ones that actaully exist
analysis_df_cleaned = analysis_df[analysis_df.aoi_type != "N/A"]

#print (analysis_df)
analysis_df_cleaned.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/{}_fixation_compiled.csv".format(subNum, subNum))


print ("processing for %s complete without error"%(subNum))
#single_image_df.to_csv("/home/slee/Desktop/eye_sample.csv")
#print (single_image_df)

