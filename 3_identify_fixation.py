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

def compute_sampling_rate(dataFrame):
	data_by_trial = list(dataFrame.groupby('image'))
	ms_between_samples = data_by_trial[0][1]['TETTime'].diff().dropna().mean()
	samples_per_second = np.round(1000 / ms_between_samples)
	return (samples_per_second)

def denoise_practice_and_pause(dataFrame):
	##### Remove first 6 practice picture trials (subset df that is not practice trials)
	denoised_df = dataFrame[(dataFrame['Running']) != 'pracList']

	##### Remove trials that are considered "pause"
	final_df = denoised_df[(denoised_df['Procedure']) != 'pause']
	return (final_df)

def denoise_fixation_cross(dataFrame):

	##### Remove trials from fixation period (~1000ms)
	# Create a new column "TimestampMilliseconds" using "TETTIME" (E-prime Clock in Microseconds)
	dataFrame['TimestampMilliseconds'] = dataFrame['TETTime']

	# Round to nearest whole number
	#data_merged['TimestampMillisecondsRounded'] = data_merged['TimestampMilliseconds'].round()

	# Group by IAPS image number and rank by time in milieseconds
	dataFrame['rank'] = dataFrame.groupby('image')['TimestampMilliseconds'].rank(ascending=True, method='dense')
	dataFrame['rank'] = dataFrame['rank'].astype(int)

	# Extract start times based on rank 1 of all IAPS images
	startTime = dataFrame.loc[dataFrame['rank'] == 1]
	startTime = startTime['TimestampMilliseconds']
	startTimeList = startTime.tolist()

	# Remove first 1000ms of each trial
	for time in startTimeList:
		fixationCrossTime = time + 1000
		dataFrame = dataFrame.drop(dataFrame[(dataFrame.TimestampMilliseconds >= time) & (dataFrame.TimestampMilliseconds <= fixationCrossTime)].index)

	final_df = dataFrame
	return (final_df)

def denoise_invalid(dataFrame):
	##### Filter data by validity
	# Keep trials with AT LEAST one good (valid) eye gaze 
	# Use anything from 0 , 1 , or 2 in at least one eye
	data_denoised = dataFrame[(dataFrame['ValidityLeftEye'] <= 2) | (dataFrame['ValidityRightEye'] <= 2)] 
	# For some reason, keep including "4" so manually drop trila that have 4 in BOTH left and right eye
	data_denoised = data_denoised.drop(data_denoised[(data_denoised.ValidityLeftEye == 4) & (data_denoised.ValidityRightEye ==4)].index) 

	final_df = data_denoised
	return (final_df)
	# Remove trials with invalid distance data 
	# We might not have to do this depdning on whether we use this in later computations
	#data_denoised = data_denoised.drop(data_denoised[(data_denoised.DistanceLeftEye == -1)].index)
	#data_denoised = data_denoised.drop(data_denoised[(data_denoised.DistanceRightEye == -1)].index)

def read_in_AOI(dataFrame):
	imageList = dataFrame['image'].unique()
	IAPSlist = []
	coordinateList = []
	objectNumList = []
	noAOIlist = []

	for image in imageList:
		filePath = "/study/reference/public/IAPS/IAPS/IAPS_2008_1-20_800x600BMP/IAPS_2008_AOIs/%s.OBT"%(image)


		#parser = re.compile(r"[^=]+=(\d+), (\d+), (\d+), (\d+), (\d+)")

		#IAPSnumb = filePath[-8:-4]
		try:
			with open(filePath, 'rU') as f:
				for line in f:
					a = line.split('=')
					if len(a) > 1 and a[1] != "0\n":
						objectNum = a[0]
						objectCoordinate = a[1][:-10]
		
						IAPSlist.append(image)
						coordinateList.append(objectCoordinate)
						objectNumList.append(objectNum)

			df1 = pd.DataFrame(IAPSlist, columns=['image'])
			df2 = pd.DataFrame(coordinateList, columns=['coordinate'])
			df3 = pd.DataFrame(objectNumList, columns=['objectNumber'])
			df4 = pd.concat([df1, df2], axis=1)
			df5 = pd.concat([df4, df3], axis=1)

		except OSError as e:
			print ("no AOI for %s"%(image))
			noAOIlist.append(image)

	# Drop coordinates that indicate grid (object01)
	final_df = df5[df5.objectNumber.str.contains("Object01") == False].reset_index()


	# Fetch AOI type information and recode 
	final_df['AOItype'] = final_df.coordinate.str[:1]
	final_df['AOItype'] = final_df['AOItype'].replace(['1'], 'Rectangle')
	final_df['AOItype'] = final_df['AOItype'].replace(['2'], 'Ellipse')

	return (final_df)
def scale_rectangle_aoi(dataFrame):

	rectangle_coordinate_list = []

	for index, row in rectangle_aoi_df.iterrows():
		a = (row['coordinate']).split(",")
		rectangle_coordinates = a[1:]
		#df5.loc[index,'coordinate'] = rectangle_coordinates
		rectangle_coordinate_list.append(rectangle_coordinates)

	rectangle_coordinate_df = pd.DataFrame(rectangle_coordinate_list, columns=['Xmin','Ymax','Xmax','Ymin'])

	# Merge new coordinate information with AOI dataframe
	rectangle_aoi_data = pd.concat([rectangle_aoi_df, rectangle_coordinate_df], axis=1)
	#df5["rectangle_coordinates"] = pd.Series(rectangle_coordinate_list, index=df5.index)

	# Cast Float to coordinate values
	rectangle_aoi_data[['Xmax', 'Xmin', 'Ymax', 'Ymin']] = rectangle_aoi_data[['Xmax', 'Xmin', 'Ymax', 'Ymin']].astype(float)

	# Scale AOI data to align with Tobbi Stimuli
	# Resample AOI data to 800 x 600
	rectangle_aoi_data['Xmax'] = (rectangle_aoi_data['Xmax'] * 1280) / 800

	rectangle_aoi_data['Xmin'] = (rectangle_aoi_data['Xmin'] * 1280) / 800

	rectangle_aoi_data['Ymax'] = (600 - rectangle_aoi_data['Ymax']) 
	rectangle_aoi_data['Ymax'] = (rectangle_aoi_data['Ymax'] * 1024) / 600

	rectangle_aoi_data['Ymin'] = (600 - rectangle_aoi_data['Ymin']) 
	rectangle_aoi_data['Ymin'] = (rectangle_aoi_data['Ymin'] * 1024) / 600

	# Change all negatvie values to 0 because it's a simple coordiante extension error (safe to assume it's at the edge of IAPS, thus 0)
	rectangle_aoi_data.loc[(rectangle_aoi_data['Ymin'] < 0)] = 0

	final_df = rectangle_aoi_data

	return (final_df)

def scale_ellipse_aoi(dataFrame):
	ellipse_coordinate_list = []

	for index, row in ellipse_aoi_df.iterrows():
		a = (row['coordinate']).split(",")
		ellipse_coordinates = a[1:]
		#df5.loc[index,'coordinate'] = ellipse_coordinates
		ellipse_coordinate_list.append(ellipse_coordinates)

	ellipse_coordinate_df = pd.DataFrame(ellipse_coordinate_list, columns=['Xcenter','Ycenter','Height','Width'])

	# Merge new coordinate information with AOI dataframe
	ellipse_aoi_data = pd.concat([ellipse_aoi_df, ellipse_coordinate_df], axis=1)
	#df5["ellipse_coordinates"] = pd.Series(ellipse_coordinate_list, index=df5.index)


	# Cast Float to coordinate values
	ellipse_aoi_data[['Xcenter','Ycenter','Height','Width']] = ellipse_aoi_data[['Xcenter','Ycenter','Height','Width']].astype(float)

	# Scale AOI data to align with Tobbi Stimuli
	# Resample AOI data to 800 x 600
	ellipse_aoi_data['Xcenter'] = (ellipse_aoi_data['Xcenter'] * 1280) / 800
	ellipse_aoi_data['Width'] = (ellipse_aoi_data['Width'] * 1280) / 800
	ellipse_aoi_data['Ycenter'] = (ellipse_aoi_data['Ycenter'] * 1024) / 600
	ellipse_aoi_data['Height'] = (ellipse_aoi_data['Height'] * 1024) / 600

	final_df = ellipse_aoi_data

	return (final_df)

def meidan_filter(dataFrame, medianWitdhMax, samplingRate):
	# Transform 'CursorX' and 'CursorY' into 2D arrays
	x = dataFrame['CursorX'].astype(float).values
	y = dataFrame['CursorY'].astype(float).values

	# Apply 1/20s width median filter (Instantiated in the beggining)
	# MEDIAN_WIDTH_MAX = 1.0 / 20
	filter_width = samplingRate * medianWitdhMax

	if filter_width % 2 == 0 :
		# It has to be an odd number
		filter_width -= 1
	filter_width = int(filter_width)

	x_filtered = signal.medfilt(x, filter_width)
	y_filtered = signal.medfilt(y, filter_width)
	dataFrame['x_filtered'] = x_filtered
	dataFrame['y_filtered'] = y_filtered

	final_df = dataFrame

	return (final_df)

def remove_blinks(dataFrame, medianWitdhMax, maxBlinkSecond, samplingRate):
	#MAX_BLINK_SEC = 0.4

	max_blink_samples = int(np.round(maxBlinkSecond * samplingRate))

	#reload(deblink)

	# Transform 'CursorX' and 'CursorY' into 2D arrays
	x = dataFrame['CursorX'].astype(float).values
	y = dataFrame['CursorY'].astype(float).values

	filter_width = samplingRate * medianWitdhMax
	if filter_width % 2 == 0 :
		filter_width -= 1
	filter_width = int(filter_width)

	x_filtered = signal.medfilt(x, filter_width)
	y_filtered = signal.medfilt(y, filter_width)
	#dataFrame['x_filtered'] = x_filtered
	#dataFrame['y_filtered'] = y_filtered

	l_valid = dataFrame['ValidityLeftEye'] < 2  # Maybe < 4? I mostly see 0 and 4.
	r_valid = dataFrame['ValidityRightEye'] < 2
	l_valid_filt = signal.medfilt(l_valid, filter_width)
	r_valid_filt = signal.medfilt(r_valid, filter_width)
	valid = (l_valid_filt * r_valid_filt).astype(float)

	# This will find us all the invalid periods of data
	missing_data_segments = list(deblink.find_zero_areas(valid))
	potential_blinks = [(start, end, length) for start, end, length in missing_data_segments if length <= max_blink_samples]
	x_to_interp = x_filtered[:] # This copies the array
	y_to_interp = y_filtered[:]
	for start, end, length in potential_blinks:
	    x_to_interp[start:end] = np.nan
	    y_to_interp[start:end] = np.nan

	dataFrame['x_to_deblink'] = x_to_interp
	dataFrame['y_to_deblink'] = y_to_interp
	dataFrame['x_deblinked'] = dataFrame['x_to_deblink'].fillna(method='ffill')
	dataFrame['y_deblinked'] = dataFrame['y_to_deblink'].fillna(method='ffill')

	final_df = dataFrame

	return (final_df)

def detect_saccade(dataFrame, samplingRate):
	gaze_array = dataFrame[['x_deblinked', 'y_deblinked']].fillna(0).values
	saccade_detector = nystrom_saccade_detector.AdaptiveDetector(gaze_array, samplingRate, threshold_sd_scale=3)
	saccade_detector._compute_saccades()
	dataFrame['saccade_candidate'] = saccade_detector._candidates


	final_df = dataFrame

	return (final_df)	

def rectangle_clean_gaze_and_coordinate(dataFrame):
	# Clean X gaze
	dataFrame['x_deblinked'] = dataFrame['x_deblinked'].astype(float)
	dataFrame['Xmax'] = dataFrame['Xmax'].astype(float)
	dataFrame['Xmin'] = dataFrame['Xmin'].astype(float)
	#dataFrame = dataFrame.dropna(subset=['x_deblinked'])
	
	# Clen Y gaze
	dataFrame['y_deblinked'] = dataFrame['y_deblinked'].astype(float)
	dataFrame['Ymax'] = dataFrame['Ymax'].astype(float)
	dataFrame['Ymin'] = dataFrame['Ymin'].astype(float)
	#dataFrame = dataFrame.dropna(subset=['y_deblinked'])

	final_df = dataFrame
	return (final_df)

def rectangle_compute_gaze_in_AOI(dataFrame):
	final_df = dataFrame[(dataFrame['x_deblinked'] > dataFrame['Xmin']) & (dataFrame['x_deblinked'] < dataFrame['Xmax']) & (dataFrame['y_deblinked'] > dataFrame['Ymin']) & (dataFrame['y_deblinked'] < dataFrame['Ymax'])]
	
	return (final_df)

def ellipse_clean_gaze_and_coordinate(dataFrame):
	# Clean X gaze
	dataFrame['x_deblinked'] = dataFrame['x_deblinked'].astype(float)
	dataFrame['Xcenter'] = dataFrame['Xcenter'].astype(float)
	dataFrame['Width'] = dataFrame['Width'].astype(float)
	#dataFrame = dataFrame.dropna(subset=['x_deblinked'])
	
	# Clen Y gaze
	dataFrame['y_deblinked'] = dataFrame['y_deblinked'].astype(float)
	dataFrame['Ycenter'] = dataFrame['Ycenter'].astype(float)
	dataFrame['Height'] = dataFrame['Height'].astype(float)
	#dataFrame = dataFrame.dropna(subset=['y_deblinked'])

	final_df = dataFrame
	return (final_df)

def ellipse_compute_gaze_in_AOI(dataFrame):
	ellipse_point_list = []
	for index, row in dataFrame.iterrows():
		Xcoordinate = row['x_deblinked']
		Ycoordinate = row['y_deblinked']
		Xcenter = row['Xcenter']
		Ycenter = row['Ycenter']
		width = row['Width']
		height = row['Height']

		dx = Xcenter - Xcoordinate
		dy = Ycenter - Ycoordinate

		point = (( dx * dx ) / ( width * width )) + (( dy * dy ) / ( height * height ))
		
		ellipse_point_list.append(point)
	
	df = pd.DataFrame({'ellipse_value':ellipse_point_list})
	final_df = pd.concat([dataFrame, df], axis=1)
	return (final_df)

# Notes
# definately have to merge before applying any filters
# Q's
# How to correct for delay - no need I can just use RTTime (E-PRIME CLOCK)
# Why 6 timestamps, not 5? - don't use "timestamp", use "TETTime"
# Some missing AOIs - skip them 
# How to filter by validity? - initaially use tobii validty data and use Nate's deblink fucntion
# Sampling rate varies by subject - always compute sampling rate in the beggning 
# Intermediate files - QA plots
# Filter out Blinks??? - use Nate's deblink function
# CSV for each subject with total good trials, and good per each IAPS # AND stim in chronological order - completed
# why NAN's in deblinked data???
# Re scale AOI to 800 x 600 - done
# account for ellipse grid computation?
# Recheck AOI coordinate xmin xmax etc

# Nate's Suggestions
#Offset two series by small y-distance to Compare
# which workflow works the beest? giant file? or lots of little files - have little physical files and combine htmls for qa purposes
# how to record okay vs not okay (excel??)
# whats okay and not okay?


subNum = sys.argv[1]
COORD_LIMITS = (0, 1280)
MEDIAN_WIDTH_MAX = 1.0 / 20
MAX_BLINK_SEC = 0.4
MINIMUM_FIXATION_DURATION = 60

#--------------------Gaze Data--------------------
# Read in gaze data 
gaze_file = "/study/midusref/DATA/Eyetracking/david_analysis/raw_data/MIDUSref_startle_order*_FINAL_VERSION-%s-%s.gazedata"%(subNum, subNum)
gaze_file = glob.glob(gaze_file)
gaze_file = gaze_file[0]
gaze = pd.read_csv(gaze_file, sep='\t')

#gaze.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/MIDUSref_startle_order1_FINAL_VERSION-%s-%s.csv"%(subNum, subNum))

#--------------------E-prime Data--------------------
# Convert Eprime file in to tsv
eprime_input = "/study/midusref/DATA/Eyetracking/david_analysis/raw_data/MIDUSref_startle_order*_FINAL_VERSION-%s-%s.txt"%(subNum, subNum)
eprime_input = glob.glob(eprime_input)
eprime_input = eprime_input[0]

os.makedirs("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/%s"%(subNum), exist_ok=True)
os.system("eprime2tabfile %s > /study/midusref/DATA/Eyetracking/david_analysis/data_processed/%s/MIDUSref_FINAL_VERSION-%s-%s.tsv"%(eprime_input, subNum, subNum, subNum))

# Read in Eprime data
e_prime = pd.read_csv("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/%s/MIDUSref_FINAL_VERSION-%s-%s.tsv"%(subNum, subNum, subNum), sep='\t')

#--------------------Gaze and E-prime Data Merged--------------------
data_merged = pd.merge(gaze, e_prime, on='image')

#--------------------Sampling Rate--------------------
SAMPLE_PER_SECOND = compute_sampling_rate(data_merged)
print ("Sampling rate for %s: %s/s"%(subNum, SAMPLE_PER_SECOND))

ONE_SAMPLE_TIME = round((1/SAMPLE_PER_SECOND) * 1000, 1)

#--------------------Denoising1: Remove 6 Practice Picture, Pause, and Fixation Cross (~1000ms) Trials (Applies Universally)--------------------

data_merged = denoise_practice_and_pause(data_merged)

#### Total number of raw trials (after denoising 1 though)
raw_gaze_count = len(data_merged.index)
print ("raw_trials: " + str(raw_gaze_count))

data_merged = denoise_fixation_cross(data_merged)

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
data_denoised = denoise_invalid(data_merged)

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
print ("Percent Good Data for subject {}: {}%".format(subNum, percent_good_data_subject))

#--------------------AOI data--------------------

aoi = read_in_AOI(data_denoised)

rectangle_aoi_df = aoi[(aoi['AOItype'] == 'Rectangle')]
rectangle_aoi_df = rectangle_aoi_df.reset_index(drop=True)

ellipse_aoi_df = aoi[(aoi['AOItype'] == 'Ellipse')]
ellipse_aoi_df = ellipse_aoi_df.reset_index(drop=True)


### Refine coordinates for Rectangles
rectangle_aoi_data = scale_rectangle_aoi(rectangle_aoi_df)

### Refine coordinates for Ellipses
ellipse_aoi_data = scale_ellipse_aoi(ellipse_aoi_df)

# Create a empty list for data to compile
image_number_list = []
aoi_type_list = []
object_number_list = []
time_to_first_fixation_list = []
first_fixation_duration_list = []
number_fixations_list = []

#--------------------Raw vs. Interpolated Data--------------------
# stim_counter = 0

# IAPSList = [] + missingIAPSList
# StimList = [] + missing_stim_number_list

# GoodPercentList = [0] * len(missing_stim_number_list)

for image in postDenoise_imageList:

	# #if image == 2580:
	# stim_counter += 1 
	# stim_name = "stim_" + str(stim_counter)

	# if stim_name in StimList:
	# 	stim_counter += 1
	# 	stim_name = "stim_" + str(stim_counter)
	
	# Work with data relavant to single IAPS image at a time
	single_image_df = data_denoised.loc[data_denoised['image'] == image]

	# # Number of good raw trials (before interpolation)
	# raw_trials = len(single_image_df)

	# Figure out missing values due to previous denoising and fill in with "NaN"
	indexList = indexListDict[image]
	# indexLength = indexLengthDict[image]
	#print ("Original Index Range: {}".format(indexList))
	#print ("Original Index Length: {}".format(indexLength))

	single_image_df = single_image_df.reindex(indexList, fill_value=np.nan)

	# Re-set the index from 0 
	# "drop = True" drops old indices
	# "inplace = True" modifies the DataFrame in place (do not create a new object)
	# Here both are used to preserve "nan" values
	single_image_df.reset_index(drop=True, inplace=True)

	# # Number of total trials (after interpolation)
	# with_interpolated_trials = len(single_image_df)

	# index_verfication = with_interpolated_trials/indexLength
	# if index_verfication == 1:
	# 	print ("Index Length verification successful")

	# else:
	# 	print ("Index Length verification failure, recheck index values")

	# # Percent raw data for single IAPS
	# try:
	# 	percent_good_data_IAPS = round((raw_trials/with_interpolated_trials) * 100, 2)
	# #print ("percent good data %s: %s percent"%(image, percent_good_data_IAPS))
	# #print ("percent good data %s: %s percent"%(stim_name, percent_good_data_IAPS))
	# except ZeroDivisionError:
	# 	percent_good_data_IAPS = 0

	# # Append to lists to be used in later QA steps
	# IAPSList.append(image)
	# StimList.append(stim_name)
	# GoodPercentList.append(percent_good_data_IAPS)


	# Fill in the empty coordinate columns with 0's
	single_image_df['CursorX'] = single_image_df['CursorX'].fillna(0)
	single_image_df['CursorY'] = single_image_df['CursorY'].fillna(0)

	#--------------------Denoising 3: Median Filtering per each IAPS--------------------
	# Handles short (1-sample) dropouts and x & y values surrounding blinks
	median_filtered_df = meidan_filter(single_image_df, MEDIAN_WIDTH_MAX, SAMPLE_PER_SECOND)

	# Create Offset values for QA purposes
	median_filtered_df['raw_x_offset_column'] = median_filtered_df['CursorX'] + 30
	median_filtered_df['raw_y_offset_column'] = median_filtered_df['CursorY'] + 30

	#Plot median filterded
	#COORD_LIMITS = (0, 1280)
	# fig = plt.figure(figsize=(14, 4))
	# plt.ylim(COORD_LIMITS)
	# fig.suptitle('subject%s %s Denoise 1: Median Filtered'%(subNum, image))
	# plt.ylabel("Coordinates")
	# plt.xlabel("Trials")
	# plt.plot(median_filtered_df['raw_x_offset_column'], 'k', alpha=0.5)
	# plt.plot(median_filtered_df['x_filtered'], 'b', alpha=0.5)
	# plt.plot(median_filtered_df['raw_y_offset_column'], 'g', alpha=0.5)
	# plt.plot(median_filtered_df['y_filtered'], 'y', alpha=0.5)
	# plt.legend(['raw_X', 'filtered_X', 'raw_Y', 'filtered_Y'], loc='upper left')
	#plt.show()
	
	# os.makedirs('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/'.format(subNum), exist_ok = True)
	# print ("creating median filtered plot for {}".format(image))
	# fig.savefig('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/{}_{}_1.median_filtered.png'.format(subNum, subNum, image))

	
	#--------------------Denoising4: Remove Blinks--------------------
	# Blink range == 50 ~ 400ms
	# Currently cut-off long end of blinks
	# Inerpolate these cuts
	# Simple forward-fill (alternative would be linear interpolation)

	deblinked_df = remove_blinks(median_filtered_df, MEDIAN_WIDTH_MAX, MAX_BLINK_SEC, SAMPLE_PER_SECOND)

	# Create Offset values for QA purposes
	deblinked_df['filtered_x_offset_column'] = deblinked_df['x_to_deblink'] + 30
	deblinked_df['filtered_y_offset_column'] = deblinked_df['y_to_deblink'] + 30

	# Plot deblinked
	# fig = plt.figure(figsize=(14, 4))
	# plt.ylim(COORD_LIMITS)
	# fig.suptitle('subject%s %s Denoise 2: Deblinked'%(subNum, image))
	# plt.ylabel("Coordinates")
	# plt.xlabel("Trials")
	# plt.plot(deblinked_df['filtered_x_offset_column'], color='k', alpha=0.5)
	# plt.plot(deblinked_df['x_deblinked'], color='b', alpha=0.5)
	# plt.plot(deblinked_df['filtered_y_offset_column'], color='g', alpha=0.5)
	# plt.plot(deblinked_df['y_deblinked'], color='y', alpha=0.5)
	# plt.legend(['filtered_X', 'deblinked_X', 'filtered_Y', 'deblinked_Y'], loc='upper left')
	#plt.show()
	
	# os.makedirs('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}'.format(subNum), exist_ok = True)
	# print ("creating deblinked plot for {}".format(image))
	# fig.savefig('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/{}_{}_2.deblinked.png'.format(subNum, subNum, image))

	#--------------------Detect Saccades--------------------
	saccade_df = detect_saccade(deblinked_df, SAMPLE_PER_SECOND)

	# Get indices that are saccades 
	candidate_t = (saccade_df[saccade_df['saccade_candidate'] == True]).index

	# Plot vertical lines at saccades
	# fig = plt.figure(figsize=(14, 4))
	# plt.ylim(COORD_LIMITS)
	# plt.plot(saccade_df['x_deblinked'], color='k', alpha=0.8)
	# plt.plot(saccade_df['y_deblinked'], color='g', alpha=0.8)
	# for t in candidate_t:
	#     plt.axvline(t, 0, 1, color='r')
	# fig.suptitle('subject%s %s Denoise 3: Saccades'%(subNum, image))
	# plt.ylabel("Coordinates")
	# plt.xlabel("Trials")
	# plt.legend(['X', 'Y'], loc='upper left')
	#plt.show()

	#saccade_df.to_csv('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/saccade.csv')

	# # Create Plotsplt.suptitle('subject%s %s'%(subNum, image))
	# os.makedirs('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}'.format(subNum), exist_ok = True)
	# print ("creating saccade plot for {}".format(image))
	# fig.savefig('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/{}_{}_3.saccade.png'.format(subNum, subNum, image))
	
	#--------------------Detect Fixations--------------------

	# Get indices that are not saccades (fixations)
	candidate_t = (saccade_df[saccade_df['saccade_candidate'] == False]).index

	# Create dataFrame of all fixations
	fixation_df = saccade_df.loc[saccade_df['saccade_candidate'] == False]

	# fig = plt.figure(figsize=(14, 4))
	# plt.ylim(COORD_LIMITS)
	# fig.suptitle('subject%s %s Fixations'%(subNum, image))
	# plt.ylabel("Coordinates")
	# plt.xlabel("Trials")
	# plt.plot(fixation_df['x_deblinked'], color='b', alpha=0.5)
	# plt.plot(fixation_df['y_deblinked'], color='y', alpha=0.5)
	# plt.legend(['X', 'Y'], loc='upper left')
	# #plt.show()
	
	# #fixation_df.to_csv('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/fixation.csv')

	# os.makedirs('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}'.format(subNum), exist_ok = True)
	# print ("creating fixation plot for {}".format(image))
	# fig.savefig('/study/midusref/DATA/Eyetracking/david_analysis/data_processed/{}/{}_{}_4.fixation.png'.format(subNum, subNum, image))
	
	# --------------------Detect Fixations in AOI--------------------

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
				cleand_fixation_df = rectangle_clean_gaze_and_coordinate(merged)

				# Compute if a gaze is in the AOI grid or not
				fixation_in_aoi_df = rectangle_compute_gaze_in_AOI(cleand_fixation_df)

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
				cleaned_fixation_df = ellipse_clean_gaze_and_coordinate(merged)
				
				#print (row['x_deblinked'], row['y_deblinked'], row['Xcenter'], row['Ycenter'], row['Width'], row['Height'])
				fixation_in_aoi_df = ellipse_compute_gaze_in_AOI(cleaned_fixation_df)
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

					# If final plot needed
					# # Reset Index
					# diff = max(indexList) - min(indexList)
					# newIndexList = range(0, diff)
					# fixation_in_aoi_df = fixation_in_aoi_df.reindex(newIndexList, fill_value=np.nan)
					# #print (len(fixation_in_aoi_df))
					# # Reindex with to get full 4 minute timeflow

					# fig = plt.figure(figsize=(14, 4))
					# plt.ylim(COORD_LIMITS)
					# plt.xlim(0, max(newIndexList))
					# plt.plot(fixation_in_aoi_df['x_deblinked'], color='k', alpha=0.8)
					# plt.plot(fixation_in_aoi_df['y_deblinked'], color='g', alpha=0.8)
					# #for t in candidate_t:
					#     #plt.axvline(t, 0, 1, color='r')
					# fig.suptitle('subject%s %s'%(subNum, image))
					# plt.ylabel("Coordinates")
					# plt.xlabel("Trials")
					# plt.legend(['X', 'Y'], loc='upper left')
					# #plt.show()
					# fig.savefig('/study/midusref/DATA/Eyetracking/david_analysis/{}_{}_ellipse_fixation_{}.png'.format(subNum, image, str(aoi_counter)))
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
#--------------------Create QA tables--------------------
# # Create Intermediate QA files
# good_data_IAPS_number_df = pd.DataFrame({subNum:GoodPercentList}) # "Stim":StimList, {"IAPS":IAPSList}
# good_data_IAPS_number_df_transposed = good_data_IAPS_number_df.T
# good_data_IAPS_number_df_transposed.columns = IAPSList
# good_data_IAPS_number_df_transposed.insert(loc=0, column='TotalValid', value=[percent_good_data_subject])
# #print (good_data_IAPS_number_df_transposed)
# good_data_IAPS_number_df_transposed.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/%s/%s_valid_IAPS.csv"%(subNum, subNum))

# good_data_STIM_number_df = pd.DataFrame({subNum:GoodPercentList}) # "Stim":StimList, {"IAPS":IAPSList}
# good_data_STIM_number_df_transposed = good_data_STIM_number_df.T
# good_data_STIM_number_df_transposed.columns = StimList
# good_data_STIM_number_df_transposed.insert(loc=0, column='TotalValid', value=[percent_good_data_subject])
# #print (good_data_STIM_number_df_transposed)
# good_data_STIM_number_df_transposed.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/data_processed/%s/%s_valid_STIM.csv"%(subNum, subNum))


print ("processing for %s complete without error"%(subNum))
#single_image_df.to_csv("/home/slee/Desktop/eye_sample.csv")
#print (single_image_df)


