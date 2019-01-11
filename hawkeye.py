#!/usr/bin/env python

__author__ = "David Lee"
__credits__ = ["David Lee", "Nate Vack"]
__version__ = "1.0"
__maintainer__ = "David Lee"
__email__ = "david.s.lee@wisc.edu"
__status__ = "Production"


import pandas as pd
import os
import sys
import re
import math
import numpy as np
import matplotlib
matplotlib.use('Agg') # Comment this out if graphing is not working
import matplotlib.pyplot as plt
import glob
from scipy import signal
import more_itertools as mit
from itertools import chain

# These are Nate Vack's Work. Should be included in the package with Nate Vack's ownership
import deblink
import nystrom_saccade_detector

class GazeReader:

	def __init__(self, subject_number):
		self.subject_number = subject_number

	def read_gaze_data(self, gaze_file_dir="/study/midusref/DATA/Eyetracking/david_analysis/raw_data/"):
		# Read in gaze data 
		gaze_file = f'{gaze_file_dir}MIDUSref_startle_order*_FINAL_VERSION-{self.subject_number}-{self.subject_number}.gazedata'
		gaze_file = glob.glob(gaze_file)
		gaze_file = gaze_file[0]
		gaze_df = pd.read_csv(gaze_file, sep='\t')

		return gaze_df

class EPrimeReader:
	
	def __init__(self, subject_number):
		self.subject_number = subject_number

	def read_eprime_data(self, eprime_raw_dir="/study/midusref/DATA/Eyetracking/david_analysis/raw_data/", eprime_out_dir="/study/midusref/DATA/Eyetracking/david_analysis/data_processed/"):
		# Convert Eprime file in to tsv
		eprime_input = f'{eprime_raw_dir}MIDUSref_startle_order*_FINAL_VERSION-{self.subject_number}-{self.subject_number}.txt'
		eprime_input = glob.glob(eprime_input)
		eprime_input = eprime_input[0]

		os.makedirs(f'{eprime_out_dir}{self.subject_number}', exist_ok=True)
		os.system(f'eprime2tabfile {eprime_input} > {eprime_out_dir}{self.subject_number}/MIDUSref_FINAL_VERSION-{self.subject_number}-{self.subject_number}.tsv')
		
		# Read in Eprime data
		e_prime_df = pd.read_csv(f'{eprime_out_dir}{self.subject_number}/MIDUSref_FINAL_VERSION-{self.subject_number}-{self.subject_number}.tsv', sep='\t')
		
		return e_prime_df

class GazeDenoisor:

	def __init__(self):
		pass

	def compute_sampling_rate(self, data_frame):
		data_by_trial = list(data_frame.groupby('image'))
		ms_between_samples = data_by_trial[0][1]['TETTime'].diff().dropna().mean()
		samples_per_second = np.round(1000 / ms_between_samples)
		return (samples_per_second)

	def denoise_practice_and_pause(self, data_frame):
		##### Remove first 6 practice picture trials (subset df that is not practice trials)
		denoised_df = data_frame[(data_frame['Running']) != 'pracList']

		##### Remove trials that are considered "pause"
		final_df = denoised_df[(denoised_df['Procedure']) != 'pause']
		return (final_df)

	def denoise_fixation_cross(self, data_frame):

		##### Remove trials from fixation period (~1000ms)
		# Create a new column "TimestampMilliseconds" using "TETTIME" (E-prime Clock in Microseconds)
		data_frame['TimestampMilliseconds'] = data_frame['TETTime']

		# Round to nearest whole number
		#data_merged['TimestampMillisecondsRounded'] = data_merged['TimestampMilliseconds'].round()

		# Group by IAPS image number and rank by time in milieseconds
		data_frame['rank'] = data_frame.groupby('image')['TimestampMilliseconds'].rank(ascending=True, method='dense')
		data_frame['rank'] = data_frame['rank'].astype(int)

		# Extract start times based on rank 1 of all IAPS images
		startTime = data_frame.loc[data_frame['rank'] == 1]
		startTime = startTime['TimestampMilliseconds']
		startTimeList = startTime.tolist()

		# Remove first 1000ms of each trial
		for time in startTimeList:
			fixationCrossTime = time + 1000
			data_frame = data_frame.drop(data_frame[(data_frame.TimestampMilliseconds >= time) & (data_frame.TimestampMilliseconds <= fixationCrossTime)].index)

		final_df = data_frame
		return (final_df)

	def denoise_invalid(self, data_frame):
		##### Filter data by validity
		# Keep trials with AT LEAST one good (valid) eye gaze 
		# Use anything from 0 , 1 , or 2 in at least one eye
		data_denoised = data_frame[(data_frame['ValidityLeftEye'] <= 2) | (data_frame['ValidityRightEye'] <= 2)] 
		# For some reason, keep including "4" so manually drop trila that have 4 in BOTH left and right eye
		data_denoised = data_denoised.drop(data_denoised[(data_denoised.ValidityLeftEye == 4) & (data_denoised.ValidityRightEye ==4)].index) 

		final_df = data_denoised
		return (final_df)
		# Remove trials with invalid distance data 
		# We might not have to do this depdning on whether we use this in later computations
		#data_denoised = data_denoised.drop(data_denoised[(data_denoised.DistanceLeftEye == -1)].index)
		#data_denoised = data_denoised.drop(data_denoised[(data_denoised.DistanceRightEye == -1)].index)

class AOIReader:

	def __init__(self):
		pass

	def read_in_AOI(self, data_frame):
		imageList = data_frame['image'].unique()
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

class AOIScalar:

	def __init__(self):
		pass

	def scale_rectangle_aoi(self, data_frame):

		rectangle_coordinate_list = []

		for index, row in data_frame.iterrows():
			a = (row['coordinate']).split(",")
			rectangle_coordinates = a[1:]
			#df5.loc[index,'coordinate'] = rectangle_coordinates
			rectangle_coordinate_list.append(rectangle_coordinates)

		rectangle_coordinate_df = pd.DataFrame(rectangle_coordinate_list, columns=['Xmin','Ymax','Xmax','Ymin'])

		# Merge new coordinate information with AOI data_frame
		rectangle_aoi_data = pd.concat([data_frame, rectangle_coordinate_df], axis=1)
		#df5["rectangle_coordinates"] = pd.Series(rectangle_coordinate_list, index=df5.index)

		# Cast Float to coordinate values
		rectangle_aoi_data[['Xmax', 'Xmin', 'Ymax', 'Ymin']] = rectangle_aoi_data[['Xmax', 'Xmin', 'Ymax', 'Ymin']].astype(float)

		# Scale AOI data to align with Tobbi Stimuli
		# Resample AOI data to 1024 x 1280
		rectangle_aoi_data['Xmax'] = (rectangle_aoi_data['Xmax'] * 1280) / 800

		rectangle_aoi_data['Xmin'] = (rectangle_aoi_data['Xmin'] * 1280) / 800

		rectangle_aoi_data['Ymax'] = (600 - rectangle_aoi_data['Ymax']) 
		rectangle_aoi_data['Ymax'] = (rectangle_aoi_data['Ymax'] * 1024) / 600

		rectangle_aoi_data['Ymin'] = (600 - rectangle_aoi_data['Ymin']) 
		rectangle_aoi_data['Ymin'] = (rectangle_aoi_data['Ymin'] * 1024) / 600

		# Change all negatvie values to 0 because it's a simple coordiante extension error (safe to assume it's at the edge of IAPS, thus 0)
		#rectangle_aoi_data['Ymin'][rectangle_aoi_data['Ymin'] < 0 ] = 0
		rectangle_aoi_data.loc[rectangle_aoi_data['Ymin'] < 0, 'Ymin'] = 0


		final_df = rectangle_aoi_data

		return (final_df)

	def scale_ellipse_aoi(self, data_frame):
		ellipse_coordinate_list = []

		for index, row in data_frame.iterrows():
			a = (row['coordinate']).split(",")
			ellipse_coordinates = a[1:]
			#df5.loc[index,'coordinate'] = ellipse_coordinates
			ellipse_coordinate_list.append(ellipse_coordinates)

		ellipse_coordinate_df = pd.DataFrame(ellipse_coordinate_list, columns=['Xcenter','Ycenter','Height','Width'])

		# Merge new coordinate information with AOI data_frame
		ellipse_aoi_data = pd.concat([data_frame, ellipse_coordinate_df], axis=1)
		#df5["ellipse_coordinates"] = pd.Series(ellipse_coordinate_list, index=df5.index)


		# Cast Float to coordinate values
		ellipse_aoi_data[['Xcenter','Ycenter','Height','Width']] = ellipse_aoi_data[['Xcenter','Ycenter','Height','Width']].astype(float)

		# Scale AOI data to align with Tobbi Stimuli
		# Resample AOI data to 1024 x 1280
		ellipse_aoi_data['Xcenter'] = (ellipse_aoi_data['Xcenter'] * 1280) / 800
		ellipse_aoi_data['Width'] = (ellipse_aoi_data['Width'] * 1280) / 800
		ellipse_aoi_data['Ycenter'] = (ellipse_aoi_data['Ycenter'] * 1024) / 600
		ellipse_aoi_data['Height'] = (ellipse_aoi_data['Height'] * 1024) / 600

		final_df = ellipse_aoi_data

		return (final_df)

class SignalDenoisor:
	
	def __init__(self, median_with_max, max_blink_second, sampling_rate):
		self.median_with_max = median_with_max
		self.max_blink_second = max_blink_second
		self.sampling_rate = sampling_rate

	def meidan_filter(self, data_frame):
		# Transform 'CursorX' and 'CursorY' into 2D arrays
		x = data_frame['CursorX'].astype(float).values
		y = data_frame['CursorY'].astype(float).values

		# Apply 1/20s width median filter (Instantiated in the beggining)
		# MEDIAN_WIDTH_MAX = 1.0 / 20
		filter_width = self.sampling_rate * self.median_with_max

		if filter_width % 2 == 0 :
			# It has to be an odd number
			filter_width -= 1
		filter_width = int(filter_width)

		x_filtered = signal.medfilt(x, filter_width)
		y_filtered = signal.medfilt(y, filter_width)
		data_frame['x_filtered'] = x_filtered
		data_frame['y_filtered'] = y_filtered

		final_df = data_frame

		return (final_df)

	def remove_blinks(self, data_frame):
		#MAX_BLINK_SEC = 0.4

		max_blink_samples = int(np.round(self.max_blink_second * self.sampling_rate))

		#reload(deblink)

		# Transform 'CursorX' and 'CursorY' into 2D arrays
		x = data_frame['CursorX'].astype(float).values
		y = data_frame['CursorY'].astype(float).values

		filter_width = self.sampling_rate * self.median_with_max
		if filter_width % 2 == 0 :
			filter_width -= 1
		filter_width = int(filter_width)

		x_filtered = signal.medfilt(x, filter_width)
		y_filtered = signal.medfilt(y, filter_width)
		#data_frame['x_filtered'] = x_filtered
		#data_frame['y_filtered'] = y_filtered

		l_valid = data_frame['ValidityLeftEye'] < 2  # Maybe < 4? I mostly see 0 and 4.
		r_valid = data_frame['ValidityRightEye'] < 2
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

		data_frame['x_to_deblink'] = x_to_interp
		data_frame['y_to_deblink'] = y_to_interp
		data_frame['x_deblinked'] = data_frame['x_to_deblink'].fillna(method='ffill')
		data_frame['y_deblinked'] = data_frame['y_to_deblink'].fillna(method='ffill')

		final_df = data_frame

		return (final_df)

class SaccadeDetector:

	def __init__(self, sampling_rate):
		self.sampling_rate = sampling_rate

	def detect_saccade(self, data_frame: str) -> pd.DataFrame:
		gaze_array = data_frame[['x_deblinked', 'y_deblinked']].fillna(0).values
		saccade_detector = nystrom_saccade_detector.AdaptiveDetector(point_array=gaze_array, samples_per_second=self.sampling_rate, threshold_sd_scale=2.5)
		saccade_detector._compute_saccades()
		data_frame['saccade_candidate'] = saccade_detector._candidates


		final_df = data_frame

		return (final_df)	

	def compute_saccade_interval(self, data_frame: str) -> pd.DataFrame:
	
		# Get indices that are saccades 
		saccade_candidate_t = (data_frame[data_frame['saccade_candidate'] == True]).index

		# Get indices before/after each saccade (basically saccade intervals)
		# Make a list of tuples (saccade intervals)
		saccade_interval_list = []
		for saccade_index in saccade_candidate_t:
			saccade_interval = (saccade_index - 1, saccade_index + 1)
			saccade_interval_list.append(saccade_interval)

		print (saccade_interval_list)

		# total index
		minimum_index = min(data_frame.index)
		maximum_index = max(data_frame.index)
		total_index = (minimum_index, maximum_index)
		print (total_index)

		# unleash the intervals
		#from itertools import chain
		unleashed_interval_list = list(chain.from_iterable(range(start, end+1) for start, end in saccade_interval_list))
		print (unleashed_interval_list)

		# put both together
		full_saccade_interval_list = []
		for index in range(total_index[0], total_index[1]+1):
			if index in unleashed_interval_list:
				full_saccade_interval_list.append("saccade")
			elif index not in unleashed_interval_list:
				full_saccade_interval_list.append("fixation")


		# Create saccade_interval column
		data_frame['saccade_interval'] = pd.Series(full_saccade_interval_list)

		# Rename first column of indices as "index" for future use
		data_frame.index.rename('index', inplace=True)

		return (data_frame)


class FixationDetector:

	def __init__(self, one_sample_time, minimum_fixation_duration, image):
		self.one_sample_time = one_sample_time
		self.minimum_fixation_duration = minimum_fixation_duration
		self.image = image

	# Takes a dataframe with gaze points(fixations) and returns a nested list of indicies(fixations)
	def detect_fixation(self, data_frame: str) -> pd.DataFrame:

		# Create a list of all indices that are fixations in IAPS
		fixation_indices = list(data_frame.index)

		# Group continous indicies 
		# to figure out how many distinct fixations are there
		# or to figure out how many times saccades intervened fixations
		all_fixations_list = [list(group) for group in mit.consecutive_groups(fixation_indices)]

		return (all_fixations_list)

	def detect_true_fixation(self, fixation_list):
		# Subset fixations that are longer than threshold
		true_fixation_list = []

		for fixation in fixation_list:
			start_fixation = min(fixation)
			end_fixation = max(fixation)
			index_fixation = end_fixation - start_fixation
			time_fixation_ms = index_fixation * self.one_sample_time
			
			if time_fixation_ms > self.minimum_fixation_duration:
				true_fixation_list.append(fixation)

		total_number_fixations = str(len(true_fixation_list))

		return (total_number_fixations, true_fixation_list)

	# Takes a nested list of indices (fixations) and returns total duration of all fixations
	def compute_total_duration_fixation(self, fixation_list):

		total_duration_IAPS = 0.0
		for fixations in fixation_list:
			total_duration_IAPS += (len(fixations) * self.one_sample_time)

		total_duration_IAPS = round(total_duration_IAPS, 2)
		total_duration_IAPS = str(total_duration_IAPS)

		return (total_duration_IAPS)

class GazeCompiler:
	
	def __init__(self):
		pass

	def rectangle_clean_gaze_and_coordinate(self, data_frame):
		# Clean X gaze
		data_frame['x_deblinked'] = data_frame['x_deblinked'].astype(float)
		data_frame['Xmax'] = data_frame['Xmax'].astype(float)
		data_frame['Xmin'] = data_frame['Xmin'].astype(float)
		#data_frame = data_frame.dropna(subset=['x_deblinked'])
		
		# Clen Y gaze
		data_frame['y_deblinked'] = data_frame['y_deblinked'].astype(float)
		data_frame['Ymax'] = data_frame['Ymax'].astype(float)
		data_frame['Ymin'] = data_frame['Ymin'].astype(float)
		#data_frame = data_frame.dropna(subset=['y_deblinked'])

		final_df = data_frame
		return (final_df)

	def rectangle_compute_gaze_in_AOI(self, data_frame):
		final_df = data_frame[(data_frame['x_deblinked'] > data_frame['Xmin']) & (data_frame['x_deblinked'] < data_frame['Xmax']) & (data_frame['y_deblinked'] > data_frame['Ymin']) & (data_frame['y_deblinked'] < data_frame['Ymax'])]
		
		return (final_df)

	def ellipse_clean_gaze_and_coordinate(self, data_frame):
		# Clean X gaze
		data_frame['x_deblinked'] = data_frame['x_deblinked'].astype(float)
		data_frame['Xcenter'] = data_frame['Xcenter'].astype(float)
		data_frame['Width'] = data_frame['Width'].astype(float)
		#data_frame = data_frame.dropna(subset=['x_deblinked'])
		
		# Clen Y gaze
		data_frame['y_deblinked'] = data_frame['y_deblinked'].astype(float)
		data_frame['Ycenter'] = data_frame['Ycenter'].astype(float)
		data_frame['Height'] = data_frame['Height'].astype(float)
		#data_frame = data_frame.dropna(subset=['y_deblinked'])

		final_df = data_frame
		return (final_df)

	def ellipse_compute_gaze_in_AOI(self, data_frame):
		ellipse_point_list = []
		for index, row in data_frame.iterrows():
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
		
		df = pd.data_frame({'ellipse_value':ellipse_point_list})
		final_df = pd.concat([data_frame, df], axis=1)
		return (final_df)
