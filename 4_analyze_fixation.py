#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import glob


files = "/study/midusref/DATA/Eyetracking/david_analysis/data_processed/[0-9][0-9][0-9]/*_fixation_compiled.csv"

files = glob.glob(files)

subject_number_list = []
good_fixation_percent_list = []

counter = 0

for file in files:

	subject_number = file.split("/")[7]

	df = pd.read_csv(file)

	number_no_fixation = len(df) - df['first_fixation_duration'].count()
	number_total = (len(df))
	number_fixation = number_total - number_no_fixation

	percent_at_least_one_fixation = (number_fixation/number_total) * 100

	print ("participant {} engaged in {}% of AOIs ({}/{})".format(subject_number, str(percent_at_least_one_fixation), str(number_fixation), str(number_total)))

	subject_number_list.append(subject_number)
	good_fixation_percent_list.append(percent_at_least_one_fixation)
	counter += 1

print (counter)

good_fixation_df = pd.DataFrame({"subject_number":subject_number_list,
	"percent_fixation_on_all_aois":good_fixation_percent_list})

good_fixation_df.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/good_fixation.csv")
