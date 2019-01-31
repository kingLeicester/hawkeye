# hawkeye

The repository contains a Python module (hawkeye.py) for processing, quality assessing, and analyzing Tobii eyetracking data. 

Usage: 

Before executing programs, ALWAYS check & resupply:
	- coordinate_limits
	- sample_limits
	- median_width_max
	- max_blink_sec
	- minimum_fixation_duration
	- maximum_gap_duration
	- check filepaths 

0. Source dependancies.sh
	- directs to an approporiate Python3 enviornmenet 
	
1. Execute 1.process.py <subject_number>
	- extracts eyetracking data for 4 Second picture onset period
	- performs signal denoising 
	- produces plots for each step of denoising

2. Execute 2.quality_assess.py <subject_number>
	- compiles percent valid for each stimuli 
	- produces separate datafiles(.csv) for stimuli number and order

3. Execute 3_identify_fixation.py <subject_number>
	- reads in pre-determined AOI information
	- computes the grid information of each AOI (rectangle and/or ellipse)
	- compiles:
		
		-a. Initial fixation time on an AOI
		-b. Fixation duration of initial fixation on an AOI
		-c1. Total fixation duration on an AOI
		-c2. Total fixation duration out of an AOI
		-d. Number of fixations before fixating on an AOI
		-e. percent valid data per subject
		-f. ratio of original:missing:interpolated data for each stimuli
	- produces separate datafiles(.csv) for (a-d) and (e,f)



	
