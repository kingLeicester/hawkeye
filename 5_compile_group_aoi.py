import pandas as pd
import glob
import sys
import numpy as np
# #subject_number = sys.argv[1]
# iaps_number = sys.argv[1]

# path = f"/study/midusref/DATA/Eyetracking/david_analysis/data_processed/009/009_{iaps_number}_*_aoi.csv"

# appended_data = []

# files = glob.glob(path)

# for file in files:

# 	data = pd.read_csv(file)
# 	appended_data.append(data)

# df1, df2 = appended_data


# group_aoi_df = pd.concat([df1, df2], sort=False)
# group_aoi_df.to_csv(f"/study/midusref/DATA/Eyetracking/david_analysis/data_processed/009/group_aoi.csv")

# number_aoi = str(len(group_aoi_df))
# print (f"total number of AOIs for IAPS {iaps_number}: {number_aoi}")

# true_fixation_labeld_df = pd.read_csv(f"/study/midusref/DATA/Eyetracking/david_analysis/data_processed/009/009_{iaps_number}_true_fixation.csv")
# true_fixation_df = true_fixation_labeld_df.loc[true_fixation_labeld_df['final_data_type'] == "true_fixation"]

# print (true_fixation_df)

file = "/study/midusref/DATA/Eyetracking/david_analysis/QA/final.csv"

data = pd.read_csv(file)

two_aoi_df = data.loc[data['object_number'] == "Object03", "iaps_number"]
two_aoi_list = list(set(two_aoi_df))


three_aoi_df = data.loc[data['object_number'] == "Object04", "iaps_number"]
three_aoi_list = list(set(three_aoi_df))

appended_data_list = []


for iaps in two_aoi_list:

	df = data.loc[data['iaps_number'] == iaps]
	# Remove rows with IAPS 
	df = df[df.object_number != "Object01"]
	# Remove non-duplicates in subject number 
	# this removes any trials that has no fixation in object 3
	df = df.drop(df.drop_duplicates(['subject_number'], keep=False).index)
	#df = df.groupby('subject_number').count()
	appended_data_list.append(df)

df_final = pd.concat(appended_data_list)
#df_final.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/QA/group_aoi.csv", na_rep="NA", index=False)

# add total fixation time (in and out of AOI) for group AOIs (2+ 3+ 4)
df_final['group_total_fixation_duration_in_AOI'] = df_final.groupby(['subject_number', 'iaps_number'])['total_fixation_duration_in_AOI'].transform(sum)
df_final['group_total_fixation_duration_out_AOI'] = df_final['total_fixation_duration'] - df_final['group_total_fixation_duration_in_AOI']

# delete rows to have one measure per subject
df_final = df_final[df_final.object_number != "Object02"]

# change object_number to group
df_final['object_number'] = df_final['object_number'].map({'Object03':'group', 'Object04':'group'})
df_final['aoi_type'] = df_final['aoi_type'].map({'rectangle':'group', 'ellipse':'group'})

# Remove fixation times that are in negative 
df_final = df_final[df_final.group_total_fixation_duration_out_AOI >= 0]

# select columsn of interest
df_final = df_final[['subject_number', 'iaps_number', 'valence', 'aoi_type', 'object_number', 'group_total_fixation_duration_in_AOI', 'group_total_fixation_duration_out_AOI']]

# Change Column Names
df_final.rename(columns={'group_total_fixation_duration_in_AOI':'total_fixation_duration_in_AOI','group_total_fixation_duration_out_AOI':'total_fixation_duration_out_AOI'}, inplace=True)

merged = pd.concat([data, df_final], sort=True)

merged.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/QA/final_with_group.csv", na_rep="NA", index=False)

#result = df_final.groupby(np.arange(len(df_final))//2).sum()
#result.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/QA/result.csv", na_rep="NA", index=False)
