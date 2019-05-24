#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import glob
from functools import reduce

path = "/study/midusref/DATA/Eyetracking/david_analysis/data_processed/[0-9][0-9][0-9]/*_data_type_compiled.csv"

files = sorted(glob.glob(path))

print (len(files))

appended_data = []

# file_name = 'df_'
# full_name_list = []
# file_counter = 1

for file in files:

	data = pd.read_csv(file)

	data_df = data[['iaps_number', 'percent_valid']]
	data_df['iaps_number'] = data_df['iaps_number'].astype(str)
	#subject_df = file[['subject_number']]
	subject_number = file.split('/')[7]
	data_df = data_df.rename(columns={'percent_valid':subject_number})
	
	#data_transposed = data_df.T
	#print (data_transposed)

	#print (data_df)

	appended_data.append(data_df)

	# file_counter = str(file_counter)
	# full_name = (file_name + file_counter)
	# full_name_list.append(full_name)
	# file_counter = int(file_counter)
	# file_counter += 1

# print (len(full_name_list))

# transposed_data = []

# for data in appended_data:
# 	data = data.set_index('iaps_number').T
# 	transposed_data.append(data)

df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10,df_11, df_12, df_13, df_14, df_15, df_16, df_17, df_18, df_19, df_20, df_21, df_22, df_23, df_24, df_25, df_26, df_27, df_28, df_29, df_30, df_31, df_32, df_33, df_34, df_35, df_36, df_37, df_38, df_39, df_40, df_41, df_42, df_43, df_44, df_45, df_46, df_47, df_48, df_49, df_50, df_51, df_52, df_53, df_54, df_55, df_56, df_57, df_58, df_59, df_60, df_61, df_62, df_63, df_64, df_65, df_66, df_67, df_68, df_69, df_70, df_71, df_72, df_73, df_74, df_75, df_76, df_77, df_78, df_79, df_80, df_81, df_82, df_83, df_84, df_85, df_86, df_87, df_88, df_89, df_90, df_91, df_92, df_93, df_94, df_95, df_96, df_97, df_98, df_99, df_100, df_101, df_102, df_103, df_104, df_105, df_106, df_107, df_108, df_109, df_110, df_111, df_112, df_113, df_114, df_115, df_116, df_117, df_118, df_119, df_120, df_121, df_122, df_123, df_124, df_125 = appended_data

data_list = [df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10,df_11, df_12, df_13, df_14, df_15, df_16, df_17, df_18, df_19, df_20, df_21, df_22, df_23, df_24, df_25, df_26, df_27, df_28, df_29, df_30, df_31, df_32, df_33, df_34, df_35, df_36, df_37, df_38, df_39, df_40, df_41, df_42, df_43, df_44, df_45, df_46, df_47, df_48, df_49, df_50, df_51, df_52, df_53, df_54, df_55, df_56, df_57, df_58, df_59, df_60, df_61, df_62, df_63, df_64, df_65, df_66, df_67, df_68, df_69, df_70, df_71, df_72, df_73, df_74, df_75, df_76, df_77, df_78, df_79, df_80, df_81, df_82, df_83, df_84, df_85, df_86, df_87, df_88, df_89, df_90, df_91, df_92, df_93, df_94, df_95, df_96, df_97, df_98, df_99, df_100, df_101, df_102, df_103, df_104, df_105, df_106, df_107, df_108, df_109, df_110, df_111, df_112, df_113, df_114, df_115, df_116, df_117, df_118, df_119, df_120, df_121, df_122, df_123, df_124, df_125]


# cols = list(transposed_data[0].columns)
# cols.append("iaps_number")
# print (cols)


from functools import reduce

final_df = reduce(lambda x,y: pd.merge(x,y, on="iaps_number", how='outer'), data_list)
final_df.rename(columns ={'iaps_number':'subject_number'}, inplace=True)
print (final_df)

data_transposed = final_df.T
print (data_transposed)

data_transposed.to_csv("/study/midusref/DATA/Eyetracking/david_analysis/QA/validity_by_iaps.csv", header=False, na_rep='NA')


