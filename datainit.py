"""
Adding column to main data set then saving as csv

Ideas: Parse timestamps, dummy variables for months and times of day?

- srch_ci and srch_co are dates as well, should break into dummy 
features for year,month,day. Also, can do a length of stay (in days)
and length of time before the trip variable (also in days)
"""

import pandas as pd 
import numpy as np 

work_dir = '/media/ryan/Charlemagne/kaggle/2016/expedia/data'
trainfile = work_dir + 'data/train.csv'
testfile = work_dir + 'data/test.csv'
destfile = work_dir + 'data/destinations.csv'

df_destinations = pd.read_csv(destfile)

# column is 'date_time' that we want to convert
# function is datetime.strptime(d,'%Y-%m-%d %H:%M:%S')

monthdict{1:'Jan',
		  2:'Feb',
		  3:'Mar',
		  4:'Apr',
		  5:'May',
		  6:'Jun',
		  7:'Jul',
		  8:'Aug',
		  9:'Sep',
		  10:'Oct',
		  11:'Nov',
		  12:'Dec'}

timeofday_dict{3:'00-03',
			   6:'03-06',
			   9:'06-09',
			   12:'09-12',
			   15:'12-15',
			   18:'15-18',
			   21:'18-21',
			   24:'21-24'}


"""
Pseudocode

Loop through train and test 
	import 100k rows
	add new cols for months and time of days (20 new columns)
	go through rows, one hot encoding the time stamps as above
		if df_new['date_time'][row].time < 3:
			df_new['00-03'][row index] = timeofday_dict[3]
		elif month >= 3 & month < 6:
			df_new['03-06'][row index] = timeofday_dict[6]
			.
			.
			.
	append new csv file with added columns
"""