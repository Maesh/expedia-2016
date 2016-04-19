"""
Adding column to main data set then saving as csv

Ideas: Parse timestamps, dummy variables for months and times of day?

- srch_ci and srch_co are dates as well, should break into dummy 
features for year,month,day. Also, can do a length of stay (in days)
and length of time before the trip variable (also in days)
"""

import pandas as pd 
import numpy as np 
from datetime import datetime
from gc import collect

# work_dir = '/media/ryan/Charlemagne/kaggle/2016/expedia/data/'
work_dir = '/media/maesh/Charming/Documents/Kaggle/2016/expedia/'

trainfile = work_dir + 'train.csv'
testfile = work_dir + 'test.csv'
destfile = work_dir + 'destinations.csv'

# df_destinations = pd.read_csv(destfile)

# Below code is not meant to be run concurrently, but instead
# in piecemeal chunks as needed

# On my desktop, enough RAM so no need to chunk
chunksize = 100000#37670293/19 # total rows evenly divisible by 19
reader = pd.read_csv(trainfile,iterator=True,chunksize=chunksize)
df_train = pd.concat(reader,ignore_index=True)

# Set up data store
# store = pd.HDFStore(work_dir + 'expedia.h5')

# for chunk in reader :
	# print(k)
parse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# break up that date_time column
dates_times = df_train['date_time'].apply(parse)

# Set-up df parsers
year_parse = lambda x: x.year
month_parse = lambda x: x.month
# day_parse = lambda x: x.day 
weekday_parse = lambda x: x.isoweekday()
hour_parse = lambda x: x.hour 

df_train['year'] = dates_times.apply(year_parse)
df_train['month'] = dates_times.apply(month_parse)
# chunk['day_of_month'] = dates_times.apply(day_parse)
df_train['day_of_week'] = dates_times.apply(weekday_parse)
df_train['hour'] = dates_times.apply(hour_parse)

# Now break up srch_ci and srch_co
# parse = lambda x: datetime.strptime(str(x), '%Y-%m-%d')
def parse(x):
	if type(x) is float :
		return np.nan
	else :
		return datetime.strptime(str(x), '%Y-%m-%d')

srch_ci_dts = df_train['srch_ci'].apply(parse)
srch_co_dts = df_train['srch_co'].apply(parse)

trip_length = srch_co_dts - srch_ci_dts

del dates_times
collect()
# time_to_trip = srch_ci_dts - dates_times.astype(object)

# Parser for the timedelta object
def timedelta_parse(x):
	if pd.isnull(x):
		return np.nan
	else :
		return float(x.days)

df_train['trip_length'] = trip_length.apply(timedelta_parse)
df_train.to_csv(work_dir+'train_parsed.csv',',')


############################################################
# Test Set 

df_test = pd.read_csv(testfile)

parse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# break up that date_time column
dates_times = df_test['date_time'].apply(parse)

# Set-up df parsers
year_parse = lambda x: x.year
month_parse = lambda x: x.month
weekday_parse = lambda x: x.isoweekday()
hour_parse = lambda x: x.hour 

df_test['year'] = dates_times.apply(year_parse)
df_test['month'] = dates_times.apply(month_parse)
df_test['day_of_week'] = dates_times.apply(weekday_parse)
df_test['hour'] = dates_times.apply(hour_parse)

# Now break up srch_ci and srch_co
# parse = lambda x: datetime.strptime(str(x), '%Y-%m-%d')
def parse(x):
	if type(x) is float :
		return np.nan
	else :
		try :
			return datetime.strptime(str(x), '%Y-%m-%d')
		except ValueError :
			return np.nan

srch_ci_dts = df_test['srch_ci'].apply(parse)
srch_co_dts = df_test['srch_co'].apply(parse)

trip_length = srch_co_dts - srch_ci_dts

# Parser for the timedelta object
def timedelta_parse(x):
	if pd.isnull(x):
		return np.nan
	else :
		return float(x.days)

df_test['trip_length'] = trip_length.apply(timedelta_parse)

# Drop columns if desired
df_test['is_booking'] = np.ones(len(df_test)) # add this in
df_test.drop(['srch_ci','srch_co','date_time'], axis = 1, inplace=True)
df_test.to_csv(work_dir+'test_parsed_booking_drop.csv',',')

################################################################
# Now just pull out only those training segments where a booking
# occurred

df_train = df_train[df_train['is_booking'] == 1]

parse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# break up that date_time column
dates_times = df_train['date_time'].apply(parse)

# Set-up df parsers
year_parse = lambda x: x.year
month_parse = lambda x: x.month
# day_parse = lambda x: x.day 
weekday_parse = lambda x: x.isoweekday()
hour_parse = lambda x: x.hour 

df_train['year'] = dates_times.apply(year_parse)
df_train['month'] = dates_times.apply(month_parse)
# chunk['day_of_month'] = dates_times.apply(day_parse)
df_train['day_of_week'] = dates_times.apply(weekday_parse)
df_train['hour'] = dates_times.apply(hour_parse)

# Now break up srch_ci and srch_co
# parse = lambda x: datetime.strptime(str(x), '%Y-%m-%d')
def parse(x):
	if type(x) is float :
		return np.nan
	else :
		return datetime.strptime(str(x), '%Y-%m-%d')

srch_ci_dts = df_train['srch_ci'].apply(parse)
srch_co_dts = df_train['srch_co'].apply(parse)

trip_length = srch_co_dts - srch_ci_dts

# Parser for the timedelta object
def timedelta_parse(x):
	if pd.isnull(x):
		return np.nan
	else :
		return float(x.days)

df_train['trip_length'] = trip_length.apply(timedelta_parse)

# now remove columns that are not going to help classifier
df_train.drop(['srch_ci','srch_co','date_time'], axis = 1, inplace=True)

# Save to CSV
df_train.to_csv(work_dir+'train_parsed_booking_drop.csv',',')