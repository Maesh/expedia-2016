import numpy as np 
import pandas as pd
from scipy.sparse import hstack, csr_matrix 
from sklearn import preprocessing, cross_validation, metrics

rs = 19683

work_dir = '/media/maesh/Charming/Documents/Kaggle/2016/expedia/'

trainfile = work_dir + 'train_parsed_booking_drop.csv'
testfile = work_dir + 'test_parsed_booking_drop.csv'

df_train = pd.read_csv(trainfile) # in both, dfs have negative values for trip len?
df_test = pd.read_csv(testfile)

hc_group = df_train.groupby(df_train['srch_destination_id'])
submit = pd.DataFrame()

def makespace(x):    
	return " ".join([str(int(z)) for z in x])

for idx in df_test['id'].values :
	submit['id'] = idx
	hotels = hc_group.get_group(idx)['hotel_cluster'].value_counts()[:5].index.get_values()
	submit['hotel_cluster'] = makespace(hotels)