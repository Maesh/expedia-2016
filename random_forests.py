"""
Using scikit learn to train and test a classifier
"""

import numpy as np 
import pandas as pd 

from scipy.sparse import hstack, csr_matrix
from sklearn import preprocessing, cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from scipy.stats.mstats import mode


rs = 19683

work_dir = '/media/maesh/Charming/Documents/Kaggle/2016/expedia/'

trainfile = work_dir + 'train_parsed_booking_drop.csv'
testfile = work_dir + 'test_parsed_booking_drop.csv'

df_train = pd.read_csv(trainfile) # in both, dfs have negative values for trip len?
df_test = pd.read_csv(testfile)

# make trip_length < 0 = nan
df_train.loc[df_train['trip_length']<0,'trip_length'] = np.nan
df_test.loc[df_test['trip_length']<0,'trip_length'] = np.nan
# Set up training matrix X and labels y
y = df_train['hotel_cluster'].values
df_train.drop(['hotel_cluster','Unnamed: 0','cnt'],axis=1,inplace=True)
X = df_train.values
test_IDs = df_test['id'].values
df_test = df_test[df_train.columns]

cols_to_use = ['site_name', 'user_location_region', 'is_package', \
	'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', \
	'hotel_market', 'hotel_country']

df_train = df_train[cols_to_use]
df_test = df_test[cols_to_use]

# For scikit learn, need to impute
for col in df_train.columns :
	df_train.loc[np.isnan(df_train[col]),col] = mode(df_train[col])[0][0]
	df_test.loc[np.isnan(df_test[col]),col] = mode(df_train[col])[0][0]

X = df_train.values#_sparse
# X_train, X_val, y_train, y_val = cross_validation.train_test_split(X, y, test_size=0.3)
X_test = df_test[df_train.columns].values#_sparse

rfc = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=1, random_state=rs)
eclf = BaggingClassifier(rfc, n_estimators=2, n_jobs=1,max_samples=0.1,max_features=3)

eclf.fit(X, y)

# Need to chunk to avoid memory error
test_chunks = np.array_split(df_test.values,50)
for i, chunk in enumerate(test_chunks):
	test_X = chunk
	if i > 0:
		test_y = np.concatenate( [test_y, eclf.predict_proba(test_X)])
	else:
		test_y = eclf.predict_proba(test_X)
	print(i)


test_prob = np.array(test_y)
print(test_prob.shape)

def makespace(x):    
	return " ".join([str(int(z)) for z in x])

submissions = (-test_prob).argsort()[:,:5]
submit = pd.read_csv(work_dir+'sample_submission.csv')
intermediate = np.apply_along_axis(makespace, 1, submissions)
submit['hotel_cluster'] = intermediate
submit.to_csv(work_dir+'rf100.bag10.subsamples0_5.8feats.2016.04.20.csv',header=True,index=False)