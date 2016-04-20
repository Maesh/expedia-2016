"""
Using XGBoost to train and test a classifier
"""

import numpy as np 
import pandas as pd 
import xgboost as xgb
from scipy.sparse import hstack
from sklearn import preprocessing, cross_validation, metrics

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

df_train = df_train.to_sparse(fill_value=0)
df_test = df_test.to_sparse(fill_value=0)
# [u'Unnamed: 0', u'site_name', u'posa_continent',
#        u'user_location_country', u'user_location_region',
#        u'user_location_city', u'orig_destination_distance', u'user_id',
#        u'is_mobile', u'is_package', u'channel', u'srch_adults_cnt',
#        u'srch_children_cnt', u'srch_rm_cnt', u'srch_destination_id',
#        u'srch_destination_type_id', u'is_booking', u'cnt', u'hotel_continent',
#        u'hotel_country', u'hotel_market', u'hotel_cluster', u'year', u'month',
#        u'day_of_week', u'hour', u'trip_length']

lb = preprocessing.LabelBinarizer(sparse_output=True)

binarizer_list = ['site_name','posa_continent','user_location_country',\
				'user_location_region','user_location_city','channel',\
				'srch_destination_type_id','hotel_continent','hotel_country',\
				'hotel_market','year','month','day_of_week','hour']

# Encode dummies as categorical ([df, pd.get_dummies(df['YEAR'])], axis=1)
df_train_sparse = df_train.to_sparse(fill_value=0)
df_test_sparse = df_test.to_sparse(fill_value=0)

for item in binarizer_list :
	df_train_sparse
	df_train = pd.concat([df_train, pd.DataFrame(lb.fit_transform(df_train[item]))],axis = 1)
	df_test = pd.concat([df_test, pd.DataFrame(lb.transform(df_test[item]))],axis = 1)



df_train = pd.concat([df_train, pd.get_dummies(df_train['site_name'], \
	prefix='site_name', sparse = True)], axis=1)
df_train = pd.concat([df_train, pd.get_dummies(df_train['posa_continent'], \
	prefix='posa_continent', sparse = True)], axis=1)
df_train = pd.concat([df_train, pd.get_dummies(df_train['user_location_country'], \
	prefix='user_location_country', sparse = True)], axis=1)
df_train = pd.concat([df_train, pd.get_dummies(df_train['user_location_region'], \
	prefix='user_location_region', sparse = True)], axis=1)
df_train = pd.concat([df_train, pd.get_dummies(df_train['user_location_city'], \
	prefix='user_location_city', sparse = True)], axis=1)
df_train = pd.concat([df_train, pd.get_dummies(df_train['channel'], \
	prefix='channel', sparse = True)], axis=1)
df_train = pd.concat([df_train, pd.get_dummies(df_train['srch_destination_type_id'], \
	prefix='srch_destination_type_id', sparse = True)], axis=1)
df_train = pd.concat([df_train, pd.get_dummies(df_train['hotel_continent'], \
	prefix='hotel_continent', sparse = True)], axis=1)
df_train = pd.concat([df_train, pd.get_dummies(df_train['hotel_country'], \
	prefix='hotel_country', sparse = True)], axis=1)
df_train = pd.concat([df_train, pd.get_dummies(df_train['hotel_market'], \
	prefix='hotel_market', sparse = True)], axis=1)
df_train = pd.concat([df_train, pd.get_dummies(df_train['year'], \
	prefix='year', sparse = True)], axis=1)
df_train = pd.concat([df_train, pd.get_dummies(df_train['month'], \
	prefix='month', sparse = True)], axis=1)
df_train = pd.concat([df_train, pd.get_dummies(df_train['day_of_week'], \
	prefix='day_of_week', sparse = True)], axis=1)
df_train = pd.concat([df_train, pd.get_dummies(df_train['hour'], \
	prefix='hour', sparse = True)], axis=1)

df_test = pd.concat([df_test, pd.get_dummies(df_test['site_name'])], axis=1, prefix='site_name')
df_test = pd.concat([df_test, pd.get_dummies(df_test['posa_continent'])], axis=1, prefix='posa_continent')
df_test = pd.concat([df_test, pd.get_dummies(df_test['user_location_country'])], axis=1, prefix='user_location_country')
df_test = pd.concat([df_test, pd.get_dummies(df_test['user_location_region'])], axis=1, prefix='user_location_region')
df_test = pd.concat([df_test, pd.get_dummies(df_test['user_location_city'])], axis=1, prefix='user_location_city')
df_test = pd.concat([df_test, pd.get_dummies(df_test['channel'])], axis=1, prefix='channel')
df_test = pd.concat([df_test, pd.get_dummies(df_test['srch_destination_type_id'])], axis=1, prefix='srch_destination_type_id')
df_test = pd.concat([df_test, pd.get_dummies(df_test['hotel_continent'])], axis=1, prefix='hotel_continent')
df_test = pd.concat([df_test, pd.get_dummies(df_test['hotel_country'])], axis=1, prefix='hotel_country')
df_test = pd.concat([df_test, pd.get_dummies(df_test['hotel_market'])], axis=1, prefix='hotel_market')
df_test = pd.concat([df_test, pd.get_dummies(df_test['year'])], axis=1, prefix='year')
df_test = pd.concat([df_test, pd.get_dummies(df_test['month'])], axis=1, prefix='month')
df_test = pd.concat([df_test, pd.get_dummies(df_test['day_of_week'])], axis=1, prefix='day_of_week')
df_test = pd.concat([df_test, pd.get_dummies(df_test['hour'])], axis=1, prefix='hour')


X_train, X_val, y_train, y_val = cross_validation.train_test_split(X, y, test_size=0.3)

xg_train = xgb.DMatrix(X_train,label=y_train,missing=np.nan)
xg_test = xgb.DMatrix(X_val,label=y_val,missing=np.nan)

param = {}

param['objective'] = 'multi:softprob'
param['eta'] = 0.05
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = -1
param['subsample'] = 0.8
param['colsample_bytree'] = 0.8
param['eval_metric'] = 'mlogloss'
param['num_class'] = len(np.unique(y))

watchlist = [(xg_train,'train'),(xg_test,'eval')]
num_round = 40

bst = xgb.train(param, xg_train, num_round, watchlist)


xg_test = xgb.DMatrix(df_test.values,missing=np.nan)
test_prob = bst.predict(xg_test)
print(test_prob.shape)
def makespace(x):    
	return " ".join([str(int(z)) for z in x])

submissions = (-test_prob).argsort()[:,:5]
submit = pd.read_csv(work_dir+'sample_submission.csv')
intermediate = np.apply_along_axis(makespace, 1, submissions)
submit['hotel_cluster'] = intermediate
submit.to_csv(work_dir+'xgb1.sub1.40rounds.2016.04.19.csv',header=True,index=False)