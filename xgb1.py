"""
Using XGBoost to train and test a classifier
"""

import numpy as np 
import pandas as pd 
import xgboost as xgb

from sklearn import preprocessing, cross_validation, metrics

rs = 19683

work_dir = '/media/maesh/Charming/Documents/Kaggle/2016/expedia/'

trainfile = work_dir + 'train_parsed_booking_drop.csv'
testfile = work_dir + 'test_parsed_booking_drop.csv'

df_train = pd.read_csv(trainfile)
df_test = pd.read_csv(testfile)

# Set up training matrix X and labels y
y = df_train['hotel_cluster'].values
df_train.drop(['hotel_cluster','Unnamed: 0','cnt'],axis=1,inplace=True)
X = df_train.values

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
num_round = 5

bst = xgb.train(param, xg_train, num_round, watchlist)

test_IDs = df_test['id'].values
df_test = df_test[df_train.columns]
xg_test = xgb.DMatrix(df_test.values,missing=np.nan)
test_prob = bst.predict(xg_test)
print(test_prob.shape)
def makespace(x):    
	return " ".join([str(int(z)) for z in x])

submissions = (-test_prob).argsort()[:,:5]
submit = pd.read_csv(work_dir+'sample_submission.csv')
intermediate = np.apply_along_axis(makespace, 1, submissions)
submit['hotel_cluster'] = intermediate