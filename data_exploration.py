import numpy as np 
import pandas as pd 

rs = 19683

work_dir = '/media/maesh/Charming/Documents/Kaggle/2016/expedia/'

trainfile = work_dir + 'train_parsed_booking_drop.csv'
testfile = work_dir + 'test_parsed_booking_drop.csv'

df_train = pd.read_csv(trainfile) # in both, dfs have negative values for trip len?
df_test = pd.read_csv(testfile)

