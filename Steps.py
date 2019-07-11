import sys
#sys.path.insert(0,'/data/2/gcgsgasb/data/work/NNM/AutoML')
sys.path.insert(0,'/home/ys86780')
import os
import gc
import time
import numpy as np
import model_tool
import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split
from itertools import chain, product
from model_tool import feature_eng as fe
from model_tool import iterations as iter
from model_tool import reports
import warnings
warnings.filterwarnings("ignore")

mypath = '/data/2/gcghkasb/data/work/Raghu/NBO/NBO_Nov_2018_cohort/Response_files';	# Folder where the datasets are stored
os.chdir(mypath)


#Data Import

df = pd.read_csv('dev_nov_2018_2.csv')	# Update the dataset
otv = pd.read_csv('otv_aug_2018_2.csv')	# Update the dataset




df.rename(columns={'MF_RESP' : 'target'}, inplace=True)	# Replace the response variable with the corresponding response variable
otv.rename(columns={'MF_RESP' : 'target'}, inplace=True)	# Replace the response variable with the corresponding response variable

id_vars = ['CLNT_NBR']
time_var = 'PERFORMANCE_PERIOD'
target_var = 'target'
cat_vars = ['CLNT_NBR']
df_cohort, otv_cohort = 201811, 201808	# Update the cohort months

df['PERFORMANCE_PERIOD'] = 201811	# Update the cohort months
otv['PERFORMANCE_PERIOD'] = 201808	# Update the cohort months


dev_itv=df



dev, itv = train_test_split(dev_itv, test_size = 0.3, random_state = 21)

version = 'v1'

os.chdir('/data/2/gcghkasb/data/work/Yash/HK_MF/5th_cut') #change this according to user(Results location)

if not os.path.exists('results'):
	os.makedirs('results')
	
if not os.path.exists('saved_objects'):
	os.makedirs('saved_objects')
	
if not os.path.exists('PDP'):
	os.makedirs('PDP')
	

# Data Cleaning

dev = dev.fillna(0)
itv = itv.fillna(0)
otv = otv.fillna(0)





# Model Iteration
iter.train_feature_iter(dev, id_vars, time_var, target_var, version)
iter.test_feature_iter(itv, id_vars, time_var, target_var, 'itv', version)
iter.test_feature_iter(otv, id_vars, time_var, target_var, 'otv', version)	
feature_iter_summary = iter.iteration_summary(iter_type = 'features', identifier = 'xgb_'+version)
final_features = iter.model_selection('features', rank = 1, version = 'v1') #select the rank you want

#Parameter Iteration
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [4, 5, 6],'subsample': [0.9, 1.0],'learning_rate': [0.01, 0.1, 0.05],'colsample_bytree': [ 0.9, 1.0]}
params_df = pd.DataFrame(list(product(*param_grid.values())), columns=param_grid.keys())
params_df = params_df.loc[:,['n_estimators', 'max_depth', 'subsample', 'learning_rate', 'colsample_bytree']]
iter.train_param_iter(dev, id_vars, time_var, target_var, params_df, final_features, version)
iter.test_param_iter(itv, id_vars, time_var, target_var, params_df, final_features, version, 'itv') 
iter.test_param_iter(otv, id_vars, time_var, target_var, params_df, final_features, version, 'otv')
param_iter_summary = iter.iteration_summary(iter_type = 'params', identifier = 'xgb_'+ version)
final_params = iter.model_selection('params', rank = 6, version = 'v1')#select parameters according to discretion

#Reports
path='/data/2/gcghkasb/data/work/Yash/HK_MF/5th_cut' #Results location
pdp_features = reports.pdp_var_reduction(dev, [itv, otv], ['itv', 'otv'], target_var, final_features, final_params, version, path)
reports.final_report(final_params, pdp_features, version, otv)
