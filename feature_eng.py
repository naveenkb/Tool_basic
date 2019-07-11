import pandas as pd
from os import listdir
from os.path import isfile, join
import os
import glob
import gc
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import euclidean_distances



delta_3m=list(pd.Series(range(-2,0)))
delta_6m=list(pd.Series(range(-5,0)))
delta_9m=list(pd.Series(range(-8,0)))
delta_12m=list(pd.Series(range(-11,0)))
delta=[delta_3m,delta_6m,delta_9m,delta_12m]

def monthdelta(month, delta):
    m, y = ((month%100)+delta) % 12, month//100 + ((month%100)+delta-1) // 12
    if not m: m = 12
    return (100*y+m)

""" Returns a data frame with new sequence features
    
    Parameters:
    ----------
    df : Input dataframe with id_vars, time_vars, seq_vars 
    id_vars : List of ID columns in the dataframe df. (Generally CLNT_NBR or CRD_ACCT_NBR etc)
    time_var: String column name of the time variable (PERFORMANCE_PERIOD)
    seq_vars: List of variables which are time varying continuous values. Sequence features will be created for these variables
    cohort: Cohort year month. Format accepted YYYYMM e.g 201802
    n_month: Number of months of data to use for sequence features. Default to previous 12 months
"""

def sequence_fts(df, id_vars, time_var, seq_vars, cohort, isdev,n_month = 12):
    nmths = [monthdelta(cohort, -1*i) for i in range(n_month)]
    coh_df = df[df[time_var].isin(nmths)]
    coh_df.sort_values(id_vars + [time_var], ascending = [True]*len(id_vars) + [True], inplace = True)
    coh_df.set_index(id_vars + [time_var], inplace =True)
    coh_df = coh_df.loc[:,seq_vars]
    coh_df = coh_df.unstack()
    out_df = pd.DataFrame()
    mean_df=pd.DataFrame()
    cols=list(range(1,13))
    cols.insert(0,'seq_fts')
    print('Creating sequence features..')
    for i in seq_vars:
        temp_df = coh_df.loc[:,i]
        temp_df_t = temp_df.transpose()
        mean_val = temp_df_t.mean()
        temp_df_t = temp_df_t.fillna(mean_val)
        temp_df = temp_df_t.transpose()
        
        if isdev:
            mean_vec = temp_df.mean(axis = 0)
            mean_vec = mean_vec.values
            temp_mean=mean_vec.tolist()
            temp_mean.insert(0,i)
            t=pd.DataFrame([temp_mean],columns=cols)
            mean_df=pd.concat([mean_df,t],axis=0)
            
        else:
            mean_df=joblib.load('../saved_objects/mean_df_seq')
            mean_vec=list(mean_df[mean_df.seq_fts==i].iloc[0,1:])
            mean_vec=np.array(mean_vec)
        
        dist_df = vec_dist(temp_df, i,mean_vec)
        diff_count = vec_count(temp_df, i,mean_vec)
        out_df = pd.concat([out_df, dist_df, diff_count], axis = 1)
        
    out_df.reset_index(inplace =True)
    out_df[time_var] = cohort
    
    if isdev:
      
        joblib.dump(mean_df, '../saved_objects/mean_df_seq')
        
    return out_df

    
def vec_dist(df, col,mean_vec):
    mean_vec=np.array([mean_vec])
    out = pd.DataFrame(index = df.index)
    dist = euclidean_distances(df, mean_vec)
    out[col + '_euclidean_ dist'] = dist[:,0]
    out[col + '_cosine_sim'] = df.apply(lambda x: cosine(x.values, mean_vec),axis=1)
    out[col + '_corr']=df.apply(lambda x: np.corrcoef(x.values, mean_vec)[0,1],axis=1)
    return out

    
def vec_count(df, col,mean_vec):
    mean_vec=np.array(mean_vec)
    diff = df.sub(mean_vec)
    diff = diff > 0
    diff_count = pd.DataFrame(diff.sum(axis =1))
    diff_count.columns = [col + '_diff_count']
    return diff_count


def acc_fts(df,id_vars,ft_vars,delta_periods):
    print('Creating acceleration features..')
    for count in range(1,len(delta_periods) - 1):
        for i in range(len(ft_vars)):
            df[ft_vars[i]+ '_acc_' + str(count)]=df.iloc[:,i + len(id_vars) + 1 +4*(count-1)]-df.iloc[:,i + len(id_vars) + 1 +4*(count-1) +2*len(ft_vars)]
    return df


""" Returns a data frame with new velocity and acceleration features
    
    Parameters:
    ----------
    df : Input dataframe with id_vars, time_vars, ft_vars 
    id_vars : List of ID columns in the dataframe df. (Generally CLNT_NBR or CRD_ACCT_NBR etc)
    time_var: String column name of the time variable (PERFORMANCE_PERIOD)
    ft_vars: List of variables which are time varying continuous values. Velocity features will be created for these variables
    cohort: Cohort year month. Format accepted YYYYMM e.g 201802
    n_month: Number of months of data to use for sequence features. Default to previous 12 months
"""

def velocity_fts(df, id_vars, time_var, ft_vars, cohort, n_month = 12):
    delta_list = list(range(-1*n_month, -2, 3))
    out = df[df[time_var] == cohort].loc[:, id_vars + [time_var]]
    delta_periods = [monthdelta(cohort, dm) for dm in delta_list] + [cohort]
    df.sort_values(id_vars + [time_var], ascending = [True]*len(id_vars) + [False], inplace = True)
    a= len(delta_periods) - 1
    print('Creating velocity features..')
    for i in range(a):
        temp_df = df[df[time_var].isin(delta_periods[i:i+2])]
        diff_fts = pd.concat([temp_df[id_vars], temp_df[ft_vars].diff()], axis=1)
        diff_fts = diff_fts[(diff_fts.loc[:,id_vars].shift(1) == diff_fts.loc[:,id_vars]).all(axis =1)]
        
        ratio_fts = pd.concat([temp_df[id_vars], temp_df[ft_vars].pct_change()], axis=1)
        ratio_fts = ratio_fts[(ratio_fts.loc[:,id_vars].shift(1) == ratio_fts.loc[:,id_vars]).all(axis =1)]
        if i < (len(delta_list) - 1):
            end_m = str(-1*delta_list[i+1]) 
        else:
            end_m = 'curr_'
        diff_fts.columns = id_vars + [col + '_' + str(-1*delta_list[i]) + 'm_to_' + end_m + 'm_diff' for col in ft_vars]
        diff_fts.dropna(inplace = True)
        ratio_fts.columns = id_vars +  [col + '_' + str(-1*delta_list[i]) + 'm_to_' + end_m + 'm_ratio' for col in ft_vars]
        ratio_fts.dropna(inplace = True)
        out = out.merge(diff_fts, on = id_vars, how =  'left')
        out = out.merge(ratio_fts, on = id_vars, how =  'left')
    acc=acc_fts(out,id_vars,ft_vars,delta_periods)
    return acc


""" Returns a data frame with new centrality features

    Parameters:
    ----------
    df : Dataframe 
    id_vars : Groupby Parameter. To be given as a list.
    time_var : Time Parameter. To be given as a string.
    cols : Relevant columns. To be given as list.
    Cohort : Cohort month. Single value.
    Window : List of windows.

"""    
def centrality_fts(df,id_vars,time_var,cols,cohort,window):
    df_temp=df.loc[:,id_vars+[time_var]+cols]
    # print(df_temp[time_var].head())
    #print('CLNT_NBR' in list(df_temp.columns))
    periods=[0 for i in range(0,len(window))]
    out_df = df[df[time_var]==cohort].loc[:,id_vars+[time_var]]
    print('Creating centrality features..')
    for i in range(0,len(window)):
        periods[i]=[monthdelta(cohort,d) for d in delta[i]] + [cohort]
        df_temp2=df_temp[df_temp[time_var].isin(periods[i])]
        df_temp3=df_temp2.groupby(id_vars,as_index=False)[cols].agg({'min', 'max', 'median', 'mean'})
        df_temp3.columns = df_temp3.columns.map(('_'+str(window[i])).join)
        df_temp3.reset_index(inplace=True)
        out_df=out_df.merge(df_temp3,on=id_vars,how='left')
    
    return out_df
    

""" Returns a data frame with target encoded columns for categorical columns features

    Parameters:
    ----------
    df : Dataframe 
    target : target column name
    id_vars : List of ID columns in the dataframe df. (Generally CLNT_NBR or CRD_ACCT_NBR etc)
    time_var: String column name of the time variable (PERFORMANCE_PERIOD)
    cohort: Cohort month. Single value
    categorical_cols: List of categorical columns for which target encoding needs to be done
"""        
def mean_encoding(df, target, id_vars, time_var, cohort, categorical_cols):
    coh_df = df[df[time_var] == cohort]
    out = coh_df.loc[:,id_vars+[time_var]]
    print('Creating target encoded features..')
    for i in categorical_cols:
        temp_df = df.loc[:,[i]+[target]]
        out[i + 'mean_enc']= coh_df[i].map(coh_df.groupby(i)[target].mean())
    return out    
