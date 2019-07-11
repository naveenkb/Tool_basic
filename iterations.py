from itertools import chain, product
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib
import pandas.core.algorithms as algos
import scipy.stats.stats as stats
import statsmodels.api as sm
import matplotlib
import collections
matplotlib.use('agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 7})

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.style.use('seaborn')



""" Treats missing values in the given training data
    
    Parameters:
    ----------
    df : Input dataframe
    id_vars : list of ID variables in the data
    time_var : time variable in the data
    target_var : target variable name 
	version : versin name of the iteration. This is used to store the imputation file
	cat_vars : List of categorical variables
"""
def missing_trt_train(df, id_vars, time_var, target_var, version = 'v1', cat_vars =None):
    if cat_vars is not None:
        non_x_vars = id_vars + [time_var] + cat_vars + [target_var]
    else:
        non_x_vars = id_vars + [time_var] + [target_var]
    num_vars = [i for i in df.columns if i not in non_x_vars]
    med = df.loc[:,num_vars].median()
    df = df.fillna(med)
    med = pd.DataFrame(med).reset_index()
    med.columns = ['features', 'median_val']
    med.to_csv('median_treatment_ ' + version + '.csv', index = False)
    if cat_vars is not None:
        mode = df.loc[:,cat_vars].mode()
        df = df.fillna(mode)
        mode = pd.DataFrame(med).reset_index()
        mode.columns = ['features', 'mode_val']
        mode.to_csv('mode_treatment_ ' + version + '.csv', index = False)
    return df
    

""" Treats missing values in the given test data
    
    Parameters:
    ----------
    df : Input dataframe
	version : versin name of the iteration. This is used to store the imputation file
	cat_vars : List of categorical variables
"""
    
def missing_trt_test(df, version, cat_vars = None):
    med_trn = pd.read_csv('median_treatment_ ' + version + '.csv')
    med = pd.Series(med_trn['median_val'])
    med.index = med_trn['features']
    df = df.fillna(med)
    if cat_vars is not None:
        mode_trn = pd.read_csv('mode_treatment_ ' + version + '.csv')
        mode = pd.Series(mode_trn['mode_val'])
        mode.index = mode_trn['features']
        df = df.fillna(mode)
    return df

""" Returns a KS table
    
    Parameters:
    ----------
    score : An array of Predicted probability for each observation
    response : Actual response for each observation
    identifier: output suffix dataset name to be added
"""

def KS_table(score, response, identifier):
    print('getting KS..')
    group = 10
    df = pd.DataFrame({'score': score, 'response' : response})
    df = df.sort_values(['score'], ascending = [False])
    bin_size = len(score)/group
    rem = len(score) % group
    if rem == 0:
        df['groups'] = list(np.repeat(range(rem+1,11), bin_size))
    else:
        df['groups'] = list(np.repeat(range(1,rem+1),bin_size + 1)) + list(np.repeat(range(rem+1,11), bin_size))
    grouped = df.groupby('groups', as_index =False)
    agg = pd.DataFrame({'Total_Obs': grouped.count().response})
    agg['No.Res'] = grouped.sum().response
    agg['No.Non_Res'] = agg['Total_Obs'] - agg['No.Res']
    agg['min_pred'] = grouped.min().score
    agg['max_pred'] = grouped.max().score
    agg['pred_rr'] = grouped.mean().score
    agg['cum_no_res'] = agg['No.Res'].cumsum()
    agg['cum_no_non_res'] = agg['No.Non_Res'].cumsum()
    agg['percent_cum_res'] = agg['cum_no_res']/agg['cum_no_res'].max()
    agg['percent_cum_non_res'] = agg['cum_no_non_res']/agg['cum_no_non_res'].max()
    agg['KS'] = agg['percent_cum_res'] - agg['percent_cum_non_res']
    agg.to_csv('results/KS_table_'+ identifier + '.csv', index = False)
    return(agg)




""" Train models based on mutliple set of features
    
    Parameters:
    ----------
    df : Input dataframe
    y: target column name
    version : Identifier of iteration version
"""
def train_feature_iter(df, id_vars, time_var, y, version, save_pred = False):
    print('Conducting first trial run with all features..')
    out = df.loc[:, id_vars + [time_var]]
    out['actual'] = df[y]
    model = XGBClassifier(seed=10)
    drop_vars = id_vars + [time_var] + [y]
    model.fit(df.drop(drop_vars, axis=1), df[y])
    imp_features_df = pd.DataFrame({'feature_names': df.drop(drop_vars, axis=1).columns, 'importance': model.feature_importances_})
    imp_features_df.sort_values('importance', ascending=False, inplace=True)
    imp_features_df.to_csv('feature_importance_' + version + '.csv', index=False)
    if save_pred:
        pred = model.predict_proba(df.drop(drop_vars, axis=1))
        out['pred'] = pred
        out.to_csv('results/pred_dev_overall_'+ version + '.csv', index=False)
    
    print('Iterating on different feature combination in dev..')
    summary_df =  pd.DataFrame()
    imp_features_df = pd.read_csv('feature_importance_' + version + '.csv')
    imp_features_df = imp_features_df.sort_values('importance', ascending = False)
    imp_features = imp_features_df[imp_features_df['importance'] != 0]
    feature_count = len(imp_features)
    if feature_count> 100:
        iter = list(range(10, 100, 10)) + list(range(100, feature_count, 50)) + [feature_count]
    else:
        iter = list(range(10, feature_count, 10)) + [feature_count]
    target = df[y]
    for i in iter:
        print('feature count {0}'.format(i))
        curr_features = imp_features.feature_names[:i]
        curr_X = df.loc[:,curr_features]
        model = XGBClassifier(seed = 10)
        model.fit(curr_X, target)
        joblib.dump(model, 'saved_objects/xgb_' + str(i) + '_features_'+ version + '.joblib', compress = 1)
        if save_pred:
            pred = model.predict_proba(df.drop(drop_vars, axis=1))
            out['pred'] = pred[:,1]
            out.to_csv('results/pred_dev_'+ str(i) + '_features_'+ version + '.csv', index=False)
        feature_imp = pd.DataFrame({'feature_names': curr_features, 'importance': model.feature_importances_})
        feature_imp.to_csv('results/feature_importance_' + str(i) + '_features_'+ version + '.csv', index=False)
        score = model.predict_proba(curr_X)
        ks = KS_table(score[:,1], target, 'dev_xgb_' + str(i) + '_features_' + version)
        breaks = np.diff(ks['No.Res']) > 0
        dec_break = (np.diff(ks['No.Res']) > 0).any()
        ks_val = ks.KS.max()
        ks_decile = ks.KS.idxmax() + 1
        capture = ks['percent_cum_res'][3]
        if dec_break:
            break_dec = min([idx for idx, x in enumerate(breaks) if x]) + 2
            summary_df = summary_df.append(pd.DataFrame([[i, ks_val, break_dec, ks_decile, capture]],columns=['feature_count', 'dev_ks', 'dev_ro_break', 'dev_ks_decile', 'dev_capture']))
        else:
            break_dec = np.nan
            summary_df = summary_df.append(pd.DataFrame([[i, ks_val, break_dec, ks_decile, capture]],columns=['feature_count', 'dev_ks', 'dev_ro_break', 'dev_ks_decile', 'dev_capture']))
    summary_df.to_csv('results/summary_df_features_xgb_' + version + '.csv', index =False)









""" predict on itv and otv on all the set of features
    
    Parameters:
    ----------
    df : Input dataframe
    y: target column name
    dset : dataset type of test (itv or otv)
    version : Identifier of iteration version
"""

def test_feature_iter(df, id_vars, time_var, y, dset, version, save_pred = False):
    print('Iterating on different feature combination in {} ..'.format(dset))
    out = df.loc[:, id_vars + [time_var]]
    out['actual'] = df[y]
    summary_df_test =  pd.DataFrame()
    print('in test feature iter function ..')
    imp_features_df = pd.read_csv('feature_importance_' + version + '.csv')
    imp_features_df = imp_features_df.sort_values('importance', ascending = False)
    imp_features = imp_features_df[imp_features_df['importance'] != 0]
    feature_count = len(imp_features)
    if feature_count> 100:
        iter = list(range(10, 100, 10)) + list(range(100, feature_count, 50)) + [feature_count]
    else:
        iter = list(range(10, feature_count, 10)) + [feature_count]
    target = df[y]
    summary_df = pd.read_csv('results/summary_df_features_xgb_' + version + '.csv')
    for i in iter:
        print(dset + ' iteration {}'.format(i))
        curr_features = imp_features.feature_names[:i]
        curr_X = df.loc[:,curr_features]
        model = joblib.load('saved_objects/xgb_' +  str(i) + '_features_'+ version + '.joblib')
        if save_pred:
            pred = model.predict_proba(curr_X)
            out['pred'] = pred[:,1]
            out.to_csv('results/pred_' + dset + '_' + str(i) + '_features_'+ version + '.csv', index=False)
        score = model.predict_proba(curr_X)
        ks = KS_table(score[:,1], target, dset +  '_xgb_' + str(i) + '_features_' + version)
        breaks = np.diff(ks['No.Res']) > 0
        dec_break = (np.diff(ks['No.Res']) > 0).any()
        ks_val = ks.KS.max()
        ks_decile = ks.KS.idxmax() + 1
        capture = ks['percent_cum_res'][3]
        if dset=='otv':
            temp_otv=df.loc[:,curr_features]
            temp_otv['score']=score[:,1]
            dev_ks=pd.read_csv('results/KS_table_'+ 'dev' + '_xgb_' + str(i)+'_features_' + version+ '.csv')
            temp_otv['dev_grps'] = np.where(temp_otv.score > dev_ks.min_pred[0], 0, np.where(temp_otv.score > dev_ks.min_pred[1], 1, np.where(temp_otv.score > dev_ks.min_pred[2], 2,np.where(temp_otv.score > dev_ks.min_pred[3], 3, np.where(temp_otv.score > dev_ks.min_pred[4], 4, np.where(temp_otv.score > dev_ks.min_pred[5], 5, np.where(temp_otv.score > dev_ks.min_pred[6], 6, np.where(temp_otv.score > dev_ks.min_pred[7], 7, np.where(temp_otv.score > dev_ks.min_pred[8], 8, np.where(temp_otv.score <= dev_ks.min_pred[8], 9,-99))))))))))
            grouped = temp_otv.groupby('dev_grps')
            agg1 = pd.DataFrame({'Total_Obs': grouped.count().score})
            agg1['percent_obs'] = agg1['Total_Obs']*100/agg1['Total_Obs'].sum()
            agg1['psi'] = (0.1 - agg1['percent_obs']/100) * np.log(0.1/(agg1['percent_obs']/100))
        if dec_break:
            break_dec = min([idx for idx, x in enumerate(breaks) if x]) + 2
            if dset=='otv':
                summary_df_test = summary_df_test.append(pd.DataFrame([[i, ks_val, break_dec, ks_decile, capture,agg1['psi'].sum()]],columns=['feature_count', dset+ '_ks', dset +'_ro_break', dset +'_ks_decile', dset + '_capture','psi']))
            else:    
                summary_df_test = summary_df_test.append(pd.DataFrame([[i, ks_val, break_dec, ks_decile, capture]],columns=['feature_count', dset+ '_ks', dset +'_ro_break', dset +'_ks_decile', dset + '_capture']))
        else:
            break_dec = np.nan
            if dset=='otv':
                summary_df_test = summary_df_test.append(pd.DataFrame([[i, ks_val, break_dec, ks_decile, capture,agg1['psi'].sum()]],columns=['feature_count', dset+ '_ks', dset +'_ro_break', dset +'_ks_decile', dset + '_capture','psi']))
            else:
                summary_df_test = summary_df_test.append(pd.DataFrame([[i, ks_val, break_dec, ks_decile, capture]],columns=['feature_count', dset+ '_ks', dset +'_ro_break', dset +'_ks_decile', dset + '_capture']))
    summary_df_test.reset_index(drop=True, inplace =True)
    summary_df[dset + '_ks'] = summary_df_test[dset + '_ks']
    summary_df[dset +'_ro_break'] = summary_df_test[dset +'_ro_break']
    summary_df[dset + '_ks_decile'] = summary_df_test[dset + '_ks_decile']
    summary_df[dset + '_capture'] = summary_df_test[dset + '_capture']
    summary_df['dev_' + dset + '_ks_diff'] = (summary_df['dev_ks'] - summary_df[dset + '_ks'])*100/summary_df['dev_ks']
    if dset=='otv':
        summary_df['psi']=summary_df_test['psi']
    summary_df.to_csv('results/summary_df_features_xgb_' + version + '.csv', index =False)




""" Performs grid search for given feature combination
    
    Parameters:
    ----------
    df : Input dataframe
    y: target column name
    params_df : pandas dataframe of the hyper parameter grid
    imp_features : list of important features to build the model
    version : Identifier of iteration version
"""

def train_param_iter(df,  id_vars, time_var, y, params_df, imp_features, version, save_pred = False):
    print('Iterating on different hyper parameters..')
    out = df.loc[:, id_vars + [time_var]]
    out['actual'] = df[y]
    summary_df = pd.DataFrame()
    identifier = str(len(imp_features)) + 'var'
    alias = {
    'n_estimators': 'est',
    'max_depth': 'max_dep',
    'subsample': 'sub_s',
    'learning_rate': 'learn_r',
    'colsample_bytree': 'col_samp',
	'reg_lambda': 'lambda'
	}
    for idx, row in params_df.astype(object).iterrows():
        print('Iteration {0} of {1}'.format(idx +1, params_df.shape[0]))
        tup = [i for i in zip([alias.get(row.index[j]) for j in range(len(params_df.columns))], row.values.astype(str))]
        params_str = [''.join(t) for t in tup]
        identifier = identifier  +'_'.join(params_str) + '_' + version
        param = row.to_dict()
        model = XGBClassifier(seed = 10, **param, nthread = 10)
        model.fit(df.loc[:,imp_features], df[y])
        joblib.dump(model, 'saved_objects/xgb_' + identifier)
        feature_imp = pd.DataFrame({'feature_names': imp_features, 'importance': model.feature_importances_})
        feature_imp.to_csv('results/feature_importance_' + identifier + '.csv', index=False)
        score = model.predict_proba(df.loc[:,imp_features])
        if save_pred:
            out['pred'] = score[:,1]
            out.to_csv('results/pred_dev_'+ identifier + '.csv', index=False)
        ks = KS_table(score[:,1],df[y], 'dev_xgb_' + identifier)
        breaks = np.diff(ks['No.Res']) > 0
        dec_break = (np.diff(ks['No.Res']) > 0).any()
        ks_val = ks.KS.max()
        ks_decile = ks.KS.idxmax() + 1
        capture = ks['percent_cum_res'][3]
        if dec_break:
            break_dec = min([idx for idx, x in enumerate(breaks) if x]) + 2
            summary_df = summary_df.append(pd.DataFrame([ list(row.values) + [ ks_val, break_dec, ks_decile, capture]],columns= list(row.index) + ['dev_ks', 'dev_ro_break', 'dev_ks_decile', 'dev_capture']))
        else:
            break_dec = np.nan
            summary_df = summary_df.append(pd.DataFrame([ list(row.values) + [ ks_val, break_dec, ks_decile, capture]],columns= list(row.index) + ['dev_ks', 'dev_ro_break', 'dev_ks_decile', 'dev_capture']))
        identifier = str(len(imp_features)) + 'var'
    summary_df.to_csv('results/summary_df_params_xgb_' + version + '.csv', index =False)


""" Gets ITV and OTV results for given feature combination using models built in train_param_iters
    
    Parameters:
    ----------
    df : Input dataframe
    y: target column name
    params_df : pandas dataframe of the hyper parameter grid
    imp_features : list of important features to build the model
    version : Identifier of iteration version
    dset : dataset type of test (itv or otv)
"""

def test_param_iter(df, id_vars, time_var, y, params_df, imp_features, version, dset, save_pred = False):
    print('Applying parameter iteration on {0}'.format(dset))
    out = df.loc[:, id_vars + [time_var]]
    out['actual'] = df[y]
    summary_df_test =  pd.DataFrame()
    #psi_df=pd.DataFrame()

    summary_df = pd.read_csv('results/summary_df_params_xgb_' + version + '.csv')
    identifier = str(len(imp_features)) + 'var'
    alias = {
    'n_estimators': 'est',
    'max_depth': 'max_dep',
    'subsample': 'sub_s',
    'learning_rate': 'learn_r',
    'colsample_bytree': 'col_samp',
	'reg_lambda': 'lambda'
	}
    for idx, row in params_df.astype(object).iterrows():
        print('here')
        print('Iteration {0} of {1}'.format(idx +1, params_df.shape[0]))
        tup = [i for i in zip([alias.get(row.index[j]) for j in range(len(params_df.columns))], row.values.astype(str))]
        params_str = [''.join(t) for t in tup]
        identifier = identifier +'_'.join(params_str) + '_' + version
        param = row.to_dict()
        model = joblib.load('saved_objects/xgb_' + identifier)
        score = model.predict_proba(df.loc[:,imp_features])
        if save_pred:
            out['pred'] = score[:,1]
            out.to_csv('results/pred_' + dset + '_'+ identifier + '.csv', index=False)
        
        ks = KS_table(score[:,1],df[y], dset + '_xgb_' + identifier)
        breaks = np.diff(ks['No.Res']) > 0
        dec_break = (np.diff(ks['No.Res']) > 0).any()
        ks_val = ks.KS.max()
        ks_decile = ks.KS.idxmax() + 1
        capture = ks['percent_cum_res'][3]
        if dset=='otv':
            temp_otv=df.loc[:,imp_features]
            temp_otv['score']=score[:,1]
            dev_ks=pd.read_csv('results/KS_table_'+ 'dev' + '_xgb_' + identifier + '.csv')
            temp_otv['dev_grps'] = np.where(temp_otv.score > dev_ks.min_pred[0], 0, np.where(temp_otv.score > dev_ks.min_pred[1], 1, np.where(temp_otv.score > dev_ks.min_pred[2], 2,np.where(temp_otv.score > dev_ks.min_pred[3], 3, np.where(temp_otv.score > dev_ks.min_pred[4], 4, np.where(temp_otv.score > dev_ks.min_pred[5], 5, np.where(temp_otv.score > dev_ks.min_pred[6], 6, np.where(temp_otv.score > dev_ks.min_pred[7], 7, np.where(temp_otv.score > dev_ks.min_pred[8], 8, np.where(temp_otv.score <= dev_ks.min_pred[8], 9,-99))))))))))
            grouped = temp_otv.groupby('dev_grps')
            agg1 = pd.DataFrame({'Total_Obs': grouped.count().score})
            agg1['percent_obs'] = agg1['Total_Obs']*100/agg1['Total_Obs'].sum()
            agg1['psi'] = (0.1 - agg1['percent_obs']/100) * np.log(0.1/(agg1['percent_obs']/100))   
        if dec_break:
            break_dec = min([idx for idx, x in enumerate(breaks) if x]) + 2
            if (dset=='otv'):
                summary_df_test = summary_df_test.append(pd.DataFrame([list(row.values) + [ks_val, break_dec, ks_decile, capture,agg1['psi'].sum()]], columns= list(row.index) + [dset+ '_ks', dset +'_ro_break', dset +'_ks_decile', dset + '_capture','psi']))
            else:
                summary_df_test = summary_df_test.append(pd.DataFrame([list(row.values) + [ks_val, break_dec, ks_decile, capture]], columns= list(row.index) + [dset+ '_ks', dset +'_ro_break', dset +'_ks_decile', dset + '_capture']))
        else:
            break_dec = np.nan
            if (dset=='otv'):
                summary_df_test = summary_df_test.append(pd.DataFrame([list(row.values) + [ks_val, break_dec, ks_decile, capture,agg1['psi'].sum()]], columns= list(row.index) + [dset+ '_ks', dset +'_ro_break', dset +'_ks_decile', dset + '_capture','psi']))
            else:
                summary_df_test = summary_df_test.append(pd.DataFrame([list(row.values) + [ks_val, break_dec, ks_decile, capture]], columns= list(row.index) + [dset+ '_ks', dset +'_ro_break', dset +'_ks_decile', dset + '_capture']))
        
        identifier = str(len(imp_features)) + 'var'
        
    summary_df_test.reset_index(drop=True, inplace =True)
    summary_df[dset + '_ks'] = summary_df_test[dset + '_ks']
    summary_df[dset +'_ro_break'] = summary_df_test[dset +'_ro_break']
    summary_df[dset + '_ks_decile'] = summary_df_test[dset + '_ks_decile']
    summary_df[dset + '_capture'] = summary_df_test[dset + '_capture']
    summary_df['dev_' + dset + '_ks_diff'] = (summary_df['dev_ks'] - summary_df[dset + '_ks'])*100/summary_df['dev_ks']
    if dset=='otv':
        summary_df['psi']=summary_df_test['psi']
   
    summary_df.to_csv('results/summary_df_params_xgb_' + version + '.csv', index =False)

""" Return top N best iterations base don features or paramter
    
    Parameters:
    ----------
    top : Top n best iteration
	iter_type : either features or paramter
    version : Identifier of iteration version

"""


def iteration_summary(iter_type, identifier):
    summary_df = pd.read_csv('results/summary_df_' + iter_type + '_' + identifier + '.csv')
    '''
    if top > summary_df.shape[0]:
        print('Top {} iterations with features are :'.format(top))
        print(summary_df.feature_count)
    '''
    summary_df['itv_otv_ks_diff'] = (summary_df['itv_ks'] - summary_df['otv_ks'])*100/summary_df['itv_ks']
    
    summary_df['dev_otv_diff_cat'] = np.where(summary_df['dev_otv_ks_diff'] <= 10, 1, 0)
    summary_df['otv_ro_cat'] = np.where(summary_df['otv_ro_break'].fillna(11) > 7, 1, 0)
    summary_df['itv_ro_cat'] = np.where(summary_df['itv_ro_break'].fillna(11) > 7, 1, 0)
    summary_df['dev_ro_cat'] = np.where(summary_df['dev_ro_break'].fillna(11) > 7, 1, 0)
    cols = ['dev_otv_diff_cat', 'otv_ro_cat', 'itv_ro_cat', 'dev_ro_cat']
    tups = summary_df[cols].sort_values(cols, ascending=False).apply(tuple, 1)
    f, i = pd.factorize(tups)
    factorized = pd.Series(f + 1, tups.index)
    summary_df = summary_df.assign(Rank1=factorized)
    
    tups2 = summary_df.loc[:,['Rank1', 'otv_ks']].sort_values(['Rank1', 'otv_ks'], ascending=[True, False]).apply(tuple, 1)
    f2, i2 = pd.factorize(tups2)
    factorized2 = pd.Series(f2 + 1, tups2.index)
    summary_df = summary_df.assign(Rank2 = factorized2)
    
    summary_df['dev_itv_ks_diff_score'] = 100 - abs(summary_df['dev_itv_ks_diff'])
    summary_df['dev_otv_ks_diff_score'] = 100 - abs(summary_df['dev_otv_ks_diff'])
    summary_df['itv_otv_ks_diff_score'] = 100 - abs(summary_df['itv_otv_ks_diff'])
    summary_df['dev_ro_score'] = 100*summary_df['dev_ro_break'].fillna(11)/11
    summary_df['itv_ro_score'] = 100*summary_df['itv_ro_break'].fillna(11)/11
    summary_df['otv_ro_score'] = 100*summary_df['otv_ro_break'].fillna(11)/11
    
    summary_df['stability_score'] = (summary_df['dev_itv_ks_diff_score'] + summary_df['dev_otv_ks_diff_score'] + summary_df['itv_otv_ks_diff_score'] + summary_df['dev_ro_score'] + summary_df['itv_ro_score'] + summary_df['otv_ro_score'])/6
    summary_df['stability_weighted_otv_ks'] = summary_df['stability_score'] * summary_df['otv_ks']
    
    summary_df.sort_values('stability_weighted_otv_ks', ascending=False, inplace=True)
    summary_df.to_csv('results/summary_df_' + iter_type + '_' + identifier + '_ordered.csv', index=False)
    return summary_df



def model_selection(type, rank, version):
    if type == 'features':
        summary_df=pd.read_csv('results/summary_df_features_xgb_' + version + '_ordered.csv')
        count=int(summary_df[summary_df.Rank2== rank].iloc[0]['feature_count'])
        importance_df=pd.read_csv('results/feature_importance_'+str(count)+'_features_'+version+'.csv')
        features=list(importance_df.iloc[:,0])
        return features
    elif type == 'params':
        summary_df=pd.read_csv('results/summary_df_params_xgb_' + version + '_ordered.csv')
        params = ['n_estimators', 'max_depth', 'subsample', 'learning_rate', 'colsample_bytree', 'reg_lambda', 'reg_alpha']
        param_cols = [i for i in summary_df.columns if i in params]
        
        final_param = summary_df[summary_df.Rank2== rank].iloc[0][param_cols].to_dict()
        final_param['n_estimators'] = int(final_param['n_estimators'])
        final_param['max_depth'] = int(final_param['max_depth'])
        return final_param

