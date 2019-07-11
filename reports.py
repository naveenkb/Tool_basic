from itertools import chain, product
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from pandas import ExcelWriter
import os
from sklearn.externals import joblib
import pandas.core.algorithms as algos
import scipy.stats.stats as stats
import statsmodels.api as sm
import matplotlib
import collections
#from iterations import KS_table
matplotlib.use('agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 7})

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.style.use('seaborn')

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

    
""" Creates partial dependence plots
    
    Parameters:
    ----------
    model: fitted model object for which partial dependence plots needs to be drawn
    X: dataset with the independent variables
    features: index of features in X for which partial dependence has to be calculated
    n_cols: number of columns in the plot grid
    figsize: overall plot size in inches
"""

def plot_partial_dependence(model, X, features, n_cols=3, figsize=(10, 10)):
    fig = plt.figure(figsize=figsize)
    nrows=int(np.ceil(len(features)/float(n_cols)))
    ncols=min(n_cols, len(features))
    axs = []
    pdp = []
    for i, f_id in enumerate(features):
        X_temp = X.copy().values
        ax = fig.add_subplot(nrows, ncols, i + 1)
        
        x_scan = np.linspace(np.percentile(X_temp[:, f_id], 0.1), np.percentile(X_temp[:, f_id], 99.5), 10)
        y_partial = []
        
        for point in x_scan:
            X_temp[:, f_id] = point
            X_temp_df = pd.DataFrame(X_temp, columns=X.columns)
            pred = model.predict_proba(X_temp_df)
            y_partial.append(np.average(pred[:,1]))
        
        y_partial = np.array(y_partial)
        pdp.append((x_scan, y_partial))
        
        # Plot partial dependence
        ax.plot(x_scan, y_partial, '-', color = 'green', linewidth = 1)
        ax.set_xlim(min(x_scan)-0.1*(max(x_scan)-min(x_scan)), max(x_scan)+0.1*(max(x_scan)-min(x_scan)))
        ax.set_ylim(min(y_partial)-0.1*(max(y_partial)-min(y_partial)), max(y_partial)+0.1*(max(y_partial)-min(y_partial)))
        ax.set_xlabel(X.columns[f_id])
    axs.append(ax)
    fig.subplots_adjust(bottom=0.15, top=0.7, left=0.1, right=0.95, wspace=0.6,
                        hspace=0.3)
    fig.tight_layout()
    return fig, axs, pdp






""" Generates partial dependence plots for all variables in a dataset iteratively
    
    Parameters:
    ----------
    model: fitted model object for which partial dependence plots needs to be drawn
    X: dataset with the independent variables
    feature_names: name of all features for which PDP has to be drawn
    save_loc: save location path for the PDP images
"""

def generate_pdp(model, X, feature_names, save_loc):
    num_features = len(feature_names)
    plots_per_grid = 9
    nonflat = []
    
    for i in range(int(np.ceil(num_features/plots_per_grid))):
        features = range(plots_per_grid*i, min(plots_per_grid*(i+1), num_features))
        fig, axs, pdp = plot_partial_dependence(model, X[feature_names], features, figsize=(10, 10))
        fig.savefig(os.path.join(save_loc, str(i)+'.png'), dpi = 200)
        
        for f, p in zip (features, pdp):
            if max(p[1]) - min(p[1]) > 0:
                nonflat.append(feature_names[f])
    return nonflat 

""" Reduces variables iteratively until no flat PDP is found
    
    Parameters:
    ----------
    devset: training set including response variable
    valsets: tuple of test sets including response variable
    valnames: names of validation sets, used in naming convention of result files
    y: name of response column
    feature_names: names of initial features from which reduction should start
    params: parameters for the model
    version: identifier of the iteration version
    save_loc: save location path for the PDP images
"""

def pdp_var_reduction(devset, valsets, valnames, y, feature_names, params, version, save_loc):
    num_flat = len(feature_names)
    nonflat = feature_names
    summary_df_pdp = pd.DataFrame()
    dct= collections.OrderedDict(params)
    identifier='_'.join([list(dct.keys())[i] + str(list(dct.values())[i]) for i in range(len(dct.keys()))])
    X_train = devset
    
    while num_flat > 0:
        summary_df =  pd.DataFrame()
        curr_X = X_train[nonflat]
        target = X_train[y]
        model = XGBClassifier(seed=10, **params, nthread=10)
        model.fit(curr_X, target)
        joblib.dump(model, 'saved_objects/xgb_nonflat_pdp_'+ version + '_' + identifier + '_' + str(len(nonflat)) + '.joblib', compress = 1)
        feature_imp = pd.DataFrame({'feature_names': nonflat, 'importance': model.feature_importances_})
        feature_imp.to_csv('results/feature_importance_nonflat_pdp_'+ version + '_'  + identifier + '_' + str(len(nonflat)) + '.csv', index=False)
        score = model.predict_proba(curr_X)
        ks = KS_table(score[:,1], target, 'dev' + '_xgb_nonflat_pdp_' + version + '_'  + identifier + '_'+ str(len(nonflat)))
        breaks = np.diff(ks['No.Res']) > 0
        dec_break = (np.diff(ks['No.Res']) > 0).any()
        ks_val = ks.KS.max()
        ks_decile = ks.KS.idxmax() + 1
        capture = ks['percent_cum_res'][3]
        if dec_break:
            break_dec = min([idx for idx, x in enumerate(breaks) if x]) + 2
            summary_df = summary_df.append(pd.DataFrame([[len(nonflat), ks_val, break_dec, ks_decile, capture]],columns=['feature_count', 'dev_ks', 'dev_ro_break', 'dev_ks_decile', 'dev_capture']))
        else:
            break_dec = np.nan
            summary_df = summary_df.append(pd.DataFrame([[len(nonflat), ks_val, break_dec, ks_decile, capture]],columns=['feature_count', 'dev_ks', 'dev_ro_break', 'dev_ks_decile', 'dev_capture']))
        
        for X_test, dset in zip(valsets, valnames):
            summary_df_test = pd.DataFrame()
            curr_X = X_test[nonflat]
            target = X_test[y]
            score = model.predict_proba(curr_X)
            ks = KS_table(score[:,1], target, dset + '_xgb_nonflat_pdp_' + version + '_'  + identifier + '_' + str(len(nonflat)))
            breaks = np.diff(ks['No.Res']) > 0
            dec_break = (np.diff(ks['No.Res']) > 0).any()
            ks_val = ks.KS.max()
            ks_decile = ks.KS.idxmax() + 1
            capture = ks['percent_cum_res'][3]
            if dec_break:
                break_dec = min([idx for idx, x in enumerate(breaks) if x]) + 2
                summary_df_test = summary_df_test.append(pd.DataFrame([[len(nonflat), ks_val, break_dec, ks_decile, capture]],columns=['feature_count', dset+ '_ks', dset +'_ro_break', dset +'_ks_decile', dset + '_capture']))
            else:
                break_dec = np.nan
                summary_df_test = summary_df_test.append(pd.DataFrame([[len(nonflat), ks_val, break_dec, ks_decile, capture]],columns=['feature_count', dset+ '_ks', dset +'_ro_break', dset +'_ks_decile', dset + '_capture']))
            
            summary_df_test.reset_index(drop=True, inplace =True)
            summary_df[dset + '_ks'] = summary_df_test[dset + '_ks']
            summary_df[dset +'_ro_break'] = summary_df_test[dset +'_ro_break']
            summary_df[dset + '_ks_decile'] = summary_df_test[dset + '_ks_decile']
            summary_df[dset + '_capture'] = summary_df_test[dset + '_capture']
            summary_df['dev_' + dset + '_ks_diff'] = (summary_df['dev_ks'] - summary_df[dset + '_ks'])*100/summary_df['dev_ks']
            
        summary_df_pdp = summary_df_pdp.append(summary_df)
    
        nonflat_prev = nonflat
        if not os.path.exists('PDP/' + version + '_'  + identifier + '_' + str(len(nonflat))):
            os.makedirs('PDP/' + version + '_'  + identifier + '_' + str(len(nonflat)))
        nonflat = generate_pdp(model, X_train, nonflat, os.path.join(save_loc, 'PDP/' + version + '_'  + identifier + '_' + str(len(nonflat))))
        num_flat = len(set(nonflat_prev)-set(nonflat))
    
    summary_df_pdp.to_csv('results/summary_df_nonflat_pdp_xgb_' + version + '_'  + identifier + '.csv', index=False)
    return nonflat




""" Creates Final Report
    
    Parameters:
    ----------
    dct: Dictionary with key as XGBoost parameters and value as tuned parameter value
	
    valsets: tuple of test sets including response variable
    valnames: names of validation sets, used in naming convention of result files
    y: name of response column
    feature_names: names of initial features from which reduction should start
    params: parameters for the model
    version: identifier of the iteration version
    save_loc: save location path for the PDP images
"""

def final_report(dct, features, version, X_otv):
    dct= collections.OrderedDict(dct)
    identifier='_'.join([list(dct.keys())[i] + str(list(dct.values())[i]) for i in range(len(dct.keys()))])
    gbm = joblib.load('saved_objects/xgb_nonflat_pdp_' + version + '_' + identifier + '_' + str(len(features)) + '.joblib')
    
    feature_summary = pd.read_csv('results/summary_df_features_xgb_' + version + '.csv')
    param_summary = pd.read_csv('results/summary_df_params_xgb_' + version + '.csv')
    pdp_summary = pd.read_csv('results/summary_df_nonflat_pdp_xgb_' + version + '_' + identifier +'.csv')
    writer = ExcelWriter('xgb_final_reports_'+ version +'.xlsx')
    
    feature_summary.to_excel(writer,'feature_summary', index =False)
    param_summary.to_excel(writer,'parameter_summary', index =False)
    pdp_summary.to_excel(writer,'pdp_summary', index =False)
    
    feature_names = pd.read_csv('results/feature_importance_nonflat_pdp_' + version + '_' + identifier + '_' + str(len(features))  + '.csv')
    feature_names.to_excel(writer,'feature_importance', index =False)
    feature_names = feature_names.feature_names.tolist()
    
    dev_ks = pd.read_csv('results/KS_table_dev_xgb_nonflat_pdp_' + version + '_' + identifier + '_' + str(len(features))  + '.csv')
    itv_ks = pd.read_csv('results/KS_table_itv_xgb_nonflat_pdp_' + version + '_' + identifier + '_' + str(len(features))  + '.csv')
    otv_ks = pd.read_csv('results/KS_table_otv_xgb_nonflat_pdp_' + version + '_' + identifier + '_' + str(len(features))  + '.csv')
    dev_ks.to_excel(writer,'dev_ks', index =False)
    itv_ks.to_excel(writer,'itv_ks', index =False)
    otv_ks.to_excel(writer,'otv_ks', index =False)
    
    otv = X_otv.loc[:,feature_names]
    
    score = gbm.predict_proba(otv)
    
    otv['score'] = score[:,1]
    
    otv['dev_grps'] = np.where(otv.score > dev_ks.min_pred[0], 0, np.where(otv.score > dev_ks.min_pred[1], 1, np.where(otv.score > dev_ks.min_pred[2], 2, np.where(otv.score > dev_ks.min_pred[3], 3, np.where(otv.score > dev_ks.min_pred[4], 4, np.where(otv.score > dev_ks.min_pred[5], 5, np.where(otv.score > dev_ks.min_pred[6], 6, np.where(otv.score > dev_ks.min_pred[7], 7, np.where(otv.score > dev_ks.min_pred[8], 8, np.where(otv.score <= dev_ks.min_pred[8], 9,-99))))))))))
    
    grouped = otv.groupby('dev_grps')
    agg = pd.DataFrame({'Total_Obs': grouped.count().score})
    agg['percent_obs'] = agg['Total_Obs']*100/agg['Total_Obs'].sum()
    
    agg['psi'] = (0.1 - agg['percent_obs']/100) * np.log(0.1/(agg['percent_obs']/100))
    
    print('PSI value is {}'.format(agg['psi'].sum()))
    #print('PSI table is as follows \n {}'.format(agg['psi']))
    agg.to_excel(writer,'PSI', index =False)
    writer.save()
