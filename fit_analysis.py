#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:31:18 2023

Analysis of models fitted to behaviour.

@author: sascha
"""

"----- Open Files"
# import sys
# sys.modules[__name__].__dict__.clear() # Clear variables
# del model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import utils
import analysis_tools as anal
import torch
import pickle
import arviz as az

from sklearn.linear_model import LinearRegression
import scipy
import itertools

out = utils.get_data_from_file()
post_sample_df, expdata_df, loss, params_df, num_params, sociopsy_df, elbo_tuple = out

expdata_df_pub = pickle.load(open("behav_data/preproc_data_old_published_all.p", "rb" ))[1]

# if len(out) == 7:
#     post_sample_df, expdata_df, loss, params_df, num_params, sociopsy_df, elbo_tuple = out

# elif len(out) == 6:
#     post_sample_df, expdata_df, loss, params_df, num_params, sociopsy_df = out

# else:
#     raise Exception("Output length invalid.")

param_names = params_df.iloc[:, 0:-3].columns
if 'ID' in expdata_df.columns:
    "For compatibility with param_recov data"
    ID_df = expdata_df.loc[:, ['ID', 'ag_idx', 'handedness']].drop_duplicates()
    "Add IDs"
    post_sample_df = pd.merge(post_sample_df, ID_df, on = 'ag_idx')
    
assert len(param_names) == num_params

if 'ID_x' in post_sample_df.columns:
    post_sample_df=post_sample_df.drop(['ID_x'],axis=1)
    post_sample_df=post_sample_df.rename(columns={'ID_y': 'ID'})

if 'ID' in post_sample_df:
    inf_mean_df = pd.DataFrame(post_sample_df.groupby(['model', 
                                                       'ag_idx', 
                                                       'group', 
                                                       'ID', 
                                                       'handedness'], as_index = False).mean())

else:
    inf_mean_df = pd.DataFrame(post_sample_df.groupby(['model', 
                                                       'ag_idx', 
                                                       'group'], as_index = False).mean())    

inf_mean_df = inf_mean_df.sort_values(by=['ag_idx'])

model = post_sample_df['model'][0]
num_agents = len(post_sample_df['ag_idx'].unique())

print(f"Model fit of model {model} for {num_agents} agents after %d inference steps."%len(loss))
'''
Plot ELBO
'''
fig, ax = plt.subplots()
plt.plot(loss[-2000:])
plt.title(f"ELBO for model {model} ({num_agents} agents)")
ax.set_xlabel("Number of iterations")
ax.set_ylabel("ELBO")
plt.show()
print(np.array(loss[-1000:]).mean())

'''
Plot Parameter Distributions
'''
num_plot_cols = 3
num_plot_rows = int((num_params <= num_plot_cols) * 1 + \
                (num_params > num_plot_cols) * np.ceil(num_params / num_plot_cols))

fig = plt.figure(figsize=(16,12), dpi=100)
gs = fig.add_gridspec(num_plot_rows, num_plot_cols, hspace=0.2, wspace = 0.2)
ax = gs.subplots()
# params_df.columns
for param_idx in range(num_params):
    plot_col_idx = param_idx % num_plot_cols
    plot_row_idx = (param_idx // num_plot_cols)
    
    if num_params > 3:
        ax_idxs = [plot_row_idx, plot_col_idx]
        
    else:
        ax_idxs = [plot_col_idx]
    
    ca = sns.kdeplot(inf_mean_df[post_sample_df.columns[param_idx]], ax = ax[*ax_idxs])
    ax[*ax_idxs].set_xlabel(post_sample_df.columns[param_idx])    

    if plot_col_idx > 0:
        ax[*ax_idxs].set_ylabel(None)

    if plot_row_idx > 0:
        ax[*ax_idxs].get_position().y0 += 10

    # ca.plot([0,0], ca.get_ylim())

plt.show()

'''
    Violin Plot
'''
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# plt.style.use("seaborn-v0_8-dark")

anal.violin(inf_mean_df.loc[:, [*param_names]], model)

complete_df = utils.create_complete_df(inf_mean_df, sociopsy_df, expdata_df, post_sample_df, param_names)

if 'ID' in inf_mean_df.columns:
    anal.violin(complete_df.loc[:, ['age', 'ER_stt', 'ER_dtt', 'RT', 'points_total']], model = 'sociopsy')
    
else:
    anal.violin(complete_df.loc[:, ['ER_stt', 'ER_dtt', 'points_total']], model = 'errors')

'''
    Kdpelots
'''
fig, ax = plt.subplots(len(param_names), 1, figsize=(10,35))
if model == 'Conflict':
    xlims = [[0, 0.08],
             [0, 9],
             [0, 3.0],
             [-2.5, 3],
             [0, 0.08],
            [0, 9],
            [0, 3.0],
            [-2.5, 3]]
    
for param_idx in range(len(param_names)):
    sns.kdeplot(post_sample_df[param_names[param_idx]], ax = ax[param_idx])
    ax[param_idx].set_xlabel(param_names[param_idx])
    ax[param_idx].set_xlim(xlims[param_idx])

plt.show()

#%%
'''
    Sort by counterbalancing group
'''
# complete_df_grp0 = complete_df[complete_df['group'] == 0]
# complete_df_grp0 = complete_df[complete_df['group'] == 1]
# complete_df_grp0 = complete_df[complete_df['group'] == 2]
# complete_df_grp0 = complete_df[complete_df['group'] == 3]

for group in range(4):
    anal.violin(inf_mean_df[inf_mean_df['group'] == group].loc[:, [*param_names]], model)

num_comparisons = 0
num_diffs = 0
for group1 in range(4):
    for group2 in range(group1+1, 4):
        for param in [*param_names]:
            t,p = scipy.stats.ttest_ind(inf_mean_df[inf_mean_df['group'] == group1][param], inf_mean_df[inf_mean_df['group'] == group2][param])
            num_comparisons += 1
            if p < 0.1: 
                if p < 0.05:
                    num_diffs += 1 
                print(f"\n param {param} for groups {group1} and {group2}")
                print(t)
                print(p)

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm                
df = complete_df.loc[:, ['group', *param_names]].copy()

# Fit the 2-way ANOVA model
for param in [*param_names]:
    
    if 'day1' in param:
    
        anova_df = pd.DataFrame()
        day = [1]*len(df)
        day.extend([2]*len(df))
        anova_df['day'] = day
        
        group = list(df['group'])
        group.extend(list(df['group']))
        anova_df['group'] = group
        
        anova_df['group'] = anova_df['group'].map(lambda x: 0 if (x == 0. or x == 2.) else 1)
        # anova_df['group'] = anova_df['group'].map(lambda x: 1 if group == 2 or group == 3 else x)
        
        param_val = list(df[param[0:-5] + '_day1'])
        param_val.extend(list(df[param[0:-5] + '_day2']))
        anova_df[param[0:-5]] = param_val
        
        print(f"\n for param {param[0:-5]}")
        model = ols(f'{param[0:-5]} ~ group * day', data=anova_df).fit()
    
        # Perform ANOVA and print the results
        anova_table = anova_lm(model, typ=2)
        print(anova_table)

#%%
'''
    Check for how many participants parameters are 0:
'''
threshold = 0
BF = 3.2
inf_mean_df_nozero = inf_mean_df.copy()
"Bayesian Approach"
for param in post_sample_df.columns:
    if param not in ['group', 'model', 'ag_idx', 'ID', 'handedness']:
    # if param != 'group' and param != 'model' and param != 'ag_idx' and param != 'ID' and param !=:
        print("----- Testing parameter %s"%param)
        for ag_idx in post_sample_df['ag_idx'].unique():
            lt0 = (np.array(post_sample_df[post_sample_df['ag_idx'] == ag_idx][param]) <= 0).sum()
            gt0 = (np.array(post_sample_df[post_sample_df['ag_idx'] == ag_idx][param]) > 0).sum()
            
            if gt0/lt0 <= BF and lt0/gt0 <= BF:
                print(f"No evidence that {param} is different from 0 for agent {ag_idx}")
                inf_mean_df_nozero[inf_mean_df_nozero['ag_idx'] == ag_idx] = np.nan

anal.violin(inf_mean_df_nozero.loc[:, [*param_names]], model)

#%%
'''
    Points Analysis
'''

"Predictors x"
x = np.stack((np.array(complete_df['theta_Q_day1']), 
              np.array(complete_df['theta_rep_day1'])), axis=1)

"Data y"
y = np.array(complete_df['points_day1']).reshape(-1,1)
linmodel = LinearRegression()
linmodel.fit(x, y)
print(f"slope: {linmodel.coef_}\n")

"Predictors x"
x = np.stack((np.array(complete_df['theta_Q_day2']), 
              np.array(complete_df['theta_rep_day2'])), axis=1)


"Data y"
y = np.array(complete_df['points_day2']).reshape(-1,1)
linmodel = LinearRegression()
linmodel.fit(x, y)
print(f"slope: {linmodel.coef_}\n")


#%%
'''
    Differences day 1 & day 2
'''

inf_mean_df['Q/R_day1'] = inf_mean_df.apply(lambda row: row['theta_Q_day1']/row['theta_rep_day1'], axis = 1)
inf_mean_df['Q/R_day2'] = inf_mean_df.apply(lambda row: row['theta_Q_day2']/row['theta_rep_day2'], axis = 1)

post_sample_df['Q/R_day1'] = post_sample_df.apply(lambda row: row['theta_Q_day1']/row['theta_rep_day1'], axis = 1)
post_sample_df['Q/R_day2'] = post_sample_df.apply(lambda row: row['theta_Q_day2']/row['theta_rep_day2'], axis = 1)

diffs_df = anal.daydiff(inf_mean_df)
diffs_df = pd.merge(diffs_df, sociopsy_df[sociopsy_df['ID'].isin(diffs_df['ID'])], on = 'ID')

# anal.violin(inf_mean_df, model)

diffs_df = anal.daydiff(post_sample_df, BF = 1)

anal.daydiff(post_sample_df[post_sample_df['group']==0], BF = 3.2)
anal.daydiff(post_sample_df[post_sample_df['group']==1], BF = 3.2)
anal.daydiff(post_sample_df[post_sample_df['group']==2], BF = 3.2)
anal.daydiff(post_sample_df[post_sample_df['group']==3], BF = 3.2)

anal.daydiff(post_sample_df[(post_sample_df['group']==0) | (post_sample_df['group']==1)], BF = 3.2)
anal.daydiff(post_sample_df[(post_sample_df['group']==2) | (post_sample_df['group']==3)], BF = 3.2)

anal.daydiff(post_sample_df[(post_sample_df['group']==0) | (post_sample_df['group']==2)], BF = 3.2)
anal.daydiff(post_sample_df[(post_sample_df['group']==1) | (post_sample_df['group']==3)], BF = 3.2)

#%%
'''
    PCA Analysis
'''

"----- On absolute values"
pca_parameters = ['ag_idx', 'age',
                  'theta_Q_day1', 'theta_Q_day2', 
                  'theta_rep_day1', 'theta_rep_day2',
                  'ER_total_day1', 'ER_total_day2',
                  'RT_day1', 'RT_day2']

pcomps = anal.perform_PCA(complete_df.loc[:, [*pca_parameters]], 4, correctfor='age')

"----- On within-subject correlations"
corr_df = anal.within_subject_corr(post_sample_df, ['theta_Q_day1', 'theta_Q_day2', 
                                                    'theta_rep_day1', 'theta_rep_day2'])

corr_df = pd.merge(corr_df, complete_df.loc[:, ['ID', 'age']], on ='ID')

anal.perform_PCA(corr_df.drop(['ID'], axis=1), 1, correctfor = 'age')

new_pca_df = pd.merge(corr_df, complete_df.loc[:, ['ID', *pca_parameters]], on='ID')

new_pca_df = new_pca_df.rename(columns={'age_x':'age'})
anal.perform_PCA(new_pca_df.drop(['ID', 'age_y'], axis=1), 9, correctfor = 'age')

#%%
'''
    PCA for points
'''

"Day 1"
print("Day 1")

"----- On absolute values"
pca_parameters = ['ag_idx',
                  'theta_Q_day1',
                  'theta_rep_day1',
                  'conflict_param_day1']

pcomps = anal.perform_PCA(complete_df.loc[:, [*pca_parameters]], 3)

complete_df['pcomps'] = pcomps[:, 0]
complete_df_temp = complete_df.sort_values(by='pcomps', inplace = False)

t,p = scipy.stats.ttest_ind(complete_df_temp['points_day1'][0:30], complete_df_temp['points_day1'][30:])
print(t)
print(p)

"Day 2"
print("\nDay 2")

"----- On absolute values"
pca_parameters = ['ag_idx',
                  'age',
                  'theta_Q_day2',
                  'theta_rep_day2',
                  'conflict_param_day2']

pcomps = anal.perform_PCA(complete_df.loc[:, [*pca_parameters]], 3)

complete_df['pcomps'] = pcomps[:, 0]
complete_df_temp = complete_df.sort_values(by='pcomps', inplace = False)

t,p = scipy.stats.ttest_ind(complete_df_temp['points_day2'][0:20], complete_df_temp['points_day2'][40:])
print(t)
print(p)

#%%
'''
    Theta Q_day1
'''

fig, ax = plt.subplots()
ax.scatter(complete_df['theta_Q_day1'], complete_df['theta_Q_day2-theta_Q_day1'])
ax.axhline(0, color='k')
ax.axvline(0, color='k')
ax.set_xlabel('theta_Q_day1')
ax.set_ylabel('theta_Q_day2-theta_Q_day1')
# plt.grid()
# plt.scatter(kmeans.cluster_centers_[0, 0], kmeans.cluster_centers_[0, 1], color='red')
# plt.scatter(kmeans.cluster_centers_[1, 0], kmeans.cluster_centers_[1, 1], color='red')
# plt.title('Delta R vs Delta Q')
plt.show()

r,p = scipy.stats.pearsonr(complete_df['theta_Q_day1'], complete_df['theta_Q_day2-theta_Q_day1'])
print(f'r=%.4f, p=%.4f'%(r,p))

if p < 0.05:
    x = np.array(complete_df['theta_Q_day1']).reshape(-1,1)
    y = np.array(complete_df['theta_Q_day2-theta_Q_day1']).reshape(-1,1)
    linmodel = LinearRegression()
    linmodel.fit(x, y)
    print(f"slope: {linmodel.coef_}\n")

from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
x1 = np.array(complete_df['theta_Q_day1'])
x2 = np.array(complete_df['theta_Q_day2-theta_Q_day1'])
principalComponents = pca.fit_transform(np.stack((x1,x2),axis=1))

x = np.array(complete_df[param_x]).reshape(-1,1)
y = np.array(complete_df[param_y]).reshape(-1,1)
linmodel = LinearRegression()
linmodel.fit(x1.reshape(-1,1), x2.reshape(-1,1))
print(f"slope: {linmodel.coef_}\n")

#%%

'''
    PCA on θ_Q_day1, θ_Q_day2, Δθ_Q, and Δθ_rep
'''

print("Need to normalize!!!")
pca = PCA(n_components = 1)
x1 = np.array(complete_df['theta_Q_day1'])
x2 = np.array(complete_df['theta_rep_day1'])
x3 = np.array(complete_df['theta_Q_day2-theta_Q_day1'])
x4 = np.array(complete_df['theta_rep_day2-theta_rep_day1'])
principalComponents = pca.fit_transform(np.stack((x1,x2,x3,x4),axis=1))

pca_df = pd.DataFrame(data={'ag_idx': range(num_agents), 'PCA value': principalComponents[:,0]})
# pca_df = pca_df.sort_values(by='PCA value')
pca_0_df = pca_df[pca_df['PCA value'] < 0]
pca_1_df = pca_df[pca_df['PCA value'] >= 0]


fig, ax = plt.subplots()
df = complete_df[complete_df['ag_idx'].isin(pca_0_df['ag_idx'])]
ax.scatter(df['theta_Q_day1'], df['theta_Q_day2-theta_Q_day1'], color='red')
df = complete_df[complete_df['ag_idx'].isin(pca_1_df['ag_idx'])]
ax.scatter(df['theta_Q_day1'], df['theta_Q_day2-theta_Q_day1'], color='blue')
ax.axhline(0, color='k')
ax.axvline(0, color='k')
ax.set_xlabel('theta_Q_day1')
ax.set_ylabel('theta_Q_day2-theta_Q_day1')
# plt.grid()
# plt.scatter(kmeans.cluster_centers_[0, 0], kmeans.cluster_centers_[0, 1], color='red')
# plt.scatter(kmeans.cluster_centers_[1, 0], kmeans.cluster_centers_[1, 1], color='red')
# plt.title('Delta R vs Delta Q')
plt.show()

fig, ax = plt.subplots()
df = complete_df[complete_df['ag_idx'].isin(pca_0_df['ag_idx'])]
ax.scatter(df['theta_rep_day1'], df['theta_rep_day2-theta_rep_day1'], color='red')
df = complete_df[complete_df['ag_idx'].isin(pca_1_df['ag_idx'])]
ax.scatter(df['theta_rep_day1'], df['theta_rep_day2-theta_rep_day1'], color='blue')
ax.axhline(0, color='k')
ax.axvline(0, color='k')
ax.set_xlabel('theta_rep_day1')
ax.set_ylabel('theta_rep_day2-theta_rep_day1')
# plt.grid()
# plt.scatter(kmeans.cluster_centers_[0, 0], kmeans.cluster_centers_[0, 1], color='red')
# plt.scatter(kmeans.cluster_centers_[1, 0], kmeans.cluster_centers_[1, 1], color='red')
# plt.title('Delta R vs Delta Q')
plt.show()

#%%

anal.perform_PCA(complete_df.loc[:, ['ag_idx', 'theta_Q_day1', 'theta_rep_day1',
                            'theta_Q_day2-theta_Q_day1',
                            'theta_rep_day2-theta_rep_day1']], num_components=1)

#%%
"Cluster Analysis based on absolute values"

normalized_mean_df = pd.DataFrame()
for param in [*param_names, 'age', 'RT', 'ER_dtt', 'ER_stt']:
    normalized_mean_df[param] = (complete_df.loc[:, param] - complete_df.loc[:, param].mean())/complete_df.loc[:, param].std()

kmeans, cluster_groups_bothdays, _ = anal.kmeans(complete_df.loc[:, 
                                                [*param_names, 'age', 'RT', 'ER_dtt', 'ER_stt']],
                                                 inf_mean_df, 
                                                 n_clusters = 2,
                                                 num_reps = 100)

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(cluster_groups_bothdays[0])],
                      expdata_df[expdata_df['ag_idx'].isin(cluster_groups_bothdays[1])])

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(cluster_groups_bothdays[2])],
                      expdata_df[expdata_df['ag_idx'].isin(cluster_groups_bothdays[3])])

#%%
"Cluster Analysis based on Daydifferences"

kmeans, cluster_groups_bothdays, _ = anal.kmeans(inf_mean_df.loc[:, 
                                                          ['ID', 'theta_Q_day2', 'theta_Q_day1', 
                                                          'theta_rep_day2', 'theta_rep_day1']], 
                                         inf_mean_df, 
                                         n_clusters = 2,
                                         num_reps = 100)

fig, ax = plt.subplots()
ax.scatter(diffs_df['theta_Q_day2-theta_Q_day1'], diffs_df['theta_rep_day2-theta_rep_day1'], color='blue')
ax.axhline(0, color='k')
ax.axvline(0, color='k')
plt.grid()
plt.scatter(kmeans.cluster_centers_[0, 0], kmeans.cluster_centers_[0, 1], color='red')
plt.scatter(kmeans.cluster_centers_[1, 0], kmeans.cluster_centers_[1, 1], color='red')
plt.title('Delta R vs Delta Q')
plt.show()

fig, ax = plt.subplots()
ax.scatter(diffs_df['theta_Q_day2-theta_Q_day1'], diffs_df['Q/R_day2-Q/R_day1'], color='blue')
ax.axhline(0, color='k')
ax.axvline(0, color='k')
plt.scatter(kmeans.cluster_centers_[0, 0], kmeans.cluster_centers_[0, 2], color='red')
plt.scatter(kmeans.cluster_centers_[1, 0], kmeans.cluster_centers_[1, 2], color='red')
plt.grid()
plt.title('Delta Q/R vs Delta Q')
plt.show()

fig, ax = plt.subplots()
ax.scatter(diffs_df['theta_rep_day2-theta_rep_day1'], diffs_df['Q/R_day2-Q/R_day1'], color='blue')
ax.axhline(0, color='k')
ax.axvline(0, color='k')
plt.scatter(kmeans.cluster_centers_[0, 1], kmeans.cluster_centers_[0, 2], color='red')
plt.scatter(kmeans.cluster_centers_[1, 1], kmeans.cluster_centers_[1, 2], color='red')
plt.grid()
plt.title('Delta Q/R vs Delta R')
plt.show()

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(cluster_groups_bothdays[0])],
                      expdata_df[expdata_df['ag_idx'].isin(cluster_groups_bothdays[1])])

#%%
'''
    Correlations of Daydiffs with age
'''
import scipy

num_params = len(diffs_df.columns) - 5
for param_idx in range(num_params):
    param = diffs_df.columns[param_idx]
    print(param)
    r,p = scipy.stats.pearsonr(diffs_df['age'], diffs_df[param])
    print(r)
    print(p)

#%%
'''
    Correlations between subjects
'''
import analysis_tools as anal
anal.param_corr(inf_mean_df)

#%%
'''
    Correlations within subjects
'''
corr_df = anal.within_subject_corr(post_sample_df, param_names)

# corr_dict_errors = corr_dict.copy()
# corr_dict_errors['errors_stt'] = errorrates[0, :]
# corr_dict_errors['errors_dtt'] = errorrates[1, :]

#%%
'''
Kdeplots of within-subject  correlations
'''
# for key in corr_dict.keys():
#     sns.kdeplot(corr_dict[key])
#     plt.title(key)
#     plt.show()

#%%
'''
Correlation analysis_tools both days
'''
leavnodes = anal.cluster_analysis(corr_df, title = 'all correlations exp')
kmeans, cluster_groups_bothdays, _ = anal.kmeans(corr_df, 
                                         inf_mean_df, 
                                         n_clusters = 2,
                                         num_reps = 100)

# utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(cluster_groups_bothdays[0])],
#                       expdata_df[expdata_df['ag_idx'].isin(cluster_groups_bothdays[1])])

# utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(leavnodes[13:21])],
#                       expdata_df[expdata_df['ag_idx'].isin(leavnodes[21:])])

# utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(leavnodes[0:3])],
#                       expdata_df[expdata_df['ag_idx'].isin(leavnodes[21:])])

#%%
'''
    Correlation analysis day 1
'''
droplist_for_day1 = [param for param in post_sample_df.columns if 'day2'  in param]
post_sample_df_day1 = post_sample_df.drop(droplist_for_day1, axis = 1)
corr_dict_day1 = anal.within_subject_corr(post_sample_df_day1)

leavnodes_day1 = anal.cluster_analysis(corr_dict_day1, title='day 1 exp')

# inf_mean_df[inf_mean_df['ag_idx'].isin(leavnodes[0:16])]['group']

# utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(leavnodes_day1[0:7])],
#                       expdata_df[expdata_df['ag_idx'].isin(leavnodes_day1[6:16])])

# utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(leavnodes_day1[0:16])],
#                       expdata_df[expdata_df['ag_idx'].isin(leavnodes_day1[16:])])

'''
K-means clustering for Day 1
'''
labels, cluster_groups_day1, _ = anal.kmeans(corr_dict_day1, 
                                          inf_mean_df, 
                                          n_clusters = 2,
                                          num_reps = 100)

# anal.compare_leavnodes(grp1_agidx, leavnodes_day1[0:16])

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(cluster_groups_day1[0])],
                      expdata_df[expdata_df['ag_idx'].isin(cluster_groups_day1[1])], )

#%%
'''
    kmeans with errors Day 1
'''
corr_dict_day1_errors = corr_dict_day1.copy()
corr_dict_day1_errors['errors_stt_day1'] = errorrates_day1[0, :]
corr_dict_day1_errors['errors_dtt_day1'] = errorrates_day1[1, :]

leavnodes_day1_errors = anal.cluster_analysis_tools(corr_dict_day1_errors, title='day 1 exp errors')

labels, cluster_groups_day1_errors = anal.kmeans(corr_dict_day1_errors, inf_mean_df, n_clusters = 2)

anal.compare_lists(cluster_groups_day1_errors[0], cluster_groups_day1[0])
#%%
'''
    Correlation analysis day 2
'''
droplist_for_day2 = [param for param in post_sample_df.columns if 'day1'  in param]
post_sample_df_day2 = post_sample_df.drop(droplist_for_day2, axis = 1)
corr_dict_day2 = anal.within_subject_corr(post_sample_df_day2)

leavnodes_day2 = anal.cluster_analysis(corr_dict_day2, title = 'day 2')

labels, cluster_groups_day2, c_distances = anal.kmeans(corr_dict_day2, 
                                     inf_mean_df, 
                                     n_clusters = 2,
                                     num_reps = 100)

# anal.compare_leavnodes(grp0_agidx, leavnodes_day1[0:16])

# utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(grp0_agidx)],
#                       expdata_df[expdata_df['ag_idx'].isin(grp1_agidx)])

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(cluster_groups_day2[0])],
                      expdata_df[expdata_df['ag_idx'].isin(cluster_groups_day2[1])],
                      day = 2)

'''
    kmeans with errors Day 2
'''
post_sample_df_day2 = post_sample_df.drop(['lr_day2', 'theta_Q_day2', 'theta_rep_day2'], axis = 1)
corr_dict_day2 = anal.within_subject_corr(post_sample_df_day2)

corr_dict_day2_errors = corr_dict_day2.copy()
corr_dict_day2_errors['errors_stt_day2'] = errorrates_day2[0, :]
corr_dict_day2_errors['errors_dtt_day2'] = errorrates_day2[1, :]

leavnodes_day2 = anal.cluster_analysis_tools(corr_dict_day2_errors, title = 'day 2')

labels, cluster_groups = anal.kmeans(corr_dict_day2_errors, inf_mean_df, n_clusters = 2)

#%%
'''
    Perform correlation analysis_tools only between days
'''
corr_dict = anal.within_subject_corr(post_sample_df)
corr_dict_day_between = corr_dict.copy()

dropkeys = []
for key in corr_dict_day_between.keys():
    if 'day1' in key and 'day2' not in key:
        dropkeys.append(key)
        
    elif 'day2' in key and 'day1' not in key:
        dropkeys.append(key)

for key in dropkeys:
    del corr_dict_day_between[key]

leavnodes_betweendays = anal.cluster_analysis(corr_dict_day_between, title = 'between days')

labels, cluster_groups_between, _ = anal.kmeans(corr_dict_day_between, 
                                             inf_mean_df, 
                                             n_clusters = 2,
                                             num_reps = 100)

# anal.compare_leavnodes(grp0_agidx, leavnodes_day1[0:16])

# utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(grp0_agidx)],
#                       expdata_df[expdata_df['ag_idx'].isin(grp1_agidx)])

# utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(cluster_groups_between[0])],
#                       expdata_df[expdata_df['ag_idx'].isin(cluster_groups_between[1])])

# anal.compare_lists(cluster_groups_between[1], cluster_groups_between[1])

#%%

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(cluster_groups_between[0])],
                      expdata_df[expdata_df['ag_idx'].isin(cluster_groups_between[1])])

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(cluster_groups_day1[1])], day = 2)


#%%
'''
Plot Experimental Data
'''
utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'] == 35], plot_single = False)

#%%
'''
Simulate only from means
'''
groupdata_dict, group_behav_df, params_sim, params_sim_df = utils.simulate_data(model, 
                                                                        num_agents,
                                                                        group = list(inf_mean_df['group']),
                                                                        params = inf_mean_df)

utils.plot_grouplevel(expdata_df, group_behav_df, plot_single = False)
# utils.plot_grouplevel(expdata_df, plot_single = True)


#%%
'''
Posterior Predictives
'''
complete_df = utils.posterior_predictives(post_sample_df, exp_data = expdata_df)

#%%
'''
ELBO Barplots
'''

# Assuming values is your list of values
values = [46900, 46550, 46230, 46870, 47880, 46830]

# Create a list of colors
colors = ['b', 'g', 'r', 'c', 'm']

# Create an array for the x values
models = ['Base', 'Hand', 'Seq Boost', 'Incongr.', 'Q_init', 'Habitual Tend']

# Create the bar plot
plt.bar(models, values, color=colors)
plt.ylim([46000, 48000])
# Show the plot
plt.title('ELBOs (48 agents)')
plt.show()

#%%
'''
    Clustering in Parameter Space Both days
'''
df = inf_mean_df.drop(['model', 'ag_idx', 'group', 'labels', 'lr_day1', 'lr_day2'], axis = 1)
normalized_df = (df-df.mean())/df.std()

kmeans, cluster_groups_between, _ = anal.kmeans(normalized_df.to_dict(orient='list'), 
                                             inf_mean_df, 
                                             n_clusters = 2,
                                             num_reps = 1,
                                             title = 'both days')
newdf = inf_mean_df
newdf['labels'] = kmeans.labels_

newdf0 = newdf[newdf['labels'] == 0]
newdf1 = newdf[newdf['labels'] == 1]

_, group_behav_df0, _, _ = utils.simulate_data('B', 
                                                len(newdf0),
                                                group = list(newdf0['group']),
                                                params = newdf1)

_, group_behav_df1, _, _ = utils.simulate_data('B', 
                                                len(newdf0),
                                                group = list(newdf1['group']),
                                                params = newdf1)

utils.plot_grouplevel(group_behav_df0, group_behav_df1, plot_single = False)

'''
From experimental data
'''
expdf = expdata_df.copy()
expdf['labels'] = kmeans.labels_

utils.plot_grouplevel(expdf[expdf['labels'] == 0], expdf[expdf['labels'] == 1], plot_single = False)

#%%
'''
    Clustering in Parameter Space Day 1
'''
df = inf_mean_df.drop(['model', 'ag_idx', 'group', 'lr_day2', 'theta_Q_day2', 'theta_rep_day2', 'labels'], axis = 1)
normalized_df = (df-df.mean())/df.std()

kmeans, cluster_groups_between, _ = anal.kmeans(normalized_df.to_dict(orient='list'), 
                                             inf_mean_df, 
                                             n_clusters = 2,
                                             num_reps = 100,
                                             title = 'both days')
newdf = inf_mean_df
newdf['labels'] = kmeans.labels_

newdf0 = newdf[newdf['labels'] == 0]
newdf1 = newdf[newdf['labels'] == 0]

_, group_behav_df0, _, _ = utils.simulate_data('B', 
                                                len(newdf0),
                                                group = list(newdf0['group']),
                                                params = newdf0)

_, group_behav_df1, _, _ = utils.simulate_data('B', 
                                                len(newdf0),
                                                group = list(newdf0['group']),
                                                params = newdf0)

utils.plot_grouplevel(group_behav_df0, group_behav_df1, plot_single = False)

#%%
'''
    Correlation Analyses
'''

""" age vs ER, RT, model parameters """
for param_x, param_y in itertools.product(['age'],['ER_dtt', 'ER_stt', 'ER_total',
                                                   'RT', *param_names]):
    r,p = scipy.stats.pearsonr(complete_df[param_x], complete_df[param_y])
    print(f'{param_x} vs {param_y}: r=%.4f, p=%.4f'%(r,p))
    
    if p < 0.05:
        x = np.array(complete_df[param_x]).reshape(-1,1)
        y = np.array(complete_df[param_y]).reshape(-1,1)
        linmodel = LinearRegression()
        linmodel.fit(x, y)
        print(f"slope: {linmodel.coef_}\n")

""" Errors vs model parameters """
for param_x, param_y in itertools.product(['ER_total_day1', 'ER_stt_day1', 'ER_dtt_day1',
                                           'ER_total_day2', 'ER_stt_day2', 'ER_dtt_day2'],
                                          [*param_names]):
    
    r,p = scipy.stats.pearsonr(complete_df[param_x], complete_df[param_y])
    
    if p < 0.1:
        print(f'\n{param_x} vs {param_y}: r=%.4f, p=%.4f'%(r,p))
        x = np.array(complete_df[param_x]).reshape(-1,1)
        y = np.array(complete_df[param_y]).reshape(-1,1)
        linmodel = LinearRegression()
        linmodel.fit(x, y)
        print(f"slope: {linmodel.coef_}")

""" Points vs model parameters """
"Day 1"
for param_x, param_y in itertools.product(['theta_Q_day1', 'theta_rep_day1', 'conflict_param_day1'], 
                                          ['points_day1',
                                           'points_stt_day1',
                                           'points_dtt_day1',
                                           'points_randomdtt_day1', 
                                           'points_congruent_day1', 
                                           'points_incongruent_day1']):

    print(f"\n~~~~~ {param_x} ~~~~~")
    r,p = scipy.stats.pearsonr(complete_df[param_x], complete_df[param_y])
    print(f'{param_x} vs {param_y}: r=%.4f, p=%.4f'%(r,p))
    
    if p < 0.1:
        x = np.array(complete_df[param_x]).reshape(-1,1)
        y = np.array(complete_df[param_y]).reshape(-1,1)
        linmodel = LinearRegression()
        linmodel.fit(x, y)
        print(f"slope: {linmodel.coef_}\n")

        
"Day 2"
for param_x, param_y in itertools.product(['theta_Q_day2', 'theta_rep_day2', 'conflict_param_day2'], 
                                          ['points_day2',
                                           'points_stt_day2',
                                           'points_dtt_day2',
                                           'points_randomdtt_day2', 'points_congruent_day2', 'points_incongruent_day2']):
    
    print(f"\n~~~~~ {param_x} ~~~~~")
    r,p = scipy.stats.pearsonr(complete_df[param_x], complete_df[param_y])
    print(f'{param_x} vs {param_y}: r=%.4f, p=%.4f'%(r,p))
    
    if p < 0.1:
        x = np.array(complete_df[param_x]).reshape(-1,1)
        y = np.array(complete_df[param_y]).reshape(-1,1)
        linmodel = LinearRegression()
        linmodel.fit(x, y)
        print(f"slope: {linmodel.coef_}\n")
        
        
""" Other """
r,p = scipy.stats.pearsonr(complete_df['RT_stt_seq_day1'], complete_df['points_stt_day1'])
print("\n")
print(r)
print(p)

r,p = scipy.stats.pearsonr(complete_df['RT_stt_seq_day1'], complete_df['points_stt_seq_day1'])
print("\n")
print(r)
print(p)

r,p = scipy.stats.pearsonr(complete_df['RT_stt_rand_day1'], complete_df['points_stt_rand_day1'])
print("\n")
print(r)
print(p)

#%%
'''
    Test sequence knowledge vs inferred parameters and sociopsy Data
'''

from scipy.stats import ttest_ind

for param_y in [*param_names, 'RT', 'age']:
    t_statistic, p_value = ttest_ind(pd.to_numeric(complete_df[complete_df['q_notice_a_sequence'] == 0][param_y]), 
                                     pd.to_numeric(complete_df[complete_df['q_notice_a_sequence'] == 1][param_y]))
    print(f"for noticed a seq vs {param_y}: p=%.4f, t = %.4f\n"%(p_value, t_statistic))


for param in [*param_names, 'RT']:
    t_statistic, p_value = ttest_ind(pd.to_numeric(complete_df[complete_df['gender'] == 0][param]), 
                                     pd.to_numeric(complete_df[complete_df['gender'] == 1][param]))
    print(f"for gender vs {param}: p={p_value}\n")

t_statistic, p_value = ttest_ind(complete_df[complete_df['gender'] == 0]['age'], 
                                 complete_df[complete_df['gender'] == 1]['age'])
print(f"for gender vs age: p={p_value}")

#%%
import pingouin as pg
# complete_df['points_day1'] = complete_df['points_day1'].astype('float')
# complete_df['RT_day1'] = complete_df['RT_day1'].astype('float')
pg.partial_corr(data=complete_df, x='points_day1', y='conflict_param_day1', covar='age')
pg.partial_corr(data=complete_df, x='points_day2', y='conflict_param_day2', covar='age')

#%%
'''
    Plot RT as boxplots
'''

RT_df = complete_df.loc[:, ['ID', 
                            'group',
                            'RT_stt_seq_day1', 
                            'RT_stt_rand_day1', 
                            'RT_stt_seq_day2', 
                            'RT_stt_rand_day2']]

RT_df['RT Diff Day 1'] = (np.squeeze(RT_df.loc[:, ['RT_stt_rand_day1']]) - np.squeeze(RT_df.loc[:, ['RT_stt_seq_day1']])).astype(float)
RT_df['RT Diff Day 2'] = (np.squeeze(RT_df.loc[:, ['RT_stt_rand_day2']]) - np.squeeze(RT_df.loc[:, ['RT_stt_seq_day2']])).astype(float)

# RT_df = RT_df.rename(columns={'RT_seq_day1':'Fix (Day 1)', 'RT_seq_day2':'Fix (Day 2)',
#                               'RT_rand_day1':'Rand (Day 1)', 'RT_rand_day2':'Rand (Day 2)'})

RT_df_melted = RT_df.melt(id_vars = ['ID'], value_vars=  ['RT Diff Day 1', 
                                                          'RT Diff Day 2'])

RT_df_melted['Day'] = RT_df_melted['variable'].map(lambda x: 'Day 1' if '1' in x else 'Day 2')

r,p = scipy.stats.ttest_1samp(RT_df['RT Diff Day 1'], 0)
print(r)
print(p)

r,p = scipy.stats.ttest_1samp(RT_df['RT Diff Day 2'], 0)
print(r)
print(p)

fig, ax = plt.subplots()
sns.boxplot(data=RT_df_melted, x='variable', y='value')
ax.set_ylabel(r'$\Delta$RT (rand-fix) (ms)', fontsize=20)
# ax.set_xlabel('Condition')
plt.savefig('/home/sascha/Downloads/diffs_rt.tiff', dpi = 600)
plt.show()

#%%
'''
    Plot HPCF as boxplots
'''
hpcf_df = complete_df.loc[:, ['ID', 
                              'hpcf_cong_day1', 
                              'hpcf_incong_day1',
                              'hpcf_rand_day1',
                              'hpcf_cong_day2', 
                              'hpcf_incong_day2',
                              'hpcf_rand_day2']]

hpcf_df = hpcf_df.rename(columns={'hpcf_cong_day1' : 'Cong (Day 1)',
                               'hpcf_incong_day1' : 'Incong (Day 1)',
                               'hpcf_rand_day1' : 'Random (Day 1)',
                               'hpcf_cong_day2' : 'Cong (Day 2)',
                               'hpcf_incong_day2' : 'Incong (Day 2)',
                               'hpcf_rand_day2' : 'Random (Day 2)'})

hpcf_df_melted = hpcf_df.melt(id_vars = ['ID'], value_vars=  ['Cong (Day 1)',
                                                              'Incong (Day 1)',
                                                              'Random (Day 1)',
                                                              'Cong (Day 2)',
                                                              'Incong (Day 2)',
                                                              'Random (Day 2)'])

hpcf_df_melted['value'] = hpcf_df_melted['value'].apply(lambda x: x*100)
hpcf_df_melted['Trialtype'] = hpcf_df_melted['variable'].apply(lambda x: 'Inc' if 'Incong' in x else 'Cong' if 'Cong' in x else 'Rand')
hpcf_df_melted['Day'] = hpcf_df_melted['variable'].apply(lambda x: 'Day 1' if 'Day 1' in x else 'Day 2')
hpcf_df_melted.rename(columns={'value':'HRCF'}, inplace = True)

fig, ax = plt.subplots()
sns.boxplot(data=hpcf_df_melted, x='variable', y='HRCF', hue = 'Trialtype')
ax.set_ylabel('HRCF (%)', fontsize=20)
ax.set_xlabel('Condition')
# plt.savefig('/home/sascha/Downloads/diffs_hpcf.tiff', dpi = 600)
plt.show()

r,p = scipy.stats.ttest_rel(hpcf_df['Cong (Day 1)'], hpcf_df['Incong (Day 1)'])
print(r)
print(p)

r,p = scipy.stats.ttest_rel(hpcf_df['Cong (Day 1)'], hpcf_df['Random (Day 1)'])
print(r)
print(p)

r,p = scipy.stats.ttest_rel(hpcf_df['Incong (Day 1)'], hpcf_df['Random (Day 1)'])
print(r)
print(p)

r,p = scipy.stats.ttest_rel(hpcf_df['Cong (Day 2)'], hpcf_df['Incong (Day 2)'])
print(r)
print(p)

r,p = scipy.stats.ttest_rel(hpcf_df['Cong (Day 2)'], hpcf_df['Random (Day 2)'])
print(r)
print(p)

r,p = scipy.stats.ttest_rel(hpcf_df['Incong (Day 2)'], hpcf_df['Random (Day 2)'])
print(r)
print(p)

#%%
'''
    Scatter HRCF vs. ΔRT (R-S) jeweils day 1 und day 2 gemittelt
'''

hpcf = complete_df.loc[:, ['ID', 'hpcf_cong_day1', 'hpcf_incong_day1', 'hpcf_cong_day2', 'hpcf_incong_day2']]
hpcf['HPCF Diff Day 1'] = (np.array(hpcf['hpcf_cong_day1']) - np.array(hpcf['hpcf_incong_day1'])).astype(float)
hpcf['HPCF Diff Day 2'] = (np.array(hpcf['hpcf_cong_day2']) - np.array(hpcf['hpcf_incong_day2'])).astype(float)

df = pd.merge(hpcf, RT_df, on = 'ID')

r,p = scipy.stats.pearsonr(df['HPCF Diff Day 1'], df['RT Diff Day 1'])

fig, ax = plt.subplots(2,1, figsize=(10,15))
sns.regplot(data = df, x='RT Diff Day 1', y = 'HPCF Diff Day 1', color='r', ax = ax[0])
# ax[0].tick_params(axis = 'x', fontsize= 100)
# ax[0].set_xticks(fontsize= 25)
ax[0].set_ylabel(r'$\Delta$HRCF (cong-inc)', fontsize= 25)
ax[0].set_xlabel(r'$\Delta$RT (rand-fix) (ms)', fontsize= 25)
# ax[0].set_title('Day 1', fontsize=25)
ax[0].text(-14, 0.3, 'r = %.2f\np = %.3f'%(r,p), fontsize=25)
ax[0].tick_params(axis='both', labelsize=20)

r,p = scipy.stats.pearsonr(df['HPCF Diff Day 2'], df['RT Diff Day 2'])

sns.regplot(data = df, x='RT Diff Day 2', y = 'HPCF Diff Day 2', color='r', ax = ax[1])
ax[1].set_ylabel(r'$\Delta$HRCF (cong-inc)', fontsize= 25)
ax[1].set_xlabel(r'$\Delta$RT (rand-fix) (ms)', fontsize= 25)
# ax[1].set_title('Day 2', fontsize = 25)
ax[1].text(-8, 0.3, 'r = %.2f\np = %.5f'%(r,p), fontsize=25)
ax[1].tick_params(axis='both', labelsize=20)
plt.savefig('/home/sascha/Downloads/corrs.tiff', dpi = 600)
plt.show()

'''
    Scatter θR vs. ΔRT (R-S) jeweils day 1 und day 2 gemittelt
'''

df = pd.merge(complete_df.loc[:, ['ID', 'theta_rep_day1', 'theta_rep_day2']], RT_df, on = 'ID')

r,p = scipy.stats.pearsonr(df['theta_rep_day1'], df['RT Diff Day 1'])

fig, ax = plt.subplots(2,1, figsize=(10,15))
sns.regplot(data = df, x='RT Diff Day 1', y = 'theta_rep_day1', color='r', ax = ax[0])
# ax[0].tick_params(axis = 'x', fontsize= 100)
# ax[0].set_xticks(fontsize= 25)
ax[0].set_ylabel(r'$\Delta$HRCF (cong-inc)', fontsize= 25)
ax[0].set_xlabel(r'$\Delta$RT (rand-fix) (ms)', fontsize= 25)
# ax[0].set_title('Day 1', fontsize=25)
ax[0].text(-14, 1.25, 'r = %.2f\np = %.3f'%(r,p), fontsize=25)
ax[0].tick_params(axis='both', labelsize=20)

r,p = scipy.stats.pearsonr(df['theta_rep_day2'], df['RT Diff Day 2'])

sns.regplot(data = df, x='RT Diff Day 2', y = 'theta_rep_day2', color='r', ax = ax[1])
ax[1].set_ylabel(r'$\Delta$HRCF (cong-inc)', fontsize= 25)
ax[1].set_xlabel(r'$\Delta$RT (rand-fix) (ms)', fontsize= 25)
# ax[1].set_title('Day 2', fontsize = 25)
ax[1].text(-7, 1.4, 'r = %.2f\np = %.5f'%(r,p), fontsize=25)
ax[1].tick_params(axis='both', labelsize=20)
# plt.savefig('/home/sascha/Downloads/corrs.tiff', dpi = 600)
plt.show()

#%%
'''
    Scatter HRCF vs. ΔER (R-S) jeweils day 1 und day 2 gemittelt
'''

ER_df = complete_df.loc[:, ['ID', 
                            'group',
                            'ER_stt_seq_day1', 
                            'ER_stt_rand_day1', 
                            'ER_stt_seq_day2', 
                            'ER_stt_rand_day2']]

ER_df['ER Diff Day 1'] = (np.squeeze(ER_df.loc[:, ['ER_stt_rand_day1']]) - np.squeeze(ER_df.loc[:, ['ER_stt_seq_day1']])).astype(float)
ER_df['ER Diff Day 2'] = (np.squeeze(ER_df.loc[:, ['ER_stt_rand_day2']]) - np.squeeze(ER_df.loc[:, ['ER_stt_seq_day2']])).astype(float)

df = pd.merge(hpcf, ER_df)

r,p = scipy.stats.pearsonr(df['HPCF Diff Day 1'], df['ER Diff Day 1'])

fig, ax = plt.subplots(2,1, figsize=(10,15))
sns.regplot(data = df, x='ER Diff Day 1', y = 'HPCF Diff Day 1', color='r', ax = ax[0])
# ax[0].tick_params(axis = 'x', fontsize= 100)
# ax[0].set_xticks(fontsize= 25)
ax[0].set_ylabel(r'$\Delta$HRCF (cong-inc)', fontsize= 25)
ax[0].set_xlabel(r'$\Delta$ER (rand-fix) (ms)', fontsize= 25)
# ax[0].set_title('Day 1', fontsize=25)
ax[0].text(-0.02, 0.3, 'r = %.2f\np = %.3f'%(r,p), fontsize=25)
# ax[0].tick_params(axis='both', labelsize=20)

r,p = scipy.stats.pearsonr(df['HPCF Diff Day 2'], df['ER Diff Day 2'])

sns.regplot(data = df, x='ER Diff Day 2', y = 'HPCF Diff Day 2', color='r', ax = ax[1])
ax[1].set_ylabel(r'$\Delta$HRCF (cong-inc)', fontsize= 25)
ax[1].set_xlabel(r'$\Delta$ER (rand-fix) (ms)', fontsize= 25)
# ax[1].set_title('Day 2', fontsize = 25)
ax[1].text(-0.02, 0.3, 'r = %.2f\np = %.5f'%(r,p), fontsize=25)
# ax[1].tick_params(axis='both', labelsize=20)
# plt.savefig('/home/sascha/Downloads/corrs.tiff', dpi = 600)
plt.show()

'''
    Scatter θR vs. ΔER (R-S) jeweils day 1 und day 2 gemittelt
'''

df = pd.merge(complete_df.loc[:, ['ID', 'theta_rep_day1', 'theta_rep_day2']], ER_df, on = 'ID')

r,p = scipy.stats.pearsonr(df['theta_rep_day1'], df['ER Diff Day 1'])

fig, ax = plt.subplots(2,1, figsize=(10,15))
sns.regplot(data = df, x='ER Diff Day 1', y = 'theta_rep_day1', color='r', ax = ax[0])
# ax[0].tick_params(axis = 'x', fontsize= 100)
# ax[0].set_xticks(fontsize= 25)
ax[0].set_ylabel(r'$\theta_R$', fontsize= 25)
ax[0].set_xlabel(r'$\Delta$ER (rand-fix) (ms)', fontsize= 25)
# ax[0].set_title('Day 1', fontsize=25)
ax[0].text(-0.02, 1.25, 'r = %.2f\np = %.3f'%(r,p), fontsize=25)
ax[0].tick_params(axis='both', labelsize=20)

r,p = scipy.stats.pearsonr(df['theta_rep_day2'], df['ER Diff Day 2'])

sns.regplot(data = df, x='ER Diff Day 2', y = 'theta_rep_day2', color='r', ax = ax[1])
ax[1].set_ylabel(r'$\theta_R$', fontsize= 25)
ax[1].set_xlabel(r'$\Delta$ER (rand-fix) (ms)', fontsize= 25)
# ax[1].set_title('Day 2', fontsize = 25)
ax[1].text(-0.01, 1.4, 'r = %.2f\np = %.5f'%(r,p), fontsize=25)
ax[1].tick_params(axis='both', labelsize=20)
# plt.savefig('/home/sascha/Downloads/corrs.tiff', dpi = 600)
plt.show()

#%%
'''
    Parameter distributions of highest points
'''

params_day1 = ['lr_day1', 'theta_Q_day1', 'theta_rep_day1', 'conflict_param_day1', 'ER_stt_day1', 'ER_dtt_day1',  'RT_day1']
params_day2 = ['lr_day2', 'theta_Q_day2', 'theta_rep_day2', 'conflict_param_day2', 'ER_stt_day2', 'ER_dtt_day2',  'RT_day2']

for par in params_day1:
    r,p = scipy.stats.pearsonr(complete_df[par], complete_df['points_day1'])
    
    print(f"Correlation for points_day1 with {par}: r=%.4f, p=%.4f"%(r, p))


print('\n')
for par in params_day2:
    r,p = scipy.stats.pearsonr(complete_df[par], complete_df['points_incongruent_day2'])
    
    print(f"Correlation for points_incong_day2 with {par}: r=%.4f, p=%.4f"%(r, p))
    
#%%
params_day1 = ['theta_rep_day1', 'conflict_param_day1']
params_day2 = ['theta_rep_day2', 'conflict_param_day2']

import statsmodels.api as sm
x = np.array(complete_df.loc[:, [*params_day1]], dtype='float')
y = np.array(complete_df['points_day1'], dtype = 'float')
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print(model.summary())

x = np.array(complete_df.loc[:, [*params_day2]], dtype='float')
y = np.array(complete_df['points_day1'], dtype = 'float')
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print(model.summary())

#%%
'''
    Points: Day 1
'''
complete_df = complete_df.sort_values(by = 'points_day1')
rep_conf = complete_df.loc[:, 'theta_rep_day1'] + complete_df.loc[:, 'conflict_param_day1']

complete_df.iloc[0:20, :]

for param in params_day1:
    t, p = scipy.stats.ttest_ind(np.array(complete_df.iloc[0:20,:][param], dtype='float'), np.array(complete_df.iloc[-20:,:][param], dtype='float'))
    print(f"Test for points_day1 with {param}: t=%.4f, p=%.4f"%(t, p))

t, p = scipy.stats.ttest_ind(np.array(rep_conf[0:20], dtype='float'), np.array(rep_conf[-20:], dtype='float'))
print("Test for points_day1 with rep+conf: t=%.4f, p=%.4f"%(t, p))

#%%
'''
    Points: Day 2
'''
complete_df = complete_df.sort_values(by = 'points_day2')
rep_conf = complete_df.loc[:, 'theta_rep_day2'] + complete_df.loc[:, 'conflict_param_day2']

complete_df.iloc[0:20, :]

for param in params_day2:
    t, p = scipy.stats.ttest_ind(np.array(complete_df.iloc[0:20,:][param], dtype='float'), np.array(complete_df.iloc[-20:,:][param], dtype='float'))
    print(f"Test for points_day1 with {param}: t=%.4f, p=%.4f"%(t, p))
    
t, p = scipy.stats.ttest_ind(np.array(rep_conf[0:20], dtype='float'), np.array(rep_conf[-20:], dtype='float'))
print("Test for points_day1 with rep+conf: t=%.4f, p=%.4f"%(t, p))

#%%
'''
    Check for 3 approaches:
        1) (Almost) no habit learning
                low ΔER, low ΔRT, low θR --> same points in DTT random and seq (ΔPoints (DTT_s - DTT_r) ~ 0)
                
        2) Habit learning --> use habit to free cognitive resources and increase performance
                High ΔER, low to medium ΔRT --> more points in DTT Seq than DTT Rand (ΔPoints (DTT_s - DTT_r) > 0)
                
        3) Habit learning --> habitual responding independent from points
                High ΔER, high ΔRT --> less points in DTT Seq than DTT Rand (ΔPoints (DTT_s - DTT_r) < 0)
                
        (NB: Use ΔER and ΔRT of last 2 blocks on day 1 and last 4 blocks on day 2)
'''
import statsmodels.api as sm
df = pd.merge(ER_df.loc[:, ['ID', 'group', 'ER Diff Day 1', 'ER Diff Day 2']], RT_df.loc[:, ['ID', 'RT Diff Day 1', 'RT Diff Day 2']], on = 'ID')
complete_df['ΔPoints Day 1'] = np.array(complete_df.loc[:, ['points_dtt_seq_day1']]) - np.array(complete_df.loc[:, ['points_dtt_rand_day1']])
complete_df['ΔPoints Day 2'] = np.array(complete_df.loc[:, ['points_dtt_seq_day2']]) - np.array(complete_df.loc[:, ['points_dtt_rand_day2']])
df = pd.merge(df, complete_df.loc[:, ['ID', 'ΔPoints Day 1', 'ΔPoints Day 2']], on = 'ID')

x = np.array(df.loc[:, ['ER Diff Day 1', 'RT Diff Day 1']], dtype='float')
y = np.array(df['ΔPoints Day 1'], dtype = 'float')
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print(model.summary())

x = np.array(df.loc[:, ['ER Diff Day 2', 'RT Diff Day 2']], dtype='float')
y = np.array(df['ΔPoints Day 2'], dtype = 'float')
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print(model.summary())




# sns.scatterplot(data = df, x = 'RT Diff Day 2', y = 'ΔPoints Day 2')
# sns.scatterplot(data = df, x = 'ER Diff Day 2', y = 'ΔPoints Day 2')

#%%

points_df = complete_df.loc[:, ['ID', 
                                'theta_rep_day1', 
                                'theta_rep_day2', 
                                'conflict_param_day1', 
                                'conflict_param_day2', 
                                'points_stt_day1', 
                                'points_dtt_day1', 
                                'points_stt_day2', 
                                'points_dtt_day2',
                                'hpcf_incong_day1',
                                'hpcf_incong_day2']]

points_df['day1_comb'] = points_df['theta_rep_day1']+points_df['conflict_param_day1']
points_df['ER_dtt_day1'] = complete_df['ER_dtt_day1']
points_df['day2_comb'] = points_df['theta_rep_day2']+points_df['conflict_param_day2']
points_df['ER_dtt_day2'] = complete_df['ER_dtt_day2']

fig, ax = plt.subplots(2,1, figsize=(10,15))
sns.regplot(data = points_df, x = 'day1_comb', y='hpcf_incong_day1', ax = ax[0])
ax[0].set_xlabel('theta_rep + theta_conf')
r,p = scipy.stats.pearsonr(complete_df['conflict_param_day1'],complete_df['hpcf_incong_day1'])
print(r)
print(p)

sns.regplot(data = points_df, x = 'day2_comb', y='hpcf_incong_day2', ax = ax[1])
ax[1].set_xlabel('theta_rep + theta_conf')
r,p = scipy.stats.pearsonr(complete_df['conflict_param_day2'], complete_df['ER_stt_day2'])
print(r)
print(p)
plt.show()

#%%
'''
    points ~ θ_Q, θ_R, θ_Conflict
'''
import statsmodels.api as sm

regr_params = ['theta_Q_day2', 'theta_rep_day2', 'conflict_param_day2']
x = np.array(complete_df.loc[:, [*regr_params]], dtype='float')
for rpar_idx1 in range(len(regr_params)):
    for rpar_idx2 in range(rpar_idx1+1, len(regr_params)):
        r,p = scipy.stats.pearsonr(complete_df[regr_params[rpar_idx1]], complete_df[regr_params[rpar_idx2]])
        print(f"Correlation for {regr_params[rpar_idx1]} with {regr_params[rpar_idx2]}: r=%.4f, p=%.4f"%(r,p))

y = np.array(complete_df['points_day2'], dtype = 'float')
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print(model.summary())

y = np.array(complete_df['points_stt_day2'], dtype = 'float')
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print(model.summary())

y = np.array(complete_df['points_dtt_day2'], dtype = 'float')
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print(model.summary())

y = np.array(complete_df['points_congruent_day2'], dtype = 'float')
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print(model.summary())

y = np.array(complete_df['points_incongruent_day2'], dtype = 'float')
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print(model.summary())

y = np.array(complete_df['points_randomdtt_day2'], dtype = 'float')
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print(model.summary())