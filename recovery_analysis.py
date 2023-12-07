#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:32:39 2023

@author: sascha
"""

"----- Open Files"
# import sys
# sys.modules[__name__].__dict__.clear() # Clear variables

import scipy
import utils
import pandas as pd
import seaborn as sns
import analysis_tools as anal
import numpy as np
import matplotlib.pylab as plt

out = utils.get_data_from_file()

if len(out) == 7:
    post_sample_df, expdata_df, loss, params_df, num_params, sociopsy_df, elbo_tuple = out
    
elif len(out) == 6:
    post_sample_df, expdata_df, loss, params_df, num_params, sociopsy_df = out
    BIC = loss[1]
    AIC = loss[2]
    loss = loss[0]

# params_df['paramtype'] = ['sim']*len(params_df)

inf_mean_df = pd.DataFrame(post_sample_df.groupby(['model','ag_idx','group'], as_index = False).mean())
# inf_mean_df['paramtype'] = ['inf']*len(params_df)
model = inf_mean_df['model'][0]
num_agents = len(inf_mean_df['ag_idx'].unique())
num_params = len(params_df.columns) - 3
# post_sample_df, params_sim_df, group_behav_df, loss, param_names = pickle.load(open( filenames[0], "rb" ))

print(f"Recovery of model {model} for {num_agents} agents after %d inference steps."%len(loss))
'''
Plot ELBO
'''
# import matplotlib.pyplot as plt
# import seaborn as sns
# fig, ax = plt.subplots()
# plt.plot(loss)
# plt.title(f"ELBO for model {model}")
# ax.set_xlabel("Number of iterations")
# ax.set_ylabel("ELBO")
# plt.show()

num_plot_cols = 3
num_plot_rows = int((num_params+1 <= num_plot_cols) + \
                (num_params+1 > num_plot_cols) * np.ceil((num_params+1) / num_plot_cols))

'''
Plot parameter distributions
'''
fig = plt.figure()
gs = fig.add_gridspec(num_plot_rows, num_plot_cols, hspace=0.5, wspace = 0.3)
ax = gs.subplots()
# params_df.columns
for param_idx in range(num_params):
    plot_col_idx = param_idx % num_plot_cols
    plot_row_idx = (param_idx // num_plot_cols)
    if num_params > 3:
        ca = sns.kdeplot(inf_mean_df[post_sample_df.columns[param_idx]], ax = ax[plot_row_idx, plot_col_idx])
        # ca.plot([0,0], ca.get_ylim())
        ax[plot_row_idx, plot_col_idx].set_xlabel(post_sample_df.columns[param_idx])
        
        if plot_col_idx > 0:
            ax[plot_row_idx, plot_col_idx].set_ylabel(None)
            
        if plot_row_idx > 0:
            ax[plot_row_idx, plot_col_idx].get_position().y0 += 10
        
    else:
        ca = sns.kdeplot(inf_mean_df[post_sample_df.columns[param_idx]], ax = ax[plot_col_idx])
        # ca.plot([0,0], ca.get_ylim())
        ax[plot_col_idx].set_xlabel(post_sample_df.columns[param_idx])
        if plot_col_idx > 0:
            ax[plot_col_idx].set_ylabel(None)
            
        if plot_row_idx > 0:
            ax[plot_col_idx].get_position().y0 += 10        

plt.show()

'''
Plot Inferred vs True
'''
fig = plt.figure(figsize=(16,12), dpi=100)
gs = fig.add_gridspec(num_plot_rows, num_plot_cols, hspace=0.2, wspace = 0.1)
ax = gs.subplots()
# params_df.columns
for param_idx in range(num_params+1):
    param = params_df.columns[param_idx]
    plot_col_idx = param_idx % num_plot_cols
    plot_row_idx = (param_idx // num_plot_cols)
    if num_params > 3:
        
        if param_idx < num_params:
            r,p = scipy.stats.pearsonr(params_df[param], inf_mean_df[param])
            print(f"r, and p for {param} : r=%.4f, p=%.4f"%(r,p))
            # ax[plot_row_idx, plot_col_idx].scatter(params_df[param], inf_mean_df[param])
            ax[plot_row_idx, plot_col_idx].plot(params_df[param], params_df[param], color='r', linewidth=0.05)
            sns.regplot(x = params_df[param], y=inf_mean_df[param], color='green', ax = ax[plot_row_idx, plot_col_idx])
            # ca = sns.kdeplot(inf_mean_df[post_sample_df.columns[param_idx]], ax = )
            # ca.plot([0,0], ca.get_ylim())
            ax[plot_row_idx, plot_col_idx].set_xlabel(param)
            ax[plot_row_idx, plot_col_idx].set_ylabel('inferred')
            if plot_col_idx > 0:
                ax[plot_row_idx, plot_col_idx].set_ylabel(None)
                
            if plot_row_idx > 0:
                ax[plot_row_idx, plot_col_idx].get_position().y0 += 10
                
        else:
            '''
                Plot ELBO
            '''
            ax[plot_row_idx, plot_col_idx].plot(loss)
            ax[plot_row_idx, plot_col_idx].set_xlabel('iteration')
            ax[plot_row_idx, plot_col_idx].set_ylabel('-ELBO')
            
    else:
        
        if param_idx < num_params:
            # ax[plot_col_idx].scatter(params_df[param], inf_mean_df[param])
            sns.regplot(x = params_df[param], y=inf_mean_df[param], color='green' , ax = ax[plot_col_idx])
            ax[plot_col_idx].plot(params_df[param], params_df[param], color='r', linewidth=0.05)
            # ca = sns.kdeplot(inf_mean_df[post_sample_df.columns[param_idx]], ax = )
            # ca.plot([0,0], ca.get_ylim())
            ax[plot_col_idx].set_xlabel(param)
            ax[plot_col_idx].set_ylabel('inferred')
            if plot_col_idx > 0:
                ax[plot_col_idx].set_ylabel(None)
                
            if plot_row_idx > 0:
                ax[plot_col_idx].get_position().y0 += 10
                
                
        else:
            '''
                Plot ELBO
            '''
            ax[plot_col_idx].plot(loss)
            ax[plot_col_idx].set_xlabel('iteration')
            ax[plot_col_idx].set_ylabel('-ELBO')
            
        
# fig.suptitle(f"Model {model}", fontsize = 32)
fig.suptitle('Dot = Mean of Posterior')
plt.show()
#%%
'''
Plot ELBO and Parameter Estimates
'''

fig, ax = plt.subplots()
plt.plot(loss)
plt.title(f"ELBO for model {model}")
ax.set_xlabel("Number of iterations")
ax.set_ylabel("ELBO")
plt.show()

all_params_df = pd.concat((params_df, inf_mean_df), axis = 0)

for param in params_df.columns[0:-3]:
    fig, ax = plt.subplots()
    plt.scatter(params_df[param], inf_mean_df[param])
    plt.plot(params_df[param], params_df[param])
    ax.set_xlabel('true value')
    ax.set_ylabel('inferred value')
    plt.title(param + f' for model {model}')
    plt.show()


#%%
'''
Posterior Predictives
'''
_ = utils.posterior_predictives(post_sample_df, exp_data = expdata_df)

#%%
'''
Simulate only from means
'''
groupdata_dict, group_behav_df, params_sim, params_sim_df = utils.simulate_data(model, 
                                                                        num_agents,
                                                                        group = list(inf_mean_df['group']),
                                                                        params = inf_mean_df)

utils.plot_grouplevel(expdata_df, group_behav_df, plot_single = True)

#%%
'''
Correlations between subjects
'''
anal.param_corr(inf_mean_df)

#%%
'''
Correlations within subjects
'''
corr_dict = anal.within_subject_corr(post_sample_df)

for key in corr_dict.keys():
    sns.kdeplot(corr_dict[key])
    plt.title(key)
    plt.show()
    
#%%
'''
Correlation analysis_tools both days
'''
import random
num_sims = 1000

dd = []

for sim in range(num_sims):
    print("Simulation num %d"%sim)
    num_agents = 36
    agidxs_temp = random.sample(range(52), 36)
    # agidxs_temp.sort()
    # print(agidxs_temp)
    
    post_sample_df_temp = post_sample_df[post_sample_df['ag_idx'].isin(agidxs_temp)]
    corr_dict_temp = anal.within_subject_corr(post_sample_df_temp)
    inf_mean_df_temp = inf_mean_df[inf_mean_df['ag_idx'].isin(agidxs_temp)]
    inf_mean_df_temp.loc[:, 'ag_idx'] = range(36)
    inf_mean_df_temp = inf_mean_df_temp.reset_index(drop=True)
    
    leavnodes = anal.cluster_analysis_tools(corr_dict_temp, title = 'all correlations exp')
    kmeans, cluster_groups, c_distances = anal.kmeans(corr_dict_temp, 
                                             inf_mean_df_temp, 
                                             n_clusters = 2,
                                             num_reps = 1,
                                             plotfig = False)
    dd.extend(c_distances)

#%%
'''
Correlation analysis_tools day 1
'''
import random
num_sims = 1000

dd = []

for sim in range(num_sims):
    print("Simulation num %d"%sim)
    num_agents = 36
    agidxs_temp = random.sample(range(52), 36)
    # agidxs_temp.sort()
    # print(agidxs_temp)
    post_sample_df_temp = post_sample_df.drop(['lr_day2', 'theta_Q_day2', 'theta_rep_day2'], axis = 1)
    post_sample_df_temp = post_sample_df_temp[post_sample_df_temp['ag_idx'].isin(agidxs_temp)]
    corr_dict_temp = anal.within_subject_corr(post_sample_df_temp)
    
    inf_mean_df_temp = inf_mean_df.drop(['lr_day2', 'theta_Q_day2', 'theta_rep_day2'], axis = 1)
    inf_mean_df_temp = inf_mean_df_temp[inf_mean_df_temp['ag_idx'].isin(agidxs_temp)]
    inf_mean_df_temp.loc[:, 'ag_idx'] = range(36)
    inf_mean_df_temp = inf_mean_df_temp.reset_index(drop=True)
    
    leavnodes = anal.cluster_analysis_tools(corr_dict_temp, title = 'all correlations day 1')
    kmeans, cluster_groups, c_distances = anal.kmeans(corr_dict_temp, 
                                             inf_mean_df_temp, 
                                             n_clusters = 2,
                                             num_reps = 1,
                                             plotfig = False)
    dd.extend(c_distances)

#%%
# '''
# Compare within-subject correlations with between-subject correlations
# '''
# corr_bewteen_df = inf_mean_df.drop(['ag_idx', 'model', 'group'], axis = 1)

# for col1 in range(len(corr_bewteen_df.columns)):
#     for col2 in range(col1, len(corr_bewteen_df.columns)):
#         col1_name = corr_bewteen_df.columns[col1]
#         col2_name = corr_bewteen_df.columns[col2]
#         r_between = corr_bewteen_df.iloc[:, col1].corr(corr_bewteen_df.iloc[:, col2])
        
#%%
'''
How close are subjects in correlation space?
'''

leavnodes = anal.cluster_analysis_tools(corr_dict, title = 'all correlations sim')
#%%
'''
Plot one cluster against the other
'''
utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(leavnodes[0:13])],
                      expdata_df[expdata_df['ag_idx'].isin(leavnodes[13:21])])

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(leavnodes[13:21])],
                      expdata_df[expdata_df['ag_idx'].isin(leavnodes[21:])])

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(leavnodes[0:3])],
                      expdata_df[expdata_df['ag_idx'].isin(leavnodes[21:])])

#%%
'''
Perform correlation analysis_tools only within day 1
'''
post_sample_df_day1 = post_sample_df.drop(['lr_day1', 'theta_Q_day1', 'theta_rep_day1'], axis = 1)
corr_dict_day1 = anal.within_subject_corr(post_sample_df_day1)

leavnodes_day1 = anal.cluster_analysis_tools(corr_dict_day1, title='day 1')

inf_mean_df[inf_mean_df['ag_idx'].isin(leavnodes[0:16])]['group']

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(leavnodes[0:7])],
                      expdata_df[expdata_df['ag_idx'].isin(leavnodes[6:16])])

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(leavnodes[0:16])],
                      expdata_df[expdata_df['ag_idx'].isin(leavnodes[16:])])

#%%
'''
Perform correlation analysis_tools only within day 2
'''
post_sample_df_day2 = post_sample_df.drop(['lr_day2', 'theta_Q_day2', 'theta_rep_day2'], axis = 1)
corr_dict_day2 = anal.within_subject_corr(post_sample_df_day2)

leavnodes_day2 = anal.cluster_analysis_tools(corr_dict_day2, title = 'day 2')

#%%
'''
Perform correlation analysis_tools only between days
'''

corr_dict_day_between = corr_dict.copy()
del corr_dict_day_between['lr_day1_vs_theta_Q_day1']
del corr_dict_day_between['lr_day2_vs_theta_Q_day2']

del corr_dict_day_between['lr_day1_vs_theta_rep_day1']
del corr_dict_day_between['lr_day2_vs_theta_rep_day2']

del corr_dict_day_between['theta_Q_day1_vs_theta_rep_day1']
del corr_dict_day_between['theta_Q_day2_vs_theta_rep_day2']

leavnodes_betweendays = anal.cluster_analysis_tools(corr_dict_day_between, title = 'between days')

import random
num_sims = 1000

dd = []

for sim in range(num_sims):
    print("Simulation num %d"%sim)
    num_agents = 36
    agidxs_temp = random.sample(range(52), 36)
    # agidxs_temp.sort()
    # print(agidxs_temp)
    post_sample_df_temp = post_sample_df[post_sample_df['ag_idx'].isin(agidxs_temp)]
    corr_dict_temp = anal.within_subject_corr(post_sample_df_temp)
    
    del corr_dict_temp['lr_day1_vs_theta_Q_day1']
    del corr_dict_temp['lr_day2_vs_theta_Q_day2']
    
    del corr_dict_temp['lr_day1_vs_theta_rep_day1']
    del corr_dict_temp['lr_day2_vs_theta_rep_day2']
    
    del corr_dict_temp['theta_Q_day1_vs_theta_rep_day1']
    del corr_dict_temp['theta_Q_day2_vs_theta_rep_day2']
    
    inf_mean_df_temp = inf_mean_df[inf_mean_df['ag_idx'].isin(agidxs_temp)]
    inf_mean_df_temp.loc[:, 'ag_idx'] = range(36)
    inf_mean_df_temp = inf_mean_df_temp.reset_index(drop=True)
    
    leavnodes = anal.cluster_analysis_tools(corr_dict_temp, title = 'all correlations day 1')
    kmeans, cluster_groups, c_distances = anal.kmeans(corr_dict_temp, 
                                             inf_mean_df_temp, 
                                             n_clusters = 2,
                                             num_reps = 1,
                                             plotfig = False)
    dd.extend(c_distances)

#%%
'''
K-means clustering
'''
labels, cluster_groups = anal.kmeans(corr_dict_day1, inf_mean_df, n_clusters = 2)

# anal.compare_leavnodes(grp1_agidx, leavnodes_day1[0:16])

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(cluster_groups[0])],
                      expdata_df[expdata_df['ag_idx'].isin(cluster_groups[1])], 
                      day=1)