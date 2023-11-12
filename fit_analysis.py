#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:31:18 2023

@author: sascha
"""

#%%
'''
Analysis_tools
'''
#%%
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

post_sample_df, expdata_df, loss, params_df, num_params = utils.get_data_from_file()
# if isinstance(measures, list):
#     loss = measures
    
# elif isinstance(measures, tuple):
#     loss, BIC = measures

# print("=======================================================")
# post_sample_df = post_sample_df[post_sample_df['ag_idx'] < 48]
# expdata_df = expdata_df[expdata_df['ag_idx'] < 48]
# params_df = params_df[params_df['ag_idx'] < 48]

inf_mean_df = pd.DataFrame(post_sample_df.groupby(['model', 'ag_idx', 'group'], as_index = False).mean())
model = post_sample_df['model'][0]
num_agents = len(post_sample_df['ag_idx'].unique())

print(f"Model fit of model {model} for {num_agents} agents after %d inference steps."%len(loss))
'''
Plot ELBO
'''
fig, ax = plt.subplots()
plt.plot(loss)
plt.title(f"ELBO for model {model} ({num_agents} agents)")
ax.set_xlabel("Number of iterations")
ax.set_ylabel("ELBO")
plt.show()

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
    ca = sns.kdeplot(inf_mean_df[post_sample_df.columns[param_idx]], ax = ax[plot_row_idx, plot_col_idx])
    # ca.plot([0,0], ca.get_ylim())
    ax[plot_row_idx, plot_col_idx].set_xlabel(post_sample_df.columns[param_idx])
    if plot_col_idx > 0:
        ax[plot_row_idx, plot_col_idx].set_ylabel(None)
        
    if plot_row_idx > 0:
        ax[plot_row_idx, plot_col_idx].get_position().y0 += 10

plt.show()

'''
Violin Plot
'''
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# plt.style.use("seaborn-v0_8-dark")

anal.violin(inf_mean_df)

#%%
'''
Check for how many participants Seqboost is larger than 0:
'''
sign_level = 0.001

from scipy.stats import ttest_1samp
for param in post_sample_df.columns:
    if param != 'group' and param != 'model' and param != 'ag_idx':
        print("----- Testing parameter %s"%param)
        for ag_idx in post_sample_df['ag_idx'].unique():
            t_statistic, p_value = ttest_1samp(post_sample_df[post_sample_df['ag_idx'] == ag_idx][param], 0)
            
            if p_value > sign_level:
                print(f"{param} for agent {ag_idx} is zero.")
    
#%%
'''
Differences day 1 & day 2
'''

inf_mean_df['Q/R_day1'] = inf_mean_df.apply(lambda row: row['theta_Q_day1']/row['theta_rep_day1'], axis = 1)
inf_mean_df['Q/R_day2'] = inf_mean_df.apply(lambda row: row['theta_Q_day2']/row['theta_rep_day2'], axis = 1)
post_sample_df['Q/R_day1'] = post_sample_df.apply(lambda row: row['theta_Q_day1']/row['theta_rep_day1'], axis = 1)
post_sample_df['Q/R_day2'] = post_sample_df.apply(lambda row: row['theta_Q_day2']/row['theta_rep_day2'], axis = 1)


anal.daydiff(inf_mean_df, sign_level = 0.01)
anal.violin(inf_mean_df)

anal.daydiff(post_sample_df, sign_level = 0.01)

anal.daydiff(post_sample_df[post_sample_df['group']==0], sign_level = 0.01)
anal.daydiff(post_sample_df[post_sample_df['group']==1], sign_level = 0.01)
anal.daydiff(post_sample_df[post_sample_df['group']==2], sign_level = 0.01)
anal.daydiff(post_sample_df[post_sample_df['group']==3], sign_level = 0.01)

anal.daydiff(post_sample_df[(post_sample_df['group']==0) | (post_sample_df['group']==1)], sign_level = 0.01)
anal.daydiff(post_sample_df[(post_sample_df['group']==2) | (post_sample_df['group']==3)], sign_level = 0.01)

anal.daydiff(post_sample_df[(post_sample_df['group']==0) | (post_sample_df['group']==2)], sign_level = 0.01)
anal.daydiff(post_sample_df[(post_sample_df['group']==1) | (post_sample_df['group']==3)], sign_level = 0.01)

# #%%
# '''
# Model B specific analysis_tools
# Ratios between parameters.
# '''

# Q_R_ratio_day1 = []
# Q_R_ratio_day2 = []
# for ag_idx in range(num_agents):
#     Q_R_ratio_day1.append((post_sample_df[post_sample_df['ag_idx'] == ag_idx]['theta_Q_day1'] / \
#         post_sample_df[post_sample_df['ag_idx'] == ag_idx]['theta_rep_day1']).mean())
        
#     Q_R_ratio_day2.append((post_sample_df[post_sample_df['ag_idx'] == ag_idx]['theta_Q_day2'] / \
#         post_sample_df[post_sample_df['ag_idx'] == ag_idx]['theta_rep_day2']).mean())
        
# diff = torch.tensor(Q_R_ratio_day1) - torch.tensor(Q_R_ratio_day2)
# sns.kdeplot(diff)
# plt.title('Ratio Differences Day 1 - Day 2')
# import scipy
# import numpy as np
# scipy.stats.ttest_1samp(np.asarray(diff), popmean=0)
# # post_sample_df.groupby.iloc[:, 0:-2][('ag_idx')]

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
corr_dict = anal.within_subject_corr(post_sample_df)

# corr_dict_errors = corr_dict.copy()
# corr_dict_errors['errors_stt'] = errorrates[0, :]
# corr_dict_errors['errors_dtt'] = errorrates[1, :]

#%%
'''
Kdeplots of within-subject  correlations
'''
for key in corr_dict.keys():
    sns.kdeplot(corr_dict[key])
    plt.title(key)
    plt.show()

#%%
'''
Correlation analysis_tools both days
'''
leavnodes = anal.cluster_analysis(corr_dict, title = 'all correlations exp')
kmeans, cluster_groups_bothdays, _ = anal.kmeans(corr_dict, 
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
utils.plot_grouplevel(expdata_df[expdata_df['ag_idx']==35], plot_single = False)

#%%
'''
Simulate only from means
'''
groupdata_dict, group_behav_df, params_sim, params_sim_df = utils.simulate_data(model, 
                                                                        num_agents,
                                                                        group = list(inf_mean_df['group']),
                                                                        params = inf_mean_df)

utils.plot_grouplevel(expdata_df, group_behav_df, plot_single = True)
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