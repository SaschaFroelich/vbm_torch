#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:26:59 2023

Parameter Recovery 
1. Simulation
2. Group-Level Inference
3. Analysis

@author: sascha
"""

# import sys
# sys.modules[__name__].__dict__.clear()

import ipdb

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import torch

import scipy

import models_torch as models

import env
import utils

import inferencemodels
# import inferencemodel_sarah as inferencemodels

import sys
from datetime import datetime
import pickle

import numpy as np 

#%%

model = 'BQ'
resim =  0 # whether to simulate agents with inferred parameters
method = 'svi' # "svi" or "mcmc"
num_agents = 52

assert num_agents%4 == 0, "num_agents must be divisible by 4."
# k = 4.
print(f"Running model {model}")

#%%
'''
Simulate data in parallel
'''

group = [0]*(num_agents//4)
group.extend([1]*(num_agents//4))
group.extend([2]*(num_agents//4))
group.extend([3]*(num_agents//4))

# params = {'lr_day1' : 0*torch.ones(num_agents),
#           'theta_Q_day1' : 4*torch.ones(num_agents),
#           'theta_rep_day1' : 1*torch.ones(num_agents),
#           'conflict_param_day1' : 0.*torch.ones(num_agents),
          
#           'lr_day2' : 0*torch.ones(num_agents),
#           'theta_Q_day2' : 4*torch.ones(num_agents),
#           'theta_rep_day2' : 1*torch.ones(num_agents),
#           'conflict_param_day2' : 0.*torch.ones(num_agents)}

groupdata_dict, group_behav_df, _, params_sim_df = utils.simulate_data(model, 
                                                                      num_agents,
                                                                      group = group)

# utils.plot_grouplevel(group_behav_df)
#%%
'''
Inference
'''

"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent(model, 
                         group, 
                         num_agents = num_agents)

print("===== Starting inference =====")
"----- Start Inference"
infer = inferencemodels.GeneralGroupInference(agent, groupdata_dict)
infer.infer_posterior(iter_steps = 10_000, num_particles = 10)

"----- Sample parameter estimates from posterior"
post_sample_df = infer.sample_posterior()
post_sample_df['group'] = post_sample_df['ag_idx'].map(lambda x: groupdata_dict['group'][0][x])
post_sample_df['model'] = [model]*len(post_sample_df)

"----- Save results to file"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
pickle.dump((post_sample_df, group_behav_df, infer.loss, params_sim_df), open(f"parameter_recovery/param_recov_model_{model}_{timestamp}.p", "wb" ) )

#%%
log_like = infer.compute_ll()

#%%
'''
Analysis
'''
#%%
"----- Open Files"
# import sys
# sys.modules[__name__].__dict__.clear() # Clear variables

import utils
import pandas as pd
import seaborn as sns
import analysis as anal
import numpy as np
import matplotlib.pylab as plt
import pickle
import tkinter as tk

from tkinter import filedialog

def open_files():
    global filenames
    filenames = filedialog.askopenfilenames()
    print(f'File paths: {filenames}')
    # return filenames
    root.destroy()

root = tk.Tk()
button = tk.Button(root, text="Open Files", command=open_files)
print(button)
button.pack()
root.mainloop()

res = pickle.load(open( filenames[0], "rb" ))
post_sample_df, expdata_df, loss, params_df = res

# params_df['paramtype'] = ['sim']*len(params_df)

inf_mean_df = pd.DataFrame(post_sample_df.groupby(['model','ag_idx','group'], as_index = False).mean())
# inf_mean_df['paramtype'] = ['inf']*len(params_df)
model = inf_mean_df['model'][0]
num_agents = len(inf_mean_df['ag_idx'].unique())
num_params = len(params_df.columns) - 3
# post_sample_df, params_sim_df, group_behav_df, loss, param_names = pickle.load(open( filenames[0], "rb" ))

'''
Plot ELBO
'''
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots()
plt.plot(loss[:250])
plt.title(f"ELBO for model {model}")
ax.set_xlabel("Number of iterations")
ax.set_ylabel("ELBO")
plt.show()

num_plot_cols = 3
num_plot_rows = int((num_params <= num_plot_cols) * 1 + \
                (num_params > num_plot_cols) * np.ceil(num_params / num_plot_cols))

fig = plt.figure()
gs = fig.add_gridspec(num_plot_rows, num_plot_cols, hspace=0.5, wspace = 0.3)
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
Correlation analysis both days
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
    
    leavnodes = anal.cluster_analysis(corr_dict_temp, title = 'all correlations exp')
    kmeans, cluster_groups, c_distances = anal.kmeans(corr_dict_temp, 
                                             inf_mean_df_temp, 
                                             n_clusters = 2,
                                             num_reps = 1,
                                             plotfig = False)
    dd.extend(c_distances)

#%%
'''
Correlation analysis day 1
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
    
    leavnodes = anal.cluster_analysis(corr_dict_temp, title = 'all correlations day 1')
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

leavnodes = anal.cluster_analysis(corr_dict, title = 'all correlations sim')
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
Perform correlation analysis only within day 1
'''
post_sample_df_day1 = post_sample_df.drop(['lr_day1', 'theta_Q_day1', 'theta_rep_day1'], axis = 1)
corr_dict_day1 = anal.within_subject_corr(post_sample_df_day1)

leavnodes_day1 = anal.cluster_analysis(corr_dict_day1, title='day 1')

inf_mean_df[inf_mean_df['ag_idx'].isin(leavnodes[0:16])]['group']

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(leavnodes[0:7])],
                      expdata_df[expdata_df['ag_idx'].isin(leavnodes[6:16])])

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(leavnodes[0:16])],
                      expdata_df[expdata_df['ag_idx'].isin(leavnodes[16:])])

#%%
'''
Perform correlation analysis only within day 2
'''
post_sample_df_day2 = post_sample_df.drop(['lr_day2', 'theta_Q_day2', 'theta_rep_day2'], axis = 1)
corr_dict_day2 = anal.within_subject_corr(post_sample_df_day2)

leavnodes_day2 = anal.cluster_analysis(corr_dict_day2, title = 'day 2')

#%%
'''
Perform correlation analysis only between days
'''

corr_dict_day_between = corr_dict.copy()
del corr_dict_day_between['lr_day1_vs_theta_Q_day1']
del corr_dict_day_between['lr_day2_vs_theta_Q_day2']

del corr_dict_day_between['lr_day1_vs_theta_rep_day1']
del corr_dict_day_between['lr_day2_vs_theta_rep_day2']

del corr_dict_day_between['theta_Q_day1_vs_theta_rep_day1']
del corr_dict_day_between['theta_Q_day2_vs_theta_rep_day2']

leavnodes_betweendays = anal.cluster_analysis(corr_dict_day_between, title = 'between days')

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
    
    leavnodes = anal.cluster_analysis(corr_dict_temp, title = 'all correlations day 1')
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