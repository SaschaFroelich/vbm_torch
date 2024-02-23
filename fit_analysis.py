#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Tue Nov  7 17:31:18 2023

    Analysis of models fitted to behaviour.

    @author: sascha
"""

from IPython import get_ipython
get_ipython().run_line_magic("reset", "-f")

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

post_sample_df_day2, expdata_df, loss, params_df, num_params, sociopsy_df, agent_elbo_tuple, BIC, AIC, extra_storage = utils.get_data_from_file()
post_sample_df_day2['day'] = 2
Q_init = extra_storage[0]
param_names = extra_storage[9]
if extra_storage[11] >= 1e-03:
    raise Exception("rhalt too large for IC computation.")

day = extra_storage[2]

if day != 2:
    raise Exception("Have to load day 2, bruh.")

print(f"Preceding model is {extra_storage[3]}.")
# assert len(param_names) == num_params
# blocks = extra_storage[2]

if sociopsy_df is None:
    "Experimental data"
    experimental_data = 0
    
else:
    "Simulated data"
    experimental_data = 1

# _, expdata_df_clipre = pickle.load(open("behav_data/preproc_data.p", "rb" ))
# expdata_df_pub = pickle.load(open("behav_data/preproc_data_old_published_all.p", "rb" ))[1]

if 'ID_x' in post_sample_df_day2.columns:
    raise Exception("NONONONONO")

if 'handedness' in post_sample_df_day2.columns:
    inf_mean_df_day2 = pd.DataFrame(post_sample_df_day2.groupby(['model', 
                                                       'ag_idx', 
                                                       'group', 
                                                       'ID', 
                                                       'handedness'], as_index = False).mean())

else:
    inf_mean_df_day2 = pd.DataFrame(post_sample_df_day2.groupby(['model', 
                                                       'ag_idx', 
                                                       'group', 
                                                       'ID'], as_index = False).mean())  

inf_mean_df_day2['day'] = 2
inf_mean_df_day2 = inf_mean_df_day2.sort_values(by=['ag_idx'])

model = post_sample_df_day2['model'][0]
num_agents = len(post_sample_df_day2['ag_idx'].unique())

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
# print(np.array(loss[-1000:]).mean())

'''
    Plot Parameter Distributions
'''
if model != 'Bullshitmodel':
    print("Plot parameter distributions")
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
        
        # ca = sns.kdeplot(inf_mean_df_day2[post_sample_df_day2.columns[param_idx]], ax = ax[*ax_idxs], cut = 0)
        ca = sns.histplot(inf_mean_df_day2[post_sample_df_day2.columns[param_idx]], ax = ax[*ax_idxs], kde = False)
        ax[*ax_idxs].set_xlabel(post_sample_df_day2.columns[param_idx], fontsize = 20)    
    
        if plot_col_idx > 0:
            ax[*ax_idxs].set_ylabel(None)
            
        else:
            ax[*ax_idxs].set_ylabel('Density', fontsize = 20)
            ax[*ax_idxs].set_ylabel('Counts', fontsize = 20)
            
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
# anal.violin(inf_mean_df_day2, param_names, model)

complete_df_day2 = utils.create_complete_df(inf_mean_df_day2, sociopsy_df, expdata_df, post_sample_df_day2, param_names)
complete_df_day2['day'] == 2

# if 'RT' in complete_df.columns:
#     anal.violin(complete_df.loc[:, ['age', 'ER_stt', 'ER_dtt', 'RT', 'points']], model = 'sociopsy')
    
# else:
#     anal.violin(complete_df.loc[:, ['ER_stt', 'ER_dtt', 'points']], model = 'errors')

if 0:
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
        sns.kdeplot(post_sample_df_day2[param_names[param_idx]], ax = ax[param_idx])
        ax[param_idx].set_xlabel(param_names[param_idx])
        # ax[param_idx].set_xlim(xlims[param_idx])
    
    plt.show()

'''
    Simulate from means
'''
'''
er:
    0 : STT
    1 : Random
    2 : Congruent
    3 : incongruent
'''
er = torch.zeros((4, num_agents))
er[0, :] = torch.tensor(complete_df_day2['ER_stt']) # stt
er[1, :] = torch.tensor(complete_df_day2['ER_randomdtt']) # random
er[2, :] = torch.tensor(complete_df_day2['ER_congruent']) # congruent
er[3, :] = torch.tensor(complete_df_day2['ER_incongruent']) # incongruent

if day == 2:
    seq_counter_day2 = extra_storage[6]
    groupdata_dict_day2, sim_group_behav_df_day2, params_sim_df_day2, _ = utils.simulate_data(model, 
                                                                            num_agents,
                                                                            group = list(inf_mean_df_day2['group']),
                                                                            day = day,
                                                                            STT = 0,
                                                                            Q_init = Q_init,
                                                                            seq_init = seq_counter_day2,
                                                                            params = inf_mean_df_day2.loc[:, [*param_names]],
                                                                            errorrates = er)
    
    utils.plot_grouplevel(expdata_df, sim_group_behav_df_day2, plot_single = False)
    sim_group_behav_df_day2['day'] = 2
    
    print("\n\nCreating and appending dataframe for day 1.")
    filename_day1 = extra_storage[7]
    
    if extra_storage[10] == 'recovery':
        post_sample_df_day1, expdata_df_day1, loss_day1, params_df_day1, num_params_day1, sociopsy_df_day1, agent_elbo_tuple_day1, BIC_day1, AIC_day1, extra_storage_day1 = utils.get_data_from_file('parameter_recovery/'+filename_day1+'.p')    
        
    elif extra_storage[10] == 'behav_fit':
        post_sample_df_day1, expdata_df_day1, loss_day1, params_df_day1, num_params_day1, sociopsy_df_day1, agent_elbo_tuple_day1, BIC_day1, AIC_day1, extra_storage_day1 = utils.get_data_from_file('behav_fit/'+filename_day1+'.p')    
        
    else:
        raise Exception('Error')

    post_sample_df_day1['day'] = 1
    # param_names_day1 = params_df_day1.iloc[:, 0:-3].columns
    param_names_day1 = extra_storage_day1[9]
    if extra_storage_day1[11] >= 1e-03:
        raise Exception("rhalt too large for IC computation.")
    
    if 'handedness' in post_sample_df_day1.columns:
        inf_mean_df_day1 = pd.DataFrame(post_sample_df_day1.groupby(['model', 
                                                           'ag_idx', 
                                                           'group', 
                                                           'ID', 
                                                           'handedness'], as_index = False).mean())
    
    else:
        inf_mean_df_day1 = pd.DataFrame(post_sample_df_day1.groupby(['model', 
                                                           'ag_idx', 
                                                           'group', 
                                                           'ID'], as_index = False).mean())  
    
    inf_mean_df_day1 = inf_mean_df_day1.sort_values(by=['ag_idx'])
    
    model_day1 = post_sample_df_day1['model'][0]
    
    complete_df_day1 = utils.create_complete_df(inf_mean_df_day1, 
                                                sociopsy_df_day1,
                                                expdata_df_day1,
                                                post_sample_df_day1, 
                                                param_names_day1)
    
    # rename_dict = {col: col+'_day1' if (col != 'ID') and (col != 'ag_idx') and (col != 'group') else col for col in complete_df_day1.columns}
    # complete_df_day1 = complete_df_day1.rename(columns=rename_dict)
    complete_df_day1['day'] = 1
    complete_df_all = pd.concat([complete_df_day1, complete_df_day2], ignore_index=True)
    
    inf_mean_df_day1['day'] = 1
    inf_mean_df_all = pd.concat([inf_mean_df_day1, inf_mean_df_day2], ignore_index=True)
    
    post_sample_df_all = pd.concat([post_sample_df_day1, post_sample_df_day2], ignore_index=True)
    
    er = torch.zeros((4, num_agents))
    er[0, :] = torch.tensor(complete_df_day1['ER_stt']) # stt
    er[1, :] = torch.tensor(complete_df_day1['ER_randomdtt']) # random
    er[2, :] = torch.tensor(complete_df_day1['ER_congruent']) # congruent
    er[3, :] = torch.tensor(complete_df_day1['ER_incongruent']) # incongruent
    groupdata_dict_day1, sim_group_behav_df_day1, params_sim_df_day1, _ = utils.simulate_data(model_day1, 
                                                                            num_agents,
                                                                            group = list(inf_mean_df_day1['group']),
                                                                            day = 1,
                                                                            STT = 0,
                                                                            params = inf_mean_df_day1.loc[:, [*param_names_day1]],
                                                                            errorrates = er)
    
    sim_group_behav_df_day1['day'] = 1
    
    sim_df = pd.concat((sim_group_behav_df_day1, sim_group_behav_df_day2), ignore_index = True)
    
    del day
    del post_sample_df_day1, post_sample_df_day2

#%%
'''
    Plot behaviour both days
'''

# HPCF_DF = complete_df_all.loc[:, ['hpcf_cong', 'hpcf_incong',
#                                   'hpcf_seq', 'hpcf_rand', 'day', 'ID']]

utils.plot_hpcf(complete_df_all, title='Experiment')

hpcf_day1 = utils.compute_hpcf(sim_group_behav_df_day1)
hpcf_day1['day'] = 1
hpcf_day2 = utils.compute_hpcf(sim_group_behav_df_day2)
hpcf_day2['day'] = 2

hpcf_df_all = pd.concat((hpcf_day1, hpcf_day2), ignore_index = True)

utils.plot_hpcf(hpcf_df_all, title=f'{model}', post_pred = False)

#%%
'''
    Correlations with behavioural measures
'''

for day in [1,2]:
    print(f"=========== Day {day} =================")
    print("HPCF vs θ_Q")
    r,p = scipy.stats.spearmanr(complete_df_all[complete_df_all['day'] == day]['hpcf'], 
                               complete_df_all[complete_df_all['day'] == day]['theta_Q'])
    print(f"r={r}, p={p}")
    
    print("CI-spread vs θ_rep")
    r,p = scipy.stats.spearmanr(complete_df_all[complete_df_all['day'] == day]['CIspread'], 
                               complete_df_all[complete_df_all['day'] == day]['theta_rep'])
    print(f"r={r}, p={p}")
    
    print("CR-spread vs θ_rep")
    r,p = scipy.stats.spearmanr(complete_df_all[complete_df_all['day'] == day]['CRspread'], 
                               complete_df_all[complete_df_all['day'] == day]['theta_rep'])
    print(f"r={r}, p={p}")
    
    print("RI-spread vs θ_rep")
    r,p = scipy.stats.spearmanr(complete_df_all[complete_df_all['day'] == day]['RIspread'], 
                               complete_df_all[complete_df_all['day'] == day]['theta_rep'])
    print(f"r={r}, p={p}")
    
    print("CR-spread vs RI-spread")
    r,p = scipy.stats.spearmanr(complete_df_all[complete_df_all['day'] == day]['CRspread'], 
                               complete_df_all[complete_df_all['day'] == day]['RIspread'])
    print(f"r={r}, p={p}")

    if model == 'Repbias_Conflict_lr':
        print("CI-spread vs θ_Conflict")
        r,p = scipy.stats.spearmanr(complete_df_all[complete_df_all['day'] == day]['CIspread'], 
                                   complete_df_all[complete_df_all['day'] == day]['theta_conflict'])
        print(f"r={r}, p={p}")
        
        print("CR-spread vs θ_Conflict")
        r,p = scipy.stats.spearmanr(complete_df_all[complete_df_all['day'] == day]['CRspread'], 
                                   complete_df_all[complete_df_all['day'] == day]['theta_conflict'])
        print(f"r={r}, p={p}")
        
        print("RI-spread vs θ_Conflict")
        r,p = scipy.stats.spearmanr(complete_df_all[complete_df_all['day'] == day]['RIspread'], 
                                   complete_df_all[complete_df_all['day'] == day]['theta_conflict'])
        print(f"r={r}, p={p}")

        exploiters_df_day1 = complete_df_all[complete_df_all['day'] == 1].copy()
        mean = exploiters_df_day1['theta_conflict'].mean()
        std_dev = exploiters_df_day1['theta_conflict'].std()
        exploiters_df_day1.loc[:, 'theta_conflict_z'] = (exploiters_df_day1['theta_conflict'] - mean) / std_dev
        
        exploiters_df_day1['expl_conflict_z'] = exploiters_df_day1['theta_conflict_z'].map(lambda x: 1 if x > 1 else 0)
        
        "--> LOWER RT-SPREAD (at x>1)!!"
        r,p = scipy.stats.ttest_ind(exploiters_df_day1[exploiters_df_day1['expl_conflict_z'] == 1]['RT_diff_stt'], 
                              exploiters_df_day1[exploiters_df_day1['expl_conflict_z'] == 0]['RT_diff_stt'])
        print(f"r={r}, p={p}")
        
        r,p = scipy.stats.ttest_ind(exploiters_df_day1[exploiters_df_day1['expl_conflict_z'] == 1]['CRspread'], 
                              exploiters_df_day1[exploiters_df_day1['expl_conflict_z'] == 0]['CRspread'])
        print(f"r={r}, p={p}")
        
        "--> LOWER RI-SPREAD (at x>1)!!"
        r,p = scipy.stats.ttest_ind(exploiters_df_day1[exploiters_df_day1['expl_conflict_z'] == 1]['RIspread'], 
                              exploiters_df_day1[exploiters_df_day1['expl_conflict_z'] == 0]['RIspread'])
        print(f"r={r}, p={p}")
        
        r,p = scipy.stats.ttest_ind(exploiters_df_day1[exploiters_df_day1['expl_conflict_z'] == 1]['CIspread'], 
                              exploiters_df_day1[exploiters_df_day1['expl_conflict_z'] == 0]['CIspread'])
        print(f"r={r}, p={p}")
        
        r,p = scipy.stats.ttest_ind(exploiters_df_day1[exploiters_df_day1['expl_conflict_z'] == 1]['hpcf_rand'], 
                              exploiters_df_day1[exploiters_df_day1['expl_conflict_z'] == 0]['hpcf_rand'])
        print(f"r={r}, p={p}")
        
        exploiters_df_day2 = complete_df_all[complete_df_all['day'] == 2].copy()
        mean = exploiters_df_day2['theta_conflict'].mean()
        std_dev = exploiters_df_day2['theta_conflict'].std()
        exploiters_df_day2.loc[:, 'theta_conflict_z'] = (exploiters_df_day2['theta_conflict'] - mean) / std_dev
        
        exploiters_df_day2['expl_conflict_z'] = exploiters_df_day2['theta_conflict_z'].map(lambda x: 1 if x > 1 else 0)
        
        r,p = scipy.stats.pearsonr(exploiters_df_day1['theta_conflict_z'], exploiters_df_day2['theta_conflict_z'])
        print(f"r={r}, p={p}")
        
        r,p = scipy.stats.pearsonr(exploiters_df_day1['RT_diff_stt'], exploiters_df_day1['theta_conflict_z'])
        print(f"r={r}, p={p}")
        
        r,p = scipy.stats.pearsonr(exploiters_df_day2['RT_diff_stt'], exploiters_df_day2['theta_conflict_z'])
        print(f"r={r}, p={p}")
        
        r,p = scipy.stats.pearsonr(exploiters_df_day2['ER_diff_stt'], exploiters_df_day2['theta_conflict_z'])
        print(f"r={r}, p={p}")
        
        r,p = scipy.stats.pearsonr(exploiters_df_day1['RT_diff_stt'], exploiters_df_day1['theta_conflict'])
        print(f"r={r}, p={p}")
        
        r,p = scipy.stats.pearsonr(exploiters_df_day2['RT_diff_stt'], exploiters_df_day2['theta_conflict'])
        print(f"r={r}, p={p}")
        
        r,p = scipy.stats.pearsonr(exploiters_df_day2['ER_diff_stt'], exploiters_df_day2['theta_conflict'])
        print(f"r={r}, p={p}")
        
    elif model == 'Repbias_Interaction_lr':
        pass

#%%
'''
    Exploiters
'''

# exploiters_df_day1.loc[:, 'theta_rep_plus_conflict'] = exploiters_df_day1['theta_rep'] + exploiters_df_day1['theta_conflict']
# mean = exploiters_df_day1['theta_rep_plus_conflict'].mean()
# std_dev = exploiters_df_day1['theta_rep_plus_conflict'].std()
# exploiters_df_day1.loc[:, 'theta_rep_plus_conflict_z'] = (exploiters_df_day1['theta_rep_plus_conflict'] - mean) / std_dev
# exploiters_df_day1['expl_repconflict_z'] = exploiters_df_day1['theta_rep_plus_conflict_z'].map(lambda x: 1 if x > 1 else 0)


# exploiters_df_day2.loc[:, 'theta_rep_plus_conflict'] = exploiters_df_day2['theta_rep'] + exploiters_df_day2['theta_conflict']
# mean = exploiters_df_day2['theta_rep_plus_conflict'].mean()
# std_dev = exploiters_df_day2['theta_rep_plus_conflict'].std()
# exploiters_df_day2.loc[:, 'theta_rep_plus_conflict_z'] = (exploiters_df_day2['theta_rep_plus_conflict'] - mean) / std_dev
# exploiters_df_day2['expl_repconflict_z'] = exploiters_df_day2['theta_rep_plus_conflict_z'].map(lambda x: 1 if x > 1 else 0)

# r,p = scipy.stats.pearsonr(exploiters_df_day1['theta_rep_plus_conflict_z'], exploiters_df_day2['theta_rep_plus_conflict_z'])
# print(f"r={r}, p={p}")

# r,p = scipy.stats.pearsonr(exploiters_df_day1['RT_diff_stt'], exploiters_df_day1['theta_rep_plus_conflict_z'])
# print(f"r={r}, p={p}")

# r,p = scipy.stats.pearsonr(exploiters_df_day2['RT_diff_stt'], exploiters_df_day2['theta_rep_plus_conflict_z'])
# print(f"r={r}, p={p}")

# r,p = scipy.stats.pearsonr(exploiters_df_day2['ER_diff_stt'], exploiters_df_day2['theta_rep_plus_conflict_z'])
# print(f"r={r}, p={p}")

#%%
'''
    Sort by counterbalancing group
'''

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
    Check for how many participants parameters are different from 0:
'''
threshold = 0
BF = 5
inf_mean_df_nozero = inf_mean_df.copy()
"Bayesian Approach"
for param in param_names:
    if param not in ['group', 'model', 'ag_idx', 'ID', 'handedness']:
    # if param != 'group' and param != 'model' and param != 'ag_idx' and param != 'ID' and param !=:
        print("----- Testing parameter %s"%param)
        for ag_idx in post_sample_df_all['ag_idx'].unique():
            for day in [1,2]:
                lt0 = (np.array(post_sample_df_all[(post_sample_df_all['ag_idx'] == ag_idx) & (post_sample_df_all['day'] == day)][param]) <= 0).sum()
                gt0 = (np.array(post_sample_df_all[(post_sample_df_all['ag_idx'] == ag_idx) & (post_sample_df_all['day'] == day)][param]) > 0).sum()
                
                if gt0/lt0 <= BF and lt0/gt0 <= BF:
                    print(f"No evidence that {param} is different from 0 for agent {ag_idx} for day {day}")
                    inf_mean_df_nozero[inf_mean_df_nozero['ag_idx'] == ag_idx] = np.nan

anal.violin(inf_mean_df_nozero.loc[:, [*param_names]], model)

#%%
'''
    Fig for paper
    Individual posteriors
'''
fig, ax = plt.subplots(num_params, 2, figsize = (15, 15))

for param in range(num_params):
    for day in range(1, 3):
        for ag in range(num_agents):
            if param == 0:
                sns.kdeplot(post_sample_df_all[(post_sample_df_all['ag_idx']==ag) & (post_sample_df_all['day']==day)][param_names[param]], 
                            ax = ax[param, day-1], clip=(0,1))
                
            else:
                sns.kdeplot(post_sample_df_all[(post_sample_df_all['ag_idx']==ag) & (post_sample_df_all['day']==day)][param_names[param]], 
                            ax = ax[param, day-1])
                
            # if param == 4:
            #     dfgh
            ax[0, day-1].set_xlabel(r'learning rate')
            ax[1, day-1].set_xlabel(r'$\Theta_Q$')
            ax[2, day-1].set_xlabel(r'$\Theta_R$')
            ax[3, day-1].set_xlabel(r'$\Theta_\text{Conflict}$')
            ax[0, day-1].set_ylabel('')
            ax[1, day-1].set_ylabel('')
            ax[2, day-1].set_ylabel('')
            ax[3, day-1].set_ylabel('')
            # if param == 0:
                # 'lr'
            ax[0, day-1].set_xlim([-0.025, 0.08])
            ax[1, day-1].set_xlim([-1, 8])
            ax[2, day-1].set_xlim([-1, 5])
            ax[3, day-1].set_xlim([-2, 4])

# plt.savefig('posterior_diffs.svg')
plt.show()

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

post_sample_df_all['Q/R'] = post_sample_df_all.apply(lambda row: row['theta_Q']/row['theta_rep'], axis = 1)
param_names = list(param_names)
param_names.append('Q/R')

# diffs_df = anal.daydiff(inf_mean_df)
# diffs_df = pd.merge(diffs_df, sociopsy_df[sociopsy_df['ID'].isin(diffs_df['ID'])], on = 'ID')

# anal.violin(inf_mean_df, model)

diffs_df, clean_means_df = anal.daydiffBF(post_sample_df_all, BF = 5, parameter_names = list(param_names))

# anal.daydiff(post_sample_df_all, threshold = 0.05)

utils.lineplot_daydiffs(pd.melt(inf_mean_df_all, id_vars=['ag_idx', 'day'], value_vars = ['lr', 'theta_Q', 'theta_rep', 'theta_conflict'], var_name = 'parameter', value_name ='mean'))

utils.lineplot_daydiffs(clean_means_df)
utils.scatterplot_daydiffs(clean_means_df)

post_sample_df_all['daydiffBF'] = post_sample_df_all[param].apply(lambda row: diffs_df[diffs_df['parameter'] == row[]]['BF'])

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
anal.param_corr(inf_mean_df, method = 'spearman')

#%%
'''
    Plot Experimental Data
'''
utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'] == 35], plot_single = False)

#%%
'''
    Simulate only from means
'''

groupdata_dict, sim_group_behav_df, params_sim_df, _ = utils.simulate_data(model, 
                                                                        num_agents,
                                                                        group = list(inf_mean_df['group']),
                                                                        day = day,
                                                                        params = inf_mean_df.loc[:, [*param_names]])

utils.plot_grouplevel(expdata_df, sim_group_behav_df, plot_single = False)
# utils.plot_grouplevel(expdata_df, plot_single = True)


#%%

utils.plot_grouplevel(sim_group_behav_df, plot_single = False)

# df_temp = expdata_df[expdata_df['choices_GD']>=0]
# df_temp = df_temp[df_temp['trialsequence']>10]
# cgd = pd.DataFrame(df_temp.loc[:, ['ID', 'choices_GD', 'jokertypes']].groupby(['ID', 'jokertypes'], as_index = False).mean())

# cgd[cgd['jokertypes'] == 0]['choices_GD'].mean() # Random
# cgd[cgd['jokertypes'] == 1]['choices_GD'].mean() # Congruent
# cgd[cgd['jokertypes'] == 2]['choices_GD'].mean() # Incongruent
#%%
'''
    Posterior Predictives from posterior samples.
'''

# ers = anal.compute_errors(expdata_df)
out=utils.posterior_predictives(post_sample_df_all[post_sample_df_all['day']==2], 
                                param_names, 
                                Q_init = Q_init,
                                seq_init = seq_counter_day2,
                                errorrates=er)

#%%
'''
    Correlation Analyses of different behavioural measures
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

#%% 

'''
    ANOVA
'''
groupdf = utils.plot_grouplevel(expdata_df)
groupdf['plottingcat'] = groupdf.apply(lambda row: 'random1' if row['DTT Types'] == 'random' and row['day'] == 1 else\
                                       'congruent1' if row['DTT Types'] == 'congruent' and row['day'] == 1 else\
                                        'incongruent1' if row['DTT Types'] == 'incongruent' and row['day'] == 1 else\
                                        'random2' if row['DTT Types'] == 'random' and row['day'] == 2 else\
                                        'congruent2' if row['DTT Types'] == 'congruent' and row['day'] == 2 else\
                                        'incongruent2', axis = 1)

custom_palette = ['#c7028c', '#63040f', '#96e6c7'] # congruent, incongruent, random


RT_df = pd.DataFrame()
RTs = []
RTs.extend(list(complete_df['RT_stt_seq_day1']))
RTs.extend(list(complete_df['RT_stt_rand_day1']))
RTs.extend(list(complete_df['RT_stt_seq_day2']))
RTs.extend(list(complete_df['RT_stt_rand_day2']))

days = []
days.extend([1]*len(complete_df['RT_stt_seq_day1']))
days.extend([1]*len(complete_df['RT_stt_rand_day1']))
days.extend([2]*len(complete_df['RT_stt_seq_day2']))
days.extend([2]*len(complete_df['RT_stt_rand_day2']))

condition = []
condition.extend(['F']*len(complete_df['RT_stt_seq_day1']))
condition.extend(['R']*len(complete_df['RT_stt_rand_day1']))
condition.extend(['F']*len(complete_df['RT_stt_seq_day2']))
condition.extend(['R']*len(complete_df['RT_stt_rand_day2']))

ag_idx = []
ag_idx.extend(list(complete_df['ag_idx']))
ag_idx.extend(list(complete_df['ag_idx']))
ag_idx.extend(list(complete_df['ag_idx']))
ag_idx.extend(list(complete_df['ag_idx']))

RT_df['RT'] = RTs
RT_df['day'] = days
RT_df['Condition'] = condition
RT_df['ag_idx'] = ag_idx

ER_df = pd.DataFrame()
ERs = []
ERs.extend(list(complete_df['ER_stt_seq_day1']))
ERs.extend(list(complete_df['ER_stt_rand_day1']))
ERs.extend(list(complete_df['ER_stt_seq_day2']))
ERs.extend(list(complete_df['ER_stt_rand_day2']))

days = []
days.extend([1]*len(complete_df['ER_stt_seq_day1']))
days.extend([1]*len(complete_df['ER_stt_rand_day1']))
days.extend([2]*len(complete_df['ER_stt_seq_day2']))
days.extend([2]*len(complete_df['ER_stt_rand_day2']))

condition = []
condition.extend(['F']*len(complete_df['ER_stt_seq_day1']))
condition.extend(['R']*len(complete_df['ER_stt_rand_day1']))
condition.extend(['F']*len(complete_df['ER_stt_seq_day2']))
condition.extend(['R']*len(complete_df['ER_stt_rand_day2']))

ER_df['ER'] = ERs
ER_df['day'] = days
ER_df['Condition'] = condition
ER_df['ag_idx'] = ag_idx

groupdf['choices_GD'] *=  100
ER_df['ER'] *= 100

groupdf['DTT Types'] = groupdf['DTT Types'].str.title()

'''
    Plot HPCF by day.
'''

fig, ax = plt.subplots(1,3, figsize = (15, 5))
sns.lineplot(x = "day",
            y = "choices_GD",
            hue = "DTT Types",
            data = groupdf,
            palette = custom_palette,
            err_style = "bars",
            errorbar = ("se", 1),
            ax = ax[0])
ax[0].set_xticks([1,2])
# ax[0].set_ylim([60, 100])
ax[0].set_xlabel('Day')
ax[0].set_ylabel('HRCF (%)')


sns.lineplot(x = "day",
            y = "ER",
            hue = "Condition",
            data = ER_df,
            # palette = custom_palette,
            err_style = "bars",
            errorbar = ("se", 1),
            ax = ax[1])
ax[1].set_xticks([1,2])
# ax[2].set_ylim([0.61, 1])
ax[1].set_xlabel('Day')
ax[1].set_ylabel('ER (%)')

sns.lineplot(x = "day",
            y = "RT",
            hue = "Condition",
            data = RT_df,
            # palette = custom_palette,
            err_style = "bars",
            errorbar = ("se", 1),
            ax = ax[2])
ax[2].set_xticks([1,2])
# ax[1].set_ylim([0.61, 1])
ax[2].set_xlabel('Day')
ax[2].set_ylabel('RT (ms)')
# plt.savefig('/home/sascha/Desktop/Paper 2024/KW2.png', dpi=600)
# plt.savefig('/home/sascha/Desktop/Nextcloud/work/FENS2024/Abstract/results.png', dpi=600)
plt.show()

#%%
'''
    High ΔRT and low CI
'''
# complete_df_all[(complete_df_all['RT_diff_stt_day2'] > complete_df_all['RT_diff_stt_day2'].mean()) & 
#             (complete_df_all['CIspread_day2'] < complete_df_all['CIspread_day2'].mean())]

'''
    High ΔER and low CI
'''
# complete_df_all[(complete_df_all['ER_diff_stt_day2'] > complete_df_all['ER_diff_stt_day2'].mean()) & 
#             (complete_df_all['CIspread_day2'] < complete_df_all['CIspread_day2'].mean())]


'''
    High ΔER & ΔRT and low CI
'''
df_day2 = complete_df_all[complete_df_all['day'] == 2]
test_df = df_day2[(df_day2['ER_diff_stt'] > df_day2['ER_diff_stt'].mean()) & 
            (df_day2['RT_diff_stt'] > df_day2['RT_diff_stt'].mean()) & 
            (df_day2['CIspread'] < df_day2['CIspread'].mean())]

'''
    High ΔER & ΔRT and low RI
'''
test_df = df_day2[(df_day2['ER_diff_stt'] > df_day2['ER_diff_stt'].mean()) & 
            (df_day2['RT_diff_stt'] > df_day2['RT_diff_stt'].mean()) & 
            (df_day2['RIspread'] < df_day2['RIspread'].mean())]

anal.violin(df_day2, param_names, model)
anal.violin(df_day2[df_day2['ID'].isin(test_df['ID'])], param_names, model)

#%%
'''
    Responses from posterior predictives
'''

predictive_choices = extra_storage[14]
obs_mask = extra_storage[15]

df_pp_day2 = utils.post_pred_sim_df(predictive_choices, obs_mask, model, num_agents, inf_mean_df, day = 2)


predictive_choices_day1 = extra_storage_day1[14]
obs_mask_day1 = extra_storage_day1[15]

df_pp_day1 = utils.post_pred_sim_df(predictive_choices_day1, obs_mask_day1, model_day1, num_agents, inf_mean_df_day1, day = 1)

# sim_group_behav_df['post_pred_GD'] = sim_group_behav_df.apply(lambda row: print(row['group']))

#%%
'''
    Plot the predictive choices
'''

utils.plot_hpcf(complete_df_all, title='Experiment')

hpcf_pp_day1 = utils.compute_hpcf(df_pp_day2)
hpcf_pp_day1['day'] = 1
hpcf_pp_day2 = utils.compute_hpcf(df_pp_day1)
hpcf_pp_day2['day'] = 2

hpcf_pp_df_all = pd.concat((hpcf_pp_day1, hpcf_pp_day2), ignore_index = True)

utils.plot_hpcf(hpcf_pp_df_all, title=f'{model}', post_pred = True)

#%%
df_day2new = df_day2[df_day2['trialsequence']> 10]
df_day2new = df_day2new[df_day2new['choices'] != -2]
df_day2new = df_day2new[df_day2new['choices'] != -1]
df_day2new = df_day2new.reset_index(drop = True)

df_day2new = df_day2new.drop(['day', 'outcomes', 'choices_GD', 'ID', 'blockidx', 'blocktype'], axis = 1)

for rowidx in range(len(df_day2new)):
    if not df_day2new.iloc[rowidx]['post_pred_GD'] != df_day2new.iloc[rowidx]['post_pred'] and not df_day2new.iloc[rowidx]['post_pred_GD'] != (1-df_day2new.iloc[rowidx]['post_pred']):
        print(f"rowidx {rowidx}")
        
#%%
'''
    p-values for day 2
'''

"Compute ER variance in R-condition"

IDs = []
df = expdata_df[expdata_df['choices'] != -1]
stepsize = 480//8
ER_stt_forvar = np.zeros((num_agents, 32))

ag_idx = 0
for ID in df['ID'].unique():
    IDs.append(ID)
    df_ag = df[df['ID'] == ID]
    
    partidx = 0
    for blidx in range(6, 14):
        if df_ag[df_ag['blockidx'] == blidx]['blocktype'].unique() == 1: #random
            trialidx_from = df_ag[df_ag['blockidx'] == blidx]['trialidx'].min()
            trialidx_to = df_ag[df_ag['blockidx'] == blidx]['trialidx'].max()
            
            for tidx in range(trialidx_from, trialidx_to, stepsize):
                partial_df = df_ag[(df_ag['blockidx'] == blidx) & 
                                   (df_ag['trialidx'] >= tidx) & 
                                   (df_ag['trialidx'] < tidx+stepsize)]
                
                num_stt = len(partial_df[partial_df['trialsequence']<10])
                num_errs = len(partial_df[partial_df['choices']==-2])
                
                ER_stt_forvar[ag_idx, partidx] = num_errs/ num_stt
                partidx += 1
                
    ag_idx += 1
            
    
#%%
'''
    Find weak and strong sequence learners.    

    Response Strategies
    Habitual Responders
    GD Responders
    Modulators
'''

seqlearners_df_day1, notseqlearners_df_day1 = anal.find_seqlearners(expdata_df_day1,
                                                          complete_df_all, 
                                                          day = 1,
                                                          correctp = True)


seqlearners_df_day2, notseqlearners_df_day2 = anal.find_seqlearners(expdata_df,
                                                          complete_df_all, 
                                                          day = 2,
                                                          correctp = True)

seqall = pd.concat((seqlearners_df_day1, seqlearners_df_day2))
notseqall = pd.concat((notseqlearners_df_day1, notseqlearners_df_day2))

seq_plotdf_day2 = utils.plot_grouplevel(expdata_df[expdata_df['ID'].isin(seqlearners_df_day2['ID'].unique())], plot_single = False)
seq_plotdf_day1 = utils.plot_grouplevel(expdata_df_day1[expdata_df_day1['ID'].isin(seqlearners_df_day1['ID'].unique())], plot_single = False)

notseq_plotdf_day2 = utils.plot_grouplevel(expdata_df[expdata_df['ID'].isin(notseqlearners_df_day2['ID'].unique())], plot_single = False)
notseq_plotdf_day1 = utils.plot_grouplevel(expdata_df_day1[expdata_df_day1['ID'].isin(notseqlearners_df_day1['ID'].unique())], plot_single = False)


notseq_plotdf_all = pd.concat((notseq_plotdf_day1, notseq_plotdf_day2))
seq_plotdf_all = pd.concat((seq_plotdf_day1, seq_plotdf_day2))

seq_plotdf_all['type'] = 'strong sequence learner'
notseq_plotdf_all['type'] = 'weak sequence learner'
seq_combined = pd.concat((seq_plotdf_all, notseq_plotdf_all))

'''
    Plot as barplots
    Bottom: Weak sequence learner vs 
    Top: strong sequence learners
    y: HRC
'''
# colors1 = ['#67b798', '#BE54C6', '#7454C7'] # random, congruent, incongruent]
colors1 = {'Random': '#67b798', 'Congruent': '#BE54C6', 'Incongruent': '#7454C7'}
# colors2 = ['#67b798', '#bd97c6'] # random, fix
colors2 = {'Random': '#67b798', 'Fix': '#bd97c6'} # random, fix

seq_plotdf_all['choices_GD'] = seq_plotdf_all['choices_GD'].map(lambda x: x*100)
if 0:
    fig, ax = plt.subplots(1,2, sharey = True, sharex=True, figsize = (15, 6))
    sns.barplot(ax = ax[0],
                data = seq_plotdf_all,
                x = 'day',
                y = 'choices_GD',
                hue = 'DTT Types',
                palette = colors1,
                errorbar = ('se', 1))
    
    ax[0].set_title('strong sequence learners', fontsize = 15)
    ax[0].tick_params(axis='x', labelsize=15)  # For x-axis tick labels
    ax[0].tick_params(axis='y', labelsize=15)  # For x-axis tick labels
    ax[0].set_xlabel('Day', fontsize = 18)
    ax[0].set_ylabel('HRC (%)', fontsize = 18)
    ax[0].set_ylim([60, 100])
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title= 'DTT Type')
    
    notseq_plotdf_all['choices_GD'] = notseq_plotdf_all['choices_GD'].map(lambda x: x*100)
    sns.barplot(ax = ax[1],
                data = notseq_plotdf_all, 
                x = 'day',
                y = 'choices_GD',
                hue = 'DTT Types',
                palette = colors1,
                errorbar = ('se', 1))
    
    ax[1].set_title('weak sequence learners', fontsize = 15)
    ax[1].set_xlabel('Day', fontsize = 18)
    ax[1].tick_params(axis='x', labelsize=15)  # For x-axis tick labels
    ax[1].tick_params(axis='y', labelsize=15)  # For x-axis tick labels
    ax[1].set_ylabel('HRC (%)', fontsize = 18)
    ax[1].set_ylim([60, 100])
    ax[1].get_legend().set_visible(False)
    plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig2a/strong_vs_weak_seqlearners_python.svg', bbox_inches = 'tight')
    plt.show()

seq_combined['choices_GD'] = seq_combined['choices_GD']*100
notseq_plotdf_all['choices_GD'] = notseq_plotdf_all['choices_GD'].map(lambda x: x*100)
'''
    Fig for paper
    2 subplots
    left: Day 1
        x : strong / weak sequence learners
        y : HRC
    right: Day 2
'''
fig, ax = plt.subplots(1,2, sharey = True, sharex=True, figsize = (15, 6))
hue_order = ['Random', 'Congruent', 'Incongruent']
sns.barplot(ax = ax[0],
            data = seq_combined[seq_combined['day'] == 1],
            x = 'type',
            y = 'choices_GD',
            hue = 'DTT Types',
            hue_order = hue_order,
            palette = colors1,
            errorbar = ('se', 1))

ax[0].set_title('Day 1', fontsize = 15)
ax[0].tick_params(axis='x', labelsize=15)  # For x-axis tick labels
ax[0].tick_params(axis='y', labelsize=15)  # For x-axis tick labels
ax[0].set_xlabel('', fontsize = 18)
ax[0].set_ylabel('HRC (%)', fontsize = 18)
ax[0].set_ylim([60, 100])
ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title= 'DTT Type')

sns.barplot(ax = ax[1],
            data = seq_combined[seq_combined['day'] == 2], 
            x = 'type',
            y = 'choices_GD',
            hue = 'DTT Types',
            hue_order = hue_order,
            palette = colors1,
            errorbar = ('se', 1))

ax[1].set_title('Day 2', fontsize = 15)
ax[1].set_xlabel('', fontsize = 18)
ax[1].tick_params(axis='x', labelsize=15)  # For x-axis tick labels
ax[1].tick_params(axis='y', labelsize=15)  # For x-axis tick labels
ax[1].set_ylabel('HRC (%)', fontsize = 18)
ax[1].set_ylim([60, 100])
ax[1].get_legend().set_visible(False)
plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig2a/strong_vs_weak_seqlearners_python.svg', bbox_inches = 'tight')
plt.show()

complete_df_all_day1 = complete_df_all[complete_df_all['day'] == 1]
complete_df_all_day2 = complete_df_all[complete_df_all['day'] == 2]

complete_df_all['strong_seq_day1'] = None
complete_df_all['strong_seq_day2'] = None
for rowidx in range(len(complete_df_all)):
    if complete_df_all.loc[rowidx, 'ID'] in seq_plotdf_day1['ID'].unique():
        complete_df_all.loc[rowidx, 'strong_seq_day1']  = 'strong sequence learner'
        
    elif complete_df_all.loc[rowidx, 'ID'] in notseq_plotdf_day1['ID'].unique():
        complete_df_all.loc[rowidx, 'strong_seq_day1']  = 'weak sequence learner'
        
    if complete_df_all.loc[rowidx, 'ID'] in seq_plotdf_day2['ID'].unique():
        complete_df_all.loc[rowidx, 'strong_seq_day2']  = 'strong sequence learner'
        
    elif complete_df_all.loc[rowidx, 'ID'] in notseq_plotdf_day2['ID'].unique():
        complete_df_all.loc[rowidx, 'strong_seq_day2']  = 'weak sequence learner'


'''
    Figure
    Scatterplots
    x: ΔRT
    y: ΔER
    color: weak & strong sequence learners
'''
fig, ax = plt.subplots(1, 2, sharey = True, sharex = True, figsize = (15, 6))

"Day 1"
plot_df_all = complete_df_all.copy()
plot_df_all['ER_diff_stt'] = plot_df_all['ER_diff_stt']*100
sns.scatterplot(x='RT_diff_stt',
                y='ER_diff_stt',
                hue = 'strong_seq_day1',
                ax = ax[0],
                data = plot_df_all[plot_df_all['day'] == 1])

ax[0].axvline(0, color = 'k')
ax[0].axhline(0, color = 'k')
ax[0].legend(title='')
ax[0].set_title('Day 1', fontsize = 15)
ax[0].tick_params(axis='x', labelsize=15)  # For x-axis tick labels
ax[0].tick_params(axis='y', labelsize=15)  # For x-axis tick labels
ax[0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
ax[0].set_ylabel(r'$\Delta$ER (p.p)', fontsize = 20)
ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

"Day 2"
sns.scatterplot(x='RT_diff_stt',
                y='ER_diff_stt',
                ax = ax[1],
                hue = 'strong_seq_day2',
                data = plot_df_all[plot_df_all['day'] == 2])

ax[1].axvline(0, color = 'k')
ax[1].axhline(0, color = 'k')
ax[1].legend(title='')
ax[1].set_title('Day 2', fontsize = 15)
ax[1].tick_params(axis='x', labelsize=15)  # For x-axis tick labels
ax[1].tick_params(axis='y', labelsize=15)  # For x-axis tick labels
ax[1].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
ax[1].set_ylabel(r'$\Delta$ER (p.p)', fontsize = 20)
ax[1].set_xlim([-15, 40])
ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.legend(fontsize = 16, loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig2a/strong_vs_weak_seqlearners_RT_ER_python.svg', bbox_inches = 'tight')
plt.show()

dfseq = seq_plotdf_all.loc[:, ['ID', 'DTT Types', 'choices_GD']].groupby(['ID', 'DTT Types'], as_index = False).mean()

#%%

print("Hypothesis Tests for pevious figure.")
"======= Day 1 ========="


print("Seqlearners Day 1")
seq_plotdf_all_day1 = seq_plotdf_all[seq_plotdf_all['day'] == 1].loc[:, ['ID', 'DTT Types', 'choices_GD']].groupby(['ID', 'DTT Types'], as_index = False).mean()
incong_day1 = seq_plotdf_all_day1[seq_plotdf_all_day1['DTT Types'] == 'Incongruent'].sort_values(by='ID').reset_index(drop=True)
cong_day1 = seq_plotdf_all_day1[seq_plotdf_all_day1['DTT Types'] == 'Congruent'].sort_values(by='ID').reset_index(drop=True)
rand_day1 = seq_plotdf_all_day1[seq_plotdf_all_day1['DTT Types'] == 'Random'].sort_values(by='ID').reset_index(drop=True)

print(len(seq_plotdf_all_day1['ID'].unique()))

t,p = scipy.stats.ttest_rel(cong_day1['choices_GD'], incong_day1['choices_GD'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(cong_day1['choices_GD'], rand_day1['choices_GD'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(incong_day1['choices_GD'], rand_day1['choices_GD'])
print(f"t={t}, p={p}")



print("Notseqlearners Day 1")
notseq_plotdf_all_day1 = notseq_plotdf_all[notseq_plotdf_all['day'] == 1].loc[:, ['ID', 'DTT Types', 'choices_GD']].groupby(['ID', 'DTT Types'], as_index = False).mean()
incong_day1 = notseq_plotdf_all_day1[notseq_plotdf_all_day1['DTT Types'] == 'Incongruent'].sort_values(by='ID').reset_index(drop=True)
cong_day1 = notseq_plotdf_all_day1[notseq_plotdf_all_day1['DTT Types'] == 'Congruent'].sort_values(by='ID').reset_index(drop=True)
rand_day1 = notseq_plotdf_all_day1[notseq_plotdf_all_day1['DTT Types'] == 'Random'].sort_values(by='ID').reset_index(drop=True)

print(len(notseq_plotdf_all_day1['ID'].unique()))


t,p = scipy.stats.ttest_rel(cong_day1['choices_GD'], incong_day1['choices_GD'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(cong_day1['choices_GD'], rand_day1['choices_GD'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(incong_day1['choices_GD'], rand_day1['choices_GD'])
print(f"t={t}, p={p}")

print("Day 1")
print(len(seq_plotdf_all_day1['ID'].unique()))
print(len(notseq_plotdf_all_day1['ID'].unique()))

"======= Day 2 ========="

print("Seqlearners Day 2")
seq_plotdf_all_day2 = seq_plotdf_all[seq_plotdf_all['day'] == 2].loc[:, ['ID', 'DTT Types', 'choices_GD']].groupby(['ID', 'DTT Types'], as_index = False).mean()
incong_day2 = seq_plotdf_all_day2[seq_plotdf_all_day2['DTT Types'] == 'Incongruent'].sort_values(by='ID').reset_index(drop=True)
cong_day2 = seq_plotdf_all_day2[seq_plotdf_all_day2['DTT Types'] == 'Congruent'].sort_values(by='ID').reset_index(drop=True)
rand_day2 = seq_plotdf_all_day2[seq_plotdf_all_day2['DTT Types'] == 'Random'].sort_values(by='ID').reset_index(drop=True)

print(len(seq_plotdf_all_day2['ID'].unique()))


t,p = scipy.stats.ttest_rel(cong_day2['choices_GD'], incong_day2['choices_GD'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(cong_day2['choices_GD'], rand_day2['choices_GD'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(incong_day2['choices_GD'], rand_day2['choices_GD'])
print(f"t={t}, p={p}")

print("Notseqlearners Day 2")
notseq_plotdf_all_day2 = notseq_plotdf_all[notseq_plotdf_all['day'] == 2].loc[:, ['ID', 'DTT Types', 'choices_GD']].groupby(['ID', 'DTT Types'], as_index = False).mean()
incong_day2 = notseq_plotdf_all_day2[notseq_plotdf_all_day2['DTT Types'] == 'Incongruent'].sort_values(by='ID').reset_index(drop=True)
cong_day2 = notseq_plotdf_all_day2[notseq_plotdf_all_day2['DTT Types'] == 'Congruent'].sort_values(by='ID').reset_index(drop=True)
rand_day2 = notseq_plotdf_all_day2[notseq_plotdf_all_day2['DTT Types'] == 'Random'].sort_values(by='ID').reset_index(drop=True)

print(len(notseq_plotdf_all_day2['ID'].unique()))

t,p = scipy.stats.ttest_rel(cong_day2['choices_GD'], rand_day2['choices_GD'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(cong_day2['choices_GD'], incong_day2['choices_GD'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(incong_day2['choices_GD'], rand_day2['choices_GD'])
print(f"t={t}, p={p}")

print("Day 2")
print(len(seq_plotdf_all_day2['ID'].unique()))
print(len(notseq_plotdf_all_day2['ID'].unique()))

#%%
'''
    Among strong sequence learners, find habitual learners, GD learners, modulators, and others.
'''

%matplotlib inline
habitual_df, GD_df, modulators_df, antimods_df = anal.find_strategies(expdata_df[expdata_df['ID'].isin(seq_plotdf_all_day2['ID'].unique())], 
                                                         day = 2, 
                                                         num_sections = 4,
                                                         plot_single = False,
                                                         correctp = False)

print("Day 2")
print(len(habitual_df['ID'].unique()))
print(len(GD_df['ID'].unique()))
print(len(modulators_df['ID'].unique()))
print(len(antimods_df['ID'].unique()))

haball = complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['ID'].isin(habitual_df[habitual_df['day']==2]['ID'].unique()))].loc[:, ['CRspread', 'RIspread']]
r,p = scipy.stats.spearmanr(haball['CRspread'], haball['RIspread'])
print(f"r={r}, p={p}")

GDall = complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['ID'].isin(GD_df[GD_df['day']==2]['ID'].unique()))]
r,p = scipy.stats.spearmanr(GDall['CRspread'], GDall['RIspread'])
print(f"r={r}, p={p}")

modulatoprsall = complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['ID'].isin(modulators_df[modulators_df['day']==2]['ID'].unique()))]
r,p = scipy.stats.spearmanr(modulatoprsall['CRspread'], modulatoprsall['RIspread'])
print(f"r={r}, p={p}")

antimods_all = complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['ID'].isin(antimods_df[antimods_df['day']==2]['ID'].unique()))]

'''
    Barplots
    Left: Habitual
    Middle: GD suppressers
    Right: GD modulators
    y: HRC
'''

habitual_plot_df = habitual_df[habitual_df['day']==2].loc[:, ['ID', 'DTT Types', 'choices_GD']]#
habitual_plot_df['choices_GD'] = habitual_plot_df['choices_GD']*100
habitual_plot_df['Type'] = 'Habitual'
fig, ax = plt.subplots(1,4, sharey = True, sharex=True, figsize = (15, 6))
sns.barplot(ax = ax[0],
            data = habitual_plot_df,
            # x = 'DTT Types',
            y = 'choices_GD',
            hue = 'DTT Types',
            palette = colors1,
            errorbar = ('se', 1))

ax[0].set_title('Habitual Responders', fontsize = 12)
ax[0].tick_params(axis='x', labelsize=12)  # For x-axis tick labels
ax[0].set_xlabel('Day', fontsize = 15)
ax[0].set_ylabel('HRC (%)', fontsize = 15)
ax[0].set_ylim([60, 100])
ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title= 'DTT Type')

pGD_plot_df = GD_df[GD_df['day']==2].loc[:, ['ID', 'DTT Types', 'choices_GD']]
pGD_plot_df['choices_GD'] = pGD_plot_df['choices_GD']*100
pGD_plot_df['Type'] = 'Suppresser'
sns.barplot(ax = ax[1],
            data = pGD_plot_df, 
            # x = 'day',
            y = 'choices_GD',
            hue = 'DTT Types',
            palette = colors1,
            errorbar = ('se', 1))

ax[1].set_title('Purely Goal-Directed Responders', fontsize = 12)
ax[1].set_xlabel('Day', fontsize = 15)
ax[1].set_ylabel('HRC (%)', fontsize = 15)
ax[1].set_ylim([60, 100])
ax[1].get_legend().set_visible(False)

GDm_plot_df = modulators_df[modulators_df['day']==2].loc[:, ['ID', 'DTT Types', 'choices_GD']]
GDm_plot_df['choices_GD'] = GDm_plot_df['choices_GD'].map(lambda x: x*100)
GDm_plot_df['Type'] = 'Exploiter'
sns.barplot(ax = ax[2],
            data = GDm_plot_df, 
            # x = 'day',
            y = 'choices_GD',
            hue = 'DTT Types',
            palette = colors1,
            errorbar = ('se', 1))

ax[2].set_title('Goal-directed exploiters', fontsize = 12)
ax[2].set_xlabel('Day', fontsize = 15)
ax[2].set_ylabel('HRC (%)', fontsize = 15)
ax[2].set_ylim([60, 100])
ax[2].get_legend().set_visible(False)

anti_df = antimods_df[antimods_df['day']==2].loc[:, ['ID', 'DTT Types', 'choices_GD']]
anti_df['choices_GD'] = anti_df['choices_GD']*100
anti_df['Type'] = 'Anti-GD'
sns.barplot(ax = ax[3],
            data = anti_df, 
            # x = 'day',
            y = 'choices_GD',
            hue = 'DTT Types',
            palette = colors1,
            errorbar = ('se', 1))

ax[3].set_title('Anti', fontsize = 12)
ax[3].set_xlabel('Day', fontsize = 15)
ax[3].set_ylabel('HRC (%)', fontsize = 15)
ax[3].set_ylim([60, 100])
ax[3].get_legend().set_visible(False)
# plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig2b/res_fig2b_python.svg', bbox_inches = 'tight')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

dfplot = pd.concat((habitual_plot_df, pGD_plot_df, GDm_plot_df, anti_df))
fig, ax = plt.subplots()
sns.barplot(ax = ax,
            data = dfplot,
            x = 'Type',
            order = ['Suppresser', 'Exploiter', 'Habitual', 'Anti-GD'],
            y = 'choices_GD',
            hue = 'DTT Types',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))

# ax[2].set_title('Goal-directed exploiters', fontsize = 12)
# ax[2].set_xlabel('Day', fontsize = 15)
ax.set_ylabel('HRC (%)', fontsize = 15)
ax.set_ylim([60, 100])
# ax[2].get_legend().set_visible(False)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig2b/res_fig2b_a_python.svg', bbox_inches = 'tight')
plt.show()

print(len(dfplot[dfplot['Type'] == 'Habitual']['ID'].unique()))
print(len(dfplot[dfplot['Type'] == 'Suppresser']['ID'].unique()))
print(len(dfplot[dfplot['Type'] == 'Exploiter']['ID'].unique()))
print(len(dfplot[dfplot['Type'] == 'Anti-GD']['ID'].unique()))

"=== Supp vs Expl ==="
t,p = scipy.stats.ttest_ind(np.array(dfplot[(dfplot['Type'] == 'Suppresser') & (dfplot['DTT Types'] == 'Congruent')]['choices_GD'], dtype='float'),
np.array(dfplot[(dfplot['Type'] == 'Exploiter') & (dfplot['DTT Types'] == 'Congruent')]['choices_GD'], dtype='float'))
print(f"t={t}, p={p}")

"=== Supp vs habitual ==="
t,p = scipy.stats.ttest_ind(np.array(dfplot[(dfplot['Type'] == 'Suppresser') & (dfplot['DTT Types'] == 'Congruent')]['choices_GD'], dtype='float'),
np.array(dfplot[(dfplot['Type'] == 'Habitual') & (dfplot['DTT Types'] == 'Congruent')]['choices_GD'], dtype='float'))
print(f"t={t}, p={p}")


"=== Supp vs Anti ==="
t,p = scipy.stats.ttest_ind(np.array(dfplot[(dfplot['Type'] == 'Suppresser') & (dfplot['DTT Types'] == 'Congruent')]['choices_GD'], dtype='float'),
np.array(dfplot[(dfplot['Type'] == 'Anti-GD') & (dfplot['DTT Types'] == 'Congruent')]['choices_GD'], dtype='float'))
print(f"t={t}, p={p}")

"=== Exploiter vs Habitual ==="
t,p = scipy.stats.ttest_ind(np.array(dfplot[(dfplot['Type'] == 'Exploiter') & (dfplot['DTT Types'] == 'Congruent')]['choices_GD'], dtype='float'),
np.array(dfplot[(dfplot['Type'] == 'Habitual') & (dfplot['DTT Types'] == 'Congruent')]['choices_GD'], dtype='float'))
print(f"t={t}, p={p}")

"=== Exploiter vs Anti ==="
t,p = scipy.stats.ttest_ind(np.array(dfplot[(dfplot['Type'] == 'Exploiter') & (dfplot['DTT Types'] == 'Congruent')]['choices_GD'], dtype='float'),
np.array(dfplot[(dfplot['Type'] == 'Anti-GD') & (dfplot['DTT Types'] == 'Congruent')]['choices_GD'], dtype='float'))
print(f"t={t}, p={p}")

"=== Habitual vs Anti ==="
t,p = scipy.stats.ttest_ind(np.array(dfplot[(dfplot['Type'] == 'Habitual') & (dfplot['DTT Types'] == 'Congruent')]['choices_GD'], dtype='float'),
np.array(dfplot[(dfplot['Type'] == 'Anti-GD') & (dfplot['DTT Types'] == 'Congruent')]['choices_GD'], dtype='float'))
print(f"t={t}, p={p}")

"=== Habitual ==="
t,p = scipy.stats.ttest_rel(np.array(habitual_plot_df[habitual_plot_df['DTT Types']=='Congruent']['choices_GD']), 
                            np.array(habitual_plot_df[habitual_plot_df['DTT Types']=='Random']['choices_GD']))
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(np.array(habitual_plot_df[habitual_plot_df['DTT Types']=='Congruent']['choices_GD']), 
                            np.array(habitual_plot_df[habitual_plot_df['DTT Types']=='Incongruent']['choices_GD']))
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(np.array(habitual_plot_df[habitual_plot_df['DTT Types']=='Random']['choices_GD']), 
                            np.array(habitual_plot_df[habitual_plot_df['DTT Types']=='Incongruent']['choices_GD']))
print(f"t={t}, p={p}")

# del habitual_plot_df

"=== purely GD ==="
t,p = scipy.stats.ttest_rel(np.array(pGD_plot_df[pGD_plot_df['DTT Types']=='Congruent']['choices_GD']), 
                            np.array(pGD_plot_df[pGD_plot_df['DTT Types']=='Random']['choices_GD']))
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(np.array(pGD_plot_df[pGD_plot_df['DTT Types']=='Congruent']['choices_GD']), 
                            np.array(pGD_plot_df[pGD_plot_df['DTT Types']=='Incongruent']['choices_GD']))
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(np.array(pGD_plot_df[pGD_plot_df['DTT Types']=='Random']['choices_GD']), 
                            np.array(pGD_plot_df[pGD_plot_df['DTT Types']=='Incongruent']['choices_GD']))
print(f"t={t}, p={p}")

# del pGD_plot_df

"=== GD-modulators ==="
t,p = scipy.stats.ttest_rel(np.array(GDm_plot_df[GDm_plot_df['DTT Types']=='Congruent']['choices_GD']), 
                            np.array(GDm_plot_df[GDm_plot_df['DTT Types']=='Random']['choices_GD']))
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(np.array(GDm_plot_df[GDm_plot_df['DTT Types']=='Congruent']['choices_GD']), 
                            np.array(GDm_plot_df[GDm_plot_df['DTT Types']=='Incongruent']['choices_GD']))
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(np.array(GDm_plot_df[GDm_plot_df['DTT Types']=='Random']['choices_GD']), 
                            np.array(GDm_plot_df[GDm_plot_df['DTT Types']=='Incongruent']['choices_GD']))
print(f"t={t}, p={p}")

"=== Habitual vs purely GD ==="
t,p = scipy.stats.ttest_ind(np.array(pGD_plot_df[pGD_plot_df['DTT Types']=='Congruent']['choices_GD'], dtype = 'float'), 
                            np.array(habitual_plot_df[habitual_plot_df['DTT Types']=='Congruent']['choices_GD'], dtype = 'float'))
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_ind(np.array(pGD_plot_df[pGD_plot_df['DTT Types']=='Random']['choices_GD'], dtype = 'float'), 
                            np.array(habitual_plot_df[habitual_plot_df['DTT Types']=='Random']['choices_GD'], dtype = 'float'))
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_ind(np.array(pGD_plot_df[pGD_plot_df['DTT Types']=='Incongruent']['choices_GD'], dtype = 'float'), 
                            np.array(habitual_plot_df[habitual_plot_df['DTT Types']=='Incongruent']['choices_GD'], dtype = 'float'))
print(f"t={t}, p={p}")

"=== purely GD vs GD modulators ==="
t,p = scipy.stats.ttest_ind(np.array(pGD_plot_df[pGD_plot_df['DTT Types']=='Congruent']['choices_GD'], dtype = 'float'), 
                            np.array(GDm_plot_df[GDm_plot_df['DTT Types']=='Congruent']['choices_GD'], dtype = 'float'))
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_ind(np.array(pGD_plot_df[pGD_plot_df['DTT Types']=='Random']['choices_GD'], dtype = 'float'), 
                            np.array(GDm_plot_df[GDm_plot_df['DTT Types']=='Random']['choices_GD'], dtype = 'float'))
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_ind(np.array(pGD_plot_df[pGD_plot_df['DTT Types']=='Incongruent']['choices_GD'], dtype = 'float'), 
                            np.array(GDm_plot_df[GDm_plot_df['DTT Types']=='Incongruent']['choices_GD'], dtype = 'float'))
print(f"t={t}, p={p}")

"=== Habitual vs GD modulators ==="
t,p = scipy.stats.ttest_ind(np.array(habitual_plot_df[habitual_plot_df['DTT Types']=='Congruent']['choices_GD'], dtype = 'float'), 
                            np.array(GDm_plot_df[GDm_plot_df['DTT Types']=='Congruent']['choices_GD'], dtype = 'float'))
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_ind(np.array(habitual_plot_df[habitual_plot_df['DTT Types']=='Random']['choices_GD'], dtype = 'float'), 
                            np.array(GDm_plot_df[GDm_plot_df['DTT Types']=='Random']['choices_GD'], dtype = 'float'))
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_ind(np.array(habitual_plot_df[habitual_plot_df['DTT Types']=='Incongruent']['choices_GD'], dtype = 'float'), 
                            np.array(GDm_plot_df[GDm_plot_df['DTT Types']=='Incongruent']['choices_GD'], dtype = 'float'))
print(f"t={t}, p={p}")

#%%
'''
    Two-way ANOVA for paper
    HRC ~ DTT type, Day
'''

hpcf_df_day2 = utils.compute_hpcf(expdata_df)
hpcf_df_day2['day'] = 2
hpcf_df_day2 = hpcf_df_day2.loc[:, ['ID', 'hpcf_cong', 'hpcf_incong', 'hpcf_rand', 'day']].reset_index(drop=True)

hpcf_df_day1 = utils.compute_hpcf(expdata_df_day1)
hpcf_df_day1['day'] = 1
hpcf_df_day1 = hpcf_df_day1.loc[:, ['ID', 'hpcf_cong', 'hpcf_incong', 'hpcf_rand', 'day']].reset_index(drop=True)

hpcf_df_all = pd.concat((hpcf_df_day1, hpcf_df_day2))

anova_df = pd.melt(hpcf_df_all, id_vars = ['day', 'ID'], value_vars=['hpcf_cong', 'hpcf_incong', 'hpcf_rand'])
anova_df['day'] = anova_df['day'].map(lambda x: 'one' if x==1 else 'two')
# anova_df['bla'] = np.random.rand()

anova_df['value'] = anova_df['value'].astype(float)
anova_df = anova_df.rename(columns={'variable': 'DTT Type', 
                                    'value': 'HRC'})

# import statsmodels.api as sm
# from statsmodels.formula.api import ols

# # model = ols('value ~ variable * day', data=anova_df).fit()
# model = ols('value ~ C(variable) + C(day) + C(variable):C(day)', data=anova_df).fit()

# # Perform ANOVA and print the results
# anova_table = sm.stats.anova_lm(model, typ=2)
# print(anova_table)

# eta_squared = anova_table['sum_sq']['C(Group)'] / anova_table['sum_sq'].sum()

import pingouin as pg
aov = pg.rm_anova(dv = 'HRC',
                  within = ['DTT Type', 'day'], 
                  subject = 'ID', 
                  data = anova_df, 
                  detailed = True,
                  effsize = 'np2')

print(aov)
print(aov.loc[:, ['Source', 'ddof1', 'ddof2', 'F', 'p-unc', 'np2']])
# print("p-values")
# print("p-values")
# print(aov['p-unc'])
# print(aov['p-unc'])
# print("Effect sizes")
# print(aov['np2'])

#%%
'''
    Two-way ANOVA for paper
    ER ~ Condition, Day
'''

ER_all = complete_df_all.loc[:, ['ID',
                                 'day',
                                 'ER_stt_rand', 
                                   'ER_stt_seq']]

ER_all = ER_all.melt(id_vars=['ID', 'day'], value_vars=['ER_stt_rand', 'ER_stt_seq'])
ER_all['variable'] = ER_all['variable'].map(lambda x: "Fix" if x == 'ER_stt_seq' else
                                                'Random' if x == 'ER_stt_rand' else
                                                'None')
ER_all = ER_all.rename(columns={'variable': 'Condition',
                                'value': 'ER'})
import pingouin as pg
aov = pg.rm_anova(dv = 'ER', 
                  within = ['Condition', 'day'], 
                  subject = 'ID', 
                  data = ER_all, 
                  detailed = True,
                  effsize = 'np2')
print(aov.loc[:, ['Source', 'ddof1', 'ddof2', 'F', 'p-unc', 'np2']])


#%%
'''
    Two-way ANOVA for paper
    RT ~ Condition, Day
'''

RT_all = complete_df_all.loc[:, ['ID',
                                 'day',
                                 'RT_stt_rand', 
                                   'RT_stt_seq']]

RT_all = RT_all.melt(id_vars=['ID', 'day'], value_vars=['RT_stt_rand', 'RT_stt_seq'])
RT_all['variable'] = RT_all['variable'].map(lambda x: "Fix" if x == 'RT_stt_seq' else
                                                'Random' if x == 'RT_stt_rand' else
                                                'None')

RT_all = RT_all.rename(columns={'variable': 'Condition',
                                'value': 'RT'})

import pingouin as pg
aov = pg.rm_anova(dv = 'RT', 
                  within=['Condition', 'day'], 
                  subject = 'ID', 
                  data = RT_all, 
                  detailed = True,
                  effsize = 'np2')

print(aov.loc[:, ['Source', 'ddof1', 'ddof2', 'F', 'p-unc', 'np2']])

#%%
'''
    t-tests for paper
'''

"======= HRC ======="
"======= === Both days"

t,p = scipy.stats.ttest_rel(np.array(complete_df_all['hpcf_cong']), 
                          np.array(complete_df_all['hpcf_rand']))

print(np.array(complete_df_all['hpcf_cong']).mean())
print(np.array(complete_df_all['hpcf_rand']).mean())
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(np.array(complete_df_all['hpcf_rand']), 
                          np.array(complete_df_all['hpcf_incong']))

print(np.array(complete_df_all['hpcf_rand']).mean())
print(np.array(complete_df_all['hpcf_incong']).mean())
print(f"t={t}, p={p}")


"======= === Day 1"
# t,p = scipy.stats.ttest_rel(np.array(complete_df_all[complete_df_all['day']==1]['hpcf_cong']), 
#                           np.array(complete_df_all[complete_df_all['day']==1]['hpcf_incong']))

# print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(np.array(complete_df_all[complete_df_all['day']==1]['hpcf_cong']), 
                          np.array(complete_df_all[complete_df_all['day']==1]['hpcf_rand']))

print(np.array(complete_df_all[complete_df_all['day']==1]['hpcf_cong']).mean())
print(np.array(complete_df_all[complete_df_all['day']==1]['hpcf_rand']).mean())
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(np.array(complete_df_all[complete_df_all['day']==1]['hpcf_rand']), 
                          np.array(complete_df_all[complete_df_all['day']==1]['hpcf_incong']))

print(np.array(complete_df_all[complete_df_all['day']==1]['hpcf_rand']).mean())
print(np.array(complete_df_all[complete_df_all['day']==1]['hpcf_incong']).mean())
print(f"t={t}, p={p}")


"======= === Day 2"
# t,p = scipy.stats.ttest_rel(np.array(complete_df_all[complete_df_all['day']==2]['hpcf_cong']), 
#                           np.array(complete_df_all[complete_df_all['day']==2]['hpcf_incong']))

# print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(np.array(complete_df_all[complete_df_all['day']==2]['hpcf_cong']), 
                          np.array(complete_df_all[complete_df_all['day']==2]['hpcf_rand']))

print(np.array(complete_df_all[complete_df_all['day']==2]['hpcf_cong']).mean())
print(np.array(complete_df_all[complete_df_all['day']==2]['hpcf_rand']).mean())
print(f"t={t}, p={p}")


t,p = scipy.stats.ttest_rel(np.array(complete_df_all[complete_df_all['day']==2]['hpcf_rand']), 
                          np.array(complete_df_all[complete_df_all['day']==2]['hpcf_incong']))

print(np.array(complete_df_all[complete_df_all['day']==2]['hpcf_rand']).mean())
print(np.array(complete_df_all[complete_df_all['day']==2]['hpcf_incong']).mean())
print(f"t={t}, p={p}")

"======= RT & ER ======="

t,p = scipy.stats.ttest_rel(np.array(complete_df_all[complete_df_all['day']==1]['RT_stt_seq']), 
                          np.array(complete_df_all[complete_df_all['day']==1]['RT_stt_rand']))

print(np.array(complete_df_all[complete_df_all['day']==1]['RT_stt_seq']).mean())
print(np.array(complete_df_all[complete_df_all['day']==1]['RT_stt_rand']).mean())
print(f"t={t}, p={p}")


t,p = scipy.stats.ttest_rel(np.array(complete_df_all[complete_df_all['day']==1]['ER_stt_seq']), 
                          np.array(complete_df_all[complete_df_all['day']==1]['ER_stt_rand']))

print(np.array(complete_df_all[complete_df_all['day']==1]['ER_stt_seq']).mean())
print(np.array(complete_df_all[complete_df_all['day']==1]['ER_stt_rand']).mean())
print(f"t={t}, p={p}")


t,p = scipy.stats.ttest_rel(np.array(complete_df_all[complete_df_all['day']==2]['RT_stt_seq']), 
                          np.array(complete_df_all[complete_df_all['day']==2]['RT_stt_rand']))

print(np.array(complete_df_all[complete_df_all['day']==2]['RT_stt_seq']).mean())
print(np.array(complete_df_all[complete_df_all['day']==2]['RT_stt_rand']).mean())
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(np.array(complete_df_all[complete_df_all['day']==2]['ER_stt_seq']), 
                          np.array(complete_df_all[complete_df_all['day']==2]['ER_stt_rand']))

print(np.array(complete_df_all[complete_df_all['day']==2]['ER_stt_seq']).mean())
print(np.array(complete_df_all[complete_df_all['day']==2]['ER_stt_rand']).mean())
print(f"t={t}, p={p}")

#%%
'''
    Fig
    A: HRC Day 1 & Day 2
    B: ER Day 1 & Day 2
    C: RT Day 1 & Day 2
'''

hpcf_all = complete_df_all.loc[:, ['ID',
                                   'day',
                                   'hpcf_cong', 
                                   'hpcf_incong',
                                   'hpcf_rand']]

hpcf_all = hpcf_all.melt(id_vars=['ID', 'day'], value_vars=['hpcf_cong', 'hpcf_incong', 'hpcf_rand'])
hpcf_all = hpcf_all.rename(columns={'variable': 'DTT Type', 'value': 'HRC'})
hpcf_all['DTT Type'] = hpcf_all['DTT Type'].map(lambda x: "Congruent" if x == 'hpcf_cong' else
                                                'Incongruent' if x == 'hpcf_incong' else
                                                'Random' if x == 'hpcf_rand' else
                                                'None')

colors1 = ['#BE54C6', '#7454C7', '#67b798', ] # random, congruent, incongruent]
colors2 = ['#67b798', '#bd97c6'] # random, fix
hpcf_all['HRC'] = hpcf_all['HRC']*100
fig, ax = plt.subplots(1, 3, figsize = (20, 10))
sns.barplot(data = hpcf_all,
            x = 'day',
            y = 'HRC',
            hue = 'DTT Type',
            palette = colors1,
            errorbar=('se', 1),
            ax = ax[0])

ax[0].legend(title="", fontsize = 14)
ax[0].set_ylabel("HRC (%)", fontsize = 20)
ax[0].set_ylim([60, 100])
ax[0].set_xlabel("Day", fontsize = 20)
ax[0].tick_params(axis='both', labelsize=14)

ER_all['ER'] = ER_all['ER']*100
sns.barplot(data = ER_all,
            x = 'day',
            y = 'ER',
            hue = 'Condition',
            palette = colors2,
            errorbar=('se', 1),
            ax = ax[1])


ax[1].legend(title="", fontsize = 14)
ax[1].set_ylabel("ER STT (%)", fontsize = 20)
ax[1].set_ylim([3, 9])
ax[1].set_xlabel("Day", fontsize = 20)
ax[1].tick_params(axis='both', labelsize=14)

sns.barplot(data = RT_all,
            x = 'day',
            y = 'RT',
            hue = 'Condition',
            palette = colors2,
            errorbar=('se', 1),
            ax = ax[2])

ax[2].legend(title="", fontsize = 14)
ax[2].set_ylabel("RT STT (ms)", fontsize = 20)
ax[2].set_ylim([320, 420])
ax[2].set_xlabel("Day", fontsize = 20)
ax[2].tick_params(axis='both', labelsize=14)
plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig0/res_fig0_python.svg')
plt.show()

"=== Analyses ==="
"HPCF"
"=== Day 1 ==="
t,p = scipy.stats.ttest_rel(np.array(hpcf_all[(hpcf_all['day']==1) & (hpcf_all['DTT Type'] == 'Congruent')]['HRC']),
                            np.array(hpcf_all[(hpcf_all['day']==1) & (hpcf_all['DTT Type'] == 'Random')]['HRC']))
print(f"t={t}, p={p}.")

t,p = scipy.stats.ttest_rel(np.array(hpcf_all[(hpcf_all['day']==1) & (hpcf_all['DTT Type'] == 'Congruent')]['HRC']),
                            np.array(hpcf_all[(hpcf_all['day']==1) & (hpcf_all['DTT Type'] == 'Incongruent')]['HRC']))
print(f"t={t}, p={p}.")

t,p = scipy.stats.ttest_rel(np.array(hpcf_all[(hpcf_all['day']==1) & (hpcf_all['DTT Type'] == 'Random')]['HRC']),
                            np.array(hpcf_all[(hpcf_all['day']==1) & (hpcf_all['DTT Type'] == 'Incongruent')]['HRC']))
print(f"t={t}, p={p}.")

"=== Day 2 ==="
t,p = scipy.stats.ttest_rel(np.array(hpcf_all[(hpcf_all['day']==2) & (hpcf_all['DTT Type'] == 'Congruent')]['HRC']),
                            np.array(hpcf_all[(hpcf_all['day']==2) & (hpcf_all['DTT Type'] == 'Random')]['HRC']))
print(f"t={t}, p={p}.")

t,p = scipy.stats.ttest_rel(np.array(hpcf_all[(hpcf_all['day']==2) & (hpcf_all['DTT Type'] == 'Congruent')]['HRC']),
                            np.array(hpcf_all[(hpcf_all['day']==2) & (hpcf_all['DTT Type'] == 'Incongruent')]['HRC']))
print(f"t={t}, p={p}.")

t,p = scipy.stats.ttest_rel(np.array(hpcf_all[(hpcf_all['day']==2) & (hpcf_all['DTT Type'] == 'Random')]['HRC']),
                            np.array(hpcf_all[(hpcf_all['day']==2) & (hpcf_all['DTT Type'] == 'Incongruent')]['HRC']))
print(f"t={t}, p={p}.")

"ER"
"=== Day 1 ==="
t,p = scipy.stats.ttest_rel(np.array(ER_all[(ER_all['day']==1) & (ER_all['Condition'] == 'Random')]['ER']),
                            np.array(ER_all[(ER_all['day']==1) & (ER_all['Condition'] == 'Fix')]['ER']))
print(f"t={t}, p={p}.")

"=== Day 2 ==="
t,p = scipy.stats.ttest_rel(np.array(ER_all[(ER_all['day']==2) & (ER_all['Condition'] == 'Random')]['ER']),
                            np.array(ER_all[(ER_all['day']==2) & (ER_all['Condition'] == 'Fix')]['ER']))
print(f"t={t}, p={p}.")

"RT"
"=== Day 1 ==="
t,p = scipy.stats.ttest_rel(np.array(RT_all[(RT_all['day']==1) & (RT_all['Condition'] == 'Random')]['RT']),
                            np.array(RT_all[(RT_all['day']==1) & (RT_all['Condition'] == 'Fix')]['RT']))
print(f"t={t}, p={p}.")

"=== Day 2 ==="
t,p = scipy.stats.ttest_rel(np.array(RT_all[(RT_all['day']==2) & (RT_all['Condition'] == 'Random')]['RT']),
                            np.array(RT_all[(RT_all['day']==2) & (RT_all['Condition'] == 'Fix')]['RT']))
print(f"t={t}, p={p}.")

#%%
'''
    Does the Conflict Parameter differentiate between habitual learners, GD, and modulators
    in strong sequence learners?
'''

%matplotlib inline
# habitual_df, GD_df, modulators_df

# BF = anal.compute_BF(post_sample_df_all[(post_sample_df_all['day'] == 2) & ])

if model == 'Repbias_Conflict_lr':
    testparam1 = 'theta_rep'
    testparam2 = 'theta_conflict'
    
elif model == 'Repbias_Interaction_lr':
    testparam1 = 'theta_rep'
    testparam2 = 'theta_interact'
    
elif model == 'Repbias_3Q_lr':
    testparam1 = 'theta_repcong'
    testparam2 = 'theta_repinc'
    
complete_df_all[f'BF_{testparam2}'] = complete_df_all.apply(lambda row: anal.compute_BF(post_sample_df_all[(post_sample_df_all['day'] == row['day']) &
                                                                                                  (post_sample_df_all['ID'] == row['ID'])][f'{testparam2}'], 0), axis=1)

complete_df_all[f'BF_{testparam1}'] = complete_df_all.apply(lambda row: anal.compute_BF(post_sample_df_all[(post_sample_df_all['day'] == row['day']) &
                                                                                                  (post_sample_df_all['ID'] == row['ID'])][f'{testparam1}'], 0), axis=1)

strong_df = complete_df_all[complete_df_all['strong_seq_day2'] == 'strong sequence learner']
weak_df = complete_df_all[complete_df_all['strong_seq_day2'] == 'weak sequence learner']

strong_df['strategy'] = None

strong_df['strategy'] = strong_df['ID'].map(lambda x: 'Habitual' if x in habitual_df['ID'].unique() else
                                            'Suppresser' if x in GD_df['ID'].unique() else
                                            'Exploiter' if x in modulators_df['ID'].unique() else 
                                            'Anti-GD' if x in antimods_df['ID'].unique() else
                                            'none')

# r,p = scipy.stats.spearmanr(np.array(strong_df['']))

fig, ax = plt.subplots()
ax.scatter(strong_df[strong_df['day']==2][f'{testparam1}'], 
           strong_df[strong_df['day']==2][f'{testparam2}'], label = 'all')

x = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Habitual')][f'{testparam1}']
y = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Habitual')][f'{testparam2}']
ax.scatter(x, 
           y, color='g', label = 'Habitual')

x = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Suppresser')][f'{testparam1}']
y = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Suppresser')][f'{testparam2}']
ax.scatter(x, 
           y, color='b', label = 'Suppresser')

x = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Exploiter')][f'{testparam1}']
y = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Exploiter')][f'{testparam2}']
ax.scatter(x, 
           y, color='r', label = 'Exploiter')

x = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Anti-GD')][f'{testparam1}']
y = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Anti-GD')][f'{testparam2}']
ax.scatter(x, 
           y, color='m', label = 'Anti-GD')

ax.set_xlabel(f'{testparam1}')
ax.set_ylabel(f'{testparam2}')

plt.show()

strong_df['stratgroup'] = strong_df['strategy'].map(lambda x: "Exploiter + Suppresser" if x == "Exploiter" or x == "Suppresser" else "Habitual + Anti-GD")

fig, ax = plt.subplots()
sns.scatterplot(strong_df[strong_df['day'] == 1],
                x = f'{testparam1}',
                y = f'{testparam2}',
                hue = 'strategy')
plt.title('Day 1')
plt.show()

fig, ax = plt.subplots()
sns.scatterplot(strong_df[strong_df['day'] == 2],
                x = f'{testparam1}',
                y = f'{testparam2}',
                palette = ['r', 'g', 'b', 'm'],
                hue = 'strategy')
plt.title('Day 2')
ax.set_xlabel(r'$\theta_{Rep}$', fontsize = 15)
ax.set_ylabel(r'$\theta_{Switch}$', fontsize = 15)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig5b/res_fig5b_a_python.svg', bbox_inches = 'tight')
plt.show()

fig, ax = plt.subplots()
sns.scatterplot(strong_df[strong_df['day'] == 2],
                x = f'{testparam1}',
                y = f'{testparam2}',
                palette = ['r', 'b'],
                hue = 'stratgroup')
plt.title('Day 2')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig5b/res_fig5b_b_python.svg', bbox_inches = 'tight')
plt.show()

#%%
'''
    Fig for Retreat
'''
fig, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize = (10, 4))
sns.scatterplot(strong_df[strong_df['day'] == 2],
                x = f'{testparam1}',
                y = f'{testparam2}',
                palette = ['r', 'g', 'b', 'm'],
                hue = 'strategy',
                ax = ax[0])
ax[0].set_title('Day 1')

ax[0].set_xlabel(r'$\theta_{Rep}$', fontsize = 15)
ax[0].set_ylabel(r'$\theta_{Switch}$', fontsize = 15)
ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

sns.scatterplot(strong_df[strong_df['day'] == 2],
                x = f'{testparam1}',
                y = f'{testparam2}',
                palette = ['r', 'b'],
                hue = 'stratgroup',
                ax = ax[1])
ax[1].set_xlabel(r'$\theta_{Rep}$', fontsize = 15)
ax[1].set_ylabel(r'$\theta_{Switch}$', fontsize = 15)
ax[1].set_title('Day 2')
ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('/home/sascha/Desktop/presentations/AST/2024_03_Weimar/strategies_model_scatter_python.svg')
plt.show()

#%%
strong_df['ratio_repconf'] = strong_df.apply(lambda row: row['theta_rep']/row['theta_conflict'], axis = 1)
strong_df['ratio_confrep'] = strong_df.apply(lambda row: row['theta_conflict']/row['theta_rep'], axis = 1)

strong_df['ratio_repconfdiff'] = strong_df.apply(lambda row: row['theta_rep']-row['theta_conflict'], axis = 1)
strong_df['ratio_confrepdiff'] = strong_df.apply(lambda row: row['theta_conflict']-row['theta_rep'], axis = 1)

fig, ax = plt.subplots(1,3, figsize = (12,4))
sns.scatterplot(strong_df[strong_df['day'] == 2],
                x = f'{testparam1}',
                y = f'{testparam2}',
                hue = 'strategy',
                palette = ['r', 'g', 'b', 'm'],
                ax = ax[0])
ax[0].get_legend().remove()

sns.scatterplot(strong_df[strong_df['day'] == 2],
                x = f'{testparam1}',
                y = 'theta_Q',
                hue = 'strategy',
                palette = ['r', 'g', 'b', 'm'],
                ax = ax[1])

ax[1].get_legend().remove()
sns.scatterplot(strong_df[strong_df['day'] == 2],
                y = 'theta_Q',
                x = f'{testparam2}',
                hue = 'strategy',
                palette = ['r', 'g', 'b', 'm'],
                ax = ax[2])

# x = np.array([1, 2.5])
# ax.plot(x, x-0.5)
ax[2].legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


fig, ax = plt.subplots(1,3, figsize = (12,4))
sns.scatterplot(strong_df[strong_df['day'] == 2],
                x = f'{testparam1}',
                y = f'{testparam2}',
                hue = 'stratgroup',
                palette = ['r', 'b'],
                ax = ax[0])
ax[0].get_legend().remove()

sns.scatterplot(strong_df[strong_df['day'] == 2],
                x = f'{testparam1}',
                y = 'theta_Q',
                hue = 'stratgroup',
                palette = ['r', 'b'],
                ax = ax[1])

ax[1].get_legend().remove()
sns.scatterplot(strong_df[strong_df['day'] == 2],
                y = 'theta_Q',
                x = f'{testparam2}',
                hue = 'stratgroup',
                palette = ['r', 'b'],
                ax = ax[2])

# x = np.array([1, 2.5])
# ax.plot(x, x-0.5)
ax[2].legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig5b/res_fig5b_c_python.svg', bbox_inches ='tight')
plt.show()

fig, ax = plt.subplots()
sns.scatterplot(strong_df[strong_df['day'] == 2],
                x = 'theta_rep',
                y = 'ratio_repconf',
                hue = 'strategy')

plt.show()

fig, ax = plt.subplots()
sns.scatterplot(strong_df[strong_df['day'] == 2],
                x = 'theta_conflict',
                y = 'ratio_repconf',
                hue = 'strategy')

plt.show()

fig, ax = plt.subplots()
sns.scatterplot(strong_df[strong_df['day'] == 2],
                x = 'theta_rep',
                y = 'ratio_repconfdiff',
                hue = 'strategy')

plt.show()

fig, ax = plt.subplots()
sns.scatterplot(strong_df[strong_df['day'] == 2],
                x = 'theta_conflict',
                y = 'ratio_repconfdiff',
                hue = 'strategy')

plt.show()

# fig, ax = plt.subplots()
# sns.scatterplot(complete_df_all[complete_df_all['day'] == 2],
#                 x = f'{testparam1}',
#                 y = f'{testparam2}',
#                 hue = 'strategy')
# plt.title('All day 2')
# plt.show ()

#%%

# sns.kdeplot(post_sample_df_all[(post_sample_df_all['day']==2) & (post_sample_df_all['ID']=='57c4761195a3ea00016e5992')]['theta_rep'])
# sns.kdeplot(post_sample_df_all[(post_sample_df_all['day']==2) & (post_sample_df_all['ID']=='5eec9ee7d900510326d78fc8')]['theta_rep'])
# plt.show()

# sns.kdeplot(post_sample_df_all[(post_sample_df_all['day']==2) & (post_sample_df_all['ID']=='57c4761195a3ea00016e5992')]['theta_conflict'])
# sns.kdeplot(post_sample_df_all[(post_sample_df_all['day']==2) & (post_sample_df_all['ID']=='5eec9ee7d900510326d78fc8')]['theta_conflict'])
# plt.show()

# r,p=scipy.stats.pearsonr(post_sample_df_all[(post_sample_df_all['day']==2) & (post_sample_df_all['ID']=='57c4761195a3ea00016e5992')]['theta_rep'],
#                          post_sample_df_all[(post_sample_df_all['day']==2) & (post_sample_df_all['ID']=='57c4761195a3ea00016e5992')]['theta_conflict'])

# r,p=scipy.stats.pearsonr(post_sample_df_all[(post_sample_df_all['day']==2) & (post_sample_df_all['ID']=='5eec9ee7d900510326d78fc8')]['theta_rep'],
#                          post_sample_df_all[(post_sample_df_all['day']==2) & (post_sample_df_all['ID']=='5eec9ee7d900510326d78fc8')]['theta_conflict'])


#%%
%matplotlib qt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

strong_df['corr_day2'] = strong_df.apply(lambda row: post_sample_df_all[(post_sample_df_all['day']==2) & 
                                                                        (post_sample_df_all['ID']==row['ID'])][f'{testparam1}'].corr(post_sample_df_all[(post_sample_df_all['day']==2) & 
                                                                                                                                                (post_sample_df_all['ID']==row['ID'])][f'{testparam2}']), axis = 1)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Habitual')][f'{testparam1}']
y = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Habitual')][f'{testparam2}']
z = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Habitual')]['corr_day2']
ax.scatter(x, 
           y, 
           z,
           color='b', label = 'Habitual')

x = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Suppresser')][f'{testparam1}']
y = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Suppresser')][f'{testparam2}']
z = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Suppresser')]['corr_day2']

ax.scatter(x, 
           y, 
           z, color='g', label = 'Suppresser')

x = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Exploiter')][f'{testparam1}']
y = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Exploiter')][f'{testparam2}']
z = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Exploiter')]['corr_day2']

ax.scatter(x, 
           y, 
           z,color='k', label = 'Adpater')

# for i, (x_pos, y_pos) in enumerate(zip(x, y)):
#     plt.annotate(i, # this is the text to show
#                  (x_pos, y_pos), # these are the coordinates to position the text
#                  textcoords="offset points", # how to position the text
#                  xytext=(0,10), # distance from text to points (x,y)
#                  ha='center') # horizontal alignment can be left, right or center

x = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Anti-GD')][f'{testparam1}']
y = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Anti-GD')][f'{testparam2}']
z = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Anti-GD')]['corr_day2']

ax.scatter(x, 
           y,
           z, color='m', label = 'Anti')

# ax.scatter(weak_df[(weak_df['day']==2)]['theta_rep'], 
#            weak_df[weak_df['day']==2]['theta_conflict'], color='y')

# ax.plot([1, 1.75], [1, 1.75])

# ax.set_aspect('equal')
ax.set_xlabel(f'{testparam1}')
ax.set_ylabel(f'{testparam2}')
ax.set_zlabel('Correlation')
plt.show()

#%%

%matplotlib qt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Habitual')][f'{testparam1}']
y = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Habitual')][f'{testparam2}']
z = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Habitual')]['theta_Q']
ax.scatter(x, 
           y, 
           z,
           color='g', label = 'Habitual', s=40)

x = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Suppresser')][f'{testparam1}']
y = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Suppresser')][f'{testparam2}']
z = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Suppresser')]['theta_Q']

ax.scatter(x, 
           y, 
           z, color='b', label = 'Suppresser', s=40)

x = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Exploiter')][f'{testparam1}']
y = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Exploiter')][f'{testparam2}']
z = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Exploiter')]['theta_Q']

ax.scatter(x, 
           y, 
           z,color='r', label = 'Exploiter', s=40)

# for i, (x_pos, y_pos) in enumerate(zip(x, y)):
#     plt.annotate(i, # this is the text to show
#                  (x_pos, y_pos), # these are the coordinates to position the text
#                  textcoords="offset points", # how to position the text
#                  xytext=(0,10), # distance from text to points (x,y)
#                  ha='center') # horizontal alignment can be left, right or center

x = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Anti-GD')][f'{testparam1}']
y = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Anti-GD')][f'{testparam2}']
z = strong_df[(strong_df['day']==2) & (strong_df['strategy']=='Anti-GD')]['theta_Q']

ax.scatter(x, 
           y,
           z, color='m', label = 'Anti-GD', s=40)

# ax.scatter(weak_df[(weak_df['day']==2)]['theta_rep'], 
#            weak_df[weak_df['day']==2]['theta_conflict'], color='y')

# ax.plot([1, 1.75], [1, 1.75])

# ax.set_aspect('equal')
ax.set_xlabel(f'{testparam1}')
ax.set_ylabel(f'{testparam2}')
ax.set_zlabel('theta_Q')
plt.show()