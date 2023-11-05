#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:05:13 2023

Fit model to behaviour.

@author: sascha
"""
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import pickle

import analysis as anal
import inferencemodels
import utils

'''
Modelle:
Vbm
B
Conflict
'''

model = 'BQ'
#%%
exp_behav_dict, expdata_df = utils.get_groupdata('/home/sascha/Desktop/vbm_torch/behav_data/')
utils.plot_grouplevel(expdata_df)
num_agents = len(expdata_df['ag_idx'].unique())
#%%
# '''
# Exclude participants with high error rates (only to balance groups)
# '''
# exclude = [14, 20, 24, 28, 38, 43, 45, 57, 58]
# exp_behav_df = exp_behav_df[~exp_behav_df['ag_idx'].isin(exclude)]
# num_agents = len(exp_behav_df['ag_idx'].unique())
# grouped = pd.DataFrame(exp_behav_df.groupby(['ag_idx', 'group', 'model'], as_index = False).mean())
# 14,18,17,16

#%%
'''
Prepare Inference
'''
group = exp_behav_dict['group'][0]

'''
Inference
'''

"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent(model, 
                         group, 
                         num_agents = num_agents)

print("===== Starting inference =====")
"----- Start Inference"
infer = inferencemodels.GeneralGroupInference(agent, exp_behav_dict)
infer.infer_posterior(iter_steps = 10_000, num_particles = 10)

"----- Sample parameter estimates from posterior"
post_sample_df = infer.sample_posterior()
post_sample_df['group'] = post_sample_df['ag_idx'].map(lambda x: exp_behav_dict['group'][0][x])
post_sample_df['model'] = [model]*len(post_sample_df)

"----- Save results to file"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
params_sim_df = pd.DataFrame(columns = agent.param_dict.keys())
for col in params_sim_df.columns:
    params_sim_df[col] = ['unknown']
pickle.dump( (post_sample_df, expdata_df, infer.loss, params_sim_df), open(f"behav_fit/behav_fit_model_{model}_{timestamp}.p", "wb" ) )

#%%
'''
Analysis
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
import analysis as anal
import torch
import tkinter as tk
from tkinter import filedialog
import pickle

def open_files():
    global filenames
    filenames = filedialog.askopenfilenames()
    print(f'File paths: {filenames}')
    root.destroy()
    
root = tk.Tk()
button = tk.Button(root, text="Open Files", command=open_files)
print(button)
button.pack()
root.mainloop()

# post_sample_df, df, loss, param_names = pickle.load(open( filenames[0], "rb" ))
res = pickle.load(open( filenames[0], "rb" ))
post_sample_df, expdata_df, loss, params_df = res
inf_mean_df = pd.DataFrame(post_sample_df.groupby(['model', 'ag_idx', 'group'], as_index = False).mean())
model = post_sample_df['model'][0]
num_agents = len(post_sample_df['ag_idx'].unique())
num_params = len(params_df.columns)

#%%
'''
Compute Errorrates
'''
num_agents = len(expdata_df['ag_idx'].unique())
errorrates = torch.zeros((3, num_agents)) # STT, DTT, Total
errorrates_day1 = torch.zeros((3, num_agents)) # STT, DTT, Total
errorrates_day2 = torch.zeros((3, num_agents)) # STT, DTT, Total

for ag_idx in np.sort(expdata_df['ag_idx'].unique()):
    "----- Both days"
    ag_df = expdata_df[expdata_df['ag_idx'] == ag_idx]
    "Remove new block trials"
    ag_df = ag_df[ag_df['choices'] != -1]
    "Total error rates"
    errorrates[-1, ag_idx] = (ag_df['choices'] == -2).sum() / len(ag_df)
    ag_df_stt = ag_df[ag_df['trialsequence'] < 10]
    "Error Rates STT"
    errorrates[0, ag_idx] = (ag_df_stt['choices'] == -2).sum() / len(ag_df_stt)
    ag_df_dtt = ag_df[ag_df['trialsequence'] > 10]
    "Error Rates DTT"
    errorrates[1, ag_idx] = (ag_df_dtt['choices'] == -2).sum() / len(ag_df_dtt)
    
    "----- Day 1"
    ag_df_day1 = expdata_df[(expdata_df['ag_idx'] == ag_idx) & (expdata_df['blockidx'] <= 5)]
    "Remove new block trials"
    ag_df_day1 = ag_df_day1[ag_df_day1['choices'] != -1]
    "Total error rates"
    errorrates_day1[-1, ag_idx] = (ag_df_day1['choices'] == -2).sum() / len(ag_df_day1)
    ag_df_day1_stt = ag_df_day1[ag_df_day1['trialsequence'] < 10]
    "Error Rates STT"
    errorrates_day1[0, ag_idx] = (ag_df_day1_stt['choices'] == -2).sum() / len(ag_df_day1_stt)
    ag_df_day1_dtt = ag_df_day1[ag_df_day1['trialsequence'] > 10]
    "Error Rates DTT"
    errorrates_day1[1, ag_idx] = (ag_df_day1_dtt['choices'] == -2).sum() / len(ag_df_day1_dtt)

    "----- Day 2"
    ag_df_day2 = expdata_df[(expdata_df['ag_idx'] == ag_idx) & (expdata_df['blockidx'] > 5)]
    "Remove new block trials"
    ag_df_day2 = ag_df_day2[ag_df_day2['choices'] != -1]
    "Total error rates"
    errorrates_day2[-1, ag_idx] = (ag_df_day2['choices'] == -2).sum() / len(ag_df_day2)
    ag_df_day2_stt = ag_df_day2[ag_df_day2['trialsequence'] < 10]
    "Error Rates STT"
    errorrates_day2[0, ag_idx] = (ag_df_day2_stt['choices'] == -2).sum() / len(ag_df_day2_stt)
    ag_df_day2_dtt = ag_df_day2[ag_df_day2['trialsequence'] > 10]
    "Error Rates DTT"
    errorrates_day2[1, ag_idx] = (ag_df_day2_dtt['choices'] == -2).sum() / len(ag_df_day2_dtt)


#%%
'''
Plot ELBO
'''
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots()
plt.plot(loss[-2000:])
plt.title("ELBO")
ax.set_xlabel("Number of iterations")
ax.set_ylabel("ELBO")
plt.show()

#%%
'''
Plot ELBO and Parameter Estimates
'''

fig, ax = plt.subplots()
plt.plot(loss)
plt.title("ELBO")
ax.set_xlabel("Number of iterations")
ax.set_ylabel("ELBO")
plt.show()

for param in params_df.columns:
    fig, ax = plt.subplots()
    sns.kdeplot(post_sample_df[param])
    # plt.plot(df[param+'_true'], df[param+'_true'])
    plt.show()

#%%
'''
Violin Plot
'''
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# plt.style.use("seaborn-v0_8-dark")

anal.violin(inf_mean_df)

#%%
'''
Model B specific analysis
Ratios between parameters.
'''

Q_R_ratio_day1 = []
Q_R_ratio_day2 = []
for ag_idx in range(num_agents):
    Q_R_ratio_day1.append((post_sample_df[post_sample_df['ag_idx'] == ag_idx]['theta_Q_day1'] / \
        post_sample_df[post_sample_df['ag_idx'] == ag_idx]['theta_rep_day1']).mean())
        
    Q_R_ratio_day2.append((post_sample_df[post_sample_df['ag_idx'] == ag_idx]['theta_Q_day2'] / \
        post_sample_df[post_sample_df['ag_idx'] == ag_idx]['theta_rep_day2']).mean())
        
diff = torch.tensor(Q_R_ratio_day1) - torch.tensor(Q_R_ratio_day2)
sns.kdeplot(diff)
plt.title('Ratio Differences Day 1 - Day 2')
import scipy
import numpy as np
scipy.stats.ttest_1samp(np.asarray(diff), popmean=0)
# post_sample_df.groupby.iloc[:, 0:-2][('ag_idx')]

#%%
'''
Correlations between subjects
'''
import analysis as anal
anal.param_corr(inf_mean_df)

#%%
'''
Correlations within subjects
'''
corr_dict = anal.within_subject_corr(post_sample_df)

corr_dict_errors = corr_dict.copy()
corr_dict_errors['errors_stt'] = errorrates[0, :]
corr_dict_errors['errors_dtt'] = errorrates[1, :]

#%%
for key in corr_dict.keys():
    sns.kdeplot(corr_dict[key])
    plt.title(key)
    plt.show()


#%%
'''
Correlation analysis both days
'''
leavnodes = anal.cluster_analysis(corr_dict, title = 'all correlations exp')
labels, cluster_groups = anal.kmeans(corr_dict, inf_mean_df, n_clusters = 2)

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(leavnodes[0:13])],
                      expdata_df[expdata_df['ag_idx'].isin(leavnodes[13:21])])

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(leavnodes[13:21])],
                      expdata_df[expdata_df['ag_idx'].isin(leavnodes[21:])])

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(leavnodes[0:3])],
                      expdata_df[expdata_df['ag_idx'].isin(leavnodes[21:])])

#%%
'''
Correlation analysis day 1
'''
post_sample_df_day1 = post_sample_df.drop(['lr_day1', 'theta_Q_day1', 'theta_rep_day1'], axis = 1)
corr_dict_day1 = anal.within_subject_corr(post_sample_df_day1)

leavnodes_day1 = anal.cluster_analysis(corr_dict_day1, title='day 1 exp')

# inf_mean_df[inf_mean_df['ag_idx'].isin(leavnodes[0:16])]['group']

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(leavnodes_day1[0:7])],
                      expdata_df[expdata_df['ag_idx'].isin(leavnodes_day1[6:16])])

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(leavnodes_day1[0:16])],
                      expdata_df[expdata_df['ag_idx'].isin(leavnodes_day1[16:])])


'''
K-means clustering for Day 1
'''
labels, cluster_groups_day1 = anal.kmeans(corr_dict_day1, inf_mean_df, n_clusters = 2)

# anal.compare_leavnodes(grp1_agidx, leavnodes_day1[0:16])

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(cluster_groups_day1[0])],
                      expdata_df[expdata_df['ag_idx'].isin(cluster_groups_day1[1])], 
                      day=1)

#%%
'''
kmeans with errors Day 1
'''
corr_dict_day1_errors = corr_dict_day1.copy()
corr_dict_day1_errors['errors_stt_day1'] = errorrates_day1[0, :]
corr_dict_day1_errors['errors_dtt_day1'] = errorrates_day1[1, :]

leavnodes_day1_errors = anal.cluster_analysis(corr_dict_day1_errors, title='day 1 exp errors')

labels, cluster_groups_day1_errors = anal.kmeans(corr_dict_day1_errors, inf_mean_df, n_clusters = 2)

anal.compare_lists(cluster_groups_day1_errors[0], cluster_groups_day1[0])
#%%
'''
Perform correlation analysis only within day 2
'''
post_sample_df_day2 = post_sample_df.drop(['lr_day2', 'theta_Q_day2', 'theta_rep_day2'], axis = 1)
corr_dict_day2 = anal.within_subject_corr(post_sample_df_day2)

leavnodes_day2 = anal.cluster_analysis(corr_dict_day2, title = 'day 2')

#%%
'''
K-means clustering for Day 2
'''
labels, cluster_groups = anal.kmeans(corr_dict_day2, inf_mean_df, n_clusters = 2)

# anal.compare_leavnodes(grp0_agidx, leavnodes_day1[0:16])

# utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(grp0_agidx)],
#                       expdata_df[expdata_df['ag_idx'].isin(grp1_agidx)])

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(cluster_groups[0])],
                      expdata_df[expdata_df['ag_idx'].isin(cluster_groups[1])],
                      day = 2)

#%%
'''
kmeans with errors Day 2
'''
post_sample_df_day2 = post_sample_df.drop(['lr_day2', 'theta_Q_day2', 'theta_rep_day2'], axis = 1)
corr_dict_day2 = anal.within_subject_corr(post_sample_df_day2)

corr_dict_day2_errors = corr_dict_day2.copy()
corr_dict_day2_errors['errors_stt_day2'] = errorrates_day2[0, :]
corr_dict_day2_errors['errors_dtt_day2'] = errorrates_day2[1, :]

leavnodes_day2 = anal.cluster_analysis(corr_dict_day2_errors, title = 'day 2')

labels, cluster_groups = anal.kmeans(corr_dict_day2_errors, inf_mean_df, n_clusters = 2)

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

labels, cluster_groups_between = anal.kmeans(corr_dict_day_between, inf_mean_df, n_clusters = 2)

# anal.compare_leavnodes(grp0_agidx, leavnodes_day1[0:16])

# utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(grp0_agidx)],
#                       expdata_df[expdata_df['ag_idx'].isin(grp1_agidx)])

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(cluster_groups_between[0])],
                      expdata_df[expdata_df['ag_idx'].isin(cluster_groups_between[1])])

anal.compare_leavnodes(cluster_groups_day1[1], cluster_groups_between[1])

#%%

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(cluster_groups_between[0])],
                      expdata_df[expdata_df['ag_idx'].isin(cluster_groups_between[1])])

utils.plot_grouplevel(expdata_df[expdata_df['ag_idx'].isin(cluster_groups_day1[1])], day = 1)


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