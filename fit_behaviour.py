#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:05:13 2023

Fit model to behaviour.

@author: sascha
"""
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

model = 'B'
#%%
exp_behav_dict, exp_behav_df = utils.get_groupdata('/home/sascha/Desktop/vbm_torch/behav_data/')

utils.plot_grouplevel(exp_behav_df)

num_agents = len(exp_behav_dict['trialsequence'][0])


#%%
'''
Compute Errorrates
'''
num_agents = len(exp_behav_df['ag_idx'].unique())
errorrates = torch.zeros((3, num_agents)) # STT, DTT, Total

for ag_idx in np.sort(exp_behav_df['ag_idx'].unique()):
    ag_df = exp_behav_df[exp_behav_df['ag_idx'] == ag_idx]

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
pickle.dump( (post_sample_df, exp_behav_df, infer.loss, params_sim_df), open(f"behav_fit/behav_fit_model_{model}_{timestamp}.p", "wb" ) )

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

for ag_idx in np.sort(expdata_df['ag_idx'].unique()):
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

for key in corr_dict.keys():
    sns.kdeplot(corr_dict[key])
    plt.title(key)
    plt.show()


#%%
'''
How close are subjects in correlation space?
'''
import ipdb
import scipy.cluster.hierarchy as spc
distance = np.zeros((num_agents, num_agents))

for ag_idx1 in range(num_agents):
    for ag_idx2 in range(num_agents):
        v1 = np.zeros(len(corr_dict))
        v2 = np.zeros(len(corr_dict))
        
        for key_idx in range(len(corr_dict.keys())):
            key = list(corr_dict.keys())[key_idx]
            v1[key_idx] = corr_dict[key][ag_idx1]
            v2[key_idx] = corr_dict[key][ag_idx2]

        distance[ag_idx1, ag_idx2] = np.sqrt(np.sum((v1 - v2)**2))

def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]

distance_vec = upper_tri_indexing(distance)
linkage = spc.linkage(distance_vec, method='single')
idx = spc.fcluster(linkage, 0.5 * distance_vec.max(), 'distance')
dn = spc.dendrogram(linkage)
plt.show()

"----- With spc"
pdist = spc.distance.pdist(pd.DataFrame(corr_dict))
"Linkage matrix"
linkage = spc.linkage(pdist, method='single')
idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')
dn = spc.dendrogram(linkage)
plt.show()
"---"

# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(distance)
# fig.colorbar(cax)

# corr_prep_df = post_sample_df.drop(['group', 'model'], axis = 1)

# "From Internet"

# corrdf = pd.DataFrame(corr_dict)
# corr = corrdf.corr().values
# pdist = spc.distance.pdist(corr)

#%%
'''
Plot Experimental Data
'''
utils.plot_grouplevel(expdata_df, plot_single = False)

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