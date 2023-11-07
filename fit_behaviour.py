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

import analysis_tools as anal
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
exp_behav_dict, expdata_df = utils.get_groupdata('/home/sascha/Desktop/vbm_torch/behav_data/')

errorrates, errorrates_day1, errorrates_day2 = anal.compute_errors(expdata_df)
print("-------------------------------------------------------\n\n")
print("Maximum errorrate for STT: %.2f, DTT: %.2f, total: %.2f"%(errorrates[0,:].max(),errorrates[1,:].max(),errorrates[2,:].max()))

utils.plot_grouplevel(expdata_df)
num_agents = len(expdata_df['ag_idx'].unique())
dfnew=pd.DataFrame(expdata_df.groupby(['ag_idx', 'group', 'model'], as_index = False).mean())
group_distro = [len(dfnew[dfnew['group']== grp]) for grp in range(4)]
print(group_distro)
assert np.abs(np.diff(group_distro)).sum() == 0
# print(len(dfnew[dfnew['group']==1]))
# print(len(dfnew[dfnew['group']==2]))
# print(len(dfnew[dfnew['group']==3]))
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