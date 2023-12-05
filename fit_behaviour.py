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
Vbm_twodays

B_oneday
B

Conflict

Seqboost

Bhand_oneday
Bhand

ConflictHand

BQ
BK
'''

model = 'SeqConflictHand'
#%%

exp_behav_dict, expdata_df = pickle.load(open("behav_data/preproc_data.p", "rb" ))
# dfnew = pd.DataFrame(expdata_df.groupby(['ag_idx', 'group', 'model', 'ID'], as_index = False).mean())
dfnew = pd.DataFrame(expdata_df.loc[:, ['ID', 'group']].groupby(['ID'], as_index = False).mean())
group_distro = [len(dfnew[dfnew['group']== grp]) for grp in range(4)]
print(group_distro)
assert np.abs(np.diff(group_distro)).sum() == 0
del dfnew

# er_df = anal.compute_errors(expdata_df)
print("-------------------------------------------------------\n\n")
# print("Maximum errorrate for STT: %.2f, DTT: %.2f, total: %.2f"%(errorrates[0,:].max(),errorrates[1,:].max(),errorrates[2,:].max()))

utils.plot_grouplevel(expdata_df.drop(['ID'], axis = 1, inplace=False))
num_agents = len(expdata_df['ag_idx'].unique())

print(f"Starting inference of model {model} for {num_agents} agents.")

#%%
# import time
# time.sleep(15*3600)

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
agent_elbo_tuple = infer.infer_posterior(iter_steps = 1, num_particles = 10, block_max = 14)

"----- Sample parameter estimates from posterior"
post_sample_df = infer.sample_posterior( )
post_sample_df['group'] = post_sample_df['ag_idx'].map(lambda x: exp_behav_dict['group'][0][x])
post_sample_df['model'] = [model]*len(post_sample_df)

ID_df = expdata_df.loc[:, ['ID', 'ag_idx']].drop_duplicates()
post_sample_df['ID'] = post_sample_df['ag_idx'].map(lambda x: exp_behav_dict['ID'][0][x])

"----- Save results to file"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
params_sim_df = pd.DataFrame(columns = agent.param_dict.keys())
for col in params_sim_df.columns:
    params_sim_df[col] = ['unknown']

params_sim_df['ag_idx']  = None
params_sim_df['group']  = None
params_sim_df['model']  = model
BIC, AIC = infer.compute_IC()
# ELBOs = infer.compute_ELBOS()
# loo_prediction = infer.loo_predict()
# posterior_params = (infer.guide()['m_locs'], infer.guide()['st_locs'])
pickle.dump( (post_sample_df, expdata_df, (infer.loss, BIC, AIC), params_sim_df, agent_elbo_tuple), 
            open(f"behav_fit/behav_fit_model_{model}_{timestamp}_{num_agents}agents.p", "wb" ) )