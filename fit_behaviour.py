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

    Vbm_lr
    Repbias_lr
    Repbias_Conflict
    Repbias_CongConflict_lr
    OnlyQ_lr
'''

model_day1 = 'OnlyQ_lr'
models_day2 = ['OnlyQ_nolr', 'OnlyQ_lr']
num_inf_steps = 2
halting_rtol = 1e-02 # for MLE estimation

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

print(f"Starting inference of model {model_day1} for day 1 for {num_agents} agents.")

group = exp_behav_dict['group'][0]
#%%
'''
    Fit day 1
'''
# import time
# time.sleep(8*3600)

blocks_day1 = [0,3]

'''
    Inference
'''
"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent(model_day1, 
                         group, 
                         num_agents = num_agents)

Q_init_day1 = agent.Q_init

print("===== Starting inference =====")
"----- Start Inference"
infer = inferencemodels.GeneralGroupInference(agent, exp_behav_dict, blocks = blocks_day1)
agent_elbo_tuple = infer.infer_posterior(iter_steps = num_inf_steps, num_particles = 10)

"----- Sample parameter estimates from posterior and add information to DataFrame"
firstlevel_df, secondlevel_df = infer.sample_posterior(n_samples = 5_000)

firstlevel_df['group'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict['group'][0][x])
firstlevel_df['model'] = [model_day1]*len(firstlevel_df)

ID_df = expdata_df.loc[:, ['ID', 'ag_idx']].drop_duplicates()
firstlevel_df['ID'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict['ID'][0][x])

"----- MLE & IC"
max_log_like, mle_locs = infer.train_mle(halting_rtol = halting_rtol)
BIC, AIC = infer.compute_IC()

"----- Q_init for next day"
Q_init_day2 = agent.Q[-1].detach().mean(axis=0)[None, ...]

"----- Save parameter names to DataFrame"
params_sim_df = pd.DataFrame(columns = agent.param_dict.keys())
for col in params_sim_df.columns:
    params_sim_df[col] = ['unknown']

params_sim_df['ag_idx']  = None
params_sim_df['group']  = None
params_sim_df['model']  = model_day1

"----- Store results"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
extra_storage = (Q_init_day1,
                 agent.Q[-1].detach(),
                 blocks_day1,
                 'no preceding model',
                 max_log_like,
                 mle_locs)

pickle.dump( (firstlevel_df, 
              expdata_df[(expdata_df['trialidx'] >= blocks_day1[0]*962) & (expdata_df['trialidx'] < blocks_day1[1]*962)], 
              (infer.loss, BIC, AIC), 
              params_sim_df, 
              agent_elbo_tuple, 
              extra_storage), 
            open(f"behav_fit/behav_fit_model_{model_day1}_blocks{blocks_day1[0]}{blocks_day1[1]}_{timestamp}_{num_agents}agents.p", "wb" ) )

'''
    Fit day 2
'''
for model_day2 in models_day2:
    blocks_day2 = [3,7]

    agent = utils.init_agent(model_day2, 
                             group, 
                             num_agents = num_agents,
                             Q_init = Q_init_day2)
    
    print("===== Starting inference =====")
    "----- Start Inference"
    infer = inferencemodels.GeneralGroupInference(agent, exp_behav_dict, blocks = blocks_day2)
    agent_elbo_tuple = infer.infer_posterior(iter_steps = num_inf_steps, num_particles = 10)

    "----- Sample parameter estimates from posterior and add information to DataFrame"
    firstlevel_df, secondlevel_df = infer.sample_posterior(n_samples = 5_000)
    
    firstlevel_df['group'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict['group'][0][x])
    firstlevel_df['model'] = [model_day2]*len(firstlevel_df)
    
    ID_df = expdata_df.loc[:, ['ID', 'ag_idx']].drop_duplicates()
    firstlevel_df['ID'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict['ID'][0][x])
    
    "----- MLE & IC"
    max_log_like, mle_locs = infer.train_mle(halting_rtol = halting_rtol)
    BIC, AIC = infer.compute_IC()
    
    # "----- Q_init for next day"
    # Q_init_day2 = agent.Q[-1].detach().mean(axis=0)[None, ...]
    
    "----- Save parameter names to DataFrame"
    params_sim_df = pd.DataFrame(columns = agent.param_dict.keys())
    for col in params_sim_df.columns:
        params_sim_df[col] = ['unknown']
    
    params_sim_df['ag_idx']  = None
    params_sim_df['group']  = None
    params_sim_df['model']  = model_day2
    
    "----- Store results"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    extra_storage = (Q_init_day2,  # initial Q-values
                     agent.Q[-1].detach(), # final Q-values
                     blocks_day2, # blocks
                     model_day1, # preceding model
                     max_log_like,
                     mle_locs) 
    
    pickle.dump( (firstlevel_df, 
                  expdata_df[(expdata_df['trialidx'] >= blocks_day2[0]*962) & (expdata_df['trialidx'] < blocks_day2[1]*962)], 
                  (infer.loss, BIC, AIC), 
                  params_sim_df, 
                  agent_elbo_tuple, 
                  extra_storage), 
                open(f"behav_fit/behav_fit_model2_{model_day2}_model1_{model_day1}_blocks{blocks_day2[0]}{blocks_day2[1]}_{timestamp}_{num_agents}agents.p", "wb" ) )