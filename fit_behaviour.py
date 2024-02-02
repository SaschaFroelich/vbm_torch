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

import env
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

post_pred = 0
model_day1 = 'Repbias_Conflict_lr'
models_day2 = ['Repbias_Conflict_lr', 'Repbias_Conflict_nolr']
num_inf_steps = 5_000
halting_rtol = 1e-06 # for MLE estimation
posterior_pred_samples = 5_000

#%%

"Day 1"
exp_behav_dict_day1, expdata_df_day1 = pickle.load(open("behav_data/preproc_data_day1.p", "rb" ))
num_agents = len(expdata_df_day1['ag_idx'].unique())
group = exp_behav_dict_day1['group'][0]

"Make sure same number of participants in each group"
group_distro = [(np.array(group)==grp).sum() for grp in range(4)]
assert np.abs(np.diff(group_distro)).sum() == 0

"Day 2"
exp_behav_dict_day2, expdata_df_day2 = pickle.load(open("behav_data/preproc_data_day2.p", "rb" ))
num_agents = len(expdata_df_day2['ag_idx'].unique())
group = exp_behav_dict_day2['group'][0]

"Make sure same number of participants in each group"
group_distro = [(np.array(group)==grp).sum() for grp in range(4)]
assert np.abs(np.diff(group_distro)).sum() == 0


'''
    Fit day 1
'''
# import time
# time.sleep(0.5*3600)

day = 1

'''
    Inference
'''
print(f"Starting inference of model {model_day1} for day 1 for {num_agents} agents.")
"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent(model_day1, 
                         group, 
                         num_agents = num_agents)

Q_init_day1 = agent.Q_init

print("===== Starting inference for day 1 =====")
"----- Start Inference"
infer = inferencemodels.GeneralGroupInference(agent, exp_behav_dict_day1)
agent_elbo_tuple = infer.infer_posterior(iter_steps = num_inf_steps, num_particles = 10)

"----- Sample parameter estimates from posterior and add information to DataFrame"
if post_pred:
    firstlevel_df, secondlevel_df = infer.posterior_predictives(n_samples = posterior_pred_samples)
    
else:
    firstlevel_df = infer.sample_posterior(n_samples = posterior_pred_samples)
    secondlevel_df = None

firstlevel_df['group'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict_day1['group'][0][x])
firstlevel_df['model'] = [model_day1]*len(firstlevel_df)

ID_df = expdata_df_day1.loc[:, ['ID', 'ag_idx']].drop_duplicates()
firstlevel_df['ID'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict_day1['ID'][0][x])

"----- MLE & IC"
max_log_like, mle_locs = infer.train_mle(halting_rtol = halting_rtol)
BIC, AIC = infer.compute_IC()

"----- Q_init for next day"
param_names = agent.param_names
inf_mean_df = firstlevel_df.loc[:, [*param_names, 
                      'ag_idx', 
                      'ID']].groupby(['ag_idx', 
                                      'ID'], as_index = False).mean()
assert torch.all(torch.tensor(inf_mean_df['ag_idx']) == torch.tensor(exp_behav_dict_day1['ag_idx'][0]))
assert all([inf_mean_df['ID'][i] == exp_behav_dict_day1['ID'][0][i] for i in range(num_agents)])
                                      
_, _, _, sim_agent = utils.simulate_data(model_day1, 
                                        num_agents,
                                        group = group,
                                        day = day,
                                        params = inf_mean_df.loc[:, [*param_names]])

assert sim_agent.Q[-1].shape[0] == 1 and sim_agent.Q[-1].ndim == 3
Q_init_day2 = np.squeeze(np.array(sim_agent.Q))[-10:, :, :].mean(axis=0)
Q_init_day2 = Q_init_day2[None, ...]
assert Q_init_day2.ndim == 3
Q_init_day2 = torch.tensor(Q_init_day2)

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
                 '',
                 'no preceding model',
                 max_log_like,
                 mle_locs)

pickle.dump( (firstlevel_df, 
              expdata_df_day1,
              (infer.loss, BIC, AIC), 
              params_sim_df, 
              agent_elbo_tuple, 
              extra_storage), 
            open(f"behav_fit/behav_fit_model_day1_{model_day1}_day{day}_{timestamp}_{num_agents}agents.p", "wb" ) )

'''
    Fit day 2
'''
del exp_behav_dict_day1
del expdata_df_day1
for model_day2 in models_day2:
    agent = utils.init_agent(model_day2, 
                             group, 
                             num_agents = num_agents,
                             Q_init = Q_init_day2.detach())
    
    print("===== Starting inference for day 2 =====")
    "----- Start Inference"
    infer = inferencemodels.GeneralGroupInference(agent, exp_behav_dict_day2)
    agent_elbo_tuple = infer.infer_posterior(iter_steps = num_inf_steps, num_particles = 10)

    "----- Sample parameter estimates from posterior and add information to DataFrame"
    if post_pred:
        firstlevel_df, secondlevel_df = infer.posterior_predictives(n_samples = posterior_pred_samples)
    else:
        firstlevel_df = infer.sample_posterior(n_samples = posterior_pred_samples)
        secondlevel_df = None
    
    firstlevel_df['group'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict_day2['group'][0][x])
    firstlevel_df['model'] = [model_day2]*len(firstlevel_df)
    
    ID_df = expdata_df_day2.loc[:, ['ID', 'ag_idx']].drop_duplicates()
    firstlevel_df['ID'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict_day2['ID'][0][x])
    
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
                     '', # blocks
                     model_day1, # preceding model
                     max_log_like,
                     mle_locs) 
    
    pickle.dump( (firstlevel_df, 
                  expdata_df_day2,
                  (infer.loss, BIC, AIC), 
                  params_sim_df, 
                  agent_elbo_tuple, 
                  extra_storage), 
                open(f"behav_fit/behav_fit_model_day2_{model_day2}_model1_{timestamp}_{num_agents}agents.p", "wb" ) )