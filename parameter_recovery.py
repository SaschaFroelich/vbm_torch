#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:26:59 2023

Parameter Recovery 
1. Simulation
2. Group-Level Inference

@author: sascha
"""

import utils
import inferencemodels
from datetime import datetime
import pickle
import torch
import numpy as np
import pandas as pd

'''
Modelle:

    Vbm
    Vbm_twodays
    random
    B
    B_oneday
    Conflict
    Seqboost
    Seqboost_nolr
    Bhand
    Bhand_oneday
    BQ
    BK
    HandSeq
    HandSeq2
    Random
    
    OnlyR
    OnlyQ
'''

'''
Modelle:

    Vbm_lr
    Repbias_lr
    Repbias_CongConflict_lr
    OnlyQ_lr
'''

post_pred = 0
model_day1 = 'Repbias_Conflict_lr'
models_day2 = ['Repbias_Conflict_lr', 'Repbias_Conflict_nolr']
num_inf_steps = 2
halting_rtol = 1e-02 # for MLE estimation
num_agents = 60
posterior_pred_samples = 2

#%%
'''
    Simulate data in parallel
'''

group = [0]*(num_agents//4)
group.extend([1]*(num_agents//4))
group.extend([2]*(num_agents//4))
group.extend([3]*(num_agents//4))

# utils.plot_grouplevel(group_behav_df)
#%%
'''
    Simulate Data Day 1
'''

day = 1
groupdata_dict_day1, group_behav_df_day1, params_sim_df_day1, agent_day1 = utils.simulate_data(model_day1, 
                                                                      num_agents,
                                                                      group = group,
                                                                      day = day)

#%%
'''
    Fit day 1
'''
# import time
# time.sleep(21*3600)

'''
    Inference
'''
"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent(model_day1, 
                         group, 
                         num_agents = num_agents)

Q_init_day1 = agent.Q_init

print("===== Starting inference of day 1 =====")
"----- Start Inference"
infer = inferencemodels.GeneralGroupInference(agent, groupdata_dict_day1)
agent_elbo_tuple = infer.infer_posterior(iter_steps = num_inf_steps, num_particles = 10)

"----- Sample parameter estimates from posterior and add information to DataFrame"
if post_pred:
    firstlevel_df, secondlevel_df = infer.posterior_predictives(n_samples = posterior_pred_samples)
    
else:
    firstlevel_df = infer.sample_posterior(n_samples = posterior_pred_samples)
    secondlevel_df = None

firstlevel_df['group'] = firstlevel_df['ag_idx'].map(lambda x: groupdata_dict_day1['group'][0][x])
firstlevel_df['model'] = [model_day1]*len(firstlevel_df)

ID_df = group_behav_df_day1.loc[:, ['ag_idx']].drop_duplicates()

"----- MLE & IC"
max_log_like, mle_locs = infer.train_mle(halting_rtol = halting_rtol)
BIC, AIC = infer.compute_IC()

"----- Q_init for next day"
param_names = agent.param_names
inf_mean_df = firstlevel_df.loc[:, [*param_names, 
                      'ag_idx', 
                      'ID']].groupby(['ag_idx', 
                                      'ID'], as_index = False).mean()
assert torch.all(torch.tensor(inf_mean_df['ag_idx']) == torch.tensor(groupdata_dict_day1['ag_idx'][0]))
assert all([inf_mean_df['ID'][i] == groupdata_dict_day1['ID'][0][i] for i in range(num_agents)])
                                      
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

"----- Store results"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
extra_storage = (Q_init_day1,
                 agent.Q[-1].detach(),
                 '',
                 'no preceding model',
                 max_log_like,
                 mle_locs)

pickle.dump( (firstlevel_df, 
              group_behav_df_day1,
              (infer.loss, BIC, AIC), 
              params_sim_df_day1, 
              agent_elbo_tuple, 
              extra_storage), 
            open(f"parameter_recovery/recovery_model_{model_day1}_day{day}_{timestamp}_{num_agents}agents.p", "wb" ) )

'''
    Fit day 2
'''

'''
    Simulate Data Day 2
'''
day = 2
del groupdata_dict_day1, group_behav_df_day1, params_sim_df_day1, agent_day1
# blocks_day2 = [3,7]
for model_day2 in models_day2:
    groupdata_dict_day2, group_behav_df_day2, params_sim_df_day2, agent_day2 = utils.simulate_data(model_day2, 
                                                                          num_agents,
                                                                          group = group,
                                                                          day = day,
                                                                          Q_init = Q_init_day2.detach())
    
    agent = utils.init_agent(model_day2, 
                             group, 
                             num_agents = num_agents,
                             Q_init = Q_init_day2.detach())
    
    print("===== Starting inference of day 2 =====")
    "----- Start Inference"
    infer = inferencemodels.GeneralGroupInference(agent, groupdata_dict_day2)
    agent_elbo_tuple = infer.infer_posterior(iter_steps = num_inf_steps, num_particles = 10)
    
    "----- Sample parameter estimates from posterior and add information to DataFrame"
    if post_pred:
        firstlevel_df, secondlevel_df = infer.posterior_predictives(n_samples = posterior_pred_samples)
        
    else:
        firstlevel_df = infer.sample_posterior(n_samples = posterior_pred_samples)
        secondlevel_df = None
        
    firstlevel_df['group'] = firstlevel_df['ag_idx'].map(lambda x: groupdata_dict_day2['group'][0][x])
    firstlevel_df['model'] = [model_day2]*len(firstlevel_df)
    
    ID_df = group_behav_df_day2.loc[:, ['ag_idx']].drop_duplicates()
    # firstlevel_df['ID'] = firstlevel_df['ag_idx'].map(lambda x: groupdata_dict_day2['ID'][0][x])
    
    "----- MLE & IC"
    max_log_like, mle_locs = infer.train_mle(halting_rtol = halting_rtol)
    BIC, AIC = infer.compute_IC()
    
    # "----- Q_init for next day"
    # Q_init_day2 = agent.Q[-1].detach().mean(axis=0)[None, ...]
    
    
    "----- Store results"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    extra_storage = (Q_init_day2,  # initial Q-values
                     agent.Q[-1].detach(), # final Q-values
                     '', # blocks
                     model_day1, # preceding model
                     max_log_like,
                     mle_locs) 
    
    pickle.dump( (firstlevel_df, 
                  group_behav_df_day2, 
                  (infer.loss, BIC, AIC), 
                  params_sim_df_day2, 
                  agent_elbo_tuple, 
                  extra_storage), 
                open(f"parameter_recovery/recovery_model2_{model_day2}_model1_{model_day1}_day{day}_{timestamp}_{num_agents}agents.p", "wb" ) )


