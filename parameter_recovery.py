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
    Repbias_Conflict
    Repbias_CongConflict_lr
    OnlyQ_lr
'''

model_day1 = 'Repbias_lr'
model_day2 = 'Repbias_lr'
num_inf_steps = 4_000
halting_rtol = 1e-06 # for MLE estimation
num_agents = 60
posterior_pred_samples = 4_000

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
# import time
# time.sleep(5*3600)
blocks_day1 = [0, 3]
groupdata_dict_day1, group_behav_df_day1, _, params_sim_df_day1, agent_day1 = utils.simulate_data(model_day1, 
                                                                      num_agents,
                                                                      group = group,
                                                                      blocks = blocks_day1)

'''
    Simulate Data Day 2
'''
# import time
# time.sleep(5*3600)

blocks_day2 = [3, 7]
groupdata_dict_day2, group_behav_df_day2, _, params_sim_df_day2, agent_day2 = utils.simulate_data(model_day2, 
                                                                      num_agents,
                                                                      group = group,
                                                                      blocks = blocks_day2,
                                                                      Q_init = agent_day1.Q[-1])

#%%
'''
    Fit day 1
'''
# import time
# time.sleep(8*3600)

# blocks_day1 = [0,3]

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
firstlevel_df, secondlevel_df = infer.sample_posterior(n_samples = posterior_pred_samples)

firstlevel_df['group'] = firstlevel_df['ag_idx'].map(lambda x: groupdata_dict_day1['group'][0][x])
firstlevel_df['model'] = [model_day1]*len(firstlevel_df)

ID_df = group_behav_df_day1.loc[:, ['ag_idx']].drop_duplicates()
# firstlevel_df['ID'] = firstlevel_df['ag_idx'].map(lambda x: groupdata_dict_day1['ID'][0][x])

"----- MLE & IC"
max_log_like, mle_locs = infer.train_mle(halting_rtol = halting_rtol)
BIC, AIC = infer.compute_IC()

"----- Q_init for next day"
Q_init_day2 = agent.Q[-1].detach().mean(axis=0)[None, ...]

"----- Store results"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
extra_storage = (Q_init_day1,
                 agent.Q[-1].detach(),
                 blocks_day1,
                 'no preceding model',
                 max_log_like,
                 mle_locs)

pickle.dump( (firstlevel_df, 
              group_behav_df_day1,
              (infer.loss, BIC, AIC), 
              params_sim_df_day1, 
              agent_elbo_tuple, 
              extra_storage), 
            open(f"parameter_recovery/recovery_model_{model_day1}_blocks{blocks_day1[0]}{blocks_day1[1]}_{timestamp}_{num_agents}agents.p", "wb" ) )

'''
    Fit day 2
'''

# blocks_day2 = [3,7]

agent = utils.init_agent(model_day2, 
                         group, 
                         num_agents = num_agents,
                         Q_init = Q_init_day2)

print("===== Starting inference of day 2 =====")
"----- Start Inference"
infer = inferencemodels.GeneralGroupInference(agent, groupdata_dict_day2)
agent_elbo_tuple = infer.infer_posterior(iter_steps = num_inf_steps, num_particles = 10)

"----- Sample parameter estimates from posterior and add information to DataFrame"
firstlevel_df, secondlevel_df = infer.sample_posterior(n_samples = posterior_pred_samples)

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
                 blocks_day2, # blocks
                 model_day1, # preceding model
                 max_log_like,
                 mle_locs) 

pickle.dump( (firstlevel_df, 
              group_behav_df_day2, 
              (infer.loss, BIC, AIC), 
              params_sim_df_day2, 
              agent_elbo_tuple, 
              extra_storage), 
            open(f"parameter_recovery/recovery_model2_{model_day2}_model1_{model_day1}_blocks{blocks_day2[0]}{blocks_day2[1]}_{timestamp}_{num_agents}agents.p", "wb" ) )


