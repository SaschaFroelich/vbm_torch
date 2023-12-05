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
'''

model = 'SeqConflictHand'
resim =  0 # whether to simulate agents with inferred parameters
method = 'svi' # "svi" or "mcmc"
num_agents = 60

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

groupdata_dict, group_behav_df, _, params_sim_df = utils.simulate_data(model, 
                                                                      num_agents,
                                                                      group = group)

# utils.plot_grouplevel(group_behav_df)
#%%
'''
Inference
'''
import time
time.sleep(10*3600)
"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent(model, 
                         group, 
                         num_agents = num_agents)

print("===== Starting inference =====")
"----- Start Inference"
infer = inferencemodels.GeneralGroupInference(agent, groupdata_dict)
agent_elbo_tuple = infer.infer_posterior(iter_steps = 8_000, num_particles = 10)

"----- Sample parameter estimates from posterior"
post_sample_df = infer.sample_posterior()
post_sample_df['group'] = post_sample_df['ag_idx'].map(lambda x: groupdata_dict['group'][0][x])
post_sample_df['model'] = [model]*len(post_sample_df)
BIC, AIC = infer.compute_IC()

"----- Save results to file"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# posterior_params = (infer.guide()['m_locs'], infer.guide()['st_locs'])
pickle.dump((post_sample_df, group_behav_df, (infer.loss, BIC, AIC), params_sim_df, agent_elbo_tuple), 
            open(f"parameter_recovery/param_recov_model_{model}_{timestamp}_{num_agents}agents.p", "wb" ) )

#%%
# log_like = infer.compute_ll()

