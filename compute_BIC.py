#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:39:57 2023

@author: sascha
"""

import utils
import inferencemodels
from datetime import datetime
import pickle

sim_with = 'Bhand'
fit_with = 'Vbm_twodays'
num_agents = 48
assert num_agents%4 == 0, "num_agents must be divisible by 4."
#%%
group = [0]*(num_agents//4)
group.extend([1]*(num_agents//4))
group.extend([2]*(num_agents//4))
group.extend([3]*(num_agents//4))

print(f"Data simulation with model {sim_with} for {num_agents} agents.")
groupdata_dict, group_behav_df, _, params_sim_df = utils.simulate_data(sim_with, 
                                                                      num_agents,
                                                                      group = group)

print(f"Model fit of model {fit_with} for {num_agents} agents.")
"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent(fit_with, 
                         group, 
                         num_agents = num_agents)

print("===== Starting inference =====")
"----- Start Inference"
infer = inferencemodels.GeneralGroupInference(agent, groupdata_dict)
infer.infer_posterior(iter_steps = 6_000, num_particles = 10)

"----- Sample parameter estimates from posterior"
post_sample_df = infer.sample_posterior()
post_sample_df['group'] = post_sample_df['ag_idx'].map(lambda x: groupdata_dict['group'][0][x])
post_sample_df['sim_with'] = [sim_with]*len(post_sample_df)
post_sample_df['fit_with'] = [fit_with]*len(post_sample_df)
BIC = infer.compute_IC()

"----- Save results to file"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
pickle.dump((post_sample_df, group_behav_df, (infer.loss, BIC), params_sim_df), open(f"BICs/sim_{sim_with}_fit_{fit_with}_{timestamp}.p", "wb" ) )
