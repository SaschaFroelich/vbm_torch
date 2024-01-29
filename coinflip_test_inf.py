#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Mon Jan 29 10:44:36 2024
    
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

num_inf_steps = 2000

exp_behav_dict, expdata_df = pickle.load(open("behav_data/preproc_data.p", "rb" ))


num_agents = len(expdata_df['ag_idx'].unique())
group = exp_behav_dict['group'][0]

'''
    Inference
'''
"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent('Coinflip_test', 
                         group, 
                         num_agents = num_agents)

num_trials = 500

coinflip_data = torch.tensor(np.random.choice([0,1], p=[0.11, 0.89], size = (num_agents, num_trials))) # 0 heads, 1 tails

infer = inferencemodels.CoinflipGroupInference(agent, coinflip_data)

#%%
'''
    Inference
'''
print("===== Starting inference =====")
"----- Start Inference"
infer.infer_posterior(iter_steps = num_inf_steps, num_particles = 10)

post_sample_df = infer.sample_posterior()

#%%
'''
    MLE
'''
max_log_like, mle_locs = infer.train_mle(iter_steps = 800, halting_rtol=1e-01)
BIC, AIC = infer.compute_IC()
