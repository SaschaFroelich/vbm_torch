#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 13:36:58 2023

@author: sascha
"""

import utils
import inferencemodels
from datetime import datetime
import pickle
import torch

model = 'Conflict'
resim =  0 # whether to simulate agents with inferred parameters
method = 'svi' # "svi" or "mcmc"
num_agents = 60

assert num_agents%4 == 0, "num_agents must be divisible by 4."
# k = 4.
print(f"Running model {model}")

'''
    Simulate data in parallel
'''
group = [0]*(num_agents//4)
group.extend([1]*(num_agents//4))
group.extend([2]*(num_agents//4))
group.extend([3]*(num_agents//4))

params = {}
params['lr_day1'] = torch.ones(num_agents)*0.01
params['theta_Q_day1'] = torch.ones(num_agents)*5
params['theta_rep_day1'] = torch.ones(num_agents)*1.5
params['conflict_param_day1'] = torch.ones(num_agents)*0

params['lr_day2'] = torch.ones(num_agents)*0.01
params['theta_Q_day2'] = torch.ones(num_agents)*5
params['theta_rep_day2'] = torch.ones(num_agents)*1.5
params['conflict_param_day2'] = torch.ones(num_agents)*10

groupdata_dict, group_behav_df, _, params_sim_df = utils.simulate_data(model, 
                                                                      num_agents,
                                                                      group = group,
                                                                      params = params)

utils.plot_grouplevel(group_behav_df)