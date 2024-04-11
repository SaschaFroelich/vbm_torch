#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:45:43 2024

@author: sascha
"""

import utils
import inferencemodels
from datetime import datetime
import pickle
import torch
import numpy as np
import pandas as pd

sim_model_1_day1 = 'Repbias_Conflict_Repdiff_lr'
sim_model_2_day1 = 'Repbias_lr'

num_agents = 60

'''
    Simulate data in parallel
'''

group = [0]*(num_agents//4)
group.extend([1]*(num_agents//4))
group.extend([2]*(num_agents//4))
group.extend([3]*(num_agents//4))

# utils.plot_grouplevel(group_behav_df)

'''
    Simulate Data Day 1
'''

day = 1
er_day1 = torch.rand((4, num_agents))*0.2

groupdata_dict_day1_1, group_behav_df_day1_1, _, _ = utils.simulate_data(sim_model_1_day1, 
                                                                      num_agents,
                                                                      group = group,
                                                                      day = day,
                                                                      STT = False,
                                                                      errorrates = er_day1)

groupdata_dict_day1_2, group_behav_df_day1_2, _, _ = utils.simulate_data(sim_model_2_day1, 
                                                                      num_agents,
                                                                      group = group,
                                                                      day = day,
                                                                      STT = False,
                                                                      errorrates = er_day1)



groupdata_dict_day1 = {}
for key in groupdata_dict_day1_1.keys():
    groupdata_dict_day1[key] = []
    
    for listidx in range(len(groupdata_dict_day1_1[key])):
        insertlist = groupdata_dict_day1_1[key][listidx][0:num_agents//2]
        insertlist.extend(groupdata_dict_day1_2[key][listidx][num_agents//2:])
        
        groupdata_dict_day1[key].append(insertlist)

group_behav_df_day1_1 = group_behav_df_day1_1[group_behav_df_day1_1['ag_idx'] < 30]
group_behav_df_day1_2 = group_behav_df_day1_2[group_behav_df_day1_2['ag_idx'] >= 30]

group_behav_df_day1 = pd.concat((group_behav_df_day1_1, group_behav_df_day1_2))

pickle.dump((groupdata_dict_day1, group_behav_df_day1), open("behav_data/mixed_sim_data_2_day1.p", "wb" ) )