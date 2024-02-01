#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:00:09 2023

@author: sascha
"""

import pickle
import utils

exp_behav_dict, expdata_df = utils.get_groupdata('/home/sascha/Desktop/vbm_torch/behav_data/', getall = False)
pickle.dump((exp_behav_dict, expdata_df), open("behav_data/preproc_data.p", "wb" ) )

# curr_day = 1
# idx = -1
# while curr_day == 1:
#     idx += 1
#     if exp_behav_dict['blockidx'][idx][0] <= 5:
#         curr_day = 1
        
#     elif exp_behav_dict['blockidx'][idx][0] > 5:
#         curr_day = 2
        
"Day 1"
exp_behav_dict_day1= {}
for key in exp_behav_dict.keys():   
    exp_behav_dict_day1[key] = exp_behav_dict[key][0:2886]

pickle.dump((exp_behav_dict_day1, expdata_df[expdata_df['blockidx'] <= 5]), open("behav_data/preproc_data_day1.p", "wb" ) )

"Day 2"
exp_behav_dict_day2= {}
for key in exp_behav_dict.keys():   
    exp_behav_dict_day2[key] = exp_behav_dict[key][2886:]

pickle.dump((exp_behav_dict_day2, expdata_df[expdata_df['blockidx'] > 5]), open("behav_data/preproc_data_day2.p", "wb" ) )

#%% Published data
exp_behav_dict, expdata_df = utils.get_old_groupdata('/home/sascha/Desktop/vbm_torch/behav_data/published/Data/', getall = True, oldpub = True)

pickle.dump((exp_behav_dict, expdata_df), open("behav_data/preproc_data_old_published_all.p", "wb" ) )