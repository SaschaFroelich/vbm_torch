#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:00:09 2023

@author: sascha
"""

import pickle
import utils

exp_behav_dict, expdata_df = utils.get_groupdata('/home/sascha/Desktop/AST2_All_Data/', getall = True)
pickle.dump((exp_behav_dict, expdata_df), open("behav_data/preproc_data_all.p", "wb" ) )

exp_behav_dict, expdata_df = utils.get_groupdata('/home/sascha/Desktop/AST2_All_Data/', getall = False)
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

utils.check_debriefing_quest(expdata_df)

pickle.dump((exp_behav_dict_day1, expdata_df[expdata_df['blockidx'] <= 5]), open("behav_data/preproc_data_day1.p", "wb" ) )

"Day 2"
exp_behav_dict_day2= {}
for key in exp_behav_dict.keys():   
    exp_behav_dict_day2[key] = exp_behav_dict[key][2886:]

pickle.dump((exp_behav_dict_day2, expdata_df[expdata_df['blockidx'] > 5]), open("behav_data/preproc_data_day2.p", "wb" ) )

#%%
'''
    Published data
'''
exp_behav_dict, expdata_df = utils.get_old_groupdata('/home/sascha/Desktop/vbm_torch/behav_data/published/Data/', getall = True, oldpub = True)

pickle.dump((exp_behav_dict, expdata_df), open("behav_data/preproc_data_old_published_all.p", "wb" ) )

#%% 
'''
    RT AST
'''
exp_behav_dict, expdata_df = utils.get_groupdata('/home/sascha/proni/AST/AST2/AST2RT_Online/data/', getall = False, RTAST = True)
pickle.dump((exp_behav_dict, expdata_df), open("behav_data/preproc_data_RTAST.p", "wb" ) )

"Day 1"
exp_behav_dict_day1= {}
for key in exp_behav_dict.keys():   
    exp_behav_dict_day1[key] = exp_behav_dict[key][0:2886]

pickle.dump((exp_behav_dict_day1, expdata_df[expdata_df['blockidx'] <= 5]), open("behav_data/preproc_data_RTAST_day1.p", "wb" ) )

"Day 2"
exp_behav_dict_day2= {}
for key in exp_behav_dict.keys():   
    exp_behav_dict_day2[key] = exp_behav_dict[key][2886:]

pickle.dump((exp_behav_dict_day2, expdata_df[expdata_df['blockidx'] > 5]), open("behav_data/preproc_data_RTAST_day2.p", "wb" ) )


#%%

import pickle
import pandas as pd
import numpy as np

preproc_df = pickle.load(open('/home/sascha/Desktop/vbm_torch/behav_data/preproc_data.p', "rb" ))[1]

preproc_all_old_df = pickle.load(open('/home/sascha/Desktop/vbm_torch/behav_data/preproc_data_all_old.p', "rb" ))[1]
overview_csv = pd.read_csv('/home/sascha/Downloads/overview.csv')

# for ID in overview_csv['PC']:
#     if ID not in preproc_all_df['ID'].unique():
#         print(f"{ID} not in.")

num_pp = 0
for ID in overview_csv['preproc_data_all.p']:
    if ID not in preproc_all_df['ID'].unique():
        print(f"{ID} not in preproc_all_df.")
            
    else:
        num_pp += 1
        
num_pp = 0
for ID in preproc_all_df['ID'].unique():
    if ID not in overview_csv['preproc_data_all.p'].unique():
        print(f"{ID} not in preproc_all_df.")
            
    else:
        num_pp += 1
        
num_pp = 0
for ID in preproc_all_df['ID'].unique():
    if ID not in overview_csv['preproc_data_all.p'].unique():
        print(f"{ID} missing.")
            
    else:
        num_pp += 1
        

#%%

"Studies of day 1"
prolific_df_1 = pd.read_csv('/home/sascha/Desktop/AST2_All_Data/prolific_export_6564759e1ad8e31c4aad6c99.csv')
prolific_df_2 = pd.read_csv('/home/sascha/Desktop/AST2_All_Data/prolific_export_645d47bb8df383730a684319.csv')
prolific_df_3 = pd.read_csv('/home/sascha/Desktop/AST2_All_Data/prolific_export_64c3628ddcc1f9e7babd121b.csv')
prolific_df_4 = pd.read_csv('/home/sascha/Desktop/AST2_All_Data/prolific_export_62e23b5aa496e999ec194973.csv')

prolific_df_1['Study']=1
prolific_df_2['Study']=2
prolific_df_3['Study']=3
prolific_df_4['Study']=4


df = pd.concat([prolific_df_1, prolific_df_2, prolific_df_3, prolific_df_4])

for ID in df['Participant id'].unique():
    if len(df[df['Participant id'] == ID]) > 1:
        print("=======")
        print(f"{ID}: {len(df[df['Participant id'] == ID])}")
        for index, row in df[df['Participant id'] == ID].iterrows():
            print(f"{row['Study']}" + ", " + row['Status'])
            
overview_csv = pd.read_csv('/home/sascha/Desktop/AST2_All_Data/overview_python.csv')
            
IDs = []
status = []
for index, row in df.iterrows():
    if row['Participant id'] not in overview_csv['Prolific ID'].unique():
        print(f"{row['Participant id']} not found in overview, status {row['Status']}")
        
        
        if row['Participant id'] not in IDs:
            IDs.append(row['Participant id'])
            status.append(row['Status'])

bla = 0
for ID in overview_csv[(overview_csv['Group'].isnull()) & ~(overview_csv['Prolific ID'].isnull())]['Prolific ID'].unique():
    if ID in df['Participant id'].unique():
        pass
        # print(f"{ID}: {df[df['Participant id']==ID]['Status'].iloc[0]}")
        
    if ID in df['Participant id'].unique():
        if df[df['Participant id']==ID]['Status'].iloc[0] == 'REJECTED':
            print(f"{ID}: {df[df['Participant id']==ID]['Status'].iloc[0]}")
            bla += 1
        
    # df[df['Participant id']==ID]['Status']
            
#%%

preproc_all_df = pickle.load(open('/home/sascha/Desktop/vbm_torch/behav_data/preproc_data_all.p', "rb" ))[1]