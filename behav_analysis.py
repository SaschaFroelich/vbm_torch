#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:04:04 2023

@author: sascha
"""

import pickle
import pandas as pd
import numpy as np

exp_behav_dict, expdata_df = pickle.load(open("behav_data/preproc_data.p", "rb" ))

# expdata_df = expdata_df[expdata_df['choices'] != -1]
# expdata_df = expdata_df[expdata_df['jokertypes'] != -1]

num_agents = 60
'''
    Prob. of choosing GD if the last x (last dim) jokers were of the same jokertype.
    jokertypes: -1/0/1/2 : no joker/random/congruent/incongruent
'''
num_choices_gd = np.zeros((num_agents, 3, 20))
counts = np.zeros((num_agents, 3, 20))
for ID_idx in range(len(expdata_df['ID'].unique())):
    prev_type = -1
    num_prev = 0
    
    ag_df = expdata_df[expdata_df['ID'] == expdata_df['ID'].unique()[ID_idx]]
    
    for row_idx in range(len(ag_df)):
        current_jokertype = ag_df.loc[row_idx, 'jokertypes']
        if current_jokertype != -1 and current_jokertype != 0:
            # print(current_jokertype)
            if ag_df.loc[row_idx, 'choices_GD'] == 0 or ag_df.loc[row_idx, 'choices_GD'] == 1:
                if current_jokertype == prev_type:
                    # print("Kuckuck")
                    num_choices_gd[ID_idx, current_jokertype, num_prev] += ag_df.loc[row_idx, 'choices_GD']
                    counts[ID_idx, current_jokertype, num_prev] += 1
                    num_prev += 1
                    
                elif current_jokertype != prev_type:
                    num_choices_gd[ID_idx, current_jokertype, 0] += ag_df.loc[row_idx, 'choices_GD']
                    counts[ID_idx, current_jokertype, 0] += 1
                    num_prev = 0
                    
                prev_type = current_jokertype
                    
            elif ag_df.loc[row_idx, 'choices_GD'] == -2:
                prev_type = -1
                num_prev = 0
            
props = num_choices_gd[:, [1,2], 0:4] / counts[:, [1,2], 0:4]
print(props.mean(axis=0))
print(props.std(axis=0))

import matplotlib.pyplot as plt
import numpy as np

"Jokertypes 1"
# Sample data
categories = ['0', '1', '2', '3']
values = props.mean(axis=0)[0,:]
errors = props.std(axis=0)[0,:]

# Create a bar plot with error bars
plt.bar(categories, values, yerr=errors, capsize=5, color='skyblue', alpha=0.7)

# Add labels and title
plt.xlabel('num preceding congruent DTT')
plt.ylabel('% GD choice')
plt.title('Congruent DTT')

# Show the plot
plt.show()

"Jokertypes 2"
# Sample data
categories = ['0', '1', '2', '3']
values = props.mean(axis=0)[1,:]
errors = props.std(axis=0)[1,:]

# Create a bar plot with error bars
plt.bar(categories, values, yerr=errors, capsize=5, color='skyblue', alpha=0.7)

# Add labels and title
plt.xlabel('num preceding incongruent DTT')
plt.ylabel('% GD choice')
plt.title('Incongruent DTT')

# Show the plot
plt.show()

'''
    Prob. of choosing GD suboptimal response in the last joker.
'''
print("---> Influence of previous joker response.")
'''
    0th column: suboptimal choice
    1st column: optimal choice
'''
num_choices_gd_seq = np.zeros((num_agents, 2)) 
num_choices_gd_rand = np.zeros((num_agents, 2)) 
for ID_idx in range(len(expdata_df['ID'].unique())):
    prev_choice = -1
    
    ag_df = expdata_df[expdata_df['ID'] == expdata_df['ID'].unique()[ID_idx]]
    
    for row_idx in range(len(ag_df)):
        current_jokertype = ag_df.loc[row_idx, 'jokertypes']
        if current_jokertype != -1 and current_jokertype == 0:
            if prev_choice == 0 or prev_choice == 1:
                num_choices_gd_rand[ID_idx, prev_choice] += ag_df.loc[row_idx, 'choices_GD']
                
        elif current_jokertype != -1 and current_jokertype != 0:
            if prev_choice == 0 or prev_choice == 1:
                num_choices_gd_seq[ID_idx, prev_choice] += ag_df.loc[row_idx, 'choices_GD']
                
        prev_choice = ag_df.loc[row_idx, 'choices_GD']

print("For seq condition %.4f +- %.4f"%((num_choices_gd_seq[:,1]/num_choices_gd_seq[:,0]).mean(),
                                        (num_choices_gd_seq[:,1]/num_choices_gd_seq[:,0]).std()))

print("For random condition %.4f +- %.4f"%((num_choices_gd_rand[:,1]/num_choices_gd_seq[:,0]).mean(),
                                           (num_choices_gd_rand[:,1]/num_choices_gd_rand[:,0]).std()))

import scipy.stats
t,p = scipy.stats.ttest_rel(num_choices_gd_seq[:,0], num_choices_gd_seq[:,1])
print(t)
print(p)

t,p = scipy.stats.ttest_rel(num_choices_gd_rand[:,0], num_choices_gd_rand[:,1])
print(t)
print(p)
