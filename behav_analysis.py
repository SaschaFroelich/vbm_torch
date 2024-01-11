#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:04:04 2023

@author: sascha
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import numpy as np

exp_behav_dict, expdata_df = pickle.load(open("behav_data/preproc_data.p", "rb" ))
# expdata_df = expdata_df[expdata_df['choices'] != -1]
# expdata_df = expdata_df[expdata_df['jokertypes'] != -1]

num_agents = 60

#%%
'''
    Prob of choosing GD choice if chose GD choice in prev joker
'''
print("---> Influence of previous joker response.")
'''
    0th column: suboptimal choice
    1st column: optimal choice
'''
"Sequential condition"
num_choices_gd_seq = np.zeros((num_agents, 2)) # num of gd choices where choice in prev joker was not GD (0th column) or GD (1st column)
occurrences_seq = np.zeros((num_agents, 2))

"Random Condition"
num_choices_gd_rand = np.zeros((num_agents, 2)) 
occurrences_rand = np.zeros((num_agents, 2))

for ID_idx in range(len(expdata_df['ID'].unique())):
    prev_choice = -1
    prevprev_choice = -1
    
    ag_df = expdata_df[expdata_df['ID'] == expdata_df['ID'].unique()[ID_idx]]
    
    for row_idx in range(len(ag_df)):
        current_jokertype = ag_df.loc[row_idx, 'jokertypes']
        
        if current_jokertype == 0:
            "Random Condition"
            if prev_choice == 0 or prev_choice == 1:
                num_choices_gd_rand[ID_idx, prev_choice] += ag_df.loc[row_idx, 'choices_GD']
                occurrences_rand[ID_idx, prev_choice] += 1
                
        elif current_jokertype > 0:
            "Sequential Condition"
            if prev_choice == 0 or prev_choice == 1:
                num_choices_gd_seq[ID_idx, prev_choice] += ag_df.loc[row_idx, 'choices_GD']
                occurrences_seq[ID_idx, prev_choice] += 1
        
        prevprev_choice = prev_choice
        prev_choice = ag_df.loc[row_idx, 'choices_GD']
        
categories = ['previously not GD', 'previously GD']
values = (num_choices_gd_rand/ occurrences_rand).mean(axis=0)
errors = (num_choices_gd_rand/ occurrences_rand).std(axis=0)

# Create a bar plot with error bars
plt.bar(categories, values, yerr=errors, capsize=5, color='skyblue', alpha=0.7)

# Add labels and title
plt.xlabel('previous action')
plt.ylabel('% GD choice')
plt.title('Random Condition')

# Show the plot
plt.show()

categories = ['previously not GD', 'previously GD']
values = (num_choices_gd_seq/ occurrences_seq).mean(axis=0)
errors = (num_choices_gd_seq/ occurrences_seq).std(axis=0)

# Create a bar plot with error bars
plt.bar(categories, values, yerr=errors, capsize=5, color='skyblue', alpha=0.7)

# Add labels and title
plt.xlabel('previous action')
plt.ylabel('% GD choice')
plt.title('sequential Condition')

# Show the plot
plt.show()

import scipy.stats
t,p = scipy.stats.ttest_rel((num_choices_gd_seq/ occurrences_seq)[:,0], (num_choices_gd_seq/ occurrences_seq)[:,1])
print(t)
print(p)

t,p = scipy.stats.ttest_rel((num_choices_gd_rand/ occurrences_rand)[:,0], (num_choices_gd_rand/ occurrences_rand)[:,1])
print(t)
print(p)

#%%

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

#%%
'''
    ΔER & ΔRT are measures of how strongly the sequences are learned.
'''

'''
    ΔER & ΔRT are correlated.
'''
fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['ER_diff_stt_day2'])
sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', ax = ax)
ax.text(22, 0.035, 'r=%.4f, p = %.4f'%(r,p))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.set_xlabel(r'$\Delta$RT')
ax.set_ylabel(r'$\Delta$ER')
ax.axhline(0, color = 'k', linewidth = 0.5)
ax.axvline(0, color = 'k', linewidth = 0.5)
plt.show()

'''
    ΔRT is more strongly correlated with RIspread than ΔER is.
'''
fig, ax = plt.subplots(1,2, figsize=(15,5))
slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['RIspread_day2'])
sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'RIspread_day2', ax = ax[0])
ax[0].text(22,-0.03, 'r=%.4f, p = %.4f'%(r,p))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[0].axhline(0, color = 'k', linewidth = 0.5)
ax[0].axvline(0, color = 'k', linewidth = 0.5)
ax[0].set_ylabel('RI-Spread (pp)')
ax[0].set_xlabel(r'$\Delta$RT (ms)')

slope, intercept, r, p, std_err = stats.linregress(complete_df['ER_diff_stt_day2'], complete_df['RIspread_day2'])
sns.regplot(data = complete_df, x = 'ER_diff_stt_day2', y = 'RIspread_day2', ax = ax[1])
ax[1].text(0.023, -0.03, 'r=%.4f, p = %.4f'%(r,p))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[1].axhline(0, color = 'k', linewidth = 0.5)
ax[1].axvline(0, color = 'k', linewidth = 0.5)
ax[1].set_xlabel(r'$\Delta$ER (pp)')
ax[1].set_ylabel('RI-Spread (pp)')
plt.show()

'''
    ΔRT & ΔEr with CI Spread like in paper
'''
fig, ax = plt.subplots(1,2, figsize=(15,5))
slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['CIspread_day2'])
sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'CIspread_day2', ax = ax[0])
ax[0].text(21,-0.07, 'r=%.4f, p = %.4f'%(r,p))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[0].axhline(0, color='k', linewidth=0.5)
ax[0].axvline(0, color='k', linewidth=0.5)
ax[0].set_xlabel(r'$\Delta$RT (ms)')
ax[0].set_ylabel(r'$\Delta$HRCF (Cong-Inc) (pp)')

slope, intercept, r, p, std_err = stats.linregress(complete_df['ER_diff_stt_day2'], complete_df['CIspread_day2'])
sns.regplot(data = complete_df, x = 'ER_diff_stt_day2', y = 'CIspread_day2', ax = ax[1])
ax[1].text(0.022,-0.07, 'r=%.4f, p = %.4f'%(r,p))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[1].axhline(0, color='k', linewidth=0.5)
ax[1].axvline(0, color='k', linewidth=0.5)
ax[1].set_xlabel(r'$\Delta$ER (pp)')
ax[1].set_ylabel(r'$\Delta$HRCF (Cong-Inc) (pp)')
plt.show()


'''
    The correlations with CRspread are not that strong, since there is a 
    ceiling effect for congruent DTT.
'''
fig, ax = plt.subplots(1,2, figsize=(15,5))
slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['CRspread_day2'])
sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'CRspread_day2', ax = ax[0])
ax[0].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[0].axhline(0)
ax[0].axvline(0)

slope, intercept, r, p, std_err = stats.linregress(complete_df['ER_diff_stt_day2'], complete_df['CRspread_day2'])
sns.regplot(data = complete_df, x = 'ER_diff_stt_day2', y = 'CRspread_day2', ax = ax[1])
ax[1].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[1].axhline(0)
ax[1].axvline(0)
plt.show()

'''
    Generally, ΔRT and ΔER do not strongly differentiate 
    between goal-directed and non-goal-directed participants...
    
    ΔER differentiates stronger.
    
    For N = 100 dataset, the differentiation is the case
    
    ToDo: Outlier detection.
'''
fig, ax = plt.subplots(1,2, figsize=(15,5))
slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['hpcf_rand_day2'])
sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[0])
ax[0].text(0,0.6, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
# ax[0].axhline(0)
# ax[0].axvline(0)

slope, intercept, r, p, std_err = stats.linregress(complete_df['ER_diff_stt_day2'], complete_df['hpcf_rand_day2'])
sns.regplot(data = complete_df, x = 'ER_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[1])
ax[1].text(0,0.6, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
# ax[1].axhline(0)
# ax[1].axvline(0)
plt.show()

no_outlier_df = complete_df.copy()
no_outlier_df['hpcf_rand_day2 zscore'] = stats.zscore(no_outlier_df['hpcf_rand_day2'])
no_outlier_df = no_outlier_df[abs(no_outlier_df['hpcf_rand_day2 zscore']) < 3]

"With Outlier Detection"
fig, ax = plt.subplots(1,2, figsize=(15,5))
slope, intercept, r, p, std_err = stats.linregress(no_outlier_df['RT_diff_stt_day2'], no_outlier_df['hpcf_rand_day2'])
sns.regplot(data = no_outlier_df, x = 'RT_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[0])
ax[0].text(21,0.97, 'r=%.4f, p = %.4f'%(r,p))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
# ax[0].axhline(0, color = 'k', linewidth=0.5)
ax[0].set_xlabel(r'$\Delta$RT')
ax[0].set_ylabel(r'HRCF (Random DTT)')
ax[0].axvline(0, color = 'k', linewidth=0.5)

slope, intercept, r, p, std_err = stats.linregress(no_outlier_df['ER_diff_stt_day2'], no_outlier_df['hpcf_rand_day2'])
sns.regplot(data = no_outlier_df, x = 'ER_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[1])
ax[1].text(0.021,0.97, 'r=%.4f, p = %.4f'%(r,p))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
# ax[1].axhline(0, color = 'k', linewidth=0.5)
ax[1].axvline(0, color = 'k', linewidth=0.5)
ax[1].set_xlabel(r'$\Delta$ER')
ax[1].set_ylabel(r'HRCF (Random DTT)')
plt.show()

'''
    ... and a higher goal-directedness in random trials does not present
    a different RI-Spread.
    
    Problem: Influence of hpcf_rand_day2 on CR-Spread not easy to evaluate
    bc of ceiling effect.
    
    N=100: Higher Goal-Directedness -> smaller spread
    
    -> SEQUENCE LEARNERS GENERALLY DO NOT HAVE LOWER GOAL-DIRECTEDNESS (per se)
'''
fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(complete_df['hpcf_rand_day2'], complete_df['RIspread_day2'])
sns.regplot(data = complete_df, x = 'hpcf_rand_day2', y = 'RIspread_day2', ax = ax)
ax.text(0.6,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
# ax[0].axhline(0)
# ax[0].axvline(0)
# ax[0].legend(title='RI Spread')
plt.show()

'''
    High ΔRT and high ΔER lead to large spreads
'''
fig, ax = plt.subplots(1,2, figsize=(15,5))
slope, intercept, r, p, std_err = stats.linregress(complete_df['RT Diff STT Day 1'], complete_df['ER Diff STT Day 1'])
sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue='RIspread_day2', ax = ax[0])
ax[0].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[0].axhline(0, color='k', linewidth = 0.5)
ax[0].axvline(0, color='k', linewidth = 0.5)
sns.move_legend(ax[0], "lower right")
ax[0].set_xlabel(r'$\Delta$RT')
ax[0].set_ylabel(r'$\Delta$ER')
ax[0].legend(title='RI Spread')

slope, intercept, r, p, std_err = stats.linregress(complete_df['RT Diff STT Day 1'], complete_df['ER Diff STT Day 1'])
sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue='CRspread_day2', ax = ax[1])
ax[1].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[1].axhline(0, color='k', linewidth = 0.5)
ax[1].axvline(0, color='k', linewidth = 0.5)
ax[1].set_xlabel(r'$\Delta$RT')
ax[1].set_ylabel(r'$\Delta$ER')
ax[1].legend(title='CR Spread')
sns.move_legend(ax[1], "lower right")
# plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/DeltasvsSpreads.svg')
plt.show()

'''
    RI-Spread and CR-Spread are correlated, but not as much as you'd think,
    which makes sense, since there is a ceiling effect for the CR-spread.
'''
fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(complete_df['ER_diff_stt_day2'], complete_df['RIspread_day2'])
sns.regplot(data = complete_df, x = 'CRspread_day2', y = 'RIspread_day2', ax = ax)
ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.axhline(complete_df['RIspread_day2'].mean())
ax.axvline(complete_df['CRspread_day2'].mean())
ax.axhline(0, color='k', linewidth = 0.5)
ax.axvline(0, color='k', linewidth = 0.5)
plt.show()


'''
    -> Strong sequence-learning is not associated with strongly reduced 
    traits of goal-directedness in general. But sequence learning 
'''

'''
    ΔRT & ΔER are both indicators of how strongly the sequence is learned.
'''

'''
    Age is not associated with ΔRT and ΔER
    This suggests age is also not associated with spread.
'''
fig, ax = plt.subplots(1,2, figsize=(15,5))
slope, intercept, r, p, std_err = stats.linregress(complete_df['age'], complete_df['RT_diff_stt_day2'])
sns.regplot(data = complete_df, x = 'age', y = 'RT_diff_stt_day2', ax = ax[0])
ax[0].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[0].axhline(0)
ax[0].axvline(0)

slope, intercept, r, p, std_err = stats.linregress(complete_df['age'], complete_df['ER_diff_stt_day2'])
sns.regplot(data = complete_df, x = 'age', y = 'ER_diff_stt_day2', ax = ax[1])
ax[1].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[1].axhline(0)
ax[1].axvline(0)
plt.show()

'''
    Age is not associated with spread.
'''
fig, ax = plt.subplots(1,2, figsize=(15,5))
slope, intercept, r, p, std_err = stats.linregress(complete_df['age'], complete_df['CRspread_day2'])
sns.regplot(data = complete_df, x = 'age', y = 'CRspread_day2', ax = ax[0])
ax[0].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[0].axhline(0)
ax[0].axvline(0)

slope, intercept, r, p, std_err = stats.linregress(complete_df['age'], complete_df['RIspread_day2'])
sns.regplot(data = complete_df, x = 'age', y = 'RIspread_day2', ax = ax[1])
ax[1].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[1].axhline(0)
ax[1].axvline(0)
plt.show()

'''
    Age is slightly associated with goal-directedness
'''
fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(complete_df['age'], complete_df['hpcf_rand_day2'])
sns.regplot(data = complete_df, x = 'age', y = 'hpcf_rand_day2', ax = ax)
ax.text(25,0.55, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
# ax.axhline(complete_df['RIspread_day2'].mean())
# ax.axvline(complete_df['CRspread_day2'].mean())
plt.show()

#%%
'''
    Exploiters
'''

# complete_df['score'] = complete_df['RT_diff_stt_day2'] /  complete_df['RT_diff_stt_day2'].std() + \
#                         complete_df['ER_diff_stt_day2'] /  complete_df['ER_diff_stt_day2'].std() + \
#                         complete_df['conflict_param_day2'] - complete_df['theta_rep_day2']

complete_df['score'] = complete_df['RT_diff_stt_day2'] / complete_df['RT_diff_stt_day2'].std() + \
                        complete_df['ER_diff_stt_day2'] / complete_df['ER_diff_stt_day2'].std() + \
                        - complete_df['RIspread_day2']

# exploiters_df = complete_df[(complete_df['CRspread_day2'] > complete_df['CRspread_day2'].mean()) &\
#                              (complete_df['RIspread_day2'] < complete_df['RIspread_day2'].mean())]

exploiters_df = complete_df[(complete_df['RIspread_day2'] < complete_df['RIspread_day2'].mean()) &\
                            (complete_df['ER_diff_stt_day2'] > 0)]
    
no_exploiters_df = complete_df[~complete_df['ID'].isin(exploiters_df['ID'])]

exploiters_df['RT_diff_stt_day2'].mean()
exploiters_df['ER_diff_stt_day2'].mean()

fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(complete_df['RT Diff STT Day 1'], complete_df['ER Diff STT Day 1'])
sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue='RIspread_day2', ax = ax)
# ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.axhline(0, color='k', linewidth = 0.5)
ax.axvline(0, color='k', linewidth = 0.5)
# sns.move_legend(ax, "lower right")
ax.set_xlabel(r'$\Delta$RT')
ax.set_ylabel(r'$\Delta$ER')
ax.set_xlim([-10, 40])
ax.set_ylim([-0.01, 0.04])
ax.legend(title='RI Spread')
# plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/Deltas_CISpread_hues.svg')
plt.show()

fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(complete_df['RT Diff STT Day 1'], complete_df['ER Diff STT Day 1'])
sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue='CIspread_day2', ax = ax)
# ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.axhline(0, color='k', linewidth = 0.5)
ax.axvline(0, color='k', linewidth = 0.5)
sns.move_legend(ax, "lower right")
ax.set_xlabel(r'$\Delta$RT (ms)')
ax.set_ylabel(r'$\Delta$ER (pp)')
ax.set_xlim([-10, 40])
ax.set_ylim([-0.01, 0.04])
ax.legend(title='CI Spread (pp)')
# plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/Deltas_CISpread_hues.svg')
plt.show()

fig, ax = plt.subplots()
sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue='RIspread_day2', ax = ax)
sns.scatterplot(data = exploiters_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', color ='red')
# ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.axhline(0, color='k', linewidth = 0.5)
ax.axvline(0, color='k', linewidth = 0.5)
sns.move_legend(ax, "lower right")
ax.set_xlabel(r'$\Delta$RT')
ax.set_ylabel(r'$\Delta$ER')
ax.legend(title='RI Spread')
ax.set_xlim([-10, 40])
ax.set_ylim([-0.01, 0.04])
plt.show()


fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(exploiters_df['RT_diff_stt_day2'], exploiters_df['RIspread_day2'])
sns.scatterplot(data = exploiters_df, x = 'hpcf_rand_day2', y = 'CRspread_day2', ax = ax, color = 'red')
# ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.axhline(0, color='k', linewidth = 0.5)
ax.axhline(exploiters_df['CRspread_day2'].mean(), color='r', linewidth = 0.5)
ax.axhline(no_exploiters_df['CRspread_day2'].mean(), color='b', linewidth = 0.5)
ax.axvline(exploiters_df['hpcf_rand_day2'].mean(), color='r', linewidth = 0.5)
ax.axvline(no_exploiters_df['hpcf_rand_day2'].mean(), color='b', linewidth = 0.5)
ax.set_ylabel('CR-Spread')
ax.set_xlabel('HRCF Random DTT')
# ax.axvline(0)
plt.title('Exploiters')
plt.show()

scipy.stats.ttest_ind(exploiters_df['CRspread_day2'], no_exploiters_df['CRspread_day2'])
scipy.stats.ttest_ind(exploiters_df['hpcf_rand_day2'], no_exploiters_df['hpcf_rand_day2'])


#%%
'''
    Habitualizers
'''

habitualizers_df = complete_df[(complete_df['RIspread_day2'] >= complete_df['RIspread_day2'].mean())]
no_habitualizers = complete_df[~complete_df['ID'].isin(habitualizers_df['ID'])]


fig, ax = plt.subplots(figsize=(10,5))
slope, intercept, r, p, std_err = stats.linregress(complete_df['RT Diff STT Day 1'], complete_df['ER Diff STT Day 1'])
sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue='RIspread_day2', ax = ax)
# ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.axhline(0, color='k', linewidth = 0.5)
ax.axvline(0, color='k', linewidth = 0.5)
# sns.move_legend(ax, "lower right")
ax.set_xlabel(r'$\Delta$RT')
ax.set_ylabel(r'$\Delta$ER')
ax.set_xlim([-10, 40])
ax.set_ylim([-0.01, 0.04])
ax.legend(title='RI Spread')
plt.show()


fig, ax = plt.subplots(figsize=(10,5))
sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue='RIspread_day2', ax = ax)
sns.scatterplot(data = habitualizers_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', color ='red')
# ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.axhline(0, color='k', linewidth = 0.5)
ax.axvline(0, color='k', linewidth = 0.5)
sns.move_legend(ax, "lower right")
ax.set_xlabel(r'$\Delta$RT')
ax.set_ylabel(r'$\Delta$ER')
ax.legend(title='RI Spread')
ax.set_xlim([-10, 40])
ax.set_ylim([-0.01, 0.04])
plt.show()

fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(habitualizers_df['RT_diff_stt_day2'], habitualizers_df['RIspread_day2'])
sns.scatterplot(data = habitualizers_df, x = 'hpcf_rand_day2', y = 'CRspread_day2', ax = ax, color ='red')
# ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.axhline(0, color='k', linewidth = 0.5)
ax.axhline(habitualizers_df['CRspread_day2'].mean(), color='r', linewidth = 0.5)
ax.axhline(exploiters_df['CRspread_day2'].mean(), color='orange', linewidth = 0.5)
ax.axhline(no_habitualizers['CRspread_day2'].mean(), color='b', linewidth = 0.5)
ax.axvline(habitualizers_df['hpcf_rand_day2'].mean(), color='r', linewidth = 0.5)
ax.axvline(exploiters_df['hpcf_rand_day2'].mean(), color='orange', linewidth = 0.5)
ax.axvline(no_habitualizers['hpcf_rand_day2'].mean(), color='b', linewidth = 0.5)
ax.set_ylabel('CR-Spread')
ax.set_xlabel('HRCF Random DTT')
# ax.axvline(0)
plt.title('Habitualizers')
# sns.move_legend(ax, "lower left")
plt.show()

scipy.stats.ttest_ind(habitualizers_df['CRspread_day2'], no_habitualizers['CRspread_day2'])
scipy.stats.ttest_ind(habitualizers_df['CRspread_day2'], exploiters_df['CRspread_day2'])
scipy.stats.ttest_ind(habitualizers_df['hpcf_rand_day2'], no_habitualizers['hpcf_rand_day2'])
scipy.stats.ttest_ind(habitualizers_df['hpcf_rand_day2'], exploiters_df['hpcf_rand_day2'])


#%%

'''
    CR-Spread and RI-Spread for seqtrait
'''
fig, ax = plt.subplots(1,2, figsize=(15,5))
slope, intercept, r, p, std_err = stats.linregress(complete_df['seqtrait'], complete_df['CRspread_day2'])
sns.regplot(data = complete_df, x = 'seqtrait', y = 'CRspread_day2', ax = ax[0])
ax[0].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[0].axhline(0)
ax[0].axvline(0)

slope, intercept, r, p, std_err = stats.linregress(complete_df['seqtrait'], complete_df['RIspread_day2'])
sns.regplot(data = complete_df, x = 'seqtrait', y = 'RIspread_day2', ax = ax[1])
ax[1].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[1].axhline(0)
ax[1].axvline(0)
plt.show()


fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(complete_df['seqtrait'], complete_df['hpcf_rand_day2'])
slope, intercept, r, p, std_err = stats.linregress(complete_df['seqtrait'], complete_df['hpcf_rand_day2'])
sns.regplot(data = complete_df, x = 'seqtrait', y = 'hpcf_rand_day2', ax = ax)
ax.text(0,0.75, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
plt.show()

#%%
# noseqlearners_df = complete_df[(complete_df['ER_diff_stt_day2'] <= 0) & (complete_df['RT_diff_stt_day2'] <= 0)]
# print(f'number of no sequence learners are {len(noseqlearners_df)}')
# utils.plot_grouplevel(expdata_df[expdata_df['ID'].isin(noseqlearners_df['ID'])], plot_single = True)
# noseqlearners_df = complete_df[(complete_df['RIspread_day2'] <= 0) & (complete_df['CRspread_day2'] <= 0)]

fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['RIspread_day2'])
slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['RIspread_day2'])
sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'RIspread_day2', ax = ax)
ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.axhline(0)
ax.axvline(0)
plt.show()

fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(complete_df['ER_diff_stt_day2'], complete_df['RIspread_day2'])
slope, intercept, r, p, std_err = stats.linregress(complete_df['ER_diff_stt_day2'], complete_df['RIspread_day2'])
sns.regplot(data = complete_df, x = 'ER_diff_stt_day2', y = 'RIspread_day2', ax = ax)
ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.axhline(0)
ax.axvline(0)
plt.show()

'''
    GLM RIspread ~ ΔER + ΔRT + ΔER * ΔRT
'''
"Correct ER for RT"
from sklearn.linear_model import LinearRegression
y = np.array(complete_df['ER_diff_stt_day2']).reshape(-1,1)
x = np.array(complete_df['RT_diff_stt_day2']).reshape(-1,1)
linmodel = LinearRegression()
linmodel.fit(x, y)
predictions = linmodel.predict(x)
residuals = y - predictions

complete_df['ER_diff_stt_day2 corrected'] = np.squeeze(residuals)
# complete_df['interaction']   = 

import statsmodels.api as sm
x = np.array(complete_df.loc[:, ['RT_diff_stt_day2', 'ER_diff_stt_day2 corrected']], dtype='float')
y = np.array(complete_df['RIspread_day2'], dtype = 'float')
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print(model.summary())

# x = np.array(df.loc[:, ['ER Diff Day 2', 'RT Diff Day 2']], dtype='float')
# y = np.array(df['ΔPoints Day 2'], dtype = 'float')
# X = sm.add_constant(x)
# model = sm.OLS(y, X).fit()
# print(model.summary())

#%%
'''
    Check High ΔRT → High ΔER but not High ΔER → High ΔRT
'''

fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(complete_df['RT Diff STT Day 1'], complete_df['ER Diff STT Day 1'])
sns.regplot(data = complete_df, x = 'RT Diff STT Day 1', y = 'ER Diff STT Day 1', ax = ax)
# ax.text(0,0, 'r=%.4f, p = %.4f'%(r,p))
ax.axhline(complete_df['ER Diff STT Day 1'].mean())
ax.axvline(complete_df['RT Diff STT Day 1'].mean())
# ax.axhline(0)
# ax.axvline(0)
plt.show()

fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['ER_diff_stt_day2'])
sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', ax = ax)
ax.text(0,0, 'r=%.4f, p = %.4f'%(r,p))
ax.axhline(complete_df['ER_diff_stt_day2'].mean())
ax.axvline(complete_df['RT_diff_stt_day2'].mean())
# ax.axhline(0)
# ax.axvline(0)
plt.show()

# test_df = complete_df.loc[:, ['RT_diff_stt_day2', 'ER_diff_stt_day2']]

# test_df = (test_df-test_df.mean())/test_df.std()
# test_df.sort_values(by='RT_diff_stt_day2', inplace = True, ascending = False)
# test_df[0:30]['ER_diff_stt_day2'].mean()

# test_df.sort_values(by='ER_diff_stt_day2', inplace = True, ascending = False)
# test_df[0:30]['ER_diff_stt_day2'].mean()

#%%


'''
    Check High ΔHRCF & High ΔER & High ΔRT → High R-Inc Spread
'''
fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['RIspread_day2'])
slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['RIspread_day2'])
sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'RIspread_day2', ax = ax)
ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.axhline(0)
ax.axvline(0)
plt.show()

fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(complete_df['ER_diff_stt_day2'], complete_df['RIspread_day2'])
slope, intercept, r, p, std_err = stats.linregress(complete_df['ER_diff_stt_day2'], complete_df['RIspread_day2'])
sns.regplot(data = complete_df, x = 'ER_diff_stt_day2', y = 'RIspread_day2', ax = ax)
ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.axhline(0)
ax.axvline(0)
plt.show()

"Day 2"
# seqlearners_df = complete_df[(complete_df['CRspread_day2'] > complete_df['CRspread_day2'] .mean()) & \
#                       (complete_df['ER_diff_stt_day2'] > complete_df['ER_diff_stt_day2'].mean())]

'''
    We define sequence learners as those whose ΔER > 0
'''
# seqlearners_df = complete_df[complete_df['ER_diff_stt_day2'] > complete_df['ER_diff_stt_day2'].mean()]
# antiseqlearners_df = complete_df[~complete_df['ID'].isin(seqlearners_df['ID'])]

seqlearners_df = complete_df[complete_df['ER_diff_stt_day2'] > 0]
noseqlearners_df = complete_df[complete_df['ER_diff_stt_day2'] <= 0]

# seqlearners_df = seqlearners_df[seqlearners_df['RT_diff_stt_day2'] < 60]

len(seqlearners_df)
'''
    larger ΔRT is associated with larger RIspread in sequence learners
'''
fig, ax = plt.subplots(1,2, figsize=(15,5))
slope, intercept, r, p, std_err = stats.linregress(seqlearners_df['RT_diff_stt_day2'], seqlearners_df['RIspread_day2'])
sns.regplot(data = seqlearners_df, x = 'RT_diff_stt_day2', y = 'RIspread_day2', ax = ax[0])
ax[0].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[0].axhline(0)
ax[0].axvline(0)

slope, intercept, r, p, std_err = stats.linregress(seqlearners_df['RT_diff_stt_day2'], seqlearners_df['ER_diff_stt_day2'])
sns.regplot(data = seqlearners_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', ax = ax[1])
ax[1].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[1].axhline(0)
ax[1].axvline(0)
plt.show()

'''
    nosequence learners
'''
fig, ax = plt.subplots(1,2, figsize=(15,5))
slope, intercept, r, p, std_err = stats.linregress(noseqlearners_df['RT_diff_stt_day2'], noseqlearners_df['RIspread_day2'])
sns.regplot(data = noseqlearners_df, x = 'RT_diff_stt_day2', y = 'RIspread_day2', ax = ax[0])
ax[0].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[0].axhline(0)
ax[0].axvline(0)

slope, intercept, r, p, std_err = stats.linregress(noseqlearners_df['RT_diff_stt_day2'], noseqlearners_df['ER_diff_stt_day2'])
sns.regplot(data = noseqlearners_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', ax = ax[1])
ax[1].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[1].axhline(0)
ax[1].axvline(0)
plt.show()

'''
    Are exploiters generally more AO than habitualizers?
'''
fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(seqlearners_df['RT_diff_stt_day2'], seqlearners_df['hpcf_rand_day2'])
sns.regplot(data = seqlearners_df, x = 'RT_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax)
ax.text(0,0, 'r=%.4f, p = %.4f'%(r,p))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.axhline(0)
ax.axvline(0)
plt.show()

'''
    Generally, sequence-learners have lower goal-directedness than non-sequence learners.
    This works only for ΔER since sequence learners can have low ΔRT if they are exploiters.
'''
fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(complete_df['ER_diff_stt_day2'], complete_df['hpcf_rand_day2'])
sns.regplot(data = complete_df, x = 'ER_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax)
ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.axhline(0)
ax.axvline(0)
plt.show()

fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(complete_df['CRspread_day2'], complete_df['hpcf_rand_day2'])
sns.regplot(data = complete_df, x = 'CRspread_day2', y = 'hpcf_rand_day2', ax = ax)
ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.axhline(0)
ax.axvline(0)
plt.show()

scipy.stats.ttest_ind(seqlearners_df['hpcf_rand_day2'], antiseqlearners_df['hpcf_rand_day2'])

'''
    Have sequence learners noticed the sequence more than non-sequence learners?
    -> Yes
'''
(seqlearners_df['q_notice_a_sequence']==1).sum() / len(seqlearners_df['q_notice_a_sequence'])
(antiseqlearners_df['q_notice_a_sequence']==1).sum() / len(antiseqlearners_df['q_notice_a_sequence'])

'''
    Have exploiters noticed the sequence more than habitualizers?
    -> Not really
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

seqlearners_regdf = seqlearners_df[seqlearners_df['q_notice_a_sequence'] < 2]
X = np.array(seqlearners_regdf['RT_diff_stt_day2']).reshape(14,-1)
y = np.array(seqlearners_regdf['q_notice_a_sequence']).reshape(14)

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Plotting the logistic regression curve
X_test = np.linspace(min(X), max(X), 300)
y_prob = model.predict_proba(X_test.reshape(-1, 1))[:, 1]

plt.scatter(X, y, color='blue', marker='o', label='Data Points')
plt.plot(X_test, y_prob, color='red', label='Logistic Regression Curve')
plt.xlabel('RT_diff_stt_day2')
plt.ylabel('Y-axis (Binary)')
plt.title('Logistic Regression Visualization')
plt.legend()
plt.show()
#%%
'''
    Check Low ΔHRCF & High ΔER → High R-Inc Spread
'''

"Day 2"
test_df = complete_df[complete_df['spread_day2'] < complete_df['spread_day2'] .mean()]
test_df = test_df[test_df['ER_diff_stt_day2'] > 0]

fig, ax = plt.subplots()
slope, intercept, r, p, std_err = stats.linregress(test_df['ER_diff_stt_day2'], test_df['RIspread_day2'])
sns.scatterplot(data = test_df, x = 'ER_diff_stt_day2', y = 'RIspread_day2', ax = ax)
ax.text(0,0, 'r=%.4f, p = %.4f'%(r,p))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax.axhline(0)
ax.axvline(0)
plt.show()

#%%
'''
    Exploit-Score
'''

complete_df['learnscore'] = complete_df['RT_diff_stt_day2'] / complete_df['RT_diff_stt_day2'].std() + \
                        complete_df['ER_diff_stt_day2'] / complete_df['ER_diff_stt_day2'].std() + \
                        + complete_df['theta_rep_day2']

complete_df['exploitscore'] = complete_df['learnscore'] + complete_df['conflict_param_day2']
