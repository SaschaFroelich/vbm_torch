#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:09:53 2024

@author: sascha
"""

from IPython import get_ipython
get_ipython().run_line_magic("reset", "-f")

"----- Open Files"
# import sys
# sys.modules[__name__].__dict__.clear() # Clear variables
# del model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import utils
import analysis_tools as anal
import torch
import pickle
import arviz as az

from sklearn.linear_model import LinearRegression
import scipy
import statsmodels.api as sm
import itertools

complete_df_all, inf_mean_df_all, expdata_df_all, post_sample_df_all, sim_df, \
param_names, Q_init_day2, seq_counter_day2, er_day2, extra_storage_day2, \
extra_storage_day1, agent_elbo_tuple_day2, agent_elbo_tuple_day1 = utils.load_data()


#%%
model = complete_df_all['model'].unique()[0]
hue_order = ['Random', 'Congruent', 'Incongruent']

if 'theta_rep' in param_names:
    param1 = 'theta_rep'
    param2 = 'theta_conflict'
    
elif 'theta_repcong' in param_names:
    param1 = 'theta_repcong'
    param2 = 'theta_repinc'
    
elif 'theta_Q_congdiff' in param_names:
    param1 = 'theta_Q_congdiff'
    param2 = 'theta_Q_conflict'

#%%
'''
    Plot behaviour on both days
    
    HPCF stands for 'high-probability choice frequency' and is identical to
    ratio of goal-directed responses.
'''

# HPCF_DF = complete_df_all.loc[:, ['hpcf_cong', 'hpcf_incong',
#                                   'hpcf_seq', 'hpcf_rand', 'day', 'ID']]

utils.plot_hpcf(complete_df_all, title='Experiment')

hpcf_day1 = utils.compute_hpcf(sim_df[sim_df['day']==1])
hpcf_day1['day'] = 1
hpcf_day2 = utils.compute_hpcf(sim_df[sim_df['day']==2])
hpcf_day2['day'] = 2

hpcf_df_all = pd.concat((hpcf_day1, hpcf_day2), ignore_index = True)

utils.plot_hpcf(hpcf_df_all, title=f'{model}', post_pred = False)

#%%
'''
    ER ~ Condition, Day
'''

ER_stt_all = complete_df_all.loc[:, ['ID',
                                 'day',
                                 'ER_stt_rand', 
                                  'ER_stt_seq']]

ER_stt_all = ER_stt_all.melt(id_vars=['ID', 'day'], value_vars=['ER_stt_rand', 'ER_stt_seq'])
ER_stt_all['variable'] = ER_stt_all['variable'].map(lambda x: "Repeating" if x == 'ER_stt_seq' else
                                                'Random' if x == 'ER_stt_rand' else
                                                'None')
ER_stt_all = ER_stt_all.rename(columns={'variable': 'Condition',
                                'value': 'ER'})
import pingouin as pg
aov = pg.rm_anova(dv = 'ER', 
                  within = ['Condition', 'day'], 
                  subject = 'ID', 
                  data = ER_stt_all, 
                  detailed = True,
                  effsize = 'np2')
print(aov.loc[:, ['Source', 'ddof1', 'ddof2', 'F', 'p-unc', 'np2']])

'''
    ER
    Differences within days
'''
"Rand vs Rep"
t,p = scipy.stats.ttest_rel(ER_stt_all[(ER_stt_all['day'] == 1) & (ER_stt_all['Condition'] == 'Random')]['ER'], 
                            ER_stt_all[(ER_stt_all['day'] == 1) & (ER_stt_all['Condition'] == 'Repeating')]['ER'])
print(f"t={t}, p={p}")

"Rand vs Rep"
t,p = scipy.stats.ttest_rel(ER_stt_all[(ER_stt_all['day'] == 2) & (ER_stt_all['Condition'] == 'Random')]['ER'], 
                            ER_stt_all[(ER_stt_all['day'] == 2) & (ER_stt_all['Condition'] == 'Repeating')]['ER'])
print(f"t={t}, p={p}")


'''
    ER
    Differences between days
'''
"Rand vs Rand"
t,p = scipy.stats.ttest_rel(ER_stt_all[(ER_stt_all['day'] == 1) & (ER_stt_all['Condition'] == 'Random')]['ER'], 
                            ER_stt_all[(ER_stt_all['day'] == 2) & (ER_stt_all['Condition'] == 'Random')]['ER'])
print(f"t={t}, p={p}")

"Rep vs Rep"
t,p = scipy.stats.ttest_rel(ER_stt_all[(ER_stt_all['day'] == 1) & (ER_stt_all['Condition'] == 'Repeating')]['ER'], 
                            ER_stt_all[(ER_stt_all['day'] == 2) & (ER_stt_all['Condition'] == 'Repeating')]['ER'])
print(f"t={t}, p={p}")

'''
    RT ~ Condition, Day
'''

RT_all = complete_df_all.loc[:, ['ID',
                                 'day',
                                 'RT_stt_rand', 
                                   'RT_stt_seq']]

RT_all = RT_all.melt(id_vars=['ID', 'day'], value_vars=['RT_stt_rand', 'RT_stt_seq'])
RT_all['variable'] = RT_all['variable'].map(lambda x: "Repeating" if x == 'RT_stt_seq' else
                                                'Random' if x == 'RT_stt_rand' else
                                                'None')

RT_all = RT_all.rename(columns={'variable': 'Condition',
                                'value': 'RT'})

import pingouin as pg
aov = pg.rm_anova(dv = 'RT', 
                  within=['Condition', 'day'], 
                  subject = 'ID', 
                  data = RT_all, 
                  detailed = True,
                  effsize = 'np2')

print(aov.loc[:, ['Source', 'ddof1', 'ddof2', 'F', 'p-unc', 'np2']])

'''
    RT 
    Differences within Days
'''

"Rand vs Rep"
t,p = scipy.stats.ttest_rel(RT_all[(RT_all['day'] == 1) & (RT_all['Condition'] == 'Random')]['RT'], 
                            RT_all[(RT_all['day'] == 1) & (RT_all['Condition'] == 'Repeating')]['RT'])
print(f"t={t}, p={p}")

"Rand vs Rep"
t,p = scipy.stats.ttest_rel(RT_all[(RT_all['day'] == 2) & (RT_all['Condition'] == 'Random')]['RT'], 
                            RT_all[(RT_all['day'] == 2) & (RT_all['Condition'] == 'Repeating')]['RT'])
print(f"t={t}, p={p}")

'''
    RT 
    Differences between Days
'''
"Rand vs Rand"
t,p = scipy.stats.ttest_rel(np.array(RT_all[(RT_all['day'] == 2) & (RT_all['Condition'] == 'Random')]['RT']),
                            np.array(RT_all[(RT_all['day'] == 1) & (RT_all['Condition'] == 'Random')]['RT']))
print(f"t={t}, p={p}")

"Rep vs Rep"
t,p = scipy.stats.ttest_rel(np.array(RT_all[(RT_all['day'] == 2) & (RT_all['Condition'] == 'Repeating')]['RT']),
                            np.array(RT_all[(RT_all['day'] == 1) & (RT_all['Condition'] == 'Repeating')]['RT']))
print(f"t={t}, p={p}")


#%%

'''
    Results Figure 1
    Left: HRC Day 1 & Day 2
    Middle: ER Day 1 & Day 2
    Right: RT Day 1 & Day 2
'''

colors1 = {'Random': '#67b798', 'Congruent': '#BE54C6', 'Incongruent': '#7454C7'}
colors2 = {'Random': '#67b798', 'Repeating': '#bd97c6'} # random, rep
colors2_stt = {'Random': '#8EC2AE', 'Repeating': '#bd97c6'} # random, rep

hpcf_all = complete_df_all.loc[:, ['ID',
                                   'day',
                                   'hpcf_cong', 
                                   'hpcf_incong',
                                   'hpcf_rand']]

hpcf_all = hpcf_all.melt(id_vars=['ID', 'day'], value_vars=['hpcf_cong', 'hpcf_incong', 'hpcf_rand'])
hpcf_all = hpcf_all.rename(columns={'variable': 'DTT Type', 'value': 'HRC'})
hpcf_all['DTT Type'] = hpcf_all['DTT Type'].map(lambda x: "Congruent" if x == 'hpcf_cong' else
                                                'Incongruent' if x == 'hpcf_incong' else
                                                'Random' if x == 'hpcf_rand' else
                                                'None')

hpcf_all['HRC'] = hpcf_all['HRC']*100
hpcf_all['day'] = hpcf_all['day'].astype(int)


fig, ax = plt.subplots(1, 3, figsize = (20, 10))
sns.barplot(data = hpcf_all,
            x = 'day',
            y = 'HRC',
            hue = 'DTT Type',
            hue_order = hue_order,
            palette = colors1,
            errorbar=('se', 1),
            ax = ax[0])

ax[0].legend(title="", fontsize = 18)
ax[0].set_ylabel("Goal-Directed Responses (%)", fontsize = 20)
ax[0].set_ylim([60, 100])
ax[0].set_xlabel("Day", fontsize = 20)
ax[0].tick_params(axis='both', labelsize=18)

ER_stt_all['day'] = ER_stt_all['day'].astype(int)
ER_stt_all['ER'] = ER_stt_all['ER']*100
sns.barplot(data = ER_stt_all,
            x = 'day',
            y = 'ER',
            hue = 'Condition',
            palette = colors2_stt,
            errorbar=('se', 1),
            ax = ax[1])

ax[1].legend(title="", fontsize = 18)
ax[1].set_ylabel("ER STT (%)", fontsize = 20)
ax[1].set_ylim([3, 9])
ax[1].set_xlabel("Day", fontsize = 20)
ax[1].tick_params(axis='both', labelsize=18)

RT_all['day'] = RT_all['day'].astype(int)
sns.barplot(data = RT_all,
            x = 'day',
            y = 'RT',
            hue = 'Condition',
            palette = colors2_stt,
            errorbar=('se', 1),
            ax = ax[2])

ax[2].legend(title="", fontsize = 18)
ax[2].set_ylabel("RT STT (ms)", fontsize = 20)
ax[2].set_ylim([320, 420])
ax[2].set_xlabel("Day", fontsize = 20)
ax[2].tick_params(axis='both', labelsize=18)
plt.savefig('/home/sascha/Desktop/Paper_2024/May/res_fig0/res_fig0_python.svg', bbox_inches = 'tight')
plt.show()

'''
    HRC
    Differences within days: Day 1
'''

"Rand vs Cong"
t,p = scipy.stats.ttest_rel(hpcf_all[(hpcf_all['day'] == 1) & (hpcf_all['DTT Type'] == 'Random')]['HRC'], 
                            hpcf_all[(hpcf_all['day'] == 1) & (hpcf_all['DTT Type'] == 'Congruent')]['HRC'])
print(f"HRC All, Rand vs Cong, Day 1: t={t}, p={p}")

"Rand vs Inc"
t,p = scipy.stats.ttest_rel(hpcf_all[(hpcf_all['day'] == 1) & (hpcf_all['DTT Type'] == 'Random')]['HRC'], 
                            hpcf_all[(hpcf_all['day'] == 1) & (hpcf_all['DTT Type'] == 'Incongruent')]['HRC'])
print(f"HRC All, Rand vs Incong, Day 1: t={t}, p={p}")

"Cong vs Inc"
t,p = scipy.stats.ttest_rel(hpcf_all[(hpcf_all['day'] == 1) & (hpcf_all['DTT Type'] == 'Congruent')]['HRC'], 
                            hpcf_all[(hpcf_all['day'] == 1) & (hpcf_all['DTT Type'] == 'Incongruent')]['HRC'])
print(f"HRC All, Cong vs Incong, Day 1: t={t}, p={p}")

'''
    HRC
    Differences within days: Day 2
'''

"Rand vs Cong"
t,p = scipy.stats.ttest_rel(hpcf_all[(hpcf_all['day'] == 2) & (hpcf_all['DTT Type'] == 'Random')]['HRC'], 
                            hpcf_all[(hpcf_all['day'] == 2) & (hpcf_all['DTT Type'] == 'Congruent')]['HRC'])
print(f"HRC All, Rand vs Cong, Day 1: t={t}, p={p}")

"Rand vs Inc"
t,p = scipy.stats.ttest_rel(hpcf_all[(hpcf_all['day'] == 2) & (hpcf_all['DTT Type'] == 'Random')]['HRC'], 
                            hpcf_all[(hpcf_all['day'] == 2) & (hpcf_all['DTT Type'] == 'Incongruent')]['HRC'])
print(f"HRC All, Rand vs Incong, Day 1: t={t}, p={p}")

"Cong vs Inc"
t,p = scipy.stats.ttest_rel(hpcf_all[(hpcf_all['day'] == 2) & (hpcf_all['DTT Type'] == 'Congruent')]['HRC'], 
                            hpcf_all[(hpcf_all['day'] == 2) & (hpcf_all['DTT Type'] == 'Incongruent')]['HRC'])
print(f"HRC All, Cong vs Incong, Day 1: t={t}, p={p}")

'''
    HRC
    Differences between days
'''
"Random vs Random"
t,p = scipy.stats.ttest_rel(hpcf_all[(hpcf_all['day'] == 1) & (hpcf_all['DTT Type'] == 'Random')]['HRC'], 
                            hpcf_all[(hpcf_all['day'] == 2) & (hpcf_all['DTT Type'] == 'Random')]['HRC'])
print(f"HRC All, Rand vs Cong, Day 1: t={t}, p={p}")

"Cong vs Cong"
t,p = scipy.stats.ttest_rel(hpcf_all[(hpcf_all['day'] == 1) & (hpcf_all['DTT Type'] == 'Congruent')]['HRC'], 
                            hpcf_all[(hpcf_all['day'] == 2) & (hpcf_all['DTT Type'] == 'Congruent')]['HRC'])
print(f"HRC All, Rand vs Incongruent, Day 1: t={t}, p={p}")

"Incong vs Incong"
t,p = scipy.stats.ttest_rel(hpcf_all[(hpcf_all['day'] == 1) & (hpcf_all['DTT Type'] == 'Incongruent')]['HRC'], 
                            hpcf_all[(hpcf_all['day'] == 2) & (hpcf_all['DTT Type'] == 'Incongruent')]['HRC'])
print(f"HRC All, Congruent vs Incongruent, Day 1: t={t}, p={p}")

'''
    ANOVA
    HRC ~ DTT Type, Day
'''

import pingouin as pg
aov = pg.rm_anova(dv = 'HRC', 
                  within=['DTT Type', 'day'], 
                  subject = 'ID', 
                  data = hpcf_all, 
                  detailed = True,
                  effsize = 'np2')

print(aov.loc[:, ['Source', 'ddof1', 'ddof2', 'F', 'p-unc', 'np2']])

RT_all.to_csv('RT_stt.csv')
ER_stt_all.to_csv('ER_stt.csv')
complete_df_all.to_csv('complete_df.csv')

#%%
'''
    Do spreads significantly differ between days?
'''
t,p = scipy.stats.ttest_rel(complete_df_all[complete_df_all['day'] == 1]['CRspread'], 
                            complete_df_all[complete_df_all['day'] == 2]['CRspread'])
print(f"CR spread day 1 vs day 2: t={t}, p={p}")

t,p = scipy.stats.ttest_rel(complete_df_all[complete_df_all['day'] == 1]['CIspread'], 
                            complete_df_all[complete_df_all['day'] == 2]['CIspread'])
print(f"CI spread day 1 vs day 2: t={t}, p={p}")

t,p = scipy.stats.ttest_rel(complete_df_all[complete_df_all['day'] == 1]['RIspread'], 
                            complete_df_all[complete_df_all['day'] == 2]['RIspread'])
print(f"RI spread day 1 vs day 2: t={t}, p={p}")


'''
    Does ΔRT and ΔER significantly differ between days?
    Do they correlate
'''

t,p = scipy.stats.ttest_rel(complete_df_all[complete_df_all['day'] == 1]['ER_diff_stt'], 
                            complete_df_all[complete_df_all['day'] == 2]['ER_diff_stt'])
print(f"ΔER day 1 vs day 2: t={t}, p={p}")

t,p = scipy.stats.ttest_rel(complete_df_all[complete_df_all['day'] == 1]['RT_diff_stt'], 
                            complete_df_all[complete_df_all['day'] == 2]['RT_diff_stt'])
print(f"ΔRT day 1 vs day 2: t={t}, p={p}")


r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 1]['ER_diff_stt'], 
                            complete_df_all[complete_df_all['day'] == 1]['RT_diff_stt'])
print(f"ΔER & ΔRT Correlation Day 1: r={t}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['ER_diff_stt'], 
                            complete_df_all[complete_df_all['day'] == 2]['RT_diff_stt'])
print(f"ΔER & ΔRT Correlation Day 2: r={t}, p={p}")


#%%
'''
    Poisitive measures of habit
    Corr ΔRT ~ ΔCI (Cong-Inc)
'''
r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day']==1]['RT_diff_stt'], 
                           complete_df_all[complete_df_all['day']==1]['CIspread'])
print(f"Correlation ΔRT vs CIspread, Day 1: r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day']==2]['RT_diff_stt'], 
                           complete_df_all[complete_df_all['day']==2]['CIspread'])
print(f"Correlation ΔRT vs CIspread, Day 2: r={r}, p={p}")

"ΔRT Day 1 vs Day 2"
t,p = scipy.stats.ttest_rel(complete_df_all[complete_df_all['day']==1]['RT_diff_stt'], 
                            complete_df_all[complete_df_all['day']==2]['RT_diff_stt'])
print(f"ΔRT Day 1 vs Day 2, t={t}, p={p}")

t,p = scipy.stats.ttest_rel(complete_df_all[complete_df_all['day']==1]['CIspread'], 
                            complete_df_all[complete_df_all['day']==2]['CIspread'])
print(f"ΔCI Day 1 vs Day 2, t={t}, p={p}")

'''
    Corr ER ~ ΔCI
'''
r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day']==1]['ER_diff_stt'], 
                           complete_df_all[complete_df_all['day']==1]['CIspread'])
print(f"Correlation ΔER vs CIspread, Day 1: r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day']==2]['ER_diff_stt'], 
                           complete_df_all[complete_df_all['day']==2]['CIspread'])
print(f"Correlation ΔER vs CIspread, Day 2: r={r}, p={p}")

"ΔER Day 1 vs Day 2"
t,p = scipy.stats.ttest_rel(complete_df_all[complete_df_all['day']==1]['ER_diff_stt'], 
                            complete_df_all[complete_df_all['day']==2]['ER_diff_stt'])
print(f"ΔER Day 1 vs Day 2: t={t}, p={p}")


#%%
'''
    Find weak and strong sequence learners.    

    Response Strategies
    Habitual Responders
    GD Responders
    Modulators
'''
print("===================================")
print("STRONG VS WEAK HABIT LEARNERS")

expdata_df_day1 = expdata_df_all[expdata_df_all['day'] == 1]
expdata_df_day2 = expdata_df_all[expdata_df_all['day'] == 2]

print("===================================")
print("DAY 1")
seqlearners_df_day1, notseqlearners_df_day1, seqlearn_df_day1 = anal.find_seqlearners(expdata_df_day1,
                                                          day = 1,
                                                          correctp = True)

print("===================================")
print("DAY 2")
seqlearners_df_day2, notseqlearners_df_day2, seqlearn_df_day2 = anal.find_seqlearners(expdata_df_day2,
                                                          day = 2,
                                                          correctp = True)



seqall = pd.concat((seqlearners_df_day1, seqlearners_df_day2))
notseqall = pd.concat((notseqlearners_df_day1, notseqlearners_df_day2))

seq_plotdf_day2 = utils.plot_grouplevel(expdata_df_day2[expdata_df_day2['ID'].isin(seqlearners_df_day2['ID'].unique())], plot_single = False)
seq_plotdf_day1 = utils.plot_grouplevel(expdata_df_day1[expdata_df_day1['ID'].isin(seqlearners_df_day1['ID'].unique())], plot_single = False)

notseq_plotdf_day2 = utils.plot_grouplevel(expdata_df_day2[expdata_df_day2['ID'].isin(notseqlearners_df_day2['ID'].unique())], plot_single = False)
notseq_plotdf_day1 = utils.plot_grouplevel(expdata_df_day1[expdata_df_day1['ID'].isin(notseqlearners_df_day1['ID'].unique())], plot_single = False)


notseq_plotdf_all = pd.concat((notseq_plotdf_day1, notseq_plotdf_day2))
seq_plotdf_all = pd.concat((seq_plotdf_day1, seq_plotdf_day2))

seq_plotdf_all['type'] = 'strong sequence learner'
notseq_plotdf_all['type'] = 'weak sequence learner'
seq_combined_temp = pd.concat((seq_plotdf_all, notseq_plotdf_all))

seq_combined = seq_combined_temp.loc[:, ['ID', 
                                         'day', 
                                         'type', 
                                         'choices_GD', 
                                         'DTT Types', 
                                         'block_num']].groupby(['ID', 
                                                                'day', 
                                                                'type', 
                                                                'DTT Types'], as_index = False).mean()

'''
    Plot as barplots
    Bottom: Weak sequence learner vs 
    Top: strong sequence learners
    y: HRC
'''
# colors1 = ['#67b798', '#BE54C6', '#7454C7'] # random, congruent, incongruent]
colors1 = {'Random': '#67b798', 'Congruent': '#BE54C6', 'Incongruent': '#7454C7'}
colors2 = {'Random': '#67b798', 'Repeating': '#bd97c6'} # random, fix

seq_plotdf_all['choices_GD'] = seq_plotdf_all['choices_GD'].map(lambda x: x*100)

seq_combined['choices_GD'] = seq_combined['choices_GD']*100
notseq_plotdf_all['choices_GD'] = notseq_plotdf_all['choices_GD'].map(lambda x: x*100)
'''
    Fig for paper
    2 subplots
    left: Day 1
        x : strong / weak sequence learners
        y : HRC
    right: Day 2
'''
fig, ax = plt.subplots(1,2, sharey = True, sharex=True, figsize = (15, 6))
sns.barplot(ax = ax[0],
            data = seq_combined[seq_combined['day'] == 1],
            x = 'type',
            y = 'choices_GD',
            hue = 'DTT Types',
            hue_order = hue_order,
            palette = colors1,
            errorbar = ('se', 1))

ax[0].set_title('Day 1', fontsize = 18)
ax[0].tick_params(axis='x', labelsize=15)  # For x-axis tick labels
ax[0].tick_params(axis='y', labelsize=15)  # For x-axis tick labels
ax[0].set_xlabel('', fontsize = 18)
ax[0].set_ylabel('Goal-Directed Responses (%)', fontsize = 18)
ax[0].set_ylim([60, 100])
ax[0].set_xticklabels(['strong habit learners', 'weak habit learners'])
custom_labels = ["Random", "Congruent", "Incongruent"]  # Define your custom labels here
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles, custom_labels, fontsize = 15, loc='upper left', bbox_to_anchor=(1, 1), title= '')

sns.barplot(ax = ax[1],
            data = seq_combined[seq_combined['day'] == 2], 
            x = 'type',
            y = 'choices_GD',
            hue = 'DTT Types',
            hue_order = hue_order,
            palette = colors1,
            errorbar = ('se', 1))

ax[1].set_title('Day 2', fontsize = 18)
ax[1].set_xlabel('', fontsize = 18)
ax[1].tick_params(axis='x', labelsize=15)  # For x-axis tick labels
ax[1].tick_params(axis='y', labelsize=15)  # For x-axis tick labels
ax[1].set_ylim([60, 100])
ax[1].set_xticklabels(['strong habit learners', 'weak habit learners'])
ax[1].get_legend().set_visible(False)
plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig1/res_fig1_python.svg')
plt.show()

# fig, ax = plt.subplots()
# sns.barplot(ax = ax,
#             data = seq_combined[seq_combined['day'] == 2],
#             x = 'type',
#             y = 'choices_GD',
#             hue = 'DTT Types',
#             hue_order = hue_order,
#             palette = colors1,
#             errorbar = ('se', 1))
# ax.set_title('Day 2', fontsize = 18)
# ax.tick_params(axis='x', labelsize=15)  # For x-axis tick labels
# ax.tick_params(axis='y', labelsize=15)  # For x-axis tick labels
# ax.set_xlabel('', fontsize = 18)
# ax.set_ylabel('Goal-Directed Responses (%)', fontsize = 18)
# ax.set_ylim([60, 100])
# ax.set_xticklabels(['strong habit learners', 'weak habit learners'])
# custom_labels = ["Random", "Congruent", "Incongruent"]  # Define your custom labels here
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, custom_labels, fontsize = 15, loc='upper left', bbox_to_anchor=(1, 1), title= '')
# plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig1/res_fig1_python.svg')
# plt.show()

"===== Strong learners Day 1"
t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Random')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Congruent')]['choices_GD'])
print(f"Rand-Cong, t={t}, p={p}")

t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Congruent')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Incongruent')]['choices_GD'])
print(f"Cong-Inc, t={t}, p={p}")

t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Random')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Incongruent')]['choices_GD'])
print(f"Rand-Inc, t={t}, p={p}")

"===== Weak learners Day 1"
t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Random')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Congruent')]['choices_GD'])
print(f"Rand-Cong, t={t}, p={p}")

t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Congruent')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Incongruent')]['choices_GD'])
print(f"Cong-Inc, t={t}, p={p}")

t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Random')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Incongruent')]['choices_GD'])
print(f"Rand-Inc, t={t}, p={p}")

"===== Strong learners Day 2"
t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Random')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Congruent')]['choices_GD'])
print(f"Rand-Cong, t={t}, p={p}")

t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Congruent')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Incongruent')]['choices_GD'])
print(f"Cong-Inc, t={t}, p={p}")

t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Random')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Incongruent')]['choices_GD'])
print(f"Rand-Inc, t={t}, p={p}")

"===== Weak learners Day 2"
t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Random')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Congruent')]['choices_GD'])
print(f"Rand-Cong, t={t}, p={p}")

t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Congruent')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Incongruent')]['choices_GD'])
print(f"Cong-Inc, t={t}, p={p}")

t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Random')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Incongruent')]['choices_GD'])
print(f"Rand-Inc, t={t}, p={p}")

#%%
'''
    Negative and no Sequence impact.
'''

spreads_df = complete_df_all[complete_df_all['day'] == 2].loc[:, ['ID', 
                                                                   'RIspread']].groupby('ID', as_index  = False).mean()

"------ HRCF Day 1"
hrc_df_day1_temp = complete_df_all[complete_df_all['day'] == 1].loc[:, ['ID', 
                                                                   'hpcf_cong', 
                                                                   'hpcf_rand', 
                                                                   'hpcf_incong']].groupby('ID', as_index  = False).mean()
df_day1 = expdata_df_all[expdata_df_all['day'] == 1].loc[:, ['ID', 
                        'trialsequence', 
                        'choices', 
                        'choices_GD', 
                        'jokertypes',
                        'day']]

IDs, ps_cr, ps_ri, ps_ci, _, _, _ = anal.hpcf_within(df_day1)
df_ps_day1 = pd.DataFrame({'ID':IDs, 'ps_cr':ps_cr, 'ps_ri':ps_ri, 'ps_ci': ps_ci})
df_ps_day1['day'] = 1

hrc_df_day1_temp = hrc_df_day1_temp.rename(columns={"hpcf_cong": "Congruent", 
                                          "hpcf_incong":"Incongruent", 
                                          'hpcf_rand':'Random'})

hrc_df_day1 = pd.melt(hrc_df_day1_temp, id_vars = ['ID'], value_vars=['Congruent', 
                                                                 'Incongruent', 
                                                                 'Random'])

hrc_df_day1 = hrc_df_day1.rename(columns={"variable": "Trial Type", 
                                          "value":"HRC"})

hrc_df_day1 = pd.merge(hrc_df_day1, df_ps_day1, on = 'ID')
hrc_df_day1['HRC'] = hrc_df_day1['HRC']*100

hrc_df_day1['median'] = hrc_df_day1['ID'].map(lambda x: 'upper_half' if spreads_df[spreads_df['ID'] == x]['RIspread'].item() > spreads_df['RIspread'].median() else 'lower_half')

"------ HRCF Day 2"
hrc_df_day2_temp = complete_df_all[complete_df_all['day'] == 2].loc[:, ['ID', 
                                                                   'hpcf_cong', 
                                                                   'hpcf_rand', 
                                                                   'hpcf_incong']].groupby('ID', as_index  = False).mean()
df_day2 = expdata_df_all[expdata_df_all['day'] == 2].loc[:, ['ID', 
                        'trialsequence', 
                        'choices', 
                        'choices_GD', 
                        'jokertypes',
                        'day']]

IDs, ps_cr, ps_ri, ps_ci, chis_cr, chis_ri, chis_ci = anal.hpcf_within(df_day2)
df_ps_day2 = pd.DataFrame({'ID':IDs, 
                           'ps_cr':ps_cr, 
                           'ps_ri':ps_ri, 
                           'ps_ci': ps_ci,
                           'chis_ri': chis_ri})
df_ps_day2['day'] = 2

hrc_df_day2_temp = hrc_df_day2_temp.rename(columns={"hpcf_cong": "Congruent", 
                                          "hpcf_incong":"Incongruent", 
                                          'hpcf_rand':'Random'})

hrc_df_day2 = pd.melt(hrc_df_day2_temp, id_vars = ['ID'], value_vars=['Congruent', 
                                                                 'Incongruent', 
                                                                 'Random'])

hrc_df_day2 = hrc_df_day2.rename(columns={"variable": "Trial Type", 
                                          "value":"HRC"})

hrc_df_day2 = pd.merge(hrc_df_day2, df_ps_day2, on = 'ID')
hrc_df_day2['HRC'] = hrc_df_day2['HRC']*100

hrc_df_day2['median'] = hrc_df_day2['ID'].map(lambda x: 'upper_half' if spreads_df[spreads_df['ID'] == x]['RIspread'].item() > spreads_df['RIspread'].median() else 'lower_half')

if 0:
    hab_mask_day1 = hrc_df_day1['median'] == 'upper_half'
    GD_mask_day1 =  hrc_df_day1['median'] == 'lower_half'
    
    hab_mask_day2 = hrc_df_day2['median'] == 'upper_half'
    GD_mask_day2 =  hrc_df_day2['median'] == 'lower_half'
    
else:
    hab_mask_day1 = hrc_df_day1['ps_ri'] < 0.05
    GD_mask_day1 = hrc_df_day1['ps_ri'] > 0.05

    hab_mask_day2 = hrc_df_day2['ps_ri'] < 0.05
    GD_mask_day2 = hrc_df_day2['ps_ri'] > 0.05

"------"

'''
    Figure
    Left: Negative efffect of habit
    Right: No negative effeect of habit
'''
fig, ax = plt.subplots(1,2, sharey=True)
sns.barplot(ax = ax[0],
            data = hrc_df_day2[hab_mask_day2],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[0].set_ylim([40, 100])
ax[0].set_title("Negative effect of habit")
ax[0].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)
custom_labels = ["Random", "Congruent", "Incongruent"]  # Define your custom labels here
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles, custom_labels, fontsize = 12, loc='upper left', bbox_to_anchor=(1, 1))

sns.stripplot(ax = ax[0],
    y="HRC", 
    hue="Trial Type", 
    hue_order = ['Random', 'Congruent', 'Incongruent'],
    palette = colors1,
    data=hrc_df_day2[hab_mask_day2], dodge=True, alpha=0.6,
    edgecolor = 'k', linewidth = 1,
)

handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles[:-3], labels[:-3], title='Trial Type')

sns.barplot(ax = ax[1],
            data = hrc_df_day2[GD_mask_day2],
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))

sns.stripplot(ax = ax[1],
    y="HRC", 
    hue="Trial Type", 
    hue_order = ['Random', 'Congruent', 'Incongruent'],
    palette = colors1,
    data=hrc_df_day2[GD_mask_day2], dodge=True, alpha=0.6,
    edgecolor = 'k', linewidth = 1,
)

handles, labels = ax[1].get_legend_handles_labels()
ax[1].legend(handles[:-3], labels[:-3], title='Trial Type')

ax[1].set_ylim([40, 100])
ax[1].set_ylabel("Goal-Directed Responses (%)", fontsize = 20)
ax[1].set_title("No negative effect of habit")
plt.savefig("/home/sascha/Desktop/Paper_2024/Mar/res_fig3/hab_vs_GD_python.svg")
plt.show()

" ==== HRC Differences within Group?"
t,p = scipy.stats.ttest_rel(hrc_df_day2[(hrc_df_day2['ps_ri'] < 0.05) & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[(hrc_df_day2['ps_ri'] < 0.05) & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"HRC Random vs Incongruent, Negative habit effect, Day 2: t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day2[(hrc_df_day2['ps_ri'] < 0.05) & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[(hrc_df_day2['ps_ri'] < 0.05) & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'])
print(f"HRC Random vs Congruent, Negative habit effect, Day 2: t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day2[(hrc_df_day2['ps_ri'] < 0.05) & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_day2[(hrc_df_day2['ps_ri'] < 0.05) & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"HRC Congruent vs Incongruent, Negative habit effect, Day 2: t={t}, p={p}")

" ==== HRC Differences within Group?"
t,p = scipy.stats.ttest_rel(hrc_df_day2[(hrc_df_day2['ps_ri'] > 0.05) & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[(hrc_df_day2['ps_ri'] > 0.05) & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"HRC Random vs Incongruent, No negative habit effect, Day 2: t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day2[(hrc_df_day2['ps_ri'] > 0.05) & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[(hrc_df_day2['ps_ri'] > 0.05) & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'])
print(f"HRC Random vs Congruent, No negative habit effect, Day 2: t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day2[(hrc_df_day2['ps_ri'] > 0.05) & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_day2[(hrc_df_day2['ps_ri'] > 0.05) & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"HRC Congruent vs Incongruent, No negative habit effect, Day 2: t={t}, p={p}")

" ==== HRC Differences between Groups?"
t,p = scipy.stats.ttest_ind(hrc_df_day2[(hrc_df_day2['ps_ri'] > 0.05) & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[(hrc_df_day2['ps_ri'] < 0.05) & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'])
print(f"t={t}, p={p}")


print(f"There are {len(hrc_df_day2[hab_mask_day2]['ID'].unique())} participants in the 'habitual' group on day 2.")
print(f"There are {len(hrc_df_day2[GD_mask_day2]['ID'].unique())}  participants in the 'GD' group on day 2.")

#%%
'''
    Exploiters vs Suppressers Day 1
'''

exploit_mask_day1 = GD_mask_day1 & (hrc_df_day1['ps_cr'] < 0.05)
suppress_mask_day1 = GD_mask_day1 & (hrc_df_day1['ps_cr'] > 0.05)
    
fig, ax = plt.subplots(2,2, sharey=True, figsize = (8,8))

sns.barplot(ax = ax[0,0],
            data = hrc_df_day1[hab_mask_day1],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))

sns.stripplot(ax = ax[0,0],
    y="HRC", 
    hue="Trial Type", 
    hue_order = ['Random', 'Congruent', 'Incongruent'],
    palette = colors1,
    data = hrc_df_day1[hab_mask_day1], dodge=True, alpha=0.6,
    edgecolor = 'k', linewidth = 1,
)

handles, labels = ax[0,0].get_legend_handles_labels()
ax[0,0].legend(handles[:-3], labels[:-3], title='Trial Type')


ax[0,0].tick_params(axis='both', labelsize=14)
ax[0,0].set_title("Negative effect of habit")
ax[0,0].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)

sns.barplot(ax = ax[0,1],
            data = hrc_df_day1[GD_mask_day1],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[0,1].set_ylim([40, 100])
ax[0,1].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)
ax[0,1].set_title("No negative effect of habit")

sns.stripplot(ax = ax[0,1],
    y="HRC", 
    hue="Trial Type", 
    hue_order = ['Random', 'Congruent', 'Incongruent'],
    palette = colors1,
    data = hrc_df_day1[GD_mask_day1], dodge=True, alpha=0.6,
    edgecolor = 'k', linewidth = 1,
)

sns.barplot(ax = ax[1,0],
            data = hrc_df_day1[exploit_mask_day1],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[1,0].set_ylim([40, 100])
ax[1,0].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)
ax[1,0].set_title('"Adapters"')
ax[1,0].tick_params(axis='both', labelsize=14)
# plt.savefig("/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/hrc_python.svg")
# plt.title(f"Habedahabeda 2 (N={len(hrc_df_day1[hrc_df_day1['ps_ri'] < 0.05])/3})")

sns.stripplot(ax = ax[1,0],
    y="HRC", 
    hue="Trial Type", 
    hue_order = ['Random', 'Congruent', 'Incongruent'],
    palette = colors1,
    data = hrc_df_day1[exploit_mask_day1], dodge=True, alpha=0.6,
    edgecolor = 'k', linewidth = 1,
)


"===== Differences within group"
t,p = scipy.stats.ttest_rel(hrc_df_day1[exploit_mask_day1 & (hrc_df_day1['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day1[exploit_mask_day1 & (hrc_df_day1['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day1[exploit_mask_day1 & (hrc_df_day1['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day1[exploit_mask_day1 & (hrc_df_day1['Trial Type'] == 'Congruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day1[exploit_mask_day1 & (hrc_df_day1['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_day1[exploit_mask_day1 & (hrc_df_day1['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

sns.barplot(ax = ax[1,1],
            data = hrc_df_day1[suppress_mask_day1],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[1,1].set_ylim([40, 100])
ax[1,1].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)
ax[1,1].set_title('"Inhibitors"')
ax[1,1].get_legend().set_visible(False)
ax[1,0].get_legend().set_visible(False)
ax[0,1].get_legend().set_visible(False)

sns.stripplot(ax = ax[1,1],
    y="HRC", 
    hue="Trial Type", 
    hue_order = ['Random', 'Congruent', 'Incongruent'],
    palette = colors1,
    data = hrc_df_day1[suppress_mask_day1], dodge=True, alpha=0.6,
    edgecolor = 'k', linewidth = 1,
)

# plt.savefig("/home/sascha/Desktop/Paper_2024/Mar/res_fig3/res_fig3_python.svg")
plt.show()

"===== Differences within group"
t,p = scipy.stats.ttest_rel(hrc_df_day1[suppress_mask_day1 & (hrc_df_day1['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day1[suppress_mask_day1 & (hrc_df_day1['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day1[suppress_mask_day1 & (hrc_df_day1['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day1[suppress_mask_day1 & (hrc_df_day1['Trial Type'] == 'Congruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day1[suppress_mask_day1 & (hrc_df_day1['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_day1[suppress_mask_day1 & (hrc_df_day1['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

"===== Differences between groups"
t,p = scipy.stats.ttest_ind(hrc_df_day1[suppress_mask_day1 & (hrc_df_day1['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day1[exploit_mask_day1 & (hrc_df_day1['Trial Type'] == 'Random')]['HRC'])
print(f"Random Type, between groups: t={t}, p={p}")

t,p = scipy.stats.ttest_ind(hrc_df_day1[suppress_mask_day1 & (hrc_df_day1['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_day1[exploit_mask_day1 & (hrc_df_day1['Trial Type'] == 'Congruent')]['HRC'])
print(f"Congruent Type, between groups: t={t}, p={p}")

t,p = scipy.stats.ttest_ind(hrc_df_day1[suppress_mask_day1 & (hrc_df_day1['Trial Type'] == 'Incongruent')]['HRC'], 
                            hrc_df_day1[exploit_mask_day1 & (hrc_df_day1['Trial Type'] == 'Incongruent')]['HRC'])
print(f"Incongruent Type, between groups: t={t}, p={p}")

print(f"There are {len(hrc_df_day1[exploit_mask_day1]['ID'].unique())} participants in the exploit group on day 1.")
print(f"There are {len(hrc_df_day1[suppress_mask_day1]['ID'].unique())} participants in the suppress group on day 1.")


#%%
'''
    Exploiters vs Suppressers Day 2
'''

exploit_mask_day2 = GD_mask_day2 & (hrc_df_day2['ps_cr'] < 0.05)
suppress_mask_day2 = GD_mask_day2 & (hrc_df_day2['ps_cr'] > 0.05)
    
fig, ax = plt.subplots(2,2, sharey=True, figsize = (8,8))

sns.barplot(ax = ax[0,0],
            data = hrc_df_day2[hab_mask_day2],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[0,0].tick_params(axis='both', labelsize=14)
ax[0,0].set_title("Negative effect of habit")
ax[0,0].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)

sns.stripplot(ax = ax[0, 0],
    y="HRC", 
    hue="Trial Type", 
    hue_order = ['Random', 'Congruent', 'Incongruent'],
    palette = colors1,
    data = hrc_df_day2[hab_mask_day2], dodge=True, alpha=0.6,
    edgecolor = 'k', linewidth = 1,
)

sns.barplot(ax = ax[0,1],
            data = hrc_df_day2[GD_mask_day2],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[0,1].set_ylim([40, 100])
ax[0,1].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)
ax[0,1].set_title("No negative effect of habit")

sns.stripplot(ax = ax[0, 1],
    y="HRC", 
    hue="Trial Type", 
    hue_order = ['Random', 'Congruent', 'Incongruent'],
    palette = colors1,
    data = hrc_df_day2[GD_mask_day2], dodge=True, alpha=0.6,
    edgecolor = 'k', linewidth = 1,
)


sns.barplot(ax = ax[1,0],
            data = hrc_df_day2[exploit_mask_day2],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[1,0].set_ylim([40, 100])
ax[1,0].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)
ax[1,0].set_title('"Adapters"')
ax[1,0].tick_params(axis='both', labelsize=14)
# plt.savefig("/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/hrc_python.svg")
# plt.title(f"Habedahabeda 2 (N={len(hrc_df_day2[hrc_df_day2['ps_ri'] < 0.05])/3})")

sns.stripplot(ax = ax[1, 0],
    y="HRC", 
    hue="Trial Type", 
    hue_order = ['Random', 'Congruent', 'Incongruent'],
    palette = colors1,
    data = hrc_df_day2[exploit_mask_day2], dodge=True, alpha=0.6,
    edgecolor = 'k', linewidth = 1,
)

sns.barplot(ax = ax[1,1],
            data = hrc_df_day2[suppress_mask_day2],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[1,1].set_ylim([40, 100])
ax[1,1].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)
ax[1,1].set_title('"Inhibitors"')
ax[1,1].get_legend().set_visible(False)
ax[1,0].get_legend().set_visible(False)
ax[0,1].get_legend().set_visible(False)

sns.stripplot(ax = ax[1, 1],
    y="HRC",
    hue="Trial Type", 
    hue_order = ['Random', 'Congruent', 'Incongruent'],
    palette = colors1,
    data = hrc_df_day2[suppress_mask_day2], dodge=True, alpha=0.6,
    edgecolor = 'k', linewidth = 1,
)

handles, labels = ax[1,1].get_legend_handles_labels()
ax[1,1].legend(handles[:-3], labels[:-3], title='Trial Type')

plt.savefig("/home/sascha/Desktop/Paper_2024/Mar/res_fig2/res_fig2_python.svg")
plt.show()

"===== Differences within group"
t,p = scipy.stats.ttest_rel(hrc_df_day2[exploit_mask_day2 & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[exploit_mask_day2 & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day2[exploit_mask_day2 & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[exploit_mask_day2 & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day2[exploit_mask_day2 & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_day2[exploit_mask_day2 & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

"===== Differences within group"
'Inhibitors'
t,p = scipy.stats.ttest_rel(hrc_df_day2[suppress_mask_day2 & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[suppress_mask_day2 & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day2[suppress_mask_day2 & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[suppress_mask_day2 & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'])
print(f"t={t}, p={p}")
t
t,p = scipy.stats.ttest_rel(hrc_df_day2[suppress_mask_day2 & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_day2[suppress_mask_day2 & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

"===== Differences between groups"
t,p = scipy.stats.ttest_ind(hrc_df_day2[suppress_mask_day2 & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[exploit_mask_day2 & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'])
print(f"Random Type, between groups: t={t}, p={p}")

t,p = scipy.stats.ttest_ind(hrc_df_day2[suppress_mask_day2 & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_day2[exploit_mask_day2 & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'])
print(f"Congruent Type, between groups: t={t}, p={p}")

t,p = scipy.stats.ttest_ind(hrc_df_day2[suppress_mask_day2 & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'], 
                            hrc_df_day2[exploit_mask_day2 & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"Incongruent Type, between groups: t={t}, p={p}")

print(f"There are {len(hrc_df_day2[exploit_mask_day2]['ID'].unique())} participants in the exploit group on day 2.")
print(f"There are {len(hrc_df_day2[suppress_mask_day2]['ID'].unique())} participants in the suppress group on day 2.")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['CRspread'], 
                           complete_df_all[complete_df_all['day'] == 2]['hpcf_rand'])

print(f"Pearson corr CR vs HRC Baseline: r={r}, p={p}.")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['RIspread'], 
                           complete_df_all[complete_df_all['day'] == 2]['hpcf_rand'])

print(f"Pearson corr RI vs HRC Baseline: r={r}, p={p}.")

'''
    Check if any group noticed the sequence more
'''
hrc_df_day2['q_notice_a_sequence']  = pd.Series(dtype=float)
hrc_df_day2['q_notice_a_sequence'] = hrc_df_day2.apply(lambda row: complete_df_all[complete_df_all['ID'] == row['ID']]['q_notice_a_sequence'].unique()[0], axis = 1)

"Negative effect of habit"
tempdf = hrc_df_day2[hab_mask_day2].loc[:, ['ID', 'q_notice_a_sequence']].groupby(['ID'], as_index=False).mean()
print(f"{(tempdf['q_notice_a_sequence']==1).sum()*100/ len(tempdf)}% noticed a sequence.")

"No negative effect of habit"
tempdf = hrc_df_day2[GD_mask_day2].loc[:, ['ID', 'q_notice_a_sequence']].groupby(['ID'], as_index=False).mean()
print(f"{(tempdf['q_notice_a_sequence']==1).sum()*100/ len(tempdf)}% noticed a sequence.")

"Adapters"
tempdf = hrc_df_day2[exploit_mask_day2].loc[:, ['ID', 'q_notice_a_sequence']].groupby(['ID'], as_index=False).mean()
print(f"{(tempdf['q_notice_a_sequence']==1).sum()*100/ len(tempdf)}% noticed a sequence.")

"Inhibitors"
tempdf = hrc_df_day2[suppress_mask_day2].loc[:, ['ID', 'q_notice_a_sequence']].groupby(['ID'], as_index=False).mean()
print(f"{(tempdf['q_notice_a_sequence']==1).sum()*100/ len(tempdf)}% noticed a sequence.")

#%%
'''
    CR / RI Table Day two
    
    
            | RI > 0 | RI = 0 |
    CR > 0  |        |        |
    ----------------------------------
    CR = 0  |        |        |
    ----------------------------------
            |        |        |
    
'''

fig, ax = plt.subplots(3,3, sharey=True, figsize = (18, 18))

ps_cr_cond = [["hrc_df_day2['ps_cr']<0.05", "hrc_df_day2['ps_cr']<0.05", "hrc_df_day2['ps_cr']<0.05"], 
              ["hrc_df_day2['ps_cr']>=0.05", "hrc_df_day2['ps_cr']>=0.05", "hrc_df_day2['ps_cr']>=0.05"], 
              ["hrc_df_day2['ps_cr']<=1", "hrc_df_day2['ps_cr']<=1", "hrc_df_day2['ps_cr']<=1"]]

ps_ri_cond = [["hrc_df_day2['ps_ri']<0.05", "hrc_df_day2['ps_ri']>=0.05", "hrc_df_day2['ps_ri']<=1"], 
              ["hrc_df_day2['ps_ri']<0.05", "hrc_df_day2['ps_ri']>=0.05", "hrc_df_day2['ps_ri']<=1"], 
              ["hrc_df_day2['ps_ri']<0.05", "hrc_df_day2['ps_ri']>=0.05", "hrc_df_day2['ps_ri']<=1"]]

for row in range(3):
    for col in range(3):
        
        mask = eval(ps_cr_cond[row][col]) & eval(ps_ri_cond[row][col])
        
        sns.barplot(ax = ax[row, col],
                    data = hrc_df_day2[mask],
                    # x = 'Trial Type',
                    y = 'HRC',
                    hue = 'Trial Type',
                    hue_order = ['Random', 'Congruent', 'Incongruent'],
                    palette = colors1,
                    errorbar = ('se', 1))
        
        ax[row,col].set_ylabel('GD Responses (%)', fontsize = 25)
        ax[row, col].tick_params(labelsize = 30)
        ax[row, col].set_ylim(50, 100)
        
        ax[row,col].text(0.1, 52, f'N={len(hrc_df_day2[mask])//3}', fontsize = 30)

plt.savefig('/home/sascha/Desktop/Nextcloud/work/TAC/behav_table_day2_python.svg')
plt.show()

#%%
'''
    Adaptation score
'''

complete_df_day2 = complete_df_all[complete_df_all['day'] == 2]

seqlearn_df_day1['day'] = 1
seqlearn_df_day2['day'] = 2

habitual_df, GD_df, modulators_df, antimods_df, ps_df_day1 = anal.find_strategies(expdata_df_day1, 
                                                         plot_single = False,
                                                         correctp = False)

habitual_df, GD_df, modulators_df, antimods_df, ps_df_day2 = anal.find_strategies(expdata_df_day2, 
                                                         plot_single = False,
                                                         correctp = False)

ps_df_day1['day'] = 1
ps_df_day2['day'] = 2

df_day2 = pd.merge(complete_df_day2, seqlearn_df_day2, on ='ID')
df_day2 = pd.merge(df_day2, ps_df_day2, on ='ID')
df_day2['ri_0'] = df_day2['ps_ri'].map(lambda x: 'yes' if x > 0.05 else 'no')

df_day2['exploit_score'] = df_day2.apply(lambda row: row[f'{param2}']/row[f'{param1}'], axis=1)

#%%
'''
    Fig
    Adaptation score
'''

limit_to_ssl = 0 # limit to strong sequence learners?

if limit_to_ssl:
    dfadapt_day2 = df_day2[df_day2['ID'].isin(seqlearners_df_day2['ID'].unique())]
    
else:
    dfadapt_day2 = df_day2
    

fig, ax = plt.subplots(1,2, sharey=True, sharex = True, figsize = (8,4))
sns.scatterplot(dfadapt_day2,
                x = f'{param1}',
                y = f'{param2}',
                # palette = ['r', 'b'],
                hue = 'ri_0',
                hue_order = ['yes', 'no'],
                ax = ax[0])
custom_labels = ["no negative effect of habit", "negative effect of habit"]  # Define your custom labels here
handles, labels = ax[0].get_legend_handles_labels()

ax[0].legend(handles, custom_labels, fontsize = 15, loc='upper left', bbox_to_anchor=(1, 1))
ax[0].set_xlabel(r'$\theta_{Rep}$', fontsize = 20)
ax[0].set_ylabel(r'$\theta_{Switch}$', fontsize = 20)
sns.scatterplot(dfadapt_day2,
                x = f'{param1}',
                y = f'{param2}',
                # palette = ['r', 'b'],
                hue = 'exploit_score',
                ax = ax[1])

ax[1].legend(title='Adaptation Score', loc='upper left', bbox_to_anchor=(1, 1))
ax[1].get_legend().get_title().set_fontsize(14)
# plt.plot([0, 2.5], [0, 2.5], color='k', linewidth = 0.5)
# plt.plot([0, 2.5], [0, 1.1])
# if param1 == 'theta_rep':
#     ax[1].set_xlim([0.25, 2])
    # ax[1].set_ylim([0, 2.5])
ax[1].set_xlabel(r'$\theta_{Rep}$', fontsize = 20)
ax[1].set_ylabel(r'$\theta_{Switch}$', fontsize = 20)

plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig3/res_fig3_python.svg', bbox_inches = 'tight')
plt.show()

# plt.plot([0, 2.5], [0, 2.5], color='k', linewidth = 0.5)
# plt.plot([0, 2.5], [0, 1.1])
# ax[0].set_xlim([0, 2.5])
# ax[0].set_ylim([0, 2.5])
# plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig5b/res_fig5b_b_python.svg', bbox_inches = 'tight')

#%%
'''
    Plots of posterior means
'''

dfadapt_day2['seqlearner'] = dfadapt_day2['ID'].map(lambda x: 'strong' if x in seqlearners_df_day2['ID'].unique() else 
                                          'weak' if x in notseqlearners_df_day2['ID'].unique() else
                                          'None')

dfadapt_day2['strategy'] = dfadapt_day2['ID'].map(lambda x: 'Inhibitor' if x in hrc_df_day2[suppress_mask_day2]['ID'].unique() else 
                                          'Adapter' if x in hrc_df_day2[exploit_mask_day2]['ID'].unique() else
                                          'negative effect of habit' if x in hrc_df_day2[hab_mask_day2]['ID'].unique() else
                                          'None')


fig, ax = plt.subplots()
sns.scatterplot(dfadapt_day2,
                x = f'{param1}',
                y = f'{param2}',
                # palette = ['r', 'b'],
                hue = 'seqlearner',
                # hue_order = ['weak', 'strong'],
                ax = ax)
# custom_labels = ["weak habit learner", "strong habit learner"]  # Define your custom labels here
# handles, labels = ax.get_legend_handles_labels()

# ax.legend(handles, custom_labels, fontsize = 15, loc='upper left', bbox_to_anchor=(1, 1))
ax.set_xlabel(r'$\theta_{Rep}$', fontsize = 20)
ax.set_ylabel(r'$\theta_{Switch}$', fontsize = 20)

# plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig3/res_fig3_python.svg', bbox_inches = 'tight')
plt.show()

fig, ax = plt.subplots()
sns.scatterplot(dfadapt_day2,
                x = f'{param1}',
                y = f'{param2}',
                palette = ['g', 'y', 'm'],
                hue = 'strategy',
                # hue_order = ['weak', 'strong'],
                ax = ax)
# custom_labels = ["weak habit learner", "strong habit learner"]  # Define your custom labels here
# handles, labels = ax.get_legend_handles_labels()

# ax.legend(handles, custom_labels, fontsize = 15, loc='upper left', bbox_to_anchor=(1, 1))
ax.set_xlabel(r'$\theta_{Rep}$', fontsize = 20)
ax.set_ylabel(r'$\theta_{Switch}$', fontsize = 20)

# plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig3/res_fig3_python.svg', bbox_inches = 'tight')
plt.show()


fig, ax = plt.subplots(1,3, sharey=True, sharex = True, figsize = (12,4))
"Ax[0]"
sns.scatterplot(dfadapt_day2,
                x = f'{param1}',
                y = f'{param2}',
                # palette = ['r', 'b'],
                hue = 'ri_0',
                hue_order = ['yes', 'no'],
                palette = {'yes': '#d62728', 'no': '#1f77b4'},
                ax = ax[0])
custom_labels = ["no negative effect of habit", "negative effect of habit"]  # Define your custom labels here
handles, labels = ax[0].get_legend_handles_labels()
# ax[0].plot(np.arange(0.5,2),np.arange(0.5,2)-0.8, color='k')
ax[0].legend(handles, custom_labels, fontsize = 13, loc='upper left', bbox_to_anchor=(1, 1))

ax[0].set_xlabel(r'$\theta_{Rep}$', fontsize = 20)
ax[0].set_ylabel(r'$\theta_{Switch}$', fontsize = 20)

"Ax[1]"
sns.scatterplot(dfadapt_day2,
                x = f'{param1}',
                y = f'{param2}',
                # palette = ['r', 'b'],
                hue = 'strategy',
                hue_order = ['Adapter', 'Inhibitor', 'negative effect of habit'],
                palette = {'negative effect of habit': '#1f77b4', 
                           'Adapter': '#e377c2', 'Inhibitor': '#9467bd'},
                ax = ax[1])

ax[1].legend(title='', fontsize = 13)
# plt.plot([0, 2.5], [0, 2.5], color='k', linewidth = 0.5)
# plt.plot([0, 2.5], [0, 1.1])
# if param1 == 'theta_rep':
#     ax[1].set_xlim([0.25, 2])
    # ax[1].set_ylim([0, 2.5])
ax[1].set_xlabel(r'$\theta_{Rep}$', fontsize = 20)
ax[1].set_ylabel(r'$\theta_{Switch}$', fontsize = 20)

"Ax[2]"
sns.scatterplot(dfadapt_day2,
                x = f'{param1}',
                y = f'{param2}',
                # palette = ['r', 'b'],
                hue = 'exploit_score',
                ax = ax[2])

ax[2].legend(title='Adaptation Score', loc='upper left', bbox_to_anchor=(1, 1))
ax[2].get_legend().get_title().set_fontsize(13)
# plt.plot([0, 2.5], [0, 2.5], color='k', linewidth = 0.5)
# plt.plot([0, 2.5], [0, 1.1])
# if param1 == 'theta_rep':
#     ax[2].set_xlim([0.25, 2])
    # ax[2].set_ylim([0, 2.5])
ax[2].set_xlabel(r'$\theta_{Rep}$', fontsize = 20)
ax[2].set_ylabel(r'$\theta_{Switch}$', fontsize = 20)

plt.savefig('/home/sascha/Desktop/Paper_2024/May/res_fig3/res_fig3_python.svg', bbox_inches = 'tight')
plt.show()

#%%
'''
    Scatterplots of posterior means colour-coded by CR/ RI table.
'''

IDs1 = hrc_df_day2[(hrc_df_day2['ps_cr']<0.05) & (hrc_df_day2['ps_ri']<0.05)]['ID'].unique()
IDs2 = hrc_df_day2[(hrc_df_day2['ps_cr']<0.05) & (hrc_df_day2['ps_ri']>0.05)]['ID'].unique() 
IDs3 = hrc_df_day2[(hrc_df_day2['ps_cr']>0.05) & (hrc_df_day2['ps_ri']<0.05)]['ID'].unique() 
IDs4 = hrc_df_day2[(hrc_df_day2['ps_cr']>0.05) & (hrc_df_day2['ps_ri']>0.05)]['ID'].unique() 

dfadapt_day2['behav_group'] = dfadapt_day2['ID'].map(lambda x: 'CR > 0, RI > 0' if (x in IDs1) else 
                                                     'CR > 0, RI ~ 0' if x in IDs2 else 
                                                     'CR ~ 0, RI > 0' if x in IDs3 else
                                                     'CR ~ 0, RI ~ 0' if x in IDs4 else None)

# dfadapt_day2[dfadapt_day2['ID'].isin(IDs)]['behav_group'] = 1

# dfadapt_day2[dfadapt_day2['ID'].isin(hrc_df_day2[(hrc_df_day2['ps_cr']<0.05) & 
#                                                  (hrc_df_day2['ps_ri']>0.05)]['ID'].unique())]['behav_group'] = 2

# dfadapt_day2[dfadapt_day2['ID'].isin(hrc_df_day2[(hrc_df_day2['ps_cr']>0.05) & 
#                                                  (hrc_df_day2['ps_ri']<0.05)]['ID'].unique())]['behav_group'] = 3

# dfadapt_day2[dfadapt_day2['ID'].isin(hrc_df_day2[(hrc_df_day2['ps_cr']>0.05) & 
#                                                  (hrc_df_day2['ps_ri']>0.05)]['ID'].unique())]['behav_group'] = 4

fig, ax = plt.subplots()

sns.scatterplot(dfadapt_day2,
                x = f'{param1}',
                y = f'{param2}',
                palette = ['r', 'b', 'g', '#A0522D'],
                # palette = ['#D4AFB9', '#D1CFE2', '#7EC4CF', '#FA9189'],
                hue = 'behav_group',
                ax = ax)

# ax.legend(title='Adaptation Score', loc='upper left', bbox_to_anchor=(1, 1))
# ax.get_legend().get_title().set_fontsize(13)
# plt.plot([0, 2.5], [0, 2.5], color='k', linewidth = 0.5)
# plt.plot([0, 2.5], [0, 1.1])
# if param1 == 'theta_rep':
#     ax.set_xlim([0.25, 2])
    # ax.set_ylim([0, 2.5])
ax.set_xlabel(r'$\theta_{Rep}$', fontsize = 20)
ax.set_ylabel(r'$\theta_{Switch}$', fontsize = 20)
plt.savefig('/home/sascha/Desktop/TAC/posteriors_behav_table_python.svg')

plt.show()

#%%
'''
    Annotate
'''
for rowidx in range(60):
    x = dfadapt_day2.iloc[rowidx,:]['theta_rep']
    y = dfadapt_day2.iloc[rowidx,:]['theta_conflict']
    # z = dfadapt_day2.iloc[rowidx,:]['theta_Q']
    z = dfadapt_day2.iloc[rowidx,:]['ID']
    
    fig, ax = plt.subplots()
    sns.scatterplot(dfadapt_day2,
                    x = f'{param1}',
                    y = f'{param2}',
                    # palette = ['r', 'b'],
                    hue = 'strategy',
                    hue_order = ['Adapter', 'Inhibitor', 'negative effect of habit'],
                    palette = {'negative effect of habit': '#1f77b4', 
                               'Adapter': '#e377c2', 'Inhibitor': '#9467bd'},
                    ax = ax)
    
    plt.annotate(f'{z}',
                 xy=(x, y),
                 xytext=(x+0.1, y+0.1),
                 arrowprops=dict(facecolor='blue', shrink=0.05))
    
    
    ax.legend(title='', fontsize = 13)
    # plt.plot([0, 2.5], [0, 2.5], color='k', linewidth = 0.5)
    # plt.plot([0, 2.5], [0, 1.1])
    # if param1 == 'theta_rep':
    #     ax.set_xlim([0.25, 2])
        # ax[1].set_ylim([0, 2.5])
    ax.set_xlabel(r'$\theta_{Rep}$', fontsize = 20)
    ax.set_ylabel(r'$\theta_{Switch}$', fontsize = 20)
    
    ax.legend(title='', fontsize = 13)
    # plt.plot([0, 2.5], [0, 2.5], color='k', linewidth = 0.5)
    # plt.plot([0, 2.5], [0, 1.1])
    # if param1 == 'theta_rep':
    #     ax.set_xlim([0.25, 2])
        # ax[1].set_ylim([0, 2.5])
    ax.set_xlabel(r'$\theta_{Rep}$', fontsize = 20)
    ax.set_ylabel(r'$\theta_{Switch}$', fontsize = 20)
    ax.get_legend().set_visible(False)
    plt.show()
    
#%%
'''
    Histogram of Habit Exploitation Score
'''
fig, ax = plt.subplots(figsize = (6,6))
# sns.histplot(data = df_day2, x='exploit_score', bins = 9, binrange=[0.2, 1.5], ax = ax)
sns.histplot(data = dfadapt_day2, x='exploit_score', ax = ax)
ax.set_xlabel('Adaptation Score')
ax.set_ylabel('Count', fontsize = 20)
ax.set_xlabel('Adaptation Score', fontsize = 20)
ax.tick_params(axis='both', labelsize=14)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title(f"{model}")
ax2 = ax.twinx()
sns.kdeplot(data = dfadapt_day2, x='exploit_score', ax = ax2)
ax2.grid(False)
ax2.get_yaxis().set_visible(False)
plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig3/res_fig3c_python.svg', bbox_inches = 'tight')
plt.show()

fig, ax = plt.subplots()
# sns.histplot(data = dfadapt_day2, x='exploit_score', bins = 9, binrange=[0.2, 1.5], ax = ax)
sns.scatterplot(data = dfadapt_day2, x='exploit_score', y='theta_rep')
ax.set_xlabel('Adaptation Score')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title(f"{model}")
# plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig3/res_fig3c_python.svg', bbox_inches = 'tight')
plt.show()


#%%
if 0:
    '''
        2 Example participants, from 3 sections each
    '''
    IDs = ['60a3f8075b013de7b5518e96', '57deda2591b7fc0001493e95', 
           '5fb46dd5d9ece50422838e7a', '5d5a75c570a7c1000152623e', 
           '63174af7d57182f9bf90c094', '58aca85e0da7f10001de92d4', '596f961cfe061d00011e3e03']
    
    # Exploit scores > 1 and hpcf_rand < 0.9
    IDs = ['5eaadc0a7adeb404eea9c3c0', '5b5e0e86902ad10001cfcc59', '62c97799bd8ab72a531abde0']
    
    for ID in dfadapt_day2.sort_values(by='exploit_score')['ID'].unique():
        utils.plot_hpcf(complete_df_all[complete_df_all['ID'] == ID], 
                        title='score = %.2f, ID %s'%(dfadapt_day2[dfadapt_day2['ID'] == ID]['exploit_score'], ID))

#%%
'''
    Ideally, theta_Q does not correlate with the other model params.
'''
r,p = scipy.stats.pearsonr(dfadapt_day2[f'{param1}'], dfadapt_day2['theta_Q'])
print(f"{param1} vs theta_Q: r={r}, p={p}")

r,p = scipy.stats.pearsonr(dfadapt_day2[f'{param2}'], dfadapt_day2['theta_Q'])
print(f"{param2} vs theta_Q: r={r}, p={p}")

r,p = scipy.stats.pearsonr(dfadapt_day2['exploit_score'], dfadapt_day2['theta_Q'])
print(f"exploit_score vs theta_Q: r={r}, p={p}")

r,p = scipy.stats.pearsonr(dfadapt_day2['exploit_score'], dfadapt_day2['chis_ri'])
print(f"exploit_score vs chis_ri: r={r}, p={p}")

r,p = scipy.stats.pearsonr(dfadapt_day2['exploit_score'], dfadapt_day2['chis_cr'])
print(f"exploit_score vs chis_cr: r={r}, p={p}")

#%%
'''
    Correlations of exploit score
'''

'''
    Group
'''

expl_means = []
expl_means.append(dfadapt_day2[dfadapt_day2['group']==0]['exploit_score'].mean())
expl_means.append(dfadapt_day2[dfadapt_day2['group']==1]['exploit_score'].mean())
expl_means.append(dfadapt_day2[dfadapt_day2['group']==2]['exploit_score'].mean())
expl_means.append(dfadapt_day2[dfadapt_day2['group']==3]['exploit_score'].mean())

expl_stdevs = []
expl_stdevs.append(dfadapt_day2[dfadapt_day2['group']==0]['exploit_score'].std())
expl_stdevs.append(dfadapt_day2[dfadapt_day2['group']==1]['exploit_score'].std())
expl_stdevs.append(dfadapt_day2[dfadapt_day2['group']==2]['exploit_score'].std())
expl_stdevs.append(dfadapt_day2[dfadapt_day2['group']==3]['exploit_score'].std())

sns.barplot(data = dfadapt_day2,
            y='exploit_score',
            x='group',
            errorbar=('se', 1))

import pingouin as pg
aov = pg.anova(dv = 'exploit_score',
                  between = ['group'], 
                  data = dfadapt_day2, 
                  detailed = True,
                  effsize = 'np2')

# print(aov)
print(aov.loc[:, ['Source', 'F', 'p-unc', 'np2']])

dfadapt_day2['blockordergroup'] = dfadapt_day2['group'].map(lambda x: 1 if (x == 0 or x == 2) else 2 if x == 1 or x == 3 else 4)

fig, ax = plt.subplots()
sns.barplot(data = dfadapt_day2,
            y='exploit_score',
            x='blockordergroup',
            errorbar=('se', 1))

sns.stripplot(data = dfadapt_day2,
            y='exploit_score',
            x='blockordergroup')
ax.set_ylabel('Adaptation score', fontsize = 20)
ax.set_xlabel('Block order', fontsize = 20)
plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig3/adapt_by_group_python.svg')
plt.show()

t,p = scipy.stats.ttest_ind(dfadapt_day2[dfadapt_day2['blockordergroup'] == 1]['exploit_score'],
                            dfadapt_day2[dfadapt_day2['blockordergroup'] == 2]['exploit_score'])

print(f"t={t}, p={p}")

'''
    Age
'''
r,p = scipy.stats.pearsonr(dfadapt_day2['age'], dfadapt_day2['exploit_score'])
print(f"r={r}, p={p}")
fig, ax = plt.subplots()
sns.regplot(data=dfadapt_day2,
           x='age',
           y='exploit_score')
ax.text(45, 1.4, "r=%.2f, p=%.2f"%(r,p))
plt.show()

'''
    Noticing a sequence
'''
from scipy.stats import chi2_contingency
t,p = scipy.stats.ttest_ind(dfadapt_day2[dfadapt_day2['q_notice_a_sequence']==0]['exploit_score'], 
                            dfadapt_day2[dfadapt_day2['q_notice_a_sequence']==1]['exploit_score'])

print(f"t={t}, p={p}")
# _ = utils.plot_grouplevel(expdata_df_day2[expdata_df_day2['ID'].isin(df_day2[df_day2['q_notice_a_sequence']==0]['ID'].unique())], plot_single = False)
# _ = utils.plot_grouplevel(expdata_df_day2[expdata_df_day2['ID'].isin(df_day2[df_day2['q_notice_a_sequence']==1]['ID'].unique())], plot_single = False)

'''
    Model parameter Q
'''
r,p = scipy.stats.pearsonr(dfadapt_day2['theta_Q'], dfadapt_day2['exploit_score'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(dfadapt_day2['theta_rep'], dfadapt_day2['exploit_score'])
print(f"r={r}, p={p}")

'''
    Performance
'''
r,p = scipy.stats.pearsonr(dfadapt_day2['points'], dfadapt_day2['exploit_score'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(dfadapt_day2['points_dtt'], dfadapt_day2['exploit_score'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(dfadapt_day2['points_stt'], dfadapt_day2['exploit_score'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(dfadapt_day2['hpcf_rand'], dfadapt_day2['exploit_score'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(dfadapt_day2['hpcf_cong'], dfadapt_day2['exploit_score'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(dfadapt_day2['hpcf_incong'], dfadapt_day2['exploit_score'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(dfadapt_day2['CRspread'], dfadapt_day2['exploit_score'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(dfadapt_day2['RIspread'], dfadapt_day2['exploit_score'])
print(f"r={r}, p={p}")

fig, ax = plt.subplots()
sns.regplot(data=dfadapt_day2, x='RIspread', y='exploit_score')
plt.show()

#%%
'''
    HRC for upper and lower median of exploit_score.
'''

hrc_df_day2 = complete_df_all[complete_df_all['day'] == 2].loc[:, ['ID', 'hpcf_rand', 'hpcf_cong', 'hpcf_incong']]
hrc_df_day2 = pd.merge(hrc_df_day2, dfadapt_day2.loc[:, ['ID', 'exploit_score']], on = 'ID')
hrc_df_day2['exploit'] = hrc_df_day2['exploit_score'].map(lambda x: 'upper_half' if x > hrc_df_day2['exploit_score'].median() else
                                                          'lower_half')

hrc_df_day2 = hrc_df_day2.melt(id_vars = ['ID', 'exploit'], value_vars=  ['hpcf_rand', 
                                                                    'hpcf_incong',
                                                                    'hpcf_cong'])

hrc_df_day2['variable'] = hrc_df_day2['variable'].map(lambda x: 'Random' if x == 'hpcf_rand' else
                                                      'Congruent' if x == 'hpcf_cong' else
                                                      'Incongruent' if x == 'hpcf_incong' else
                                                      'none')

hrc_df_day2['value'] = hrc_df_day2['value']*100

colors = {'Random': '#67b798', 'Congruent': '#BE54C6', 'Incongruent': '#7454C7'}
fig, ax = plt.subplots(1, 2, sharey = True)
sns.barplot(ax = ax[0],
            data = hrc_df_day2[hrc_df_day2['exploit'] == 'upper_half'],
            # x = 'Trial Type',
            y = 'value',
            hue = 'variable',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors,
            errorbar = ('se', 1))
ax[0].legend(title='Trial Type')
ax[0].set_ylim([60, 100])
ax[0].set_ylabel('Goal-Directed Responses(%)')

sns.barplot(ax = ax[1],
            data = hrc_df_day2[hrc_df_day2['exploit'] == 'lower_half'],
            # x = 'Trial Type',
            y = 'value',
            hue = 'variable',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors,
            errorbar = ('se', 1))
ax[1].set_ylim([60, 100])
# ax.set_ylabel("Goal-Directed Responses (%)", fontsize = 18)
# custom_labels = ["Random", "Congruent (Habit helps)", "Incongruent (Habit hinders)"]  # Define your custom labels here
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, custom_labels, fontsize = 15, loc='upper left', bbox_to_anchor=(1, 1))
ax[1].legend(title='Trial Type')
plt.show()

t,p=scipy.stats.ttest_ind(hrc_df_day2[(hrc_df_day2['exploit'] == 'lower_half') & (hrc_df_day2['variable'] == 'Random')]['value'],
hrc_df_day2[(hrc_df_day2['exploit'] == 'upper_half') & (hrc_df_day2['variable'] == 'Random')]['value'])
print(f"exploit_score median split, upper vs lower half, Random DTT: t={t}, p={p}")

t,p=scipy.stats.ttest_ind(hrc_df_day2[(hrc_df_day2['exploit'] == 'lower_half') & (hrc_df_day2['variable'] == 'Congruent')]['value'],
hrc_df_day2[(hrc_df_day2['exploit'] == 'upper_half') & (hrc_df_day2['variable'] == 'Congruent')]['value'])
print(f"exploit_score median split, upper vs lower half, Congruent DTT:  t={t}, p={p}")

t,p=scipy.stats.ttest_ind(hrc_df_day2[(hrc_df_day2['exploit'] == 'lower_half') & (hrc_df_day2['variable'] == 'Incongruent')]['value'],
hrc_df_day2[(hrc_df_day2['exploit'] == 'upper_half') & (hrc_df_day2['variable'] == 'Incongruent')]['value'])
print(f"exploit_score median split, upper vs lower half, Incongruent DTT:  t={t}, p={p}")

#%%
'''
    Model parameters correlate with participant behaviour.
'''
print("===========================")
print("Correlation Model Parameters & Spreads resp HRC.")
r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_Q'],
                           complete_df_all[complete_df_all['day'] == 2]['hpcf_rand'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rep'],
                           complete_df_all[complete_df_all['day'] == 2]['hpcf_rand'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_conflict'],
                           complete_df_all[complete_df_all['day'] == 2]['hpcf_rand'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['lr'],
                           complete_df_all[complete_df_all['day'] == 2]['hpcf_rand'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rep'],
                           complete_df_all[complete_df_all['day'] == 2]['CRspread'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_conflict'],
                           complete_df_all[complete_df_all['day'] == 2]['RIspread'])
print(f"r={r}, p={p}")

'''
    Model parameters do not correlate with each other.
'''
print("===========================")
print("Correlation among Model Parameters.")
r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_Q'],
                           complete_df_all[complete_df_all['day'] == 2]['theta_rep'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_Q'],
                           complete_df_all[complete_df_all['day'] == 2]['theta_conflict'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rep'],
                           complete_df_all[complete_df_all['day'] == 2]['theta_conflict'])
print(f"r={r}, p={p}")

complete_df_all['comb'] = complete_df_all['theta_rep']  - complete_df_all['theta_conflict'] 

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['comb'],
                           complete_df_all[complete_df_all['day'] == 2]['theta_Q'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['comb'],
                           complete_df_all[complete_df_all['day'] == 2]['RIspread'])
print(f"r={r}, p={p}")

print("===========================")
print("Correlation Model Parameters & Spreads, resp HRC.")
r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['CRspread'],
                           complete_df_all[complete_df_all['day'] == 2]['theta_conflict'])
print(f"r={r}, p={p}")

# r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_Q'],
#                            complete_df_all[complete_df_all['day'] == 2]['comb'])
# print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rep'],
                           complete_df_all[complete_df_all['day'] == 2]['hpcf_rand'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_conflict'],
                           complete_df_all[complete_df_all['day'] == 2]['hpcf_rand'])
print(f"r={r}, p={p}")

'''
    Age effects
'''
print("===========================")
print("Age effects")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_Q'],
                           complete_df_all[complete_df_all['day'] == 2]['age'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rep'],
                           complete_df_all[complete_df_all['day'] == 2]['age'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_conflict'],
                           complete_df_all[complete_df_all['day'] == 2]['age'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['hpcf_rand'],
                           complete_df_all[complete_df_all['day'] == 2]['age'])
print(f"Pearson r GD(Random DTT) vs Age: r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['hpcf_cong'],
                           complete_df_all[complete_df_all['day'] == 2]['age'])
print(f"Pearson r GD(Cong DTT) vs Age: r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['hpcf_incong'],
                           complete_df_all[complete_df_all['day'] == 2]['age'])
print(f"Pearson r GD(Incong DTT) vs Age: r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['CRspread'],
                           complete_df_all[complete_df_all['day'] == 2]['age'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['RIspread'],
                           complete_df_all[complete_df_all['day'] == 2]['age'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['CIspread'],
                           complete_df_all[complete_df_all['day'] == 2]['age'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['points'],
                           complete_df_all[complete_df_all['day'] == 2]['age'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['ER_dtt'],
                           complete_df_all[complete_df_all['day'] == 2]['age'])
print(f"Pearson r ER(DTT) vs Age: r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['ER_stt'],
                           complete_df_all[complete_df_all['day'] == 2]['age'])
print(f"Pearson r ER(STT) vs Age: r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['ER_diff_stt'],
                           complete_df_all[complete_df_all['day'] == 2]['age'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['RT_stt'],
                           complete_df_all[complete_df_all['day'] == 2]['age'])
print(f"Pearson r RT(STT) vs Age: r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['RT_dtt'],
                           complete_df_all[complete_df_all['day'] == 2]['age'])
print(f"Pearson r RT(DTT) vs Age: r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['RT_diff_stt'],
                           complete_df_all[complete_df_all['day'] == 2]['age'])
print(f"r={r}, p={p}")

'''
    Effect of noticing a sequence
'''
print("===========================")
print("Noticed a sequence effects")

t,p = scipy.stats.ttest_ind(complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 1)]['age'],
                                        complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 0)]['age'])
dof = len(complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 1)]['age']) + len(complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 0)]['age']) -2
print(f"Age: t={t}, p={p}, dof = {dof}")
del dof

t,p = scipy.stats.ttest_ind(complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 1)]['hpcf_rand'],
                           complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 0)]['hpcf_rand'])

dof = len(complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 1)]['hpcf_rand']) + len(complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 0)]['hpcf_rand']) -2
print(f"GD(Rand): t={t}, p={p}, dof = {dof}")
del dof

t,p = scipy.stats.ttest_ind(complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 1)]['CRspread'],
                           complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 0)]['CRspread'])
dof = len(complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 1)]['CRspread']) + len(complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 0)]['CRspread']) -2
print(f"CRspread: t={t}, p={p}, dof = {dof}")
del dof

t,p = scipy.stats.ttest_ind(complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 1)]['CIspread'],
                           complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 0)]['CIspread'])
dof = len(complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 1)]['CRspread']) + len(complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 0)]['CRspread']) - 2
print(f"CIspread: t={t}, p={p}, dof = {dof}")

t,p = scipy.stats.ttest_ind(complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 1)]['RIspread'],
                           complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 0)]['RIspread'])
dof = len(complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 1)]['RIspread']) + len(complete_df_all[(complete_df_all['day'] == 2) & (complete_df_all['q_notice_a_sequence'] == 0)]['RIspread']) -2
print(f"RIspread: t={t}, p={p}, dof = {dof}")
del dof

# t,p = scipy.stats.ttest_ind(dfadapt_day2[(dfadapt_day2['day'] == 2) & (dfadapt_day2['q_notice_a_sequence'] == 1)]['exploit_score'],
#                            dfadapt_day2[(dfadapt_day2['day'] == 2) & (dfadapt_day2['q_notice_a_sequence'] == 0)]['exploit_score'])
# dof = len(dfadapt_day2[(dfadapt_day2['day'] == 2) & (dfadapt_day2['q_notice_a_sequence'] == 1)]['exploit_score']) + len(dfadapt_day2[(dfadapt_day2['day'] == 2) & (dfadapt_day2['q_notice_a_sequence'] == 0)]['exploit_score']) -2
# print(f"t={t}, p={p}, dof={dof}")
# del dof

#%%
'''
    Simulate behaviour
'''

num_agents = 60

theta_rep_values = []
theta_conflict_values = []

for theta_rep_sim in [dfadapt_day2.theta_rep.min(), dfadapt_day2.theta_rep.max()]:
    for theta_conflict_sim in [dfadapt_day2.theta_conflict.min(), dfadapt_day2.theta_conflict.max()]:
        theta_rep_values.append(theta_rep_sim)
        theta_conflict_values.append(theta_conflict_sim)
        print(" \n\ntheta_rep_sim=%.2f, theta_conflict_sim=%.2f"%(theta_rep_sim, theta_conflict_sim))
        parameters = inf_mean_df_all[inf_mean_df_all['day'] == 2].loc[:, [*param_names]]
        parameters['theta_rep'] = theta_rep_sim
        parameters['theta_conflict'] = theta_conflict_sim
        
        groupdata_dict, sim_group_behav_df, params_sim_df, _ = utils.simulate_data(model, 
                                                                                num_agents,
                                                                                group = list(inf_mean_df_all['group'])[0:num_agents],
                                                                                day = 2,
                                                                                params = parameters,
                                                                                STT = False,
                                                                                Q_init = Q_init_day2,
                                                                                seq_init = seq_counter_day2,
                                                                                errorrates = er_day2)
        
        
        utils.plot_grouplevel(sim_group_behav_df, plot_single = False, day = 2)
        
        
        # utils.plot_grouplevel(sim_group_behav_df, expdata_df_all[(expdata_df_all['day']==2) & 
        #                                                          (expdata_df_all['ID']==2)], plot_single = False, day = 2)
        
        # utils.plot_grouplevel(expdata_df_all, expdata_df_all, plot_single = False, day = 2)
   
extreme_values_df = pd.DataFrame({'theta_rep': theta_rep_values, 'theta_conflict': theta_conflict_values})

#%%
'''
    Plot behaviour of IDs, and simulate model behaviour
'''

# plot_IDs = ['5b5e0e86902ad10001cfcc59', '5c321ebf6558270001bd79aa', 
#             '60f816ff1fa74fcfab532378', '5db4ef4a2986a3000be1f886', 
#             '56f699e876348f000c883bba', '654abe303c4940ec0502538e', 
#             '596f961cfe061d00011e3e03', '629f6b8c65fcae219e245284',
#             '654abe303c4940ec0502538e']

plot_IDs = ['595e7974af78da0001a21c3a', '654abe303c4940ec0502538e',
            '5db32244dbe39d000be72fb0', '5db4ef4a2986a3000be1f886',
            '59dd90f6e75b450001a68dac', '6329b1add3dcd53cb9c9cab8',
            '615739949cf5767509a7e29a']

# utils.plot_grouplevel(expdata_df_all[expdata_df_all['ID'].isin(plot_IDs)], plot_single = True, day = 2)
num_agents = 60
for ID in plot_IDs:
# ID = plot_IDs[0]
    parameters = inf_mean_df_all[(inf_mean_df_all['day'] == 2) & (inf_mean_df_all['ID'] == ID)].loc[:, [*param_names]]
    parameters = parameters.iloc[np.repeat(0, num_agents)]
    
    groupdata_dict, sim_group_behav_df, params_sim_df, _ = utils.simulate_data(model, 
                                                                                num_agents,
                                                                                group = list(inf_mean_df_all['group'])[0:num_agents],
                                                                                day = 2,
                                                                                params = parameters,
                                                                                STT = False,
                                                                                Q_init = Q_init_day2,
                                                                                seq_init = seq_counter_day2,
                                                                                errorrates = er_day2)
    
    print(ID)
    utils.plot_grouplevel(sim_group_behav_df, expdata_df_all[(expdata_df_all['day'] == 2) 
                                                             & (expdata_df_all['ID'] == ID)], plot_single = False, day = 2)
    
#%%
'''
    Scatterplot of posterior means, plus extreme values as black dots
'''

fig, ax = plt.subplots()
sns.scatterplot(dfadapt_day2,
                x = f'{param1}',
                y = f'{param2}',
                # palette = ['r', 'b'],
                # hue = 'ri_0',
                # hue_order = ['yes', 'no'],
                # palette = {'yes': '#d62728', 'no': '#1f77b4'},
                ax = ax)
custom_labels = ["no negative effect of habit", "negative effect of habit"]  # Define your custom labels here
handles, labels = ax.get_legend_handles_labels()
# ax[0].plot(np.arange(0.5,2),np.arange(0.5,2)-0.8, color='k')
ax.legend(handles, custom_labels, fontsize = 13, loc='upper left', bbox_to_anchor=(1, 1))
ax.set_xlabel(r'$\theta_{Rep}$', fontsize = 20)
ax.set_ylabel(r'$\theta_{Switch}$', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


sns.scatterplot(extreme_values_df,
                x = f'{param1}',
                y = f'{param2}',
                # palette = ['r', 'b'],
                # hue = 'ri_0',
                # hue_order = ['yes', 'no'],
                # palette = {'yes': '#d62728', 'no': '#1f77b4'},
                color = 'k',
                ax = ax)
custom_labels = ["no negative effect of habit", "negative effect of habit"]  # Define your custom labels here
handles, labels = ax.get_legend_handles_labels()
# ax[0].plot(np.arange(0.5,2),np.arange(0.5,2)-0.8, color='k')
ax.legend(handles, custom_labels, fontsize = 13, loc='upper left', bbox_to_anchor=(1, 1))
ax.set_xlabel(r'$\theta_{Rep}$', fontsize = 20)
ax.set_ylabel(r'$\theta_{Switch}$', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.savefig('/home/sascha/Desktop/Paper_2024/May/res_fig3/inferred_thetas_python.svg')
plt.show()

#%%

'''
    Model Comparison
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import pickle
import arviz as az

import scipy
import itertools

sns.set()

errors_day1_waic = [187, 187, 187, 186, 187, 187, 186, 187]
# errors_day1_dic = [210, 190, 180, 170, 160, 150, 150, 140]
errors_day2_waic = [221, 221, 222, 224, 224, 222, 223, 223]
# errors_day2_dic = [310, 290, 280, 270, 260, 250, 150, 140]

df_day1 = pd.DataFrame({'WAIC': {0: 19987, 1: 19982, 2: 19939, 3: 20550, 4: 20551, 5: 20306, 6: 20316, 7: 20345},
                      'DIC': {0: 19997, 1: 19988, 2: 19946, 3: 20559, 4: 20561, 5: 20316, 6: 20324, 7: 20347},
                      'Model_Names': {0: u'M 1',1: u'M 2',2: u'M 3',3: u'M 4.A', 
                                      4: u'M 4.B', 5: u'M 5.A', 6: u'M 5.B', 7: u'M 6'}})

df_day2 = pd.DataFrame({'WAIC': {0: 23916, 1: 23815, 2: 23742, 3: 24960, 4: 24958, 5: 24391, 6: 24400, 7: 24428},
                      'DIC': {0: 23909, 1: 23801, 2: 23729, 3: 24956, 4: 24953, 5: 24384, 6: 24393, 7: 24420},
                      'Model_Names': {0: u'M 1',1: u'M 2',2: u'M 3',3: u'M 4.A', 
                                      4: u'M 4.B', 5: u'M 5.A', 6: u'M 5.B', 7: u'M 6'}})

df_day1 = df_day1.set_index('Model_Names')
df_day2 = df_day2.set_index('Model_Names')

fig, ax = plt.subplots(1, 2, figsize=(12, 5)) # Create matplotlib figure

# ax = fig.add_subplot(111) # Create matplotlib axes
# ax2 = ax.twinx() # Create another axes that shares the same x-axis as a
width = .3

df_day1.WAIC.plot(kind='bar', color='deepskyblue',ax=ax[0], width=width, position=0, yerr=errors_day1_waic)
df_day1.DIC.plot(kind='bar', color='green', ax=ax[0], width = width, position=1)
ax[0].axvline(x = 2.5, color = 'k', label = 'axvline - full height')

df_day2.WAIC.plot(kind='bar', color='deepskyblue', ax=ax[1], width=width, position=0, yerr=errors_day2_waic)
df_day2.DIC.plot(kind='bar', color='green', ax=ax[1], width = width,position=1)
ax[1].axvline(x = 2.5, color = 'k', label = 'axvline - full height')

for i, label in enumerate(ax[0].get_xticklabels()):
    if label.get_text() in ['M 2', 'M 3', 'M 5.A', 'M 5.B', 'M 6']:
        label.set_color('red')

for i, label in enumerate(ax[1].get_xticklabels()):
    if label.get_text() in ['M 2', 'M 3', 'M 5.A', 'M 5.B', 'M 6']:
        label.set_color('red')


ax[0].set_ylabel('IC', fontsize = 20)
ax[0].set_xlabel('Model', fontsize = 20)
ax[1].set_xlabel('Model', fontsize = 20)

ax[0].set_xlim(-1, 8)
ax[0].set_ylim(19_600, 21_300)

ax[1].set_xlim(-1, 8)
ax[1].set_ylim(23_000, 26_500)

ax[0].legend(handles=ax[0].get_legend_handles_labels()[0][1:3], labels=['WAIC', 'DIC'], fontsize = 16)
ax[1].legend(handles=ax[1].get_legend_handles_labels()[0][1:3], labels=['WAIC', 'DIC'], fontsize = 16)

ax[0].text(-1, 21200, 'repetition bias', fontsize = 15)
ax[0].text(3, 21200, 'No repetition bias', fontsize = 15)

ax[1].text(-1, 26200, 'repetition bias', fontsize = 15)
ax[1].text(3, 26200, 'No repetition bias', fontsize = 15)

plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Poster_FENS_2024/MC_python.svg')

plt.show()


#%%
'''
    Points Analysis
'''

'''
    Day 2
'''
"Predictors x"
x = np.stack((np.array(complete_df_all[complete_df_all['day']==2]['theta_Q']), 
              np.array(complete_df_all[complete_df_all['day']==2]['theta_rep'])), axis=1)

"Data y"
y = np.array(complete_df_all[complete_df_all['day']==2]['points']).reshape(-1,1)
linmodel = LinearRegression()
linmodel.fit(x, y)
# print(f"slope: {linmodel.coef_}\n")


'''
    θ_Q
'''
r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_Q'], 
                            complete_df_all[complete_df_all['day'] == 2]['points'])
print("θ_Q vs points r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_Q'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_congruent'])
print("θ_Q vs points(Cong) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_Q'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_incongruent'])
print("θ_Q vs points(Incong) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_Q'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_dtt'])
print("θ_Q vs points(DTT) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_Q'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_dtt_rand'])
print("θ_Q vs points(DTT Rand) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_Q'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_dtt_seq'])
print("θ_Q vs points(DTT Rep) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_Q'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_stt'])
print("θ_Q vs points(STT) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_Q'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_stt_rand'])
print("θ_Q vs points(STT Rand) r=%.2f, p=%.4f"%(r,p))


r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_Q'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_stt_seq'])
print("θ_Q vs points(STT Rep) r=%.2f, p=%.4f\n\n"%(r,p))


'''
    θ_Rep
'''
r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rep'], 
                            complete_df_all[complete_df_all['day'] == 2]['points'])
print("θ_Rep vs points r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rep'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_congruent'])
print("θ_Rep vs points(Cong) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rep'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_incongruent'])
print("θ_Rep vs points(Incong) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rep'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_dtt'])
print("θ_Rep vs points(DTT) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rep'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_dtt_rand'])
print("θ_Rep vs points(DTT Rand) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rep'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_dtt_seq'])
print("θ_Rep vs points(DTT Rep) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rep'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_stt'])
print("θ_Rep vs points(STT) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rep'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_stt_rand'])
print("θ_Rep vs points(STT Rand) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rep'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_stt_seq'])
print("θ_Rep vs points(STT Rep) r=%.2f, p=%.4f\n\n"%(r,p))

'''
    θ_Switch
'''
r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_conflict'], 
                            complete_df_all[complete_df_all['day'] == 2]['points'])
print("θ_Switch vs points r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_conflict'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_congruent'])
print("θ_Switch vs points(Cong) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_conflict'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_incongruent'])
print("θ_Switch vs points(Incong) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_conflict'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_dtt'])
print("θ_Switch vs points(DTT) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_conflict'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_dtt_rand'])
print("θ_Switch vs points(DTT Rand) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_conflict'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_dtt_seq'])
print("θ_Switch vs points(DTT Rep) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_conflict'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_stt'])
print("θ_Switch vs points(STT) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_conflict'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_stt_rand'])
print("θ_Switch vs points(STT Rand) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_conflict'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_stt_seq'])
print("θ_Switch vs points(STT Rep) r=%.2f, p=%.4f\n\n"%(r,p))

'''
    θ_Rep - θ_Switch
'''
complete_df_all['theta_rpc'] = complete_df_all.apply(lambda row: row['theta_rep']-row['theta_conflict'], axis = 1)

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points'])
print("θ_Rep - θ_Switch vs points r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_congruent'])
print("θ_Rep - θ_Switch vs points(Cong) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_incongruent'])
print("θ_Rep - θ_Switch vs points(Incong) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_dtt'])
print("θ_Rep - θ_Switch vs points(DTT) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_dtt_rand'])
print("θ_Rep - θ_Switch vs points(DTT Rand) r=%.2f, p=%.4f"%(r,p))


r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_dtt_seq'])
print("θ_Rep - θ_Switch vs points(DTT Rep) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_stt'])
print("θ_Rep - θ_Switch vs points(STT) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_stt_rand'])
print("θ_Rep - θ_Switch vs points(STT Rand) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_stt_seq'])
print("θ_Rep - θ_Switch vs points(STT Rep) r=%.2f, p=%.4f\n\n"%(r,p))


'''
    θ_Rep + θ_Switch
'''
complete_df_all['theta_rpc'] = complete_df_all.apply(lambda row: row['theta_rep']+row['theta_conflict'], axis = 1)

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points'])
print("θ_Rep + θ_Switch vs points r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_congruent'])
print("θ_Rep + θ_Switch vs points(Cong) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_incongruent'])
print("θ_Rep + θ_Switch vs points(Incong) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_dtt'])
print("θ_Rep + θ_Switch vs points(DTT) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_dtt_rand'])
print("θ_Rep + θ_Switch vs points(DTT Rand) r=%.2f, p=%.4f"%(r,p))


r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_dtt_seq'])
print("θ_Rep + θ_Switch vs points(DTT Rep) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_stt'])
print("θ_Rep + θ_Switch vs points(STT) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_stt_rand'])
print("θ_Rep + θ_Switch vs points(STT Rand) r=%.2f, p=%.4f"%(r,p))

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['theta_rpc'], 
                            complete_df_all[complete_df_all['day'] == 2]['points_stt_seq'])
print("θ_Rep + θ_Switch vs points(STT Rep) r=%.2f, p=%.4f"%(r,p))

#%%
'''
    Correlate with DDM results
'''

DDM_res = pd.read_csv('DDM_model_params_conflictmodel_May_29_2024.csv')

DDM_res['phi_learnrate_day1'] = scipy.stats.norm(0, 1).pdf(DDM_res['phi_lernrate_day1'])
DDM_res['phi_learnrate_day2'] = scipy.stats.norm(0, 1).pdf(DDM_res['phi_learnrate_day2'])

DDM_params = ['drift_theta_Q_day2', 'drift_theta_rep_day2', 'drift_theta_switch_day2',
              'bias_theta_rep_congruent_day2', 'bias_theta_rep_incongruent_day2',
              'bias_baseline_day2', 'non_decision_time_day2', 'phi_learnrate_day2']

model_params = []

num_comparisons = len(DDM_params)*len(param_names)
print("------ Day 2")
for param1 in DDM_params:
    for param2 in param_names:
        r,p = scipy.stats.spearmanr(DDM_res[param1], inf_mean_df_all[inf_mean_df_all['day']==2][param2])
        
        if p < 0.05:
            print(f"{param1} vs {param2}: r=%.2f, p=%.4f"%(r,p))
            
plt.figure()
plt.scatter(DDM_res['phi_learnrate_day1'], inf_mean_df_all[inf_mean_df_all['day']==1]['lr'])
plt.show()
            
plt.figure()
plt.scatter(DDM_res['phi_learnrate_day2'], inf_mean_df_all[inf_mean_df_all['day']==2]['lr'])
plt.show()

#%%
'''
    Supplementary Material
'''

'''
    Fig for paper
    Individual posteriors
'''
num_params = len(param_names)
num_agents = len(complete_df_all['ID'].unique())

fig, ax = plt.subplots(num_params, 2, figsize = (15, 19))

for param in range(num_params):
    for day in range(1, 3): 
        for ag in range(num_agents):
            if param == 0:
                sns.kdeplot(post_sample_df_all[(post_sample_df_all['ag_idx']==ag) & 
                                               (post_sample_df_all['day']==day)][param_names[param]], 
                            ax = ax[param, day-1], clip=(0,1))
                
            else:
                sns.kdeplot(post_sample_df_all[(post_sample_df_all['ag_idx']==ag) & (post_sample_df_all['day']==day)][param_names[param]], 
                            ax = ax[param, day-1])
                
            # if param == 4:
            #     dfgh
            ax[0, day-1].set_xlabel(r'learning rate', fontsize = 20)
            ax[1, day-1].set_xlabel(r'$\Theta_Q$', fontsize = 20)
            ax[2, day-1].set_xlabel(r'$\Theta_{Rep}$', fontsize = 20)
            ax[3, day-1].set_xlabel(r'$\Theta_\text{Switch}$', fontsize = 20)
            ax[0, day-1].set_ylabel('')
            ax[1, day-1].set_ylabel('')
            ax[2, day-1].set_ylabel('')
            ax[3, day-1].set_ylabel('')
            # if param == 0:
                # 'lr'
            ax[0, day-1].set_xlim([-0.025, 0.2])
            ax[1, day-1].set_xlim([-1, 8])
            ax[2, day-1].set_xlim([-1, 5])
            ax[3, day-1].set_xlim([-2, 4])
            
            ax[0, day-1].set_ylim([0, 80])
            ax[1, day-1].set_ylim([0, 20])
            ax[2, day-1].set_ylim([0, 20])
            ax[3, day-1].set_ylim([0, 20])

plt.savefig('/home/sascha/Desktop/Paper_2024/May/suppl_fig1/posteriors_python.svg')
plt.show()

#%%
'''
    Differences day 1 & day 2
'''
inf_mean_df_all['Q/R'] = inf_mean_df_all.apply(lambda row: row['theta_Q']/row['theta_rep'], axis = 1)

post_sample_df_all['Q/R'] = post_sample_df_all.apply(lambda row: row['theta_Q']/row['theta_rep'], axis = 1)
param_names = list(param_names)
param_names.append('Q/R')

utils.lineplot_daydiffs(pd.melt(inf_mean_df_all, 
                                id_vars=['ag_idx', 'day'], 
                                value_vars = ['lr', 'theta_Q', 'theta_rep', 'theta_conflict'], 
                                var_name = 'parameter', value_name ='mean'))


#%%
'''
    Debriefing Questionnaire
'''

num_never_easier=(expdata_df_all[expdata_df_all['trialidx']==0]['q_sometimes_easier'] == 0).astype(int).sum()
num_sometimes_easier=(expdata_df_all[expdata_df_all['trialidx']==0]['q_sometimes_easier'] == 1).astype(int).sum()
num_easier_dunno=(expdata_df_all[expdata_df_all['trialidx']==0]['q_sometimes_easier'] == 2).astype(int).sum()

num_not_noticed_seq = (complete_df_all[complete_df_all['day']==2]['q_notice_a_sequence'] == 0).astype(int).sum()
num_noticed_seq = (complete_df_all[complete_df_all['day']==2]['q_notice_a_sequence'] == 1).astype(int).sum()
num_seqnoticed_dunno = (complete_df_all[complete_df_all['day']==2]['q_notice_a_sequence'] == 2).astype(int).sum()

utils.check_debriefing_quest(expdata_df_all)

num_male = (complete_df_all[complete_df_all['day']==2]['gender']=='male').astype(int).sum()
num_female = (complete_df_all[complete_df_all['day']==2]['gender']=='female').astype(int).sum()


#%%
'''
    Compute Bayes Factors
    BF = p(y_A)/p(y_B) = exp[log p(y_A) - log p(y_B)] = exp[elbo_A - elbo_B]
    
    BF > 1 indicate a preference for model model_names[compidx]
'''

elbos = np.zeros((num_agents, 2))

for midx in range(2):
    _, _, _, _, _, _, _, _, _, _, _, agent_elbo_tuple_day2, agent_elbo_tuple_day1 = utils.load_data()
    
    elbo_combined = -agent_elbo_tuple_day2[0] -agent_elbo_tuple_day1[0]
    
    elbos[:, model] = elbo_combined

compidx = 0

num_comparisons = num_models - 1

BF = np.zeros((num_comparisons, num_agents))
BF_2nd_lvl = np.zeros(num_comparisons)

compnumb = 0
for i in range(num_models):
    if i != compidx:
        BF[compnumb, :] = np.exp(elbos[:, compidx] - elbos[:, i])
        BF_2nd_lvl[compnumb] = np.exp(elbos_2nd_lvl[compidx] - elbos_2nd_lvl[i])
        # BF[compnumb, :] = np.exp(elbos[:, compidx]) / np.exp(elbos[:, i])
        compnumb += 1
        
fig, ax = plt.subplots()
sns.histplot(BF[0,:])
plt.legend([],[], frameon=False)
plt.show()

fig, ax = plt.subplots()
sns.histplot(1/BF[0,:])
plt.legend([],[], frameon=False)
plt.show()