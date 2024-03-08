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
import itertools

complete_df_all, expdata_df_all, post_sample_df_all, sim_df, param_names, _, _, _ = utils.load_data()
model = complete_df_all['model'].unique()[0]
hue_order = ['Random', 'Congruent', 'Incongruent']

if 'theta_rep' in param_names:
    param1 = 'theta_rep'
    param2 = 'theta_conflict'
    
elif 'theta_repcong' in param_names:
    param1 = 'theta_repcong'
    param2 = 'theta_repinc'
    
#%%
'''
    ER ~ Condition, Day
'''

ER_all = complete_df_all.loc[:, ['ID',
                                 'day',
                                 'ER_stt_rand', 
                                   'ER_stt_seq']]

ER_all = ER_all.melt(id_vars=['ID', 'day'], value_vars=['ER_stt_rand', 'ER_stt_seq'])
ER_all['variable'] = ER_all['variable'].map(lambda x: "Repeating" if x == 'ER_stt_seq' else
                                                'Random' if x == 'ER_stt_rand' else
                                                'None')
ER_all = ER_all.rename(columns={'variable': 'Condition',
                                'value': 'ER'})
import pingouin as pg
aov = pg.rm_anova(dv = 'ER', 
                  within = ['Condition', 'day'], 
                  subject = 'ID', 
                  data = ER_all, 
                  detailed = True,
                  effsize = 'np2')
print(aov.loc[:, ['Source', 'ddof1', 'ddof2', 'F', 'p-unc', 'np2']])


t,p = scipy.stats.ttest_rel(ER_all[(ER_all['day'] == 1) & (ER_all['Condition'] == 'Random')]['ER'], 
                            ER_all[(ER_all['day'] == 1) & (ER_all['Condition'] == 'Repeating')]['ER'])
print(f"t={t}, p={p}")


t,p = scipy.stats.ttest_rel(ER_all[(ER_all['day'] == 2) & (ER_all['Condition'] == 'Random')]['ER'], 
                            ER_all[(ER_all['day'] == 2) & (ER_all['Condition'] == 'Repeating')]['ER'])
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

t,p = scipy.stats.ttest_rel(RT_all[(RT_all['day'] == 1) & (RT_all['Condition'] == 'Random')]['RT'], 
                            RT_all[(RT_all['day'] == 1) & (RT_all['Condition'] == 'Repeating')]['RT'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(RT_all[(RT_all['day'] == 2) & (RT_all['Condition'] == 'Random')]['RT'], 
                            RT_all[(RT_all['day'] == 2) & (RT_all['Condition'] == 'Repeating')]['RT'])
print(f"t={t}, p={p}")

#%%

'''
    Results Figure 1
    Left: HRC Day 1 & Day 2
    Middle: ER Day 1 & Day 2
    Right: RT Day 1 & Day 2
'''

colors1 = {'Random': '#67b798', 'Congruent': '#BE54C6', 'Incongruent': '#7454C7'}
colors2 = {'Random': '#67b798', 'Repeating': '#bd97c6'} # random, fix

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

ER_all['day'] = ER_all['day'].astype(int)
ER_all['ER'] = ER_all['ER']*100
sns.barplot(data = ER_all,
            x = 'day',
            y = 'ER',
            hue = 'Condition',
            palette = colors2,
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
            palette = colors2,
            errorbar=('se', 1),
            ax = ax[2])

ax[2].legend(title="", fontsize = 18)
ax[2].set_ylabel("RT STT (ms)", fontsize = 20)
ax[2].set_ylim([320, 420])
ax[2].set_xlabel("Day", fontsize = 20)
ax[2].tick_params(axis='both', labelsize=18)
plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig0/res_fig0_python.svg', bbox_inches = 'tight')
plt.show()

t,p = scipy.stats.ttest_rel(hpcf_all[(hpcf_all['day'] == 1) & (hpcf_all['DTT Type'] == 'Random')]['HRC'], 
                            hpcf_all[(hpcf_all['day'] == 1) & (hpcf_all['DTT Type'] == 'Congruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hpcf_all[(hpcf_all['day'] == 1) & (hpcf_all['DTT Type'] == 'Random')]['HRC'], 
                            hpcf_all[(hpcf_all['day'] == 1) & (hpcf_all['DTT Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hpcf_all[(hpcf_all['day'] == 1) & (hpcf_all['DTT Type'] == 'Congruent')]['HRC'], 
                            hpcf_all[(hpcf_all['day'] == 1) & (hpcf_all['DTT Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

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

#%%
'''
    Corr RT ~ ΔCI
'''
r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day']==1]['RT_diff_stt'], complete_df_all[complete_df_all['day']==1]['CIspread'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day']==2]['RT_diff_stt'], complete_df_all[complete_df_all['day']==2]['CIspread'])
print(f"r={r}, p={p}")

"ΔRT Day 1 vs Day 2"
t,p = scipy.stats.ttest_rel(complete_df_all[complete_df_all['day']==1]['RT_diff_stt'], complete_df_all[complete_df_all['day']==2]['RT_diff_stt'])
print(f"r={t}, p={p}")

t,p = scipy.stats.ttest_rel(complete_df_all[complete_df_all['day']==1]['CIspread'], complete_df_all[complete_df_all['day']==2]['CIspread'])
print(f"r={t}, p={p}")

'''
    Corr ER ~ ΔCI
'''
r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day']==1]['ER_diff_stt'], complete_df_all[complete_df_all['day']==1]['CIspread'])
print(f"r={r}, p={p}")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day']==2]['ER_diff_stt'], complete_df_all[complete_df_all['day']==2]['CIspread'])
print(f"r={r}, p={p}")

"ΔER Day 1 vs Day 2"
t,p = scipy.stats.ttest_rel(complete_df_all[complete_df_all['day']==1]['ER_diff_stt'], complete_df_all[complete_df_all['day']==2]['ER_diff_stt'])
print(f"r={t}, p={p}")


#%%
'''
    Find weak and strong sequence learners.    

    Response Strategies
    Habitual Responders
    GD Responders
    Modulators
'''

expdata_df_day1 = expdata_df_all[expdata_df_all['day'] == 1]
expdata_df_day2 = expdata_df_all[expdata_df_all['day'] == 2]

seqlearners_df_day1, notseqlearners_df_day1, seqlearn_df_day1 = anal.find_seqlearners(expdata_df_day1,
                                                          day = 1,
                                                          correctp = True)


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

"===== Strong learners"
t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Random')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Congruent')]['choices_GD'])
print(f"Rand-Cong, t={t}, p={p}")

t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Congruent')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Incongruent')]['choices_GD'])
print(f"Cong-Inc, t={t}, p={p}")

t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Random')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Incongruent')]['choices_GD'])
print(f"Rand-Inc, t={t}, p={p}")

"===== Strong learners"
t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Random')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Congruent')]['choices_GD'])
print(f"Rand-Cong, t={t}, p={p}")

t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Congruent')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Incongruent')]['choices_GD'])
print(f"Cong-Inc, t={t}, p={p}")

t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Random')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 1) & (seq_combined['type'] == 'weak sequence learner') & (seq_combined['DTT Types'] == 'Incongruent')]['choices_GD'])
print(f"Rand-Inc, t={t}, p={p}")


"===== Strong learners"
t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Random')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Congruent')]['choices_GD'])
print(f"Rand-Cong, t={t}, p={p}")

t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Congruent')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Incongruent')]['choices_GD'])
print(f"Cong-Inc, t={t}, p={p}")

t,p = scipy.stats.ttest_rel(seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Random')]['choices_GD'], 
                            seq_combined[(seq_combined['day'] == 2) & (seq_combined['type'] == 'strong sequence learner') & (seq_combined['DTT Types'] == 'Incongruent')]['choices_GD'])
print(f"Rand-Inc, t={t}, p={p}")

"===== Strong learners"
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

IDs, ps_cr, ps_ri, ps_ci, _, _, _ = anal.hpcf_within(df_day2)
df_ps_day2 = pd.DataFrame({'ID':IDs, 'ps_cr':ps_cr, 'ps_ri':ps_ri, 'ps_ci': ps_ci})
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
    hab_mask_day1 = hrc_df_day2['ps_ri'] < 0.05
    GD_mask_day1 = hrc_df_day2['ps_ri'] > 0.05

    hab_mask_day2 = hrc_df_day2['ps_ri'] < 0.05
    GD_mask_day2 = hrc_df_day2['ps_ri'] > 0.05

"------"

fig, ax = plt.subplots(1,2, sharey=True)
sns.barplot(ax = ax[0],
            data = hrc_df_day2[hab_mask_day2],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[0].set_ylim([60, 100])
ax[0].set_title("Negative effect of habit")
ax[0].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)
custom_labels = ["Random", "Congruent", "Incongruent"]  # Define your custom labels here
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles, custom_labels, fontsize = 12, loc='upper left', bbox_to_anchor=(1, 1))

" ==== HRC Differences within Group?"
t,p = scipy.stats.ttest_rel(hrc_df_day2[(hrc_df_day2['ps_ri'] < 0.05) & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[(hrc_df_day2['ps_ri'] < 0.05) & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day2[(hrc_df_day2['ps_ri'] < 0.05) & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[(hrc_df_day2['ps_ri'] < 0.05) & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day2[(hrc_df_day2['ps_ri'] < 0.05) & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_day2[(hrc_df_day2['ps_ri'] < 0.05) & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

sns.barplot(ax = ax[1],
            data = hrc_df_day2[GD_mask_day2],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[1].set_ylim([60, 100])
ax[1].set_ylabel("Goal-Directed Responses (%)", fontsize = 20)
ax[1].set_title("No negative effect of habit")
plt.savefig("/home/sascha/Desktop/Paper_2024/Mar/res_fig3/hab_vs_GD_python.svg")
plt.show()

" ==== HRC Differences within Group?"
t,p = scipy.stats.ttest_rel(hrc_df_day2[(hrc_df_day2['ps_ri'] > 0.05) & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[(hrc_df_day2['ps_ri'] > 0.05) & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day2[(hrc_df_day2['ps_ri'] > 0.05) & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[(hrc_df_day2['ps_ri'] > 0.05) & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day2[(hrc_df_day2['ps_ri'] > 0.05) & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_day2[(hrc_df_day2['ps_ri'] > 0.05) & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

" ==== HRC Differences between Groups?"
t,p = scipy.stats.ttest_ind(hrc_df_day2[(hrc_df_day2['ps_ri'] > 0.05) & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[(hrc_df_day2['ps_ri'] < 0.05) & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'])
print(f"t={t}, p={p}")


print(f"There are {len(hrc_df_day2[hab_mask_day2]['ID'].unique())} participants in the 'habitual' group")
print(f"There are {len(hrc_df_day2[GD_mask_day2]['ID'].unique())}  participants in the 'GD' group")

#%%
'''
    Exploiters vs Suppressers Day 1
'''

exploit_mask = GD_mask_day1 & (hrc_df_day1['ps_cr'] < 0.05)
suppress_mask = GD_mask_day1 & (hrc_df_day1['ps_cr'] > 0.05)
    
fig, ax = plt.subplots(2,2, sharey=True, figsize = (8,8))

sns.barplot(ax = ax[0,0],
            data = hrc_df_day1[hab_mask_day1],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
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
ax[0,1].set_ylim([60, 100])
ax[0,1].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)
ax[0,1].set_title("No negative effect of habit")

sns.barplot(ax = ax[1,0],
            data = hrc_df_day1[exploit_mask],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[1,0].set_ylim([60, 100])
ax[1,0].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)
ax[1,0].set_title('"Adapters"')
ax[1,0].tick_params(axis='both', labelsize=14)
# plt.savefig("/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/hrc_python.svg")
# plt.title(f"Habedahabeda 2 (N={len(hrc_df_day1[hrc_df_day1['ps_ri'] < 0.05])/3})")

"===== Differences within group"
t,p = scipy.stats.ttest_rel(hrc_df_day1[exploit_mask & (hrc_df_day1['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day1[exploit_mask & (hrc_df_day1['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day1[exploit_mask & (hrc_df_day1['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day1[exploit_mask & (hrc_df_day1['Trial Type'] == 'Congruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day1[exploit_mask & (hrc_df_day1['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_day1[exploit_mask & (hrc_df_day1['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

sns.barplot(ax = ax[1,1],
            data = hrc_df_day1[suppress_mask],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[1,1].set_ylim([60, 100])
ax[1,1].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)
ax[1,1].set_title('"Inhibitors"')
ax[1,1].get_legend().set_visible(False)
ax[1,0].get_legend().set_visible(False)
ax[0,1].get_legend().set_visible(False)
# plt.savefig("/home/sascha/Desktop/Paper_2024/Mar/res_fig3/res_fig3_python.svg")
plt.show()

"===== Differences within group"
t,p = scipy.stats.ttest_rel(hrc_df_day1[suppress_mask & (hrc_df_day1['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day1[suppress_mask & (hrc_df_day1['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day1[suppress_mask & (hrc_df_day1['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day1[suppress_mask & (hrc_df_day1['Trial Type'] == 'Congruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day1[suppress_mask & (hrc_df_day1['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_day1[suppress_mask & (hrc_df_day1['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

"===== Differences between groups"
t,p = scipy.stats.ttest_ind(hrc_df_day1[suppress_mask & (hrc_df_day1['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day1[exploit_mask & (hrc_df_day1['Trial Type'] == 'Random')]['HRC'])
print(f"Random Type, between groups: t={t}, p={p}")

t,p = scipy.stats.ttest_ind(hrc_df_day1[suppress_mask & (hrc_df_day1['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_day1[exploit_mask & (hrc_df_day1['Trial Type'] == 'Congruent')]['HRC'])
print(f"Congruent Type, between groups: t={t}, p={p}")

t,p = scipy.stats.ttest_ind(hrc_df_day1[suppress_mask & (hrc_df_day1['Trial Type'] == 'Incongruent')]['HRC'], 
                            hrc_df_day1[exploit_mask & (hrc_df_day1['Trial Type'] == 'Incongruent')]['HRC'])
print(f"Incongruent Type, between groups: t={t}, p={p}")

print(f"There are {len(hrc_df_day1[exploit_mask]['ID'].unique())} participants in the exploit group")
print(f"There are {len(hrc_df_day1[suppress_mask]['ID'].unique())} participants in the suppress group")

#%%
'''
    Exploiters vs Suppressers Day 2
'''

exploit_mask = GD_mask_day2 & (hrc_df_day2['ps_cr'] < 0.05)
suppress_mask = GD_mask_day2 & (hrc_df_day2['ps_cr'] > 0.05)
    
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

sns.barplot(ax = ax[0,1],
            data = hrc_df_day2[GD_mask_day2],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[0,1].set_ylim([60, 100])
ax[0,1].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)
ax[0,1].set_title("No negative effect of habit")

sns.barplot(ax = ax[1,0],
            data = hrc_df_day2[exploit_mask],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[1,0].set_ylim([60, 100])
ax[1,0].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)
ax[1,0].set_title('"Adapters"')
ax[1,0].tick_params(axis='both', labelsize=14)
# plt.savefig("/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/hrc_python.svg")
# plt.title(f"Habedahabeda 2 (N={len(hrc_df_day2[hrc_df_day2['ps_ri'] < 0.05])/3})")

"===== Differences within group"
t,p = scipy.stats.ttest_rel(hrc_df_day2[exploit_mask & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[exploit_mask & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day2[exploit_mask & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[exploit_mask & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day2[exploit_mask & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_day2[exploit_mask & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

sns.barplot(ax = ax[1,1],
            data = hrc_df_day2[suppress_mask],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[1,1].set_ylim([60, 100])
ax[1,1].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)
ax[1,1].set_title('"Inhibitors"')
ax[1,1].get_legend().set_visible(False)
ax[1,0].get_legend().set_visible(False)
ax[0,1].get_legend().set_visible(False)
plt.savefig("/home/sascha/Desktop/Paper_2024/Mar/res_fig3/res_fig3_python.svg")
plt.show()

"===== Differences within group"
t,p = scipy.stats.ttest_rel(hrc_df_day2[suppress_mask & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[suppress_mask & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day2[suppress_mask & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[suppress_mask & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_day2[suppress_mask & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_day2[suppress_mask & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

"===== Differences between groups"
t,p = scipy.stats.ttest_ind(hrc_df_day2[suppress_mask & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_day2[exploit_mask & (hrc_df_day2['Trial Type'] == 'Random')]['HRC'])
print(f"Random Type, between groups: t={t}, p={p}")

t,p = scipy.stats.ttest_ind(hrc_df_day2[suppress_mask & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_day2[exploit_mask & (hrc_df_day2['Trial Type'] == 'Congruent')]['HRC'])
print(f"Congruent Type, between groups: t={t}, p={p}")

t,p = scipy.stats.ttest_ind(hrc_df_day2[suppress_mask & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'], 
                            hrc_df_day2[exploit_mask & (hrc_df_day2['Trial Type'] == 'Incongruent')]['HRC'])
print(f"Incongruent Type, between groups: t={t}, p={p}")

print(f"There are {len(hrc_df_day2[exploit_mask]['ID'].unique())} participants in the exploit group")
print(f"There are {len(hrc_df_day2[suppress_mask]['ID'].unique())} participants in the suppress group")

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['CRspread'], 
                           complete_df_all[complete_df_all['day'] == 2]['hpcf_rand'])

r,p = scipy.stats.pearsonr(complete_df_all[complete_df_all['day'] == 2]['RIspread'], 
                           complete_df_all[complete_df_all['day'] == 2]['hpcf_rand'])

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

df = pd.merge(complete_df_day2, seqlearn_df_day2, on ='ID')
df = pd.merge(df, ps_df_day2, on ='ID')
df['ri_0'] = df['ps_ri'].map(lambda x: 'yes' if x > 0.05 else 'no')

df['exploit_score'] = df.apply(lambda row: row[f'{param2}']/row[f'{param1}'], axis=1)

'''
    Fig
    Adaptation score
'''
fig, ax = plt.subplots(1,2, sharey=True, sharex = True, figsize = (8,4))
sns.scatterplot(df,
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
sns.scatterplot(df,
                x = f'{param1}',
                y = f'{param2}',
                # palette = ['r', 'b'],
                hue = 'exploit_score',
                ax = ax[1])
ax[1].legend(title='Adaptation Score', loc='upper left', bbox_to_anchor=(1, 1))
ax[1].get_legend().get_title().set_fontsize(14)
# plt.plot([0, 2.5], [0, 2.5], color='k', linewidth = 0.5)
# plt.plot([0, 2.5], [0, 1.1])
ax[1].set_xlim([0, 2.5])
ax[1].set_ylim([0, 2.5])
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
    Histogram of Habit Exploitation Score
'''
fig, ax = plt.subplots()
sns.histplot(data = df, x='exploit_score', bins = 9, binrange=[0.2, 1.5], ax = ax)
ax.set_xlabel('Adaptation Score')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title(f"{model}")
ax2 = ax.twinx()
sns.kdeplot(data = df, x='exploit_score', ax = ax2)
ax2.grid(False)
ax2.get_yaxis().set_visible(False)
plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig3/res_fig3c_python.svg', bbox_inches = 'tight')
plt.show()

#%%
'''
    2 Example participants, from 3 sections each
'''
IDs = ['60a3f8075b013de7b5518e96', '57deda2591b7fc0001493e95', 
       '5fb46dd5d9ece50422838e7a', '5d5a75c570a7c1000152623e', 
       '63174af7d57182f9bf90c094', '58aca85e0da7f10001de92d4', '596f961cfe061d00011e3e03']

# Exploit scores > 1 and hpcf_rand < 0.9
IDs = ['5eaadc0a7adeb404eea9c3c0', '5b5e0e86902ad10001cfcc59', '62c97799bd8ab72a531abde0']

for ID in df.sort_values(by='exploit_score')['ID'].unique():
    utils.plot_hpcf(complete_df_all[complete_df_all['ID'] == ID], title='score = %.2f, ID %s'%(df[df['ID'] == ID]['exploit_score'], ID))


#%%
'''
    Ideally, theta_Q does not correlate with the other model params.
'''
r,p = scipy.stats.pearsonr(df[f'{param1}'], df['theta_Q'])
print(f"{param1} vs theta_Q: r={r}, p={p}")

r,p = scipy.stats.pearsonr(df[f'{param2}'], df['theta_Q'])
print(f"{param2} vs theta_Q: r={r}, p={p}")

r,p = scipy.stats.pearsonr(df['exploit_score'], df['theta_Q'])
print(f"exploit_score vs theta_Q: r={r}, p={p}")

#%%
'''
    Does exploit score depend on group?
'''

expl_means = []
expl_means.append(df[df['group']==0]['exploit_score'].mean())
expl_means.append(df[df['group']==1]['exploit_score'].mean())
expl_means.append(df[df['group']==2]['exploit_score'].mean())
expl_means.append(df[df['group']==3]['exploit_score'].mean())


expl_stdevs = []
expl_stdevs.append(df[df['group']==0]['exploit_score'].std())
expl_stdevs.append(df[df['group']==1]['exploit_score'].std())
expl_stdevs.append(df[df['group']==2]['exploit_score'].std())
expl_stdevs.append(df[df['group']==3]['exploit_score'].std())

sns.barplot(data=df,
            y='exploit_score',
            x='group',
            errorbar=('se', 1))

import pingouin as pg
aov = pg.anova(dv = 'exploit_score',
                  between = ['group'], 
                  data = df, 
                  detailed = True,
                  effsize = 'np2')

# print(aov)
print(aov.loc[:, ['Source', 'F', 'p-unc', 'np2']])

'''
    Does exploit score depend on age?
'''
r,p = scipy.stats.pearsonr(df['age'], df['exploit_score'])
print(f"r={r}, p={p}")
fig, ax = plt.subplots()
sns.regplot(data=df,
           x='age',
           y='exploit_score')
ax.text(45, 1.4, "r=%.2f, p=%.2f"%(r,p))
plt.show()

'''
    Does exploit score depend on noticing a sequence?
'''
from scipy.stats import chi2_contingency
t,p = scipy.stats.ttest_ind(df[df['q_notice_a_sequence']==0]['exploit_score'], 
                            df[df['q_notice_a_sequence']==1]['exploit_score'])

_ = utils.plot_grouplevel(expdata_df_day2[expdata_df_day2['ID'].isin(df[df['q_notice_a_sequence']==0]['ID'].unique())], plot_single = False)
_ = utils.plot_grouplevel(expdata_df_day2[expdata_df_day2['ID'].isin(df[df['q_notice_a_sequence']==1]['ID'].unique())], plot_single = False)

#%%
'''
    HRC for upper and lower median of exploit_score.
'''

hrc_df_day2 = complete_df_all[complete_df_all['day'] == 2].loc[:, ['ID', 'hpcf_rand', 'hpcf_cong', 'hpcf_incong']]
hrc_df_day2 = pd.merge(hrc_df_day2, df.loc[:, ['ID', 'exploit_score']], on = 'ID')
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