#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Tue Nov  7 17:31:18 2023

    THIS. IS. WEIMAR!!!

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

complete_df_all, expdata_df_all, post_sample_df_all, sim_df, param_names, _, _, _  = utils.load_data()

#%%
'''
    Correlation plots
    Left: ΔCI vs ΔRT
    Right: ΔCI vs ΔER
'''

fig, ax = plt.subplots(1,2, sharey = True, figsize = (12, 6))

plot_df_all = complete_df_all.copy()
plot_df_all['CIspread'] = plot_df_all['CIspread']*100
plot_df_all['ER_diff_stt'] = plot_df_all['ER_diff_stt']*100
sns.regplot(x='RT_diff_stt',
                y='CIspread',
                # hue = 'CIspread',
                ax = ax[0],
                data = plot_df_all[plot_df_all['day'] == 2])

ax[0].axvline(0, color = 'k')
ax[0].axhline(0, color = 'k')
ax[0].legend(title='')
# ax[0].set_title('Day 1', fontsize = 15)
ax[0].tick_params(axis='x', labelsize=15)  # For x-axis tick labels
ax[0].tick_params(axis='y', labelsize=15)  # For x-axis tick labels
ax[0].set_xlabel('RT (Rand - Rep) (ms)', fontsize = 20)
ax[0].set_ylabel('Cong-Inc (%)', fontsize = 20)
# ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
r, p = scipy.stats.pearsonr(plot_df_all[plot_df_all['day'] == 2]['CIspread'], plot_df_all[plot_df_all['day'] == 2]['RT_diff_stt'])
ax[0].text(10, 40, "r=%.2f, p<0.001"%r, fontsize = 20)

sns.regplot(x='ER_diff_stt',
                y='CIspread',
                # hue = 'CIspread',
                ax = ax[1],
                data = plot_df_all[plot_df_all['day'] == 2])

ax[1].axvline(0, color = 'k')
ax[1].axhline(0, color = 'k')
ax[1].legend(title='')
# ax[1].set_title('Day 1', fontsize = 15)
ax[1].tick_params(axis='x', labelsize=15)  # For x-axis tick labels
ax[1].tick_params(axis='y', labelsize=15)  # For x-axis tick labels
ax[1].set_xlabel('ER (Rand - Rep) (%)', fontsize = 20)
ax[1].set_ylabel("", fontsize = 20)
# ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
r, p = scipy.stats.pearsonr(plot_df_all[plot_df_all['day'] == 2]['CIspread'], plot_df_all[plot_df_all['day'] == 2]['ER_diff_stt'])
ax[1].text(1, 40, "r=%.2f, p=%.3f"%(r,p), fontsize = 20)
plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/pos_to_instrumental_python.svg', bbox_inches = 'tight')
plt.show()


fig, ax = plt.subplots(1,2, sharey = True, figsize = (12, 6))

plot_df_all['RIspread'] = plot_df_all['RIspread']*100
sns.regplot(x='RT_diff_stt',
                y='RIspread',
                # hue = 'CIspread',
                ax = ax[0],
                data = plot_df_all[plot_df_all['day'] == 2])

ax[0].axvline(0, color = 'k')
ax[0].axhline(0, color = 'k')
ax[0].legend(title='')
# ax[0].set_title('Day 1', fontsize = 15)
ax[0].tick_params(axis='x', labelsize=15)  # For x-axis tick labels
ax[0].tick_params(axis='y', labelsize=15)  # For x-axis tick labels
ax[0].set_xlabel('RT (Rand - Rep) (ms)', fontsize = 20)
ax[0].set_ylabel('Rand-Inc (%)', fontsize = 20)
# ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
r, p = scipy.stats.pearsonr(plot_df_all[plot_df_all['day'] == 2]['RIspread'], plot_df_all[plot_df_all['day'] == 2]['RT_diff_stt'])
ax[0].text(10, -0.03, "r=%.2f, p<0.001"%r, fontsize = 20)

sns.regplot(x='ER_diff_stt',
                y='RIspread',
                # hue = 'CIspread',
                ax = ax[1],
                data = plot_df_all[plot_df_all['day'] == 2])

ax[1].axvline(0, color = 'k')
ax[1].axhline(0, color = 'k')
ax[1].legend(title='')
# ax[1].set_title('Day 1', fontsize = 15)
ax[1].tick_params(axis='x', labelsize=15)  # For x-axis tick labels
ax[1].tick_params(axis='y', labelsize=15)  # For x-axis tick labels
ax[1].set_xlabel('ER (Rand - Rep) (%)', fontsize = 20)
ax[1].set_ylabel("", fontsize = 20)
# ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
r, p = scipy.stats.pearsonr(plot_df_all[plot_df_all['day'] == 2]['RIspread'], plot_df_all[plot_df_all['day'] == 2]['ER_diff_stt'])
ax[1].text(0.009, -0.03, "r=%.2f, p=%.3f"%(r,p), fontsize = 20)
# plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/pos_to_instrumental_python.svg', bbox_inches = 'tight')
plt.show()

#%%
'''
    Weimar 2
    HRC by Trialtype pooled across days
'''
hrc_df_pooled = complete_df_all.loc[:, ['ID', 'hpcf_cong', 'hpcf_rand', 'hpcf_incong']].groupby('ID', as_index  = False).mean()
df = expdata_df_all.loc[:, ['ID', 
                        'trialsequence', 
                        'choices', 
                        'choices_GD', 
                        'jokertypes']]
df['day'] = 2.5
IDs, ps_cr, ps_ri, ps_ci, _, _, _ = anal.hpcf_within(df, correctp = False)
df_ps_pooled = pd.DataFrame({'ID':IDs, 'ps_cr':ps_cr, 'ps_ri':ps_ri, 'ps_ci': ps_ci})

hrc_df_pooled = hrc_df_pooled.rename(columns={"hpcf_cong": "Congruent", "hpcf_incong":"Incongruent", 'hpcf_rand':'Random'})

hrc_df_pooled = pd.melt(hrc_df_pooled, id_vars = ['ID'], value_vars=['Congruent', 'Incongruent', 'Random'])
hrc_df_pooled = hrc_df_pooled.rename(columns={"variable": "Trial Type", "value":"HRC"})

hrc_df_pooled = pd.merge(hrc_df_pooled, df_ps_pooled, on = 'ID')

colors1 = {'Random': '#67b798', 'Congruent': '#BE54C6', 'Incongruent': '#7454C7'}

hrc_df_pooled['HRC'] = hrc_df_pooled['HRC']*100
fig, ax = plt.subplots()
sns.barplot(ax = ax,
            data = hrc_df_pooled,
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax.set_ylim([60, 100])
ax.set_ylabel("Goal-Directed Responses (%)", fontsize = 18)
custom_labels = ["Random", "Congruent (Habit helps)", "Incongruent (Habit hinders)"]  # Define your custom labels here
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, custom_labels, fontsize = 15, loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig("/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/hrc_python.svg",  bbox_inches ='tight')
plt.title("Habedahabeda")
plt.show()

t, p = scipy.stats.ttest_rel(hrc_df_pooled[hrc_df_pooled['Trial Type'] == 'Random']['HRC'], 
                             hrc_df_pooled[hrc_df_pooled['Trial Type'] == 'Incongruent']['HRC'])
print(f"t={t}, p={p}")

t, p = scipy.stats.ttest_rel(hrc_df_pooled[hrc_df_pooled['Trial Type'] == 'Random']['HRC'], 
                             hrc_df_pooled[hrc_df_pooled['Trial Type'] == 'Congruent']['HRC'])
print(f"t={t}, p={p}")

t, p = scipy.stats.ttest_rel(hrc_df_pooled[hrc_df_pooled['Trial Type'] == 'Incongruent']['HRC'], 
                             hrc_df_pooled[hrc_df_pooled['Trial Type'] == 'Congruent']['HRC'])
print(f"t={t}, p={p}")


#%%
'''
    ER ~ Condition, Day
'''

ER_all = complete_df_all.loc[:, ['ID',
                                 'day',
                                 'ER_stt_rand', 
                                   'ER_stt_seq']]

ER_all = ER_all.melt(id_vars=['ID', 'day'], value_vars=['ER_stt_rand', 'ER_stt_seq'])
ER_all['variable'] = ER_all['variable'].map(lambda x: "Fix" if x == 'ER_stt_seq' else
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


'''
    RT ~ Condition, Day
'''

RT_all = complete_df_all.loc[:, ['ID',
                                 'day',
                                 'RT_stt_rand', 
                                   'RT_stt_seq']]

RT_all = RT_all.melt(id_vars=['ID', 'day'], value_vars=['RT_stt_rand', 'RT_stt_seq'])
RT_all['variable'] = RT_all['variable'].map(lambda x: "Fix" if x == 'RT_stt_seq' else
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
    Weimar 3
    Left: HRC Day 1 & Day 2
    Middle: ER Day 1 & Day 2
    Right: RT Day 1 & Day 2
'''

colors2 = {'Random': '#67b798', 'Fix': '#bd97c6'} # random, fix

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
fig, ax = plt.subplots(1, 3, figsize = (20, 10))
sns.barplot(data = hpcf_all,
            x = 'day',
            y = 'HRC',
            hue = 'DTT Type',
            palette = colors1,
            errorbar=('se', 1),
            ax = ax[0])

ax[0].legend(title="", fontsize = 18)
ax[0].set_ylabel("HRC (%)", fontsize = 20)
ax[0].set_ylim([60, 100])
ax[0].set_xlabel("Day", fontsize = 20)
ax[0].tick_params(axis='both', labelsize=18)

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
plt.show()


#%%
'''
    Fig for Weimar
'''
RT_pooled = RT_all.groupby(['ID', 'Condition'], as_index=False).mean()
ER_pooled = ER_all.groupby(['ID', 'Condition'], as_index=False).mean()

fig, ax = plt.subplots(1, 2, figsize = (13, 10))
sns.barplot(data = RT_pooled,
            # x = 'day',
            y = 'RT',
            hue = 'Condition',
            hue_order = ['Random', 'Fix'],
            palette = colors2,
            errorbar=('se', 1),
            ax = ax[0])

ax[0].legend(title="", fontsize = 25)
ax[0].set_ylabel("RT (ms)", fontsize = 28)
ax[0].set_ylim([320, 400])
# ax[0].set_xlabel("Day", fontsize = 25)
ax[0].tick_params(axis='both', labelsize=25)
handles, labels = ax[0].get_legend_handles_labels()
custom_labels = ["Random", "Repeating"]  # Define your custom labels here
ax[0].legend(handles, custom_labels, fontsize = 25)

r,p = scipy.stats.ttest_rel(RT_pooled[RT_pooled['Condition']=='Random']['RT'], 
                            RT_pooled[RT_pooled['Condition']=='Fix']['RT'])
print(f"r={r}, p={p}")

sns.barplot(data = ER_pooled,
            # x = 'day',
            y = 'ER',
            hue = 'Condition',
            hue_order = ['Random', 'Fix'],
            palette = colors2,
            errorbar=('se', 1),
            ax = ax[1])

r,p = scipy.stats.ttest_rel(ER_pooled[ER_pooled['Condition']=='Random']['ER'], 
                            ER_pooled[ER_pooled['Condition']=='Fix']['ER'])
print(f"r={r}, p={p}")

ax[1].legend(title="", fontsize = 25)
ax[1].set_ylabel("Errors (%)", fontsize = 28)
ax[1].set_ylim([3, 9])
# ax[1].set_xlabel("Day", fontsize = 20)
ax[1].tick_params(axis='both', labelsize=25)

handles, labels = ax[1].get_legend_handles_labels()
custom_labels = ["Random", "Repeating"]  # Define your custom labels here
ax[1].legend(handles, custom_labels, fontsize = 25)

plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/positive_measures_python.svg')
plt.show()

#%%

model_plus_spreads_df = complete_df_all.loc[:, ['ID', 'theta_conflict', 'theta_rep', 'day', 'CIspread', 'RIspread', 'CRspread']].copy()

model_plus_spreads_df_pooled = model_plus_spreads_df.groupby(['ID'], as_index = False).mean()

model_plus_spreads_df_pooled = pd.merge(model_plus_spreads_df_pooled, df_ps_pooled, on = 'ID')
model_plus_spreads_df_pooled['repBF'] = anal.compute_BF(model_plus_spreads_df_pooled['theta_rep'], 0)

fig, ax = plt.subplots(1,2, figsize=(8,4))
r,p = scipy.stats.pearsonr(model_plus_spreads_df_pooled['theta_rep'], model_plus_spreads_df_pooled['CIspread'])
sns.regplot(ax = ax[0],
                x = 'theta_rep',
                y = 'CIspread',
                data = model_plus_spreads_df_pooled)
ax[0].set_ylabel('Cong-Inc', fontsize=15)
ax[0].set_xlabel(r'$\theta_{Rep}$', fontsize=15)
ax[0].text(0.5, 0.4, "r=%.2f, p<0.001"%r, fontsize=14)

r,p = scipy.stats.pearsonr(model_plus_spreads_df_pooled['theta_conflict'], model_plus_spreads_df_pooled['RIspread'])
sns.regplot(ax = ax[1],
                x = 'theta_conflict',
                y = 'RIspread',
                data = model_plus_spreads_df_pooled)

ax[1].set_ylabel('Random-Inc', fontsize=15)
ax[1].set_xlabel(r'$\theta_{Switch}$', fontsize=15)
ax[1].text(-0.25, 0.201, "r=%.2f, p<0.001"%r, fontsize=14)
# plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/thetas_vs_spreads_python.svg')
plt.show()

# fig, ax = plt.subplots()
# r,p = scipy.stats.pearsonr(model_plus_spreads_df_pooled['theta_rep'], model_plus_spreads_df_pooled['CRspread'])
# sns.regplot(ax = ax,
#                 x = 'theta_rep',
#                 y = 'CRspread',
#                 data = model_plus_spreads_df_pooled)
# ax.set_ylabel('Cong-Rand', fontsize=15)
# ax.set_xlabel(r'$\theta_{Rep}$', fontsize=15)
# ax.text(0.5, 0.32, "r=%.2f, p<0.001"%r, fontsize=14)
# # plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/Appendix/CR_vs_theta.svg')
# plt.show()

fig, ax = plt.subplots(2,3, figsize=(16, 8))
r,p = scipy.stats.pearsonr(model_plus_spreads_df_pooled['theta_rep'], model_plus_spreads_df_pooled['CRspread'])
sns.regplot(ax = ax[0,0],
                x = 'theta_rep',
                y = 'CRspread',
                data = model_plus_spreads_df_pooled)
ax[0,0].set_ylabel('Cong-Rand', fontsize=15)
ax[0,0].set_xlabel(r'$\theta_{Rep}$', fontsize=15)
ax[0,0].text(0.5, 0.27, "r=%.2f, p<0.001"%r, fontsize=14)

r,p = scipy.stats.pearsonr(model_plus_spreads_df_pooled['theta_rep'], model_plus_spreads_df_pooled['RIspread'])
sns.regplot(ax = ax[0,1],
                x = 'theta_rep',
                y = 'RIspread',
                data = model_plus_spreads_df_pooled)
ax[0,1].set_ylabel('Rand-Inc', fontsize=15)
ax[0,1].set_xlabel(r'$\theta_{Rep}$', fontsize=15)
ax[0,1].text(0.5, 0.2, "r=%.2f, p<0.001"%r, fontsize=14)

r,p = scipy.stats.pearsonr(model_plus_spreads_df_pooled['theta_rep'], model_plus_spreads_df_pooled['CIspread'])
sns.regplot(ax = ax[0,2],
                x = 'theta_rep',
                y = 'CIspread',
                data = model_plus_spreads_df_pooled)

ax[0,2].set_ylabel('Cong-Inc', fontsize=15)
ax[0,2].set_xlabel(r'$\theta_{Rep}$', fontsize=15)
ax[0,2].text(0.5, 0.4, "r=%.2f, p<0.001"%r, fontsize=14)

r,p = scipy.stats.pearsonr(model_plus_spreads_df_pooled['theta_conflict'], model_plus_spreads_df_pooled['CRspread'])
sns.regplot(ax = ax[1,0],
                x = 'theta_conflict',
                y = 'CRspread',
                data = model_plus_spreads_df_pooled)
ax[1,0].set_ylabel('Cong-Rand', fontsize=15)
ax[1,0].set_xlabel(r'$\theta_{Switch}$', fontsize=15)
ax[1,0].text(-0.25, 0.27, "r=%.2f, p<0.001"%r, fontsize=14)

r,p = scipy.stats.pearsonr(model_plus_spreads_df_pooled['theta_conflict'], model_plus_spreads_df_pooled['RIspread'])
sns.regplot(ax = ax[1,1],
                x = 'theta_conflict',
                y = 'RIspread',
                data = model_plus_spreads_df_pooled)
ax[1,1].set_ylabel('Rand-Inc', fontsize=15)
ax[1,1].set_xlabel(r'$\theta_{Switch}$', fontsize=15)
ax[1,1].text(-0.25, 0.2, "r=%.2f, p<0.001"%r, fontsize=14)

r,p = scipy.stats.pearsonr(model_plus_spreads_df_pooled['theta_conflict'], model_plus_spreads_df_pooled['CIspread'])
sns.regplot(ax = ax[1,2],
                x = 'theta_conflict',
                y = 'CIspread',
                data = model_plus_spreads_df_pooled)
ax[1,2].set_ylabel('Cong-Inc', fontsize=15)
ax[1,2].set_xlabel(r'$\theta_{Switch}$', fontsize=15)
ax[1,2].text(-0.25, 0.33, "r=%.2f, p<0.001"%r, fontsize=14)
plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/Appendix/spreads_thetas.svg')
plt.show()

fig, ax = plt.subplots()
r,p = scipy.stats.pearsonr(model_plus_spreads_df_pooled['theta_rep'], model_plus_spreads_df_pooled['theta_conflict'])
sns.regplot(ax = ax,
                x = 'theta_rep',
                y = 'theta_conflict',
                data = model_plus_spreads_df_pooled)
ax.set_ylabel(r'$\theta_{Switch}$', fontsize=15)
ax.set_xlabel(r'$\theta_{Rep}$', fontsize=15)
ax.text(0.5, 0.95, "r=%.2f, p<0.001"%r, fontsize=14)
plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/Appendix/thetas.svg')
plt.show()


#%%

fig, ax = plt.subplots(1,2, sharey=True)
sns.barplot(ax = ax[0],
            data = hrc_df_pooled[hrc_df_pooled['ps_ri'] < 0.05],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[0].set_ylim([60, 100])
ax[0].set_ylabel("Goal-Directed Responses (%)", fontsize = 16)
custom_labels = ["Random", "Congruent (Habit helps)", "Incongruent (Habit hinders)"]  # Define your custom labels here
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles, custom_labels, fontsize = 12, loc='upper left', bbox_to_anchor=(1, 1))

" ==== HRC Differences within Group?"
t,p = scipy.stats.ttest_rel(hrc_df_pooled[(hrc_df_pooled['ps_ri'] < 0.05) & (hrc_df_pooled['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_pooled[(hrc_df_pooled['ps_ri'] < 0.05) & (hrc_df_pooled['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_pooled[(hrc_df_pooled['ps_ri'] < 0.05) & (hrc_df_pooled['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_pooled[(hrc_df_pooled['ps_ri'] < 0.05) & (hrc_df_pooled['Trial Type'] == 'Congruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_pooled[(hrc_df_pooled['ps_ri'] < 0.05) & (hrc_df_pooled['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_pooled[(hrc_df_pooled['ps_ri'] < 0.05) & (hrc_df_pooled['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

sns.barplot(ax = ax[1],
            data = hrc_df_pooled[hrc_df_pooled['ps_ri'] > 0.05],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[1].set_ylim([60, 100])
ax[1].set_ylabel("Goal-Directed Responses (%)", fontsize = 20)
# plt.savefig("/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/hab_vs_GD_python.svg")
# plt.savefig("/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/Appendix/hab_vs_GD_BHcorrect_python.svg")
plt.show()

" ==== HRC Differences within Group?"
t,p = scipy.stats.ttest_rel(hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05) & (hrc_df_pooled['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05) & (hrc_df_pooled['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05) & (hrc_df_pooled['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05) & (hrc_df_pooled['Trial Type'] == 'Congruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05) & (hrc_df_pooled['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05) & (hrc_df_pooled['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

" ==== HRC Differences between Groups?"
t,p = scipy.stats.ttest_ind(hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05) & (hrc_df_pooled['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_pooled[(hrc_df_pooled['ps_ri'] < 0.05) & (hrc_df_pooled['Trial Type'] == 'Random')]['HRC'])
print(f"t={t}, p={p}")

"====== Num of participants per group"
print(f"There are {len(hrc_df_pooled[hrc_df_pooled['ps_ri'] < 0.05])/3} participants in the 'negative effects' group.")
print(f"There are {len(hrc_df_pooled[hrc_df_pooled['ps_ri'] > 0.05])/3} participants in the 'no negative effects' group.")
#%%
'''
    Exploiters vs Suppressers
'''

fig, ax = plt.subplots(1,2, sharey=True, figsize=(4, 4))
sns.barplot(ax = ax[0],
            data = hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05) & (hrc_df_pooled['ps_cr'] < 0.05)],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[0].set_ylim([60, 100])
ax[0].set_ylabel("High-Reward Choices (%)")
# plt.savefig("/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/hrc_python.svg")
# plt.title(f"Habedahabeda 2 (N={len(hrc_df_pooled[hrc_df_pooled['ps_ri'] < 0.05])/3})")

t,p = scipy.stats.ttest_rel(hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05)& (hrc_df_pooled['ps_cr'] < 0.05) & (hrc_df_pooled['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05)& (hrc_df_pooled['ps_cr'] < 0.05) & (hrc_df_pooled['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05)& (hrc_df_pooled['ps_cr'] < 0.05) & (hrc_df_pooled['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05)& (hrc_df_pooled['ps_cr'] < 0.05) & (hrc_df_pooled['Trial Type'] == 'Congruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05)& (hrc_df_pooled['ps_cr'] < 0.05) & (hrc_df_pooled['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05)& (hrc_df_pooled['ps_cr'] < 0.05) & (hrc_df_pooled['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

sns.barplot(ax = ax[1],
            data = hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05) & (hrc_df_pooled['ps_cr'] > 0.05)],
            # x = 'Trial Type',
            y = 'HRC',
            hue = 'Trial Type',
            hue_order = ['Random', 'Congruent', 'Incongruent'],
            palette = colors1,
            errorbar = ('se', 1))
ax[1].set_ylim([60, 100])
ax[1].set_ylabel("High-Reward Choices (%)")
# plt.savefig("/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/GD_vs_adapt_python.svg")
# plt.savefig("/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/Appendix/GD_vs_adapt_BHcorrect_python.svg")
# plt.title(f"Habedahabeda 2 (N={len(hrc_df_pooled[hrc_df_pooled['ps_ri'] > 0.05])/3})")
plt.show()

t,p = scipy.stats.ttest_rel(hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05)& (hrc_df_pooled['ps_cr'] > 0.05) & (hrc_df_pooled['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05)& (hrc_df_pooled['ps_cr'] > 0.05) & (hrc_df_pooled['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05)& (hrc_df_pooled['ps_cr'] > 0.05) & (hrc_df_pooled['Trial Type'] == 'Random')]['HRC'], 
                            hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05)& (hrc_df_pooled['ps_cr'] > 0.05) & (hrc_df_pooled['Trial Type'] == 'Congruent')]['HRC'])
print(f"t={t}, p={p}")

t,p = scipy.stats.ttest_rel(hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05)& (hrc_df_pooled['ps_cr'] > 0.05) & (hrc_df_pooled['Trial Type'] == 'Congruent')]['HRC'], 
                            hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05)& (hrc_df_pooled['ps_cr'] > 0.05) & (hrc_df_pooled['Trial Type'] == 'Incongruent')]['HRC'])
print(f"t={t}, p={p}")

print(f"There are {len(hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05)& (hrc_df_pooled['ps_cr'] < 0.05)])/3} participants in the 'adapter' group.")
print(f"There are {len(hrc_df_pooled[(hrc_df_pooled['ps_ri'] > 0.05)& (hrc_df_pooled['ps_cr'] > 0.05)])/3} participants in the 'suppresser' group.")

#%%
'''
    Appendix Figures
'''
if 0:
    strong_df['ratio_repconf'] = strong_df.apply(lambda row: row['theta_rep']/row['theta_conflict'], axis = 1)
    strong_df['ratio_confrep'] = strong_df.apply(lambda row: row['theta_conflict']/row['theta_rep'], axis = 1)
    
    strong_df['ratio_repconfdiff'] = strong_df.apply(lambda row: row['theta_rep']-row['theta_conflict'], axis = 1)
    strong_df['ratio_confrepdiff'] = strong_df.apply(lambda row: row['theta_conflict']-row['theta_rep'], axis = 1)
    
    fig, ax = plt.subplots(2,3, figsize = (12,8))
    sns.scatterplot(strong_df[strong_df['day'] == 1],
                    x = f'{testparam1}',
                    y = f'{testparam2}',
                    hue = 'strategy',
                    hue_order=['Exploiter', 'pure GD', 'Habitual', 'Anti-GD'],
                    palette = ['r', 'g', 'b', 'm'],
                    ax = ax[0,0])
    ax[0,0].get_legend().remove()
    
    sns.scatterplot(strong_df[strong_df['day'] == 1],
                    x = f'{testparam1}',
                    y = 'theta_Q',
                    hue = 'strategy',
                    hue_order=['Exploiter', 'pure GD', 'Habitual', 'Anti-GD'],
                    palette = ['r', 'g', 'b', 'm'],
                    ax = ax[0, 1])
    
    ax[0,1].get_legend().remove()
    sns.scatterplot(strong_df[strong_df['day'] == 1],
                    y = 'theta_Q',
                    x = f'{testparam2}',
                    hue = 'strategy',
                    hue_order=['Exploiter', 'pure GD', 'Habitual', 'Anti-GD'],
                    palette = ['r', 'g', 'b', 'm'],
                    ax = ax[0, 2])
    
    ax[0,2].legend(loc='upper left', bbox_to_anchor=(1, 1))
    sns.scatterplot(strong_df[strong_df['day'] == 2],
                    x = f'{testparam1}',
                    y = f'{testparam2}',
                    hue = 'strategy',
                    hue_order=['Exploiter', 'pure GD', 'Habitual', 'Anti-GD'],
                    palette = ['r', 'g', 'b', 'm'],
                    ax = ax[1,0])
    ax[1,0].get_legend().remove()
    
    sns.scatterplot(strong_df[strong_df['day'] == 2],
                    x = f'{testparam1}',
                    y = 'theta_Q',
                    hue = 'strategy',
                    hue_order=['Exploiter', 'pure GD', 'Habitual', 'Anti-GD'],
                    palette = ['r', 'g', 'b', 'm'],
                    ax = ax[1, 1])
    
    ax[1,1].get_legend().remove()
    sns.scatterplot(strong_df[strong_df['day'] == 2],
                    y = 'theta_Q',
                    x = f'{testparam2}',
                    hue = 'strategy',
                    hue_order=['Exploiter', 'pure GD', 'Habitual', 'Anti-GD'],
                    palette = ['r', 'g', 'b', 'm'],
                    ax = ax[1, 2])
    
    ax[1,2].get_legend().remove()
    # x = np.array([1, 2.5])
    # ax.plot(x, x-0.5)
    ax[1,2].legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/appim1.svg')
    plt.show()
    
    
    fig, ax = plt.subplots(2,3, figsize = (12,8))
    
    sns.scatterplot(strong_df[strong_df['day'] == 1],
                    x = f'{testparam1}',
                    y = f'{testparam2}',
                    hue = 'stratgroup',
                    palette = ['r', 'b'],
                    ax = ax[0,0])
    ax[0,0].get_legend().remove()
    
    sns.scatterplot(strong_df[strong_df['day'] == 1],
                    x = f'{testparam1}',
                    y = 'theta_Q',
                    hue = 'stratgroup',
                    palette = ['r', 'b'],
                    ax = ax[0,1])
    
    ax[0,1].get_legend().remove()
    sns.scatterplot(strong_df[strong_df['day'] == 1],
                    y = 'theta_Q',
                    x = f'{testparam2}',
                    hue = 'stratgroup',
                    palette = ['r', 'b'],
                    ax = ax[0,2])
    
    # x = np.array([1, 2.5])
    # ax.plot(x, x-0.5)
    ax[0,2].legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    sns.scatterplot(strong_df[strong_df['day'] == 2],
                    x = f'{testparam1}',
                    y = f'{testparam2}',
                    hue = 'stratgroup',
                    palette = ['r', 'b'],
                    ax = ax[1,0])
    ax[1,0].get_legend().remove()
    
    sns.scatterplot(strong_df[strong_df['day'] == 2],
                    x = f'{testparam1}',
                    y = 'theta_Q',
                    hue = 'stratgroup',
                    palette = ['r', 'b'],
                    ax = ax[1,1])
    
    ax[1,1].get_legend().remove()
    sns.scatterplot(strong_df[strong_df['day'] == 2],
                    y = 'theta_Q',
                    x = f'{testparam2}',
                    hue = 'stratgroup',
                    palette = ['r', 'b'],
                    ax = ax[1,2])
    
    # x = np.array([1, 2.5])
    # ax.plot(x, x-0.5)
    ax[1,2].legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.savefig('/home/sascha/Desktop/Paper_2024/Mar/res_fig5b/res_fig5b_c_python.svg', bbox_inches ='tight')
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/appim2.svg')
    plt.show()


#%%

fig, ax = plt.subplots(1,2, sharey=True, sharex = True, figsize = (10,4))

model_plus_spreads_df_pooled['RI_sign'] = model_plus_spreads_df_pooled['ps_ri'].map(lambda x: 1 if x < 0.05 else 0)

model_plus_spreads_df_pooled['stratgroup'] = model_plus_spreads_df_pooled['ps_ri']
model_plus_spreads_df_pooled['stratgroup'] = model_plus_spreads_df_pooled['stratgroup'].map(lambda x: 'negative habit effect' if x < 0.05 else 0)
model_plus_spreads_df_pooled['stratgroup'] = model_plus_spreads_df_pooled.apply(lambda row: 'negative habit effect' if row['ps_ri'] < 0.05 else 
                                                                                'Habit-Exploiters' if row['ps_ri'] > 0.05 and row['ps_cr'] < 0.05 else
                                                                                'Habit-Suppressers' if row['ps_ri'] > 0.05 and row['ps_cr'] > 0.05 else
                                                                                'Habedahabeda', axis =1)

sns.scatterplot(ax = ax[0], 
                data = model_plus_spreads_df_pooled,
                y='theta_conflict',
                x='theta_rep',
                hue='RI_sign',
                palette = ['orangered', 'b'],
                hue_order = [0, 1])
ax[0].set_xlabel(r"$\theta_{Rep}$", fontsize = 20)
ax[0].set_ylabel(r"$\theta_{Switch}$", fontsize = 20)
custom_labels = ["no negative habit effect", "negative habit effect"]
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles, custom_labels, fontsize = 15, loc='upper left', bbox_to_anchor=(1, 1))

sns.scatterplot(ax = ax[1],
                data = model_plus_spreads_df_pooled,
                y='theta_conflict',
                x='theta_rep',
                hue='stratgroup',
                palette = ['fuchsia', 'r', 'b'],
                hue_order = ['Habit-Exploiters', 'Habit-Suppressers', 'negative habit effect'])
ax[1].set_xlabel(r"$\theta_{Rep}$", fontsize = 20)
ax[1].set_ylabel(r"$\theta_{Switch}$", fontsize = 20)
ax[1].legend(fontsize = 15, loc='upper left', bbox_to_anchor=(1, 1))
plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/thetas_vs_RI_python.svg')
plt.show()

#%%

points_pooled = complete_df_all.loc[:, ['ID', 'points_dtt', 'day']].groupby(['ID'], as_index = False).mean()
points_pooled = pd.merge(points_pooled, model_plus_spreads_df_pooled, on = 'ID')

fig, ax = plt.subplots()

sns.barplot(y = 'points_dtt',
            hue = 'stratgroup',
            palette = ['fuchsia', 'r', 'b'],
            hue_order = ['Habit-Exploiters', 'Habit-Suppressers', 'negative habit effect'],
            data = points_pooled,
            errorbar=('se', 1))
ax.set_ylim([300, 400])
ax.legend(title="")
ax.set_ylabel('Total Points (Dual-Target Trials)')
plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/AST/2024_03_Weimar/points_by_strategy.svg')
plt.show()

t,p = scipy.stats.ttest_ind(points_pooled[points_pooled['stratgroup'] == 'Habit-Exploiters']['points_dtt'], 
                            points_pooled[points_pooled['stratgroup'] == 'Habit-Suppressers']['points_dtt'])
print(f"t={t},p={p}")

t,p = scipy.stats.ttest_ind(points_pooled[points_pooled['stratgroup'] == 'Habit-Exploiters']['points_dtt'], 
                            points_pooled[points_pooled['stratgroup'] == 'negative habit effect']['points_dtt'])
print(f"t={t},p={p}")

t,p = scipy.stats.ttest_ind(points_pooled[points_pooled['stratgroup'] == 'Habit-Suppressers']['points_dtt'], 
                            points_pooled[points_pooled['stratgroup'] == 'negative habit effect']['points_dtt'])
print(f"t={t},p={p}")