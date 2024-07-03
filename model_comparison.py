#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:01:41 2024

@author: sascha
"""

import pandas as pd
import numpy as np
import utils
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

num_models = 2
num_agents = 60
elbos_2nd_lvl = np.zeros(num_models)
elbos = np.zeros((num_agents, num_models))
AICs = np.zeros((num_agents, num_models))
BICs = np.zeros((num_agents, num_models))
WAIC = np.zeros(num_models)
WAIC_var = np.zeros(num_models)
individual_WAIC = np.zeros((num_agents, num_models))
DIC = np.zeros(num_models)
individual_DIC = np.zeros((num_agents, num_models))
log_likelihood = np.zeros(num_models)
# model_files = ['behav_fit_model_B_2023-11-25_60agents.p',
# 'behav_fit_model_Bhand_2023-11-28 23:25:11.p',
# 'behav_fit_model_Conflict_2023-12-02 00:14:47_60agents.p',
# 'behav_fit_model_ConflictHand_2023-12-03 03:38:21_60agents.p',
# 'behav_fit_model_Seqboost_2023-12-02 01:29:31_60agents.p',
# 'behav_fit_model_SeqConflict_2023-12-02 15:31:29_60agents.p',
# 'behav_fit_model_SeqConflictHand_2023-12-03 05:39:36_60agents.p',
# 'behav_fit_model_SeqHand_2023-12-02 13:21:27_60agents.p']
model_names = []
sim_models = []
inf_models = []

participants = pd.DataFrame()

for model in range(num_models):
    post_sample_df, expdata_df, loss, params_df, num_params, sociopsy_df, agent_elbo_tuple, BIC, AIC, extra_storage, filename = utils.get_data_from_file()
    elbos[:, model] = (-agent_elbo_tuple[0]).tolist()
    elbos_2nd_lvl[model] = -np.array(loss[-10:]).mean()
    AICs[:, model] = np.squeeze(AIC.detach().numpy())
    BICs[:, model] = np.squeeze(BIC.detach().numpy())
    WAIC[model] = np.squeeze(extra_storage[12])
    # log_likelihood[model] = np.squeeze(extra_storage[13])
    WAIC_var[model] = np.squeeze(extra_storage[16])
    individual_WAIC[:, model] = np.squeeze(extra_storage[17])
    DIC[model] = np.squeeze(extra_storage[18])
    individual_DIC[:, model] = np.squeeze(extra_storage[20])
    # if len(extra_storage) >= 10:
    #     if extra_storage[11] >= 1e-03:
    #         print("rhalt too large for IC computation.")
    
    day = extra_storage[2]
    
    if 'model' in post_sample_df.columns:
        model_names.append(post_sample_df['model'][0])
        
    else:
        model_names.append(post_sample_df['inf_model'][0])

    "Check that all models have the same ag_idx -> ID mapping."    
    if model == 1:
        participants = expdata_df.loc[:, ['ag_idx', 'ID']].drop_duplicates(subset=['ID', 'ag_idx'])
        
    elif model >= 2:
        assert np.all(participants == expdata_df.loc[:, ['ag_idx', 'ID']].drop_duplicates(subset=['ID', 'ag_idx']))
        
    if 'recovery' in filename:
        strlist = filename.split('/')[-1].split('_')
        simname = strlist[2]
        
        if day == 1:
            stridx = strlist.index('infmodelday1')
            stridx2 = strlist.index('day1')
            infname = strlist[stridx + 1]
            
            for idx in range(3, stridx):
                simname += '_' + strlist[idx]
                
            for idx in range(stridx+2, stridx2):
                infname += '_' + strlist[idx]
            
            sim_models.append(simname)
            inf_models.append(infname)
        
    elif 'behav_fit' in filename:
        strlist = filename.split('/')[-1].split('_')
        
        if day == 1:
            stridx = strlist.index('day1')
            infname = strlist[stridx + 1]
            
            for idx in range(stridx+2, len(strlist)-2):
                infname += '_' + strlist[idx]
            
            inf_models.append(infname)
            
        elif day == 2:
            stridx = strlist.index('day2')
            infname = strlist[stridx + 1]
            
            for idx in range(stridx+2, len(strlist)-2):
                infname += '_' + strlist[idx]
            
            inf_models.append(infname)
            
#%%
import csv

with open('IC.csv', 'w', newline='') as csvfile:

    for midx in range(len(sim_models)):

        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([f'{sim_models[midx]},', f'{inf_models[midx]},', '%.0f,'%WAIC[midx], '%.0f,'%DIC[midx]])
        
    if len(sim_models) == 0:

        for midx in range(len(inf_models)):        

            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([f'{inf_models[midx]},', '%.0f,'%WAIC[midx], '%.0f,'%DIC[midx]])
            
#%%
'''
    Compute Bayes Factors
    BF = p(y_A)/p(y_B) = exp[log p(y_A) - log p(y_B)] = exp[elbo_A - elbo_B]
    
    BF > 1 indicate a preference for model model_names[compidx]
'''

# compidx = model_names.index('Bullshitmodel')
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


        
#%%
'''
    Group-Level WAIC
'''

markershapes = ['o', 'D', '^', '>', '*', '+', 'D', 'x']
colors = ['blue', 'orange', 'green', 'red', 'purple', 'black']

fig, ax = plt.subplots(1, figsize = (10, 5))
# ax.scatter(range(num_models), WAIC)
ax.errorbar(range(num_models), 
            WAIC, 
            yerr=WAIC_var, 
            fmt='o', 
            ecolor='r', 
            capsize=5, 
            linestyle='None', 
            label='Data points')
    
# ax.legend()
ax.title.set_text(f'WAIC (day {day})')
ax.set_xlabel('Model no.')
ax.set_xticks(range(num_models))
ax.set_xticklabels(model_names)
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

plt.savefig('BMS/WAIC_day{day}.svg')
plt.show()

#%%
'''
    Individual WAIC 
'''

markershapes = ['o', 'D', '^', '>', '*', '+', 'D', 'x']
colors = ['blue', 'orange', 'green', 'red', 'purple', 'black']

fig, ax = plt.subplots(sharey = True, figsize = (15,5))
for midx in range(num_models):
    # ax[0].scatter(range(num_agents), AICs[:, midx], s=20, label=model_names[midx], marker = markershapes[midx])
    ax.scatter(range(num_agents), individual_WAIC[:, midx], 
                  # marker=markershapes[midx], 
                  # edgecolor=colors[midx], 
                  # facecolors='none', 
                  linewidth=1, 
                  label=model_names[midx])
    
ax.legend()
ax.title.set_text(f'WAIC (day {day})')
ax.set_xlabel('Agent no.')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()

plt.savefig('BMS/ICs.png')
plt.show()

if AICs.shape[1] == 2:
    '''
        Sort ICs
    '''
    WAIC_diff = individual_WAIC[:, 0] - individual_WAIC[:, 1]
    WAIC_sort_idxs = np.argsort(WAIC_diff)
    WAIC_diff_argmin_idx = np.abs(WAIC_diff[WAIC_sort_idxs]).argmin()
    
    DIC_diff = individual_DIC[:, 0] - individual_DIC[:, 1]
    DIC_sort_idxs = np.argsort(DIC_diff)
    DIC_diff_argmin_idx = np.abs(DIC_diff[DIC_sort_idxs]).argmin()
    
    '''
        Plot 1
        Scatterplot WAIC for both models
    '''
    fig, ax = plt.subplots(sharey = True, figsize = (15,5))
    for midx in range(num_models):
        # ax[0].scatter(range(num_agents), AICs[:, midx], s=20, label=model_names[midx], marker = markershapes[midx])
        ax.scatter(range(num_agents), individual_WAIC[WAIC_sort_idxs, midx], 
                      # marker=markershapes[midx], 
                      # edgecolor=colors[midx], 
                      # facecolors='none', 
                      linewidth=1, 
                      label=model_names[midx])
        
    ax.axvline(WAIC_diff_argmin_idx)
    ax.legend()
    ax.title.set_text(f'WAIC (day {day})')
    ax.set_xlabel('Agent no.')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    # plt.savefig('BMS/ICs.png')
    plt.show()
    
    '''
        Plot 2
        Sorted Barplot WAIC model 1 - model 2<
    '''
    WAIC_sorted = -WAIC_diff[WAIC_sort_idxs]
    DIC_waic_sorted = -DIC_diff[WAIC_sort_idxs]
    Participant = np.arange(1, num_agents+1)
    
    # data = {'ΔWAIC': WAIC_sorted,
    #         'ΔDIC': DIC_waic_sorted,
    #         'Participant': Participant}
    
    PB = list(Participant)
    PB.extend(list(Participant))
    
    DeltaIC = list(WAIC_sorted)
    DeltaIC.extend(list(DIC_waic_sorted))
    
    IC = ['WAIC']*len(list(WAIC_sorted))
    IC.extend(['DIC']*len(list(WAIC_sorted)))
    data = {'ΔIC': DeltaIC,
            'IC': IC,
            'Participant': PB}
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(sharey = True, figsize = (15,5))
    # ax[0].scatter(range(num_agents), AICs[:, midx], s=20, label=model_names[midx], marker = markershapes[midx])
    sns.barplot(WAIC_sorted, 
                  # marker=markershapes[midx], 
                  # edgecolor=colors[midx], 
                  # facecolors='none', 
                  color = 'deepskyblue',
                  linewidth = 1, 
                  label = model_names[midx],
                  ax = ax)
        
    ax.axvline(WAIC_diff_argmin_idx+0.5, color ='k')
    ax.legend()
    ax.title.set_text('WAIC (Model 3 - Model 1)')
    ax.set_xlabel('Participants', fontsize = 20)
    ax.set_ylabel(r'$\Delta$WAIC', fontsize = 20)
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xticks(np.arange(1, num_agents+1), minor = True)
    plt.tight_layout()

    # plt.savefig('/home/sascha/Desktop/Paper_2024/May/res_fig4/individual_diffs_python.svg')
    plt.show()
    
    
    
    '''
        Plot 1
        Scatterplot DIC for both models
    '''
    fig, ax = plt.subplots(sharey = True, figsize = (15,5))
    for midx in range(num_models):
        # ax[0].scatter(range(num_agents), AICs[:, midx], s=20, label=model_names[midx], marker = markershapes[midx])
        ax.scatter(range(num_agents), individual_DIC[DIC_sort_idxs, midx], 
                      # marker=markershapes[midx], 
                      # edgecolor=colors[midx], 
                      # facecolors='none', 
                      linewidth=1, 
                      label=model_names[midx])
        
    ax.axvline(DIC_diff_argmin_idx)
    ax.legend()
    ax.title.set_text(f'DIC (day {day})')
    ax.set_xlabel('Agent no.')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    # plt.savefig('BMS/ICs.png')
    plt.show()
    
    '''
        Plot 2
        Sorted Barplot DIC model 1 - model 2<
    '''
    fig, ax = plt.subplots(sharey = True, figsize = (15,5))
    # ax[0].scatter(range(num_agents), AICs[:, midx], s=20, label=model_names[midx], marker = markershapes[midx])
    sns.barplot(-DIC_diff[DIC_sort_idxs], 
                  # marker=markershapes[midx], 
                  # edgecolor=colors[midx], 
                  # facecolors='none', 
                  color = 'deepskyblue',
                  linewidth = 1, 
                  label = model_names[midx],
                  ax = ax)
        
    ax.axvline(DIC_diff_argmin_idx+0.5, color ='k')
    ax.legend()
    ax.title.set_text('DIC (Model 3 - Model 1)')
    ax.set_xlabel('Participants', fontsize = 20)
    ax.set_ylabel(r'$\Delta$DIC', fontsize = 20)
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xticks(np.arange(1, num_agents+1), minor = True)
    plt.tight_layout()

    # plt.savefig('/home/sascha/Desktop/Paper_2024/May/res_fig4/individual_diffs_python.svg')
    plt.show()


    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Participant', y='ΔIC', hue='IC', data=df, ax = ax)
    ax.set_xlabel('Participant', fontsize = 20)
    ax.set_ylabel('ΔIC', fontsize = 20)
    ax.tick_params(axis='y', labelsize=16)
    plt.savefig('/home/sascha/Desktop/Paper_2024/May/suppl_fig3/suppl_fig3.svg')
    plt.show()


#%%
wins = []
winning_model = []
IC = []

for midx in range(num_models):
    wins.append((individual_WAIC.argmin(axis=1) == midx).astype(int).sum())
    winning_model.append(model_names[midx])
    IC.append('WAIC')
    
    wins.append((individual_DIC.argmin(axis=1) == midx).astype(int).sum())
    winning_model.append(model_names[midx])
    IC.append('DIC')

IC_performance_df = pd.DataFrame({'Wins':wins, 'IC':IC, 'Model':winning_model})

fig, ax = plt.subplots()
sns.barplot(IC_performance_df,
            x = 'Model',
            y = 'Wins',
            hue = 'IC')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig(f'winning_models_day{day}.svg')

ax.set_xlabel('Model', fontsize=20)
ax.set_ylabel('Count', fontsize=20)

plt.show()

#%%
'''
    Group-Level DIC
'''

markershapes = ['o', 'D', '^', '>', '*', '+', 'D', 'x']
colors = ['blue', 'orange', 'green', 'red', 'purple', 'black']

fig, ax = plt.subplots(1, figsize = (10, 5))
# ax.scatter(range(num_models), WAIC)
ax.errorbar(range(num_models), 
            DIC, 
            # yerr=WAIC_var, 
            fmt='o', 
            ecolor='r', 
            capsize=5, 
            linestyle='None', 
            label='Data points')
    
# ax.legend()
ax.title.set_text(f'DIC (day {day})')
ax.set_xlabel('Model no.')
ax.set_xticks(range(num_models))
ax.set_xticklabels(model_names)
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

plt.savefig('BMS/DIC_day{day}.svg')
plt.show()

#%%
'''
    Individual DIC 
'''

markershapes = ['o', 'D', '^', '>', '*', '+', 'D', 'x']
colors = ['blue', 'orange', 'green', 'red', 'purple', 'black']

fig, ax = plt.subplots(sharey = True, figsize = (15,5))
for midx in range(num_models):
    # ax[0].scatter(range(num_agents), AICs[:, midx], s=20, label=model_names[midx], marker = markershapes[midx])
    ax.scatter(range(num_agents), individual_DIC[:, midx], 
                  # marker=markershapes[midx], 
                  # edgecolor=colors[midx], 
                  # facecolors='none', 
                  linewidth=1, 
                  label=model_names[midx])
    
ax.legend()
ax.title.set_text(f'DIC (day {day})')
ax.set_xlabel('Agent no.')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()

plt.savefig('BMS/ICs.png')
plt.show()

if AICs.shape[1] == 2:
    '''
        Sort DIC
    '''
    DIC_diff = individual_DIC[:, 0] - individual_DIC[:, 1]
    DIC_sort_idxs = np.argsort(DIC_diff)
    DIC_diff_argmin_idx = np.abs(DIC_diff[DIC_sort_idxs]).argmin()
    
    fig, ax = plt.subplots(sharey = True, figsize = (15,5))
    for midx in range(num_models):
        # ax[0].scatter(range(num_agents), AICs[:, midx], s=20, label=model_names[midx], marker = markershapes[midx])
        ax.scatter(range(num_agents), individual_DIC[DIC_sort_idxs, midx], 
                      # marker=markershapes[midx], 
                      # edgecolor=colors[midx], 
                      # facecolors='none', 
                      linewidth=1, 
                      label=model_names[midx])
        
    ax.axvline(DIC_diff_argmin_idx)
    ax.legend()
    ax.title.set_text(f'DIC (day {day})')
    ax.set_xlabel('Agent no.')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    # plt.savefig('BMS/ICs.png')
    plt.show()

#%%
# from scipy import stats
# model1idx = 2
# model2idx = 4

# print("Check these formulas")
# # Pooled variance
# sp2 = ((60 - 1) * WAIC_var[model1idx] + (60 - 1) * WAIC_var[model2idx]) / (60 + 60 - 2)

# # t-value
# t_value = (WAIC[model1idx] - WAIC[model2idx]) / np.sqrt(sp2 * (1/60 + 1/60))
# print(f"t={t_value}")

# # Degrees of freedom
# df = 60 + 60 - 2

# # p-value
# p_value = 2 * stats.t.sf(np.abs(t_value), df)  # Two-tailed test
# print(f'p-value: {p_value}')

#%%
# '''
#     Plot log-likelihood
# '''
# fig, ax = plt.subplots(1, figsize = (10, 5))
# # ax.scatter(range(num_models), WAIC)
# ax.scatter(range(num_models), log_likelihood)
    
# # ax.legend()
# ax.title.set_text(f'Log-Likelihood (day {day})')
# ax.set_xlabel('Model no.')
# ax.set_xticks(range(num_models))
# ax.set_xticklabels(model_names)
# # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.tight_layout()

# plt.savefig('BMS/loglike_day{day}.svg')
# plt.show()


#%%
# '''
#     Plot 2nd-level ELBOS
# '''
# fig, ax = plt.subplots(1, figsize = (10, 5))
# # ax.scatter(range(num_models), WAIC)
# ax.scatter(range(num_models), elbos_2nd_lvl)
    
# # ax.legend()
# ax.title.set_text(f'ELBO (day {day})')
# ax.set_xlabel('Model no.')
# ax.set_xticks(range(num_models))
# ax.set_xticklabels(model_names)
# # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.tight_layout()

# plt.savefig('BMS/loglike_day{day}.svg')
# plt.show()

#%%
'''
    AIC & BIC
'''

markershapes = ['o', 'D', '^', '>', '*', '+', 'D', 'x']
colors = ['blue', 'orange', 'green', 'red', 'purple', 'black']

fig, ax = plt.subplots(1,2, sharey = True, figsize = (15,5))
for midx in range(num_models):
    # ax[0].scatter(range(num_agents), AICs[:, midx], s=20, label=model_names[midx], marker = markershapes[midx])
    ax[0].scatter(range(num_agents), AICs[:, midx], 
                  # marker=markershapes[midx], 
                  # edgecolor=colors[midx], 
                  # facecolors='none', 
                  linewidth=1, 
                  label=model_names[midx])
    
ax[0].legend()
ax[0].title.set_text(f'AIC (day {day})')
ax[0].set_xlabel('Agent no.')
ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))


for midx in range(num_models):
    ax[1].scatter(range(num_agents), BICs[:, midx], 
                  # marker=markershapes[midx], 
                  # edgecolor=colors[midx], 
                  # facecolors='none', 
                  linewidth=1, 
                  label=model_names[midx])
    
ax[1].legend()
ax[1].title.set_text(f'BIC (day {day})')
ax[1].set_xlabel('Agent no.')
ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

plt.savefig('BMS/ICs.png')
plt.show()

if AICs.shape[1] == 2:
    '''
        Sort AIC & BIC
    '''
    AIC_diff = AICs[:, 0] - AICs[:, 1]
    AIC_sort_idxs = np.argsort(AIC_diff)
    AIC_diff_argmin_idx = np.abs(AIC_diff[AIC_sort_idxs]).argmin()
    
    BIC_diff = BICs[:, 0] - BICs[:, 1]
    BIC_sort_idxs = np.argsort(BIC_diff)
    BIC_diff_argmin_idx = np.abs(BIC_diff[BIC_sort_idxs]).argmin()
    
    fig, ax = plt.subplots(1,2, sharey = True, figsize = (15,5))
    for midx in range(num_models):
        # ax[0].scatter(range(num_agents), AICs[:, midx], s=20, label=model_names[midx], marker = markershapes[midx])
        ax[0].scatter(range(num_agents), AICs[AIC_sort_idxs, midx], 
                      marker=markershapes[midx], 
                      edgecolor=colors[midx], 
                      facecolors='none', 
                      linewidth=1, 
                      label=model_names[midx])
        
    ax[0].axvline(AIC_diff_argmin_idx)
    ax[0].legend()
    ax[0].title.set_text(f'AIC (day {day})')
    ax[0].set_xlabel('Agent no.')
    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))


    for midx in range(num_models):
        ax[1].scatter(range(num_agents), BICs[BIC_sort_idxs, midx], 
                      marker=markershapes[midx], 
                      edgecolor=colors[midx], 
                      facecolors='none', 
                      linewidth=1, 
                      label=model_names[midx])
    
    ax[1].axvline(BIC_diff_argmin_idx)
    ax[1].legend()
    ax[1].title.set_text(f'BIC (day {day})')
    ax[1].set_xlabel('Agent no.')
    ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # plt.savefig('BMS/ICs.png')
    plt.show()