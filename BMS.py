#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:52:23 2023

@author: sascha
"""

import pymc as pm
import pandas as pd
import numpy as np
import utils
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

def exceedance_probability(traces):
    """Takes the traces of the hierarchical model defined in
    mosq.sequential.hierarchical() and calculates the exceedance
    probabilities for all the componentes of
    traces.get_values('model_probs').    """
    samples = traces #.get_values('model_probs')
    exc_prob = -np.ones(samples.shape[1])
    for best_ix in range(samples.shape[1]):
        exc_prob[best_ix] = sweep_probs(samples, best_ix)
    return exc_prob

def sweep_probs(samples, index):
    """Finds the sweet spot where the best model, indexed by --index--,
    is better than any other.    """
    other_ix = [x for x in range(samples.shape[1]) if x != index]
    c_point = 0.75
    step_size = 0.1
    c_sign = - 1
    while step_size > 0.0001:
        count_other = (samples[:, other_ix] > c_point).sum(axis=0).max()
        count_this = (samples[:, index] <= c_point).sum()
        old_sign = c_sign
        c_sign = 1 - 2 * (count_other < count_this)
        if old_sign != c_sign:
            step_size *= 0.1
        c_point += c_sign * step_size
    return (samples[:, index] >= c_point).sum() / samples.shape[0]

num_models = 8
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
        
        if day == 2:
            filename_day1 = extra_storage[7]
            
            print(f"Filename for day 1 is {filename_day1}")
        
    elif model >= 2:
        assert np.all(participants == expdata_df.loc[:, ['ag_idx', 'ID']].drop_duplicates(subset=['ID', 'ag_idx']))
        
        if day == 2:
            print(f"Filename for day 1 is {extra_storage[7]}")
            # assert filename_day1 == extra_storage[7]
        
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
with pm.Model() as BMS:
    tau = pm.HalfCauchy('hyper_tau', beta=1.0)
    model_probs = pm.Dirichlet('model_probs', a=pt.ones(num_models) / tau,
                                shape=(num_models))

    def logp(model_evidence, model_probs=model_probs):
        log_likelihood = pm.math.log(model_probs, ) + model_evidence
        return pm.math.sum(pm.logsumexp(log_likelihood, axis=1))
    pm.DensityDist('log_joint', model_probs, logp=logp,
                    observed=elbos)
    
    BMSinferenceData = pm.sample(chains = 4, draws = 18_000, tune = 6000)

az.summary(BMSinferenceData)

#%%

posteriorsModelProbs = az.extract(data=BMSinferenceData, var_names=['model_probs']).to_numpy().T
exceedance_probability(posteriorsModelProbs)

#%%
'''
    Plot Bayesian Model Selection
'''
figname = 'Repbias_1day_vs_2days_wo_Q'

# extract samples of all chains
posteriorsModelProbs = az.extract(data=BMSinferenceData, var_names=['model_probs']).to_numpy().T
fig, ax = plt.subplots()
# sns.histplot(posteriorsModelProbs, stat='density', element='step', bins=30, alpha=.1, ax = ax) #, fill=False)

for modelidx in range(num_models):
    sns.kdeplot(posteriorsModelProbs[:, modelidx], linewidth=4, label = model_names[modelidx])
    
plt.ylabel(f'$p(r_i\mid{{data}})$', fontsize=16)
plt.xlabel('Posterior probability of the model', fontsize=16)
plt.xlim([0,1])
plt.yticks([])
# plt.legend(['Bullshit', 'Repbias'], fontsize=14, title='Model', title_fontsize=14)
plt.legend()
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
# plt.yscale('log')
plt.ylim([0, 10])
plt.savefig(f'BMS/{num_models}_models_{figname}_day{day}.png', dpi = 300)
plt.savefig(f'BMS/{num_models}_models_{figname}_day{day}.svg')
plt.title(f"Day {day}")
plt.show()

import pickle

# pickle.dump((model_names, posteriorsModelProbs), open(f"BMS/{num_models}_models_{figname}.p", "wb"))