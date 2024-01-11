#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:52:23 2023

@author: sascha
"""

import pymc as pm
import numpy as np
import utils
import pytensor.tensor as pt
import arviz as az

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

num_models = 5
num_agents = 60
elbos = np.zeros((num_agents, num_models))
# model_files = ['behav_fit_model_B_2023-11-25_60agents.p',
# 'behav_fit_model_Bhand_2023-11-28 23:25:11.p',
# 'behav_fit_model_Conflict_2023-12-02 00:14:47_60agents.p',
# 'behav_fit_model_ConflictHand_2023-12-03 03:38:21_60agents.p',
# 'behav_fit_model_Seqboost_2023-12-02 01:29:31_60agents.p',
# 'behav_fit_model_SeqConflict_2023-12-02 15:31:29_60agents.p',
# 'behav_fit_model_SeqConflictHand_2023-12-03 05:39:36_60agents.p',
# 'behav_fit_model_SeqHand_2023-12-02 13:21:27_60agents.p']

for model in range(num_models):
    out = utils.get_data_from_file()
    post_sample_df, expdata_df, loss, params_df, num_params, sociopsy_df, elbo_tuple = out
    elbos[:, model] = (-elbo_tuple[0]).tolist()
    
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
    
    BMSinferenceData = pm.sample(chains=4, draws=6000, tune=6000)
    
az.summary(BMSinferenceData)

#%%
posteriorsModelProbs = az.extract(data=BMSinferenceData, var_names=['model_probs']).to_numpy().T
exceedance_probability(posteriorsModelProbs)

#%%
import matplotlib.pyplot as plt
import seaborn as sns
# extract samples of all chains
posteriorsModelProbs = az.extract(data=BMSinferenceData, var_names=['model_probs']).to_numpy().T
plt.figure()
sns.histplot(posteriorsModelProbs, stat='density', element='step', bins=30, alpha=.1) #, fill=False)
sns.kdeplot(posteriorsModelProbs, linewidth=4)
plt.ylabel(f'$p(r_i\mid{{data}})$', fontsize=16)
plt.xlabel('Posterior probability of the model', fontsize=16)
plt.xlim([0,1])
plt.yticks([])
# plt.legend(['1', '2', '3', '4', '5'], fontsize=14, title='Model', title_fontsize=14)
plt.show()