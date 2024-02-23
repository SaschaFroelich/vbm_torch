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

num_models = 2
num_agents = 60
elbos_2nd_lvl = np.zeros(num_models)
elbos = np.zeros((num_agents, num_models))
AICs = np.zeros((num_agents, num_models))
BICs = np.zeros((num_agents, num_models))
WAIC = np.zeros(num_models)
WAIC_var = np.zeros(num_models)
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

participants = pd.DataFrame()

for model in range(num_models):
    post_sample_df, expdata_df, loss, params_df, num_params, sociopsy_df, agent_elbo_tuple, BIC, AIC, extra_storage = utils.get_data_from_file()
    elbos[:, model] = (-agent_elbo_tuple[0]).tolist()
    elbos_2nd_lvl[model] = -np.array(loss[-10:]).mean()
    AICs[:, model] = np.squeeze(AIC.detach().numpy())
    BICs[:, model] = np.squeeze(BIC.detach().numpy())
    WAIC[model] = np.squeeze(extra_storage[12])
    WAIC_var[model] = np.squeeze(extra_storage[16])
    log_likelihood[model] = np.squeeze(extra_storage[13])
    if len(extra_storage) >= 10:
        if extra_storage[11] >= 1e-03:
            raise Exception("rhalt too large for IC computation.")
    
    day = extra_storage[2]
    
    model_names.append(post_sample_df['model'][0])

    "Check that all models have the same ag_idx -> ID mapping."    
    if model == 1:
        participants = expdata_df.loc[:, ['ag_idx', 'ID']].drop_duplicates(subset=['ID', 'ag_idx'])
        
    elif model >= 2:
        assert np.all(participants == expdata_df.loc[:, ['ag_idx', 'ID']].drop_duplicates(subset=['ID', 'ag_idx']))
    
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
                  marker=markershapes[midx], 
                  edgecolor=colors[midx], 
                  facecolors='none', 
                  linewidth=1, 
                  label=model_names[midx])
    
ax[0].legend()
ax[0].title.set_text(f'AIC (day {day})')
ax[0].set_xlabel('Agent no.')
ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))


for midx in range(num_models):
    ax[1].scatter(range(num_agents), BICs[:, midx], 
                  marker=markershapes[midx], 
                  edgecolor=colors[midx], 
                  facecolors='none', 
                  linewidth=1, 
                  label=model_names[midx])
    
ax[1].legend()
ax[1].title.set_text(f'BIC (day {day})')
ax[1].set_xlabel('Agent no.')
ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

plt.savefig('BMS/ICs.png')
plt.show()

#%%
'''
    Compute Bayes Factors
    BF = p(y_A)/p(y_B) = exp[log p(y_A) - log p(y_B)] = exp[elbo_A - elbo_B]
'''

compidx = model_names.index('Bullshitmodel')

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
        
#%%
'''
    Plot WAIC
'''

markershapes = ['o', 'D', '^', '>', '*', '+', 'D', 'x']
colors = ['blue', 'orange', 'green', 'red', 'purple', 'black']

fig, ax = plt.subplots(1, figsize = (10, 5))
# ax.scatter(range(num_models), WAIC)
ax.errorbar(range(num_models), WAIC, yerr=WAIC_var, fmt='o', ecolor='r', capsize=5, linestyle='None', label='Data points')

    
# ax.legend()
ax.title.set_text(f'WAIC (day {day})')
ax.set_xlabel('Model no.')
ax.set_xticks(range(num_models))
ax.set_xticklabels(model_names)
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

plt.savefig('BMS/WAIC_day{day}.svg')
plt.show()

from scipy import stats
model1idx = 2
model2idx = 4

print("Check these formulas")
# Pooled variance
sp2 = ((60 - 1) * WAIC_var[model1idx] + (60 - 1) * WAIC_var[model2idx]) / (60 + 60 - 2)

# t-value
t_value = (WAIC[model1idx] - WAIC[model2idx]) / np.sqrt(sp2 * (1/60 + 1/60))
print(f"t={t_value}")

# Degrees of freedom
df = 60 + 60 - 2

# p-value
p_value = 2 * stats.t.sf(np.abs(t_value), df)  # Two-tailed test
print(f'p-value: {p_value}')

'''
    Plot log-likelihood
'''
fig, ax = plt.subplots(1, figsize = (10, 5))
# ax.scatter(range(num_models), WAIC)
ax.scatter(range(num_models), log_likelihood)
    
# ax.legend()
ax.title.set_text(f'Log-Likelihood (day {day})')
ax.set_xlabel('Model no.')
ax.set_xticks(range(num_models))
ax.set_xticklabels(model_names)
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

plt.savefig('BMS/loglike_day{day}.svg')
plt.show()