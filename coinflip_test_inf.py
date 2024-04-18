#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Mon Jan 29 10:44:36 2024
    
    @author: sascha
"""

from IPython import get_ipython
get_ipython().run_line_magic("reset", "-f")

import ipdb
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import models_torch as models

import analysis_tools as anal
import inferencemodels
import utils

import numpy as np
import arviz as az
import xarray as xr

phead = 0.89
num_inf_steps = 1_00
num_waic_samples = 100
num_agents = 20
num_trials = 100

'''
    Inference
'''
"----- Initialize new agent object with num_agents agents for inference"
print(f"\nGood Model with phead={phead}")
agent = models.Coinflip_test(num_agents)

# print(f"\nBullshit Model with phead={phead}")
# agent = utils.init_agent('Coinflip_test_bullshit', 
#                           group, 
#                           num_agents = num_agents)

coinflip_data = torch.tensor(np.random.choice([0, 1], 
                                              p = [1-phead, phead], 
                                              size = (num_agents, num_trials))) # 0 heads, 1 tails

infer = inferencemodels.CoinflipGroupInference_flat(agent, coinflip_data)

'''
    Inference
'''
print("===== Starting inference =====")
print(f"num_trials = {num_trials}, num_agents={num_agents}")
"----- Start Inference"
infer.infer_posterior(iter_steps = num_inf_steps, num_particles = 10)
plt.plot(infer.loss)

post_sample_df = infer.sample_posterior(n_samples=5000)
post_mean_df = post_sample_df.groupby(['ag_idx'], as_index = False).mean()


'''
    Compute log-likelihood and WAIC
'''
# max_log_like, mle_locs = infer.train_mle(iter_steps = 800, halting_rtol=1e-07)
_, _, WAIC, ll, WAIC_var, subject_WAIC, DIC, subject_DIC, all_post_loc_samples, all_post_samples, ll_complete = infer.compute_IC(num_samples = num_waic_samples)
# all_locs = infer.compute_IC_test(num_samples = num_waic_samples)
# # print("\n\nWAIC = %.0f pm %.0f, DIC = %.0f, loglike=%.0f"%(WAIC, WAIC_var, DIC, ll))
# print(torch.sigmoid(all_post_loc_samples.mean(axis=0)).mean())
# print(all_post_samples['heads_p'].mean(axis=1).mean())
 
#%%
'''
    idata with posterior & likelihood
'''
new_loglike = torch.zeros((1, num_waic_samples, num_trials*num_agents))

for widx in range(num_waic_samples):
    for nt in range(num_trials):
        for na in range(num_agents):
            # print(f"[0, {widx}, {na*num_trials+nt}]")
            new_loglike[0, widx, na*num_trials+nt]  = ll_complete[widx, na, nt]

#%%
# lltemp = torch.clone(new_loglike)
# post_temp = all_post_samples.copy()

'''
    Arange all_post_samples and ll
'''
datadict = all_post_samples
idata = az.convert_to_inference_data(datadict)
# log_likelihood_values = ll_complete[None,...]
log_likelihood_values = new_loglike

# Convert log_likelihood_values to an xarray DataArray with the correct dimensions
log_likelihood_da = xr.DataArray(
    log_likelihood_values,
    dims=["chain", "draw", "observation"],
    coords={
        "chain": [0],  # Assuming a single chain for simplicity
        "draw": np.arange(log_likelihood_values.shape[1]),
        "observation": np.arange(log_likelihood_values.shape[2])
    }
)

# Add the log_likelihood DataArray to the InferenceData object
if hasattr(idata, 'log_likelihood'):
    idata.log_likelihood["log_likelihood"] = log_likelihood_da
    
else:
    idata.add_groups(log_likelihood={"log_likelihood": log_likelihood_da})
    
    
# Print a summary of the InferenceData object
print(idata)

# Optionally, explore the log_likelihood group specifically
print(idata.log_likelihood)

print("\n\n")
print(az.waic(idata))
az.waic(idata)['elpd_waic']
az.waic(idata)['p_waic']

waic = -2*az.waic(idata)['elpd_waic']

print(f"num_trials = {num_trials}, num_agents={num_agents}")
print(f"\narviz waic computed as {waic}")
print(f"own waic computed as {WAIC} pm {WAIC_var}\n")
print("\n\n")

#%%
lldf = idata.log_likelihood.to_dataframe()
xdata = xr.Dataset.from_dataframe(lldf.groupby(['chain','draw','observations_dim_0']).mean())
idata = az.InferenceData(log_likelihood=xdata)
print(az.waic(idata))

lldf = lldf.reset_index()

dfnew = lldf.reset_index()
vardf = dfnew.loc[:, ['log_likelihood_dim_0', 'log_likelihood']].groupby(['log_likelihood_dim_0']).var()
print(vardf)
print(vardf.sum())

#%%
'''
    idata without posterior
'''

tryhier = 0

print("From log-likelihood... \n\n\n")

chain = [0 for _ in range(num_agents*num_trials*num_waic_samples)]
draw = [i for i in range(num_waic_samples) for _ in range(num_trials*num_agents)]
if tryhier:
    agent = [x for _ in range(num_waic_samples) for x in range(num_agents) for _ in range(num_trials)]
    observations_dim_0 = [i for _ in range(num_agents*num_waic_samples) for i in range(num_trials)]
    
else:
    observations_dim_0 = [i for _ in range(num_waic_samples) for i in range(num_agents*num_trials)]
    
observations = ll_complete.flatten()

if tryhier:
    lldf = pd.DataFrame({'chain': chain,
                          'draw': draw,
                          'agent': agent,
                          'observations_dim_0':observations_dim_0,
                          'observations':observations})
    
else:
    lldf = pd.DataFrame({'chain': chain,
                          'draw': draw,
                          'observations_dim_0':observations_dim_0,
                          'observations':observations})

if tryhier:
    xdata = xr.Dataset.from_dataframe(lldf.groupby(['chain','draw','agent', 'observations_dim_0']).mean())
    
else:
    xdata = xr.Dataset.from_dataframe(lldf.groupby(['chain','draw','observations_dim_0']).mean())

# log_likelihood_xr = xr.Dataset({
#     "log_likelihood": (("chain", "draw", "observation_dim_0"), ll_final[None, ...])
# }, coords={"chain": [0], 
#            "draw": np.arange(ll_final.shape[0]),
#            "observation_dim_0": np.arange(ll_final.shape[1])})

idata = az.InferenceData(log_likelihood=xdata)

print(az.waic(idata))
waic_result = az.waic(idata, scale="log")

waic = -2*az.waic(idata)['elpd_waic']
print(waic)

#%%

dfnew = lldf.reset_index()
vardf = dfnew.loc[:, ['observations_dim_0', 'observations']].groupby(['observations_dim_0']).var()
print(vardf)
print(vardf.sum())