#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Wed Oct 25 11:05:13 2023

    Fit model to behaviour.

    @author: sascha
"""
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import pickle

import env
import analysis_tools as anal
import inferencemodels
import utils

'''
Modelle:

    Vbm_lr
    Repbias_lr
    Repbias_Conflict
    Repbias_CongConflict_lr
    OnlyQ_lr
'''

waithrs = 15
post_pred = 0

model_day1 = 'Repbias_CongConflict_lr'
models_day2 = ['Repbias_CongConflict_lr', 'Repbias_CongConflict_nolr']
num_inf_steps_day1 = 3_000
halting_rtol_day1 = 1e-02 # for MLE estimation
posterior_pred_samples_day1 = 5_000

num_inf_steps_day2 = 3_000
halting_rtol_day2 = 1e-06 # for MLE estimation
posterior_pred_samples_day2 = 5_000

"Only temporarily to speed up inference"
Q_init_day2 = torch.tensor([[[0.7305, 0.1772, 0.1805, 0.7223],
         [0.7325, 0.2006, 0.2010, 0.7628],
         [0.8230, 0.2532, 0.2596, 0.7975],
         [0.7342, 0.1661, 0.1554, 0.7284],
         [0.7836, 0.1943, 0.1795, 0.7822],
         [0.7777, 0.1780, 0.1884, 0.7453],
         [0.6813, 0.1659, 0.1544, 0.7254],
         [0.7757, 0.2259, 0.1588, 0.8235],
         [0.8382, 0.1596, 0.2114, 0.8409],
         [0.7542, 0.1798, 0.2108, 0.7646],
         [0.7637, 0.1880, 0.1957, 0.7717],
         [0.8157, 0.1505, 0.3074, 0.7511],
         [0.7732, 0.1720, 0.2227, 0.7530],
         [0.8802, 0.2181, 0.1988, 0.8753],
         [0.5888, 0.1182, 0.1302, 0.5948],
         [0.7011, 0.1748, 0.1778, 0.7389],
         [0.7159, 0.1984, 0.1592, 0.7300],
         [0.6777, 0.1428, 0.1410, 0.6678],
         [0.7243, 0.1922, 0.1408, 0.7040],
         [0.7836, 0.2128, 0.1774, 0.7910],
         [0.7366, 0.1636, 0.1407, 0.6985],
         [0.7812, 0.2190, 0.2202, 0.8130],
         [0.6725, 0.1437, 0.1474, 0.6578],
         [0.8010, 0.2431, 0.1872, 0.7459],
         [0.6847, 0.1791, 0.1981, 0.7438],
         [0.6719, 0.1457, 0.1349, 0.6461],
         [0.8304, 0.1426, 0.1994, 0.7111],
         [0.7221, 0.1499, 0.1519, 0.7065],
         [0.7817, 0.1735, 0.1672, 0.7622],
         [0.7796, 0.1926, 0.1957, 0.7718],
         [0.1407, 0.6276, 0.6250, 0.1205],
         [0.1744, 0.7421, 0.7754, 0.1972],
         [0.1568, 0.6988, 0.7177, 0.1551],
         [0.1191, 0.7423, 0.8040, 0.1918],
         [0.2067, 0.8063, 0.7901, 0.1922],
         [0.1756, 0.7352, 0.7286, 0.1658],
         [0.2051, 0.8031, 0.8225, 0.2721],
         [0.1620, 0.8244, 0.7901, 0.2160],
         [0.1927, 0.7716, 0.8143, 0.1687],
         [0.1751, 0.8391, 0.8861, 0.1964],
         [0.1984, 0.7647, 0.7460, 0.2108],
         [0.1929, 0.8433, 0.8147, 0.1857],
         [0.1803, 0.8167, 0.8406, 0.2152],
         [0.1748, 0.8424, 0.8436, 0.2255],
         [0.1969, 0.7814, 0.8302, 0.1728],
         [0.2097, 0.7505, 0.8183, 0.1601],
         [0.2190, 0.7952, 0.8364, 0.2440],
         [0.1766, 0.7745, 0.7452, 0.1948],
         [0.1807, 0.7218, 0.8200, 0.2279],
         [0.2091, 0.7971, 0.8331, 0.1934],
         [0.1818, 0.7473, 0.8080, 0.2448],
         [0.2428, 0.7474, 0.7738, 0.1720],
         [0.1778, 0.7983, 0.7697, 0.1600],
         [0.1629, 0.7297, 0.7963, 0.1514],
         [0.1710, 0.7035, 0.7137, 0.1784],
         [0.2381, 0.8256, 0.7420, 0.1580],
         [0.1822, 0.8219, 0.7668, 0.2237],
         [0.2014, 0.8355, 0.7941, 0.2794],
         [0.1993, 0.7762, 0.8443, 0.2386],
         [0.3121, 0.7267, 0.8235, 0.2677]]])

#%%

"Day 1"
exp_behav_dict_day1, expdata_df_day1 = pickle.load(open("behav_data/preproc_data_day1.p", "rb" ))
num_agents = len(expdata_df_day1['ag_idx'].unique())
group = exp_behav_dict_day1['group'][0]

"Make sure same number of participants in each group"
group_distro = [(np.array(group)==grp).sum() for grp in range(4)]
assert np.abs(np.diff(group_distro)).sum() == 0

"Day 2"
exp_behav_dict_day2, expdata_df_day2 = pickle.load(open("behav_data/preproc_data_day2.p", "rb" ))
num_agents = len(expdata_df_day2['ag_idx'].unique())
group = exp_behav_dict_day2['group'][0]

"Make sure same number of participants in each group"
group_distro = [(np.array(group)==grp).sum() for grp in range(4)]
assert np.abs(np.diff(group_distro)).sum() == 0


'''
    Fit day 1
'''
import time
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Waiting for {waithrs} hours, starting at {timestamp}.")
time.sleep(waithrs*3600)

'''
    Inference
'''
print(f"Starting inference of model {model_day1} for day 1 for {num_agents} agents.")
"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent(model_day1, 
                         group, 
                         num_agents = num_agents)

Q_init_day1 = agent.Q_init

print("===== Starting inference for day 1 =====")
"----- Start Inference"
infer = inferencemodels.GeneralGroupInference(agent, exp_behav_dict_day1)
agent_elbo_tuple = infer.infer_posterior(iter_steps = num_inf_steps_day1, num_particles = 10)

"----- Sample parameter estimates from posterior and add information to DataFrame"
if post_pred:
    firstlevel_df, secondlevel_df = infer.posterior_predictives(n_samples = posterior_pred_samples_day1)

else:
    firstlevel_df = infer.sample_posterior(n_samples = posterior_pred_samples_day1)
    secondlevel_df = None

firstlevel_df['group'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict_day1['group'][0][x])
firstlevel_df['model'] = [model_day1]*len(firstlevel_df)

ID_df = expdata_df_day1.loc[:, ['ID', 'ag_idx']].drop_duplicates()
firstlevel_df['ID'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict_day1['ID'][0][x])

"----- MLE & IC"
max_log_like, mle_locs = infer.train_mle(halting_rtol = halting_rtol_day1)
BIC, AIC = infer.compute_IC()

"----- Q_init & seqcounter for next day"
seq_counter_day2 = infer.agent.seq_counter.detach()

param_names = agent.param_names
inf_mean_df = firstlevel_df.loc[:, [*param_names, 
                      'ag_idx', 
                      'ID']].groupby(['ag_idx', 
                                      'ID'], as_index = False).mean()
assert torch.all(torch.tensor(inf_mean_df['ag_idx']) == torch.tensor(exp_behav_dict_day1['ag_idx'][0]))
assert all([inf_mean_df['ID'][i] == exp_behav_dict_day1['ID'][0][i] for i in range(num_agents)])
                                      
_, _, _, sim_agent = utils.simulate_data(model_day1, 
                                        num_agents,
                                        group = group,
                                        day = 1,
                                        params = inf_mean_df.loc[:, [*param_names]])

assert sim_agent.Q[-1].shape[0] == 1 and sim_agent.Q[-1].ndim == 3
# Q_init_day2 = np.squeeze(np.array(sim_agent.Q))[-10:, :, :].mean(axis=0)
# Q_init_day2 = Q_init_day2[None, ...]
# assert Q_init_day2.ndim == 3
# Q_init_day2 = torch.tensor(Q_init_day2)

"----- Save parameter names to DataFrame"
params_sim_df = pd.DataFrame(columns = agent.param_dict.keys())
for col in params_sim_df.columns:
    params_sim_df[col] = ['unknown']

params_sim_df['ag_idx']  = None
params_sim_df['group']  = None
params_sim_df['model']  = model_day1

"----- Store results"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
extra_storage = (Q_init_day1,
                 agent.Q[-1].detach(),
                 1, # day
                 'no preceding model',
                 max_log_like,
                 mle_locs,
                 '',
                 '',
                 secondlevel_df)

filename_day1 = f'behav_fit_model_day1_{model_day1}_{timestamp}_{num_agents}agents'
if num_inf_steps_day1 > 100:
    pickle.dump( (firstlevel_df, 
                  expdata_df_day1,
                  (infer.loss, BIC, AIC), 
                  params_sim_df, 
                  agent_elbo_tuple, 
                  extra_storage), 
                open(f"behav_fit/{filename_day1}.p", "wb" ) )

'''
    Fit day 2
'''
del exp_behav_dict_day1
del expdata_df_day1
for model_day2 in models_day2:
    agent = utils.init_agent(model_day2, 
                             group, 
                             num_agents = num_agents,
                             Q_init = Q_init_day2.detach(),
                             seq_init = seq_counter_day2)
    
    print("===== Starting inference for day 2 =====")
    "----- Start Inference"
    infer = inferencemodels.GeneralGroupInference(agent, exp_behav_dict_day2)
    agent_elbo_tuple = infer.infer_posterior(iter_steps = num_inf_steps_day2, num_particles = 10)

    "----- Sample parameter estimates from posterior and add information to DataFrame"
    if post_pred:
        firstlevel_df, secondlevel_df = infer.posterior_predictives(n_samples = posterior_pred_samples_day2)
    else:
        firstlevel_df = infer.sample_posterior(n_samples = posterior_pred_samples_day2)
        secondlevel_df = None
    
    firstlevel_df['group'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict_day2['group'][0][x])
    firstlevel_df['model'] = [model_day2]*len(firstlevel_df)
    
    ID_df = expdata_df_day2.loc[:, ['ID', 'ag_idx']].drop_duplicates()
    firstlevel_df['ID'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict_day2['ID'][0][x])
    
    "----- MLE & IC"
    max_log_like, mle_locs = infer.train_mle(halting_rtol = halting_rtol_day2)
    BIC, AIC = infer.compute_IC()
    
    # "----- Q_init for next day"
    # Q_init_day2 = agent.Q[-1].detach().mean(axis=0)[None, ...]
    
    "----- Save parameter names to DataFrame"
    params_sim_df = pd.DataFrame(columns = agent.param_dict.keys())
    for col in params_sim_df.columns:
        params_sim_df[col] = ['unknown']
    
    params_sim_df['ag_idx']  = None
    params_sim_df['group']  = None
    params_sim_df['model']  = model_day2
    
    "----- Store results"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    extra_storage = (Q_init_day2,  # initial Q-values
                     agent.Q[-1].detach(), # final Q-values
                     2, # day
                     model_day1, # preceding model name
                     max_log_like,
                     mle_locs,
                     seq_counter_day2,
                     filename_day1,
                     secondlevel_df) 
    
    if num_inf_steps_day2 > 1:
        pickle.dump( (firstlevel_df, 
                      expdata_df_day2,
                      (infer.loss, BIC, AIC), 
                      params_sim_df, 
                      agent_elbo_tuple, 
                      extra_storage), 
                    open(f"behav_fit/behav_fit_model_day2_{model_day2}_model1_{model_day1}_{timestamp}_{num_agents}agents.p", "wb" ) )
        
        
from IPython import get_ipython
get_ipython().run_line_magic("reset", "-f")