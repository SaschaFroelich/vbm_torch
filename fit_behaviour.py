#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Wed Oct 25 11:05:13 2023

    Fit model to behaviour.

    @author: sascha
"""
from IPython import get_ipython
get_ipython().run_line_magic("reset", "-f")

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

    Vbm_lr - 2 parameters
    Repbias - 3 parameters
    Repbias_Conflict - 4 parameters
    Repbias_Interaction - 4 parameters
    OnlyQ - 4 parameters
    Bullshitmodel - 6 parameters
    Repbias_3Q_lr - 4 parameters
    Repbias_Conflict_Repdiff_lr  - 4 parameters
    OnlyQ_Qdiff_lr - 4 parameters
'''

waithrs = 0
post_pred = 1
STT = 0

model_day1 = 'OnlyQ_Qdiff_lr'
models_day2 = ['OnlyQ_Qdiff_lr']
num_inf_steps_day1 = 3_000
halting_rtol_day1 = 1e-07 # for MLE estimation
posterior_pred_samples_day1 = 3_000
num_waic_samples = 3_000

num_inf_steps_day2 = num_inf_steps_day1
halting_rtol_day2 = halting_rtol_day1 # for MLE estimation
posterior_pred_samples_day2 = posterior_pred_samples_day1

#%%

"Day 1"
exp_behav_dict_day1, expdata_df_day1 = pickle.load(open("behav_data/preproc_data_day1.p", "rb" ))
exp_behav_dict_day2 = utils.RT_err_to_m2(exp_behav_dict_day1)

num_agents = len(expdata_df_day1['ag_idx'].unique())
group = exp_behav_dict_day1['group'][0]

error_df_day1 = anal.compute_errors(expdata_df_day1)
er_day1 = torch.zeros((4, num_agents))
er_day1[0, :] = torch.tensor(error_df_day1['ER_stt']) # stt
er_day1[1, :] = torch.tensor(error_df_day1['ER_randomdtt']) # random
er_day1[2, :] = torch.tensor(error_df_day1['ER_congruent']) # congruent
er_day1[3, :] = torch.tensor(error_df_day1['ER_incongruent']) # incongruent

"Make sure same number of participants in each group"
group_distro = [(np.array(group)==grp).sum() for grp in range(4)]
assert np.abs(np.diff(group_distro)).sum() == 0

"Day 2"
exp_behav_dict_day2, expdata_df_day2 = pickle.load(open("behav_data/preproc_data_day2.p", "rb" ))
exp_behav_dict_day2 = utils.RT_err_to_m2(exp_behav_dict_day2)
num_agents = len(expdata_df_day2['ag_idx'].unique())
group = exp_behav_dict_day2['group'][0]

"Make sure same number of participants in each group"
group_distro = [(np.array(group)==grp).sum() for grp in range(4)]
assert np.abs(np.diff(group_distro)).sum() == 0


'''
    Fit day 1
'''

'''
    Inference
'''
print(f"Starting inference of model {model_day1} for day 1 for {num_agents} agents.")
"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent(model_day1, 
                         group, 
                         num_agents = num_agents)

import time
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Waiting for {waithrs} hours, starting at {timestamp}.")
time.sleep(waithrs*3600)

Q_init_day1 = agent.Q_init

print("===== Starting inference for day 1 =====")
"----- Start Inference"
if STT:
    infer = inferencemodels.GeneralGroupInferenceSTT(agent, exp_behav_dict_day1, 1)
    
else:
    infer = inferencemodels.GeneralGroupInference(agent, exp_behav_dict_day1)
agent_elbo_tuple = infer.infer_posterior(iter_steps = num_inf_steps_day1, num_particles = 10)

"----- Sample parameter estimates from posterior and add information to DataFrame"
if post_pred:
    firstlevel_df, secondlevel_df, predictive_choices, obs_mask = infer.posterior_predictives(n_samples = posterior_pred_samples_day1)
    
else:
    firstlevel_df = infer.sample_posterior(n_samples = posterior_pred_samples_day1)
    secondlevel_df = None
    predictive_choices = None
    obs_mask = None

firstlevel_df['group'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict_day1['group'][0][x])
firstlevel_df['model'] = [model_day1]*len(firstlevel_df)

ID_df = expdata_df_day1.loc[:, ['ID', 'ag_idx']].drop_duplicates()
firstlevel_df['ID'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict_day1['ID'][0][x])

"----- MLE & IC"
max_log_like, mle_locs = infer.train_mle(halting_rtol = halting_rtol_day1)
BIC, AIC, WAIC, ll, WAIC_var = infer.compute_IC(num_samples = num_waic_samples)

"----- Q_init & seqcounter for next day"
seq_counter_day2 = infer.agent.seq_counter.detach()

param_names_day1 = agent.param_names
inf_mean_df = firstlevel_df.loc[:, [*param_names_day1, 
                      'ag_idx', 
                      'ID']].groupby(['ag_idx', 
                                      'ID'], as_index = False).mean()
assert torch.all(torch.tensor(inf_mean_df['ag_idx']) == torch.tensor(exp_behav_dict_day1['ag_idx'][0]))
assert all([inf_mean_df['ID'][i] == exp_behav_dict_day1['ID'][0][i] for i in range(num_agents)])

_, _, _, sim_agent = utils.simulate_data(model_day1, 
                                        num_agents,
                                        group = group,
                                        day = 1,
                                        STT = STT,
                                        params = inf_mean_df.loc[:, [*param_names_day1]],
                                        errorrates = er_day1)

assert sim_agent.Q[-1].shape[0] == 1 and sim_agent.Q[-1].ndim == 3

if STT:
    Q_init_day2 = None
    
else:
    Q_init_day2 = np.squeeze(np.array(sim_agent.Q))[-10:, :, :].mean(axis=0)
    Q_init_day2 = Q_init_day2[None, ...]
    assert Q_init_day2.ndim == 3
    Q_init_day2 = torch.tensor(Q_init_day2)

"----- Save parameter names to DataFrame"
params_sim_df = pd.DataFrame(columns = agent.param_dict.keys())
for col in params_sim_df.columns:
    params_sim_df[col] = ['unknown']

params_sim_df['ag_idx']  = None
params_sim_df['group']  = None
params_sim_df['model']  = model_day1

"----- Store results"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
extra_storage = (Q_init_day1, # 0 (Q_init))
                 agent.Q[-1].detach(), # 1 (Q-final)
                 1, # day # 2 (day)
                 'no preceding model', # 3 (preceding model)
                max_log_like, # 4 (Maximum Log Likelihood)
                mle_locs, # 5 (MLE estimates)
                 '', # 6 (initial seq_counter - only for day 2)
                 '', # 7 (filename day 1 - only for day 2)
                 secondlevel_df, # 8
                 param_names_day1, # 9
                 'behav_fit', # 10
                 halting_rtol_day1, # 11 (halting r_tol)
                 WAIC, # 12
                 ll, # 13
                 predictive_choices, # 14
                 obs_mask, #15
                 WAIC_var) # 16

filename_day1 = f'behav_fit_model_day1_{model_day1}_{timestamp}_{num_agents}agents'
if num_inf_steps_day1 > 1:
    print("Storing results for day 1.")
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
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"======== INFERENCE DAY 2 ({timestamp}) ===========.")
    agent = utils.init_agent(model_day2, 
                             group, 
                             num_agents = num_agents,
                             seq_init = seq_counter_day2)
    
    param_names_day2 = agent.param_names
    print("===== Starting inference for day 2 =====")
    "----- Start Inference"
    if STT:
        infer = inferencemodels.GeneralGroupInferenceSTT(agent, exp_behav_dict_day2, 1)
        
    else:
        infer = inferencemodels.GeneralGroupInference(agent, exp_behav_dict_day2)
        
    agent_elbo_tuple = infer.infer_posterior(iter_steps = num_inf_steps_day2, num_particles = 10)

    "----- Sample parameter estimates from posterior and add information to DataFrame"
    if post_pred:
        firstlevel_df, secondlevel_df, predictive_choices, obs_mask = infer.posterior_predictives(n_samples = posterior_pred_samples_day2)
        
    else:
        firstlevel_df = infer.sample_posterior(n_samples = posterior_pred_samples_day2)
        secondlevel_df = None
        predictive_choices = None
        obs_mask = None
    
    firstlevel_df['group'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict_day2['group'][0][x])
    firstlevel_df['model'] = [model_day2]*len(firstlevel_df)
    
    ID_df = expdata_df_day2.loc[:, ['ID', 'ag_idx']].drop_duplicates()
    firstlevel_df['ID'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict_day2['ID'][0][x])
    
    "----- MLE & IC"
    max_log_like, mle_locs = infer.train_mle(halting_rtol = halting_rtol_day2)
    BIC, AIC, WAIC, ll, WAIC_var = infer.compute_IC(num_samples = num_waic_samples)
    
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
                     '',
                     '',
                     seq_counter_day2, # seq counter day 2
                     filename_day1, # filename day 1
                     secondlevel_df,
                     param_names_day2,
                     'behav_fit',
                     halting_rtol_day2, # halting r_tol
                     WAIC,
                     ll,
                     predictive_choices,
                     obs_mask,
                     WAIC_var)
    
    if num_inf_steps_day2 > 1:
        print("Storing results for day two.")
        pickle.dump( (firstlevel_df, 
                      expdata_df_day2,
                      (infer.loss, BIC, AIC), 
                      params_sim_df, 
                      agent_elbo_tuple, 
                      extra_storage), 
                    open(f"behav_fit/behav_fit_model_day2_{model_day2}_model1_{model_day1}_{timestamp}_{num_agents}agents.p", "wb" ) )
        
        
from IPython import get_ipython
get_ipython().run_line_magic("reset", "-f")

quit()