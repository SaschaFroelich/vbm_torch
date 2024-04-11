#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:26:59 2023

Parameter Recovery 
1. Simulation
2. Group-Level Inference

@author: sascha
"""

from IPython import get_ipython
get_ipython().run_line_magic("reset", "-f")

import utils
import inferencemodels
from datetime import datetime
import pickle
import torch
import numpy as np
import pandas as pd

'''
Modelle:

    Vbm
    Vbm_twodays
    random
    B
    B_oneday
    Conflict
    Seqboost
    Seqboost_nolr
    Bhand
    Bhand_oneday
    BQ
    BK
    HandSeq
    HandSeq2
    Random
    
    OnlyR
    OnlyQ
    
    Repbias_lr
    
    ---- Repdiff Models ----
    Repbias_Conflict_Repdiff_onlyseq_onlyseq_lr
    Repbias_Conflict_Repdiff_onlyseq_onlyseq_nobound_lr
    Repbias_Conflict_Repdiff_onlyseq_both
    Repbias_Conflict_Repdiff_onlyseq_both_nobound
    Repbias_Conflict_Repdiff_onlyseq_lr
    Repbias_Conflict_Repdiff_onlyseq_nobound
    Repbias_Conflict_Repdiff_lr
    Repbias_Conflict_Repdiff_lr_nobound
    
    ---- Conflict Models ----
    Repbias_Conflict_onlyseq_onlyseq_lr
    Repbias_Conflict_onlyseq_onlyseq_nobound_lr
    Repbias_Conflict_onlyseq_both
    Repbias_Conflict_onlyseq_both_nobound
    Repbias_Conflict_both_onlyseq
    Repbias_Conflict_both_onlyseq_nobound
    Repbias_Conflict_both_both
    Repbias_Conflict_both_both_nobound
'''

waithrs = 0
post_pred = 0
sim_model_day1 = 'OnlyQ_Qdiff_onlyseq_lr'
sim_models_day2 = ['Repbias_Conflict_Repdiff_lr']

if 0:
    inf_model_day1 = sim_model_day1
    inf_models_day2 = sim_models_day2
    
else:
    inf_model_day1 = 'Repbias_Conflict_Repdiff_lr'
    inf_models_day2 = ['OnlyQ_Qdiff_onlyseq_lr']

num_agents = 60
num_inf_steps_day1 = 3_000
halting_rtol_day1 = 1e-07 # for MLE estimation
posterior_pred_samples_day1 = 1
num_waic_samples_day1 = 3_000

num_inf_steps_day2 = 1
halting_rtol_day2 = 1e-02 # for MLE estimation
posterior_pred_samples_day2 = 1
num_waic_samples_day2 = 1

#%%
'''
    Simulate data in parallel
'''

group = [0]*(num_agents//4)
group.extend([1]*(num_agents//4))
group.extend([2]*(num_agents//4))
group.extend([3]*(num_agents//4))

# utils.plot_grouplevel(group_behav_df)

'''
    Simulate Data Day 1
'''

day = 1
er_day1 = torch.rand((4, num_agents))*0.2

groupdata_dict_day1, group_behav_df_day1, params_sim_df_day1, agent_day1 = utils.simulate_data(sim_model_day1, 
                                                                      num_agents,
                                                                      group = group,
                                                                      day = day,
                                                                      STT = False,
                                                                      errorrates = er_day1)


#%%
'''
    Check that all other models exist
'''
_, _, _, _ = utils.simulate_data(inf_model_day1, 
                                num_agents,
                                group = group,
                                day = day,
                                STT = False,
                                errorrates = er_day1)

for im2 in inf_models_day2:
    _, _, _, _ = utils.simulate_data(im2, 
                                    num_agents,
                                    group = group,
                                    day = day,
                                    STT = False,
                                    errorrates = er_day1)

for sm2 in sim_models_day2:
    _, _, _, _ = utils.simulate_data(sm2, 
                                    num_agents,
                                    group = group,
                                    day = day,
                                    STT = False,
                                    errorrates = er_day1)


print("\n\n")
print("Sim Model for day 1 is")
print(sim_model_day1)
print("Inf Model for day 1 is")
print(inf_model_day1)

print("Sim Models for day 2 are")
for sm2 in sim_models_day2:
    print(sm2)

print("Inf Models for day 2 are")
for im2 in inf_models_day2:
    print(sm2)
import time
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"\n\nWaiting for {waithrs} hours, starting at {timestamp}.")
time.sleep(waithrs*3600)


'''
    Fit day 1
'''
# import time
# waithrs = 12
# timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# print(f"Waiting for {waithrs} hours, starting at {timestamp}.")
# time.sleep(waithrs*3600)

'''
    Inference
'''
"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent(inf_model_day1, 
                         group, 
                         num_agents = num_agents)

Q_init_day1 = agent.Q_init

print("\n\n===== Starting inference of day 1 =====\n\n")
"----- Start Inference"
infer = inferencemodels.GeneralGroupInference(agent, groupdata_dict_day1)
agent_elbo_tuple, loss = infer.infer_posterior(iter_steps = num_inf_steps_day1, num_particles = 10)

"----- Sample parameter estimates from posterior and add information to DataFrame"
if post_pred:
    firstlevel_df, secondlevel_df, predictive_choices, obs_mask = infer.posterior_predictives(n_samples = posterior_pred_samples_day1)
    
else:
    firstlevel_df = infer.sample_posterior(n_samples = posterior_pred_samples_day1)
    secondlevel_df = None
    predictive_choices = None
    obs_mask = None

firstlevel_df['group'] = firstlevel_df['ag_idx'].map(lambda x: groupdata_dict_day1['group'][0][x])
firstlevel_df['sim_model'] = [sim_model_day1]*len(firstlevel_df)
firstlevel_df['inf_model'] = [inf_model_day1]*len(firstlevel_df)

ID_df = group_behav_df_day1.loc[:, ['ag_idx']].drop_duplicates()

"----- WAIC & DIC"
WAIC, ll, WAIC_var, subject_WAIC, DIC, _, pwaic = infer.compute_WAIC_DIC(num_samples = num_waic_samples_day1)

"----- Q_init & seqcounter for next day"
seq_counter_day2 = infer.agent.seq_counter.detach()
param_names_day1 = agent.param_names
inf_mean_df = firstlevel_df.loc[:, [*param_names_day1, 
                      'ag_idx', 
                      'ID']].groupby(['ag_idx', 
                                      'ID'], as_index = False).mean()
assert torch.all(torch.tensor(inf_mean_df['ag_idx']) == torch.tensor(groupdata_dict_day1['ag_idx'][0]))
assert all([inf_mean_df['ID'][i] == groupdata_dict_day1['ID'][0][i] for i in range(num_agents)])

_, _, _, sim_agent = utils.simulate_data(inf_model_day1, 
                                        num_agents,
                                        group = group,
                                        day = day,
                                        STT = False,
                                        params = inf_mean_df.loc[:, [*param_names_day1]],
                                        errorrates = er_day1)

assert sim_agent.Q[-1].shape[0] == 1 and sim_agent.Q[-1].ndim == 3

Q_init_day2 = torch.zeros((1, num_agents, 4))
idxgenerator = range(10,0,-1)
for lastidx in idxgenerator:
    print(lastidx)
    Q_init_day2 += sim_agent.Q[-lastidx]/len(idxgenerator)

assert Q_init_day2.ndim == 3
assert Q_init_day2.shape[0] == 1
# print("Q_init_day2 starting as")
# print(Q_init_day2)

'''
    Compute MLE & AIC, BIC
'''
max_log_like, mle_locs = infer.train_mle(halting_rtol = halting_rtol_day1)
BIC, AIC = infer.compute_BIC_AIC()

"----- Store results"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
extra_storage = (Q_init_day1, # 0 
                 agent.Q[-1].detach(), # 1
                 1, # 2) day
                 'no preceding model', # 3
                '', # 4
                '', # 5
                 '', # 6
                 '', # 7
                 secondlevel_df, # 8
                 param_names_day1, # 9
                 'recovery', # 10
                 '', # 11
                 WAIC, # 12
                 ll, # 13
                 predictive_choices, # 14
                 obs_mask, # 15
                 WAIC_var, # 16
                 subject_WAIC, # 17
                 DIC, # 18
                 pwaic) # 19

filename_day1 = f'recovery_simmodelday1_{sim_model_day1}_infmodelday1_{inf_model_day1}_day{day}_{timestamp}_{num_agents}agents'

if num_inf_steps_day1 > 1:
    pickle.dump( (firstlevel_df, 
                  group_behav_df_day1,
                  (infer.loss, BIC, AIC), 
                  params_sim_df_day1, 
                  agent_elbo_tuple, 
                  extra_storage), 
                open(f"parameter_recovery/{filename_day1}.p", "wb" ) )

'''
    Fit day 2
'''

'''
    Simulate Data Day 2
'''
day = 2
del groupdata_dict_day1, group_behav_df_day1, params_sim_df_day1, agent_day1
# blocks_day2 = [3,7]
for md2_idx in range(len(sim_models_day2)):
    print("Start day 2.")
    er_day2 = er_day1
    groupdata_dict_day2, group_behav_df_day2, params_sim_df_day2, agent_day2 = utils.simulate_data(sim_models_day2[md2_idx], 
                                                                          num_agents,
                                                                          group = group,
                                                                          day = day,
                                                                          STT = False,
                                                                          Q_init = Q_init_day2.detach(),
                                                                          seq_init = seq_counter_day2,
                                                                          errorrates = er_day2)
    
    agent = utils.init_agent(inf_models_day2[md2_idx], 
                             group, 
                             num_agents = num_agents,
                             Q_init = Q_init_day2.detach(),
                             seq_init = seq_counter_day2)
    
    param_names_day2 = agent.param_names
    
    print("\n\n===== Starting inference of day 2 =====\n\n")
    "----- Start Inference"
    infer = inferencemodels.GeneralGroupInference(agent, groupdata_dict_day2)
    agent_elbo_tuple, loss = infer.infer_posterior(iter_steps = num_inf_steps_day2, num_particles = 10)
    
    "----- Sample parameter estimates from posterior and add information to DataFrame"
    if post_pred:
        firstlevel_df, secondlevel_df, predictive_choices, obs_mask = infer.posterior_predictives(n_samples = posterior_pred_samples_day2)
        
    else:
        firstlevel_df = infer.sample_posterior(n_samples = posterior_pred_samples_day2)
        secondlevel_df = None
        
    firstlevel_df['group'] = firstlevel_df['ag_idx'].map(lambda x: groupdata_dict_day2['group'][0][x])
    firstlevel_df['inf_model'] = [inf_models_day2[md2_idx]]*len(firstlevel_df)
    firstlevel_df['sim_model'] = [sim_models_day2[md2_idx]]*len(firstlevel_df)
    
    ID_df = group_behav_df_day2.loc[:, ['ag_idx']].drop_duplicates()
    # firstlevel_df['ID'] = firstlevel_df['ag_idx'].map(lambda x: groupdata_dict_day2['ID'][0][x])
    
    "----- WAIC & DIC"
    WAIC, ll, WAIC_var, subject_WAIC, DIC, _, pwaic = infer.compute_WAIC_DIC(num_samples = num_waic_samples_day1)
    
    # "----- Q_init for next day"
    # Q_init_day2 = agent.Q[-1].detach().mean(axis=0)[None, ...]
    
    
    '''
        Compute MLE & AIC, BIC
    '''
    max_log_like, mle_locs = infer.train_mle(halting_rtol = halting_rtol_day2)
    BIC, AIC = infer.compute_BIC_AIC()
    
    "----- Store results"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    extra_storage = (Q_init_day2,  # 1) initial Q-values
                     agent.Q[-1].detach(), # 2) final Q-values
                     2, # 3) day
                     sim_model_day1, # 4) preceding sim model
                     '', # 5)
                     '', # 6)
                     seq_counter_day2, # 7) seq counter day 2
                     filename_day1, # 8) filename day 1
                     secondlevel_df, # 9)
                     param_names_day2, # 10)
                     'recovery', # 11)
                     '', # 12)
                     WAIC, # 13)
                     ll, # 14)
                     predictive_choices, # 15)
                     obs_mask, # 16)
                     WAIC_var, # 17
                     subject_WAIC, # 18 
                     DIC, # 19
                     pwaic) # 20
    
    if num_inf_steps_day2 > 1:
        pickle.dump( (firstlevel_df, 
                      group_behav_df_day2, 
                      (infer.loss, BIC, AIC), 
                      params_sim_df_day2, 
                      agent_elbo_tuple, 
                      extra_storage), 
                    open(f"parameter_recovery/recovery_simmodelday2_{sim_models_day2[md2_idx]}_infmodelday2_{inf_models_day2[md2_idx]}_simmodel1_{sim_model_day1}_day{day}_{timestamp}_{num_agents}agents.p", "wb" ) )

print("Done.")
from IPython import get_ipython
get_ipython().run_line_magic("reset", "-f")

quit()