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
    OnlyQ_Qdiff_lr - 4 parameters
    OnlyQ_Qdiff_onlyseq_lr
    
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

import tracemalloc
tracemalloc.start()

waithrs = 0
saveQ = True
post_pred = 1
STT = 0

import sys

models = ['Repbias_Conflict_both_onlyseq_inferinc']

# models = ['Repbias_lr',
#                 'Repbias_Conflict_Repdiff_onlyseq_lr_inferinc',
#                 'Repbias_Conflict_both_onlyseq_inferinc',
#                 'OnlyQ_Qdiff_noswitch_onlyseq_onlyseq',
#                 'OnlyQ_Qdiff_noswitch_onlyseq_onlyseq_DQ',
#                 'OnlyQ_Qdiff_onlyseq_lr_C', 
#                 'OnlyQ_Qdiff_onlyseq_lr_D',
#                 'OnlyQ_lr']

seqlength_from = 3
seqlength_to = 3

num_inf_steps = 7_000
halting_rtol = 1e-07 # for MLE estimation
if saveQ:
    posterior_pred_samples = 100 # For Q-values
else:
    posterior_pred_samples = 1
    
num_waic_samples = 3_000

#%%
if 0:
    "Day 1"
    datafile_day1 = 'preproc_data_day1.p'
    exp_behav_dict_day1, expdata_df_day1 = pickle.load(open(f"behav_data/{datafile_day1}", "rb" ))
    exp_behav_dict_day1 = utils.RT_err_to_m2(exp_behav_dict_day1)
    
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
    datafile_day2 = 'preproc_data_day2.p'
    exp_behav_dict_day2, expdata_df_day2 = pickle.load(open(f"behav_data/{datafile_day2}", "rb" ))
    exp_behav_dict_day2 = utils.RT_err_to_m2(exp_behav_dict_day2)
    num_agents = len(expdata_df_day2['ag_idx'].unique())
    group = exp_behav_dict_day2['group'][0]
    
    "Make sure same number of participants in each group"
    group_distro = [(np.array(group)==grp).sum() for grp in range(4)]
    assert np.abs(np.diff(group_distro)).sum() == 0

"Both days"
datafile = 'preproc_data.p'
exp_behav_dict, expdata_df = pickle.load(open(f"behav_data/{datafile}", "rb" ))
exp_behav_dict = utils.RT_err_to_m2(exp_behav_dict)
num_agents = len(expdata_df['ag_idx'].unique())
group = exp_behav_dict['group'][0]

"Make sure same number of participants in each group"
group_distro = [(np.array(group)==grp).sum() for grp in range(4)]
assert np.abs(np.diff(group_distro)).sum() == 0

import time
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Waiting for {waithrs} hours, starting at {timestamp}.")
time.sleep(waithrs*3600)

'''
    - Set up agent
    - Inference
'''
for model in models:
    print(f"Starting inference of model {model} (datafiles {datafile}) for {num_agents} agents.")
    "----- Initialize new agent object with num_agents agents for inference"
    
    for seqlength in range(seqlength_from, seqlength_to+1):
        agent = utils.init_agent(model, 
                                 group, 
                                 num_agents = num_agents,
                                 seqlength = seqlength)
        
        infer = inferencemodels.GeneralGroupInference(agent, exp_behav_dict)
        
        Q_init = agent.Q_init
        
        print(f"===== Starting inference with seqlength {seqlength} =====")
        "----- Start Inference"
        agent_elbo_tuple, loss = infer.infer_posterior(iter_steps = num_inf_steps, num_particles = 10)
        
        "----- Sample parameter estimates from posterior and add information to DataFrame"
        if post_pred:
            firstlevel_df, secondlevel_df, predictive_choices, obs_mask, Qvalues = infer.posterior_predictives(n_samples = posterior_pred_samples,
                                                                                                      saveQ = saveQ)
            
        else:
            firstlevel_df = infer.sample_posterior(n_samples = posterior_pred_samples)
            secondlevel_df = None
            predictive_choices = None
            obs_mask = None
        
        firstlevel_df['group'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict['group'][0][x])
        firstlevel_df['model'] = [model]*len(firstlevel_df)
        
        ID_df = expdata_df.loc[:, ['ID', 'ag_idx']].drop_duplicates()
        firstlevel_df['ID'] = firstlevel_df['ag_idx'].map(lambda x: exp_behav_dict['ID'][0][x])
        
        "----- WAIC & DIC"
        WAIC, _, WAIC_var, individual_WAIC, DIC, loglike, pwaic, individual_DIC = infer.compute_WAIC_DIC(num_samples = num_waic_samples)
        
        param_names = agent.param_names
        
        if 0:
            '''
                Q_init & seqcounter for next day
            '''
            seq_counter_day2 = infer.agent.seq_counter.detach()
            
            inf_mean_df = firstlevel_df.loc[:, [*param_names, 
                                  'ag_idx', 
                                  'ID']].groupby(['ag_idx', 
                                                  'ID'], as_index = False).mean()
            assert torch.all(torch.tensor(inf_mean_df['ag_idx']) == torch.tensor(exp_behav_dict['ag_idx'][0]))
            assert all([inf_mean_df['ID'][i] == exp_behav_dict['ID'][0][i] for i in range(num_agents)])
            
            _, _, _, sim_agent = utils.simulate_data(model, 
                                                    num_agents,
                                                    group = group,
                                                    day = 1,
                                                    STT = STT,
                                                    params = inf_mean_df.loc[:, [*param_names]],
                                                    errorrates = er)
            
            del er
            assert sim_agent.Q[-1].shape[0] == 1 and sim_agent.Q[-1].ndim == 3
            
            if STT:
                Q_init_day2 = None
                
            else:
                Q_init_day2 = torch.zeros((1, num_agents, 4))
                idxgenerator = range(10,0,-1)
                for lastidx in idxgenerator:
                    print(lastidx)
                    Q_init_day2 += sim_agent.Q[-lastidx]/len(idxgenerator)
                
                assert Q_init_day2.ndim == 3
                assert Q_init_day2.shape[0] == 1
            
            print("Q_init_day2 starting as")
            print(Q_init_day2)
            
        "----- Save parameter names to DataFrame"
        params_sim_df = pd.DataFrame(columns = agent.param_dict.keys())
        for col in params_sim_df.columns:
            params_sim_df[col] = ['unknown']
        
        params_sim_df['ag_idx']  = None
        params_sim_df['group']  = None
        params_sim_df['model']  = model
        
        
        '''
            Compute MLE & AIC, BIC
        '''
        max_log_like, mle_locs = infer.train_mle(halting_rtol = halting_rtol)
        BIC, AIC = infer.compute_BIC_AIC()
        
        "----- Store results"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        extra_storage = (Q_init, # 0 (Q_init))
                         agent.Q[-1].detach(), # 1 (Q-final)
                         3, # day # 2 (day) (3 == bothdays)
                         'no preceding model', # 3 (preceding model)
                        '', # 4 (Maximum Log Likelihood)
                        '', # 5 (MLE estimates)
                         '', # 6 (initial seq_counter - only for day 2)
                         '', # 7 (filename day 1 - only for day 2)
                         secondlevel_df, # 8
                         param_names, # 9
                         'behav_fit', # 10
                         halting_rtol, # 11 (halting r_tol)
                         WAIC, # 12
                         None, # 13 (loglike 2D)
                         predictive_choices, # 14
                         obs_mask, # 15
                         WAIC_var, # 16
                         individual_WAIC, # 17
                         DIC, # 18
                         pwaic, # 19
                         individual_DIC,
                         '') # 20
        
        filename = f'BehavFitModelBothDays_{model}_{timestamp}_{num_agents}agents_SeqLength_{seqlength}'
        if num_inf_steps > 1:
            print("Storing results.")
            pickle.dump( (firstlevel_df, 
                          expdata_df,
                          (loss, BIC, AIC), 
                          params_sim_df, 
                          agent_elbo_tuple, 
                          extra_storage), 
                        open(f"behav_fit/{filename}.p", "wb" ) )
            
            print("Storing Q-values.")
            Qvalues['Q1'] = Qvalues['Q1'].map(lambda x: x.item())
            Qvalues['Q2'] = Qvalues['Q2'].map(lambda x: x.item())
            Qvalues['Q3'] = Qvalues['Q3'].map(lambda x: x.item())
            Qvalues['Q4'] = Qvalues['Q4'].map(lambda x: x.item())
            
            Qvalues.to_csv(f'behav_fit/{filename}_Qvalues.csv')
            
            print("Saving loglike.")
            assert loglike.ndim == 3
            assert loglike.shape[0] == num_waic_samples
            assert loglike.shape[-1] == num_agents
            num_loglike_entries = loglike.shape[1]
        
            column_indices = torch.arange(num_agents)
            agent_indexer = column_indices.unsqueeze(0).repeat(num_waic_samples, loglike.shape[1], 1)
                
            loglike = torch.reshape(loglike, (num_waic_samples, num_loglike_entries*num_agents))
            agent_indexer = torch.reshape(agent_indexer, (num_waic_samples, num_loglike_entries*num_agents))
            
            nan_mask = torch.isnan(loglike)
            all_nan_columns = nan_mask.all(dim=0)
            
            loglike = loglike[:, ~all_nan_columns]
            agent_indexer = agent_indexer[:, ~all_nan_columns]
            agent_indexer = agent_indexer[0, :]
            
            pickle.dump( (loglike, agent_indexer, num_inf_steps), 
                        open(f"behav_fit/IC/{filename}_loglike.p", "wb" ) )
            
            
            print("Done.")
            # del loglike, agent_indexer
        
from IPython import get_ipython
get_ipython().run_line_magic("reset", "-f")
quit()