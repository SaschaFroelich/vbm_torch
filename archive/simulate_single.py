#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:36:19 2023

For remote parameter recovery.

@author: sascha
"""

import ipdb

import pandas as pd
import matplotlib.pylab as plt

import numpy

import models_torch as models
import env 

import sys
from datetime import datetime
import os
import pickle

plt.style.use("classic")

#%%
model = sys.argv[1]
resim =  int(sys.argv[2]) # whether to simulate agents with inferred parameters
method = sys.argv[3] # "svi" or "mcmc"
k = float(sys.argv[4])

if resim:
    df = pickle.load( open(f"model{model}_param_anal.p", "rb" ) )
    pb_idx = sys.argv[3]

if method == 'svi':
    savedir = f"param_recov/model{model}/k_{k}/"

elif method == 'mcmc':
    savedir = f"param_recov/mcmc/model{model}/k_{k}/"

#%%
"Check if there are already files in the savedir directory, and arrange prefix for filenames"

num_versions = 0
if len(os.listdir(savedir)) > 0:
    "Already some files"
    "Check how many different inference 'versions' there are"
    
    for file in os.listdir(savedir):
         filename = os.fsdecode(file)
         if filename.endswith(".p") and "prior" in filename:
             num_versions += 1
             
current_version = num_versions + 1

savedir = savedir + f"version{current_version}_"

#%%

"Simulations for Parameter Recovery (Single Inference)"

if model == 'original':
    npar = 3
    
    if resim:
        dectemp = df.iloc[pb_idx, :].inf_dectemp
        lr = df.iloc[pb_idx, :].inf_lr
        omega = df.iloc[pb_idx, :].inf_omega
        
    else:
        "Simulate with random parameters"
        parameter = numpy.random.uniform(0,1, npar)
        dectemp = parameter[0]*5
        lr = parameter[1]
        omega = parameter[2]
    
    newagent = models.Vbm(dectemp = dectemp, \
                          lr = lr, \
                          omega = omega, \
                          k=k,\
                          Q_init=[0.2, 0., 0., 0.2])

elif model == 'B' or model == 'B_onlydual':
    npar = 6
    
    if resim:
        theta_rep_day1 = df.iloc[pb_idx, :].theta_rep_day1
        theta_rep_day2 = df.iloc[pb_idx, :].theta_rep_day2
        theta_Q_day1 = df.iloc[pb_idx, :].theta_Q_day1
        theta_Q_day2 = df.iloc[pb_idx, :].theta_Q_day2
        lr_day1 = df.iloc[pb_idx, :].lr_day1
        lr_day2 = df.iloc[pb_idx, :].lr_day2
        
    else:
        "Simulate with random parameters"
        parameter = numpy.random.uniform(0,1, npar)
        theta_rep_day1 = parameter[0]*6
        theta_rep_day2 = parameter[1]*6
        theta_Q_day1 = parameter[2]*6
        theta_Q_day2 = parameter[3]*6
        lr_day1 = parameter[4]*0.1
        lr_day2 = parameter[5]*0.1

    if model == 'B':
        newagent = models.Vbm_B(theta_rep_day1 = theta_rep_day1, \
                              theta_rep_day2 = theta_rep_day2, \
                              lr_day1 = lr_day1, \
                              lr_day2 = lr_day2, \
                              theta_Q_day1 = theta_Q_day1, \
                              theta_Q_day2 = theta_Q_day2, \
                              k=k,\
                              Q_init=[0.2, 0., 0., 0.2])
            
    elif model == 'B_onlydual':
        newagent = models.Vbm_B_onlydual(theta_rep_day1 = theta_rep_day1, \
                              theta_rep_day2 = theta_rep_day2, \
                              lr_day1 = lr_day1, \
                              lr_day2 = lr_day2, \
                              theta_Q_day1 = theta_Q_day1, \
                              theta_Q_day2 = theta_Q_day2, \
                              k=k,\
                              Q_init=[0.2, 0., 0., 0.2])

elif model == 'B_2':
    npar = 9
    
    if resim:
        pass
        
    else:
        "Simulate with random parameters"
        parameter = numpy.random.uniform(0,1, npar)
        theta_rep_day1_1 = parameter[0]*6
        theta_rep_day1_2 = parameter[1]*6
        theta_rep_day2 = parameter[2]*6
        theta_Q_day1_1 = parameter[3]*6
        theta_Q_day1_2 = parameter[4]*6
        theta_Q_day2 = parameter[5]*6
        lr_day1_1 = parameter[6]*0.1
        lr_day1_2 = parameter[7]*0.1
        lr_day2 = parameter[8]*0.1

    newagent = models.Vbm_B_2(theta_rep_day1_1 = theta_rep_day1_1, \
                            theta_rep_day1_2 = theta_rep_day1_2, \
                          theta_rep_day2 = theta_rep_day2, \
                          lr_day1_1 = lr_day1_1, \
                          lr_day1_2 = lr_day1_2, \
                          lr_day2 = lr_day2, \
                          theta_Q_day1_1 = theta_Q_day1_1, \
                          theta_Q_day1_2 = theta_Q_day1_2, \
                          theta_Q_day2 = theta_Q_day2, \
                          k=k,\
                          Q_init=[0.2, 0., 0., 0.2])
        
elif model == 'B_3':
    npar = 6
    
    if resim:
        pass
        
    else:
        "Simulate with random parameters"
        parameter = numpy.random.uniform(0,1, npar)
        theta_rep_day1_1 = parameter[0]*6
        theta_rep_day1_2 = parameter[1]*6
        theta_rep_day2 = parameter[2]*6
        theta_Q_day1_1 = parameter[3]*6
        theta_Q_day1_2 = parameter[4]*6
        theta_Q_day2 = parameter[5]*6


    newagent = models.Vbm_B_3(theta_rep_day1_1 = theta_rep_day1_1, \
                            theta_rep_day1_2 = theta_rep_day1_2, \
                          theta_rep_day2 = theta_rep_day2, \
                          theta_Q_day1_1 = theta_Q_day1_1, \
                          theta_Q_day1_2 = theta_Q_day1_2, \
                          theta_Q_day2 = theta_Q_day2, \
                          k=k,\
                          Q_init=[0.8, 0., 0., 0.8])

elif model == 'F':
    "No lr, theta q and thetar_r are linearly time-dependent"
    npar = 8
    parameter = numpy.random.uniform(0,1, npar)
    
    # theta_rep0_day1 = parameter[0]*6
    theta_rep0_day1 = 0
    theta_replambda_day1 = parameter[1]*0.02
    # theta_Q0_day1 = parameter[2]*6
    theta_Q0_day1 = 0
    theta_Qlambda_day1 = parameter[3]*0.02
    
    theta_rep0_day2 = parameter[4]*6
    theta_replambda_day2 = (parameter[5]-0.5)*0.02
    theta_Q0_day2 = parameter[6]*6
    theta_Qlambda_day2 = (parameter[7]-0.5)*0.02

    newagent = models.Vbm_F(theta_rep0_day1 = theta_rep0_day1, \
                          theta_replambda_day1 = theta_replambda_day1, \
                          theta_Q0_day1 = theta_Q0_day1, \
                          theta_Qlambda_day1 = theta_Qlambda_day1, \
                          theta_rep0_day2 = theta_rep0_day2, \
                          theta_replambda_day2 = theta_replambda_day2, \
                          theta_Q0_day2 = theta_Q0_day2, \
                          theta_Qlambda_day2 = theta_Qlambda_day2, \
                          k=k,\
                          Q_init=[0.8, 0.2, 0.2, 0.8])

#%%

newenv = env.Env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir = './matlabcode/clipre/')

newenv.run()
data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
        "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
        "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}

#%%
if model == 'original':
    infer = models.SingleInference(newagent, data)
    loss, params = infer.infer_posterior(iter_steps=150, num_particles = 10)
        
    inf_dectemp = params["dectemp_conc"][0]/params["dectemp_rate"][0]
    dectemp_variance = models.gamma_variance(params["dectemp_conc"][0], params["dectemp_rate"][0])
    
    inf_lr = params["lr_alpha"][0]/(params["lr_alpha"][0]+params["lr_beta"][0])
    lr_variance = models.beta_variance(params["lr_alpha"][0], params["lr_beta"][0])
    
    inf_omega = params["omega_alpha"][0]/(params["omega_alpha"][0]+params["omega_beta"][0])
    omega_variance = models.beta_variance(params["omega_alpha"][0], params["omega_beta"][0])

    pickle_data = {"dectemp" : [dectemp],\
                   "inf_dectemp" : [inf_dectemp], \
                   "dectemp_variance" : [dectemp_variance], \
                   "lr" : [lr], \
                   "inf_lr" : [inf_lr], \
                   "lr_variance" : [lr_variance], \
                   "omega" : [omega], \
                   "inf_omega" : [inf_omega], \
                   "omega_variance" : [omega_variance], \
                    "model" : [model]}

    df = pd.DataFrame (data=pickle_data)

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    random_num = numpy.random.rand()
    
elif model == 'B' or model == 'B_onlydual':
    print("Starting inference for model B")
    infer = models.SingleInference_modelB(newagent, data, k=k)
    
    if method == 'svi':
        loss, params = infer.infer_posterior(iter_steps=150, num_particles = 10)
        
        " ========================= Save Priors ========================"
        lr_day1_alpha = infer.lr_day1_alpha
        lr_day1_beta = infer.lr_day1_beta
        
        theta_Q_day1_conc = infer.theta_Q_day1_conc
        theta_Q_day1_rate = infer.theta_Q_day1_rate
        
        theta_rep_day1_conc = infer.theta_rep_day1_conc
        theta_rep_day1_rate = infer.theta_rep_day1_rate
        
        lr_day2_alpha = infer.lr_day2_alpha
        lr_day2_beta = infer.lr_day2_beta
        
        theta_Q_day2_conc = infer.theta_Q_day2_conc
        theta_Q_day2_rate = infer.theta_Q_day2_rate
        
        theta_rep_day2_conc = infer.theta_rep_day2_conc
        theta_rep_day2_rate = infer.theta_rep_day2_rate
        
        prior_data = {'lr_day1_alpha' : [lr_day1_alpha], \
                'lr_day1_beta' : [lr_day1_beta], \
                    
                'theta_Q_day1_conc' : [theta_Q_day1_conc],\
                'theta_Q_day1_rate' : [theta_Q_day1_rate], \
                    
                'theta_rep_day1_conc' : [theta_rep_day1_conc],\
                'theta_rep_day1_rate' : [theta_rep_day1_rate], \
                    
                'lr_day2_alpha' : [lr_day2_alpha], \
                'lr_day2_beta' : [lr_day2_beta], \
                    
                'theta_Q_day2_conc' : [theta_Q_day2_conc],\
                'theta_Q_day2_rate' : [theta_Q_day2_rate], \
                    
                'theta_rep_day2_conc' : [theta_rep_day2_conc],\
                'theta_rep_day2_rate' : [theta_rep_day2_rate],\
                    
                'Q_init' : [infer.agent.Q_init]}
        
        " ========================= Inference Results ========================"
        "--- Day 1 ---"
        inf_lr_day1 = params["lr_day1_alpha"][0]/(params["lr_day1_alpha"][0]+params["lr_day1_beta"][0])
        lr_day1_variance = models.beta_variance(params["lr_day1_alpha"][0], params["lr_day1_beta"][0])
        
        inf_theta_Q_day1 = params["theta_Q_day1_conc"][0]/params["theta_Q_day1_rate"][0]
        theta_Q_day1_variance = models.gamma_variance(params["theta_Q_day1_conc"][0], params["theta_Q_day1_rate"][0])
        
        inf_theta_rep_day1 = params["theta_rep_day1_conc"][0]/params["theta_rep_day1_rate"][0]
        theta_rep_day1_variance = models.gamma_variance(params["theta_rep_day1_conc"][0], params["theta_rep_day1_rate"][0])
        
        "--- Day 2 ---"
        inf_lr_day2 = params["lr_day2_alpha"][0]/(params["lr_day2_alpha"][0]+params["lr_day2_beta"][0])
        lr_day2_variance = models.beta_variance(params["lr_day2_alpha"][0], params["lr_day2_beta"][0])
        
        inf_theta_Q_day2 = params["theta_Q_day2_conc"][0]/params["theta_Q_day2_rate"][0]
        theta_Q_day2_variance = models.gamma_variance(params["theta_Q_day2_conc"][0], params["theta_Q_day2_rate"][0])
        
        inf_theta_rep_day2 = params["theta_rep_day2_conc"][0]/params["theta_rep_day2_rate"][0]
        theta_rep_day2_variance = models.gamma_variance(params["theta_rep_day2_conc"][0], params["theta_rep_day2_rate"][0])
                
        pickle_data = {'lr_day1' : [lr_day1], \
                       'inf_lr_day1' : [inf_lr_day1], \
                       'lr_day1_variance' : [lr_day1_variance], \
                           
                       'theta_Q_day1' : [theta_Q_day1],\
                       'inf_theta_Q_day1' : [inf_theta_Q_day1], \
                       'theta_Q_day1_variance' : [theta_Q_day1_variance], \
                           
                       'theta_rep_day1' : [theta_rep_day1],\
                       'inf_theta_rep_day1' : [inf_theta_rep_day1], \
                       'theta_rep_day1_variance' : [theta_rep_day1_variance], \
                           
                       'lr_day2' : [lr_day2], \
                       'inf_lr_day2' : [inf_lr_day2], \
                       'lr_day2_variance' : [lr_day2_variance], \
                           
                       'theta_Q_day2' : [theta_Q_day2],\
                       'inf_theta_Q_day2' : [inf_theta_Q_day2], \
                       'theta_Q_day2_variance' : [theta_Q_day2_variance], \
                           
                       'theta_rep_day2' : [theta_rep_day2],\
                       'inf_theta_rep_day2' : [inf_theta_rep_day2], \
                       'theta_rep_day2_variance' : [theta_rep_day2_variance]}
        
    elif method == 'mcmc':
        mcmc_res = infer.mcmc(num_samples = 10_000, burnin = 1_000)
        
        "--- Day 1 ---"
        inf_lr_day1 = mcmc_res.get_samples()['lr_day1'].mean().item()
        lr_day1_variance = mcmc_res.get_samples()['lr_day1'].var().item()
        lr_day1_rhat = mcmc_res.diagnostics()["lr_day1"]["r_hat"].item()
        
        inf_theta_Q_day1 = mcmc_res.get_samples()['theta_Q_day1'].mean().item()
        theta_Q_day1_variance = mcmc_res.get_samples()['theta_Q_day1'].var().item()
        theta_Q_day1_rhat = mcmc_res.diagnostics()["theta_Q_day1"]["r_hat"].item()
        
        inf_theta_rep_day1 = mcmc_res.get_samples()['theta_rep_day1'].mean().item()
        theta_rep_day1_variance = mcmc_res.get_samples()['theta_rep_day1'].var().item()
        theta_rep_day1_rhat = mcmc_res.diagnostics()["theta_rep_day1"]["r_hat"].item()
        
        "--- Day 2 ---"
        inf_lr_day2 = mcmc_res.get_samples()['lr_day2'].mean().item()
        lr_day2_variance = mcmc_res.get_samples()['lr_day2'].var().item()
        lr_day2_rhat = mcmc_res.diagnostics()["lr_day2"]["r_hat"].item()
        
        inf_theta_Q_day2 = mcmc_res.get_samples()['theta_Q_day2'].mean().item()
        theta_Q_day2_variance = mcmc_res.get_samples()['theta_Q_day2'].var().item()
        theta_Q_day2_rhat = mcmc_res.diagnostics()["theta_Q_day2"]["r_hat"].item()
        
        inf_theta_rep_day2 = mcmc_res.get_samples()['theta_rep_day2'].mean().item()
        theta_rep_day2_variance = mcmc_res.get_samples()['theta_rep_day2'].var().item()
        theta_rep_day2_rhat = mcmc_res.diagnostics()["theta_rep_day2"]["r_hat"].item()
        
        pickle_data = {'lr_day1' : [lr_day1], \
                       'inf_lr_day1' : [inf_lr_day1], \
                       'lr_day1_variance' : [lr_day1_variance], \
                       'lr_day1_rhat' : [lr_day1_rhat], \
                           
                       'theta_Q_day1' : [theta_Q_day1],\
                       'inf_theta_Q_day1' : [inf_theta_Q_day1], \
                       'theta_Q_day1_variance' : [theta_Q_day1_variance], \
                       'theta_Q_day1_rhat' : [theta_Q_day1_rhat],\
                           
                       'theta_rep_day1' : [theta_rep_day1],\
                       'inf_theta_rep_day1' : [inf_theta_rep_day1], \
                       'theta_rep_day1_variance' : [theta_rep_day1_variance], \
                       'theta_rep_day1_rhat' : [theta_rep_day1_rhat],\
                           
                       'lr_day2' : [lr_day2], \
                       'inf_lr_day2' : [inf_lr_day2], \
                       'lr_day2_variance' : [lr_day2_variance], \
                       'lr_day2_rhat' : [lr_day2_rhat], \
                           
                       'theta_Q_day2' : [theta_Q_day2],\
                       'inf_theta_Q_day2' : [inf_theta_Q_day2], \
                       'theta_Q_day2_variance' : [theta_Q_day2_variance], \
                       'theta_Q_day2_rhat' : [theta_Q_day2_rhat],\
                           
                       'theta_rep_day2' : [theta_rep_day2],\
                       'inf_theta_rep_day2' : [inf_theta_rep_day2], \
                       'theta_rep_day2_variance' : [theta_rep_day2_variance], \
                       'theta_rep_day2_rhat' : [theta_rep_day2_rhat]}
        
    else:
        raise Exception("Need to specify method.")
        
    prior_df = pd.DataFrame(data = prior_data)        
    df = pd.DataFrame(data = pickle_data)
        
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    random_num = numpy.random.rand()
    
elif model == 'B_2':
    print("Starting inference for model B_2")
    infer = models.SingleInference_modelB_2(newagent, data, k = k)
    loss, params = infer.infer_posterior(iter_steps=150, num_particles = 10)
    
    " ========================= Save Priors ========================= "
    lr_day1_1_alpha = infer.lr_day1_1_alpha
    lr_day1_1_beta = infer.lr_day1_1_beta
    
    "Θ_Q"
    theta_Q_day1_1_conc = infer.theta_Q_day1_1_conc
    theta_Q_day1_1_rate = infer.theta_Q_day1_1_rate
    
    "Θ_rep"
    theta_rep_day1_1_conc = infer.theta_rep_day1_1_conc
    theta_rep_day1_1_rate = infer.theta_rep_day1_1_rate
    
    "--- Day 1 : 2 ---"
    "Learning Rate: Beta Distribution"
    lr_day1_2_alpha = infer.lr_day1_2_alpha
    lr_day1_2_beta = infer.lr_day1_2_beta
    
    "Θ_Q"
    theta_Q_day1_2_conc = infer.theta_Q_day1_2_conc
    theta_Q_day1_2_rate = infer.theta_Q_day1_2_rate
    
    "Θ_rep"
    theta_rep_day1_2_conc = infer.theta_rep_day1_2_conc
    theta_rep_day1_2_rate = infer.theta_rep_day1_2_rate
    
    "--- Day 2 ---"
    "Learning Rate: Beta Distribution"
    lr_day2_alpha = infer.lr_day2_alpha
    lr_day2_beta = infer.lr_day2_beta
    
    "Θ_Q"
    theta_Q_day2_conc = infer.theta_Q_day2_conc
    theta_Q_day2_rate = infer.theta_Q_day2_rate
    
    "Θ_rep"
    theta_rep_day2_conc = infer.theta_rep_day2_conc
    theta_rep_day2_rate = infer.theta_rep_day2_rate
    
    prior_data = {'lr_day1_1_alpha' : [lr_day1_1_alpha], \
            'lr_day1_1_beta' : [lr_day1_1_beta], \
                
            'theta_Q_day1_1_conc' : [theta_Q_day1_1_conc],\
            'theta_Q_day1_1_rate' : [theta_Q_day1_1_rate], \
                
            'theta_rep_day1_1_conc' : [theta_rep_day1_1_conc],\
            'theta_rep_day1_1_rate' : [theta_rep_day1_1_rate], \
                
            'lr_day1_2_alpha' : [lr_day1_2_alpha], \
            'lr_day1_2_beta' : [lr_day1_2_beta], \
            
            'theta_Q_day1_2_conc' : [theta_Q_day1_2_conc],\
            'theta_Q_day1_2_rate' : [theta_Q_day1_2_rate], \
            
            'theta_rep_day1_2_conc' : [theta_rep_day1_2_conc],\
            'theta_rep_day1_2_rate' : [theta_rep_day1_2_rate], \
                
            'lr_day2_alpha' : [lr_day2_alpha], \
            'lr_day2_beta' : [lr_day2_beta], \
                
            'theta_Q_day2_conc' : [theta_Q_day2_conc],\
            'theta_Q_day2_rate' : [theta_Q_day2_rate], \
                
            'theta_rep_day2_conc' : [theta_rep_day2_conc],\
            'theta_rep_day2_rate' : [theta_rep_day2_rate],\
                
            'Q_init': [infer.agent.Q_init]}
    
    " ========================= Inference Results ======================== "
    "--- Day 1 : 1 ---"
    inf_lr_day1_1 = params["lr_day1_1_alpha"][0]/(params["lr_day1_1_alpha"][0]+params["lr_day1_1_beta"][0])
    lr_day1_1_variance = models.beta_variance(params["lr_day1_1_alpha"][0], params["lr_day1_1_beta"][0])
    
    inf_theta_Q_day1_1 = params["theta_Q_day1_1_conc"][0]/params["theta_Q_day1_1_rate"][0]
    theta_Q_day1_1_variance = models.gamma_variance(params["theta_Q_day1_1_conc"][0], params["theta_Q_day1_1_rate"][0])
    
    inf_theta_rep_day1_1 = params["theta_rep_day1_1_conc"][0]/params["theta_rep_day1_1_rate"][0]
    theta_rep_day1_1_variance = models.gamma_variance(params["theta_rep_day1_1_conc"][0], params["theta_rep_day1_1_rate"][0])
    
    "--- Day 1 : 2 ---"
    inf_lr_day1_2 = params["lr_day1_2_alpha"][0]/(params["lr_day1_2_alpha"][0]+params["lr_day1_2_beta"][0])
    lr_day1_2_variance = models.beta_variance(params["lr_day1_2_alpha"][0], params["lr_day1_2_beta"][0])
    
    inf_theta_Q_day1_2 = params["theta_Q_day1_2_conc"][0]/params["theta_Q_day1_2_rate"][0]
    theta_Q_day1_2_variance = models.gamma_variance(params["theta_Q_day1_2_conc"][0], params["theta_Q_day1_2_rate"][0])
    
    inf_theta_rep_day1_2 = params["theta_rep_day1_2_conc"][0]/params["theta_rep_day1_2_rate"][0]
    theta_rep_day1_2_variance = models.gamma_variance(params["theta_rep_day1_2_conc"][0], params["theta_rep_day1_2_rate"][0])
    
    "--- Day 2 ---"
    inf_lr_day2 = params["lr_day2_alpha"][0]/(params["lr_day2_alpha"][0]+params["lr_day2_beta"][0])
    lr_day2_variance = models.beta_variance(params["lr_day2_alpha"][0], params["lr_day2_beta"][0])
    
    inf_theta_Q_day2 = params["theta_Q_day2_conc"][0]/params["theta_Q_day2_rate"][0]
    theta_Q_day2_variance = models.gamma_variance(params["theta_Q_day2_conc"][0], params["theta_Q_day2_rate"][0])
    
    inf_theta_rep_day2 = params["theta_rep_day2_conc"][0]/params["theta_rep_day2_rate"][0]
    theta_rep_day2_variance = models.gamma_variance(params["theta_rep_day2_conc"][0], params["theta_rep_day2_rate"][0])
    
    pickle_data = {'lr_day1_1' : [lr_day1_1], \
                   'inf_lr_day1_1' : [inf_lr_day1_1], \
                   'lr_day1_1_variance' : [lr_day1_1_variance], \
                   'theta_Q_day1_1' : [theta_Q_day1_1],\
                   'inf_theta_Q_day1_1' : [inf_theta_Q_day1_1], \
                   'theta_Q_day1_1_variance' : [theta_Q_day1_1_variance], \
                   'theta_rep_day1_1' : [theta_rep_day1_1],\
                   'inf_theta_rep_day1_1' : [inf_theta_rep_day1_1], \
                   'theta_rep_day1_1_variance' : [theta_rep_day1_1_variance], \
                       
                   'lr_day1_2' : [lr_day1_2], \
                   'inf_lr_day1_2' : [inf_lr_day1_2], \
                   'lr_day1_2_variance' : [lr_day1_2_variance], \
                   'theta_Q_day1_2' : [theta_Q_day1_2],\
                   'inf_theta_Q_day1_2' : [inf_theta_Q_day1_2], \
                   'theta_Q_day1_2_variance' : [theta_Q_day1_2_variance], \
                   'theta_rep_day1_2' : [theta_rep_day1_2],\
                   'inf_theta_rep_day1_2' : [inf_theta_rep_day1_2], \
                   'theta_rep_day1_2_variance' : [theta_rep_day1_2_variance], \
                       
                   'lr_day2' : [lr_day2], \
                   'inf_lr_day2' : [inf_lr_day2], \
                   'lr_day2_variance' : [lr_day2_variance], \
                   'theta_Q_day2' : [theta_Q_day2],\
                   'inf_theta_Q_day2' : [inf_theta_Q_day2], \
                   'theta_Q_day2_variance' : [theta_Q_day2_variance], \
                   'theta_rep_day2' : [theta_rep_day2],\
                   'inf_theta_rep_day2' : [inf_theta_rep_day2], \
                   'theta_rep_day2_variance' : [theta_rep_day2_variance]}

    prior_df = pd.DataFrame(data = prior_data)       
    df = pd.DataFrame(data = pickle_data)
        
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    random_num = numpy.random.rand()
    
    
elif model == 'B_3':
    print("Starting inference for model B_3")
    infer = models.SingleInference_modelB_3(newagent, data, k = k)
    loss, params = infer.infer_posterior(iter_steps=150, num_particles = 10)
    
    " ========================= Save Priors ========================= "
    "Θ_Q"
    theta_Q_day1_1_conc = infer.theta_Q_day1_1_conc
    theta_Q_day1_1_rate = infer.theta_Q_day1_1_rate
    
    "Θ_rep"
    theta_rep_day1_1_conc = infer.theta_rep_day1_1_conc
    theta_rep_day1_1_rate = infer.theta_rep_day1_1_rate
    
    "--- Day 1 : 2 ---"
    "Θ_Q"
    theta_Q_day1_2_conc = infer.theta_Q_day1_2_conc
    theta_Q_day1_2_rate = infer.theta_Q_day1_2_rate
    
    "Θ_rep"
    theta_rep_day1_2_conc = infer.theta_rep_day1_2_conc
    theta_rep_day1_2_rate = infer.theta_rep_day1_2_rate
    
    "--- Day 2 ---"
    "Θ_Q"
    theta_Q_day2_conc = infer.theta_Q_day2_conc
    theta_Q_day2_rate = infer.theta_Q_day2_rate
    
    "Θ_rep"
    theta_rep_day2_conc = infer.theta_rep_day2_conc
    theta_rep_day2_rate = infer.theta_rep_day2_rate
    
    prior_data = {'theta_Q_day1_1_conc' : [theta_Q_day1_1_conc],\
            'theta_Q_day1_1_rate' : [theta_Q_day1_1_rate], \
                
            'theta_rep_day1_1_conc' : [theta_rep_day1_1_conc],\
            'theta_rep_day1_1_rate' : [theta_rep_day1_1_rate], \
            
            'theta_Q_day1_2_conc' : [theta_Q_day1_2_conc],\
            'theta_Q_day1_2_rate' : [theta_Q_day1_2_rate], \
            
            'theta_rep_day1_2_conc' : [theta_rep_day1_2_conc],\
            'theta_rep_day1_2_rate' : [theta_rep_day1_2_rate], \
                
            'theta_Q_day2_conc' : [theta_Q_day2_conc],\
            'theta_Q_day2_rate' : [theta_Q_day2_rate], \
                
            'theta_rep_day2_conc' : [theta_rep_day2_conc],\
            'theta_rep_day2_rate' : [theta_rep_day2_rate],\
                
            'Q_init': [infer.agent.Q_init]}
    
    " ========================= Inference Results ======================== "
    "--- Day 1 : 1 ---"
    inf_theta_Q_day1_1 = params["theta_Q_day1_1_conc"][0]/params["theta_Q_day1_1_rate"][0]
    theta_Q_day1_1_variance = models.gamma_variance(params["theta_Q_day1_1_conc"][0], params["theta_Q_day1_1_rate"][0])
    
    inf_theta_rep_day1_1 = params["theta_rep_day1_1_conc"][0]/params["theta_rep_day1_1_rate"][0]
    theta_rep_day1_1_variance = models.gamma_variance(params["theta_rep_day1_1_conc"][0], params["theta_rep_day1_1_rate"][0])
    
    "--- Day 1 : 2 ---"
    inf_theta_Q_day1_2 = params["theta_Q_day1_2_conc"][0]/params["theta_Q_day1_2_rate"][0]
    theta_Q_day1_2_variance = models.gamma_variance(params["theta_Q_day1_2_conc"][0], params["theta_Q_day1_2_rate"][0])
    
    inf_theta_rep_day1_2 = params["theta_rep_day1_2_conc"][0]/params["theta_rep_day1_2_rate"][0]
    theta_rep_day1_2_variance = models.gamma_variance(params["theta_rep_day1_2_conc"][0], params["theta_rep_day1_2_rate"][0])
    
    "--- Day 2 ---"
    inf_theta_Q_day2 = params["theta_Q_day2_conc"][0]/params["theta_Q_day2_rate"][0]
    theta_Q_day2_variance = models.gamma_variance(params["theta_Q_day2_conc"][0], params["theta_Q_day2_rate"][0])
    
    inf_theta_rep_day2 = params["theta_rep_day2_conc"][0]/params["theta_rep_day2_rate"][0]
    theta_rep_day2_variance = models.gamma_variance(params["theta_rep_day2_conc"][0], params["theta_rep_day2_rate"][0])
    
    pickle_data = {'theta_Q_day1_1' : [theta_Q_day1_1],\
                   'inf_theta_Q_day1_1' : [inf_theta_Q_day1_1], \
                   'theta_Q_day1_1_variance' : [theta_Q_day1_1_variance], \
                   'theta_rep_day1_1' : [theta_rep_day1_1],\
                   'inf_theta_rep_day1_1' : [inf_theta_rep_day1_1], \
                   'theta_rep_day1_1_variance' : [theta_rep_day1_1_variance], \
                       
                   'theta_Q_day1_2' : [theta_Q_day1_2],\
                   'inf_theta_Q_day1_2' : [inf_theta_Q_day1_2], \
                   'theta_Q_day1_2_variance' : [theta_Q_day1_2_variance], \
                   'theta_rep_day1_2' : [theta_rep_day1_2],\
                   'inf_theta_rep_day1_2' : [inf_theta_rep_day1_2], \
                   'theta_rep_day1_2_variance' : [theta_rep_day1_2_variance], \
                       
                   'theta_Q_day2' : [theta_Q_day2],\
                   'inf_theta_Q_day2' : [inf_theta_Q_day2], \
                   'theta_Q_day2_variance' : [theta_Q_day2_variance], \
                   'theta_rep_day2' : [theta_rep_day2],\
                   'inf_theta_rep_day2' : [inf_theta_rep_day2], \
                   'theta_rep_day2_variance' : [theta_rep_day2_variance]}

    prior_df = pd.DataFrame(data = prior_data)       
    df = pd.DataFrame(data = pickle_data)
        
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    random_num = numpy.random.rand()
    
elif model == 'F':
    infer = models.SingleInference_modelF(newagent, data)
    loss, params = infer.infer_posterior(iter_steps=150, num_particles = 10)
    
    " ========================= Save Priors ========================= "
    prior_data = {'theta_Q_day1_1_conc' : [theta_Q_day1_1_conc],\
            'theta_Q_day1_1_rate' : [theta_Q_day1_1_rate], \
                
            'theta_rep_day1_1_conc' : [theta_rep_day1_1_conc],\
            'theta_rep_day1_1_rate' : [theta_rep_day1_1_rate], \
            
            'theta_Q_day1_2_conc' : [theta_Q_day1_2_conc],\
            'theta_Q_day1_2_rate' : [theta_Q_day1_2_rate], \
            
            'theta_rep_day1_2_conc' : [theta_rep_day1_2_conc],\
            'theta_rep_day1_2_rate' : [theta_rep_day1_2_rate], \
                
            'theta_Q_day2_conc' : [theta_Q_day2_conc],\
            'theta_Q_day2_rate' : [theta_Q_day2_rate], \
                
            'theta_rep_day2_conc' : [theta_rep_day2_conc],\
            'theta_rep_day2_rate' : [theta_rep_day2_rate],\
                
            'Q_init': [infer.agent.Q_init]}
    
    " ========================= Inference Results ======================== "
    "--- Day 1 ---"
    inf_theta_replambda_day1 = (params["theta_replambda_day1_alpha"][0]/(params["theta_replambda_day1_alpha"][0]+params["theta_replambda_day1_beta"][0]))
    theta_replambda_day1_variance = models.beta_variance(params["theta_replambda_day1_alpha"][0], params["theta_replambda_day1_beta"][0])
    
    inf_theta_rep0_day1 = params["theta_rep0_day1_conc"][0]/params["theta_rep0_day1_rate"][0]
    theta_rep0_day1_variance = models.gamma_variance(params["theta_rep0_day1_conc"][0], params["theta_rep0_day1_rate"][0])

    inf_theta_Qlambda_day1 = (params["theta_Qlambda_day1_alpha"][0]/(params["theta_Qlambda_day1_alpha"][0]+params["theta_Qlambda_day1_beta"][0]))
    theta_Qlambda_day1_variance = models.beta_variance(params["theta_Qlambda_day1_alpha"][0], params["theta_Qlambda_day1_beta"][0])
    
    inf_theta_Q0_day1 = params["theta_Q0_day1_conc"][0]/params["theta_Q0_day1_rate"][0]
    theta_Q0_day1_variance = models.gamma_variance(params["theta_Q0_day1_conc"][0], params["theta_Q0_day1_rate"][0])

    "--- Day 2 ---"
    inf_theta_replambda_day2 = (params["theta_replambda_day2_alpha"][0]/(params["theta_replambda_day2_alpha"][0]+params["theta_replambda_day2_beta"][0]))
    theta_replambda_day2_variance = models.beta_variance(params["theta_replambda_day2_alpha"][0], params["theta_replambda_day2_beta"][0])
    
    inf_theta_rep0_day2 = params["theta_rep0_day2_conc"][0]/params["theta_rep0_day2_rate"][0]
    theta_rep0_day2_variance = models.gamma_variance(params["theta_rep0_day2_conc"][0], params["theta_rep0_day2_rate"][0])

    inf_theta_Qlambda_day2 = (params["theta_Qlambda_day2_alpha"][0]/(params["theta_Qlambda_day2_alpha"][0]+params["theta_Qlambda_day2_beta"][0]))
    theta_Qlambda_day2_variance = models.beta_variance(params["theta_Qlambda_day2_alpha"][0], params["theta_Qlambda_day2_beta"][0])
    
    inf_theta_Q0_day2 = params["theta_Q0_day2_conc"][0]/params["theta_Q0_day2_rate"][0]
    theta_Q0_day2_variance = models.gamma_variance(params["theta_Q0_day2_conc"][0], params["theta_Q0_day2_rate"][0])
    
    pickle_data = {'theta_replambda_day1' : [theta_replambda_day1], \
                   'inf_theta_replambda_day1' : [inf_theta_replambda_day1], \
                   'theta_replambda_day1_variance' : [theta_replambda_day1_variance], \
                   'theta_rep0_day1' : [theta_rep0_day1], \
                   'inf_theta_rep0_day1' : [inf_theta_rep0_day1], \
                   'theta_rep0_day1_variance' : [theta_rep0_day1_variance], \
                   'theta_Qlambda_day1' : [theta_Qlambda_day1], \
                   'inf_theta_Qlambda_day1' : [inf_theta_Qlambda_day1], \
                   'theta_Qlambda_day1_variance' : [theta_Qlambda_day1_variance], \
                   'theta_Q0_day1' : [theta_Q0_day1], \
                   'inf_theta_Q0_day1' : [inf_theta_Q0_day1], \
                   'theta_Q0_day1_variance' : [theta_Q0_day1_variance], \
                   'theta_replambda_day2' : [theta_replambda_day2], \
                   'inf_theta_replambda_day2' : [inf_theta_replambda_day2], \
                   'theta_replambda_day2_variance' : [theta_replambda_day2_variance], \
                   'theta_rep0_day2' : [theta_rep0_day2], \
                   'inf_theta_rep0_day2' : [inf_theta_rep0_day2], \
                   'theta_rep0_day2_variance' : [theta_rep0_day2_variance], \
                   'theta_Qlambda_day2' : [theta_Qlambda_day2], \
                   'inf_theta_Qlambda_day2' : [inf_theta_Qlambda_day2], \
                   'theta_Qlambda_day2_variance' : [theta_Qlambda_day2_variance], \
                   'theta_Q0_day2' : [theta_Q0_day2], \
                   'inf_theta_Q0_day2' : [inf_theta_Q0_day2], \
                   'theta_Q0_day2_variance' : [theta_Q0_day2_variance]}
        
    df = pd.DataFrame(data = pickle_data)

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    random_num = numpy.random.rand()

if method == 'svi':
    pickle.dump( prior_df, open(savedir + "prior.p", "wb" ) )
     
pickle.dump( df, open( savedir + f"param_recovs_{dt_string}_{random_num}.p", "wb" ) )
pickle.dump( loss, open( savedir + f"ELBO_param_recovs_{dt_string}_{random_num}.p", "wb" ) )