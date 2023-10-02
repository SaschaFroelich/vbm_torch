#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:22:51 2023

@author: sascha
"""

import ipdb
import pickle
import pandas as pd
import numpy as np
import torch

import sys
import os

import models_torch as models
import inferencemodels
import utils

remote = 0

if remote:
    file_day1 = sys.argv[1] # Complete path
    group = int(sys.argv[2])
    model = sys.argv[3]
    num_reps = int(sys.argv[4]) # How often to repeat the parameter inference per participant (0 to infer just once)
    published_results = int(sys.argv[5])
    k = float(sys.argv[6])
    
    print("==============================================================")
    print(f"Doing file {file_day1} for group {group} and model {model}.")

else:
    file_day1 = '/home/sascha/Desktop/vb_model/vbm_torch/behav_data/Grp1/csv/it6_5b5e0e86902ad10001cfcc59_Tag1_Grp1.mat' # Complete path
    group = 0
    model = 'B'
    num_reps = 0
    published_results = 0
    k = 4.
    
if published_results:
    data_dir = "/home/sascha/Desktop/vb_model/torch/behav_data/published/"
else:
    data_dir = "/home/sascha/Desktop/vb_model/vbm_torch/behav_data/"

if published_results:
    "Published"
    savedir = f"behav_fit/published/model{model}/k_{k}/"

else:
    "Clipre"
    savedir = f"behav_fit/model{model}/k_{k}/"

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

assert(num_reps == 0)

"Fit participants (single inference)"

data, prolific_ID = utils.get_participant_data(file_day1, 
                                               group, 
                                               data_dir, 
                                               published_results = published_results)

if group == 0 or group == 1:
    Qinit = torch.tensor([[[0.2, 0., 0., 0.2]]])
    
elif group == 2 or group == 3:
    Qinit = torch.tensor([[[0., 0.2, 0.2, 0.]]])
    
if model == 'original':
    newagent = models.Vbm(omega=0.5, dectemp=2., lr=0., k=k, Q_init=Qinit)
    
elif model == 'B':
    newagent = models.Vbm_B(lr_day1 = torch.tensor([[0.5]]), \
    theta_Q_day1 = torch.tensor([[2.]]), \
    theta_rep_day1 = torch.tensor([[2.]]), \
    lr_day2 = torch.tensor([[0.5]]), \
    theta_Q_day2 = torch.tensor([[2.]]), \
    theta_rep_day2 = torch.tensor([[2.]]), \
    k = torch.tensor([[k]]), \
    Q_init = Qinit)
        
elif model == 'B_2':
    newagent = models.Vbm_B_2(lr_day1_1=0.5, \
                              lr_day1_2=0.5, \
                                  
    theta_Q_day1_1=2., \
        theta_Q_day1_2=2., \
            
    theta_rep_day1_1=2., \
        theta_rep_day1_2=2., \
    lr_day2=0.5, \
    theta_Q_day2=2., \
    theta_rep_day2=2., \
    k=k, \
    Q_init=Qinit)

elif model == 'B_3':
    if group == 0 or group == 1:
        Qinit = [0.8, 0., 0., 0.8]
        
    elif group == 2 or group == 3:
        Qinit = [0., 0.8, 0.8, 0.]

    newagent = models.Vbm_B_3(theta_Q_day1_1=2., \
        theta_Q_day1_2=2., \
        theta_rep_day1_1=2., \
        theta_rep_day1_2=2., \
        theta_Q_day2=2., \
        theta_rep_day2=2., \
        k=k, \
        Q_init=Qinit)
            
elif model == 'B_onlydual':
    newagent = models.Vbm_B_onlydual(lr_day1=0.5, \
    theta_Q_day1=2., \
    theta_rep_day1=2., \
    lr_day2=0.5, \
    theta_Q_day2=2., \
    theta_rep_day2=2., \
    k=k, \
    Q_init=Qinit)

"--- Inference ---"
for rep in range(num_reps+1):
    if model == 'original':
        infer = inferencemodels.SingleInference(newagent, data)
        
    elif model == 'B' or model == 'B_onlydual':
        infer = inferencemodels.SingleInference_modelB(newagent, data, k=k)
        
    elif model == 'B_2':
        infer = inferencemodels.SingleInference_modelB_2(newagent, data, k=k)
        
    elif model == 'B_3':
        infer = inferencemodels.SingleInference_modelB_3(newagent, data, k=k)
        
    loss, params = infer.infer_posterior(iter_steps=150, num_particles = 10)
    loss_mean = np.array(loss[-2:]).mean()
    
    if model == 'original':
        inf_dectemp = params["dectemp_conc"][0]/params["dectemp_rate"][0]
        dectemp_variance = models.gamma_variance(params["dectemp_conc"][0], params["dectemp_rate"][0])
        
        inf_lr = params["lr_alpha"][0]/(params["lr_alpha"][0]+params["lr_beta"][0])
        lr_variance = models.beta_variance(params["lr_alpha"][0], params["lr_beta"][0])
        
        inf_omega = params["omega_alpha"][0]/(params["omega_alpha"][0]+params["omega_beta"][0])
        omega_variance = models.beta_variance(params["omega_alpha"][0], params["omega_beta"][0])
    
        post_sample_df=infer.sample_posterior()
            
        pickle_data = {"inf_dectemp" : [inf_dectemp], \
                       "dectemp_variance" : [dectemp_variance], \
                       "inf_lr" : [inf_lr], \
                       "lr_variance" : [lr_variance], \
                       "inf_omega" : [inf_omega], \
                       "omega_variance" : [omega_variance], \
                        "model" : [model], \
                        "loss": [loss_mean]}

        df = pd.DataFrame (data=pickle_data)
    
    elif model == 'B' or model == 'B_onlydual':
        
        " ========================= Save Priors ========================"
        prior_data = {'lr_day1_alpha' : [infer.lr_day1_alpha], \
                'lr_day1_beta' : [infer.lr_day1_beta], \
                    
                'theta_Q_day1_conc' : [infer.theta_Q_day1_conc],\
                'theta_Q_day1_rate' : [infer.theta_Q_day1_rate], \
                    
                'theta_rep_day1_conc' : [infer.theta_rep_day1_conc],\
                'theta_rep_day1_rate' : [infer.theta_rep_day1_rate], \
                    
                'lr_day2_alpha' : [infer.lr_day2_alpha], \
                'lr_day2_beta' : [infer.lr_day2_beta], \
                    
                'theta_Q_day2_conc' : [infer.theta_Q_day2_conc],\
                'theta_Q_day2_rate' : [infer.theta_Q_day2_rate], \
                    
                'theta_rep_day2_conc' : [infer.theta_rep_day2_conc],\
                'theta_rep_day2_rate' : [infer.theta_rep_day2_rate],\
                    
                'Q_init': [infer.agent.Q_init]}
        
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
        
        pickle_data = {'inf_lr_day1' : [inf_lr_day1], \
                       'lr_day1_variance' : [lr_day1_variance], \
                       'inf_theta_Q_day1' : [inf_theta_Q_day1], \
                       'theta_Q_day1_variance' : [theta_Q_day1_variance], \
                       'inf_theta_rep_day1' : [inf_theta_rep_day1], \
                       'theta_rep_day1_variance' : [theta_rep_day1_variance], \
                       'inf_lr_day2' : [inf_lr_day2], \
                       'lr_day2_variance' : [lr_day2_variance], \
                       'inf_theta_Q_day2' : [inf_theta_Q_day2], \
                       'theta_Q_day2_variance' : [theta_Q_day2_variance], \
                       'inf_theta_rep_day2' : [inf_theta_rep_day2], \
                       'theta_rep_day2_variance' : [theta_rep_day2_variance], \
                       'model' : [model], \
                       'loss': [loss_mean]}
    
        prior_df = pd.DataFrame(data = prior_data)
        df = pd.DataFrame(data = pickle_data)
        
        
    elif model == 'B_2':
        " ========================= Save Priors ========================= "
        prior_data = {'lr_day1_1_alpha' : [infer.lr_day1_1_alpha], \
                'lr_day1_1_beta' : [infer.lr_day1_1_beta], \
                    
                'theta_Q_day1_1_conc' : [infer.theta_Q_day1_1_conc],\
                'theta_Q_day1_1_rate' : [infer.theta_Q_day1_1_rate], \
                    
                'theta_rep_day1_1_conc' : [infer.theta_rep_day1_1_conc],\
                'theta_rep_day1_1_rate' : [infer.theta_rep_day1_1_rate], \
                    
                'lr_day1_2_alpha' : [infer.lr_day1_2_alpha], \
                'lr_day1_2_beta' : [infer.lr_day1_2_beta], \
                
                'theta_Q_day1_2_conc' : [infer.theta_Q_day1_2_conc],\
                'theta_Q_day1_2_rate' : [infer.theta_Q_day1_2_rate], \
                
                'theta_rep_day1_2_conc' : [infer.theta_rep_day1_2_conc],\
                'theta_rep_day1_2_rate' : [infer.theta_rep_day1_2_rate], \
                    
                'lr_day2_alpha' : [infer.lr_day2_alpha], \
                'lr_day2_beta' : [infer.lr_day2_beta], \
                    
                'theta_Q_day2_conc' : [infer.theta_Q_day2_conc],\
                'theta_Q_day2_rate' : [infer.theta_Q_day2_rate], \
                    
                'theta_rep_day2_conc' : [infer.theta_rep_day2_conc],\
                'theta_rep_day2_rate' : [infer.theta_rep_day2_rate],\
                    
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
        
        pickle_data = {'inf_lr_day1_1' : [inf_lr_day1_1], \
                       'lr_day1_1_variance' : [lr_day1_1_variance], \
                       
                       'inf_theta_Q_day1_1' : [inf_theta_Q_day1_1], \
                       'theta_Q_day1_1_variance' : [theta_Q_day1_1_variance], \
                       
                       'inf_theta_rep_day1_1' : [inf_theta_rep_day1_1], \
                       'theta_rep_day1_1_variance' : [theta_rep_day1_1_variance], \
                           
                       'inf_lr_day1_2' : [inf_lr_day1_2], \
                       'lr_day1_2_variance' : [lr_day1_2_variance], \
                       
                       'inf_theta_Q_day1_2' : [inf_theta_Q_day1_2], \
                       'theta_Q_day1_2_variance' : [theta_Q_day1_2_variance], \
                       
                       'inf_theta_rep_day1_2' : [inf_theta_rep_day1_2], \
                       'theta_rep_day1_2_variance' : [theta_rep_day1_2_variance], \
                           
                       'inf_lr_day2' : [inf_lr_day2], \
                       'lr_day2_variance' : [lr_day2_variance], \
                       
                       'inf_theta_Q_day2' : [inf_theta_Q_day2], \
                       'theta_Q_day2_variance' : [theta_Q_day2_variance], \
                       
                       'inf_theta_rep_day2' : [inf_theta_rep_day2], \
                       'theta_rep_day2_variance' : [theta_rep_day2_variance]}
    
        prior_df = pd.DataFrame(data = prior_data)       
        df = pd.DataFrame(data = pickle_data)
        
    elif model == 'B_3':
        " ========================= Save Priors ========================= "
        prior_data = {
                'theta_Q_day1_1_conc' : [infer.theta_Q_day1_1_conc],\
                'theta_Q_day1_1_rate' : [infer.theta_Q_day1_1_rate], \
                    
                'theta_rep_day1_1_conc' : [infer.theta_rep_day1_1_conc],\
                'theta_rep_day1_1_rate' : [infer.theta_rep_day1_1_rate], \

                'theta_Q_day1_2_conc' : [infer.theta_Q_day1_2_conc],\
                'theta_Q_day1_2_rate' : [infer.theta_Q_day1_2_rate], \
                
                'theta_rep_day1_2_conc' : [infer.theta_rep_day1_2_conc],\
                'theta_rep_day1_2_rate' : [infer.theta_rep_day1_2_rate], \

                'theta_Q_day2_conc' : [infer.theta_Q_day2_conc],\
                'theta_Q_day2_rate' : [infer.theta_Q_day2_rate], \
                    
                'theta_rep_day2_conc' : [infer.theta_rep_day2_conc],\
                'theta_rep_day2_rate' : [infer.theta_rep_day2_rate],\
                    
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
        
        pickle_data = {'inf_theta_Q_day1_1' : [inf_theta_Q_day1_1], \
                       'theta_Q_day1_1_variance' : [theta_Q_day1_1_variance], \
                       
                       'inf_theta_rep_day1_1' : [inf_theta_rep_day1_1], \
                       'theta_rep_day1_1_variance' : [theta_rep_day1_1_variance], \
                                                  
                       'inf_theta_Q_day1_2' : [inf_theta_Q_day1_2], \
                       'theta_Q_day1_2_variance' : [theta_Q_day1_2_variance], \
                       
                       'inf_theta_rep_day1_2' : [inf_theta_rep_day1_2], \
                       'theta_rep_day1_2_variance' : [theta_rep_day1_2_variance], \
                                                  
                       'inf_theta_Q_day2' : [inf_theta_Q_day2], \
                       'theta_Q_day2_variance' : [theta_Q_day2_variance], \
                       
                       'inf_theta_rep_day2' : [inf_theta_rep_day2], \
                       'theta_rep_day2_variance' : [theta_rep_day2_variance]}
    
        prior_df = pd.DataFrame(data = prior_data)       
        df = pd.DataFrame(data = pickle_data)
                    
    if group == 0:
        pickle.dump( prior_df, open(savedir + "prior.p", "wb" ) )
    pickle.dump( df, open(savedir + f"participant_{prolific_ID}.p", "wb" ) )
    pickle.dump( loss, open(savedir + f"ELBO_participant_{prolific_ID}.p", "wb" ) )
    
            
            