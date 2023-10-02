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

import sys

import models_torch as models
import utils

remote = 1

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
    file_day1 = '/home/sascha/Desktop/vb_model/vbm_torch/behav_data/published/Grp1/csv/it6_00420_Tag1_Grp1.mat' # Complete path
    group = 0
    model = 'A'
    num_reps = 0
    published_results = 1
    k = 4.
    
if published_results:
    data_dir = "/home/sascha/Desktop/vb_model/torch/behav_data/published/"
else:
    data_dir = "/home/sascha/Desktop/vb_model/torch/behav_data/"

#%%

"Fit participants (single inference)"

data, prolific_ID = utils.get_participant_data(file_day1, 
                                               group, 
                                               data_dir, 
                                               published_results = published_results)

if group == 0 or group == 1:
    Qinit = [0.2, 0., 0., 0.2]
    
elif group == 2 or group == 3:
    Qinit = [0., 0.2, 0.2, 0.]
    
if model == 'original':
    newagent = models.Vbm(omega=0.5, dectemp=2., lr=0., k=k, Q_init=Qinit, num_blocks=14)
    
elif model == 'A':
    newagent = models.Vbm_A(omega_day1=0.5, \
                            omega_day2=0.5, \
                            dectemp_day1=2., \
                            dectemp_day2=2., \
                            lr_day1=0.5, \
                            lr_day2=0.5, \
                            k=k, \
                            Q_init=Qinit, \
                            num_blocks=14)
    
elif model == 'B':
    newagent = models.Vbm_B(lr_day1=0.5, \
    theta_Q_day1=2., \
    theta_rep_day1=2., \
    lr_day2=0.5, \
    theta_Q_day2=2., \
    theta_rep_day2=2., \
    k=k, \
    Q_init=Qinit, \
    num_blocks=14)
    
elif model == 'C':
    newagent = models.Vbm_C(lr0_day1=0.5, \
    lr_lambda_day1=0.5, \
    theta_Q_day1=2., \
    theta_rep_day1=2., \
    lr0_day2=0.5, \
    lr_lambda_day2=0.5, \
    theta_Q_day2=2., \
    theta_rep_day2=2., \
    k=k, \
    Q_init=Qinit, \
    num_blocks=14)

elif model == 'Bayesianprior':
    newagent = models.Vbm_A_Bayesian(omega_day1=1.5, \
                            omega_day2=1.5, \
                            dectemp_day1=2., \
                            dectemp_day2=2., \
                            lr_day1=0.5, \
                            lr_day2=0.5, \
                            k=k, \
                            Q_init=Qinit, \
                            num_blocks=14)

"--- Inference ---"
for rep in range(num_reps+1):
    if model == 'original':
        infer = models.SingleInference(newagent, data)
        
    elif model == 'A':
        infer = models.SingleInference_modelA(newagent, data)
        
    elif model == 'B':
        infer = models.SingleInference_modelB(newagent, data, k=k)
        
    elif model == 'C':
        infer = models.SingleInference_modelC(newagent, data)
        
    elif model == 'Bayesianprior':
        infer = models.SingleInference_modelA_Bayesian(newagent, data)
        
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
    
    elif model == 'A':
        "--- Day 1 ---"
        inf_dectemp_day1 = params["dectemp_day1_conc"][0]/params["dectemp_day1_rate"][0]
        dectemp_day1_variance = models.gamma_variance(params["dectemp_day1_conc"][0], params["dectemp_day1_rate"][0])
        
        inf_lr_day1 = params["lr_day1_alpha"][0]/(params["lr_day1_alpha"][0]+params["lr_day1_beta"][0])
        lr_day1_variance = models.beta_variance(params["lr_day1_alpha"][0], params["lr_day1_beta"][0])
        
        inf_omega_day1 = params["omega_day1_alpha"][0]/(params["omega_day1_alpha"][0]+params["omega_day1_beta"][0])
        omega_day1_variance = models.beta_variance(params["omega_day1_alpha"][0], params["omega_day1_beta"][0])
        
        "--- Day 2 ---"
        inf_dectemp_day2 = params["dectemp_day2_conc"][0]/params["dectemp_day2_rate"][0]
        dectemp_day2_variance = models.gamma_variance(params["dectemp_day2_conc"][0], params["dectemp_day2_rate"][0])
        
        inf_lr_day2 = params["lr_day2_alpha"][0]/(params["lr_day2_alpha"][0]+params["lr_day2_beta"][0])
        lr_day2_variance = models.beta_variance(params["lr_day2_alpha"][0], params["lr_day2_beta"][0])
        
        inf_omega_day2 = params["omega_day2_alpha"][0]/(params["omega_day2_alpha"][0]+params["omega_day2_beta"][0])
        omega_day2_variance = models.beta_variance(params["omega_day2_alpha"][0], params["omega_day2_beta"][0])
            
        pickle_data = {"inf_dectemp_day1" : [inf_dectemp_day1], \
                       "dectemp_day1_variance" : [dectemp_day1_variance], \
                       "inf_lr_day1" : [inf_lr_day1], \
                       "lr_day1_variance" : [lr_day1_variance], \
                       "inf_omega_day1" : [inf_omega_day1], \
                       "omega_day1_variance" : [omega_day1_variance], \
                       "inf_dectemp_day2" : [inf_dectemp_day2], \
                       "dectemp_day2_variance" : [dectemp_day2_variance], \
                       "inf_lr_day2" : [inf_lr_day2], \
                       "lr_day2_variance" : [lr_day2_variance], \
                       "inf_omega_day2" : [inf_omega_day2], \
                       "omega_day2_variance" : [omega_day2_variance], \
                       "model" : [model], \
                       "loss": [loss_mean]}
            
        df = pd.DataFrame(data = pickle_data)
    
    elif model == 'B':
        
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
                'theta_rep_day2_rate' : [theta_rep_day2_rate]}
        
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
        
    elif model == 'C':
        "--- Day 1 ---"
        inf_lr0_day1 = params["lr0_day1_alpha"][0]/(params["lr0_day1_alpha"][0]+params["lr0_day1_beta"][0])
        lr0_day1_variance = models.beta_variance(params["lr0_day1_alpha"][0], params["lr0_day1_beta"][0])
        
        inf_lr_lambda_day1 = params["lr_lambda_day1_alpha"][0]/(params["lr_lambda_day1_alpha"][0]+params["lr_lambda_day1_beta"][0])
        lr_lambda_day1_variance = models.beta_variance(params["lr_lambda_day1_alpha"][0], params["lr_lambda_day1_beta"][0])
        
        inf_theta_Q_day1 = params["theta_Q_day1_conc"][0]/params["theta_Q_day1_rate"][0]
        theta_Q_day1_variance = models.gamma_variance(params["theta_Q_day1_conc"][0], params["theta_Q_day1_rate"][0])
        
        inf_theta_rep_day1 = params["theta_rep_day1_conc"][0]/params["theta_rep_day1_rate"][0]
        theta_rep_day1_variance = models.gamma_variance(params["theta_rep_day1_conc"][0], params["theta_rep_day1_rate"][0])
        
        "--- Day 2 ---"
        inf_lr0_day2 = params["lr0_day2_alpha"][0]/(params["lr0_day2_alpha"][0]+params["lr0_day2_beta"][0])
        lr0_day2_variance = models.beta_variance(params["lr0_day2_alpha"][0], params["lr0_day2_beta"][0])
        
        inf_lr_lambda_day2 = params["lr_lambda_day2_alpha"][0]/(params["lr_lambda_day2_alpha"][0]+params["lr_lambda_day2_beta"][0])
        lr_lambda_day2_variance = models.beta_variance(params["lr_lambda_day2_alpha"][0], params["lr_lambda_day2_beta"][0])
        
        inf_theta_Q_day2 = params["theta_Q_day2_conc"][0]/params["theta_Q_day2_rate"][0]
        theta_Q_day2_variance = models.gamma_variance(params["theta_Q_day2_conc"][0], params["theta_Q_day2_rate"][0])
        
        inf_theta_rep_day2 = params["theta_rep_day2_conc"][0]/params["theta_rep_day2_rate"][0]
        theta_rep_day2_variance = models.gamma_variance(params["theta_rep_day2_conc"][0], params["theta_rep_day2_rate"][0])
        
        pickle_data = {'inf_lr0_day1' : [inf_lr0_day1], \
                       'lr0_day1_variance' : [lr0_day1_variance], \
                       'inf_lr_lambda_day1' : [inf_lr_lambda_day1], \
                       'lr_lambda_day1_variance' : [lr_lambda_day1_variance], \
                       'inf_theta_Q_day1' : [inf_theta_Q_day1], \
                       'theta_Q_day1_variance' : [theta_Q_day1_variance], \
                       'inf_theta_rep_day1' : [inf_theta_rep_day1], \
                       'theta_rep_day1_variance' : [theta_rep_day1_variance], \
                       'inf_lr0_day2' : [inf_lr0_day2], \
                       'lr0_day2_variance' : [lr0_day2_variance], \
                       'inf_lr_lambda_day2' : [inf_lr_lambda_day2], \
                       'lr_lambda_day2_variance' : [lr_lambda_day2_variance], \
                       'inf_theta_Q_day2' : [inf_theta_Q_day2], \
                       'theta_Q_day2_variance' : [theta_Q_day2_variance], \
                       'inf_theta_rep_day2' : [inf_theta_rep_day2], \
                       'theta_rep_day2_variance' : [theta_rep_day2_variance], \
                       'model': [model], \
                       'loss': [loss_mean]}
            
    elif model == 'Bayesianprior':
        "--- Day 1 ---"
        inf_dectemp_day1 = params["dectemp_day1_conc"][0]/params["dectemp_day1_rate"][0]
        dectemp_day1_variance = models.gamma_variance(params["dectemp_day1_conc"][0], params["dectemp_day1_rate"][0])
        
        inf_lr_day1 = params["lr_day1_alpha"][0]/(params["lr_day1_alpha"][0]+params["lr_day1_beta"][0])
        lr_day1_variance = models.beta_variance(params["lr_day1_alpha"][0], params["lr_day1_beta"][0])
        
        inf_omega_day1 = params["omega_day1_conc"][0]/params["omega_day1_rate"][0]
        omega_day1_variance = models.gamma_variance(params["omega_day1_conc"][0], params["omega_day1_rate"][0])
        
        "--- Day 2 ---"
        inf_dectemp_day2 = params["dectemp_day2_conc"][0]/params["dectemp_day2_rate"][0]
        dectemp_day2_variance = models.gamma_variance(params["dectemp_day2_conc"][0], params["dectemp_day2_rate"][0])
        
        inf_lr_day2 = params["lr_day2_alpha"][0]/(params["lr_day2_alpha"][0]+params["lr_day2_beta"][0])
        lr_day2_variance = models.beta_variance(params["lr_day2_alpha"][0], params["lr_day2_beta"][0])
        
        inf_omega_day2 = params["omega_day2_conc"][0]/params["omega_day2_rate"][0]
        omega_day2_variance = models.gamma_variance(params["omega_day2_conc"][0], params["omega_day2_rate"][0])

        pickle_data = {"inf_dectemp_day1" : [inf_dectemp_day1], \
                       "dectemp_day1_variance" : [dectemp_day1_variance], \
                       "inf_lr_day1" : [inf_lr_day1], \
                       "lr_day1_variance" : [lr_day1_variance], \
                       "inf_omega_day1" : [inf_omega_day1], \
                       "omega_day1_variance" : [omega_day1_variance], \
                       "inf_dectemp_day2" : [inf_dectemp_day2], \
                       "dectemp_day2_variance" : [dectemp_day2_variance], \
                       "inf_lr_day2" : [inf_lr_day2], \
                       "lr_day2_variance" : [lr_day2_variance], \
                       "inf_omega_day2" : [inf_omega_day2], \
                       "omega_day2_variance" : [omega_day2_variance], \
                       "model" : [model], \
                       "loss": [loss_mean]}
            
        df = pd.DataFrame(data = pickle_data)

    if num_reps == 0:
        if published_results:
            "Published"
            pickle.dump( prior_df, open(f"behav_fit/published/model{model}/prior.p", "wb" ) )
            pickle.dump( df, open(f"behav_fit/published/model{model}/participant_{prolific_ID}.p", "wb" ) )
            pickle.dump( loss, open(f"behav_fit/published/model{model}/ELBO_participant_{prolific_ID}.p", "wb" ) )

        else:
            "Clipre"
            pickle.dump( prior_df, open(f"behav_fit/model{model}/prior.p", "wb" ) )
            pickle.dump( df, open(f"behav_fit/model{model}/participant_{prolific_ID}.p", "wb" ) )
            pickle.dump( loss, open(f"behav_fit/model{model}/ELBO_participant_{prolific_ID}.p", "wb" ) )
        
    else:
        if published_results:
            "Published"
            pickle.dump( prior_df, open(f"behav_fit/published/model{model}/prior.p", "wb" ) )
            pickle.dump( df, open(f"behav_fit/published/model{model}/participant_{prolific_ID}_rep{rep}.p", "wb" ) )
            pickle.dump( loss, open(f"behav_fit/published/model{model}/ELBO_participant_{prolific_ID}_rep{rep}.p", "wb" ) )
            
        else:
            "Clipre"
            pickle.dump( prior_df, open(f"behav_fit/model{model}/prior.p", "wb" ) )
            pickle.dump( df, open(f"behav_fit/model{model}/participant_{prolific_ID}_rep{rep}.p", "wb" ) )
            pickle.dump( loss, open(f"behav_fit/model{model}/ELBO_participant_{prolific_ID}_rep{rep}.p", "wb" ) )
            
            
            