#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:59:40 2023

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
import pickle

plt.style.use("classic")

#%%
model = sys.argv[1]
resim =  int(sys.argv[2]) # whether to simulate agents with inferred parameters
method = sys.argv[3] # "svi" or "mcmc"
k = int(sys.argv[4])

if resim:
    df = pickle.load( open(f"model{model}_param_anal.p", "rb" ) )
    pb_idx = sys.argv[3]
    
#%%

if model == 'A':
    npar = 6
    
    if resim:
        omega_day1 = df.iloc[pb_idx, :].inf_omega_day1
        omega_day2 = df.iloc[pb_idx, :].inf_omega_day2
        lr_day1 = df.iloc[pb_idx, :].inf_lr_day1
        lr_day2 = df.iloc[pb_idx, :].inf_lr_day2
        dectemp_day1 = df.iloc[pb_idx, :].inf_dectemp_day1
        dectemp_day2 = df.iloc[pb_idx, :].inf_dectemp_day2
        
    else:
        "Simulate with random parameters"
        parameter = numpy.random.uniform(0,1, npar)
        omega_day1 = parameter[0]
        omega_day2 = parameter[1]
        lr_day1 = parameter[2]
        lr_day2 = parameter[3]
        dectemp_day1 = parameter[4]*6
        dectemp_day2 = parameter[5]*6

    newagent = models.vbm_A(omega_day1 = omega_day1, \
                          omega_day2 = omega_day2, \
                          lr_day1 = lr_day1, \
                          lr_day2 = lr_day2, \
                          dectemp_day1 = dectemp_day1, \
                          dectemp_day2 = dectemp_day2, \
                          k=k,\
                          Q_init=[0.4, 0., 0., 0.4])
        

elif model == 'C':
    npar = 8
    parameter = numpy.random.uniform(0,1, npar)
    theta_rep_day1 = parameter[0]*10
    theta_rep_day2 = parameter[1]*10
    theta_Q_day1 = parameter[2]*10
    theta_Q_day2 = parameter[3]*10
    lr0_day1 = parameter[4]
    lr0_day2 = parameter[5]
    lr_lambda_day1 = parameter[6]*0.1
    lr_lambda_day2 = parameter[7]*0.1

    newagent = models.vbm_C(theta_rep_day1 = theta_rep_day1, \
                          theta_rep_day2 = theta_rep_day2, \
                          lr0_day1 = lr0_day1, \
                          lr0_day2 = lr0_day2, \
                          lr_lambda_day1 = lr_lambda_day1, \
                          lr_lambda_day2 = lr_lambda_day2, \
                          theta_Q_day1 = theta_Q_day1, \
                          theta_Q_day2 = theta_Q_day2, \
                          k=k,\
                          Q_init=[0.4, 0., 0., 0.4])

elif model == 'Bayesianprior':
    "Adapted from model A"
    npar = 6
    
    if resim:
        omega_day1 = df.iloc[pb_idx, :].inf_omega_day1
        omega_day2 = df.iloc[pb_idx, :].inf_omega_day2
        lr_day1 = df.iloc[pb_idx, :].inf_lr_day1
        lr_day2 = df.iloc[pb_idx, :].inf_lr_day2
        dectemp_day1 = df.iloc[pb_idx, :].inf_dectemp_day1
        dectemp_day2 = df.iloc[pb_idx, :].inf_dectemp_day2
        
    else:
        "Simulate with random parameters"
        parameter = numpy.random.uniform(0,1, npar)
        omega_day1 = parameter[0]*6
        omega_day2 = parameter[1]*6
        lr_day1 = parameter[2]
        lr_day2 = parameter[3]
        dectemp_day1 = parameter[4]*6
        dectemp_day2 = parameter[5]*6

    newagent = models.vbm_A_Bayesian(omega_day1 = omega_day1, \
                          omega_day2 = omega_day2, \
                          lr_day1 = lr_day1, \
                          lr_day2 = lr_day2, \
                          dectemp_day1 = dectemp_day1, \
                          dectemp_day2 = dectemp_day2, \
                          k=k,\
                          Q_init=[0.4, 0., 0., 0.4])

elif model == 'D':
    npar = 8
    parameter = numpy.random.uniform(0,1, npar)
    
    theta_rep0_day1 = parameter[0]*6
    theta_replambda_day1 = parameter[1]*0.1
    theta_Q0_day1 = parameter[2]*6
    theta_Qlambda_day1 = parameter[3]*0.1
    
    theta_rep0_day2 = parameter[4]*6
    theta_replambda_day2 = parameter[5]*0.1
    theta_Q0_day2 = parameter[6]*6
    theta_Qlambda_day2 = parameter[7]*0.1


    newagent = models.vbm_D(theta_rep0_day1 = theta_rep0_day1, \
                          theta_replambda_day1 = theta_replambda_day1, \
                          theta_Q0_day1 = theta_Q0_day1, \
                          theta_Qlambda_day1 = theta_Qlambda_day1, \
                              
                          theta_rep0_day2 = theta_rep0_day2, \
                          theta_replambda_day2 = theta_replambda_day2, \
                          theta_Q0_day2 = theta_Q0_day2, \
                          theta_Qlambda_day2 = theta_Qlambda_day2, \
                          k=k,\
                          Q_init=[0.8, 0.2, 0.2, 0.8])
        
elif model == 'D_simple':
    npar = 2
    parameter = numpy.random.uniform(0,1, npar)
    
    theta_Qlambda_day1 = parameter[0]*0.1
    theta_Qlambda_day2 = parameter[1]*0.1


    newagent = models.vbm_D_simple(theta_Qlambda_day1 = theta_Qlambda_day1, \
                          theta_Qlambda_day2 = theta_Qlambda_day2, \
                          k=k,\
                          Q_init=[0.8, 0.2, 0.2, 0.8])
        
#%%

newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir = './matlabcode/clipre/')

newenv.run()
data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
        "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
        "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}

#%%

if model == 'A':
    infer = models.SingleInference_modelA(newagent, data)
    loss, params = infer.infer_posterior(iter_steps=150, num_particles = 10)
    
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
        
    pickle_data = {"dectemp_day1" : [dectemp_day1], \
                   "inf_dectemp_day1" : [inf_dectemp_day1], \
                   "dectemp_day1_variance" : [dectemp_day1_variance], \
                   "lr_day1" : [lr_day1], \
                   "inf_lr_day1" : [inf_lr_day1], \
                   "lr_day1_variance" : [lr_day1_variance], \
                   "omega_day1" : [omega_day1],\
                   "inf_omega_day1" : [inf_omega_day1], \
                   "omega_day1_variance" : [omega_day1_variance], \
                   "dectemp_day2" : [dectemp_day2], \
                   "inf_dectemp_day2" : [inf_dectemp_day2], \
                   "dectemp_day2_variance" : [dectemp_day2_variance], \
                   "lr_day2" : [lr_day2], \
                   "inf_lr_day2" : [inf_lr_day2], \
                   "lr_day2_variance" : [lr_day2_variance], \
                   "omega_day2" : [omega_day2],\
                   "inf_omega_day2" : [inf_omega_day2], \
                   "omega_day2_variance" : [omega_day2_variance]}
        
    df = pd.DataFrame(data = pickle_data)
        
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    random_num = numpy.random.rand()
    

elif model == 'C':
    infer = models.SingleInference_modelC(newagent, data)
    loss, params = infer.infer_posterior(iter_steps=150, num_particles = 10)
    
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
    
    pickle_data = {'lr0_day1' : [lr0_day1], \
                   'inf_lr0_day1' : [inf_lr0_day1], \
                   'lr0_day1_variance' : [lr0_day1_variance], \
                   'lr_lambda_day1' : [lr_lambda_day1], \
                   'inf_lr_lambda_day1' : [inf_lr_lambda_day1], \
                   'lr_lambda_day1_variance' : [lr_lambda_day1_variance], \
                   'theta_Q_day1' : [theta_Q_day1],\
                   'inf_theta_Q_day1' : [inf_theta_Q_day1], \
                   'theta_Q_day1_variance' : [theta_Q_day1_variance], \
                   'theta_rep_day1' : [theta_rep_day1],\
                   'inf_theta_rep_day1' : [inf_theta_rep_day1], \
                   'theta_rep_day1_variance' : [theta_rep_day1_variance], \
                   'lr0_day2' : [lr0_day2], \
                   'inf_lr0_day2' : [inf_lr0_day2], \
                   'lr0_day2_variance' : [lr0_day2_variance], \
                   'lr_lambda_day2' : [lr_lambda_day2], \
                   'inf_lr_lambda_day2' : [inf_lr_lambda_day2], \
                   'lr_lambda_day2_variance' : [lr_lambda_day2_variance], \
                   'theta_Q_day2' : [theta_Q_day2],\
                   'inf_theta_Q_day2' : [inf_theta_Q_day2], \
                   'theta_Q_day2_variance' : [theta_Q_day2_variance], \
                   'theta_rep_day2' : [theta_rep_day2],\
                   'inf_theta_rep_day2' : [inf_theta_rep_day2], \
                   'theta_rep_day2_variance' : [theta_rep_day2_variance]}
        
    df = pd.DataFrame(data = pickle_data)

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    random_num = numpy.random.rand()
    
elif model == 'D':
    infer = models.SingleInference_modelD(newagent, data)
    loss, params = infer.infer_posterior(iter_steps=150, num_particles = 10)
    
    "--- Day 1 ---"
    inf_theta_replambda_day1 = params["theta_replambda_day1_alpha"][0]/(params["theta_replambda_day1_alpha"][0]+params["theta_replambda_day1_beta"][0])
    theta_replambda_day1_variance = models.beta_variance(params["theta_replambda_day1_alpha"][0], params["theta_replambda_day1_beta"][0])
    
    inf_theta_rep0_day1 = params["theta_rep0_day1_conc"][0]/params["theta_rep0_day1_rate"][0]
    theta_rep0_day1_variance = models.gamma_variance(params["theta_rep0_day1_conc"][0], params["theta_rep0_day1_rate"][0])

    inf_theta_Qlambda_day1 = params["theta_Qlambda_day1_alpha"][0]/(params["theta_Qlambda_day1_alpha"][0]+params["theta_Qlambda_day1_beta"][0])
    theta_Qlambda_day1_variance = models.beta_variance(params["theta_Qlambda_day1_alpha"][0], params["theta_Qlambda_day1_beta"][0])
    
    inf_theta_Q0_day1 = params["theta_Q0_day1_conc"][0]/params["theta_Q0_day1_rate"][0]
    theta_Q0_day1_variance = models.gamma_variance(params["theta_Q0_day1_conc"][0], params["theta_Q0_day1_rate"][0])

    "--- Day 2 ---"
    inf_theta_replambda_day2 = params["theta_replambda_day2_alpha"][0]/(params["theta_replambda_day2_alpha"][0]+params["theta_replambda_day2_beta"][0])
    theta_replambda_day2_variance = models.beta_variance(params["theta_replambda_day2_alpha"][0], params["theta_replambda_day2_beta"][0])
    
    inf_theta_rep0_day2 = params["theta_rep0_day2_conc"][0]/params["theta_rep0_day2_rate"][0]
    theta_rep0_day2_variance = models.gamma_variance(params["theta_rep0_day2_conc"][0], params["theta_rep0_day2_rate"][0])

    inf_theta_Qlambda_day2 = params["theta_Qlambda_day2_alpha"][0]/(params["theta_Qlambda_day2_alpha"][0]+params["theta_Qlambda_day2_beta"][0])
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
    
elif model == 'D_simple':
    # infer = models.SingleInference_modelD_simple(newagent, data)
    infer = models.SingleInference_modelD_simple(newagent, data)
    loss, params = infer.infer_posterior(iter_steps=150, num_particles = 10)
    
    "--- Day 1 ---"
    inf_theta_Qlambda_day1 = params["theta_Qlambda_day1_alpha"][0]/(params["theta_Qlambda_day1_alpha"][0]+params["theta_Qlambda_day1_beta"][0])
    theta_Qlambda_day1_variance = models.beta_variance(params["theta_Qlambda_day1_alpha"][0], params["theta_Qlambda_day1_beta"][0])

    "--- Day 2 ---"
    inf_theta_Qlambda_day2 = params["theta_Qlambda_day2_alpha"][0]/(params["theta_Qlambda_day2_alpha"][0]+params["theta_Qlambda_day2_beta"][0])
    theta_Qlambda_day2_variance = models.beta_variance(params["theta_Qlambda_day2_alpha"][0], params["theta_Qlambda_day2_beta"][0])
    
    pickle_data = {'theta_Qlambda_day1' : [theta_Qlambda_day1], \
                   'inf_theta_Qlambda_day1' : [inf_theta_Qlambda_day1], \
                   'theta_Qlambda_day1_variance' : [theta_Qlambda_day1_variance], \
                   'theta_Qlambda_day2' : [theta_Qlambda_day2], \
                   'inf_theta_Qlambda_day2' : [inf_theta_Qlambda_day2], \
                   'theta_Qlambda_day2_variance' : [theta_Qlambda_day2_variance]}
        
    df = pd.DataFrame(data = pickle_data)

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    random_num = numpy.random.rand()

elif model == 'Bayesianprior':
    infer = models.SingleInference_modelA_Bayesian(newagent, data)
    loss, params = infer.infer_posterior(iter_steps=150, num_particles = 10)
    
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
        
    pickle_data = {"dectemp_day1" : [dectemp_day1], \
                   "inf_dectemp_day1" : [inf_dectemp_day1], \
                   "dectemp_day1_variance" : [dectemp_day1_variance], \
                   "lr_day1" : [lr_day1], \
                   "inf_lr_day1" : [inf_lr_day1], \
                   "lr_day1_variance" : [lr_day1_variance], \
                   "omega_day1" : [omega_day1],\
                   "inf_omega_day1" : [inf_omega_day1], \
                   "omega_day1_variance" : [omega_day1_variance], \
                   "dectemp_day2" : [dectemp_day2], \
                   "inf_dectemp_day2" : [inf_dectemp_day2], \
                   "dectemp_day2_variance" : [dectemp_day2_variance], \
                   "lr_day2" : [lr_day2], \
                   "inf_lr_day2" : [inf_lr_day2], \
                   "lr_day2_variance" : [lr_day2_variance], \
                   "omega_day2" : [omega_day2],\
                   "inf_omega_day2" : [inf_omega_day2], \
                   "omega_day2_variance" : [omega_day2_variance]}
        
    df = pd.DataFrame(data = pickle_data)
        
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    random_num = numpy.random.rand()
    
 
#%%
    
if method == 'svi':
    pickle.dump( df, open(f"param_recov/model{model}/k_{k}/param_recovs_{dt_string}_{random_num}.p", "wb" ) )
    pickle.dump( loss, open(f"param_recov/model{model}/k_{k}/ELBO_param_recovs_{dt_string}_{random_num}.p", "wb" ) )
    
elif method == 'mcmc':
    pickle.dump( df, open(f"param_recov/mcmc/model{model}/k_{k}/param_recovs_{dt_string}_{random_num}.p", "wb" ) )
    pickle.dump( loss, open(f"param_recov/mcmc/model{model}/k_{k}/ELBO_param_recovs_{dt_string}_{random_num}.p", "wb" ) )