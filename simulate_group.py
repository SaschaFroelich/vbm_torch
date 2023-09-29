#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:41:39 2023

Script for group inference on simulated data

@author: sascha
"""

import ipdb

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

import numpy
import scipy

import models_torch as models
import env 

import sys
from datetime import datetime
import pickle

plt.style.use("classic")

#%%
# model = sys.argv[1]
# resim =  int(sys.argv[2]) # whether to simulate agents with inferred parameters
# method = sys.argv[3] # "svi" or "mcmc"
# num_agents = 50

model = 'B'
resim =  0 # whether to simulate agents with inferred parameters
method = 'svi' # "svi" or "mcmc"
num_agents = 10
k = 4.

#%%

if resim:
    raise Exception("Not implemented yet, buddy!")
    
if model == 'original':
    npar = 3
    
    agents = []
    groupdata = []
    
    dectemp_true = []
    lr_true = []
    omega_true = []
    
    "Simulate with random parameters"
    for ag in range(num_agents):
        print("Simulating agent no. %d"%ag)
        parameter = numpy.random.uniform(0,1, npar)
        dectemp = parameter[0]*5
        lr = parameter[1]*0.1
        omega = parameter[2]
        
        dectemp_true.append(dectemp)
        lr_true.append(lr)
        omega_true.append(omega)
    
        newagent = models.vbm(dectemp = dectemp, \
                              lr = lr, \
                              omega = omega, \
                              k=k,\
                              Q_init=[0.2, 0., 0., 0.2])

        agents.append(newagent)
        
        newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir = './matlabcode/clipre/')
        
        newenv.run()
        data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
                "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
                "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}
            
        groupdata.append(data)
        
    infer = models.GroupInference(agents, groupdata)
    loss, params = infer.infer_posterior(iter_steps=250, num_particles = 10)
    inference_df = infer.sample_posterior()
    pickle.dump( (dectemp_true, lr_true, omega_true, inference_df), open(f"param_recov/groupinference/model{model}/k_{k}/group_inference.p", "wb" ) )
    pickle.dump( loss, open(f"param_recov/groupinference/model{model}/k_{k}/ELBO_group_inference.p", "wb" ) )
    
elif model == 'B':
    npar = 6
    
    agents = []
    groupdata = []
    
    lr_day1_true = []
    theta_Q_day1_true = []
    theta_rep_day1_true = []
    
    lr_day2_true = []
    theta_Q_day2_true = []
    theta_rep_day2_true = []
    
    "Simulate with random parameters"
    for ag in range(num_agents):
        print("Simulating agent no. %d"%ag)
        parameter = numpy.random.uniform(0,1, npar)
        lr_day1 = parameter[0]*0.1
        theta_Q_day1 = parameter[1]*6
        theta_rep_day1 = parameter[2]*6
        
        lr_day2 = parameter[0]*0.1
        theta_Q_day2 = parameter[1]*6
        theta_rep_day2 = parameter[2]*6
        
        lr_day1_true.append(lr_day1)
        theta_Q_day1_true.append(theta_Q_day1)
        theta_rep_day1_true.append(theta_rep_day1)
        
        lr_day2_true.append(lr_day2)
        theta_Q_day2_true.append(theta_Q_day2)
        theta_rep_day2_true.append(theta_rep_day2)

        newagent = models.vbm_B(lr_day1 = lr_day1, \
                              theta_Q_day1 = theta_Q_day1, \
                              theta_rep_day1 = theta_rep_day1, \
                                  
                              lr_day2 = lr_day2, \
                            theta_Q_day2 = theta_Q_day2, \
                            theta_rep_day2 = theta_rep_day2, \
                              k=k,\
                              Q_init=[0.2, 0., 0., 0.2])

        agents.append(newagent)
        
        newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir = './matlabcode/clipre/')
        
        newenv.run()
        data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
                "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
                    "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}
            
        groupdata.append(data)

    infer = models.GroupInference_modelB(agents, groupdata)
    loss, params = infer.infer_posterior(iter_steps=150, num_particles = 10)
    inference_df = infer.sample_posterior()
    pickle.dump( (lr_day1_true, \
                  theta_Q_day1_true, \
                  theta_rep_day1_true, \
                  lr_day2_true, \
                  theta_Q_day2_true, \
                  theta_rep_day2_true, \
                  inference_df), open(f"param_recov/groupinference/model{model}/k_{k}/group_inference.p", "wb" ) )
        
    pickle.dump( loss, open(f"param_recov/groupinference/model{model}/k_{k}/ELBO_group_inference.p", "wb" ) )
    
#%%
"Plot ELBO"
plt.plot(loss)