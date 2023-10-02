#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:26:59 2023

Simulation and parallel group-level inference.

@author: sascha
"""

# import sys
# sys.modules[__name__].__dict__.clear()

import ipdb

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import torch

import numpy
import scipy

import models_torch as models

import env
import utils

import inferencemodels
# import inferencemodel_sarah as inferencemodels

import sys
from datetime import datetime
import pickle

plt.style.use("classic")

#%%
# model = sys.argv[1]
# resim =  int(sys.argv[2]) # whether to simulate agents with inferred parameters
# method = sys.argv[3] # "svi" or "mcmc"
# num_agents = 50

model = 'original'
resim =  0 # whether to simulate agents with inferred parameters
method = 'svi' # "svi" or "mcmc"
num_agents = 20
k = 4.
print(f"Running model {model}")

#%%

if resim:
    raise Exception("Not implemented yet, buddy!")
    
if model == 'original':
    npar = 3
    
    agents = []
    groupdata = []
    
    omega_true = []
    dectemp_true = []
    lr_true = []

    Q_init_group = []
    
    "Simulate with random parameters"
    for ag in range(num_agents):
        print("Simulating agent no. %d"%ag)
        parameter = numpy.random.uniform(0,1, npar)
        omega = parameter[1]
        dectemp = (parameter[2]+1)*3
        lr = parameter[0]*0.1
        
        omega_true.append(omega)
        dectemp_true.append(dectemp)
        lr_true.append(lr)
        
        Q_init = [0.2, 0., 0., 0.2]
        Q_init_group.append(Q_init)
        newagent = models.Vbm(omega = torch.tensor([[omega]]),
                              dectemp = torch.tensor([[dectemp]]),
                              lr = torch.tensor([[lr]]),
                              k=torch.tensor([k]),
                              Q_init=torch.tensor([[Q_init]]))
            
        newenv = env.Env(newagent, 
                         rewprobs=[0.8, 0.2, 0.2, 0.8], 
                         matfile_dir = './matlabcode/clipre/')
        
        newenv.run()
        
        data = {"Choices": newenv.choices, 
                "Outcomes": newenv.outcomes,
                "Trialsequence": newenv.data["trialsequence"], 
                "Blocktype": newenv.data["blocktype"],
                "Jokertypes": newenv.data["jokertypes"], 
                "Blockidx": newenv.data["blockidx"]}
            
        utils.plot_results(pd.DataFrame(data), group = 0)
            
        groupdata.append(data)
   
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
    
    Q_init_group = []
    
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

        Q_init = [0.2, 0., 0., 0.2]
        Q_init_group.append(Q_init)
        newagent = models.Vbm_B(lr_day1 = torch.tensor([[lr_day1]]),
                              theta_Q_day1 = torch.tensor([[theta_Q_day1]]),
                              theta_rep_day1 = torch.tensor([[theta_rep_day1]]),
                                  
                              lr_day2 = torch.tensor([[lr_day2]]),
                              theta_Q_day2 = torch.tensor([[theta_Q_day2]]),
                              theta_rep_day2 = torch.tensor([[theta_rep_day2]]),
                              k=torch.tensor([k]),
                              Q_init=torch.tensor([[Q_init]]))
            
        newenv = env.Env(newagent, 
                         rewprobs=[0.8, 0.2, 0.2, 0.8], 
                         matfile_dir = './matlabcode/clipre/')
        
        newenv.run()
        
        data = {"Choices": newenv.choices, 
                "Outcomes": newenv.outcomes,
                "Trialsequence": newenv.data["trialsequence"], 
                "Blocktype": newenv.data["blocktype"],
                "Jokertypes": newenv.data["jokertypes"], 
                "Blockidx": newenv.data["blockidx"]}
            
        utils.plot_results(pd.DataFrame(data), group = 0)
            
        groupdata.append(data)  

elif model == 'testmodel':
    npar = 2
    
    agents = []
    groupdata = []
    
    prob1_true = []
    prob2_true = []
    
    "Simulate with random parameters"
    for ag in range(num_agents):
        print("Simulating agent no. %d"%ag)
        parameter = numpy.random.uniform(0,1, npar)
        prob1 = parameter[0]
        prob2 = parameter[1]
        
        prob1_true.append(prob1)
        prob2_true.append(prob2)

        newagent = models.testmodel(prob1 = torch.tensor([[prob1]]), 
                                    prob2 = torch.tensor([[prob2]]))
            
        newenv = env.Env(newagent, 
                         rewprobs=[0.8, 0.2, 0.2, 0.8], 
                         matfile_dir = './matlabcode/clipre/')
        
        newenv.run()
        
        data = {"Choices": newenv.choices, 
                "Outcomes": newenv.outcomes,
                "Trialsequence": newenv.data["trialsequence"], 
                "Blocktype": newenv.data["blocktype"],
                "Jokertypes": newenv.data["jokertypes"], 
                "Blockidx": newenv.data["blockidx"]}
            
        utils.plot_results(pd.DataFrame(data), group = 0)
            
        groupdata.append(data)  

newgroupdata = utils.comp_groupdata(groupdata, for_ddm = 0)

if model == 'original':
    agent = models.Vbm(omega = torch.tensor([omega_true]),
                       dectemp = torch.tensor([dectemp_true]),
                       lr = torch.tensor([lr_true]),
                       k = torch.tensor(k),
                       Q_init = torch.tensor([Q_init_group]))

elif model == 'B':
    agent = models.Vbm_B(lr_day1 = torch.tensor([lr_day1_true]),
                          theta_Q_day1 = torch.tensor([theta_Q_day1_true]),
                          theta_rep_day1 = torch.tensor([theta_rep_day1_true]),

                          lr_day2 = torch.tensor([lr_day2_true]),
                          theta_Q_day2 = torch.tensor([theta_Q_day2_true]),
                          theta_rep_day2 = torch.tensor([theta_rep_day2_true]),
                          k = torch.tensor(k),
                          Q_init = torch.tensor([Q_init_group]))
        
elif model == 'testmodel':
    agent = models.testmodel(prob1 = torch.tensor([prob1_true]), prob2 = torch.tensor([prob2_true]))

print("===== Starting inference =====")
infer = inferencemodels.GeneralGroupInference(agent, num_agents, newgroupdata)
# loss, params = infer.infer_posterior(iter_steps=100, num_particles = 10)
infer.infer_posterior(iter_steps=250, num_particles = 10)
inference_df = infer.sample_posterior()
# pickle.dump( (lr_day1_true, \
#               theta_Q_day1_true, \
#               theta_rep_day1_true, \
#               lr_day2_true, \
#               theta_Q_day2_true, \
#               theta_rep_day2_true, \
#               inference_df), open(f"param_recov/groupinference/model{model}/k_{k}/group_inference.p", "wb" ) )

# pickle.dump( loss, open(f"param_recov/groupinference/model{model}/k_{k}/ELBO_group_inference.p", "wb" ) )

df = inference_df.groupby(['subject']).mean()
#%%
import numpy as np
# (lr_day1_true, theta_Q_day1_true, \
#               theta_rep_day1_true, \
#               lr_day2_true, \
#               theta_Q_day2_true, \
#               theta_rep_day2_true, \
#               inference_df) = pickle.load( open(f"param_recov/groupinference/model{model}/k_{k}/group_inference.p", "rb" ) )
    
# loss = pickle.load( open(f"param_recov/groupinference/model{model}/k_{k}/ELBO_group_inference.p", "rb" ) )

"Plot ELBO"
plt.plot(infer.loss)
plt.show()


# for par in range(npar):
# plt.scatter(lr_day1_true, df.iloc[])