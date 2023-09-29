#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:00:28 2023

Script for inferring experimental data on group level

@author: sascha
"""

import ipdb
import pickle
import pandas as pd
import numpy as np

import sys
import glob

import env

import models_torch as models
import utils

remote = 0
k = 4.

# assert(remote)

if remote:
    model = sys.argv[1]
    num_reps = int(sys.argv[2]) # How often to repeat the parameter inference per participant (0 to infer just once)
    published_results = int(sys.argv[3])
    
else:
    #file_day1 = '/home/sascha/Desktop/vb_model/vbm_torch/behav_data/published/Grp1/csv/it6_00420_Tag1_Grp1.mat' # Complete path
    file_day1 = '/home/sascha/Desktop/vb_model/vbm_torch/behav_data/Grp1/csv/it6_5b5e0e86902ad10001cfcc59_Tag1_Grp1.mat' # Complete path
    group = 0
    model = 'B'
    num_reps = 0
    published_results = 0
    
if published_results:
    if remote:
        data_dir = "/home/sascha/Desktop/vb_model/torch/behav_data/published/"
    else: 
        data_dir = "/home/sascha/Desktop/vb_model/vbm_torch/behav_data/published/"
else:
    if remote:
        data_dir = "/home/sascha/Desktop/vb_model/torch/behav_data/"
    else:
        data_dir = "/home/sascha/Desktop/vb_model/vbm_torch/behav_data/"
        
if model == 'B':
    npar = 6
    
    agents = []
    groupdata = []
    
    pb = -1
    for group in range(4):
        files_day1 = glob.glob(data_dir + "Grp%d/csv/*Tag1*.mat"%(group+1))
        
        for file1 in files_day1:
            "Loop over participants"
            pb += 1
            data, _ = utils.get_participant_data(file1, group, data_dir, remote = remote, published_results = published_results)
            groupdata.append(data)

            if group == 0 or group == 1:
                Q_init=[0.2, 0., 0., 0.2]

            elif group == 2 or group == 3:
                Q_init=[0., 0.2, 0.2, 0.]

            "--- Initialize new agent with whatever parameter values ---"
            parameter = np.random.uniform(0,1, npar)
            lr_day1 = parameter[0]*0.1
            theta_Q_day1 = parameter[1]*6
            theta_rep_day1 = parameter[2]*6
            
            lr_day2 = parameter[0]*0.1
            theta_Q_day2 = parameter[1]*6
            theta_rep_day2 = parameter[2]*6
        
            newagent = models.Vbm_B(lr_day1 = lr_day1, \
                                  theta_Q_day1 = theta_Q_day1, \
                                  theta_rep_day1 = theta_rep_day1, \
                                  lr_day2 = lr_day2, \
                                  theta_Q_day2 = theta_Q_day2, \
                                  theta_rep_day2 = theta_rep_day2, \
                                  k=k,\
                                  Q_init=Q_init)
    
            agents.append(newagent)

    infer = models.GroupInference_modelB(agents, groupdata)
    loss, params = infer.infer_posterior(iter_steps=250, num_particles = 10)
    inference_df = infer.sample_posterior()
    pickle.dump( inference_df, open(f"behav_fit/groupinference/model{model}/k_{k}/group_inference.p", "wb" ) )
    pickle.dump( loss, open(f"behav_fit/groupinference/model{model}/k_{k}/ELBO_group_inference.p", "wb" ) )