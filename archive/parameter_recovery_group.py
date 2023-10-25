#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:52:20 2023

Script to analyse the parameter recovery at group level

@author: sascha
"""

import ipdb
import torch
import pyro
import pyro.distributions as dist
from tqdm import tqdm
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

import numpy
import scipy

import models_torch as models
import env 

import pickle
import os

model = 'B'
# limited_range = 0
# onlydual = 0
k = 4.

if model == 'original':
    npar = 3

elif model == 'B':
    npar = 6

directory = os.fsencode(f'/home/sascha/Desktop/vb_model/vbm_torch/param_recov/groupinference/model{model}/k_{k}/')

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".p") and "ELBO" not in filename and "prior" not in filename:
         print("opening %s"%(directory + file))
         
         if model == 'B':
             (lr_day1_true, \
              theta_Q_day1_true, \
              theta_rep_day1_true, \
              lr_day2_true, \
              theta_Q_day2_true, \
              theta_rep_day2_true, \
              inference_df) = pickle.load( open(directory + file, "rb" ) )
                 
         elif model == 'original':
             (dectemp_true, lr_true, omega_true, inference_df) = pickle.load( open(directory + file, "rb" ) )
             
                           
"ELBOs"
fig, ax = plt.subplots()
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".p") and "ELBO" in filename:
         print("opening %s"%(directory + file))
         
         loss = pickle.load( open(directory + file, "rb" ) )

         ax.plot(loss)
         
plt.title("ELBO")
ax.set_ylabel("ELBO")
ax.set_xlabel("iteration steps")
plt.show()

#%%
num_agents = inference_df.subject.max()+1

if model == 'B':
    "Plot Parameter Recovery with pandas dataframe"
    df = pd.DataFrame({"lr_day1":[], \
                       "inf_lr_day1":[],  \
                           
                       "theta_Q_day1":[], \
                       "inf_theta_Q_day1":[], \
                           
                       "theta_rep_day1":[], \
                       "inf_theta_rep_day1":[], \
                      
                      "lr_day2":[], \
                      "inf_lr_day2":[],  \
                       
                      "theta_Q_day2":[], \
                      "inf_theta_Q_day2":[], \
                       
                      "theta_rep_day2":[], \
                      "inf_theta_rep_day2":[]})
    
    for ag in range(num_agents):
        df = pd.concat((df, pd.DataFrame({"lr_day1":[lr_day1_true[ag]], \
                                    "inf_lr_day1":[inference_df[inference_df["subject"]==ag].lr_day1.mean()], \
                                        
                                    "theta_Q_day1":[theta_Q_day1_true[ag]], \
                                    "inf_theta_Q_day1":[inference_df[inference_df["subject"]==ag].theta_Q_day1.mean()], \
                                        
                                    "theta_rep_day1":[theta_rep_day1_true[ag]], \
                                    "inf_theta_rep_day1":[inference_df[inference_df["subject"]==ag].theta_rep_day1.mean()], \
                                    
                                    "lr_day2":[lr_day2_true[ag]], \
                                    "inf_lr_day2":[inference_df[inference_df["subject"]==ag].lr_day2.mean()], \
                                        
                                    "theta_Q_day2":[theta_Q_day2_true[ag]], \
                                    "inf_theta_Q_day2":[inference_df[inference_df["subject"]==ag].theta_Q_day2.mean()], \
                                        
                                    "theta_rep_day2":[theta_rep_day2_true[ag]], \
                                    "inf_theta_rep_day2":[inference_df[inference_df["subject"]==ag].theta_rep_day2.mean()]})))

elif model == 'original':
    "Plot Parameter Recovery with pandas dataframe"
    df = pd.DataFrame({"dectemp":[], "inf_dectemp":[],  "omega":[], "inf_omega":[], "lr":[], "inf_lr":[]})
    for ag in range(num_agents):
        df = pd.concat((df, pd.DataFrame({"dectemp":[dectemp_true[ag]], \
                                    "inf_dectemp":[inference_df[inference_df["subject"]==ag].dectemp.mean()], \
                                    "omega":[omega_true[ag]], \
                                    "inf_omega":[inference_df[inference_df["subject"]==ag].omega.mean()], \
                                    "lr":[lr_true[ag]], \
                                    "inf_lr":[inference_df[inference_df["subject"]==ag].lr.mean()]})))
    
plt.style.use("classic")

fig, ax = plt.subplots(2, -(npar//-2), figsize=(20,20))

param = 0
for row in range(2):
    for col in range(-(npar//-2)):
        if param < npar:
            # ipdb.set_trace()
            "Plot parameter recovery scatterplot"
            im0 = ax[row, col].scatter(df.iloc[:, 2*param], df.iloc[:, 2*param+1])
            "Plot diagonal line"
            ax[row, col].plot(df.iloc[:, 2*param], df.iloc[:, 2*param])
            #ax[0].set_title("2*parameter Recovery for lr")
            ax[row, col].set_xlabel(df.iloc[:, 2*param].name, fontsize = 30.0)
            ax[row, col].set_ylabel(df.iloc[:, 2*param+1].name[4:], fontsize = 30.0)
            
            # r,p = scipy.stats.pearsonr(df.iloc[:, 2*param], df.iloc[:, 2*param+1])
            # print(f"Correlation for {df.iloc[:, 2*param].name} with {df.iloc[:, 2*param+1].name} : {r}, p = {p}")
        
            #ax[0].annotate("omega 0.05, dectemp 1.6", xy=(true_lrs[14],inf_lrs[14]), xytext=(0.4, -0.1), arrowprops={"arrowstyle":"->", "color":"gray"})
            #ax[0].annotate("omega 0.11, dectemp 2.2", xy=(true_lrs[25],inf_lrs[25]), xytext=(-0.1, 0.8), arrowprops={"arrowstyle":"->", "color":"gray"})
            #ax[0].annotate("omega 0.74, dectemp 0.4", xy=(true_lrs[36],inf_lrs[36]), xytext=(0.2, 1.), arrowprops={"arrowstyle":"->", "color":"gray"})
            #divider = make_axes_locatable(ax[0])
            #cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im0, ax=ax[row, col], orientation='vertical', location='right', shrink = 0.4, anchor = (0, 0.4))
        
            chartBox = ax[row, col].get_position()
            ax[row, col].set_position([chartBox.x0, 
                              chartBox.y0,
                              chartBox.width,
                              chartBox.height * 0.9])
            
            #ax[row, col].set_aspect('equal')
    
            param += 1
    
plt.show()

#%%
for ag in range(num_agents):
    sns.kdeplot(inference_df[inference_df['subject'] == ag]['dectemp'])
    plt.plot([dectemp_true[ag], dectemp_true[ag]], [0,1])
    plt.title("Participant %d"%ag)
    plt.show()