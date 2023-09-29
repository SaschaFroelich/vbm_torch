#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:34:51 2023

Parameter Analysis for Behavioural Fits.

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
from scipy.stats import ttest_rel

import os

import models_torch as models
import env 
import analysis as anal
import utils

import pickle

#%%

model = 'B'
published = 0
k = 4.
version=1

if model == 'original':
    npar = 3

elif model == 'A':
    "Daywise"
    npar = 6

elif model == 'B' or model == 'B_onlydual' or model == 'B_3':
    "Daywise. Parameters: lr_day1, theta_Q_day1, theta_rep_day1,  lr_day2, theta_Q_day2, theta_rep_day2"
    npar = 6

elif model == 'B_2':
    npar = 9

elif model == 'C':
    "Daywise with decaying learning rate. Parameters: lr0_day1, lr_lambda_day1, theta_Q_day1, theta_rep_day1,  lr0_day2, lr_lambda_day2, theta_Q_day2, theta_rep_day2"
    npar = 8

"Get data from pickle-files"
if published:
    directory = f'/home/sascha/Desktop/vb_model/vbm_torch/behav_fit/published/model{model}/k_{k}/'

else:
    directory = f'/home/sascha/Desktop/vb_model/vbm_torch/behav_fit/model{model}/k_{k}/'

df = pd.DataFrame()

for file in os.listdir(os.fsencode(directory)):
     filename = os.fsdecode(file)
     if filename.endswith(".p") and "ELBO" not in filename and f"version{version}" in filename and "prior" not in filename:
         # print("opening %s"%(os.fsencode(directory) + file))
         
         pickle_df = pickle.load( open(os.fsencode(directory) + file, "rb" ) )
         
         assert(version < 10)
         pickle_df["participant"] = filename[21:-2]
                  
         df = pd.concat((df, pickle_df))

df = df.reset_index()
df.drop(columns={'index'}, inplace = True)

"ELBOs"
for file in os.listdir(os.fsencode(directory)):
     filename = os.fsdecode(file)
     if filename.endswith(".p") and "ELBO" in filename and f"version{version}" in filename and "prior" not in filename:
         # print("opening %s"%(os.fsencode(directory) + file))
         
         loss = pickle.load( open(os.fsencode(directory) + file, "rb" ) )

         plt.plot(loss)

plt.ylabel('ELBO')
plt.xlabel('iteration step')
plt.show()

"Prior"
prior_df = pickle.load( open(os.fsencode(directory +  f'version{version}_' +  "prior.p"), "rb" ) )
print("========================================================")
for col in prior_df.columns:
    if col == 'Q_init':
        print("%s prior = "%col)
        print(prior_df[col][0])
    else:
        print("%s prior = %d"%(col, prior_df[col][0]))
    
    
"Save inferred parameters in a single pickle-file"
if published:
    pickle.dump( df, open(f"model{model}_pub_param_anal.p", "wb" ) )         
else:
    pickle.dump( df, open(f"model{model}_param_anal.p", "wb" ) )         

#%%
"Prepare for violin plot"
parameter = []
inferred = []
variance = []

for row in range(len(df)):
    for par in range(npar):
        parameter.append(df.columns[2*par][4:])
        inferred.append(df.iloc[row, 2*par])
        variance.append(df.iloc[row, 2*par+1])

data_new = {"parameter": parameter, "inferred": inferred, "variance": variance}
df_viol = pd.DataFrame(data=data_new)

"violin plot"
anal.violin(df_viol, with_colbar = 0)

#%%

"Compare theta_Q and theta_rep"
if model == 'B' or model == 'B_onlydual':
    inf_tq_tr_day1 = (df["inf_theta_Q_day1"]/df["inf_theta_rep_day1"]).tolist()
    inf_tq_tr_day2 = (df["inf_theta_Q_day2"]/df["inf_theta_rep_day2"]).tolist()
    
    parameter = ["thetaQ_thetaR_day1"]*len(inf_tq_tr_day1)
    parameter.extend(["thetaQ_thetaR_day2"]*len(inf_tq_tr_day2))
    
    inferred = inf_tq_tr_day1
    inferred.extend(inf_tq_tr_day2)
    
    # variance = [0]*len(inferred)
    
    df_tqtr = pd.DataFrame(data={"parameter":parameter, "inferred": inferred})
    sns.violinplot(x="parameter", y="inferred", data=df_tqtr, color=".8")
    
    sns.stripplot(x="parameter", \
                  y="inferred", \
                  edgecolor = 'gray', \
                  linewidth = 1, \
                  data=df_tqtr, \
                  jitter=True, \
                  palette="coolwarm")

    # anal.violin(df_tqtr)

#%%
"Q/R"

parameter = []
inferred = []
variance = []

if model == 'B' or model == 'B_onlydual':
    df["ratio Q_R Day 1"] = df.apply(lambda x: x["inf_theta_Q_day1"]/x["inf_theta_rep_day1"], axis = 1)
    df["ratio Q_R Day 2"] = df.apply(lambda x: x["inf_theta_Q_day2"]/x["inf_theta_rep_day2"], axis = 1)
    
    for row in range(len(df)):
        for par in range(2):
            parameter.append("Q/R Day 1")
            inferred.append(df.loc[df.index[row], 'ratio Q_R Day 1']) 
            
            parameter.append("Q/R Day 2")
            inferred.append(df.loc[df.index[row], 'ratio Q_R Day 2']) 
    
elif model == 'B_2' or model == 'B_3':
    df["ratio Q_R Day 1:1"] = df.apply(lambda x: x["inf_theta_Q_day1_1"]/x["inf_theta_rep_day1_1"], axis = 1)
    df["ratio Q_R Day 1:2"] = df.apply(lambda x: x["inf_theta_Q_day1_2"]/x["inf_theta_rep_day1_2"], axis = 1)
    df["ratio Q_R Day 2"] = df.apply(lambda x: x["inf_theta_Q_day2"]/x["inf_theta_rep_day2"], axis = 1)
    
    for row in range(len(df)):
        for par in range(2):
            parameter.append("Q/R Day 1:1")
            inferred.append(df.loc[df.index[row], 'ratio Q_R Day 1:1']) 
            
            parameter.append("Q/R Day 1:2")
            inferred.append(df.loc[df.index[row], 'ratio Q_R Day 1:2']) 
            
            parameter.append("Q/R Day 2")
            inferred.append(df.loc[df.index[row], 'ratio Q_R Day 2']) 
    
    
df_viol_QR = pd.DataFrame({"parameter": parameter, "inferred":inferred})
anal.violin(df_viol_QR, with_colbar = 0, sharey = True)


#%% 
"Correlation plot"
colidx = [2*x for x in range(npar)]

"Correlations"
anal.param_corr(df.iloc[:, colidx])

#%% 
"Parameter Differences Day 1 -> Day 2"

print("Need Cohen's d for repeated measures!")

if model == 'A':
    res = ttest_rel(df["inf_dectemp_day1"], df["inf_dectemp_day2"])    
    utils.cohens_d(df["inf_dectemp_day1"], df["inf_dectemp_day2"])
    
    res = ttest_rel(df["inf_lr_day1"], df["inf_lr_day2"])    
    utils.cohens_d(df["inf_lr_day1"], df["inf_lr_day2"])
    
    res = ttest_rel(df["inf_omega_day1"], df["inf_omega_day2"])    
    utils.cohens_d(df["inf_omega_day1"], df["inf_omega_day2"])
    
elif model == "B":
    res = ttest_rel(df["inf_lr_day1"], df["inf_lr_day2"])
    utils.cohens_d(df["inf_lr_day1"], df["inf_lr_day2"])
    
    res = ttest_rel(df["inf_theta_Q_day1"], df["inf_theta_Q_day2"])
    utils.cohens_d(df["inf_theta_Q_day1"], df["inf_theta_Q_day2"])
    
    res = ttest_rel(df["inf_theta_rep_day1"], df["inf_theta_rep_day2"])
    utils.cohens_d(df["inf_theta_rep_day1"], df["inf_theta_rep_day2"])
    
#%% 
"Simulate inferred"
if published:
    data_dir = "/home/sascha/Desktop/vb_model/vbm_torch/behav_data/published/"
else:
    data_dir = "/home/sascha/1TB/TRR/AST_repo/Analyses/clipre/Data/Online_Erhebung/"

anal.simulate_inferred(data_dir = data_dir, 
                       df = df, 
                       model = model, 
                       k=k, 
                       published_results = published, 
                       Q_init = prior_df.Q_init.item())

#%%
"Compute log likelihood and ELPD and WAIC (https://docs.pyro.ai/en/stable/_modules/pyro/ops/stats.html)"

