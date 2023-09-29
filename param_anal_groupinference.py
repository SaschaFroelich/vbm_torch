#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:25:50 2023

Parameter Analysis for Behavioural Fits with group inference.

@author: sascha
"""

import numpy as np
import pandas as pd
import ipdb
import os
import matplotlib.pyplot as plt
import pickle

import analysis as anal

model = 'B'
published = 0
k = 4.
version=4

datadir = f'behav_fit/groupinference/model{model}/k_{k}/'

if model == 'B' or model == 'B_onlydual':
    param_names = ["lr_day1", "theta_Q_day1", "theta_rep_day1", "lr_day2", "theta_Q_day2", "theta_rep_day2"]

"ELBO"
fig, ax = plt.subplots()
for file in os.listdir(datadir):
      filename = os.fsdecode(file)
      if filename.endswith(".p") and "ELBO" in filename and f"version{version}" in filename:# and f'version{version}' in filename:
        loss = pickle.load( open(datadir +  file, "rb" ) )
        
        ax.plot(loss)
        plt.title(f"ELBO for model {model}")
        ax.set_ylabel("ELBO")
        ax.set_xlabel("iteration steps")
        # plt.savefig(datadir + f"version{version}_ELBO.png")
        plt.show()

df = pickle.load( open(datadir +  f"group_inference_version{version}.p", "rb" ) )
df_mean = df.groupby('subject').mean()

"Prepare for violin plot"
parameter = []
inferred = []
variance = []

for row in range(len(df_mean)):
    for par in range(len(df_mean.columns)):
        parameter.append(df_mean.columns[par])
        inferred.append(df_mean.iloc[row, par])
        variance.append(np.nan)

data_new = {"parameter": parameter, "inferred": inferred, "variance": variance}
df_viol = pd.DataFrame(data=data_new)

"violin plot"
anal.violin(df_viol, with_colbar = 0)

#%%
# "Plot theta_Q/theta_rep"
# if model == 'B' or model =='B_onlydual':
#     parameter = []
#     inferred = []
#     variance = []
    
#     for row in range(len(df_mean)):
#         for day in [1,2]:
#             parameter.append(f'theta Q/R Day {day}')
#             colidx_Q =  df_mean.columns.get_loc(f'inf_theta_Q_day{day}')
#             colidx_rep =  df_mean.columns.get_loc(f'inf_theta_rep_day{day}')
#             inferred.append(df_mean.iloc[row, colidx_Q] / df_mean.iloc[row, colidx_rep])
#             variance.append(np.nan)

# data_new = {"parameter": parameter, "inferred": inferred, "variance": variance}
# df_viol = pd.DataFrame(data=data_new)

# "violin plot"
# anal.violin(df_viol, with_colbar = 0, sharey=True)
#%%
"Simulate inferred"
if published:
    data_dir = "/home/sascha/Desktop/vb_model/vbm_torch/behav_data/published/"
else:
    data_dir = "/home/sascha/1TB/TRR/AST_repo/Analyses/clipre/Data/Online_Erhebung/"

df_mean = df_mean.reset_index()
df_mean.rename(columns={"subject": "participant"}, inplace = True)

for col in df_mean.columns:
    if col != "participant":
        df_mean.rename(columns={col: "inf_"+col}, inplace = True)

print("Simulate !")
anal.simulate_inferred(data_dir = data_dir, \
                       df = df_mean, \
                       model = model, \
                       k=k, \
                       published_results = published, \
                       Q_init = [0.2, 0.0, 0.0, 0.2])
