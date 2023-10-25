#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 15:03:15 2023

Parameter Recovery Analysis to load from pickle files.

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

model = 'F'
limited_range = 0
k = 4.
version = 1

if model == 'original':
    npar = 3

elif model == 'A':
    "Daywise"
    npar = 6

elif model == 'B' or model == 'B_3':
    "Daywise. Parameters: lr_day1, theta_Q_day1, theta_rep_day1, lr_day2, theta_Q_day2, theta_rep_day2"
    npar = 6

elif model == 'B_2':
    npar = 9

elif model == 'C':
    "Daywise with decaying learning rate. Parameters: lr0_day1, lr_lambda_day1, theta_Q_day1, theta_rep_day1,  lr0_day2, lr_lambda_day2, theta_Q_day2, theta_rep_day2"
    npar = 8
    
elif model == 'Bayesianprior':
    npar = 6
    
elif model == 'D':
    npar = 8

elif model == 'D_simple':
    npar = 2
    
elif model == 'F':
    npar = 8

"Get data from pickle-files"

datadir = f'/home/sascha/Desktop/vb_model/vbm_torch/param_recov/model{model}/k_{k}/'

"Prior"
prior_df = pickle.load( open(os.fsencode(datadir +  f'version{version}_' +  "prior.p"), "rb" ) )

df = pd.DataFrame()

for file in os.listdir(datadir):
     filename = os.fsdecode(file)
     if filename.endswith(".p") and "ELBO" not in filename and "prior" not in filename and f'version{version}' in filename:
         # print("opening %s"%(directory + file))
         
         pickle_df = pickle.load( open(datadir + file, "rb" ) )
                  
         df = pd.concat((df, pickle_df))
         
"ELBOs"
fig, ax = plt.subplots()
for file in os.listdir(datadir):
     filename = os.fsdecode(file)
     if filename.endswith(".p") and "ELBO" in filename and f'version{version}' in filename:
         # print("opening %s"%(directory + file))
         
         loss = pickle.load( open(datadir +  file, "rb" ) )

         ax.plot(loss)
         
print("========================================================")
for col in prior_df.columns:
    if col == 'Q_init':
        print("%s prior = "%col)
        print(prior_df[col][0])
    else:
        print("%s prior = %d"%(col, prior_df[col][0]))

plt.title("ELBO")
ax.set_ylabel("ELBO")
ax.set_xlabel("iteration steps")
plt.savefig(datadir + f"version{version}_ELBO.png")
plt.show()

#%%
"Plot Parameter Recovery with pandas dataframe"
plt.style.use("classic")

fig, ax = plt.subplots(2, -(npar//-2), figsize=(20,20))

param = 0
for row in range(2):
    for col in range(-(npar//-2)):
        if param < npar:
            cmap = df.iloc[:, 3*param+2]
            im0 = ax[row, col].scatter(df.iloc[:, 3*param], df.iloc[:, 3*param+1], c=cmap)
            ax[row, col].plot(df.iloc[:, 3*param], df.iloc[:, 3*param])
            #ax[0].set_title("Parameter Recovery for lr")
            ax[row, col].set_xlabel(df.iloc[:, 3*param].name, fontsize = 30.0)
            # ax[row, col].set_ylabel(df.iloc[:, 3*param+1].name[4:], fontsize = 30.0)
            
            r,p = scipy.stats.pearsonr(df.iloc[:, 3*param], df.iloc[:, 3*param+1])
            print(f"Correlation for {df.iloc[:, 3*param].name} with {df.iloc[:, 3*param+1].name} : {r}, p = {p}")
        
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
            
            ax[row, col].set_aspect('equal')
    
            param += 1
    
fig.savefig(datadir + f"version{version}_param_recov.png")
plt.show()

#%%
if model == 'B' or model == 'B_onlydual':
    df["ratio Q_R Day 2"] = df.apply(lambda x: x["inf_theta_Q_day2"]/x["inf_theta_rep_day2"], axis = 1)
    df["ratio Q_R Day 1"] = df.apply(lambda x: x["inf_theta_Q_day1"]/x["inf_theta_rep_day1"], axis = 1)
    
    df["true ratio Q_R Day 2"] = df.apply(lambda x: x["theta_Q_day2"]/x["theta_rep_day2"], axis = 1)
    df["true ratio Q_R Day 1"] = df.apply(lambda x: x["theta_Q_day1"]/x["theta_rep_day1"], axis = 1)
    
    fig, ax = plt.subplots(1,2)
    ax[0].scatter(df["true ratio Q_R Day 1"] , df["ratio Q_R Day 1"] )
    ax[0].plot(df["true ratio Q_R Day 1"] , df["true ratio Q_R Day 1"] )
    
    ax[1].scatter(df["true ratio Q_R Day 2"] , df["ratio Q_R Day 2"] )
    ax[1].plot(df["true ratio Q_R Day 2"] , df["true ratio Q_R Day 2"] )
    
    ax[0].set_xlim([-1, 10])
    ax[0].set_ylim([-1, 10])
    ax[0].set_ylabel(r'Inferred Ratio $\theta_Q$ /$\theta_R$')
    
    ax[1].set_xlim([-1, 10])
    ax[1].set_ylim([-1, 10])
    ax[1].set_xlabel(r'Ratio $\theta_Q$ /$\theta_R$')
    
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    plt.show()
    
elif model == 'B_2' or model == 'B_3':
    
    df["ratio Q_R Day 1:1"] = df.apply(lambda x: x["inf_theta_Q_day1_1"]/x["inf_theta_rep_day1_1"], axis = 1)
    df["ratio Q_R Day 1:2"] = df.apply(lambda x: x["inf_theta_Q_day1_2"]/x["inf_theta_rep_day1_2"], axis = 1)
    df["ratio Q_R Day 2"] = df.apply(lambda x: x["inf_theta_Q_day2"]/x["inf_theta_rep_day2"], axis = 1)
    
    df["true ratio Q_R Day 1:1"] = df.apply(lambda x: x["theta_Q_day1_1"]/x["theta_rep_day1_1"], axis = 1)
    df["true ratio Q_R Day 1:2"] = df.apply(lambda x: x["theta_Q_day1_2"]/x["theta_rep_day1_2"], axis = 1)
    df["true ratio Q_R Day 2"] = df.apply(lambda x: x["theta_Q_day2"]/x["theta_rep_day2"], axis = 1)
    
    fig, ax = plt.subplots(1,3)
    ax[0].scatter(df["true ratio Q_R Day 1:1"] , df["ratio Q_R Day 1:1"] )
    ax[0].plot(df["true ratio Q_R Day 1:1"] , df["true ratio Q_R Day 1:1"] )

    ax[1].scatter(df["true ratio Q_R Day 1:2"] , df["ratio Q_R Day 1:2"] )
    ax[1].plot(df["true ratio Q_R Day 1:2"] , df["true ratio Q_R Day 1:2"] )
    
    ax[2].scatter(df["true ratio Q_R Day 2"] , df["ratio Q_R Day 2"] )
    ax[2].plot(df["true ratio Q_R Day 2"] , df["true ratio Q_R Day 2"] )
    
    ax[0].set_xlim([-1, 12])
    ax[0].set_ylim([-1, 12])
    ax[0].set_ylabel(r'Inferred Ratio $\theta_Q$ /$\theta_R$')
    
    ax[1].set_xlim([-1, 12])
    ax[1].set_ylim([-1, 12])
    ax[1].set_xlabel(r'Ratio $\theta_Q$ /$\theta_R$')
    
    ax[2].set_xlim([-1, 12])
    ax[2].set_ylim([-1, 12])
    
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[2].set_aspect('equal')
    plt.show()

#%%
"Plot Parameter Recovery with pandas dataframe"
plt.style.use("classic")

fig, ax = plt.subplots(npar, 1, figsize=(20,15))

for param in range(npar):
    cmap = df.iloc[:, 3*param+2]
    im0 = ax[param].scatter(df.iloc[:, 3*param], df.iloc[:, 3*param+1], c=cmap)
    ax[param].plot(df.iloc[:, 3*param], df.iloc[:, 3*param])
    #ax[0].set_title("Parameter Recovery for lr")
    ax[param].set_xlabel(df.iloc[:, 3*param].name)
    ax[param].set_ylabel(df.iloc[:, 3*param+1].name[4:])
    
    r,p = scipy.stats.pearsonr(df.iloc[:, 3*param], df.iloc[:, 3*param+1])
    print(f"Correlation for {df.iloc[:, 3*param].name} with {df.iloc[:, 3*param+1].name} : {r}, p = {p}")

    #ax[0].annotate("omega 0.05, dectemp 1.6", xy=(true_lrs[14],inf_lrs[14]), xytext=(0.4, -0.1), arrowprops={"arrowstyle":"->", "color":"gray"})
    #ax[0].annotate("omega 0.11, dectemp 2.2", xy=(true_lrs[25],inf_lrs[25]), xytext=(-0.1, 0.8), arrowprops={"arrowstyle":"->", "color":"gray"})
    #ax[0].annotate("omega 0.74, dectemp 0.4", xy=(true_lrs[36],inf_lrs[36]), xytext=(0.2, 1.), arrowprops={"arrowstyle":"->", "color":"gray"})
    #divider = make_axes_locatable(ax[0])
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, ax=ax[param], orientation='vertical', location='right', anchor=(0.,0.5))

    chartBox = ax[param].get_position()
    ax[param].set_position([chartBox.x0, 
                      chartBox.y0,
                      chartBox.width,
                      chartBox.height * 0.9])

plt.show()
#%% 

if model == 'original':
    "Plot inferred lr color-coded by dectemp value"
    fig, ax = plt.subplots()
    
    cmap = df.iloc[:, 0]
    # cmap = ['r' if c < 2.5 else 'g' for c in cmap]
    # scatter plot results
    im0 = ax.scatter(df.iloc[:, 3], df.iloc[:, 4], c=cmap)
    # plot line
    ax.plot(df.iloc[:, 3], df.iloc[:, 3])
    #ax[0].set_title("Parameter Recovery for lr")
    ax.set_xlabel("lr")
    ax.set_ylabel("inferred lr")
    

    #ax[0].annotate("omega 0.05, dectemp 1.6", xy=(true_lrs[14],inf_lrs[14]), xytext=(0.4, -0.1), arrowprops={"arrowstyle":"->", "color":"gray"})
    #ax[0].annotate("omega 0.11, dectemp 2.2", xy=(true_lrs[25],inf_lrs[25]), xytext=(-0.1, 0.8), arrowprops={"arrowstyle":"->", "color":"gray"})
    #ax[0].annotate("omega 0.74, dectemp 0.4", xy=(true_lrs[36],inf_lrs[36]), xytext=(0.2, 1.), arrowprops={"arrowstyle":"->", "color":"gray"})
    #divider = make_axes_locatable(ax[0])
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, ax=ax, orientation='vertical', location='right', anchor=(0.,0.5))

    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, 
                      chartBox.y0,
                      chartBox.width,
                      chartBox.height * 0.9])


#%%
"Check if inferred variables are correlated : Both Days"

# true_lrs, inf_lrs, true_omegas, inf_omegas, true_dectemps, inf_dectemps = pickle.load( open( "40_param_recovs.p", "rb" ) )
df_inf = df.iloc[:, [3*param + 1 for param in range(npar)]]

def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)

g = sns.PairGrid(df_inf)
g.map(sns.scatterplot)

sns.set(style='white', font_scale=1.6)
g = sns.PairGrid(df_inf, aspect=1.4, diag_sharey=False)
g.map_lower(sns.regplot, ci=False, line_kws={'color': 'black'})
g.map_diag(sns.distplot, kde_kws={'color': 'black'})
g.map_upper(corrdot)

g.savefig("bla.svg")

#%%

if limited_range:
    "Check limited range correlations"
    if model == 'original':
        r,p = scipy.stats.pearsonr(df_inf[(df_inf["inf_dectemp"]<7) & (df_inf["inf_lr"]<0.15)]["inf_dectemp"], \
                                   df_inf[(df_inf["inf_dectemp"]<7) & (df_inf["inf_lr"]<0.15)]["inf_lr"])
            
        r,p = scipy.stats.pearsonr(df_inf[(df_inf["inf_dectemp"]<7) & (df_inf["inf_omega"]>0.5)]["inf_dectemp"], \
                                   df_inf[(df_inf["inf_dectemp"]<7) & (df_inf["inf_omega"]>0.5)]["inf_omega"])
            
        r,p = scipy.stats.pearsonr(df_inf[(df_inf["inf_omega"]>0.5) & (df_inf["inf_lr"]<0.15)]["inf_dectemp"], \
                                   df_inf[(df_inf["inf_omega"]>0.5) & (df_inf["inf_lr"]<0.15)]["inf_lr"])

#%%
"Parameter Differences Day 1 -> Day 2"

from scipy.stats import ttest_rel

if model == 'A':
    res = ttest_rel(df["inf_dectemp_day1"], df["inf_dectemp_day2"])
    
    res = ttest_rel(df["inf_lr_day1"], df["inf_lr_day2"])
    
    res = ttest_rel(df["inf_omega_day1"], df["inf_omega_day2"])
    
elif model == "B":
    res = ttest_rel(df["inf_lr_day1"], df["inf_lr_day2"])
    
    res = ttest_rel(df["inf_theta_Q_day1"], df["inf_theta_Q_day2"])
    
    res = ttest_rel(df["inf_theta_rep_day1"], df["inf_theta_rep_day2"])

#%%
if 0:
    "Check if inferred variables are correlated : Day 1"
    
    # true_lrs, inf_lrs, true_omegas, inf_omegas, true_dectemps, inf_dectemps = pickle.load( open( "40_param_recovs.p", "rb" ) )
    df_inf = df.iloc[:, [3*param + 1 for param in range(npar)]]
    
    def corrdot(*args, **kwargs):
        corr_r = args[0].corr(args[1], 'pearson')
        corr_text = f"{corr_r:2.2f}".replace("0.", ".")
        ax = plt.gca()
        ax.set_axis_off()
        marker_size = abs(corr_r) * 10000
        ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
                   vmin=-1, vmax=1, transform=ax.transAxes)
        font_size = abs(corr_r) * 40 + 5
        ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                    ha='center', va='center', fontsize=font_size)
    
    g = sns.PairGrid(df_inf)
    g.map(sns.scatterplot)
    
    sns.set(style='white', font_scale=1.6)
    g = sns.PairGrid(df_inf, aspect=1.4, diag_sharey=False)
    g.map_lower(sns.regplot, ci=False, line_kws={'color': 'black'})
    g.map_diag(sns.distplot, kde_kws={'color': 'black'})
    g.map_upper(corrdot)

#%%
if 0:
    "Do inferred lr with low variance have high dec temp?"
    
    if model == 'original':
        r,p = scipy.stats.pearsonr(lr_variance, dectemp)
        
        fig, ax = plt.subplots()
        ax.scatter(lr_variance, dectemp)
        ax.set_xlabel("lr variance")
        ax.set_ylabel("decision temperature")
        plt.show()
    
    
#%%

