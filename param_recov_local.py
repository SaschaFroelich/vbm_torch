#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:40:16 2023

For local parameter recovery

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

import models_torch as models
import env 

import pickle

plt.style.use("classic")

#%%
"Import pickle data"
true_lrs, inferred_lrs, true_omegas, inferred_omegas, true_dectemps, inferred_dectemps = pickle.load( open( "40_param_recovs_June2nd_2023_2.p", "rb" ) )

#%%
"Parameter Recovery (Single Inference)"

true_dectemps = []
inferred_dectemps = []
dectemp_variance = []

true_lrs = []
inferred_lrs = []
lr_variance = []

true_omegas = []
inferred_omegas = []
omega_variance = []

all_datas = {}

for i in range(40):
    num_blocks = 14

    parameter = numpy.random.uniform(0,1, 3)
    omega = parameter[0]
    lr = parameter[1]
    dectemp = parameter[2]*5

    true_omegas.append(omega)
    true_lrs.append(lr)
    true_dectemps.append(dectemp)
    
    newagent = models.vbm(omega=omega, dectemp=dectemp, lr=lr, k=4, Q_init=[0.4, 0., 0., 0.4], num_blocks=num_blocks) # change per console
    newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8])
    
    newenv.run()
    data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
            "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
                "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}

    all_datas["sim_%d"%i] = data
        
    infer = models.SingleInference(newagent, data)
    loss, params = infer.infer_posterior(iter_steps=200, num_particles = 10)
    
    fig, ax = plt.subplots()
    plt.plot(loss)
    plt.show()
    
    inferred_dectemps.append(params["conc_dectemp"][0]/params["rate_dectemp"][0])
    dectemp_variance.append(models.gamma_variance(params["conc_dectemp"][0], params["rate_dectemp"][0]))
    
    inferred_lrs.append(params["alpha_lr"][0]/(params["alpha_lr"][0]+params["beta_lr"][0]))
    lr_variance.append(models.beta_variance(params["alpha_lr"][0], params["beta_lr"][0]))
    
    inferred_omegas.append(params["alpha_omega"][0]/(params["alpha_omega"][0]+params["beta_omega"][0]))
    omega_variance.append(models.beta_variance(params["alpha_omega"][0], params["beta_omega"][0]))
    
    print("True value : Inferred value")
    print("dectemp %.3f : %.3f"%(dectemp, inferred_dectemps[-1]))
    print("lr %.3f : %.3f"%(lr, inferred_lrs[-1]))
    print("omega %.3f : %.3f"%(omega, inferred_omegas[-1]))
  
pickle_data = (true_lrs, inferred_lrs, true_omegas, inferred_omegas, true_dectemps, inferred_dectemps)
pickle.dump( pickle_data, open( "40_param_recovs_June2nd_2023_1.p", "wb" ) ) # change per console
  
#%%
"Parameter Recovery (Group Inference)"

true_lrs = []

true_omegas = []

true_dectemps = []

all_datas = {}
group_data = []
agents = []

for i in range(40):
    num_blocks = 14

    parameter = numpy.random.uniform(0,1, 3)
    omega = parameter[0]
    lr = parameter[1]
    dectemp = parameter[2]*5

    true_omegas.append(omega)
    true_lrs.append(lr)
    true_dectemps.append(dectemp)
    
    newagent = models.vbm(omega=omega, dectemp=dectemp, lr=lr, k=4, Q_init=[0.4, 0., 0., 0.4], num_blocks=num_blocks) # change per console
    newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8])
    
    newenv.run()
    data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
            "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
                "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}

    all_datas["sim_%d"%i] = data
    
    agents.append(newagent)
    group_data.append(data)

infer = models.GroupInference(agents, group_data)
loss, params = infer.infer_posterior(iter_steps=400, num_particles = 10)

fig, ax = plt.subplots()
plt.plot(loss)
plt.show()

inferred_lrs = torch.sigmoid(torch.tensor(params["m_locs"][:, 0]))
# lr_variance = []

inferred_omegas = torch.sigmoid(torch.tensor(params["m_locs"][:, 1]))
# omega_variance = []

inferred_dectemps = torch.exp(torch.tensor(params["m_locs"][:, 1]))
# dectemp_variance = []

  
pickle_data = (true_lrs, inferred_lrs, true_omegas, inferred_omegas, true_dectemps, inferred_dectemps)
pickle.dump( pickle_data, open( "40_group_param_recovs.p", "wb" ) ) # change per console

#%%
"Sample from Posterior (only for group inference)"

num_samples = 500
sample_df = infer.sample_posterior(n_samples=num_samples)

plt.figure()
sns.displot(data=sample_df, x='lr', hue="subject", kde=True)
plt.xlim([0,1])
plt.title("lr of each subject")
plt.show()

plt.figure()
sns.displot(data=sample_df, x='omega', hue="subject", kde=True)
plt.xlim([0,1])
plt.title("omega of each subject")
plt.show()

plt.figure()
sns.displot(data=sample_df, x='dectemp', hue="subject", kde=True)
# plt.xlim([0,1])
plt.title("dectemp of each subject")
plt.show()

#%%
"Get data from different pickle-files (parameters lr, omega & dectemp)"
import os

directory = os.fsencode('/home/sascha/Desktop/vb_model/vbm_torch/Parameter_Recoveries/pickle/')
    
true_lrs = []
inferred_lrs = []
lr_variance = []

true_omegas = []
inferred_omegas = []
omega_variance = []

true_dectemps = []
inferred_dectemps = []
dectemp_variance = []

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".p"):
         print("opening %s"%(directory + file))
         lr, inferred_lr, lr_variance, omega, inferred_omega, omega_variance, dectemp, \
             inferred_dectemp, dectemp_variance = pickle.load( open(directory + file, "rb" ) )
             
         true_lrs.append(lr)
         inferred_lrs.append(inferred_lr)
         lr_variance.append(lr_variance)
        
         true_omegas.append(omega)
         inferred_omegas.append(inferred_omega)
         omega_variance.append(omega_variance)
        
         true_dectemps.append(dectemp)
         inferred_dectemps.append(inferred_dectemp)
         dectemp_variance.append(dectemp_variance)

#%%
"Plot Parameter Recovery (Single Inference)"

#cmap = numpy.arange(0, 40)
#plt.scatter(range(0,40), range(0,40), c=cmap)
#plt.colorbar()

#fig, ax = plt.subplots(3,1, figsize=(15,30))
fig, ax = plt.subplots(3,1)
cmap = lr_variance
im0 = ax[0].scatter(true_lrs, inferred_lrs, c=cmap)
ax[0].plot(true_lrs, true_lrs)
#ax[0].set_title("Parameter Recovery for lr")
ax[0].set_xlabel("True lr")
ax[0].set_ylabel("Inferred lr")

#ax[0].annotate("omega 0.05, dectemp 1.6", xy=(true_lrs[14],inferred_lrs[14]), xytext=(0.4, -0.1), arrowprops={"arrowstyle":"->", "color":"gray"})
#ax[0].annotate("omega 0.11, dectemp 2.2", xy=(true_lrs[25],inferred_lrs[25]), xytext=(-0.1, 0.8), arrowprops={"arrowstyle":"->", "color":"gray"})
#ax[0].annotate("omega 0.74, dectemp 0.4", xy=(true_lrs[36],inferred_lrs[36]), xytext=(0.2, 1.), arrowprops={"arrowstyle":"->", "color":"gray"})
#divider = make_axes_locatable(ax[0])
#cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, ax=ax[0], orientation='vertical', location='right', anchor=(0.,0.5))

chartBox = ax[0].get_position()
ax[0].set_position([chartBox.x0, 
                  chartBox.y0,
                  chartBox.width,
                  chartBox.height * 0.9])

cmap = dectemp_variance
im1 = ax[1].scatter(true_dectemps, inferred_dectemps, c=cmap)
ax[1].plot(true_dectemps, true_dectemps)
#ax[1].set_title("Parameter Recovery for dectemp")
ax[1].set_xlabel("True dectemp")
ax[1].set_ylabel("Inferred dectemp")
fig.colorbar(im1, ax=ax[1], orientation='vertical', location='right', anchor=(0.,0.5))
chartBox = ax[1].get_position()
ax[1].set_position([chartBox.x0, 
                  chartBox.y0,
                  chartBox.width,
                  chartBox.height * 0.9])

cmap = omega_variance
im2 = ax[2].scatter(true_omegas, inferred_omegas, c=cmap)
ax[2].plot(true_omegas, true_omegas)
#ax[2].set_title("Parameter Recovery for omega")
ax[2].set_xlabel("True Omega")
ax[2].set_ylabel("Inferred Omega")
fig.colorbar(im2, ax=ax[2], orientation='vertical', location='right', anchor=(0.,0.5))

chartBox = ax[2].get_position()
ax[2].set_position([chartBox.x0, 
                  chartBox.y0,
                  chartBox.width,
                  chartBox.height * 0.9])

plt.show()

#%%

"Plot Parameter Recovery (Group Inference)"

#cmap = numpy.arange(0, 40)
#plt.scatter(range(0,40), range(0,40), c=cmap)
#plt.colorbar()

#fig, ax = plt.subplots(3,1, figsize=(15,30))
fig, ax = plt.subplots(3,1)
#cmap = lr_variance
im0 = ax[0].scatter(true_lrs, inferred_lrs)
ax[0].plot(true_lrs, true_lrs)
#ax[0].set_title("Parameter Recovery for lr")
ax[0].set_xlabel("True lr")
ax[0].set_ylabel("Inferred lr")

"""
#divider = make_axes_locatable(ax[0])
#cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, ax=ax[0], orientation='vertical', location='right', anchor=(0.,0.5))

chartBox = ax[0].get_position()
ax[0].set_position([chartBox.x0, 
                  chartBox.y0,
                  chartBox.width,
                  chartBox.height * 0.9])

cmap = dectemp_variance
"""
im1 = ax[1].scatter(true_dectemps, inferred_dectemps)
ax[1].plot(true_dectemps, true_dectemps)
#ax[1].set_title("Parameter Recovery for dectemp")
ax[1].set_xlabel("True dectemp")
ax[1].set_ylabel("Inferred dectemp")
#fig.colorbar(im1, ax=ax[1], orientation='vertical', location='right', anchor=(0.,0.5))
chartBox = ax[1].get_position()
ax[1].set_position([chartBox.x0, 
                  chartBox.y0,
                  chartBox.width,
                  chartBox.height * 0.9])

#cmap = omega_variance
im2 = ax[2].scatter(true_omegas, inferred_omegas)
ax[2].plot(true_omegas, true_omegas)
#ax[2].set_title("Parameter Recovery for omega")
ax[2].set_xlabel("True Omega")
ax[2].set_ylabel("Inferred Omega")
#fig.colorbar(im2, ax=ax[2], orientation='vertical', location='right', anchor=(0.,0.5))

chartBox = ax[2].get_position()
ax[2].set_position([chartBox.x0, 
                  chartBox.y0,
                  chartBox.width,
                  chartBox.height * 0.9])

plt.show()


#%%

import scipy
r,p = scipy.stats.pearsonr(true_lrs, inferred_lrs)
r,p = scipy.stats.pearsonr(true_dectemps, inferred_dectemps)
r,p = scipy.stats.pearsonr(true_omegas, inferred_omegas)

#%%
"Check if inferred variables are correlated"

# true_lrs, inferred_lrs, true_omegas, inferred_omegas, true_dectemps, inferred_dectemps = pickle.load( open( "40_param_recovs.p", "rb" ) )

data_sim = {"learning rate": true_lrs, "Decision Temp": true_dectemps, "omega": true_omegas}
df_sim = pd.DataFrame(data=data_sim)

data_inf = {"learning rate": inferred_lrs, "Decision Temp": inferred_dectemps, "omega": inferred_omegas}
df_inf = pd.DataFrame(data=data_inf)

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

r,p = scipy.stats.pearsonr(df_inf["learning rate"], df_inf["omega"])
r,p = scipy.stats.pearsonr(df_inf["learning rate"], df_inf["Decision Temp"])
r,p = scipy.stats.pearsonr(df_inf["omega"], df_inf["Decision Temp"])

#%%

"Simulate inferred parameters"
for sim in range(len(true_omegas)):
    omega_true = true_omegas[sim]
    dectemp_true = true_dectemps[sim]
    lr_true = true_lrs[sim]
    
    omega_inf = inferred_omegas[sim]
    dectemp_inf = inferred_dectemps[sim]
    lr_inf = inferred_lrs[sim]

    "--- Given data ---"
    data_sim = all_datas["sim_%d"%sim]
    
    "--- Inferred data simulation ---"
    newagent = models.vbm(omega=omega_inf, dectemp=dectemp_inf, lr=lr_inf, k=4, Q_init=[0., 0., 0., 0.], num_blocks=num_blocks) # change per console
    newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8]) 
    
    newenv.run()
    data_inf = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
            "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
                "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}

    "--- Plot ---"
    models.plot_results(data_sim, data_inf, omega_true=omega_true, omega_inf=omega_inf, \
                 dectemp_true=dectemp_true, dectemp_inf=dectemp_inf,\
                     lr_true=lr_true, lr_inf=lr_inf, group=0)
        
#%%

import scipy
r,p = scipy.stats.pearsonr(df_inf["learning rate"], df_inf["omega"])
r_sim,p_sim = scipy.stats.pearsonr(df_sim["learning rate"], df_sim["omega"])