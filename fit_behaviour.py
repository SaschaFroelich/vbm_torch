#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:05:13 2023

Fit model to behaviour.

@author: sascha
"""
import torch
import pandas as pd
from datetime import datetime
import pickle

import analysis as anal
import inferencemodels
import utils

'''
Modelle:
Vbm
B
Conflict
'''

model = 'Conflict'
#%%
exp_behav_dict, exp_behav_df = utils.get_groupdata('/home/sascha/Desktop/vbm_torch/behav_data/')

utils.plot_grouplevel(exp_behav_df)

num_agents = len(exp_behav_dict['trialsequence'][0])
#%%
'''
Prepare Inference
'''
group = exp_behav_dict['group'][0]

'''
Inference
'''

"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent(model, 
                         group, 
                         num_agents = num_agents)

print("===== Starting inference =====")
"----- Start Inference"
infer = inferencemodels.GeneralGroupInference(agent, exp_behav_dict)
infer.infer_posterior(iter_steps = 16_000, num_particles = 10)

"----- Sample parameter estimates from posterior"
post_sample_df = infer.sample_posterior()
post_sample_df['group'] = post_sample_df['ag_idx'].map(lambda x: exp_behav_dict['group'][0][x])
post_sample_df['model'] = [model]*len(post_sample_df)

"----- Save results to file"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
params_sim_df = pd.DataFrame(columns = agent.par_dict.keys())
for col in params_sim_df.columns:
    params_sim_df[col] = ['unknown']
pickle.dump( (post_sample_df, exp_behav_df, infer.loss, params_sim_df), open(f"behav_fit/behav_fit_model_{model}_{timestamp}.p", "wb" ) )

#%%
'''
Analysis
'''
#%%
"----- Open Files"
# import sys
# sys.modules[__name__].__dict__.clear() # Clear variables
# del model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import utils
import torch
import tkinter as tk
from tkinter import filedialog
import pickle

def open_files():
    global filenames
    filenames = filedialog.askopenfilenames()
    print(f'File paths: {filenames}')
    root.destroy()
    
root = tk.Tk()
button = tk.Button(root, text="Open Files", command=open_files)
print(button)
button.pack()
root.mainloop()

# post_sample_df, df, loss, param_names = pickle.load(open( filenames[0], "rb" ))
res = pickle.load(open( filenames[0], "rb" ))
post_sample_df, expdata_df, loss, params_df = res
inf_mean_df = pd.DataFrame(post_sample_df.groupby(['model', 'ag_idx', 'group'], as_index = False).mean())
model = post_sample_df['model'][0]
num_agents = len(post_sample_df['ag_idx'].unique())
num_params = len(params_df.columns)

#%%
'''
Plot ELBO
'''
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots()
plt.plot(loss)
plt.title("ELBO")
ax.set_xlabel("Number of iterations")
ax.set_ylabel("ELBO")
plt.show()

#%%
'''
Plot ELBO and Parameter Estimates
'''

fig, ax = plt.subplots()
plt.plot(loss)
plt.title("ELBO")
ax.set_xlabel("Number of iterations")
ax.set_ylabel("ELBO")
plt.show()

for param in params_df.columns:
    fig, ax = plt.subplots()
    sns.kdeplot(post_sample_df[param])
    # plt.plot(df[param+'_true'], df[param+'_true'])
    plt.show()

#%%
'''
Violin Plot
'''
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# plt.style.use("seaborn-v0_8-dark")

num_params = len(post_sample_df.columns[0:-3])

fig, ax = plt.subplots(1, num_params, figsize=(15,5), sharey=0)

if model == 'B':
    ylims = [[0, 0.03], # lr
             [0.5, 7.5], # theta_Q
             [0., 8.], # theta_rep
             [0, 0.03], # lr
             [0.5, 7.5], # theta_Q
             [0., 8]] # theta_rep
    
elif model == 'Conflict'or model =='conflictmodel':
    ylims = [[0, 0.03], # lr
             [0.5, 7.5], # theta_Q
             [0.5, 5], # theta_rep
             [-0.4, 0.5], # conflict param
             [0, 0.03], # lr
             [0.5, 7.5], # theta_Q
             [0.5, 5], # theta_rep
             [-0.5, 0.5]] # conflict param
    

for par in range(num_params):
    
    if 1:    
        "With colorbar"
        "ax[0]"
        dataseries = (inf_mean_df.melt()[inf_mean_df.melt()['variable'] == params_df.columns[par]])
        dataseries['value'] = pd.to_numeric(dataseries['value'], errors='coerce')
        
        sns.violinplot(ax = ax[par], 
                       x = 'variable',
                       y = 'value',
                       data = dataseries,
                       color=".8")
        
        sns.stripplot(x = 'variable',
                      y = 'value',
                      data = dataseries,
                      edgecolor = 'gray',
                      linewidth = 1,
                      jitter=True,
                      ax=ax[par])
                      # palette="coolwarm")
        
        ax[par].legend([],[], frameon=False)
        
        "Position"
        chartBox = ax[par].get_position()
        ax[par].set_position([chartBox.x0+par/64,
                          chartBox.y0,
                          chartBox.width,
                          chartBox.height])
        
        ax[par].set_ylim(ylims[par])
    
        "Colorbar"
        # variance = df[params_df.columns[par]].std()**2
        
        # normalize = mcolors.TwoSlopeNorm(vcenter=(min(variance)+max(variance))/2, 
        #                                  vmin=min(variance), 
        #                                  vmax=max(variance))
        
        # colormap = cm.coolwarm
        # scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        # scalarmappaple.set_array(variance)
        # plt.colorbar(scalarmappaple, ax = ax[par])
        
    else:
        "Without colorbar"
        "ax[0]"
        g1 = sns.violinplot(ax=ax[par], 
                            x="parameter", 
                            y="inferred", 
                            data=inf_mean_df[inf_mean_df["parameter"]==inf_mean_df["parameter"].unique()[par]], 
                            color=".8")
        
        g2 = sns.stripplot(x="parameter",
                      y="inferred",
                      edgecolor = 'gray',
                      linewidth = 1,
                      data = inf_mean_df[inf_mean_df["parameter"]==inf_mean_df["parameter"].unique()[par]],
                      jitter = True,
                      ax = ax[par])
            
        if par > 0:
            g1.set(ylabel=None)
            g2.set(ylabel=None)
        ax[par].legend([],[], frameon=False)

plt.show()

#%%
'''
Plot Experimental Data
'''
utils.plot_grouplevel(expdata_df, plot_single = False)

#%%
'''
Simulate only from means
'''
groupdata_dict, group_behav_df, params_sim, params_sim_df = utils.simulate_data(model, 
                                                                        num_agents,
                                                                        group = list(inf_mean_df['group']),
                                                                        params = inf_mean_df)

utils.plot_grouplevel(expdata_df, group_behav_df, plot_single = True)
# utils.plot_grouplevel(expdata_df, plot_single = True)


#%%
'''
Posterior Predictives
'''
complete_df = utils.posterior_predictives(post_sample_df, exp_data = expdata_df)

#%%