#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:26:59 2023

Parameter Recovery 
1. Simulation
2. Group-Level Inference
3. Analysis

@author: sascha
"""

# import sys
# sys.modules[__name__].__dict__.clear()

import ipdb

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import torch

import scipy

import models_torch as models

import env
import utils

import inferencemodels
# import inferencemodel_sarah as inferencemodels

import sys
from datetime import datetime
import pickle

import numpy as np 

#%%

model = 'Conflict'
resim =  0 # whether to simulate agents with inferred parameters
method = 'svi' # "svi" or "mcmc"
num_agents = 52

assert num_agents%4 == 0, "num_agents must be divisible by 4."
# k = 4.
print(f"Running model {model}")

#%%
'''
Simulate data in parallel
'''

group = [0]*(num_agents//4)
group.extend([1]*(num_agents//4))
group.extend([2]*(num_agents//4))
group.extend([3]*(num_agents//4))

params = {'lr_day1' : 0*torch.ones(num_agents),
          'theta_Q_day1' : 4*torch.ones(num_agents),
          'theta_rep_day1' : 1*torch.ones(num_agents),
          'conflict_param_day1' : -10.*torch.ones(num_agents),
          
          'lr_day2' : 0*torch.ones(num_agents),
          'theta_Q_day2' : 1*torch.ones(num_agents),
          'theta_rep_day2' : 4*torch.ones(num_agents),
          'conflict_param_day2' : 0.*torch.ones(num_agents)}

groupdata_dict, group_behav_df, _, params_sim_df = utils.simulate_data(model, 
                                                                      num_agents,
                                                                      group = group,
                                                                      params = params)

utils.plot_grouplevel(group_behav_df)


#%%
'''
Inference
'''

"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent(model, 
                         group, 
                         num_agents = num_agents)

print("===== Starting inference =====")
"----- Start Inference"
infer = inferencemodels.GeneralGroupInference(agent, groupdata_dict)
infer.infer_posterior(iter_steps = 16_000, num_particles = 10)

"----- Sample parameter estimates from posterior"
post_sample_df = infer.sample_posterior()
post_sample_df['group'] = post_sample_df['ag_idx'].map(lambda x: groupdata_dict['group'][0][x])
post_sample_df['model'] = [model]*len(post_sample_df)

"----- Save results to file"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
pickle.dump((post_sample_df, group_behav_df, infer.loss, params_sim_df), open(f"parameter_recovery/param_recov_model_{model}_{timestamp}.p", "wb" ) )

#%%
'''
Analysis
'''
#%%
"----- Open Files"
# import sys
# sys.modules[__name__].__dict__.clear() # Clear variables

import utils
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import pickle
import tkinter as tk
from tkinter import filedialog

def open_files():
    global filenames
    filenames = filedialog.askopenfilenames()
    print(f'File paths: {filenames}')
    # return filenames
    root.destroy()

root = tk.Tk()
button = tk.Button(root, text="Open Files", command=open_files)
print(button)
button.pack()
root.mainloop()

res = pickle.load(open( filenames[0], "rb" ))
post_sample_df, exp_data_df, loss, params_df = res

# params_df['paramtype'] = ['sim']*len(params_df)

inf_mean_df = pd.DataFrame(post_sample_df.groupby(['model','ag_idx','group'], as_index = False).mean())
# inf_mean_df['paramtype'] = ['inf']*len(params_df)
model = inf_mean_df['model'][0]
num_agents = len(inf_mean_df['ag_idx'].unique())
    
# post_sample_df, params_sim_df, group_behav_df, loss, param_names = pickle.load(open( filenames[0], "rb" ))
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

all_params_df = pd.concat((params_df, inf_mean_df), axis = 0)

for param in params_df.columns[0:-3]:
    fig, ax = plt.subplots()
    plt.scatter(params_df[param], inf_mean_df[param])
    plt.plot(params_df[param], params_df[param])
    ax.set_xlabel('true value')
    ax.set_ylabel('inferred value')
    plt.title(param)
    plt.show()

#%%
'''
Posterior Predictives
'''
_ = utils.posterior_predictives(post_sample_df, exp_data = exp_data_df)

#%%
'''
Simulate only from means
'''
groupdata_dict, group_behav_df, params_sim, params_sim_df = utils.simulate_data(model, 
                                                                        num_agents,
                                                                        group = list(inf_mean_df['group']),
                                                                        params = inf_mean_df)

utils.plot_grouplevel(exp_data_df, group_behav_df, plot_single = True)

#%%
'''
Test Correlation between Parameters within and across participants
'''

"----- Within"