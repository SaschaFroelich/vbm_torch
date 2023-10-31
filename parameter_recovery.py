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

# sequence = ([1]*(num_agents//2))
# sequence.extend(([2]*(num_agents//2)))

# blockorder = np.ones((1, num_agents//4), dtype=int)
# blockorder = np.concatenate((blockorder,np.ones((1, num_agents//4), dtype=int)*2),axis=1)
# blockorder = np.concatenate((blockorder,blockorder),axis=1).tolist()[0]
# Qs = torch.tensor([[0.2, 0., 0., 0.2],
#                    [0, 0.2, 0.2, 0.]]).tile((num_agents, 1, 1))
# Q_init = Qs[range(num_agents),torch.tensor(sequence)-1, :]

Qs = torch.tensor([[0.2, 0., 0., 0.2],
                   [0.2, 0., 0., 0.2],
                   [0, 0.2, 0.2, 0],
                   [0, 0.2, 0.2, 0.]]).tile((num_agents, 1, 1))
Q_init = Qs[range(num_agents),torch.tensor(group), :]

groupdata_dict, group_behav_df, _, params_sim_df = utils.simulate_data(model, 
                                                                      num_agents,
                                                                      Q_init = Q_init,
                                                                      group = group)

utils.plot_grouplevel(group_behav_df)


#%%
'''
Inference
'''

"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent(model, 
                         Q_init, 
                         num_agents = num_agents)

print("===== Starting inference =====")
"----- Start Inference"
infer = inferencemodels.GeneralGroupInference(agent, groupdata_dict)
infer.infer_posterior(iter_steps = 2_000, num_particles = 10)

"----- Sample parameter estimates from posterior"
post_sample_df = infer.sample_posterior()
post_sample_df['group'] = post_sample_df['ag_idx'].map(lambda x: groupdata_dict['group'][0][x])
post_sample_df['model'] = [model]*len(post_sample_df)

"----- Save results to file"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
pickle.dump((post_sample_df, params_sim_df, group_behav_df, infer.loss, agent.param_names), open(f"parameter_recovery/param_recov_{timestamp}.p", "wb" ) )

#%%
'''
Analysis
'''
#%%
"----- Open Files"
# import sys
# sys.modules[__name__].__dict__.clear() # Clear variables

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

res=pickle.load(open( filenames[0], "rb" ))
if len(res) == 5:
    post_sample_df, params_sim_df, group_behav_df, loss, param_names = res
    
elif len(res) == 3:
    all_params_df, loss, param_names = res
    all_params_df
    fig, ax = plt.subplots()
    plt.plot(loss)
    plt.title("ELBO")
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("ELBO")
    plt.show()
    
    for param in param_names:
        fig, ax = plt.subplots()
        sns.scatterplot(x=param + '_true', y=param, data=all_params_df)
        plt.plot(all_params_df[param+'_true'], all_params_df[param+'_true'])
        plt.show()

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

all_params_df = pd.concat((params_sim_df, pd.DataFrame(post_sample_df.iloc[:, 0:-2].groupby(['ag_idx'], as_index = False).mean())), axis = 1)

for param in param_names:
    fig, ax = plt.subplots()
    sns.scatterplot(x=param + '_sim', y=param + '_postsample', data=all_params_df)
    plt.plot(params_sim_df[param+'_sim'], params_sim_df[param+'_sim'])
    plt.show()


#%%
'''
Simulate data from inferred parameters
'''
groupdata, params, params_df = utils.simulate_data(model, 
                                                   num_agents, 
                                                   Q_init = Q_init,
                                                   blockorder = [1]*num_agents,
                                                   params = torch.tensor(np.array(params_sim_df.iloc[:, 0:len(param_names)]).T))