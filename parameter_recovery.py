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
# model = sys.argv[1]
# resim =  int(sys.argv[2]) # whether to simulate agents with inferred parameters
# method = sys.argv[3] # "svi" or "mcmc"
# num_agents = 50

model = 'conflictmodel'
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

sequence = ([1]*(num_agents//2))
sequence.extend(([2]*(num_agents//2)))

blockorder = np.ones((1, num_agents//4), dtype=int)
blockorder = np.concatenate((blockorder,np.ones((1, num_agents//4), dtype=int)*2),axis=1)
blockorder = np.concatenate((blockorder,blockorder),axis=1).tolist()[0]

Qs = torch.tensor([[0.2, 0., 0., 0.2],
                   [0, 0.2, 0.2, 0.]]).tile((num_agents, 1, 1))
Q_init = Qs[range(num_agents),torch.tensor(sequence)-1, :]

groupdata_dict, params, params_df = utils.simulate_data(model, 
                                                    num_agents,
                                                   Q_init = Q_init,
                                                    sequence = sequence,
                                                    blockorder = blockorder)

groupdata_df = pd.DataFrame(groupdata_dict).explode(list(groupdata_dict.keys()))

utils.plot_grouplevel(groupdata_df)

#%%
'''
Inference
'''

"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent(model, 
                         Q_init, 
                         num_agents = num_agents)

# df_true = pd.DataFrame({'lr_day1_true' : lr_day1_true,
#                         'theta_Q_day1_true' : theta_Q_day1_true,
#                         'theta_rep_day1_true' : theta_rep_day1_true,
                        
#                         'lr_day2_true' : lr_day2_true,
#                         'theta_Q_day2_true' : theta_Q_day2_true,
#                         'theta_rep_day2_true' : theta_rep_day2_true})

print("===== Starting inference =====")
"----- Start Inference"
infer = inferencemodels.GeneralGroupInference(agent, groupdata_dict)
infer.infer_posterior(iter_steps = 10_000, num_particles = 10)

"----- Sample parameter estimates from posterior"
post_sample = infer.sample_posterior()
df = post_sample.groupby(['subject']).mean()
post_sample['group'] = post_sample['subject'].map(lambda x: groupdata_dict['group'][0][x])
post_sample['model'] = [model]*len(post_sample)

"----- Save results to file"
df_all = pd.concat([df, params_df], axis = 1)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
pickle.dump((post_sample, df_all, groupdata_df, infer.loss, agent.param_names), open(f"parameter_recovery/param_recov_{timestamp}.p", "wb" ) )

#%%
'''
Analysis
'''
#%%
"----- Open Files"
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

post_sample, df_all, groupdata_df, infer_loss, param_names = pickle.load(open( filenames[0], "rb" ))
# res = pickle.load(open( filenames[0], "rb" ))
#%%
'''
Plot ELBO and Parameter Estimates
'''

fig, ax = plt.subplots()
plt.plot(infer_loss)
plt.title("ELBO")
ax.set_xlabel("Number of iterations")
ax.set_ylabel("ELBO")
plt.show()

for param in param_names:
    fig, ax = plt.subplots()
    sns.scatterplot(x=param + '_true', y=param, data=df_all)
    plt.plot(df_all[param+'_true'], df_all[param+'_true'])
    plt.show()


#%%
'''
Simulate data from inferred parameters
'''
groupdata, params, params_df = utils.simulate_data(model, 
                                                   num_agents, 
                                                   Q_init = Q_init,
                                                   blockorder = [1]*num_agents,
                                                   params = torch.tensor(np.array(df_all.iloc[:, 0:len(param_names)]).T))