#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:26:59 2023

Simulation and parallel group-level inference.

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

plt.style.use("classic")

#%%
# model = sys.argv[1]
# resim =  int(sys.argv[2]) # whether to simulate agents with inferred parameters
# method = sys.argv[3] # "svi" or "mcmc"
# num_agents = 50

model = 'original'
resim =  0 # whether to simulate agents with inferred parameters
method = 'svi' # "svi" or "mcmc"
num_agents = 40
# k = 4.
print(f"Running model {model}")

#%%
'''
Simulate data
'''

if resim:
    raise Exception("Not implemented yet, buddy!")
    
"----- Simulate data"
Q_init=[0.2, 0., 0., 0.2]
groupdata, params, params_df = utils.simulate_data(model, num_agents, Q_init = Q_init)
newgroupdata = utils.comp_groupdata(groupdata, for_ddm = 0)



#%%
'''
Inference
'''

"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent(model, 
                         Q_init, 
                         num_agents = num_agents, 
                         params = params)

# df_true = pd.DataFrame({'lr_day1_true' : lr_day1_true,
#                         'theta_Q_day1_true' : theta_Q_day1_true,
#                         'theta_rep_day1_true' : theta_rep_day1_true,
                        
#                         'lr_day2_true' : lr_day2_true,
#                         'theta_Q_day2_true' : theta_Q_day2_true,
#                         'theta_rep_day2_true' : theta_rep_day2_true})


print("===== Starting inference =====")
"----- Start Inference"
infer = inferencemodels.GeneralGroupInference(agent, num_agents, newgroupdata)
infer.infer_posterior(iter_steps = 12_500, num_particles = 10)

"----- Sample parameter estimates from posterior"
inference_df = infer.sample_posterior()
df = inference_df.groupby(['subject']).mean()

"----- Save results to file"
df_all = pd.concat([df, params_df], axis = 1)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# pickle.dump( infer.loss, open(f"parameter_recovery/loss_{timestamp}.p", "wb" ) )
pickle.dump( (df_all, infer.loss, agent.param_names), open(f"parameter_recovery/param_recov_{timestamp}.p", "wb" ) )

#%%
'''
Analysis
'''

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
# root.destroy()

#%%
'''
Plot ELBO and Parameter Estimates
'''
df_all, infer.loss, param_names = pickle.load(open( filenames[0], "rb" ))

fig, ax = plt.subplots()
plt.plot(infer.loss)
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



