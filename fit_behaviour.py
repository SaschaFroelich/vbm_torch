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
Q_init = torch.cat((torch.tensor([[0.2, 0., 0., 0.2]]).tile((num_agents//2, 1)),
                    torch.tensor([[0, 0.2, 0.2, 0.]]).tile((num_agents//2, 1))))

'''
Inference
'''

"----- Initialize new agent object with num_agents agents for inference"
agent = utils.init_agent(model, 
                         Q_init, 
                         num_agents = num_agents)

print("===== Starting inference =====")
"----- Start Inference"
infer = inferencemodels.GeneralGroupInference(agent, exp_behav_dict)
infer.infer_posterior(iter_steps = 3, num_particles = 10)

"----- Sample parameter estimates from posterior"
post_sample_df = infer.sample_posterior()
post_sample_df['group'] = post_sample_df['ag_idx'].map(lambda x: exp_behav_dict['group'][0][x])
post_sample_df['model'] = [model]*len(post_sample_df)

"----- Save results to file"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
pickle.dump( (post_sample_df, exp_behav_df, infer.loss, agent.param_names), open(f"behav_fit/behav_fit_{timestamp}.p", "wb" ) )

#%%
'''
Analysis
'''
#%%
"----- Open Files"
# import sys
# sys.modules[__name__].__dict__.clear() # Clear variables
# del model
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

if len(res) == 3:
    inf_mean_df, loss, param_names = res
    inf_mean_df['ag_idx'] = range(len(inf_mean_df))
    gruppe = [0]*9
    gruppe.extend([1]*9)
    gruppe.extend([2]*9)
    gruppe.extend([3]*9)
    inf_mean_df['group'] = gruppe
    model ='B'
    num_agents = len(inf_mean_df)
    
elif len(res) == 4:
    post_sample_df, inf_mean_df, loss, param_names = res
    model = post_sample_df['model'][0]
    post_sample_df.rename(columns={'subject': 'ag_idx'}, inplace=True)
    num_agents = len(post_sample_df['ag_idx'].unique())
    
num_params = len(param_names)
params = torch.tensor(inf_mean_df.iloc[:,0:-3].values.T, requires_grad = False)
expdata_dict, expdata_df = utils.get_groupdata('/home/sascha/Desktop/vbm_torch/behav_data/')
# groupdata_df = pd.DataFrame(expdata_dict).explode(list(expdata_dict.keys()))
# sequence_behav_fit  = [1 if group==0 or group==1 else 2 for group in expdata_dict['group']]
# blockorder_behav_fit  = [1 if group==0 or group==2 else 2 for group in expdata_dict['group']]
Q_init_behav_fit = torch.cat((torch.tensor([[0.2, 0., 0., 0.2]]).tile((num_agents//2, 1)),
                    torch.tensor([[0, 0.2, 0.2, 0.]]).tile((num_agents//2, 1))))

# # "Makes ure order of parameters is preserved"
# # assert(all([param_names[par_idx] == df.columns[par_idx] for par_idx in range(num_params)]))
# if model == 'conflictmodel':
#     model= 'Conflict'

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
Plot Experimental Data
'''
utils.plot_grouplevel(expdata_df)

#%%
'''
Simulate only from means
'''
groupdata_dict, group_behav_df, params_sim, params_sim_df = utils.simulate_data(model, 
                                                                        num_agents,
                                                                        Q_init = Q_init_behav_fit,
                                                                        group = list(inf_mean_df['group']),
                                                                        params = inf_mean_df)

utils.plot_grouplevel(expdata_df, group_behav_df, plot_single = False)
# utils.plot_grouplevel(expdata_df, plot_single = True)

#%%
'''
Plot ELBO and Parameter Estimates
'''
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()
plt.plot(loss)
plt.title("ELBO")
ax.set_xlabel("Number of iterations")
ax.set_ylabel("ELBO")
plt.show()

for param in param_names:
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

npar = len(post_sample_df.columns[0:-3])

fig, ax = plt.subplots(1, npar, figsize=(15,5), sharey=0)

if model == 'B':
    ylims = [[0, 0.03], # lr
             [0.5, 7.5], # theta_Q
             [0.5, 1.8], # theta_rep
             [0, 0.03], # lr
             [0.5, 7.5], # theta_Q
             [0.5, 1.8]] # theta_rep
    
elif model == 'Conflict'or model =='conflictmodel':
    ylims = [[0, 0.03], # lr
             [0.5, 7.5], # theta_Q
             [0.5, 5], # theta_rep
             [-0.4, 0.5], # conflict param
             [0, 0.03], # lr
             [0.5, 7.5], # theta_Q
             [0.5, 5], # theta_rep
             [-0.5, 0.5]] # conflict param
    

for par in range(npar):

    if 1:    
        "With colorbar"
        "ax[0]"
        sns.violinplot(ax = ax[par], 
                       x = 'variable',
                       y = 'value',
                       data = inf_mean_df.melt()[inf_mean_df.melt()['variable'] == param_names[par]],
                       color=".8")
        
        sns.stripplot(x = 'variable',
                      y = 'value',
                      data = inf_mean_df.melt()[inf_mean_df.melt()['variable'] == param_names[par]],
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
        # variance = df[param_names[par]].std()**2
        
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
Posterior Predictives
'''
complete_df = utils.posterior_predictives(post_sample_df, exp_data = expdata_df)