#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 10:17:20 2023

@author: sascha
"""

# SuperFastPython.com
# example of parallel map() with the process pool
import numpy as np
import torch

import utils
import models_torch as models
import env 
import utils

import pandas as pd

matfile_dir = './matlabcode/clipre/'

#%%
"Simulate the group-level behaviour of several agents"

num_agents = 40
model = 'B'
parameters = {'theta_rep_day1': 0.4,
'theta_rep_day2': 0.8,
'theta_Q_day1': 3.,
'theta_Q_day2': 3.,
'lr_day1': 0.001,
'lr_day2': 0.,
'k': 4.}

utils.simulate_model_behaviour(num_agents, model, **parameters)

#%%
import ipdb

def Qoutcomp(Qin, choices):
    """Qin shape is [num_particles, num_agents, 4]"""
    """Returns a tensor with the same shape as Qin, with zeros everywhere except for the relevant places
    as indicated by 'choices', where the values of Qin are retained. Q positions for agents with an error choice
    are replaced by 0."""
    
    Qin = Qin.type(torch.double)
    
    print("Test this with errors!!!")
    if len(Qin.shape) == 2:
        Qin = Qin[None, ...]
        
    elif len(Qin.shape) == 3:
        pass
    
    else:
        ipdb.set_trace()
        raise Exception("Fehla, digga!")
    
    try:
        no_error_mask = [1 if ch != -10 else 0 for ch in choices]
    except:
        ipdb.set_trace()
        
    "Replace error choices by the number one"
    choices_noerrors = torch.where(torch.tensor(no_error_mask).type(torch.bool), choices, torch.ones(choices.shape)).type(torch.int)

    Qout = torch.zeros(Qin.shape).double()
    choicemask = torch.zeros(Qin.shape, dtype = int)
    num_particles = Qout.shape[0] # num of particles
    num_agents = Qout.shape[1] # num_agents
    
    errormask = torch.tensor([0 if c == -10 else 1 for c in choices])
    errormask = errormask.broadcast_to(num_particles, 4, num_agents).transpose(1,2)
    
    ipdb.set_trace()
    x = torch.linspace(0, num_particles-1,num_particles, dtype = int).repeat(num_agents)
    y = torch.linspace(0, num_agents-1,num_agents, dtype = int).repeat_interleave(num_particles)
    z = choices_noerrors.repeat_interleave(num_particles)
    Qout[x, y, z] = Qin[x, y, z]

    choicemask[torch.linspace(0, num_particles-1,num_particles, dtype = int).repeat_interleave(num_agents), \
          torch.linspace(0, num_agents-1,num_agents, dtype = int).repeat(num_particles), \
          choices_noerrors.repeat(num_particles)] = 1
    
    mask = errormask*choicemask
    ipdb.set_trace()
    return Qout.double()*mask, mask

#%%
import torch

ch = torch.tensor([0,1,3])

Q=torch.tensor([[[0.6820, 0.0221, 0.1871, 0.0121],
         [0.6425, 0.6157, 0.5441, 0.1801],
         [0.8009, 0.9316, 0.0234, 0.7405]],

        [[0.1365, 0.2842, 0.0991, 0.3371],
         [0.6814, 0.8119, 0.1702, 0.9241],
         [0.9045, 0.7136, 0.2212, 0.6656]]])

res = Q[..., ch]*torch.eye(3)

res = res.diagonal(dim1=1, dim2=2).reshape(2,3,1)

#%%
# 5 particles, 3 agents, 4 actions
lr = torch.rand(5,6)
Q = torch.rand(5,6,4)
Q[0,0,2] = 0.
choices = torch.tensor([0,1,3,0,0,2])
outcomes = torch.tensor([1,1,1,0,0,0])

#%%
mask = torch.ceil(abs(Qoutcomp(Q, choices))).double()
Qnewmask = Q + lr[..., None]*((outcomes[None,...,None]-Qoutcomp(Q, choices))*mask)
Qnew = Q + lr[..., None]*((outcomes[None,...,None]-Qoutcomp(Q, choices)))

#%%
import numpy as np
import matplotlib.pyplot as plt

# Define the double-armed bandit environment
class DoubleArmedBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.true_values = np.random.normal(0, 1, num_arms)
        print("The true values are")
        print(self.true_values)
        
    def step(self, action):
        return np.random.normal(self.true_values[action], 1)

# Q-learning algorithm
def q_learning(num_arms, num_episodes, alpha, epsilon, gamma):
    Q = np.zeros(num_arms)  # Initialize Q-values to zero
    Q_hist = []
    bandit = DoubleArmedBandit(num_arms)
    rewards_per_episode = []
    
    for episode in range(num_episodes):
        total_reward = 0
        state = 0  # There is only one state in this case (start state)
        
        for t in range(num_arms):
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(num_arms)
            else:
                action = np.argmax(Q)
            
            # Take a step in the environment and observe the reward
            reward = bandit.step(action)
            total_reward += reward
            
            # Update Q-value using temporal-difference learning (Q-learning)
            next_action = np.argmax(Q)
            Q[action] += alpha * (reward + gamma * Q[next_action] - Q[action])
        
        Q_hist.append(Q)
        rewards_per_episode.append(total_reward)
    
    return Q, rewards_per_episode, Q_hist

# Parameters
num_arms = 2
num_episodes = 1000
alpha = 0.1  # Learning rate
epsilon = 0.1  # Exploration rate
gamma = 1.0  # Discount factor

# Run Q-learning
Q_values, rewards, Q_hist = q_learning(num_arms, num_episodes, alpha, epsilon, gamma)

# Plot the evolution of Q-values
plt.figure(figsize=(10, 5))
plt.plot(Q_values, label='Q-values')
plt.xlabel('Episodes')
plt.ylabel('Q-values')
plt.legend()
plt.title('Evolution of Q-values with Temporal-Difference Learning (Q-learning)')
plt.show()

# Plot the rewards per episode
plt.figure(figsize=(10, 5))
plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.show()

#%% 

Qout, mask = Qoutcomp(Qin, choices)

#%%

def mult_func(x):
    mylist = [None]
    
    for i in range(1_000_000_000):
        mylist[0] = i

    return mylist[-1]

import multiprocessing as mp
with mp.Pool(mp.cpu_count()) as pool:
    results = pool.map(mult_func, range(10))

print(results)

#%%
'''
Simulate data
'''
import numpy as np
import torch

import utils
import models_torch as models
import env 
import utils

import pandas as pd# model = sys.argv[1]
# resim =  int(sys.argv[2]) # whether to simulate agents with inferred parameters
# method = sys.argv[3] # "svi" or "mcmc"
# num_agents = 50

model = 'B_lrdec'
resim =  0 # whether to simulate agents with inferred parameters
method = 'svi' # "svi" or "mcmc"
num_agents = 60

assert num_agents%4 == 0, "num_agents must be divisible by 4."
# k = 4.
print(f"Running model {model}")

sequence = ([1]*(num_agents//2))
sequence.extend(([2]*(num_agents//2)))

blockorder = np.ones((1, num_agents//4), dtype=int)
blockorder = np.concatenate((blockorder,np.ones((1, num_agents//4), dtype=int)*2),axis=1)
blockorder = np.concatenate((blockorder,blockorder),axis=1).tolist()[0]

Qs = torch.tensor([[0.2, 0., 0., 0.2],
                   [0, 0.2, 0.2, 0.]]).tile((num_agents, 1, 1))
Q_init = Qs[range(num_agents),torch.tensor(sequence)-1, :]

params = {'lr0': torch.tensor([0.001]*num_agents),
          'lrk': torch.tensor([0.0]*num_agents),
          'theta_Q_day1': torch.tensor([1.]*num_agents),
          'theta_rep_day1': torch.tensor([1.]*num_agents),
          
          'theta_Q_day2': torch.tensor([2.]*num_agents),
          'theta_rep_day2': torch.tensor([1.]*num_agents)}

group = [0]*(num_agents//4)
group.extend([1]*(num_agents//4))
group.extend([2]*(num_agents//4))
group.extend([3]*(num_agents//4))

groupdata_dict, group_behav_df, _, params_sim_df = utils.simulate_data(model, 
                                                    num_agents,
                                                   group = group,
                                                    params = params)

groupdata_df = pd.DataFrame(groupdata_dict).explode(list(groupdata_dict.keys()))

utils.plot_grouplevel(groupdata_df)
