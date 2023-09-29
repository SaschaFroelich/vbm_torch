#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 10:17:20 2023

@author: sascha
"""

# SuperFastPython.com
# example of parallel map() with the process pool
import numpy as np

import utils
import models_torch as models
import env 
import utils

import pandas as pd
 
matfile_dir = './matlabcode/clipre/'

#%%
"Bayesian Prior Model"

dectemp_day1 = 0
lr_day1 = 0.1
omega_day1 = 0.1

dectemp_day2 = 0
lr_day2 = 0.1
omega_day2 = 0.1

newagent = models.vbm_A_Bayesian(omega_day1 = omega_day1, \
                      omega_day2 = omega_day2, \
                      lr_day1 = lr_day1, \
                      lr_day2 = lr_day2, \
                      dectemp_day1 = dectemp_day1, \
                      dectemp_day2 = dectemp_day2, \
                      k=4,\
                      Q_init=[0.4, 0., 0., 0.4])
    
newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir=matfile_dir)

newenv.run()
data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
        "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
            "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"], \
        "Qdiff": [(newenv.agent.Q[i][...,0] + newenv.agent.Q[i][...,3])/2 - (newenv.agent.Q[i][...,1] + newenv.agent.Q[i][...,2])/2 for i in range(len(newenv.choices))]}

df = pd.DataFrame(data)
utils.plot_results(df, group = 0)

dectemp_day1 = 0
lr_day1 = 0.1
omega_day1 = 3

dectemp_day2 = 0
lr_day2 = 0.1
omega_day2 = 3

newagent = models.vbm_A_Bayesian(omega_day1 = omega_day1, \
                      omega_day2 = omega_day2, \
                      lr_day1 = lr_day1, \
                      lr_day2 = lr_day2, \
                      dectemp_day1 = dectemp_day1, \
                      dectemp_day2 = dectemp_day2, \
                      k=4,\
                      Q_init=[0.4, 0., 0., 0.4])
newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir=matfile_dir)

newenv.run()
data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
        "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
            "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"], \
        "Qdiff": [(newenv.agent.Q[i][...,0] + newenv.agent.Q[i][...,3])/2 - (newenv.agent.Q[i][...,1] + newenv.agent.Q[i][...,2])/2 for i in range(len(newenv.choices))]}

df = pd.DataFrame(data)
utils.plot_results(df, group = 0)


#%%
"Model C"

lr0_day1 = 0.1
lr_lambda_day1 = 0.01
theta_Q_day1 = 2
theta_rep_day1 = 1.5

lr0_day2 = 0.1
lr_lambda_day2 = 0.01
theta_Q_day2 = 2
theta_rep_day2 = 1.5

newagent = models.vbm_C(theta_rep_day1 = theta_rep_day1, \
                      theta_rep_day2 = theta_rep_day2, \
                      lr0_day1 = lr0_day1, \
                      lr0_day2 = lr0_day2, \
                      lr_lambda_day1 = lr_lambda_day1, \
                      lr_lambda_day2 = lr_lambda_day2, \
                      theta_Q_day1 = theta_Q_day1, \
                      theta_Q_day2 = theta_Q_day2, \
                      k=4,\
                      Q_init=[0.4, 0., 0., 0.4])
    
newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir=matfile_dir)

newenv.run()
data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
        "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
            "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}

df = pd.DataFrame(data)
utils.plot_results(df, group = 0)

lr0_day1 = 0.1
lr_lambda_day1 = 0.01
theta_Q_day1 = 2
theta_rep_day1 = 1.5

lr0_day2 = 0.8
lr_lambda_day2 = 0.0
theta_Q_day2 = 2
theta_rep_day2 = 1.5

newagent = models.vbm_C(theta_rep_day1 = theta_rep_day1, \
                      theta_rep_day2 = theta_rep_day2, \
                      lr0_day1 = lr0_day1, \
                      lr0_day2 = lr0_day2, \
                      lr_lambda_day1 = lr_lambda_day1, \
                      lr_lambda_day2 = lr_lambda_day2, \
                      theta_Q_day1 = theta_Q_day1, \
                      theta_Q_day2 = theta_Q_day2, \
                      k=4,\
                      Q_init=[0.4, 0., 0., 0.4])
    
newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir=matfile_dir)

newenv.run()
data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
        "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
            "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}

df = pd.DataFrame(data)
utils.plot_results(df, group = 0)

#%% 
"Model B"

npar = 6
parameter = np.random.uniform(0,1, npar)
theta_rep_day1 = 0.8
theta_rep_day2 = 0.8
theta_Q_day1 = 3.
theta_Q_day2 = 3.
lr_day1 = 0.005
lr_day2 = 0.

newagent = models.vbm_B(theta_rep_day1 = theta_rep_day1, \
              theta_rep_day2 = theta_rep_day2, \
              lr_day1 = lr_day1, \
              lr_day2 = lr_day2, \
              theta_Q_day1 = theta_Q_day1, \
              theta_Q_day2 = theta_Q_day2, \
              k=4,\
              Q_init=[0.2, 0., 0., 0.2])
    
newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir = './matlabcode/clipre/')

newenv.run()
data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
        "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
        "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"], \
        "Qdiff": [(newenv.agent.Q[i][...,0] + newenv.agent.Q[i][...,3])/2 - (newenv.agent.Q[i][...,1] + newenv.agent.Q[i][...,2])/2 for i in range(len(newenv.choices))]}
    
df = pd.DataFrame(data)
utils.plot_results(df, group = 0)

#%%
"Model F"

"No lr, theta q and thetar_r are linearly time-dependent"
npar = 8
# parameter = numpy.random.uniform(0,1, npar)

theta_rep0_day1 = 1
theta_replambda_day1 = 0.001
theta_Q0_day1 = 0
theta_Qlambda_day1 = 0.007

theta_rep0_day2 = 432*theta_replambda_day1 + theta_rep0_day1 + 1
theta_replambda_day2 = 0.001
theta_Q0_day2 = 432*theta_Qlambda_day1 + theta_Q0_day1
theta_Qlambda_day2 = 0.0


newagent = models.vbm_F(theta_rep0_day1 = theta_rep0_day1, \
                      theta_replambda_day1 = theta_replambda_day1, \
                      theta_Q0_day1 = theta_Q0_day1, \
                      theta_Qlambda_day1 = theta_Qlambda_day1, \
                      theta_rep0_day2 = theta_rep0_day2, \
                      theta_replambda_day2 = theta_replambda_day2, \
                      theta_Q0_day2 = theta_Q0_day2, \
                      theta_Qlambda_day2 = theta_Qlambda_day2, \
                      k=4,\
                      Q_init=[0.8, 0.2, 0.2, 0.8])

newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir = './matlabcode/clipre/')

newenv.run()
data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
        "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
            "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}

df = pd.DataFrame(data)
utils.plot_results(df, group = 0)

#%%
def reassign(s_idx):
    
    if s_idx in [1,3,5]:
        return s_idx + 1
    
    elif s_idx in [8, 10, 12, 14]:
        return s_idx - 1
    
    else: 
        return s_idx

import data_for_DDM as ddm
import matplotlib.pyplot as plt

print("Does this weight every participant the same, or according to their number of trials?")

df_ddm = ddm.get_ddm_data(4)
df_ddm = df_ddm[df_ddm["Blocktype"]=='s']
df_ddm = df_ddm[df_ddm["Trialsequence"]>10]

df_ddm["Blockidx"] = df_ddm["Blockidx"].map(lambda x: reassign(x))

# df_ddm["rep1"] = df_ddm["repvals"].map(lambda x: x[0])
# df_ddm["rep2"] = df_ddm["repvals"].map(lambda x: x[1])
# df_ddm["rep3"] = df_ddm["repvals"].map(lambda x: x[2])
# df_ddm["rep4"] = df_ddm["repvals"].map(lambda x: x[3])

df_ddm["maxrepval"] = df_ddm["repvals"].map(lambda x: np.max(x))

plt.scatter(range(7), df_ddm.groupby('Blockidx')['maxrepval'].mean())    
plt.title("repetition values")
plt.show()

#%%
"Simulate the group-level behaviour of several agents"

num_agents = 40
model = 'B'
parameters = {'theta_rep_day1': 0.4, \
'theta_rep_day2': 0.8, \
'theta_Q_day1': 3., \
'theta_Q_day2': 3., \
'lr_day1': 0.001, \
'lr_day2': 0., \
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


_, mask = Qoutcomp(Q, choices)
    

#%%

exec("nval = 7")
exec("blavec = [1,2,3,4]")
exec("blavec[-1] = nval")
exec("print(blavec)")

#%%

import csv
import random

# List of fantasy countries
fantasy_countries = ["Eldoria", "Mythland", "Avaloria", "Dragonia", "Feyland"]

# Generate data for fantasy cities
cities = []
for i in range(1, 150):
    city_name = f"Fantasy City {i}"
    inhabitants = random.randint(1000, 100000)
    surface_area = random.randint(10, 1000)
    average_temperature = random.uniform(-20, 40)
    children_per_household = random.uniform(0.5, 3.5)
    country = random.choice(fantasy_countries)
    
    city_data = [city_name, inhabitants, surface_area, average_temperature, children_per_household, country]
    cities.append(city_data)

# Save data to a CSV file
header = ['City Name', 'Inhabitants', 'Surface Area (sq km)', 'Average Temperature (Celsius)', 'Children per Household', 'Country']

with open('fantasy_cities_data_with_country.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(header)
    csv_writer.writerows(cities)

print("CSV file 'fantasy_cities_data_with_country.csv' created successfully!")


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