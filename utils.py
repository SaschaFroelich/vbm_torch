#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 12:52:24 2023

@author: sascha
"""

import env 
import ipdb
import torch
import numpy as np
import scipy
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import models_torch as models

from statistics import mean, stdev
from math import sqrt

# np.random.seed(123)
# torch.manual_seed(123)

def get_groupdata(data_dir):
    '''
    

    Parameters
    ----------
    data_dir : str
        Directory with data.

    Returns
    -------
    newgroupdata : dict
        Contains experimental data.
        Keys
            trialsequence : nested list, 'shape' [num_trials, num_agents]
            choices : nested list, 'shape' [num_trials, num_agents]
            outcomes : nested list, 'shape' [num_trials, num_agents]
            blocktype : nested list, 'shape' [num_trials, num_agents]
            blockidx : nested list, 'shape' [num_trials, num_agents]
            RT : nested list, 'shape' [num_trials, num_agents]
            group : list, len [num_agents]

    '''
    
    groupdata = []
    group = []
    
    pb = -1
    for grp in range(4):
        files_day1 = glob.glob(data_dir + "Grp%d/csv/*Tag1*.mat"%(grp+1))
        for file1 in files_day1:
            "Loop over participants"
            pb += 1
            group.append(grp)
            data, _ = get_participant_data(file1, 
                                            grp, 
                                            data_dir,
                                            published_results = 0)
            
            groupdata.append(data)
                
    newgroupdata = comp_groupdata(groupdata, for_ddm = 0)
    newgroupdata['group'] = group
    
    return newgroupdata

def get_participant_data(file_day1, group, data_dir, published_results = 0):
    '''
    Parameters
    ----------
    file_day1 : TYPE
        DESCRIPTION.
        
    group : int
        Experimental Group
        
    data_dir : str
        Where experimental data is stored.
        
    published_results : int, optional
        0/1 get unpublished/ published results. The default is 0.

    Returns
    -------
    data : dict
        DESCRIPTION.
        
    ID : str
        Participant-specific ID.
        Keys:
            trialsequence
            trialsequence no jokers
            choices : list, len num_trials, -2,-1,0,1,2, or 3
            outcomes
            blocktype
            blockidx
            RT

    '''
    
    "Get data of an individual participant"
    assert(group < 4)
    "RETURN: data (dict) used for inference"
    
    if published_results:
        ID = file_day1.split("/")[-1][4:9]
    else:
        ID = file_day1.split("/")[-1][4:28] # Prolific ID

    print(data_dir)
    print(glob.glob(data_dir + "Grp%d/csv/*%s*Tag2*.mat"%(group+1, ID)))
    
    file_day2 = glob.glob(data_dir + "Grp%d/csv/*%s*Tag2*.mat"%(group+1, ID))[0]

    print("==============================================================")
    print("Doing %s and %s"%(file_day1, file_day2))
    participant_day1 = scipy.io.loadmat(file_day1)
    participant_day2 = scipy.io.loadmat(file_day2) 

    correct = [] # 0 = wrong response, 1 = correct response, 2 = too slow, 3 = two keys at once during joker-trials
    choices = []
    outcomes = []
    RT = []
    
    "Block order is switched pairwise for groups 2 & 4"
    block_order_day1 = [[0,1,2,3,4,5], [1,0,3,2,5,4], [0,1,2,3,4,5], [1,0,3,2,5,4]]
    
    for i in block_order_day1[group]:
        "Mark beginning of new block with -1"
        choices.append(-1)
        outcomes.append(-2) # will transform this to -1 further down below
        correct.append(-1)
        RT.append(-1)
        
        correct.extend(np.squeeze(participant_day1["correct_all_cell"][0][i]).tolist())
        choices.extend(np.squeeze(participant_day1["resps_response_digit_cell"][0][i]).tolist()) # Still 1-indexed
        outcomes.extend(np.squeeze(participant_day1["rew_cell"][0][i]).tolist())
        RT.extend(np.squeeze(participant_day1["RT_cell"][0][i]).tolist())
        
    block_order_day2 = [[0,1,2,3,4,5,6,7], [1,0,3,2,5,4,7,6], [0,1,2,3,4,5,6,7], [1,0,3,2,5,4,7,6]]
        
    for i in block_order_day2[group]:
        "Mark beginning of new block with -1"
        choices.append(-1)
        outcomes.append(-2) # will transform this to -1 further down below
        correct.append(-1)
        RT.append(-1)
        
        correct.extend(np.squeeze(participant_day2["correct_all_cell"][0][i]).tolist())
        choices.extend(np.squeeze(participant_day2["resps_response_digit_cell"][0][i]).tolist()) # Still 1-indexed
        outcomes.extend(np.squeeze(participant_day2["rew_cell"][0][i]).tolist())
        RT.extend(np.squeeze(participant_day2["RT_cell"][0][i]).tolist())
    
    'So far, choices is list containing -1 (new block trial), 0 (error), and resp options 1,2,3,4,'
    'Transform choices to 0-indexing, and errors to -2.'
    choices = [-2 if ch == 0 else ch for ch in choices]
    choices = [-2 if ch == -2 else -1 if ch == -1 else ch-1 for ch in choices]
    
    "Transform outcomes: 2 (no reward) -> 0, -1 (error) -> -2, 1 -> 1, -2->-1"
    outcomes = [0 if out == 2 else -2 if out == -1 else 1 if out == 1 else -1 for out in outcomes]
    
    "Check internal consistency: indexes of errors should be the same in choices and in correct"
    indices_ch = [i for i, x in enumerate(choices) if x == -2]
    indices_corr = [i for i, x in enumerate(correct) if x != 1 and x != -1]
    
    assert(indices_ch == indices_corr)
    
    "Check internal consistency: indexes of new blocks (-1) should be the same in choices and in correct"
    indices_ch = [i for i, x in enumerate(choices) if x == -1]
    indices_out = [i for i, x in enumerate(outcomes) if x == -1]
    indices_corr = [i for i, x in enumerate(correct) if x == -1]
    
    assert(indices_ch == indices_corr)
    assert(indices_ch == indices_out)
    
    trialsequence = []
    trialsequence_wo_jokers = []
    blocktype = []
    blockidx = []
    for block in range(14):
        "Mark the beginning of a new block"
        trialsequence.append(torch.tensor(-1))
        trialsequence_wo_jokers.append(torch.tensor(-1))
        blocktype.append(torch.tensor(-1))
        blockidx.append(block)
        
        seq, btype, seq_wo_jokers = get_trialseq(group, 
                                                 block, 
                                                 published_results = published_results)
        
        trialsequence.extend(seq)
        trialsequence_wo_jokers.extend(seq_wo_jokers)
        blocktype.extend([btype]*len(seq))
        blockidx.extend([block]*len(seq))
    
    assert(len(trialsequence) == len(choices))
    assert(len(outcomes) == len(choices))
    assert(len(outcomes) == len(blocktype))
    
    trialsequence = [[trialsequence[i].item()] for i in range(len(trialsequence))]
    trialsequence_wo_jokers = [[trialsequence_wo_jokers[i].item()] for i in range(len(trialsequence_wo_jokers))]
    choices = [torch.tensor([choices[i]]) for i in range(len(choices))]
    outcomes = [torch.tensor([outcomes[i]]) for i in range(len(outcomes))]
    blocktype = [[blocktype[i]] for i in range(len(blocktype))]
    blockidx = [[blockidx[i]] for i in range(len(blockidx))]
    RT = [[RT[i]] for i in range(len(RT))]
        
    data = {"trialsequence": trialsequence, \
            "trialsequence no jokers": trialsequence_wo_jokers, \
            "choices": choices, \
            "outcomes": outcomes, \
            "blocktype": blocktype, \
            "blockidx": blockidx, \
            "RT": RT}
            
    return data, ID
    
def get_trialseq(group, block_no, published_results = 0):
    "NB: in mat-files, the block order was already swapped, as if all participants saw the first group's block order! Have to correct for this!"
    
    "This is the blockorder participants actually saw"
    blockorder = [["random1", "trainblock1", 
                   "random2", "trainblock2", 
                   "random3", "trainblock3", 
                   "trainblock4", "random4", 
                   "trainblock5", "random5", 
                   "trainblock6", "random6", 
                   "trainblock7", "random7"],
                  
                  ["trainblock1", "random1", 
                   "trainblock2", "random2", 
                   "trainblock3", "random3", 
                   "random4", "trainblock4", 
                   "random5", "trainblock5", 
                   "random6", "trainblock6", 
                   "random7", "trainblock7"],
                  
                  ["mirror_random1", "mirror_trainblock1", 
                   "mirror_random2", "mirror_trainblock2", 
                   "mirror_random3", "mirror_trainblock3", 
                   "mirror_trainblock4", "mirror_random4", 
                   "mirror_trainblock5", "mirror_random5", 
                   "mirror_trainblock6", "mirror_random6", 
                   "mirror_trainblock7", "mirror_random7"],
                  
                  ["mirror_trainblock1", "mirror_random1", 
                   "mirror_trainblock2", "mirror_random2", 
                   "mirror_trainblock3", "mirror_random3", 
                   "mirror_random4", "mirror_trainblock4", 
                   "mirror_random5", "mirror_trainblock5", 
                   "mirror_random6", "mirror_trainblock6", 
                   "mirror_random7", "mirror_trainblock7"]]    

    if published_results:
        "Published"
        mat = scipy.io.loadmat("/home/sascha/Desktop/vbm_torch/matlabcode/published/%s.mat"%blockorder[group][block_no])
        
    else:
        "Clipre"
        mat = scipy.io.loadmat("/home/sascha/Desktop/vbm_torch/matlabcode/clipre/%s.mat"%blockorder[group][block_no])
        
    types = [[1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],\
             [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],\
              [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],\
              [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]]    
    
    return torch.tensor(np.squeeze(mat["sequence"])), types[group][block_no], \
        torch.tensor(np.squeeze(mat["sequence_without_jokers"]))

def replace_single_element_lists(value):
    if len(value) == 1:
        return value[0]
    return value

def arrange_data_for_plot(i, df, **kwargs):        
    df = df.applymap(replace_single_element_lists)
    
    "Only retain rows pertaining to joker trial"
    df = df[df["Jokertypes"] > -1]
    
    "Get rid of tensors"
    df = df.applymap(lambda x: x.item() if torch.is_tensor(x) else x)
    
    """
    choices -> Goal-Directed choices.
    0: Low-Reward Choice, 1: High-Reward Choice 
    groups 0 & 1: (from High-Rew: 0 & 3, Low-Rew: 1 & 2)
    groups 2 & 3: (from High-Rew: 1 & 2, Low-Rew: 0 & 3)
    """
    
    if kwargs["group"] == 0 or kwargs["group"] == 1:
        df['choices'] = df['choices'].map(lambda x: 1 if x==0 else 0 if x==1 else 0 if x==2 else 1)
        
    elif kwargs["group"] == 2 or kwargs["group"] == 3:
        df['choices'] = df['choices'].map(lambda x: 0 if x==0 else 1 if x==1 else 1 if x==2 else 0)
    
    else:
        raise Exception("Group is not correctly specified, pal!")
    
    data_new = {"HPCF": [], "Trialtype": [], "blockidx": [], "datatype" : []}
    data_Q = {"Qdiff": [], "blocktype":[], "blockidx": []}
    
    for block in df["blockidx"].unique():
        # For column "Jokertypes":
        # -1 no joker
        # 0 random 
        # 1 congruent 
        # 2 incongruent
        
        if "Qdiff" in df.columns:
            data_Q["Qdiff"].append(df[df["blockidx"]==block]["Qdiff"].mean())
        
        if df[df["blockidx"]==block]["blocktype"].unique()[0] == 1:
            "Random Block"
            
            if "Qdiff" in df.columns:
                data_Q["blocktype"].append(1)
                data_Q["blockidx"].append(block)
                
            data_new["blockidx"].append(block)
            data_new["Trialtype"].append("random")
            data_new["HPCF"].append(df[(df["blockidx"] == block) & (df["Jokertypes"] == 0)]["choices"].mean())
            
            if i == 0:
                data_new["datatype"].append("given (Group %s)"%kwargs["group"])
                
            elif i == 1:
                data_new["datatype"].append("simulated (Group %s)"%kwargs["group"])
                
            else:
                raise Exception("Fehla!")
            
        elif df[df["blockidx"]==block]["blocktype"].unique()[0] == 0:
            "Sequential Block"
            
            if "Qdiff" in df.columns:
                data_Q["blocktype"].append(0)
                data_Q["blockidx"].append(block)
            
            "Congruent Jokers"
            data_new["blockidx"].append(block)
            data_new["Trialtype"].append("congruent")
            data_new["HPCF"].append(df[(df["blockidx"] == block) & (df["Jokertypes"] == 1)]["choices"].mean())
            
            if i == 0:
                data_new["datatype"].append("given (Group %s)"%kwargs["group"])
                
            elif i == 1:
                data_new["datatype"].append("simulated (Group %s)"%kwargs["group"])
            
            else:
                raise Exception("Fehla!")
            
            "Incongruent Jokers"
            data_new["blockidx"].append(block)
            data_new["Trialtype"].append("incongruent")
            data_new["HPCF"].append(df[(df["blockidx"] == block) & (df["Jokertypes"] == 2)]["choices"].mean())
            
            if i == 0:
                data_new["datatype"].append("given (Group %s)"%kwargs["group"])
                
            elif i == 1:
                data_new["datatype"].append("simulated (Group %s)"%kwargs["group"])
                
            else:
                raise Exception("Fehla!")
            
        else:
            raise Exception("Fehla!")
    
    
    return data_new, data_Q

def plot_results(data_sim, *args, **kwargs):
    '''
    

    Parameters
    ----------
    data_sim : DataFrame
        DataFrame with data of single participant to be plotted.
        
    *args : DataFrame
        DataFrame with second set of data to be plotted (for comparison with first DataFrame).
        
    **kwargs :  group (int)
                    Experimental group.
                title (str)
                    Title of the plot
                omega_true (float)
                omega_inf (float)
                ymin (float)
                savedir (str)
                plotname (str)

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    '''
    

    '''
    Jokertypes 
    ----------
    -1 no joker
    0 random
    1 congruent
    2 incongruent
    '''
    if args:
        datas = (data_sim, args[0])
    
    elif not args: 
        datas = (data_sim,)
        
    else:
        raise Exception("Fehla!")
        
    "Create df_new, which will contain modified results for plotting and will contain column 'datatype' for simulated data and experimental data"
    df_new = pd.DataFrame()
    
    for i in range(len(datas)):
        
        #print(i)
        df = pd.DataFrame(data=datas[i])
        
        data_new, data_Q = arrange_data_for_plot(i, df, **kwargs)
        
        df_temp = pd.DataFrame(data=data_new)
        df_new = pd.concat([df_new, df_temp])
        df_Q = pd.DataFrame(data=data_Q)

    # fig, ax = plt.subplots()
    if kwargs["group"] == 0 or kwargs["group"] == 2:
        custom_palette = ['r', 'g', 'b'] # random, congruent, incongruent
        
    elif kwargs["group"] == 1 or kwargs["group"] == 3:
        custom_palette = ['g', 'b', 'r'] # congruent, incongruent, random

    sns.relplot(x="blockidx", y="HPCF", hue = "Trialtype", data=df_new, kind="line", col="datatype", palette = custom_palette)
    # plt.plot([5.5, 5.5],[0.5,1], color='black') # plot Day1/Day2 line
    
    "----- UNCOMMENT THESE TWO LINES TO PLOT QDIFF ------------"
    # if "Qdiff" in df.columns:
    #     sns.relplot(x='blockidx', y="Qdiff", hue ="blocktype", data=df_Q, kind = 'line')
    "----- --------------------------------------- ------------"

    if "ymin" in kwargs:
        ylim = kwargs["ymin"]
    else:
        ylim = 0

    plt.ylim([ylim, 1])
    
    if "omega_true" in kwargs:
        plt.text(0.05, ylim+0.14, "omega = %.2f, inf:%.2f"%(kwargs["omega_true"], kwargs["omega_inf"]))
        plt.text(0.05, ylim+0.08, "dectemp = %.2f, inf:%.2f"%(kwargs["dectemp_true"], kwargs["dectemp_inf"]))
        plt.text(0.05, ylim+0.02, "lr = %.2f, inf:%.2f"%(kwargs["lr_true"], kwargs["lr_inf"]))
        
    else:
        if "omega_inf" in kwargs:
            plt.text(0.05, ylim+0.14, "omega inf:%.2f"%(kwargs["omega_inf"]))
            plt.text(0.05, ylim+0.08, "dectemp inf:%.2f"%(kwargs["dectemp_inf"]))
            plt.text(0.05, ylim+0.02, "lr inf:%.2f"%(kwargs["lr_inf"]))
    
    if "savedir" in kwargs:
        
        if "plotname" in kwargs:
            plt.savefig(kwargs["savedir"]+"/%s.png"%kwargs["plotname"])
        else:
            plt.savefig(kwargs["savedir"]+"/plot_%d.png"%(np.random.randint(10e+9)))
        
    if 'title' in kwargs:
        plt.title("Simulated data.")
        
    plt.show()    
    
    return df_new[df_new["datatype"] == "simulated (Group %d)"%kwargs["group"]], \
        df_new[df_new["datatype"] == "given (Group %d)"%kwargs["group"]]

def cohens_d(c0,c1):
    return (mean(c0) - mean(c1)) / (sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2))

def simulate_model_behaviour(num_agents, model, **kwargs):

    assert(model == 'B')
    
    if 'k' in kwargs:
        k = kwargs['k']
        
    else:
        k = 4.

    df_all = pd.DataFrame()

    for agent in range(num_agents):
        print("Simulating agent no. %d"%agent)
        newagent = models.Vbm_B(theta_rep_day1 = kwargs['theta_rep_day1'], \
                      theta_rep_day2 = kwargs['theta_rep_day2'], \
                      lr_day1 = kwargs['lr_day1'], \
                      lr_day2 = kwargs['lr_day2'], \
                      theta_Q_day1 = kwargs['theta_Q_day1'], \
                      theta_Q_day2 = kwargs['theta_Q_day2'], \
                      k=k,\
                      Q_init=[0.2, 0., 0., 0.2])

        newenv = env.Env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir = './matlabcode/clipre/')

        newenv.run()
        data = {"choices": newenv.choices, "outcomes": newenv.outcomes,\
                "trialsequence": newenv.data["trialsequence"], "blocktype": newenv.data["blocktype"],\
                "Jokertypes": newenv.data["jokertypes"], "blockidx": newenv.data["blockidx"], \
                "Qdiff": [(newenv.agent.Q[i][...,0] + newenv.agent.Q[i][...,3])/2 - (newenv.agent.Q[i][...,1] + newenv.agent.Q[i][...,2])/2 for i in range(len(newenv.choices))]}

        df = pd.DataFrame(data)

        data_new, data_Q = arrange_data_for_plot(1, df, group = 0)
        
        df_new = pd.DataFrame(data_new)
        
        df_all = pd.concat((df_all, df_new))
        
    custom_palette = ['r', 'g', 'b'] # random, congruent, incongruent
    sns.relplot(x="blockidx", y="HPCF", hue = "Trialtype", data=df_all, kind="line", palette=custom_palette)
    plt.plot([5.5, 5.5],[0.5,1], color='black')
    plt.show()
    
def comp_groupdata(groupdata, for_ddm = 1):
    '''
    Parameters
    ----------
    groupdata : list of len num_agents
        Contains 1 dictionary per agent, with experimental data.
        Keys: 
            choices
            outcomes
            trialsequence
            blocktype
            Jokertypes
            blockidx
            
    for_ddm : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    newgroupdata : dict
        Contains keys as in groupdata.
        len = num_trials
        Each element of a key is another list with length num_agents

    '''
    
    if for_ddm:
        newgroupdata = {"trialsequence" : [],\
                        #"trialsequence no jokers" : [],\
                        "choices" : [],\
                        "outcomes" : [],\
                        "blocktype" : [],\
                        "blockidx" : [],\
                        "RT": []}
            
    else:
        newgroupdata = {"trialsequence" : [],\
                        #"trialsequence no jokers" : [],\
                        "choices" : [],\
                        "outcomes" : [],\
                        "blocktype" : [],\
                        "blockidx" : []}
    
    for trial in range(len(groupdata[0]["trialsequence"])):
        trialsequence = []
        # trialseq_no_jokers = []
        choices = []
        outcomes = []
        blocktype = []
        blockidx = []
        if for_ddm:
            RT = []

        for dt in groupdata:
            trialsequence.append(dt["trialsequence"][trial][0])
            # trialseq_no_jokers.append(dt["trialsequence no jokers"][trial][0])
            choices.append(dt["choices"][trial][0].item())
            outcomes.append(dt["outcomes"][trial][0].item())
            blocktype.append(dt["blocktype"][trial][0])
            blockidx.append(dt["blockidx"][trial][0])
            if for_ddm:
                RT.append(dt["RT"][trial][0])
            
        newgroupdata["trialsequence"].append(trialsequence)
        # newgroupdata["trialsequence no jokers"].append(trialseq_no_jokers)
        newgroupdata["choices"].append(torch.tensor(choices, dtype = int))
        newgroupdata["outcomes"].append(torch.tensor(outcomes, dtype = int))
        newgroupdata["blocktype"].append(blocktype)
        newgroupdata["blockidx"].append(blockidx)
        if for_ddm:
            newgroupdata["RT"].append(RT)
        
    return newgroupdata

def init_agent(model, Q_init, num_agents=1, params = None):
    '''
    
    Parameters
    ----------
    model : str
        The model class to simulate data with.

    Q_init : tensor, shape [num_agents, 4]
        Initial Q-Values of the agent(s).

    num_agents : int
        Number of agents in a single agent object.
        The default is 1.
        
    params : torch tensor, shape [num_params, num_agents]
        Parameters to initialize agents with
        If None, parameters will be set randomly.


    Returns
    -------
    newagent : obj of class model

    '''
    
    assert(Q_init.shape == (num_agents, 4))
    
    import models_torch as models
    
    k = 4.
    if model =='original':
        num_params = models.Vbm.num_params #number of latent model parameters
        
        if params is None:
            print("Setting random parameters.")
            params = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
        else:
            print("Setting initial parameters as provided.")
        
        omega = params[0:1, :]
        dectemp = (params[1:2, :]+1)*3
        lr = params[2:3, :]*0.01
        newagent = models.Vbm(omega = omega,
                              dectemp = dectemp,
                              lr = lr,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
        
    elif model == 'B':
        num_params = models.Vbm_B.num_params #number of latent model parameters
        
        if params is None:
            print("Setting random parameters.")
            params = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
        else:
            print("Setting initial parameters as provided.")
            
        lr_day1 = params[0:1, :]*0.01
        theta_Q_day1 = params[1:2, :]*6
        theta_rep_day1 = params[2:3, :]*6
        
        lr_day2 = params[3:4, :]*0.01
        theta_Q_day2 = params[4:5, :]*6
        theta_rep_day2 = params[5:6, :]*6
        
        newagent = models.Vbm_B(lr_day1 = lr_day1,
                              theta_Q_day1 = theta_Q_day1,
                              theta_rep_day1 = theta_rep_day1,
                                  
                              lr_day2 = lr_day2,
                              theta_Q_day2 = theta_Q_day2,
                              theta_rep_day2 = theta_rep_day2,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    else:
        raise Exception("No model specified")
        
    return newagent
        
def simulate_data(model, 
                  num_agents, 
                  Q_init, 
                  sequence = None, 
                  blockorder = None,
                  params = None):
    '''
    Simulates data and plots results.
    
    Parameters
    ----------
    model : str
        The model class to simulate data with.

    num_agents : int
        Number of agents for simulation.
        
    Q_init : tensor, shape [num_agents, 4]
        Initial Q-Values of the agent.
        
    sequence : list, len num_agents
        Whether sequence or mirror sequence.
        1/2 : sequence/ mirror sequence
        Mirror sequence has reversed reward probabilities.
        
    blockorder : list, len num_agents
        Which blockorder
        1/2 : RSRSRS SRSRSRSR / SRSRSR RSRSRSRS
        
    params : tensor, shape [num_params, num_agents]
        Contains the latent model parameters with which to simulate data.
        
    Returns
    -------
    groupdata : list of len num_agents
        Contains 1 dictionary per agent, with experimental data.
        Keys: 
            choices
            outcomes : 0/1 reward/ no reward
            trialsequence : stimuli as seen by participants
            blocktype
            jokertypes
            blockidx : number of block as experienced by participants
            
    params_true : torch tensor, shape [num_params, num_agents]
        Contains the parameter values with which the simulations were performed.

    params_true_df : DataFrame
        Contains the parameter values with which the simulations were performed.

    '''
    
    assert(torch.is_tensor(Q_init))
    
    if sequence is None:
        sequence = [1]*num_agents
        
    else: 
        raise Exception("Not implemented.")
        
    if blockorder is None:
        blockorder = [1]*num_agents
        
    else: 
        raise Exception("Not implemented.")
    
    if model == 'original':
        num_params = models.Vbm.num_params #number of latent model parameters
        
    elif model == 'B':
        num_params = models.Vbm_B.num_params #number of latent model parameters
    
    groupdata = []
    params_true = torch.zeros((num_params, num_agents))
        
    "Simulate with random parameters"
    for ag_idx in range(num_agents):
        
        print("Simulating agent no. %d"%ag_idx)
        
        if params == None:
            newagent = init_agent('original', Q_init[ag_idx, :], num_agents = 1)
        
        else:
            newagent = init_agent('original', Q_init[ag_idx, :], num_agents = 1, params = params[:, ag_idx:ag_idx+1])
        
        for param_idx in range(len(newagent.param_names)):
            params_true[param_idx, ag_idx] = newagent.par_dict[newagent.param_names[param_idx]]

        newenv = env.Env(newagent, matfile_dir = './matlabcode/clipre/')
        
        newenv.run(sequence = sequence[ag_idx],
                   blockorder = blockorder[ag_idx])
        
        data = {"choices": newenv.choices, 
                "outcomes": newenv.outcomes,
                "trialsequence": newenv.data["trialsequence"], 
                "blocktype": newenv.data["blocktype"],
                "Jokertypes": newenv.data["jokertypes"], 
                "blockidx": newenv.data["blockidx"]}
            
        plot_results(pd.DataFrame(data), group = 0, title = 'Simulated Data')
            
        groupdata.append(data)
        
    "---- Create DataFrame"
    columns = []
    for key in newagent.par_dict.keys():
        columns.append(key + '_true')
    params_true_df = pd.DataFrame(params_true.numpy().T, columns = columns)
        
    return groupdata, params_true, params_true_df