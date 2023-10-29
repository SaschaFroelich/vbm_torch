#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 12:52:24 2023

@author: sascha
"""

import time
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
import multiprocessing as mp

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
            group : list, len [num_agents]. 0-indexed

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
    Get data of an individual participant.
    
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
    
    assert group < 4
    
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
    
    assert indices_ch == indices_corr
    
    "Check internal consistency: indexes of new blocks (-1) should be the same in choices and in correct"
    indices_ch = [i for i, x in enumerate(choices) if x == -1]
    indices_out = [i for i, x in enumerate(outcomes) if x == -1]
    indices_corr = [i for i, x in enumerate(correct) if x == -1]
    
    assert indices_ch == indices_corr
    assert indices_ch == indices_out
    
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
    
    assert len(trialsequence) == len(choices)
    assert len(outcomes) == len(choices)
    assert len(outcomes) == len(blocktype)
    
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
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
    return value

def arrange_data_for_plot(dat_type, df):     
    '''
    Prepares dataframe of behaviour a single agent for plotting.
    Among other things, it computes the Optimal Response Rate (HPCF) of an agent.

    Parameters
    ----------
    dat_type : str
        'sim'/'exp': simulated/ experimental data
        
    df : DataFrame with data.
        Columns:
            choices
            choices_GD
            outcomes
            trialsequence
            blocktype
            jokertypes : -1/0/1/2 no joker/random/congruent/incongruent
            blockidx
            ag_idx (opt)
            group

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    data_new : dict
        Data arranged for plotting.
        
    data_Q : TYPE
        DESCRIPTION.

    '''
    group = df['group'][1]
    assert dat_type == 'sim' or dat_type =='exp', "Specified dat_type not recognized."
    assert group > -1 and group < 5
    assert not torch.is_tensor(group), "Group must not be tensor."

    df = df.applymap(replace_single_element_lists)
    
    "Only retain rows pertaining to joker trial"
    df = df[df['jokertypes'] > -1]
    
    "Get rid of tensors"
    df = df.applymap(lambda x: x.item() if torch.is_tensor(x) else x)
    
    # '''
    # choices -> Goal-Directed choices.
    # 0: Low-Reward Choice, 1: High-Reward Choice 
    # groups 0 & 1: (from High-Rew: 0 & 3, Low-Rew: 1 & 2)
    # groups 2 & 3: (from High-Rew: 1 & 2, Low-Rew: 0 & 3)
    # '''
    # if group == 0 or group == 1:
    #     df['choices'] = df['choices'].map(lambda x: 1 if x==0 else 0 if x==1 else 0 if x==2 else 1)
        
    # elif group == 2 or group == 3:
    #     df['choices'] = df['choices'].map(lambda x: 0 if x==0 else 1 if x==1 else 1 if x==2 else 0)
    
    data_new = {'HPCF': [], 
                'trialtype': [], 
                'blockidx': [], 
                'datatype' : [],
                'group' : []}
    
    data_Q = {'Qdiff': [], 
              'blocktype':[], 
              'blockidx': []}
    
    for block in df["blockidx"].unique():
        
        if "Qdiff" in df.columns:
            data_Q["Qdiff"].append(df[df["blockidx"]==block]["Qdiff"].mean())
        
        if df[df["blockidx"] == block]["blocktype"].unique()[0] == 1:
            "Random Block"
            
            if "Qdiff" in df.columns:
                data_Q["blocktype"].append(1)
                data_Q["blockidx"].append(block)
                
            data_new["blockidx"].append(block)
            data_new["trialtype"].append("random")
            data_new["HPCF"].append(df[(df["blockidx"] == block) & (df["jokertypes"] == 0)]["choices_GD"].mean())
            
            if dat_type == 'exp':
                data_new["datatype"].append("experimental (Group %s)"%group)
                
            elif dat_type == 'sim':
                data_new["datatype"].append("simulated (Group %s)"%group)
                
        elif df[df["blockidx"] == block]["blocktype"].unique()[0] == 0:
            "Sequential Block"
            
            if "Qdiff" in df.columns:
                data_Q["blocktype"].append(0)
                data_Q["blockidx"].append(block)
            
            "Congruent Jokers"
            data_new["blockidx"].append(block)
            data_new["trialtype"].append("congruent")
            data_new["HPCF"].append(df[(df["blockidx"] == block) & (df["jokertypes"] == 1)]["choices_GD"].mean())
            
            if dat_type == 'exp':
                data_new["datatype"].append("experimental (Group %s)"%group)
                
            elif dat_type == 'sim':
                data_new["datatype"].append("simulated (Group %s)"%group)
            
            "Incongruent Jokers"
            data_new["blockidx"].append(block)
            data_new["trialtype"].append("incongruent")
            data_new["HPCF"].append(df[(df["blockidx"] == block) & (df["jokertypes"] == 2)]["choices_GD"].mean())
            
            if dat_type == 'exp':
                data_new["datatype"].append("experimental (Group %s)"%group)
                
            elif dat_type == 'sim':
                data_new["datatype"].append("simulated (Group %s)"%group)
                
        else:
            raise Exception("Fehla!")
    
    data_new['group'] = [group]*len(data_new['blockidx'])
    return data_new, data_Q

def plot_results(data_sim, *args, **kwargs):
    '''
    Plots behaviour of a single agent.
    
    Parameters
    ----------
    data_sim : DataFrame
        DataFrame with data of single agent to be plotted.
        Columns:
            choices
            outcomes
            trialsequence
            blocktype
            jokertypes
            blockidx
            group
            
    group : int
        Experimental group of agent.
        
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
    
    group = data_sim['group'][1]
    assert not torch.is_tensor(group), "Group must not be tensor."
    
    if args:
        datas = (data_sim, args[0])
    
    elif not args: 
        datas = (data_sim,)
        
    else:
        raise Exception("Fehla!")
        
    "Create df_new, which will contain modified results for plotting and will contain column 'datatype' for simulated data and experimental data"
    df_new = pd.DataFrame()
    
    dat_type = ['exp', 'sim']
    for i in range(len(datas)):
        df = pd.DataFrame(data=datas[i])
        
        data_new, data_Q = arrange_data_for_plot(dat_type[i], df)
        
        df_temp = pd.DataFrame(data=data_new)
        df_new = pd.concat([df_new, df_temp])
        # df_Q = pd.DataFrame(data=data_Q)

    # fig, ax = plt.subplots()
    if group == 0 or group == 2:
        custom_palette = ['r', 'g', 'b'] # random, congruent, incongruent
        
    elif group == 1 or group == 3:
        custom_palette = ['g', 'b', 'r'] # congruent, incongruent, random
    # dfgh        
    # custom_palette = ['r', 'g', 'b'] # random, congruent, incongruent
    sns.relplot(x='blockidx',
                y='HPCF',
                hue = 'trialtype',
                data=df_new,
                kind='line',
                col='datatype',
                palette = custom_palette)
    
    # plt.plot([5.5, 5.5], [0.5,1], color='black') # plot Day1/Day2 line
    
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
    
    return df_new[df_new["datatype"] == "simulated (Group %d)"%group], \
        df_new[df_new["datatype"] == "experimental (Group %d)"%group]

def cohens_d(c0,c1):
    return (mean(c0) - mean(c1)) / (sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2))

def simulate_model_behaviour(num_agents, model, **kwargs):

    assert model == 'B'
    
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
                "jokertypes": newenv.data["jokertypes"], "blockidx": newenv.data["blockidx"], \
                "Qdiff": [(newenv.agent.Q[i][...,0] + newenv.agent.Q[i][...,3])/2 - (newenv.agent.Q[i][...,1] + newenv.agent.Q[i][...,2])/2 for i in range(len(newenv.choices))]}

        df = pd.DataFrame(data)

        data_new, data_Q = arrange_data_for_plot('sim', df, group = 0)
        
        df_new = pd.DataFrame(data_new)
        
        df_all = pd.concat((df_all, df_new))
        
    custom_palette = ['r', 'g', 'b'] # random, congruent, incongruent
    sns.relplot(x="blockidx", y="HPCF", hue = "trialtype", data=df_all, kind="line", palette=custom_palette)
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
            jokertypes
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
        newgroupdata["choices"].append(choices)
        newgroupdata["outcomes"].append(outcomes)
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
    
    assert Q_init.shape == (num_agents, 4)
    
    import models_torch as models
    
    k = 4.
    if model =='original':
        num_params = models.Vbm.num_params #number of latent model parameters
        
        if params is None:
            print("Setting random parameters.")
            params = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
        else:
            print("Setting initial parameters as provided.\n")
        
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
                  params = None,
                  plotres = True):
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
        
    plotres : bool, optional
        Plot results.
        
    Returns
    -------
    data : dict of lists
        Keys: 
            choices : list of len (num_trials). Each list element a list of len (num_agents).
            choices_GD : list
            outcomes : 0/1 reward/ no reward
            trialsequence : list, stimuli as seen by participants
            blocktype : list
            jokertypes : list
            blockidx : number of block as experienced by participants
            group : list
            
    params_true : torch tensor, shape [num_params, num_agents]
        Contains the parameter values with which the simulations were performed.

    params_true_df : DataFrame
        Contains the parameter values with which the simulations were performed.

    '''
    
    start = time.time()
    print("Simulating %d agents."%num_agents)
    assert torch.is_tensor(Q_init), "Q_init must be tensor."
    assert Q_init.shape == (num_agents, 4), "Q_init must have shape (num_agents, 4)."
    
    
    if sequence is None:
        print("Setting sequence to ones.")
        sequence = [1]*num_agents
        
    if blockorder is None:
        blockorder = [1]*num_agents
        print("Setting blockorder to ones.")
        
    assert len(blockorder) == num_agents
    assert len(sequence) == num_agents
    assert torch.all(torch.tensor(sequence) > 0), "list must only contain 1 and 2."
    assert torch.all(torch.tensor(blockorder) > 0), "blockorder must only contain 1 and 2."
        
    print('Inferring group.')
    seq_torch = torch.tensor(sequence)
    blockorder_torch = torch.tensor(blockorder)
    groups = (seq_torch == 1).type(torch.int) * (blockorder_torch == 1).type(torch.int)*0 +\
    (seq_torch == 1).type(torch.int) * (blockorder_torch == 2).type(torch.int)*1 +\
    (seq_torch == 2).type(torch.int) * (blockorder_torch == 1).type(torch.int)*2 +\
    (seq_torch == 2).type(torch.int) * (blockorder_torch == 2).type(torch.int)*3
    
    if model == 'original':
        num_params = models.Vbm.num_params #number of latent model parameters
        
    elif model == 'B':
        num_params = models.Vbm_B.num_params #number of latent model parameters
    
    if params is not None:
        assert params.shape[0] == num_params, "Number of parameters incorrect for selected model."
    
    # groupdata_list = []
    # groupdata_df = pd.DataFrame(columns = ['choices',
    #                                        'outcomes',
    #                                        'trialsequence',
    #                                        'blocktype',
    #                                        'jokertypes',
    #                                        'blockidx',
    #                                        'ag_idx',
    #                                        'group'])
    
    
    "New Code"
    # print("Simulating %d agents in parallel"%num_agents)
    # print("sequence:")
    # print(sequence)
    # print("blockorder:")
    # print(blockorder)
    
    "----- Initialize agent"
    if params == None:
        params_true = torch.zeros((num_params, num_agents))
        newagent = init_agent(model, 
                              Q_init, 
                              num_agents = num_agents)
    
        for param_idx in range(len(newagent.param_names)):
            params_true[param_idx, :] = newagent.par_dict[newagent.param_names[param_idx]]
    
    else:
        newagent = init_agent(model, 
                              Q_init, 
                              num_agents = num_agents, 
                              params = params)
    
        params_true = params
    

    "----- Set environment for agent"
    newenv = env.Env(newagent, matfile_dir = './matlabcode/clipre/')
    
    "----- Simulate"
    newenv.run(sequence = sequence,
               blockorder = blockorder)
    
    "----- Save Data"
    data = {'choices': newenv.choices, 
            'choices_GD' : newenv.choices_GD,
            'outcomes': newenv.outcomes,
            'trialsequence': newenv.data['trialsequence'], 
            'blocktype': newenv.data['blocktype'],
            'jokertypes': newenv.data['jokertypes'], 
            'blockidx': newenv.data['blockidx'],
            'group': [groups.tolist()]*len(newenv.choices),
            'ag_idx' : [torch.arange(num_agents).tolist()]*len(newenv.choices)}
    
    for key in data:
        assert(len(data[key])==6734)
    # dfgh
    # "Old Code"
    # "Simulate with random parameters"
    # for ag_idx in range(num_agents):
        
    #     print("Simulating agent no. %d.\n"%ag_idx)
        
    #     "----- Initialize agent"
    #     if params == None:
    #         newagent = init_agent(model, 
    #                               Q_init[ag_idx:ag_idx+1, :], 
    #                               num_agents = 1)
        
    #     else:
    #         newagent = init_agent(model, 
    #                               Q_init[ag_idx:ag_idx+1, :], 
    #                               num_agents = 1, 
    #                               params = params[:, ag_idx:ag_idx+1])
        
    #     for param_idx in range(len(newagent.param_names)):
    #         params_true[param_idx, ag_idx] = newagent.par_dict[newagent.param_names[param_idx]]

    #     "----- Set environment for agent"
    #     newenv = env.Env(newagent, matfile_dir = './matlabcode/clipre/')
    #     "----- Simulate"
    #     newenv.run(sequence = sequence[ag_idx],
    #                blockorder = blockorder[ag_idx])
    #     "----- Save Data"
    #     data = {'choices': newenv.choices, 
    #             'outcomes': newenv.outcomes,
    #             'trialsequence': newenv.data['trialsequence'], 
    #             'blocktype': newenv.data['blocktype'],
    #             'jokertypes': newenv.data['jokertypes'], 
    #             'blockidx': newenv.data['blockidx'],
    #             'ag_idx': [ag_idx]*len(newenv.choices),
    #             'group': [groups[ag_idx].item()]*len(newenv.choices)}
    #     groupdata_list.append(data)    
    #     groupdata_df = pd.concat((groupdata_df, pd.DataFrame(data)))
    # if plotres:
    #     plot_results(pd.DataFrame(data), group = groups[ag_idx].item(), title = 'Simulated Data')
        
    "----- Create DataFrame containing the parameters."
    columns = []
    for key in newagent.par_dict.keys():
        columns.append(key + '_true')
    params_true_df = pd.DataFrame(params_true.numpy().T, columns = columns)
    
    # "----- Create DataFrame with simulated data of whole group."
    # groupdata_df = groupdata_df.applymap(replace_single_element_lists)

    # if plotres:
    #     plot_grouplevel(groupdata_df)


    # if 0:
    #     return groupdata_list, groupdata_df, params_true, params_true_df
    # else:
    #     return groupdata_list
    
    print("Simulation took %.4f seconds."%(time.time()-start))
    return data, params_true, params_true_df

def plot_grouplevel(groupdata_df):
    '''
    Parameters
    ----------
    groupdata_df : DataFrame
        columns
            choices
            choices_GD
            outcomes
            trialsequence
            blocktype
            jokertypes
            blockidx
            group
            ag_idx

    Returns
    -------
    None.

    '''
    
    "Remove -1"
    groupdata_df = groupdata_df[groupdata_df['choices'] != -1]
    groupdata_df = groupdata_df[groupdata_df['jokertypes'] != -1]
    
    "---------- Create new column block_num for different blockorders"
    blocknums_blockorder2 = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]
    groupdata_df['block_num'] = groupdata_df.apply(lambda row: \
                                         blocknums_blockorder2[row['blockidx']] if row['group']==1 or row['group']==3 \
                                         else row['blockidx'], axis=1)
    groupdata_df.drop(['group', 'blockidx', 'ag_idx', 'trialsequence', 'outcomes', 'choices'], axis = 1, inplace = True)
    groupdata_df['jokertypes']=groupdata_df['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'no joker')))
    
    custom_palette = ['r', 'g', 'b'] # random, congruent, incongruent
    sns.relplot(x="block_num", y="choices_GD", hue = "jokertypes", data=groupdata_df, kind="line", palette=custom_palette)
    plt.show()
    
    # "----- Plot Group-Level behaviour"
    # all_ag_df = pd.DataFrame()
    # for ag_idx in range(len(groupdata_df['ag_idx'].unique())):
    #     df = groupdata_df[groupdata_df['ag_idx']==ag_idx]
    #     ag_dict, _ = arrange_data_for_plot('sim', df)
    #     ag_df = pd.DataFrame(ag_dict)
    #     ag_df['ag_idx'] = [ag_idx]*len(ag_df)
    #     all_ag_df = pd.concat((all_ag_df, ag_df))
    
    
    # "---------- Create new column block_num"
    # blocknums_blockorder2 = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]
    # all_ag_df['block_num'] = all_ag_df.apply(lambda row: \
    #                                          blocknums_blockorder2[row['blockidx']] if row['group']==1 or row['group']==3 \
    #                                          else row['blockidx'], axis=1)
    # all_ag_df.drop(['datatype', 'group', 'blockidx', 'ag_idx'], axis = 1, inplace = True)
    # # grouped = all_ag_df.groupby(['block_num', 'trialtype'])
    
    # # fig, ax = plt.subplots()
    # custom_palette = ['g', 'b', 'r'] # congruent, incongruent, random
    # sns.relplot(x='block_num', 
    #             y='HPCF', 
    #             hue='trialtype', 
    #             data = all_ag_df, 
    #             kind='line', 
    #             palette = custom_palette)
    # plt.title('Group-Level behaviour')
    # # plt.show()
    
# def simulate_data_parallel(model, 
#                             Q_init,
#                             sequence,
#                             blockorder,
#                             params):
#     '''
    

#     Parameters
#     ----------
#     model : TYPE
#         DESCRIPTION.
        
#     Q_init : tensor, shape [num_agents, 4]
#         Initial Q-Values of the agent.
        
#     sequence : list, len num_agents
#         Whether sequence or mirror sequence.
#         1/2 : sequence/ mirror sequence
#         Mirror sequence has reversed reward probabilities.
        
#     blockorder : list, len num_agents
#         Which blockorder
#         1/2 : RSRSRS SRSRSRSR / SRSRSR RSRSRSRS
        
#     params : tensor, shape (num_params, num_agents)
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     '''
#     assert isinstance(sequence, list), "sequence must be list"
#     assert isinstance(blockorder, list), "blockorder must be list"
#     assert torch.all(torch.tensor(sequence) > 0), "list must only contain 1 and 2."
#     assert torch.all(torch.tensor(blockorder) > 0), "blockorder must only contain 1 and 2."
    
#     num_agents = params.shape[1]
#     num_cpus = 10
#     assert num_agents%num_cpus == 0, "num_agents must be a multiple of num_cpus."
    
#     print("Simulating %d agents across %d cores.\n"%(num_agents, 10))
    
#     num_agents_per_cpu = num_agents//num_cpus
    
#     args = []
#     for ag_idx in range(0, num_agents, num_agents_per_cpu):
#         args.append((model, 
#                  num_agents_per_cpu, 
#                  Q_init[ag_idx:ag_idx+num_agents_per_cpu,:],
#                  sequence[ag_idx:ag_idx+num_agents_per_cpu],
#                  blockorder[ag_idx:ag_idx+num_agents_per_cpu],
#                  params[:, ag_idx:ag_idx+num_agents_per_cpu],
#                  False))
    
#     with mp.Pool(num_cpus) as pool:
#         result = pool.starmap(simulate_data, args)
        
#     dfgh


# def grouplevel_dict_to_df(groupdict):
    
#     num_agents = len(groupdict['choices'][0])
#     num_trials = len(groupdict['choices'])
    
#     df_data = {}
#     "Define columns"
#     for key in groupdict.keys():
#         df_data[key] = []
    
#     df_data['trialidx']
#     for trial in range(num_trials):

def posterior_predictives(post_sample, plot_single = True):
    '''
    
    Parameters
    ----------
    post_sample : DataFrame
        columns
            parameter columns
            subject
            group
            
    plot_single : bool, optional
        Whether to plot behaviour of single agents. Default is True.

    Returns
    -------
    complete_df : DataFrame
        DataFrame with average simulated behaviour for each agent.

    '''
    
    num_agents = len(post_sample['subject'].unique())
    num_reps = len(post_sample[post_sample['subject'] == 0])
    assert num_reps <= 1000, "Number of repetitions must be less than 1000."
    sequences = [1, 1, 2, 2]
    blockorders = [1, 2, 1, 2]
    Qs = torch.tensor([[0.2, 0., 0., 0.2],
                       [0, 0.2, 0.2, 0.]]).tile((num_reps, 1, 1))
    
    complete_df = pd.DataFrame()
    
    for ag_idx in range(num_agents):
        print("Simulating agent %d of %d with %d repetitions."%(ag_idx+1, num_agents, num_reps))
        post_sample_agent = post_sample[post_sample['subject'] == ag_idx]
        model = post_sample_agent['model'].unique()[0]
        sequence = [sequences[post_sample_agent['group'].unique()[0]]]*num_reps
        Q_init = Qs[range(num_reps),torch.tensor(sequence)-1,:]
        blockorder = [blockorders[post_sample_agent['group'].unique()[0]]]*num_reps
        
        
        # params = torch.tensor(np.random.rand(3, num_agents))
        params = torch.tensor(post_sample_agent.iloc[:, 0:-3].to_numpy().T)
        # dfgh #test whether correct
        ag_data_dict, params, params_df = simulate_data(model, 
                                                            num_reps,
                                                           Q_init = Q_init,
                                                            sequence = sequence,
                                                            blockorder = blockorder,
                                                           params = params)
        
        agent_df = pd.DataFrame(ag_data_dict).explode(list(ag_data_dict.keys()))
        
        groupdata_df = agent_df
        groupdata_df = groupdata_df[groupdata_df['choices'] != -1]
        groupdata_df = groupdata_df[groupdata_df['jokertypes'] != -1]
        
        "---------- Create new column block_num for different blockorders"
        blocknums_blockorder2 = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]
        groupdata_df['block_num'] = groupdata_df.apply(lambda row: \
                                             blocknums_blockorder2[row['blockidx']] if row['group']==1 or row['group']==3 \
                                             else row['blockidx'], axis=1)
        groupdata_df.drop(['group', 'blockidx', 'ag_idx', 'trialsequence', 'outcomes', 'choices'], axis = 1, inplace = True)
        groupdata_df['jokertypes']=groupdata_df['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'no joker')))
        grouped_df = pd.DataFrame(groupdata_df.groupby(['block_num','jokertypes'], as_index = False).mean())
        grouped_df['ag_idx'] = ag_idx
        
        if plot_single:
            sns.relplot(x='block_num', y= 'choices_GD', hue='jokertypes', kind='line', data = grouped_df)
            plt.title('Agent %d'%ag_idx)
            plt.show()
        complete_df = pd.concat((complete_df, grouped_df))
    
        assert list(post_sample_agent.columns[0:-3]) == [col[0:-5] for col in params_df.columns]
        
    custom_palette = ['r', 'g', 'b'] # random, congruent, incongruent
    sns.relplot(x="block_num", y="choices_GD", hue = "jokertypes", data=complete_df, kind="line", palette=custom_palette)
    print("Finished posterior predictives.")
    return complete_df