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

def load_matfiles(matfile_dir, 
                  blocknr, 
                  blocktype, 
                  sequence = 1):
    '''
    Gets stimulus sequences from .mat-files.

    Parameters
    ----------
    matfile_dir : str
        Storage directory of the .mat-files.
        
    blocknr : int
        Number of block of a given type (1-7). Not the same as blockidx.
        
    blocktype : int
        0/1 sequential/ random
        
    sequence : int, optional
        1/2 sequence/ mirror sequence. The default is 1.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    seq: list
        Stimulus sequence as seen by participants in experiment (1-indexed.)
        
    seq_no_jokers: list
        Sequence without jokers (i.e. without DTT). 1-indexed.
        
    jokertypes : list
        DTT Types
        -1/0/1/2 : no joker/random/congruent/incongruent
    '''

    if sequence == 2:
        prefix = "mirror_"
        
    else:
        prefix = ""
                    
    if blocktype == 0:
        "sequential"
        mat = scipy.io.loadmat(matfile_dir + prefix + 'trainblock' + str(blocknr+1) + '.mat')
        
    elif blocktype == 1:
        "random"
        mat = scipy.io.loadmat(matfile_dir + prefix + 'random' + str(blocknr+1) + '.mat')
        
    else:
        raise Exception("Problem with los blocktypos.")

    seq = mat['sequence'][0]
    seq_no_jokers = mat['sequence_without_jokers'][0]
    
    "----- Map Neutral Jokers to 'No Joker' (only necessary for published results)."
    seq_noneutral = [t if t != 14 and t != 23 else 1 for t in seq]    
    
    "----- Determine congruent/ incongruent jokers"
    if blocktype == 0:
        "sequential"
        jokers = [-1 if seq_noneutral[tidx]<10 else seq_no_jokers[tidx] for tidx in range(len(seq))]
        if sequence == 1:
            jokertypes = [j if j == -1 else 1 if j == 1 else 2 if j == 2 else 2 if j == 3 else 1 for j in jokers]
            
        elif sequence == 2:
            jokertypes = [j if j == -1 else 2 if j == 1 else 1 if j == 2 else 1 if j == 3 else 2 for j in jokers]
            
        else:
            raise Exception("Fehla!!")
                    
    elif blocktype == 1:
        "random"
        jokertypes = [-1 if seq_noneutral[tidx]<10 else 0 for tidx in range(len(seq_noneutral))]
    
    return torch.squeeze(torch.tensor(seq)).tolist(), \
        torch.squeeze(torch.tensor(seq_no_jokers)).tolist(), \
            jokertypes

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
            choices : nested list, 'shape' [num_trials, num_agents]
            choices_GD
            outcomes : nested list, 'shape' [num_trials, num_agents]
            trialsequence : nested list, 'shape' [num_trials, num_agents]
            blocktype : nested list, 'shape' [num_trials, num_agents]
            jokertypes
            blockidx : nested list, 'shape' [num_trials, num_agents]
            RT : nested list, 'shape' [num_trials, num_agents]
            group : list, len [num_trials, num_agents]. 0-indexed
            ag_idx
            ID
            
    groupdata_df : DataFrame

    '''
    
    groupdata = []
    group = []
    'Exclude because of errors'
    # exclude_errors = [# Grp 0
    #                 # Grp1
    #                 14, 20, 24, 28,
    #                 # Grp2
    #                 38, 43, 45, 
    #                 # Grp3
    #                 57, 58]
    
    exclude_errors = [# Grp 0
                  
                  # Grp 1
                  '5e07c976f6191f137214e91f',
                  # Grp 2
                  ]
    
    'Exclude because of execution time'
    exclude_time = [# Grp 0
                  '604fa685e33606f9a0ee8189', # pb 5
                  # Grp 1
                '63ef938aea545fb9bd19a591', '56d8df02d13f6b000e542eef',  # pb 14, 15
                '601f08cba1191642cadf59c1', '6151f20d06757e43aa1f54f9', '5f48f497fae2763d537d2e6b', # pb 23, 25, 26
                
                # Grp 2
                '62cbed33d29c28e6be511bde', '5a54d42476d1c60001aacd6c', # pb 33, 44
                '59e7232f24d7bf00012f112e', # pb 43
                # '63e5eb956eab1f2740ac6289', 
                # Grp 3
                '5eebe3d7b1914c17e6208284', '6329c510ea44255e948f8492', # pb 50, 63
                '5ea00d4b1286ee0008405450'] #pb 58
    
    exclude_random = []
    exclude_random.append(np.random.choice(range(14)))
    exclude_random.extend(list(np.random.choice([32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 45, 46, 47, 48], size = 2, replace = False)))
    exclude_random.append(np.random.choice([49, 51, 52, 53, 54, 55, 56, 57, 59,60,61,62,64]))
    
    exclude_random.sort()
    print(exclude_random)
    
    IDs_included = []
    
    hand = []
    'left/right/both: 0/1/2'
    handedness = {
                '5483702ffdf99b24c654b810'	: 1,
                '5d55b7ef6a0f930017202336'	: 1,
                '629f6b8c65fcae219e245284'	: 1,
                '5b5e0e86902ad10001cfcc59'	: 1,
                '5efb31fa8cd32f04bf048643'	: 1,
                '5cbe04b4f429ff00159de30e'	: 1,
                '5982eef79dfc3e00011d81e0'	: 1,
                '57dd186e6598aa0001992616'	: 1,
                '5d8a29c082fec30001d9c24a'	: 1,
                '56d8df02d13f6b000e542eef'	: 1,
                '62db2644ab0a3a353c0dcb54'	: 1,
                '63ef938aea545fb9bd19a591'	: 0,
                '5a54d42476d1c60001aacd6c'	: 1,
                '5b2a72f7c293b90001732b89'	: 1,
                '59dd90f6e75b450001a68dac'	: 1,
                '63d79fcd741f9cfb2f87152f'	: 1,
                '63174af7d57182f9bf90c094'	: 1,
                '5d5a75c570a7c1000152623e'	: 1,
                '5e850b0e390e520ec806b084'	: 1,
                '55ca5305fdf99b498b73f29c'	: 1,
                '5edeea354d54cb1a825aa8f9'	: 1,
                '60e2577f8c52db9d1fb5ffac'	: 1,
                '57c4761195a3ea00016e5992'	: 0,
                '63e61d24e1c77b4b584e879f'	: 1,
                '5fdcd228539a382bc5e9fae5'	: 1,
                '640103805957bbf6a1326ed0'	: 1,
                '5fdcd228539a382bc5e9fae5'	: 1,
                '5fdcd228539a382bc5e9fae5'	: 1,
                '5fd652b056a8bc53dbd1f680'	: 0,
                '5fd652b056a8bc53dbd1f680'	: 0,
                '5fbacf8a8f01f4b486dea42a'	: 1,
                '5fb46dd5d9ece50422838e7a'	: 1,
                '5ee38ef98c0c440172e022ac'	: 1,
                '5e66c77e8ebdaf4466e26326'	: 0,
                '5c4594514fe4f800016ea070'	: 1,
                '5c8ba607c5366c0001d022cb'	: 1,
                '63af557b3d4f219c3226b7d6'	: 1,
                '617406fbfced12169896d158'	: 1,
                '63f77b9ca8439f6e1620c1cd'	: 0,
                '63f77b9ca8439f6e1620c1cd'	: 0,
                '5e5fe64a8278d103d35cac54'	: 1,
                '601f08cba1191642cadf59c1'	: 1,
                '5d0245966e208b0017301561'	: 1,
                '62e02b26e879244a99e852fa'	: 1,
                '5d7ebf9e93902b0001965912'	: 1,
                '5d49d17b3dad1f0001e2aba1'	: 1,
                '5efcb50255a06b097b8d5d29'	: 1,
                '56f699e876348f000c883bba'	: 1,
                '604fa685e33606f9a0ee8189'	: 1,
                '63e5eb956eab1f2740ac6289'	: 1,
                '5f6bfe84b3d4ba1796defbfb'	: 1,
                '6044ca22bc6235555362d5bb'	: 1,
                '63eaa48e178ff1aadbf62963'	: 1,
                '5eecd506d78b3b0a7182ccb9'	: 1,
                '5eac671f0658ac0ea6c730d2'	: 2,
                '5db4ef4a2986a3000be1f886'	: 1,
                '5b266738007d870001c7c360'	: 1,
                '5eb6d09db2c9c85dd30dc94e'	: 1,
                '5b2bcfb51835660001cc6751'	: 1,
                '5f356cbffb4cea5170d04fd9'	: 1,
                '5c321ebf6558270001bd79aa'	: 1,
                '5dc5da21d999de45a504651b'	: 1,
                '58aca85e0da7f10001de92d4'	: 1,
                '646dc9848e53ff1705d8c9c5'	: 1,
                '5f16fde210d37701904c9dc2'	: 1,
                '60a3f8075b013de7b5518e96'	: 1,
                '5a9ed5046475f90001a0189e'	: 1,
                '596f961cfe061d00011e3e03'	: 1,
                '57d5ab3a722df500017f3622'	: 1,
                '5f48f497fae2763d537d2e6b'	: 1,
                '5c4b987538878c0001c7883b'	: 1,
                '5908458b1138880001bc77e7'	: 1,
                '5eebe3d7b1914c17e6208284'	: 1,
                '62b05fdedd70cd33ea6afbd2'	: 2,
                '62c8391cd913ab9b5317d5f9'	: 1,
                '6329b1add3dcd53cb9c9cab8'	: 1,
                '5eaadc0a7adeb404eea9c3c0'	: 1,
                '62b44f66a16d45783569fad6'	: 1,
                '5db32244dbe39d000be72fb0'	: 1,
                '6286672d0165aad8f1386c27'	: 1,
                '5e9ffa9ee172201b04f1a4c6'	: 1,
                '631e235a33040da66ddf51a1'	: 1,
                '5d8cef3c28e183001a335ab0'	: 1,
                '5f0f7fe1d7ad1c000b42d091'	: 1,
                '595e7974af78da0001a21c3a'	: 1,
                '5c9f26cbe57be600147042e5'	: 1,
                '6116b022b7ef87ef5828748b'	: 0,
                '5e07c976f6191f137214e91f'	: 1,
                '60f816ff1fa74fcfab532378'	: 1,
                '5daf58c627595a00117d30b7'	: 1,
                '5d5bdf323feb0c00195dcf2e'	: 1,
                '6151f20d06757e43aa1f54f9'	: 1,
                '5e8f94bcdc69fd294b61ebcc'	: 1,
                '5d7fa4fd6eff6f001a16b4aa'	: 1,
                '630389e7772b100ba592cdee'	: 1,
                '5eec9ee7d900510326d78fc8'	: 1,
                '62cbed33d29c28e6be511bde'	: 1,
                '615739949cf5767509a7e29a'	: 1,
                '6329c510ea44255e948f8492'	: 1,
                '5ea00d4b1286ee0008405450'	: 1,
                '59e7232f24d7bf00012f112e'	: 1,
        }
    
    pb = -1
    for grp in range(4):
        files_day1 = glob.glob(data_dir + "Grp%d/csv/*Tag1*.mat"%(grp+1))
        for file1 in files_day1:
            "Loop over participants"
            pb += 1
            data, ID = get_participant_data(file1, 
                                            grp, 
                                            data_dir)
            
            if ID not in handedness.keys():
                raise Exception('Handedness missing for ID %s'%ID)
            
            if ID not in exclude_time and ID not in exclude_errors and pb not in exclude_random:
                groupdata.append(data)
                group.append(grp)
                IDs_included.append(ID)
                hand.append(handedness[ID])

    newgroupdata = comp_groupdata(groupdata)
    num_trials = len(newgroupdata['trialsequence'])
    num_agents = len(newgroupdata['trialsequence'][0])
    newgroupdata['group'] = [group]*num_trials
    newgroupdata['handedness'] = [hand]*num_trials
    newgroupdata['ID'] = [IDs_included]*num_trials
    newgroupdata['ag_idx'] = [torch.arange(num_agents).tolist()]*num_trials
    newgroupdata['model'] = [['Experiment']*num_agents]*num_trials
    groupdata_df = pd.DataFrame(newgroupdata).explode(list(newgroupdata.keys()))
    
    dfnew = pd.DataFrame(groupdata_df.groupby(['ag_idx', 'group', 'model', 'ID'], as_index = False).mean())
    group_distro = [len(dfnew[dfnew['group']== grp]) for grp in range(4)]
    print(group_distro)
    assert np.abs(np.diff(group_distro)).sum() == 0
    
    return newgroupdata, groupdata_df

def get_participant_data(file_day1, group, data_dir):
    '''
    Get data of an individual participant.
    
    Parameters
    ----------
    file_day1 : str
        File of 1st part of experiment.
        
    group : int
        Experimental Group
        
    data_dir : str
        Where experimental data is stored.

    Returns
    -------
    data : dict
        Contains experimental data.
        Keys:
            trialsequence : list of list of ints
            trialsequence_no_jokers : list of list of ints
            choices : list, len num_trials, -2,-1,0,1,2, or 3
            choices_GD :
            outcomes : list of list of ints
            blocktype : list of list of ints
            jokertypes : list
            blockidx
            RT : list of list of floats (RT in ms)
        
    ID : str
        Participant-specific ID.

    '''
    
    assert group < 4
    
    ID = file_day1.split("/")[-1][4:28] # Prolific ID

    # print(data_dir)
    # print(glob.glob(data_dir + "Grp%d/csv/*%s*Tag2*.mat"%(group+1, ID)))
    
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
    jokertypes = []
    blocktype = []
    blockidx = []
    for block in range(14):
        "Mark the beginning of a new block"
        trialsequence.append(-1)
        trialsequence_wo_jokers.append(-1)
        jokertypes.append(-1)
        blocktype.append(-1)
        blockidx.append(block)
        
        seq, seq_wo_jokers, jtypes, btype = get_trialseq('./matlabcode/clipre/',
                                                 group, 
                                                 block)
        
        trialsequence.extend(seq)
        trialsequence_wo_jokers.extend(seq_wo_jokers)
        jokertypes.extend(jtypes)
        blocktype.extend([btype]*len(seq))
        blockidx.extend([block]*len(seq))
    
    assert len(trialsequence) == len(choices)
    assert len(outcomes) == len(choices)
    assert len(outcomes) == len(blocktype)
    
    num_trials = len(choices)
    
    assert len(trialsequence) == num_trials and len(trialsequence_wo_jokers) == num_trials and len(jokertypes) == num_trials and \
        len(outcomes) == num_trials and len(blocktype) == num_trials and len(blockidx) == num_trials and len(RT) == num_trials
    
    jokertypes = [[jokertypes[i]] for i in range(num_trials)]
    trialsequence = [[trialsequence[i]] for i in range(num_trials)]
    trialsequence_wo_jokers = [[trialsequence_wo_jokers[i]] for i in range(num_trials)]
    choices = [[choices[i]] for i in range(num_trials)]
    outcomes = [[outcomes[i]] for i in range(num_trials)]
    blocktype = [[blocktype[i]] for i in range(num_trials)]
    blockidx = [[blockidx[i]] for i in range(num_trials)]
    RT = [[RT[i]] for i in range(num_trials)]
        
    if group == 0 or group == 1:
        choices_GD = torch.where(torch.logical_or(torch.tensor(choices) == 0, torch.tensor(choices) == 3), torch.ones(torch.tensor(choices).shape), torch.zeros(torch.tensor(choices).shape))
        
    elif group == 2 or group == 3:
        choices_GD = torch.where(torch.logical_or(torch.tensor(choices) == 1, torch.tensor(choices) == 2), torch.ones(torch.tensor(choices).shape), torch.zeros(torch.tensor(choices).shape))
        
    choices_GD = torch.where(torch.tensor(outcomes) == -1, -1*torch.ones(torch.tensor(choices).shape), choices_GD)
    assert torch.all(choices_GD <= 1)
    
    data = {'trialsequence': trialsequence,
            'trialsequence_no_jokers': trialsequence_wo_jokers,
            'jokertypes' : jokertypes,
            'choices': choices,
            'choices_GD' : choices_GD.type(torch.int).tolist(),
            'outcomes': outcomes,
            'blocktype': blocktype,
            'blockidx': blockidx,
            'RT': RT}
    return data, ID
    
def get_trialseq(matfile_dir, 
                 group, 
                 blockidx):
    '''

    Parameters
    ----------
    matfile_dir : str
        Storage directory of the .mat-files.
        
    group : int
        Experimental group.
        
    blockidx : int
        Number of block as seen by agent. Not the same as blocknr in load_matfiles().

    Returns
    -------
    seq: list
        Stimulus sequence as seen by participants in experiment (1-indexed.)
        
    seq_no_jokers: list
        Sequence without jokers (i.e. without DTT). 1-indexed.

    jokertypes : list
        DTT Types
        -1/0/1/2 : no joker/random/congruent/incongruent

    blocktype : int
        0/1 : sequential/ random block

    '''
    
    "NB: in mat-files, the block order was already swapped, as if all participants saw the first group's block order! Have to correct for this!"
    
    "This is the blockorder participants actually saw in the 4 groups."
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

    types = [[1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
             [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
              [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
              [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]]
    
    seqs = [1, 1, 2, 2]
    sequence = seqs[group]

    blocktype = types[group][blockidx]

    mat = scipy.io.loadmat(matfile_dir + "%s.mat"%blockorder[group][blockidx])
    
    seq = mat['sequence'][0]
    seq_no_jokers = mat['sequence_without_jokers'][0]
    
    "----- Map Neutral Jokers to 'No Joker' (only necessary for published results)."
    seq_noneutral = [t if t != 14 and t != 23 else 1 for t in seq]    
    
    "----- Determine congruent/ incongruent jokers"
    if blocktype == 0:
        "sequential"
        jokers = [-1 if seq_noneutral[tidx]<10 else seq_no_jokers[tidx] for tidx in range(len(seq))]
        if sequence == 1:
            jokertypes = [j if j == -1 else 1 if j == 1 else 2 if j == 2 else 2 if j == 3 else 1 for j in jokers]
            
        elif sequence == 2:
            jokertypes = [j if j == -1 else 2 if j == 1 else 1 if j == 2 else 1 if j == 3 else 2 for j in jokers]
            
        else:
            raise Exception("Fehla!!")
                    
    elif blocktype == 1:
        "random"
        jokertypes = [-1 if seq_noneutral[tidx]<10 else 0 for tidx in range(len(seq_noneutral))]
    
    return torch.squeeze(torch.tensor(seq)).tolist(), \
        torch.squeeze(torch.tensor(seq_no_jokers)).tolist(), \
            jokertypes, blocktype

def replace_single_element_lists(value):
    '''
    if value is list and length 1, return its element.

    Parameters
    ----------
    value : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
    return value

def cohens_d(c0,c1):
    return (mean(c0) - mean(c1)) / (sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2))
    
def comp_groupdata(groupdata):
    '''
    Parameters
    ----------
    groupdata : list of dicts, len num_agents
        Contains 1 dictionary per agent, with experimental data.
        Keys: 
            choices
            choices_GD
            outcomes
            trialsequence
            blocktype
            jokertypes
            blockidx
            RT

    Returns
    -------
    newgroupdata : dict
        Contains keys as in groupdata.
        len = num_trials
        Each element of a key is another list with length num_agents

    '''
    
    newgroupdata = {'choices' : [],
                    'choices_GD' : [],
                    'outcomes' : [],
                    'trialsequence' : [],
                    'blocktype' : [],
                    'jokertypes' : [],
                    'blockidx' : [],
                    'RT': []}

        
    num_trials = len(groupdata[0]["trialsequence"])
    for trial in range(num_trials):
        trialsequence = []
        choices = []
        choices_GD = []
        outcomes = []
        blocktype = []
        jokertypes = []
        blockidx = []
        RT = []

        for dt in groupdata:
            choices.append(dt['choices'][trial][0])
            choices_GD.append(dt['choices_GD'][trial][0])
            outcomes.append(dt['outcomes'][trial][0])
            trialsequence.append(dt['trialsequence'][trial][0])
            blocktype.append(dt['blocktype'][trial][0])
            jokertypes.append(dt['jokertypes'][trial][0])
            blockidx.append(dt['blockidx'][trial][0])
            RT.append(dt['RT'][trial][0])
            
        
        newgroupdata["choices"].append(choices)
        newgroupdata["choices_GD"].append(choices_GD)
        newgroupdata["outcomes"].append(outcomes)
        newgroupdata["trialsequence"].append(trialsequence)
        newgroupdata["blocktype"].append(blocktype)
        newgroupdata["jokertypes"].append(jokertypes)
        newgroupdata["blockidx"].append(blockidx)
        newgroupdata["RT"].append(RT)
    
    return newgroupdata

def init_agent(model, group, num_agents=1, params = None):
    '''
    
    Parameters
    ----------
    model : str
        The model class to simulate data with.

    group : list, len num_agents
        Experimental group(s) of agent(s).
        Determines Q_init.

    num_agents : int
        Number of agents in a single agent object.
        The default is 1.
        
    params : dict, each values is tensor or list with shape/ len [num_agents]
        Parameters to initialize agents with
        If None, parameters will be set randomly.


    Returns
    -------
    newagent : obj of class model

    '''
    
    print("Setting Q_init.")
    Qs = torch.tensor([[0.2, 0., 0., 0.2],
                       [0.2, 0., 0., 0.2],
                       [0, 0.2, 0.2, 0],
                       [0, 0.2, 0.2, 0.]]).tile((num_agents, 1, 1))
    Q_init = Qs[range(num_agents),torch.tensor(group), :]
    
    import models_torch as models
    
    if params is not None:
        for key in params.keys():
            if isinstance(params[key], list):
                if not isinstance(params[key][0], str):
                    params[key] = torch.tensor(params[key])
            
    k = 4.
    if model =='Vbm':
        num_params = models.Vbm.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['omega'] = params_uniform[0:1, :]
            param_dict['dectemp'] = (params_uniform[1:2, :]+1)*3
            param_dict['lr'] = params_uniform[2:3, :]*0.01
            
            
        else:
            assert isinstance(params, dict)
            print("Setting initial parameters as provided.\n")
            param_dict['omega'] = params['omega'][None,...]
            param_dict['dectemp'] = params['dectemp'][None,...]
            param_dict['lr'] = params['lr'][None,...]
        
        newagent = models.Vbm(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    elif model == 'Vbm_twodays':
        num_params = models.Vbm_twodays.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['omega_day1'] = params_uniform[0:1, :]
            param_dict['dectemp_day1'] = (params_uniform[1:2, :]+1)*3
            param_dict['lr_day1'] = params_uniform[2:3, :]*0.01
            
            param_dict['omega_day2'] = params_uniform[3:4, :]
            param_dict['dectemp_day2'] = (params_uniform[4:5, :]+1)*3
            param_dict['lr_day2'] = params_uniform[5:6, :]*0.01
            
            
        else:
            assert isinstance(params, dict)
            print("Setting initial parameters as provided.\n")
            param_dict['omega_day1'] = params['omega_day1'][None,...]
            param_dict['dectemp_day1'] = params['dectemp_day1'][None,...]
            param_dict['lr_day1'] = params['lr_day1'][None,...]
            
            param_dict['omega_day2'] = params['omega_day2'][None,...]
            param_dict['dectemp_day2'] = params['dectemp_day2'][None,...]
            param_dict['lr_day2'] = params['lr_day2'][None,...]
        
        newagent = models.Vbm_twodays(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init = Q_init[None, ...])
        
        
    elif model == 'B':
        num_params = models.Vbm_B.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr_day1'] = params_uniform[0:1, :]*0.01 # shape (1, num_agents)
            param_dict['theta_Q_day1'] = params_uniform[1:2, :]*6
            param_dict['theta_rep_day1'] = params_uniform[2:3, :]*6
            
            param_dict['lr_day2'] = params_uniform[3:4, :]*0.01
            param_dict['theta_Q_day2'] = params_uniform[4:5, :]*6
            param_dict['theta_rep_day2'] = params_uniform[5:6, :]*6
            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr_day1'] = params['lr_day1'][None,...]
            param_dict['theta_Q_day1'] = params['theta_Q_day1'][None,...]
            param_dict['theta_rep_day1'] = params['theta_rep_day1'][None,...]
            
            param_dict['lr_day2'] = params['lr_day2'][None,...]
            param_dict['theta_Q_day2'] = params['theta_Q_day2'][None,...]
            param_dict['theta_rep_day2'] = params['theta_rep_day2'][None,...]
            
        newagent = models.Vbm_B(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    elif model == 'Conflict':
        num_params = models.Conflict.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr_day1'] = params_uniform[0:1, :]*0.01
            param_dict['theta_Q_day1'] = params_uniform[1:2, :]*6
            param_dict['theta_rep_day1'] = params_uniform[2:3, :]*6
            param_dict['conflict_param_day1'] = (params_uniform[3:4, :]-0.5)*6
            
            param_dict['lr_day2'] = params_uniform[4:5, :]*0.01
            param_dict['theta_Q_day2'] = params_uniform[5:6, :]*6
            param_dict['theta_rep_day2'] = params_uniform[6:7, :]*6
            param_dict['conflict_param_day2'] = (params_uniform[7:8, :]-0.5)*6
            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr_day1'] = params['lr_day1'][None,...]
            param_dict['theta_Q_day1'] = params['theta_Q_day1'][None,...]
            param_dict['theta_rep_day1'] = params['theta_rep_day1'][None,...]
            param_dict['conflict_param_day1'] = params['conflict_param_day1'][None,...]
            
            param_dict['lr_day2'] = params['lr_day2'][None,...]
            param_dict['theta_Q_day2'] = params['theta_Q_day2'][None,...]
            param_dict['theta_rep_day2'] = params['theta_rep_day2'][None,...]
            param_dict['conflict_param_day2'] = params['conflict_param_day2'][None,...]
        
        newagent = models.Conflict(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    elif model == 'Seqparam':
        num_params = models.Seqparam.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr_day1'] = params_uniform[0:1, :]*0.01
            param_dict['theta_Q_day1'] = params_uniform[1:2, :]*6
            param_dict['theta_rep_day1'] = params_uniform[2:3, :]*6
            param_dict['seq_param_day1'] = params_uniform[3:4, :]*6
            
            param_dict['lr_day2'] = params_uniform[4:5, :]*0.01
            param_dict['theta_Q_day2'] = params_uniform[5:6, :]*6
            param_dict['theta_rep_day2'] = params_uniform[6:7, :]*6
            param_dict['seq_param_day2'] = params_uniform[7:8, :]*6
            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr_day1'] = params['lr_day1'][None,...]
            param_dict['theta_Q_day1'] = params['theta_Q_day1'][None,...]
            param_dict['theta_rep_day1'] = params['theta_rep_day1'][None,...]
            param_dict['seq_param_day1'] = params['seq_param_day1'][None,...]
            
            param_dict['lr_day2'] = params['lr_day2'][None,...]
            param_dict['theta_Q_day2'] = params['theta_Q_day2'][None,...]
            param_dict['theta_rep_day2'] = params['theta_rep_day2'][None,...]
            param_dict['seq_param_day2'] = params['seq_param_day2'][None,...]
            
        newagent = models.Seqparam(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    elif model == 'Random':
        num_params = models.Random.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['p_cong_day1'] = params_uniform[0:1, :]
            param_dict['p_incong_day1'] = params_uniform[1:2, :]
            param_dict['p_rand_day1'] = params_uniform[2:3, :]
            
            param_dict['p_cong_day2'] = params_uniform[3:4, :]
            param_dict['p_incong_day2'] = params_uniform[4:5, :]
            param_dict['p_rand_day2'] = params_uniform[5:6, :]
            
        else:
            raise Exception("Not yet implemented")

            
        newagent = models.Random(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    elif model == 'BQ':
        num_params = models.Vbm_B_Q.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            # print("Setting random parameters.")
            # locs = torch.tensor(np.random.uniform(-8,4, (1, num_agents, num_params)))
            # param_dict = models.Vbm_B_Q.locs_to_pars('None', locs)
            
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr_day1'] = params_uniform[0:1, :]*0.2
            param_dict['theta_Q_day1'] = params_uniform[1:2, :]*8
            param_dict['theta_rep_day1'] = params_uniform[2:3, :]*6
            
            param_dict['lr_day2'] = params_uniform[3:4, :]*0.1
            param_dict['theta_Q_day2'] = params_uniform[4:5, :]*8
            param_dict['theta_rep_day2'] = params_uniform[5:6, :]*6
            param_dict['Qparam'] = params_uniform[6:7, :]*500
            
        else:
            raise Exception("Not yet implemented")
            # print("Setting initial parameters as provided.")
            # param_dict['lr_day1'] = params['lr_day1'][None,...]
            # param_dict['theta_Q_day1'] = params['theta_Q_day1'][None,...]
            # param_dict['theta_rep_day1'] = params['theta_rep_day1'][None,...]
            
            # param_dict['lr_day2'] = params['lr_day2'][None,...]
            # param_dict['theta_Q_day2'] = params['theta_Q_day2'][None,...]
            # param_dict['theta_rep_day2'] = params['theta_rep_day2'][None,...]
            
        newagent = models.Vbm_B_Q(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
        
    elif model == 'Bk':
        num_params = models.Vbm_B_k.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            # print("Setting random parameters.")
            # locs = torch.tensor(np.random.uniform(-2,7, (1, num_agents, num_params)))
            # param_dict = models.Vbm_B_k.locs_to_pars('None', locs)
            
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr_day1'] = params_uniform[0:1, :]*0.01
            param_dict['theta_Q_day1'] = params_uniform[1:2, :]*8
            param_dict['theta_rep_day1'] = params_uniform[2:3, :]*6
            
            param_dict['lr_day2'] = params_uniform[3:4, :]*0.01
            param_dict['theta_Q_day2'] = params_uniform[4:5, :]*8
            param_dict['theta_rep_day2'] = params_uniform[5:6, :]*6
            param_dict['kparam'] = params_uniform[6:7, :]*1000
            
        else:
            raise Exception("Not yet implemented")
            # print("Setting initial parameters as provided.")
            # param_dict['lr_day1'] = params['lr_day1'][None,...]
            # param_dict['theta_Q_day1'] = params['theta_Q_day1'][None,...]
            # param_dict['theta_rep_day1'] = params['theta_rep_day1'][None,...]
            
            # param_dict['lr_day2'] = params['lr_day2'][None,...]
            # param_dict['theta_Q_day2'] = params['theta_Q_day2'][None,...]
            # param_dict['theta_rep_day2'] = params['theta_rep_day2'][None,...]
            
        newagent = models.Vbm_B_k(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    elif model=='Bhand':
        num_params = models.Handedness.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            # print("Setting random parameters.")
            # locs = torch.tensor(np.random.uniform(-2,7, (1, num_agents, num_params)))
            # param_dict = models.Handedness.locs_to_pars('None', locs)
            
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr_day1'] = params_uniform[0:1, :]*0.01
            param_dict['theta_Q_day1'] = params_uniform[1:2, :]*6
            param_dict['theta_rep_day1'] = params_uniform[2:3, :]*6
            
            param_dict['lr_day2'] = params_uniform[3:4, :]*0.01
            param_dict['theta_Q_day2'] = params_uniform[4:5, :]*6
            param_dict['theta_rep_day2'] = params_uniform[5:6, :]*6
            param_dict['hand_param'] = (params_uniform[6:7, :]-0.5)*80
            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr_day1'] = params['lr_day1'][None,...]
            param_dict['theta_Q_day1'] = params['theta_Q_day1'][None,...]
            param_dict['theta_rep_day1'] = params['theta_rep_day1'][None,...]
            
            param_dict['lr_day2'] = params['lr_day2'][None,...]
            param_dict['theta_Q_day2'] = params['theta_Q_day2'][None,...]
            param_dict['theta_rep_day2'] = params['theta_rep_day2'][None,...]
            
            param_dict['hand_param'] = params['hand_param'][None,...]

        newagent = models.Handedness(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    else:
        raise Exception("No model specified")
        
    return newagent
        
def simulate_data(model, 
                  num_agents,
                  group = None,
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

    group : list, len num_agents
        Experimental group.
        Given group, sequence and blockorder can be inferred.
        
        Determines sequence & blockorder.
        sequence : list, len num_agents
            Whether sequence or mirror sequence.
            1/2 : sequence/ mirror sequence
            Mirror sequence has reversed reward probabilities.
            
        blockorder : list, len num_agents
            Which blockorder
            1/2 : RSRSRS SRSRSRSR / SRSRSR RSRSRSRS
        
    params : DataFrame or dict, values are tensors or list, shape [num_agents]
        Contains the latent model parameters with which to simulate data.
        
    plotres : bool, optional
        Plot results.
        
    Returns
    -------
    data : dict of nested lists.
        Needed for inference.
        Keys: 
            choices : list of len (num_trials). Each list element a list of len (num_agents).
            choices_GD : list
            outcomes : 0/1 reward/ no reward
            trialsequence : list, stimuli as seen by participants
            blocktype : list
            jokertypes : list
            blockidx : number of block as experienced by participants
            group : list
            ag_idx : 
                
                
    group_behav_df: DataFrame
            
    params_sim : torch tensor, shape [num_params, num_agents]
        Contains the parameter values with which the simulations were performed.

    params_sim_df : DataFrame
        Contains the parameter values with which the simulations were performed.

    '''
    
    
    start = time.time()
    print("Simulating %d agents."%num_agents)

    "Only group is set"
    print('Inferring sequence and blockorder from group.')
    if torch.is_tensor(group):
        assert group.ndim == 1
        group = list(group)
    assert isinstance(group, list)
    sequences = [1, 1, 2, 2]
    blockorders = [1, 2, 1, 2]
    sequence = [sequences[g] for g in group]
    blockorder = [blockorders[g] for g in group]
        
    if group == None:
        "seuqence and blockorder are set."
        print('Inferring group from sequence and blockorder.')
        seq_torch = torch.tensor(sequence)
        blockorder_torch = torch.tensor(blockorder)
        group = (seq_torch == 1).type(torch.int) * (blockorder_torch == 1).type(torch.int)*0 +\
        (seq_torch == 1).type(torch.int) * (blockorder_torch == 2).type(torch.int)*1 +\
        (seq_torch == 2).type(torch.int) * (blockorder_torch == 1).type(torch.int)*2 +\
        (seq_torch == 2).type(torch.int) * (blockorder_torch == 2).type(torch.int)*3
        group = group.tolist()
        
        
    assert len(blockorder) == num_agents
    assert len(sequence) == num_agents
    assert torch.all(torch.tensor(sequence) > 0), "list must only contain 1 and 2."
    assert torch.all(torch.tensor(blockorder) > 0), "blockorder must only contain 1 and 2."
    
    if params is not None:
        if isinstance(params, pd.DataFrame):
            params = params.to_dict(orient='list')
            
        assert isinstance(params, dict)
    
    "----- Initialize agent"
    if params == None:
        newagent = init_agent(model, 
                              group, 
                              num_agents = num_agents)
    
        params_sim = {}
        for key in newagent.param_dict.keys():
            params_sim[key] = torch.squeeze(newagent.param_dict[key])
        # for param_idx in range(len(newagent.param_names)):
            # [param_idx, :] = newagent.par_dict[newagent.param_names[param_idx]]
    
    else:
        newagent = init_agent(model, 
                              group, 
                              num_agents = num_agents, 
                              params = params)
    
        params_sim = params

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
            'group': [group]*len(newenv.choices),
            'ag_idx' : [torch.arange(num_agents).tolist()]*len(newenv.choices),
            'model' : [[model]*num_agents]*len(newenv.choices)}
    
    for key in data:
        assert len(data[key])==6734
    
    group_behav_df = pd.DataFrame(data).explode(list(data.keys()))    
    
    "----- Create DataFrame containing the parameters."
    params_sim_df = pd.DataFrame(params_sim)
    
    params_sim_df['ag_idx'] = [i for i in range(num_agents)]
    params_sim_df['group'] = group
    params_sim_df['model'] = [model]*num_agents
    
    print("Simulation took %.4f seconds."%(time.time()-start))
    return data, group_behav_df, params_sim, params_sim_df

def plot_grouplevel(df1,
                    df2 = None,
                    plot_single = False,
                    plot_pairs = None, 
                    day = None):
    '''
    Parameters
    ----------
    df1 : DataFrame
        Contains each trial for all agents.
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
            
    plot_single : bool

    Returns
    -------
    None.

    '''
    
    if 'ID' in df1.columns:
        groupdata_df_1 = df1.drop(['ID'], axis = 1)
        
    else:
        groupdata_df_1 = df1
        
    if df2 is not None and 'ID' in df2:
        groupdata_df_2 = df2.drop(['ID'], axis = 1)
        
    elif df2 is not None and 'ID' not in df2:
        groupdata_df_2 = df2
        
    plt.style.use("ggplot")
    
    if day == 1:
        groupdata_df_1 = groupdata_df_1[groupdata_df_1['blockidx'] <= 5]

    elif day == 2:
        groupdata_df_1 = groupdata_df_1[groupdata_df_1['blockidx'] > 5]
    
    "----- Model 1"
    model_1 = groupdata_df_1['model'].unique()[0]
    
    "----- Remove newblock trials (where choices == -1)"
    groupdata_df_1 = groupdata_df_1[groupdata_df_1['choices'] != -1]
    
    "----- Remove STT (where jokertypes == -1)"
    groupdata_df_1 = groupdata_df_1[groupdata_df_1['jokertypes'] != -1]
    
    "----- Remove errortrials (where choices_GD == -2)"
    groupdata_df_1 = groupdata_df_1[groupdata_df_1['choices_GD'] != -2]
    
    "---------- Create new column block_num for different blockorders"
    blocknums_blockorder2 = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]
    groupdata_df_1['block_num'] = groupdata_df_1.apply(lambda row: \
                                         blocknums_blockorder2[row['blockidx']] if row['group']==1 or row['group']==3 \
                                         else row['blockidx'], axis=1)
        
    groupdata_df_1 = groupdata_df_1.drop(['blockidx', 'trialsequence', 'outcomes', 'choices'], axis = 1)
    
    if df2 is not None:
        if day == 1:
            groupdata_df_2 = groupdata_df_2[groupdata_df_2['blockidx'] <= 5]
    
        elif day == 2:
            groupdata_df_2 = groupdata_df_2[groupdata_df_2['blockidx'] > 5]
        
        "----- Model 2"
        model_2 = groupdata_df_2['model'].unique()[0]
        
        "----- Remove newblock trials (where choices == -1)"
        groupdata_df_2 = groupdata_df_2[groupdata_df_2['choices'] != -1]
        
        "----- Remove STT (where jokertypes == -1)"
        groupdata_df_2 = groupdata_df_2[groupdata_df_2['jokertypes'] != -1]
        
        "----- Remove errortrials (where choices_GD == -2)"
        groupdata_df_2 = groupdata_df_2[groupdata_df_2['choices_GD'] != -2]
        
        "---------- Create new column block_num for different blockorders"
        blocknums_blockorder2 = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]
        groupdata_df_2['block_num'] = groupdata_df_2.apply(lambda row: \
                                             blocknums_blockorder2[row['blockidx']] if row['group']==1 or row['group']==3 \
                                             else row['blockidx'], axis=1)
            
        groupdata_df_2 = groupdata_df_2.drop(['blockidx', 'trialsequence', 'outcomes', 'choices'], axis = 1)
        
    custom_palette = ['r', 'g', 'b'] # random, congruent, incongruent
    if plot_single:
        for ag_idx in np.sort(groupdata_df_1['ag_idx'].unique()):
            agent_df_1 = groupdata_df_1[groupdata_df_1['ag_idx'] == ag_idx]
            agent_df_1['jokertypes'] = agent_df_1['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'no joker')))
            
            if groupdata_df_2 is not None:
                agent_df_2 = groupdata_df_2[groupdata_df_2['ag_idx'] == ag_idx]
                agent_df_2['jokertypes'] = agent_df_2['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'no joker')))
            
            if groupdata_df_2 is not None:
                
                # pass
                # exp_grouped_df = create_grouped(exp_data[exp_data['ag_idx']==ag_idx], ag_idx)
                plot_dual_behav(agent_df_1, agent_df_2)
            
            else:
                fig, ax = plt.subplots()
                sns.lineplot(x='block_num',
                            y= 'choices_GD',
                            hue='jokertypes',
                            ax = ax,
                            data = agent_df_1,
                            palette = custom_palette)
                plt.title(f'Agent {ag_idx}, model {model_1}')
                ax.get_legend().remove()
                plt.show()
    
    
    if plot_pairs is not None:
        num_pairs = plot_pairs.shape[0]
        
        for pair in range(num_pairs):
            agent_df_1 = groupdata_df_1[groupdata_df_1['ag_idx'] == plot_pairs[pair, 0]]
            
            if df2 is not None:
                agent_df_2 = groupdata_df_2[groupdata_df_2['ag_idx'] == plot_pairs[pair, 1]]
            
            else:
                agent_df_2 = groupdata_df_1[groupdata_df_1['ag_idx'] == plot_pairs[pair, 1]]
            
            agent_df_1['jokertypes'] = agent_df_1['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'no joker')))
            agent_df_2['jokertypes'] = agent_df_2['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'no joker')))
            
            plot_dual_behav(agent_df_1, agent_df_2)
            
    "----- Plot grouplevel"
    groupdata_df_1 = groupdata_df_1.drop(['model'], axis = 1)
    grouped_df_1 = pd.DataFrame(groupdata_df_1.groupby(['ag_idx','block_num', 'jokertypes'], as_index = False).mean())
    grouped_df_1['jokertypes'] = grouped_df_1['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'no joker')))
    
    if df2 is not None:
        groupdata_df_2 = groupdata_df_2.drop(['model'], axis = 1)
        grouped_df_2 = pd.DataFrame(groupdata_df_2.groupby(['ag_idx','block_num', 'jokertypes'], as_index = False).mean())
        grouped_df_2['jokertypes'] = grouped_df_2['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'no joker')))
    
    if df2 is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        sns.lineplot(x = 'block_num', 
                    y = 'choices_GD', 
                    hue = 'jokertypes', 
                    data = grouped_df_1,
                    ax = ax1)
        ax1.set_xticks(np.arange(15), minor = True)
        ax1.grid(which='minor', alpha=0.5)
        ax1.set_title(f'Model {model_1}')
        ax1.get_legend().remove()
        
        sns.lineplot(x = 'block_num', 
                    y = 'choices_GD', 
                    hue ='jokertypes', 
                    data = grouped_df_2,
                    ax = ax2)
        ax2.set_xticks(np.arange(15), minor = True)
        ax2.grid(which='minor', alpha=0.5)
        ax2.set_title(f'Model {model_2}')
        ax2.get_legend().remove()
        plt.show()      

    else:
        "----- Remove error trials (where choices_GD == -2"
        fig, ax = plt.subplots()
        sns.lineplot(x = "block_num",
                    y = "choices_GD",
                    hue = "jokertypes",
                    data = grouped_df_1,
                    palette = custom_palette,
                    ax = ax)
        plt.title(f'Group Behaviour for model {model_1}')
        plt.show()
        
def plot_dual_behav(agent_df_1, agent_df_2):
    '''
    For plotting the behaviour of 2 agents side by side.

    Parameters
    ----------
    agent_df_1 : TYPE
        DESCRIPTION.
    agent_df_2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    assert len(agent_df_1['ag_idx'].unique()) == 1
    assert len(agent_df_2['ag_idx'].unique()) == 1
    
    ag_idx_1 = agent_df_1['ag_idx'].unique()[0]
    ag_idx_2 = agent_df_2['ag_idx'].unique()[0]
    
    model_1 = agent_df_1['model'].unique()
    model_2 = agent_df_2['model'].unique()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    sns.lineplot(x='block_num', 
                y= 'choices_GD', 
                hue='jokertypes', 
                # kind='line', 
                data = agent_df_1,
                ax = ax1)
    ax1.set_xticks(np.arange(15), minor = True)
    ax1.grid(which='minor', alpha=0.5)
    ax1.set_title(f'{model_1}, agent {ag_idx_1}')
    ax1.get_legend().remove()
    
    sns.lineplot(x='block_num', 
                y= 'choices_GD', 
                hue='jokertypes', 
                # kind='line', 
                data = agent_df_2,
                ax = ax2)
    ax2.set_xticks(np.arange(15), minor = True)
    ax2.grid(which='minor', alpha=0.5)
    ax2.set_title(f'{model_2}, agent {ag_idx_2}')
    ax2.get_legend().remove()
    plt.show()        

def posterior_predictives(post_sample, 
                          exp_data = None,
                          plot_single = True):
    '''
    
    Parameters
    ----------
    post_sample : DataFrame
        Contains posterior samples from inferred posterior.
        columns
            parameter columns
            ag_idx
            group
            
    exp_data : DataFrame
        columns
            parameter columns
            ag_idx
            group
            
    plot_single : bool, optional
        Whether to plot behaviour of single agents. Default is True.

    Returns
    -------
    complete_df : DataFrame
        DataFrame with average simulated behaviour for each agent.

    '''
    
    num_agents = len(post_sample['ag_idx'].unique())
    num_reps = len(post_sample[post_sample['ag_idx'] == 0])
    model = post_sample['model'].unique()[0]
    assert num_reps <= 1000, "Number of repetitions must be less than 1000."
    
    complete_df = pd.DataFrame()
    
    import seaborn
    custom_palette = ['r', 'g', 'b'] # random, congruent, incongruent
    
    for ag_idx in range(num_agents):
        print("Simulating agent %d of %d with %d repetitions."%(ag_idx+1, num_agents, num_reps))
        post_sample_agent_df = post_sample[post_sample['ag_idx'] == ag_idx]
        model = post_sample_agent_df['model'].unique()[0]
        
        
        # params = torch.tensor(post_sample_agent_df.iloc[:, 0:-3].to_numpy().T)
        # dfgh #test whether correct
        params = post_sample_agent_df.iloc[:,0:-1].to_dict(orient='list')
        for key in params.keys():
            params[key] = torch.tensor(params[key])
        
        ag_data_dict, ag_data_df, params, params_df = simulate_data(model, 
                                                                    num_reps,
                                                                    group = params['group'],
                                                                    params = params)
        
        agent_df = pd.DataFrame(ag_data_dict).explode(list(ag_data_dict.keys()))
        
        grouped_df = create_grouped(agent_df, ag_idx)
        
        if plot_single:
            if exp_data is not None:
                exp_grouped_df = create_grouped(exp_data[exp_data['ag_idx']==ag_idx], ag_idx)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
                sns.lineplot(x='block_num', 
                            y= 'choices_GD', 
                            hue='jokertypes', 
                            # kind='line', 
                            data = exp_grouped_df,
                            palette = custom_palette,
                            ax = ax1)
                ax1.set_title('Experimental Data')
                
                sns.lineplot(x='block_num', 
                            y= 'choices_GD', 
                            hue='jokertypes', 
                            # kind='line', 
                            data = grouped_df,
                            palette = custom_palette,
                            ax = ax2)
                ax2.set_title(f'Post Pred (model {model})')
                ax1.get_legend().remove()
                ax2.get_legend().remove()
                plt.show()        
            
            else:
                fig, ax = plt.subplots()
                sns.lineplot(x='block_num', 
                            y= 'choices_GD', 
                            hue='jokertypes', 
                            # kind='line', 
                            data = grouped_df,
                            palette = custom_palette,
                            ax = ax)
                plt.title(f'Post Pred (agent {ag_idx}, model {model})')
                ax.get_legend().remove()
                plt.show()
        
        complete_df = pd.concat((complete_df, grouped_df))
    
    if exp_data is not None:
        "Remove new block trials"
        exp_data = exp_data[exp_data['choices'] != -1]
        "Remove error trials"
        exp_data = exp_data[exp_data['choices'] != -2]
        "Remove STT"
        exp_data = exp_data[exp_data['jokertypes'] != -1]
        
        "---------- Create new column block_num for different blockorders"
        blocknums_blockorder2 = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]
        exp_data['block_num'] = exp_data.apply(lambda row: \
                                             blocknums_blockorder2[row['blockidx']] if row['group']==1 or row['group']==3 \
                                             else row['blockidx'], axis=1)
        model_exp = exp_data['model'].unique()[0]
        exp_data = exp_data.drop(['group', 'blockidx', 'trialsequence', 'outcomes', 'choices', 'model'], axis = 1)
        exp_data['jokertypes'] = exp_data['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'no joker')))
        
        if 'ID' in exp_data.columns:
            exp_data_grouped = pd.DataFrame(exp_data.groupby(['ag_idx', 'block_num','jokertypes', 'ID'], as_index = False).mean())
        else:
            exp_data_grouped = pd.DataFrame(exp_data.groupby(['ag_idx', 'block_num','jokertypes'], as_index = False).mean())
        # exp_data_grouped_all = pd.DataFrame(exp_data_grouped.groupby(['block_num','jokertypes'], as_index = False).mean())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        
        sns.lineplot(x = 'block_num', 
                    y = 'choices_GD', 
                    hue = 'jokertypes', 
                    # kind='line', 
                    data = exp_data_grouped,
                    palette = custom_palette,
                    ax = ax1)
        ax1.set_title(f'Data model {model_exp}')
        
        sns.lineplot(x = 'block_num', 
                    y = 'choices_GD', 
                    hue ='jokertypes', 
                    # kind='line', 
                    data = complete_df,
                    palette = custom_palette,
                    ax = ax2)
        ax2.set_title(f'Post Pred (model {model})')
        
        plt.show()
    else:
        fig, ax = plt.subplots()
        sns.lineplot(x="block_num", 
                     y="choices_GD", 
                     hue = "jokertypes", 
                     data = complete_df, 
                     palette = custom_palette,
                     ax = ax)
        plt.title(f'Group-Level Post Pred (model {model})')
        plt.show()
    print("Finished posterior predictives.")
    return complete_df

def create_grouped(df, ag_idx):
    '''
    Creates grouped DataFrame for a single agent.
    Groups by block_num and jokertypes across whole dataframe.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with data of a single agent.
        Must contain columns:
            choices
            choices_GD
            jokertypes : int
            blockidx
            group
            ag_idx

    ag_idx : int
        idx for the single agent (not ag_idx from dataframe).

    Returns
    -------
    grouped_df : DataFrame
        DESCRIPTION.
    '''
    
    if 'ID' in df.columns:
        groupdata_df = df.drop(['ID'], axis = 1)
    
    else:
        groupdata_df = df
        
    "Remove new block trials"
    groupdata_df = groupdata_df[groupdata_df['choices'] != -1]
    "Remove error trials"
    groupdata_df = groupdata_df[groupdata_df['choices'] != -2]
    "Remove STT"
    groupdata_df = groupdata_df[groupdata_df['jokertypes'] != -1]
    
    "---------- Create new column block_num for different blockorders"
    blocknums_blockorder2 = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]
    groupdata_df['block_num'] = groupdata_df.apply(lambda row: \
                                         blocknums_blockorder2[row['blockidx']] if row['group']==1 or row['group']==3 \
                                         else row['blockidx'], axis=1)
    
    groupdata_df.drop(['group', 'blockidx', 'ag_idx', 'trialsequence', 'outcomes', 'choices', 'model'], axis = 1, inplace = True)
    groupdata_df['jokertypes']=groupdata_df['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'no joker')))
    grouped_df = pd.DataFrame(groupdata_df.groupby(['block_num','jokertypes'], as_index = False).mean())
    grouped_df['ag_idx'] = ag_idx
    
    return grouped_df

def get_data_from_file():
    import pickle
    import tkinter as tk
    from tkinter import filedialog
    
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
    post_sample_df, expdata_df, loss, params_df = pickle.load(open( filenames[0], "rb" ))
    
    if 'ag_idx' not in params_df.columns:
        params_df['ag_idx'] = None
        
    if 'group' not in params_df.columns:
        params_df['group'] = None

    if 'model' not in params_df.columns:
        params_df['model'] = post_sample_df['model'][0]
    
    if 'ID' in params_df.columns:
        params_df_temp = params_df.drop(['model', 'ag_idx', 'ID', 'group'], axis = 1)
        
    else:
        params_df_temp = params_df.drop(['model', 'ag_idx', 'group'], axis = 1)
        
    num_params = len(params_df_temp.columns)
    del params_df_temp
    
    return post_sample_df, expdata_df, loss, params_df, num_params