#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 12:52:24 2023

@author: sascha
"""

import time
import ipdb
import torch
import numpy as np
import scipy
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
import analysis_tools as anal

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

def get_groupdata(data_dir, getall = False):
    '''

    Parameters
    ----------
    data_dir : str
        Directory with data.
        
    getall : bool
        True: Get all data collected so far
        False: Make sure to get the same number of participants in every group

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
            jokertypes : -1/0/1/2 : no joker/random/congruent/incongruent
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
    
    include_IDs = ['5d7ebf9e93902b0001965912', '5b266738007d870001c7c360',
           '6419cedec2078147e5682474', '5d55b7ef6a0f930017202336',
           '6286672d0165aad8f1386c27', '55cca8f81676ab000ff06ef1',
           '5908458b1138880001bc77e7', '5efb31fa8cd32f04bf048643',
           '62b44f66a16d45783569fad6', '5b5e0e86902ad10001cfcc59',
           '629f6b8c65fcae219e245284', '5f356cbffb4cea5170d04fd9',
           '63af557b3d4f219c3226b7d6', '5eff5f05b92981000a2aed73',
           '5c6e8dd877955b0001ff0c58', '5d8b66f5d189bd001a378273',
           '62db2644ab0a3a353c0dcb54', '6329b1add3dcd53cb9c9cab8',
           '5c321ebf6558270001bd79aa', '5d49d17b3dad1f0001e2aba1',
           '5eadaff848b26f4483ae62d9', '60a3f8075b013de7b5518e96',
           '5eec9ee7d900510326d78fc8', '5d8a29c082fec30001d9c24a',
           '617406fbfced12169896d158', '5e66c77e8ebdaf4466e26326',
           '57deda2591b7fc0001493e95', '5982eef79dfc3e00011d81e0',
           '57dd186e6598aa0001992616', '595e7974af78da0001a21c3a',
           '5a9ed5046475f90001a0189e', '615739949cf5767509a7e29a',
           '5f16fde210d37701904c9dc2', '5dc5da21d999de45a504651b',
           '5d0245966e208b0017301561', '63d79fcd741f9cfb2f87152f',
           '59dd90f6e75b450001a68dac', '63174af7d57182f9bf90c094',
           '5eaadc0a7adeb404eea9c3c0', '62c97799bd8ab72a531abde0',
           '60f816ff1fa74fcfab532378', '6500615b226d81ec5db464d7',
           '5c4b987538878c0001c7883b', '5b2a72f7c293b90001732b89',
           '57d5ab3a722df500017f3622', '5db4ef4a2986a3000be1f886',
           '57c4761195a3ea00016e5992', '5db32244dbe39d000be72fb0',
           '5e850b0e390e520ec806b084', '6116b022b7ef87ef5828748b',
           '596f961cfe061d00011e3e03', '6044ca22bc6235555362d5bb',
           '58aca85e0da7f10001de92d4', '65389f0b0f181197c4218f6d',
           '654abe303c4940ec0502538e', '60e2577f8c52db9d1fb5ffac',
           '62e02b26e879244a99e852fa', '56f699e876348f000c883bba',
           '5fb46dd5d9ece50422838e7a', '5d5a75c570a7c1000152623e']
    
    exclude_errors = [
                  '5e07c976f6191f137214e91f' # (Grp 1)
                  ]
    
    'Exclude because of execution time'
    exclude_time = [# Grp 0
                  '604fa685e33606f9a0ee8189',
                  # Grp 1
                '63ef938aea545fb9bd19a591', '56d8df02d13f6b000e542eef',
                '601f08cba1191642cadf59c1', '6151f20d06757e43aa1f54f9', '5f48f497fae2763d537d2e6b',
                '63e5eb956eab1f2740ac6289',
                # Grp 2
                '62cbed33d29c28e6be511bde', '5a54d42476d1c60001aacd6c', 
                '59e7232f24d7bf00012f112e',
                # Grp 3
                '5eebe3d7b1914c17e6208284', '6329c510ea44255e948f8492',
                '5ea00d4b1286ee0008405450']
    
    exclude_random = []
    exclude_random.extend(list(np.random.choice(range(19), size = 2, replace = False))) # Grp 0
    # exclude_random.extend(list(np.random.choice(range(19, 41), size = 1, replace = False))) # Grp 1
    exclude_random.extend(list(np.random.choice(range(41, 61), size = 2, replace = False))) # Grp 2
    
    exclude_random.sort()
    print(exclude_random)
    
    IDs_included = []
    
    'right/left/ambidextrous: 0/1/2'
    hand = []
    'male/female : 0/1'
    gender = []
    age = []
    
    q_sometimes_easier = []
    q_notice_a_sequence = []
    q_sequence_repro = []
    q_sequence_repro_with_help = []
    
    sociopsy_df = pd.read_csv(data_dir + 'sociopsy_data.csv')
    sociopsy_df.loc[sociopsy_df['handedness'] == 'righthanded', 'handedness'] = 0
    sociopsy_df.loc[sociopsy_df['handedness'] == 'lefthanded', 'handedness'] = 1
    sociopsy_df.loc[sociopsy_df['handedness'] == 'ambidextrous', 'handedness'] = 2
    
    sociopsy_df.loc[sociopsy_df['handedness'] == 'female', 'handedness'] = 0
    sociopsy_df.loc[sociopsy_df['handedness'] == 'male', 'handedness'] = 1
    sociopsy_df.loc[sociopsy_df['handedness'] == 'other', 'handedness'] = 2
    
    pb = -1
    for grp in range(4):
        files_day1 = glob.glob(data_dir + "Grp%d/csv/*Tag1*.mat"%(grp+1))
        
        for file1 in files_day1:
            "Loop over participants"
            pb += 1
            print(f"Doing pb {pb}")
            data, ID = get_participant_data(file1, 
                                            grp, 
                                            data_dir)
            
            file_day2 = glob.glob(data_dir + "Grp%d/csv/*%s*Tag2*.mat"%(grp+1, ID))[0]
            participant_day2 = scipy.io.loadmat(file_day2)
            
            # if ID not in handedness.keys():
            #     raise Exception('handedness missing for ID %s'%ID)
            
            if getall:
                groupdata.append(data)
                group.append(grp)
                IDs_included.append(ID)
                hand.append(sociopsy_df[sociopsy_df['ID'] == ID].iloc[0]['handedness'])
                gender.append(sociopsy_df[sociopsy_df['ID'] == ID].iloc[0]['gender'])
                age.append(sociopsy_df[sociopsy_df['ID'] == ID].iloc[0]['age'])
                
                q_sometimes_easier.append(participant_day2['q1'][0,1][0])
                q_notice_a_sequence.append(participant_day2['q2'][0,1][0])
                
            else:
                # if ID not in exclude_time and ID not in exclude_errors and pb not in exclude_random:
                if ID in include_IDs:
                    groupdata.append(data)
                    group.append(grp)
                    IDs_included.append(ID)
                    hand.append(sociopsy_df[sociopsy_df['ID'] == ID].iloc[0]['handedness'])
                    gender.append(sociopsy_df[sociopsy_df['ID'] == ID].iloc[0]['gender'])
                    age.append(sociopsy_df[sociopsy_df['ID'] == ID].iloc[0]['age'])
                    
                    q_sometimes_easier.append(participant_day2['q1'][0,1][0])
                    q_notice_a_sequence.append(participant_day2['q2'][0,1][0])
                    # q_sequence_repro.append(participant_day2['q3'][0,1][0])
                    # q_sequence_repro_with_help.append(participant_day2['q4'][0,1][0])
    
    q_sometimes_easier = [1 if q=='Yes' else (0 if q == 'No' else 2) for q in q_sometimes_easier]
    q_notice_a_sequence  = [1 if q=='Yes' else (0 if q == 'No' else 2) for q in q_notice_a_sequence]
    
    newgroupdata = comp_groupdata(groupdata)
    num_trials = len(newgroupdata['trialsequence'])
    num_agents = len(newgroupdata['trialsequence'][0])
    newgroupdata['group'] = [group]*num_trials
    newgroupdata['handedness'] = [hand]*num_trials
    newgroupdata['age'] = [age]*num_trials
    newgroupdata['gender'] = [gender]*num_trials
    
    newgroupdata['q_sometimes_easier'] = [q_sometimes_easier]*num_trials
    newgroupdata['q_notice_a_sequence'] = [q_notice_a_sequence]*num_trials
    # newgroupdata['q_sequence_repro'] = [q_sequence_repro]*num_trials
    # newgroupdata['q_sequence_repro_with_help'] = [q_sequence_repro_with_help]*num_trials
    
    newgroupdata['ID'] = [IDs_included]*num_trials
    newgroupdata['ag_idx'] = [torch.arange(num_agents).tolist()]*num_trials
    newgroupdata['model'] = [['Experiment']*num_agents]*num_trials
    groupdata_df = pd.DataFrame(newgroupdata).explode(list(newgroupdata.keys()))
    
    dfnew = pd.DataFrame(groupdata_df.loc[:, ['ID', 'group']].groupby(['ID'], as_index = False).mean())
    
    group_distro = [len(dfnew[dfnew['group']== grp]) for grp in range(4)]
    print(group_distro)
    if not getall:
        assert np.abs(np.diff(group_distro)).sum() == 0
    
    return newgroupdata, groupdata_df


def get_old_groupdata(data_dir, getall = False, oldpub = True):
    '''

    Parameters
    ----------
    data_dir : str
        Directory with data.
        
    getall : bool
        True: Get all data collected so far
        False: Make sure to get the same number of participants in every group

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
            jokertypes : -1/0/1/2 : no joker/random/congruent/incongruent
            blockidx : nested list, 'shape' [num_trials, num_agents]
            RT : nested list, 'shape' [num_trials, num_agents]
            group : list, len [num_trials, num_agents]. 0-indexed
            ag_idx
            ID
            
    groupdata_df : DataFrame

    '''
    
    groupdata = []
    group = []
    
    
    IDs_included = []
    
    'right/left/ambidextrous: 0/1/2'
    # hand = []
    'male/female : 0/1'
    # gender = []
    # age = []
    
    q_sometimes_easier = []
    q_notice_a_sequence = []
    q_sequence_repro = []
    q_sequence_repro_with_help = []
    
    pb = -1
    for grp in range(4):
        files_day1 = glob.glob(data_dir + "Grp%d/csv/*Tag1*.mat"%(grp+1))
        
        for file1 in files_day1:
            "Loop over participants"
            pb += 1
            print(f"Doing pb {pb}")
            data, ID = get_participant_data(file1, 
                                            grp, 
                                            data_dir,
                                            oldpub = oldpub)
            
            file_day2 = glob.glob(data_dir + "Grp%d/csv/*%s*Tag2*.mat"%(grp+1, ID))[0]
            participant_day2 = scipy.io.loadmat(file_day2)
            
            # if ID not in handedness.keys():
            #     raise Exception('handedness missing for ID %s'%ID)
            
            groupdata.append(data)
            group.append(grp)
            IDs_included.append(ID)
            
            q_sometimes_easier.append(participant_day2['q1'][0,1][0])
            q_notice_a_sequence.append(participant_day2['q2'][0,1][0])
                
    q_sometimes_easier = [1 if q == 'Yes' else (0 if q == 'No' else 2) for q in q_sometimes_easier]
    q_notice_a_sequence  = [1 if q == 'Yes' else (0 if q == 'No' else 2) for q in q_notice_a_sequence]
    
    newgroupdata = comp_groupdata(groupdata)
    num_trials = len(newgroupdata['trialsequence'])
    num_agents = len(newgroupdata['trialsequence'][0])
    newgroupdata['group'] = [group]*num_trials
    # newgroupdata['handedness'] = [hand]*num_trials
    # newgroupdata['age'] = [age]*num_trials
    # newgroupdata['gender'] = [gender]*num_trials
    
    newgroupdata['q_sometimes_easier'] = [q_sometimes_easier]*num_trials
    newgroupdata['q_notice_a_sequence'] = [q_notice_a_sequence]*num_trials
    # newgroupdata['q_sequence_repro'] = [q_sequence_repro]*num_trials
    # newgroupdata['q_sequence_repro_with_help'] = [q_sequence_repro_with_help]*num_trials
    
    newgroupdata['ID'] = [IDs_included]*num_trials
    newgroupdata['ag_idx'] = [torch.arange(num_agents).tolist()]*num_trials
    newgroupdata['model'] = [['Experiment']*num_agents]*num_trials
    groupdata_df = pd.DataFrame(newgroupdata).explode(list(newgroupdata.keys()))
    #
    dfnew = pd.DataFrame(groupdata_df.loc[:, ['ID', 'group']].groupby(['ID'], as_index = False).mean())
    
    group_distro = [len(dfnew[dfnew['group']== grp]) for grp in range(4)]
    print(group_distro)
    if not getall:
        assert np.abs(np.diff(group_distro)).sum() == 0
    
    return newgroupdata, groupdata_df


def get_participant_data(file_day1, group, data_dir, oldpub = False):
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
                DTT Types
                (clipre) -1/0/1/2 : no joker/random/congruent/incongruent
                (published) : -1/0/1/2/3/4 : no joker/random choice/congruent/incongruent/NLP/NHP
            blockidx
            RT : list of list of floats (RT in ms)
        
    ID : str
        Participant-specific ID.

    '''
    
    assert group < 4
    
    if oldpub:
        ID = file_day1.split("/")[-1][4:9] # NicDB
    else:
        # ID = file_day1.split("/")[-1][4:28] # Prolific ID
        ID = file_day1.split("/")[-1][4:10] # PBIDXX


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
    if oldpub:
        block_order_day1 = [[0,1,2,3,4,5], [0,1,2,3,4,5], [0,1,2,3,4,5], [0,1,2,3,4,5]]
        
    else:
        block_order_day1 = [[0,1,2,3,4,5], [1,0,3,2,5,4], [0,1,2,3,4,5], [1,0,3,2,5,4]]
    
    for i in block_order_day1[group]:
        "Mark beginning of new block with -1"
        choices.append(-1)
        outcomes.append(-2) # will transform this to -1 further down below
        correct.append(-1)
        RT.append(-1)
        
        correct.extend(np.squeeze(participant_day1["correct_all_cell"][0][i]).tolist())
        choices.extend(np.squeeze(participant_day1["resps_response_digit_cell"][0][i]).tolist()) # Still 1-indexed
        if 'rew_cell' in participant_day1.keys():
            outcomes.extend(np.squeeze(participant_day1["rew_cell"][0][i]).tolist())
        RT.extend(np.squeeze(participant_day1["RT_cell"][0][i]).tolist())

    if oldpub:
        block_order_day2 = [[0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7]]
        
    else:
        block_order_day2 = [[0,1,2,3,4,5,6,7], [1,0,3,2,5,4,7,6], [0,1,2,3,4,5,6,7], [1,0,3,2,5,4,7,6]]
        
    for i in block_order_day2[group]:
        "Mark beginning of new block with -1"
        choices.append(-1)
        outcomes.append(-2) # will transform this to -1 further down below
        correct.append(-1)
        RT.append(-1)
        
        correct.extend(np.squeeze(participant_day2["correct_all_cell"][0][i]).tolist())
        choices.extend(np.squeeze(participant_day2["resps_response_digit_cell"][0][i]).tolist()) # Still 1-indexed
        if 'rew_cell' in participant_day2.keys():
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
    
    if 'rew_cell' in participant_day2.keys():
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
        
        if oldpub:
            seq, seq_wo_jokers, jtypes, btype = get_trialseq('./matlabcode/published/',
                                                     group, 
                                                     block)

        else:
            seq, seq_wo_jokers, jtypes, btype = get_trialseq('./matlabcode/clipre/',
                                                     group, 
                                                     block)
        
        trialsequence.extend(seq)
        trialsequence_wo_jokers.extend(seq_wo_jokers)
        jokertypes.extend(jtypes)
        blocktype.extend([btype]*len(seq))
        blockidx.extend([block]*len(seq))
    
    assert len(trialsequence) == len(choices)
    
    if 'rew_cell' in participant_day2.keys():
        assert len(outcomes) == len(choices)
        assert len(outcomes) == len(blocktype)
    
    num_trials = len(choices)
    
    assert  len(trialsequence) == num_trials and \
            len(trialsequence_wo_jokers) == num_trials and \
            len(jokertypes) == num_trials and \
            len(blocktype) == num_trials and \
            len(blockidx) == num_trials and \
            len(RT) == num_trials
    
    if 'rew_cell' in participant_day2.keys():
        len(outcomes) == num_trials
        
    jokertypes = [[jokertypes[i]] for i in range(num_trials)]
    trialsequence = [[trialsequence[i]] for i in range(num_trials)]
    trialsequence_wo_jokers = [[trialsequence_wo_jokers[i]] for i in range(num_trials)]
    choices = [[choices[i]] for i in range(num_trials)]
    correct = [[correct[i]] for i in range(num_trials)]
    if 'rew_cell' in participant_day2.keys():
        outcomes = [[outcomes[i]] for i in range(num_trials)]
    blocktype = [[blocktype[i]] for i in range(num_trials)]
    blockidx = [[blockidx[i]] for i in range(num_trials)]
    RT = [[RT[i]] for i in range(num_trials)]

    choices_torch = torch.squeeze(torch.tensor(choices))

    if oldpub:
        "For old published data."
        if group == 0 or group == 3:
            "High Reward prob is top left and bottom right"
            choices_GD = torch.logical_or(choices_torch == 0, choices_torch == 3) * 1 +\
                        (choices_torch == -2) * -2 + (choices_torch == -1) * -1
            
        elif group == 1 or group == 2:
            "High Reward prob is top right and bottom left"
            choices_GD = torch.logical_or(choices_torch == 1, choices_torch == 2) * 1 +\
                        (choices_torch == -2) * -2 + (choices_torch == -1) * -1
            
    else:
        "For new fresh, and crispy data."
        if group == 0 or group == 1:
            "High Reward prob is top left and bottom right"
            choices_GD = torch.logical_or(choices_torch == 0, choices_torch == 3) * 1 +\
                        (choices_torch == -2) * -2 + (choices_torch == -1) * -1
            
            
        elif group == 2 or group == 3:
            "High Reward prob is top right and bottom left"
            choices_GD = torch.logical_or(choices_torch == 1, choices_torch == 2) * 1 +\
                        (choices_torch == -2) * -2 + (choices_torch == -1) * -1
            
    
    "Set -1 (new block trial) where appropriate"
    # choices_GD = torch.where(choices_torch == -1, -1*torch.ones(choices_torch.shape), choices_GD)
    assert torch.all(choices_GD <= 1)
    choices_GD = [[cgd.item()] for cgd in choices_GD]
    
    "Make sure that all answers at NLP are either error, new block trial, or not GD"
    nlpresps = torch.squeeze(torch.tensor(choices_GD))[torch.squeeze(torch.tensor(jokertypes)) == 3]
    nhpresps = torch.squeeze(torch.tensor(choices_GD))[torch.squeeze(torch.tensor(jokertypes)) == 4]
    assert torch.all(torch.tensor([resp in [-2, -1, 0] for resp in nlpresps]))
    assert torch.all(torch.tensor([resp in [-2, -1, 1] for resp in nhpresps]))
    
    assert len(correct) == len(choices)
    
    if 'rew_cell' in participant_day2.keys():
        data = {'trialsequence': trialsequence,
                'trialsequence_no_jokers': trialsequence_wo_jokers,
                'jokertypes' : jokertypes,
                'correct' : correct,
                'choices': choices,
                'choices_GD' : choices_GD,
                'outcomes': outcomes,
                'blocktype': blocktype,
                'blockidx': blockidx,
                'RT': RT}
        
    else:
        "Because of old published data"
        data = {'trialsequence': trialsequence,
                'trialsequence_no_jokers': trialsequence_wo_jokers,
                'jokertypes' : jokertypes,
                'choices': choices,
                'correct': correct,
                'choices_GD' : choices_GD,
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
        (clipre) -1/0/1/2 : no joker/random/congruent/incongruent
        (published) : -1/0/1/2/3/4 : no joker/random choice/congruent/incongruent/NLP/NHP

    blocktype : int
        0/1 : sequential/ random block

    '''
    
    "NB: in mat-files, the block order was already swapped, as if all participants saw the first group's block order! Have to correct for this!"
    
    'This is the blockorder participants actually saw in the 4 groups.'
    
    if 'published' in matfile_dir:
        blockorder = [["random1", "trainblock1", 
                       "random2", "trainblock2", 
                       "random3", "trainblock3", 
                       "trainblock4", "random4", 
                       "trainblock5", "random5", 
                       "trainblock6", "random6", 
                       "trainblock7", "random7"],
                      
                      ["mirror_random1", "mirror_trainblock1", 
                       "mirror_random2", "mirror_trainblock2", 
                       "mirror_random3", "mirror_trainblock3", 
                       "mirror_trainblock4", "mirror_random4", 
                       "mirror_trainblock5", "mirror_random5", 
                       "mirror_trainblock6", "mirror_random6", 
                       "mirror_trainblock7", "mirror_random7"],
                      
                      ["mirror_random1", "mirror_trainblock1", 
                       "mirror_random2", "mirror_trainblock2", 
                       "mirror_random3", "mirror_trainblock3", 
                       "mirror_trainblock4", "mirror_random4", 
                       "mirror_trainblock5", "mirror_random5", 
                       "mirror_trainblock6", "mirror_random6", 
                       "mirror_trainblock7", "mirror_random7"],
                      
                    ["random1", "trainblock1", 
                     "random2", "trainblock2", 
                     "random3", "trainblock3", 
                     "trainblock4", "random4", 
                     "trainblock5", "random5", 
                     "trainblock6", "random6", 
                     "trainblock7", "random7"]]    
        
        '1 = random, 0 = seq'
        types = [[1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                 [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                 [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                 [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]]

        seqs = [1, 2, 2, 1]
        
    elif 'clipre' in matfile_dir:
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

        '1 = random, 0 = seq'
        types = [[1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                 [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                  [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]]
        
        seqs = [1, 1, 2, 2] # 2 = mirror sequence (has different reward contingency)

    sequence = seqs[group]
    blocktype = types[group][blockidx]

    mat = scipy.io.loadmat(matfile_dir + "%s.mat"%blockorder[group][blockidx])

    seq = mat['sequence'][0]
    seq_no_jokers = mat['sequence_without_jokers'][0]


    "----- Map Neutral Jokers to 'No Joker' (only necessary for published results)."
    # seq_noneutral = [t if t != 14 and t != 23 else 1 for t in seq]
    # seq_withneutral = seq.copy()

    "----- Determine congruent/ incongruent jokers"
    if 'published' in matfile_dir:
        if blocktype == 0:
            "Sequential"
            "Sequential element where jokers are"
            jokers_noneut_seqelem = [-1 if seq[tidx] not in [12, 13, 24, 34] else seq_no_jokers[tidx] for tidx in range(len(seq))]
            
            if sequence == 1:
                "(published) : -1/0/1/2/3/4 : no joker/random choice/congruent/incongruent/NLP/NHP"
                num_nlp = (np.array(seq) == 23).sum()
                num_nhp = (np.array(seq) == 14).sum()
                "groups 0 and 3"
                "-1 > -1, 1 > 1, 2 > 2, 3 > 2, 4 > 1"
                jokertypes = np.array([j if j == -1 else 1 if j == 1 else 2 if j == 2 else 2 if j == 3 else 1 if j == 4 else 0 for j in jokers_noneut_seqelem])
                jokertypes += ((np.array(seq) == 14) * 5 + (np.array(seq) == 23) * 4)
                
                assert num_nlp == ((np.array(jokertypes) == 3).sum())
                assert num_nhp == ((np.array(jokertypes) == 4).sum())
                
            elif sequence == 2:
                num_nlp = (np.array(seq) == 14).sum()
                num_nhp = (np.array(seq) == 23).sum()
                "groups 1 and 2"
                "-1 > -1, 1 > 2, 2 > 1, 3 > 1, 4 > 2"
                jokertypes = np.array([j if j == -1 else 2 if j == 1 else 1 if j == 2 else 1 if j == 3 else 2 if j == 4 else 0 for j in jokers_noneut_seqelem])
                jokertypes += ((np.array(seq) == 14) * 4 + (np.array(seq) == 23) * 5)
                
                assert num_nlp == ((np.array(jokertypes) == 3).sum())
                assert num_nhp == ((np.array(jokertypes) == 4).sum())
                
            else:
                raise Exception("Fehla!!")

        elif blocktype == 1:
            "Random"
            # jokertypes = [-1 if seq_noneutral[tidx]<10 else 0 for tidx in range(len(seq_noneutral))]

            if group == 0 or group == 3:
                jokertypes = (np.array(seq) == 14) * 4 +\
                            (np.array(seq) == 23) * 3 -\
                            (np.array(seq) < 10)

            elif group == 1 or group == 2:
                jokertypes = (np.array(seq) == 14) * 3 +\
                            (np.array(seq) == 23) * 4 -\
                            (np.array(seq) < 10)
                
            else:
                raise Exception("Group must be 0,1,2, or 3.")
            
        jokertypes = jokertypes.tolist()
        # jokertypes = torch.where(torch.logical_or(seq_withneutral == 14, seq_withneutral == 23), )
          
    elif 'clipre' in matfile_dir:
        if blocktype == 0:
            "sequential"
            "Sequential element where jokers are"
            jokers_seqelem = [-1 if seq[tidx]<10 else seq_no_jokers[tidx] for tidx in range(len(seq))]
            
            if sequence == 1:
                "Groups 1 & 2: High Reward top left & bottom right"
                "-1 > -1, 1 > 1, 2 > 2, 3 > 2, else 1"
                jokertypes = [j if j == -1 else 1 if j == 1 else 2 if j == 2 else 2 if j == 3 else 1 for j in jokers_seqelem]
                
            elif sequence == 2:
                "Groups 1 & 2: High Reward top right & bottom left"
                "Groups 3 & 4"
                jokertypes = [j if j == -1 else 2 if j == 1 else 1 if j == 2 else 1 if j == 3 else 2 for j in jokers_seqelem]
                
            else:
                raise Exception("Fehla!!")
                     
        elif blocktype == 1:
            "random"
            # jokertypes = [-1 if seq[tidx]<10 else 0 for tidx in range(len(seq))]
            
            jokertypes = ((np.array(seq) > 10) -1).tolist()
    
    assert (np.array(seq) > 10).sum() == (np.array(jokertypes)>=0).sum()
    assert (np.array(jokertypes) > 4).sum() == 0
    
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
            correct

    Returns
    -------
    newgroupdata : dict
        Contains keys as in groupdata.
        len = num_trials
        Each element of a key is another list with length num_agents

    '''
    
    if 'outcomes' in groupdata[0].keys():
        newgroupdata = {'choices' : [],
                        'choices_GD' : [],
                        'outcomes' : [],
                        'trialsequence' : [],
                        'blocktype' : [],
                        'jokertypes' : [],
                        'blockidx' : [],
                        'correct': [],
                        'RT': []}


    else:
        "Because of old published data"
        newgroupdata = {'choices' : [],
                        'choices_GD' : [],
                        'trialsequence' : [],
                        'blocktype' : [],
                        'jokertypes' : [],
                        'blockidx' : [],
                        'correct': [],
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
        correct = []

        for dt in groupdata:
            choices.append(dt['choices'][trial][0])
            choices_GD.append(dt['choices_GD'][trial][0])
            if 'outcomes' in groupdata[0].keys():
                outcomes.append(dt['outcomes'][trial][0])
            trialsequence.append(dt['trialsequence'][trial][0])
            blocktype.append(dt['blocktype'][trial][0])
            jokertypes.append(dt['jokertypes'][trial][0])
            blockidx.append(dt['blockidx'][trial][0])
            RT.append(dt['RT'][trial][0])
            correct.append(dt['correct'][trial][0])
        
        newgroupdata["choices"].append(choices)
        newgroupdata["choices_GD"].append(choices_GD)
        if 'outcomes' in groupdata[0].keys():
            newgroupdata["outcomes"].append(outcomes)
        newgroupdata["trialsequence"].append(trialsequence)
        newgroupdata["blocktype"].append(blocktype)
        newgroupdata["jokertypes"].append(jokertypes)
        newgroupdata["blockidx"].append(blockidx)
        newgroupdata["RT"].append(RT)
        newgroupdata["correct"].append(correct)
    
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
    
    
    if params is not None:
        for key in params.keys():
            if isinstance(params[key], list):
                if not isinstance(params[key][0], str):
                    params[key] = torch.tensor(params[key])
            
    k = 4.
    import models_torch_paper as models
    if model == 'Vbm':
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
        
    elif model == 'Vbm_2days':
        num_params = models.Vbm_2days_nolr.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr'] = params_uniform[0:1, :]
            param_dict['omega_day1'] = params_uniform[1:2, :]
            param_dict['dectemp_day1'] = params_uniform[2:3, :]
            
            param_dict['omega_day2'] = params_uniform[3:4, :]
            param_dict['dectemp_day2'] = params_uniform[4:5, :]
            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr'] = params['lr'][None,...]
            param_dict['omega_day1'] = params['omega_day1'][None,...]
            param_dict['dectemp_day1'] = params['dectemp_day1'][None,...]
            param_dict['omega_day2'] = params['omega_day2'][None,...]
            param_dict['dectemp_day2'] = params['dectemp_day2'][None,...]
            
        newagent = models.Vbm_2days_nolr(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    elif model == 'Repbias':
        num_params = models.Repbias_2days_nolr.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr'] = params_uniform[0:1, :]
            param_dict['theta_Q_day1'] = params_uniform[1:2, :]*6
            param_dict['theta_rep_day1'] = params_uniform[2:3, :]*6
            
            param_dict['theta_Q_day2'] = params_uniform[3:4, :]*6
            param_dict['theta_rep_day2'] = params_uniform[4:5, :]*6
            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr'] = params['lr'][None,...]
            param_dict['theta_Q_day1'] = params['theta_Q_day1'][None,...]
            param_dict['theta_rep_day1'] = params['theta_rep_day1'][None,...]
            
            param_dict['theta_rep_day2'] = params['theta_rep_day2'][None,...]
            param_dict['theta_Q_day2'] = params['theta_Q_day2'][None,...]
            
        newagent = models.Repbias_2days_nolr(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    elif model == 'Repbias_1day':
        num_params = models.Repbias_1day.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr'] = params_uniform[0:1, :]
            param_dict['theta_Q'] = params_uniform[1:2, :]*6
            param_dict['theta_rep'] = params_uniform[2:3, :]*6

            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr'] = params['lr'][None,...]
            param_dict['theta_Q'] = params['theta_Q'][None,...]
            param_dict['theta_rep'] = params['theta_rep'][None,...]
            
            
        newagent = models.Repbias_1day(param_dict,
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    elif model == 'Seqboost':
        num_params = models.Seqboost_2days_nlor.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr'] = params_uniform[0:1, :]
            param_dict['theta_Q_day1'] = params_uniform[1:2, :]*6
            param_dict['theta_rep_day1'] = params_uniform[2:3, :]*6
            param_dict['seq_param_day1'] = params_uniform[3:4, :]*6
            
            param_dict['theta_Q_day2'] = params_uniform[4:5, :]*6
            param_dict['theta_rep_day2'] = params_uniform[5:6, :]*6
            param_dict['seq_param_day2'] = params_uniform[6:7, :]*6
            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr'] = params['lr'][None,...]
            param_dict['theta_Q_day1'] = params['theta_Q_day1'][None,...]
            param_dict['theta_rep_day1'] = params['theta_rep_day1'][None,...]
            param_dict['seq_param_day1'] = params['seq_param_day1'][None,...]
            
            param_dict['theta_Q_day2'] = params['theta_Q_day2'][None,...]
            param_dict['theta_rep_day2'] = params['theta_rep_day2'][None,...]
            param_dict['seq_param_day2'] = params['seq_param_day2'][None,...]
            
        newagent = models.Seqboost_2days_nlor(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    elif model == 'Handedness':
        num_params = models.Handedness_2days_nolr.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr'] = params_uniform[0:1, :]
            param_dict['theta_Q_day1'] = params_uniform[1:2, :]*6
            param_dict['theta_rep_day1'] = params_uniform[2:3, :]*6
            param_dict['hand_param_day1'] = params_uniform[3:4, :]*6
            
            param_dict['theta_Q_day2'] = params_uniform[4:5, :]*6
            param_dict['theta_rep_day2'] = params_uniform[5:6, :]*6
            param_dict['hand_param_day2'] = params_uniform[6:7, :]*6
            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr'] = params['lr'][None,...]
            param_dict['theta_Q_day1'] = params['theta_Q_day1'][None,...]
            param_dict['theta_rep_day1'] = params['theta_rep_day1'][None,...]
            param_dict['hand_param_day1'] = params['hand_param_day1'][None,...]
            
            param_dict['theta_Q_day2'] = params['theta_Q_day2'][None,...]
            param_dict['theta_rep_day2'] = params['theta_rep_day2'][None,...]
            param_dict['hand_param_day2'] = params['hand_param_day2'][None,...]
            
        newagent = models.Handedness_2days_nolr(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    elif model == 'Conflict':
        num_params = models.Conflict_2days_nolr.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr'] = params_uniform[0:1, :]
            param_dict['theta_Q_day1'] = params_uniform[1:2, :]*6
            param_dict['theta_rep_day1'] = params_uniform[2:3, :]*6
            param_dict['theta_conflict_day1'] = params_uniform[3:4, :]*6
            
            param_dict['theta_Q_day2'] = params_uniform[4:5, :]*6
            param_dict['theta_rep_day2'] = params_uniform[5:6, :]*6
            param_dict['theta_conflict_day2'] = params_uniform[6:7, :]*6
            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr'] = params['lr'][None,...]
            param_dict['theta_Q_day1'] = params['theta_Q_day1'][None,...]
            param_dict['theta_rep_day1'] = params['theta_rep_day1'][None,...]
            param_dict['theta_conflict_day1'] = params['theta_conflict_day1'][None,...]
            
            param_dict['theta_Q_day2'] = params['theta_Q_day2'][None,...]
            param_dict['theta_rep_day2'] = params['theta_rep_day2'][None,...]
            param_dict['theta_conflict_day2'] = params['theta_conflict_day2'][None,...]
            
        newagent = models.Conflict_2days_nolr(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    elif model == 'Repbias_CongConflict_2days':
        num_params = models.Repbias_CongConflict_2days_nolr.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr'] = params_uniform[0:1, :]
            param_dict['theta_Q_day1'] = params_uniform[1:2, :]*6
            param_dict['theta_rep_day1'] = params_uniform[2:3, :]*6
            param_dict['theta_conflict_day1'] = params_uniform[3:4, :]*6
            
            param_dict['theta_Q_day2'] = params_uniform[4:5, :]*6
            param_dict['theta_rep_day2'] = params_uniform[5:6, :]*6
            param_dict['theta_conflict_day2'] = params_uniform[6:7, :]*6
            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr'] = params['lr'][None,...]
            param_dict['theta_Q_day1'] = params['theta_Q_day1'][None,...]
            param_dict['theta_rep_day1'] = params['theta_rep_day1'][None,...]
            param_dict['theta_conflict_day1'] = params['theta_conflict_day1'][None,...]
            
            param_dict['theta_Q_day2'] = params['theta_Q_day2'][None,...]
            param_dict['theta_rep_day2'] = params['theta_rep_day2'][None,...]
            param_dict['theta_conflict_day2'] = params['theta_conflict_day2'][None,...]
            
        newagent = models.Repbias_CongConflict_2days_nolr(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    elif model == 'Repbias_Interaction_2days':
        num_params = models.Repbias_Interaction_2days_nolr.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr'] = params_uniform[0:1, :]
            param_dict['theta_Q_day1'] = params_uniform[1:2, :]*6
            param_dict['theta_rep_day1'] = params_uniform[2:3, :]*6
            param_dict['interaction_param_day1'] = params_uniform[3:4, :]*6
            
            param_dict['theta_Q_day2'] = params_uniform[4:5, :]*6
            param_dict['theta_rep_day2'] = params_uniform[5:6, :]*6
            param_dict['interaction_param_day2'] = params_uniform[6:7, :]*6
            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr'] = params['lr'][None,...]
            param_dict['theta_Q_day1'] = params['theta_Q_day1'][None,...]
            param_dict['theta_rep_day1'] = params['theta_rep_day1'][None,...]
            param_dict['interaction_param_day1'] = params['interaction_param_day1'][None,...]
            
            param_dict['theta_Q_day2'] = params['theta_Q_day2'][None,...]
            param_dict['theta_rep_day2'] = params['theta_rep_day2'][None,...]
            param_dict['interaction_param_day2'] = params['interaction_param_day2'][None,...]
            
        newagent = models.Interaction_2days_nolr(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
        
    elif model == 'OnlyR':
        num_params = models.OnlyR_2days_nolr.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr'] = params_uniform[0:1, :]
            param_dict['theta_Q_day1'] = params_uniform[1:2, :]*6
            param_dict['theta_Q_day2'] = params_uniform[2:3, :]*6
            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr'] = params['lr'][None,...]
            param_dict['theta_Q_day1'] = params['theta_Q_day1'][None,...]
            param_dict['theta_Q_day2'] = params['theta_Q_day2'][None,...]
            
        newagent = models.OnlyR_2days_nolr(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])

    elif model == 'OnlyQ':
        num_params = models.OnlyQ_2days_nolr.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr'] = params_uniform[0:1, :]
            param_dict['theta_Qcong_day1'] = params_uniform[1:2, :]*6
            param_dict['theta_Qinc_day1'] = params_uniform[2:3, :]*6
            param_dict['theta_Qrand_day1'] = params_uniform[3:4, :]*6
            
            param_dict['theta_Qcong_day2'] = params_uniform[4:5, :]*6
            param_dict['theta_Qinc_day2'] = params_uniform[5:6, :]*6
            param_dict['theta_Qrand_day2'] = params_uniform[6:7, :]*6
            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr'] = params['lr'][None,...]
            param_dict['theta_Qcong_day1'] = params['theta_Qcong_day1'][None,...]
            param_dict['theta_Qinc_day1'] = params['theta_Qinc_day1'][None,...]
            param_dict['theta_Qrand_day1'] = params['theta_Qrand_day1'][None,...]

            param_dict['theta_Qcong_day2'] = params['theta_Qcong_day2'][None,...]
            param_dict['theta_Qinc_day2'] = params['theta_Qinc_day2'][None,...]
            param_dict['theta_Qrand_day2'] = params['theta_Qrand_day2'][None,...]
            
        newagent = models.OnlyQ_2days_nolr(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    elif model == 'Repbias_Conflict_1day':
        num_params = models.Repbias_Conflict_1day.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr'] = params_uniform[0:1, :]
            param_dict['theta_Q'] = params_uniform[1:2, :]*6
            param_dict['theta_rep'] = params_uniform[2:3, :]*6
            param_dict['theta_conflict'] = params_uniform[3:4, :]*6

            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr'] = params['lr'][None,...]
            param_dict['theta_Q'] = params['theta_Q'][None,...]
            param_dict['theta_rep'] = params['theta_rep'][None,...]
            param_dict['theta_conflict'] = params['theta_conflict'][None,...]

            
        newagent = models.Repbias_Conflict_1day(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    elif model == 'Repbias_Handedness_1day':
        num_params = models.Repbias_Handedness_1day.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr'] = params_uniform[0:1, :]
            param_dict['theta_Q'] = params_uniform[1:2, :]*6
            param_dict['theta_rep'] = params_uniform[2:3, :]*6
            param_dict['hand_param'] = params_uniform[3:4, :]*6

            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr'] = params['lr'][None,...]
            param_dict['theta_Q'] = params['theta_Q'][None,...]
            param_dict['theta_rep'] = params['theta_rep'][None,...]
            param_dict['hand_param'] = params['hand_param'][None,...]
            
        newagent = models.Repbias_Handedness_1day(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
        
    elif model == 'Repbias_Interaction_1day':
        num_params = models.Repbias_Interaction_1day.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr'] = params_uniform[0:1, :]
            param_dict['theta_Q'] = params_uniform[1:2, :]*6
            param_dict['theta_rep'] = params_uniform[2:3, :]*6
            param_dict['interaction_param'] = params_uniform[3:4, :]*6
            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr'] = params['lr'][None,...]
            param_dict['theta_Q'] = params['theta_Q'][None,...]
            param_dict['theta_rep'] = params['theta_rep'][None,...]
            param_dict['interaction_param'] = params['interaction_param'][None,...]
            
        newagent = models.Repbias_Interaction_1day(param_dict,
                              
                              k=torch.tensor([k]),
                              Q_init=Q_init[None, ...])
        
    elif model == 'Repbias_CongConflict_1day':
        num_params = models.Repbias_CongConflict_1day.num_params #number of latent model parameters
        param_dict = {}
        
        if params is None:
            print("Setting random parameters.")
            params_uniform = torch.tensor(np.random.uniform(0,1, (num_params, num_agents)))
            
            param_dict['lr'] = params_uniform[0:1, :]
            param_dict['theta_Q'] = params_uniform[1:2, :]*6
            param_dict['theta_rep'] = params_uniform[2:3, :]*6
            param_dict['theta_conflict'] = params_uniform[3:4, :]*6

            
        else:
            print("Setting initial parameters as provided.")
            param_dict['lr'] = params['lr'][None,...]
            param_dict['theta_Q'] = params['theta_Q'][None,...]
            param_dict['theta_rep'] = params['theta_rep'][None,...]
            param_dict['theta_conflict'] = params['theta_conflict'][None,...]
            
        newagent = models.Repbias_CongConflict_1day(param_dict,
                              
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
    
    import env 
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
            jokertypes:
                DTT Types
                (clipre) -1/0/1/2 : no joker/random/congruent/incongruent
                (published) : -1/0/1/2/3/4 : no joker/random choice/congruent/incongruent/NLP/NHP
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
        
    # groupdata_df_1 = groupdata_df_1.drop(['blockidx', 'trialsequence', 'outcomes', 'choices'], axis = 1)
    
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
        
        # groupdata_df_2 = groupdata_df_2.drop(['blockidx', 'trialsequence', 'outcomes', 'choices'], axis = 1)
        
    custom_palette = ['r', 'g', 'b'] # random, congruent, incongruent
    if plot_single:
        for ag_idx in np.sort(groupdata_df_1['ag_idx'].unique()):
            agent_df_1 = groupdata_df_1[groupdata_df_1['ag_idx'] == ag_idx]
            agent_df_1['jokertypes'] = agent_df_1['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'no joker')))
            
            if df2 is not None:
                agent_df_2 = groupdata_df_2[groupdata_df_2['ag_idx'] == ag_idx]
                agent_df_2['jokertypes'] = agent_df_2['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'no joker')))
            
            if df2 is not None:
                
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
                plt.title(f'agent {ag_idx}, model {model_1}')
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
            
            # agent_df_1['jokertypes'] = agent_df_1['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'NLP' if x == 3 else 'NHP' if x == 4 else 'no joker')))
            agent_df_1['jokertypes'] = agent_df_1['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'no joker')))
            # agent_df_2['jokertypes'] = agent_df_2['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'NLP' if x == 3 else 'NHP' if x == 4 else 'no joker')))
            agent_df_2['jokertypes'] = agent_df_2['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'no joker')))
            
            plot_dual_behav(agent_df_1, agent_df_2)
            
    "----- Plot grouplevel"
    groupdata_df_1 = groupdata_df_1.drop(['model'], axis = 1)
    grouped_df_1 = pd.DataFrame(groupdata_df_1.loc[:, ['ag_idx',
                                               'block_num',
                                               'jokertypes',
                                               'choices_GD']].groupby(['ag_idx','block_num', 'jokertypes'], as_index = False).mean())
    grouped_df_1['jokertypes'] = grouped_df_1['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'NLP' if x == 3 else 'NHP' if x == 4 else 'no joker')))
    
    if df2 is not None:
        groupdata_df_2 = groupdata_df_2.drop(['model'], axis = 1)
        grouped_df_2 = pd.DataFrame(groupdata_df_2.loc[:, ['ag_idx',
                                                   'block_num',
                                                   'jokertypes',
                                                   'choices_GD']].groupby(['ag_idx','block_num', 'jokertypes'], as_index = False).mean())
        # grouped_df_2['jokertypes'] = grouped_df_2['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'NLP' if x == 3 else 'NHP' if x == 4 else 'no joker')))
        grouped_df_2['jokertypes'] = grouped_df_2['jokertypes'].map(lambda x: 'random' if x == 0 else ('congruent' if x == 1 else ('incongruent' if x == 2 else 'no joker')))
        
        grouped_df_2['blocknum'] = grouped_df_2['block_num'].map(lambda x: 1 if x == 0 else x)
        grouped_df_2['blocknum'] = grouped_df_2['blocknum'].map(lambda x: 2 if x == 3 else x)
        grouped_df_2['blocknum'] = grouped_df_2['blocknum'].map(lambda x: 3 if x == 4 or x == 5 else x)
        grouped_df_2['blocknum'] = grouped_df_2['blocknum'].map(lambda x: 4 if x == 6 or x == 7 else x)
        grouped_df_2['blocknum'] = grouped_df_2['blocknum'].map(lambda x: 5 if x == 8 or x == 9 else x)
        grouped_df_2['blocknum'] = grouped_df_2['blocknum'].map(lambda x: 6 if x == 10 or x == 11 else x)
        grouped_df_2['blocknum'] = grouped_df_2['blocknum'].map(lambda x: 7 if x == 12 or x == 13 else x)
        grouped_df_2 = grouped_df_2[grouped_df_2['jokertypes'] != 'no joker']
    
    grouped_df_1['blocknum'] = grouped_df_1['block_num'].map(lambda x: 1 if x == 0 else x)
    grouped_df_1['blocknum'] = grouped_df_1['blocknum'].map(lambda x: 2 if x == 3 else x)
    grouped_df_1['blocknum'] = grouped_df_1['blocknum'].map(lambda x: 3 if x == 4 or x == 5 else x)
    grouped_df_1['blocknum'] = grouped_df_1['blocknum'].map(lambda x: 4 if x == 6 or x == 7 else x)
    grouped_df_1['blocknum'] = grouped_df_1['blocknum'].map(lambda x: 5 if x == 8 or x == 9 else x)
    grouped_df_1['blocknum'] = grouped_df_1['blocknum'].map(lambda x: 6 if x == 10 or x == 11 else x)
    grouped_df_1['blocknum'] = grouped_df_1['blocknum'].map(lambda x: 7 if x == 12 or x == 13 else x)
    
    grouped_df_1 = grouped_df_1[grouped_df_1['jokertypes'] != 'no joker']
    if df2 is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        sns.lineplot(x = 'blocknum', 
                    y = 'choices_GD', 
                    hue = 'jokertypes', 
                    data = grouped_df_1,
                    ax = ax1)
        ax1.set_xticks(np.arange(1, 8), minor = True)
        ax1.set_xlabel('Block no.')
        ax1.set_ylabel('HRCF (%)')
        ax1.grid(which='minor', alpha=0.5)
        ax1.set_title(f'Model {model_1}')
        ax1.axvline(3.5, color='k', linewidth=0.5)
        ax1.get_legend().remove()
        
        sns.lineplot(x = 'blocknum', 
                    y = 'choices_GD', 
                    hue ='jokertypes', 
                    data = grouped_df_2,
                    ax = ax2)

        ax2.set_xticks(np.arange(1, 8), minor = True)
        ax2.grid(which='minor', alpha=0.5)
        ax2.set_title(f'Model {model_2}')
        ax2.get_legend().remove()
        ax2.set_xlabel('Block no.')
        ax2.set_ylabel('HRCF (%)')
        ax2.axvline(3.5, color='k', linewidth=0.5)
        plt.savefig('/home/sascha/Downloads/exp_vs_sim.tiff', dpi=600)
        plt.show()      

    else:
        "----- Remove error trials (where choices_GD == -2"

        # means = grouped_df_1.groupby(['jokertypes', 'blocknum'], as_index = False).mean()
        # stdvars = grouped_df_1.groupby(['jokertypes', 'blocknum'], as_index = False).std()
        # means.rename(columns={'jokertypes':'DTT Types'}, inplace = True)
        grouped_df_1.rename(columns={'jokertypes':'DTT Types'}, inplace = True)
        
        custom_palette = ['#c7028c', '#63040f', '#96e6c7'] # congruent, incongruent, random
        grouped_df_1['day'] = grouped_df_1['blocknum'].map(lambda x: 1 if x <=3 else 2)
        
        fig, ax = plt.subplots()
        
        sns.lineplot(x = "blocknum",
                    y = "choices_GD",
                    hue = "DTT Types",
                    data = grouped_df_1,
                    palette = custom_palette,
                    err_style = "bars",
                    errorbar = ("se", 1),
                    ax = ax)
        
        # ax.axvline(3.5, color='k', linewidth=0.5)
        # sns.lineplot(x = "blocknum",
        #             y = "choices_GD",
        #             hue = "DTT Types",
        #             data = grouped_df_1,
        #             palette = custom_palette,
        #             ax = ax)
        ax.set_ylim([0.61, 0.95])
        ax.set_xlabel('Block no.')
        ax.set_ylabel('HRCF (%)')
        plt.title(f'Group Behaviour for model {model_1}')
        plt.savefig('/home/sascha/Desktop/Paper 2024/KW2.png', dpi=600)
        plt.show()
        
        fig, ax = plt.subplots(figsize = (5, 5))
        sns.lineplot(x = "day",
                    y = "choices_GD",
                    hue = "DTT Types",
                    data = grouped_df_1,
                    palette = custom_palette,
                    err_style = "bars",
                    errorbar = ("se", 1),
                    ax = ax)
        ax.set_xticks([1,2])
        ax.set_ylim([0.61, 1])
        ax.set_xlabel('Day')
        ax.set_ylabel('HRCF (%)')
        plt.savefig('/home/sascha/Desktop/Paper 2024/KW2.png', dpi=600)
        plt.show()
        # dfgh

        
        return grouped_df_1
        
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

def get_data_from_file(file_dir = None):
    import pickle
    import tkinter as tk
    from tkinter import filedialog
    
    if file_dir is None:
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
            
    else:
        res = pickle.load(open( file_dir, "rb" ))
            
    # print("len res is %d"%len(res))
    assert len(res) == 5
    assert len(res[2]) == 3
    loss = res[2][0]
    BIC = res[2][1]
    AIC = res[2][2]
    post_sample_df, expdata_df, _, params_df, agent_elbo_tuple = res
    
    try:
        print("BIC = %.2f"%BIC)
        
    except:
        pass
    
    try:
        print("AIC = %.2f"%AIC)
        
    except:
        pass
    
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
    
    sociopsy_df = get_sociopsy_df()
    return post_sample_df, expdata_df, loss, params_df, num_params, sociopsy_df, agent_elbo_tuple
        
def create_complete_df(inf_mean_df, sociopsy_df, expdata_df, post_sample_df, param_names):
    '''

    Parameters
    ----------
    inf_mean_df : DataFrame
        columns:
            model
            ag_idx
            group
            ID
            lr_day1
            theta_Q_day1
            theta_rep_day1
            lr_day2
            theta_Q_day2
            theta_rep_day2
        
    sociopsy_df : DataFrame
        columns:
            ID
            age
            gender
            handedness
        
    expdata_df : DataFrame
        columns:
            choices
            choices_GD
            outcomes
            trialsequence
            blocktype
            jokertypes
            blockidx
            group
            ID
            ag_idx
            model
        
    post_sample_df : DataFrame
        columns:
            lr_day1
            theta_Q_day1
            theta_rep_day1
            lr_day2
            theta_Q_day2
            theta_rep_day2
            ag_idx
            group
            model
            ID
            (handedness)
        
    param_names : TYPE
        DESCRIPTION.

    Returns
    -------
    complete_df : TYPE
        DESCRIPTION.

    '''
    
    if 'ID' in inf_mean_df.columns:
        '''
            For behavioral data
        '''
        
        '''
            Errors
        '''
        print("\nComputing errorrates.")
        error_df = anal.compute_errors(expdata_df)
        "error_df.columns: 'group', 'ID', 'ER_dtt', 'ER_dtt_day1', 'ER_dtt_day2', 'ER_stt','ER_total', 'ER_total_day1', 'ER_total_day2'"

        '''
            Differences between days
        '''
        diffs_df = anal.daydiff(inf_mean_df)

        '''
            Points
        '''
        points_df = compute_points(expdata_df)
        
        '''
            Within-subject correlations
        '''
        corr_df = anal.within_subject_corr(post_sample_df, [*param_names])

        '''
            HPCF
        '''    
        print("\nComputing HPCF.")
        hpcf_df = compute_hpcf(expdata_df)

        '''
            Sociopsychological Data
        '''
        print("\nRetrieving sociopsychological data.")
        complete_df_temp = pd.merge(inf_mean_df.drop(['handedness'], axis=1), 
                               sociopsy_df[sociopsy_df['ID'].isin(inf_mean_df['ID'])], 
                               on = 'ID')
        
        '''
            RT
        '''
        
        RT_df = compute_RT(expdata_df)
        
        '''
            Q: Did you notice a sequence?
        '''
        _, expdata_df_wseq = pickle.load(open("behav_data/preproc_data_all.p", "rb" ))
        '''expdata_df_wseq.columns: choices, choices_GD, outcomes, trialsequence, blocktype,
        jokertypes, blockidx, RT, group, handedness, age, gender,
        q_sometimes_easier, q_notice_a_sequence, ID, ag_idx, model'''
        # expdata_df_wseq['gender'] = expdata_df_wseq['gender'].map(lambda x: 0 if x=='female' else (1 if x == 'male' else 2))

        expdata_df_wseq = expdata_df_wseq.drop(['ag_idx'], axis=1)
        notice_seq_df = pd.DataFrame(expdata_df_wseq.loc[:, ['ID',
                                'q_notice_a_sequence']].groupby(['ID'], as_index = False).mean())

    
        '''
            Merge them dataframes
        '''
        print("\nMerging Dataframes.")
        df1 = pd.merge(error_df[error_df['ID'].isin(inf_mean_df['ID'])], complete_df_temp, on='ID')
        df2 = pd.merge(df1, RT_df, on = 'ID')
        df3 = pd.merge(df2, notice_seq_df, on='ID')   
        df4 = pd.merge(df3, diffs_df, on='ID')
        df5 = pd.merge(df4, points_df, on='ID')
        df6 = pd.merge(df5, hpcf_df, on='ID')
        complete_df = pd.merge(df6, corr_df, on='ID')
        
        '''
            Rearrange dat dataframe
        '''
        assert np.all(complete_df['ag_idx_x'] == complete_df['ag_idx_y'])
        complete_df = complete_df.drop(['ag_idx_x'], axis = 1)
        complete_df.rename(columns={'ag_idx_y': 'ag_idx'}, inplace = True)
        
        assert np.all(complete_df['group_x'] == complete_df['group_y'])
        complete_df = complete_df.drop(['group_x'], axis = 1)
        complete_df.rename(columns={'group_y': 'group'}, inplace = True)
        
        firstcolumns = ['ID', 'ag_idx', 'group', 'age', 'gender', 'handedness', *param_names]
        complete_df = complete_df[[*firstcolumns + [col for col in complete_df.columns if col not in firstcolumns]]]
        complete_df = complete_df[[col for col in complete_df.columns if col != 'model'] + ['model']]
        
    else:
        '''
            For simulated datasets
        '''
        
        '''
            Errors
        '''
        print("Computing errorrates.")
        error_df = anal.compute_errors(expdata_df, identifier = 'ag_idx')
        
        '''
            Differences between days
        '''
        diffs_df = anal.daydiff(inf_mean_df)
        
        '''
            Points
        '''
        points_df = compute_points(expdata_df, identifier = 'ag_idx')
        
        '''
            Within-subject correlations
        '''
        # corr_df = anal.within_subject_corr(post_sample_df, [*param_names])
        df1 = pd.merge(error_df[error_df['ag_idx'].isin(inf_mean_df['ag_idx'])], inf_mean_df, on='ag_idx')
        df2 = pd.merge(df1, diffs_df, on='ag_idx')
        complete_df = pd.merge(df2, points_df, on='ag_idx')
    
    assert len(complete_df) == len(inf_mean_df)
    complete_df = complete_df.sort_values(by=['ag_idx'])
    
    
    for col in complete_df.columns:
        if type(complete_df[col][0]) != str:
            complete_df[col] = complete_df[col].astype('float')
    
    return complete_df

def compute_points(expdata_df, identifier = 'ID'):
    expdata_df = expdata_df[expdata_df['outcomes'] != -1]
    expdata_df = expdata_df[expdata_df['outcomes'] != -2]
    
    "Total"
    points_total = expdata_df.loc[:, [identifier, 'outcomes']].groupby([identifier], as_index=False).sum()
    points_total.rename(columns={'outcomes': 'points_total'}, inplace = True)
    
    "----- Day 1"
    expdata_df_day1 = expdata_df[expdata_df['blockidx'] <= 5 ]
    points_day1 = expdata_df_day1.loc[:, [identifier, 'outcomes']].groupby([identifier], as_index=False).sum()
    points_day1.rename(columns={'outcomes': 'points_day1'}, inplace = True)
    
    "--- Day 1: STT"
    expdata_df_stt_day1 = expdata_df[(expdata_df['trialsequence'] < 10) & (expdata_df['blockidx'] <= 5)]
    points_stt_df_day1 = expdata_df_stt_day1.loc[:, [identifier, 'outcomes']].groupby([identifier], as_index = False).sum()
    points_stt_df_day1.rename(columns = {'outcomes' : 'points_stt_day1'}, inplace = True)
    
    "STT sequential Day 1"
    expdata_df_stt_seq_day1 = expdata_df[(expdata_df['trialsequence'] < 10) & (expdata_df['blockidx'] <= 5) & (expdata_df['blocktype'] == 0)]
    points_stt_seq_df_day1 = expdata_df_stt_seq_day1.loc[:, [identifier, 'outcomes']].groupby([identifier], as_index = False).sum()
    points_stt_seq_df_day1.rename(columns = {'outcomes' : 'points_stt_seq_day1'}, inplace = True)
    
    "STT Random Day 1"
    expdata_df_stt_rand_day1 = expdata_df[(expdata_df['trialsequence'] < 10) & (expdata_df['blockidx'] <= 5) & (expdata_df['blocktype'] == 1)]
    points_stt_rand_df_day1 = expdata_df_stt_rand_day1.loc[:, [identifier, 'outcomes']].groupby([identifier], as_index = False).sum()
    points_stt_rand_df_day1.rename(columns = {'outcomes' : 'points_stt_rand_day1'}, inplace = True)
    
    "--- Day 1: DTT"
    "DTT Jokertypes Day 1"
    "-1/0/1/2 : no joker/random/congruent/incongruent"
    expdata_df_dtt_day1 = expdata_df[(expdata_df['trialsequence'] > 10) & (expdata_df['blockidx'] <= 5)]
    points_dtt_df_jokertypes_day1 = expdata_df_dtt_day1.loc[:, [identifier, 'outcomes', 'jokertypes']].groupby([identifier, 'jokertypes'], as_index = False).sum()
    points_dtt_df_jokertypes_day1_pivoted = points_dtt_df_jokertypes_day1.pivot(index=identifier, columns='jokertypes', values = 'outcomes').reset_index()
    points_dtt_df_jokertypes_day1_pivoted.rename(columns={0: 'points_randomdtt_day1',
                                               1: 'points_congruent_day1',
                                               2: 'points_incongruent_day1'}, inplace=True)
    
    "DTT Seq"
    expdata_df_dtt_seq_day1 = expdata_df[(expdata_df['trialsequence'] > 10) & (expdata_df['blockidx'] <= 5) & (expdata_df['blocktype'] == 0)]
    points_dtt_seq_day1 = expdata_df_dtt_seq_day1.loc[:, [identifier, 'outcomes']].groupby([identifier], as_index=False).sum()
    points_dtt_seq_day1.rename(columns = {'outcomes' : 'points_dtt_seq_day1'}, inplace = True)
    
    "DTT Rand"
    expdata_df_dtt_rand_day1 = expdata_df[(expdata_df['trialsequence'] > 10) & (expdata_df['blockidx'] <= 5) & (expdata_df['blocktype'] == 1)]
    points_dtt_rand_day1 = expdata_df_dtt_rand_day1.loc[:, [identifier, 'outcomes']].groupby([identifier], as_index=False).sum()
    points_dtt_rand_day1.rename(columns = {'outcomes' : 'points_dtt_rand_day1'}, inplace = True)
    
    "--- Day 1: DTT"
    points_df_dtt_day1 = expdata_df_dtt_day1.loc[:, [identifier, 'outcomes']].groupby([identifier], as_index = False).sum()
    points_df_dtt_day1.rename(columns = {'outcomes' : 'points_dtt_day1'}, inplace = True)
    
    "----- Day 2"
    expdata_df_day2 = expdata_df[expdata_df['blockidx'] > 5 ]
    points_day2 = expdata_df_day2.loc[:, [identifier, 'outcomes']].groupby([identifier], as_index=False).sum()
    points_day2.rename(columns={'outcomes': 'points_day2'}, inplace = True)
    
    "STT Day 2"
    expdata_df_stt_day2 = expdata_df[(expdata_df['trialsequence'] < 10) & (expdata_df['blockidx'] > 5)]
    points_stt_df_day2 = expdata_df_stt_day2.loc[:, [identifier, 'outcomes']].groupby([identifier], as_index = False).sum()
    points_stt_df_day2.rename(columns = {'outcomes' : 'points_stt_day2'}, inplace = True)
    
    "STT sequential Day 2"
    expdata_df_stt_seq_day2 = expdata_df[(expdata_df['trialsequence'] < 10) & (expdata_df['blockidx'] > 5) & (expdata_df['blocktype'] == 0)]
    points_stt_seq_df_day2 = expdata_df_stt_seq_day2.loc[:, [identifier, 'outcomes']].groupby([identifier], as_index = False).sum()
    points_stt_seq_df_day2.rename(columns = {'outcomes' : 'points_stt_seq_day2'}, inplace = True)
    
    "STT Random Day 2"
    expdata_df_stt_rand_day2 = expdata_df[(expdata_df['trialsequence'] < 10) & (expdata_df['blockidx'] > 5) & (expdata_df['blocktype'] == 1)]
    points_stt_rand_df_day2 = expdata_df_stt_rand_day2.loc[:, [identifier, 'outcomes']].groupby([identifier], as_index = False).sum()
    points_stt_rand_df_day2.rename(columns = {'outcomes' : 'points_stt_rand_day2'}, inplace = True)
    
    
    "DTT Seq"
    expdata_df_dtt_seq_day2 = expdata_df[(expdata_df['trialsequence'] > 10) & (expdata_df['blockidx'] > 5) & (expdata_df['blocktype'] == 0)]
    points_dtt_seq_day2 = expdata_df_dtt_seq_day2.loc[:, [identifier, 'outcomes']].groupby([identifier], as_index=False).sum()
    points_dtt_seq_day2.rename(columns = {'outcomes' : 'points_dtt_seq_day2'}, inplace = True)
    
    "DTT Rand"
    expdata_df_dtt_rand_day2 = expdata_df[(expdata_df['trialsequence'] > 10) & (expdata_df['blockidx'] > 5) & (expdata_df['blocktype'] == 1)]
    points_dtt_rand_day2 = expdata_df_dtt_rand_day2.loc[:, [identifier, 'outcomes']].groupby([identifier], as_index=False).sum()
    points_dtt_rand_day2.rename(columns = {'outcomes' : 'points_dtt_rand_day2'}, inplace = True)
    
    "DTT Jokertypes Day 2"
    expdata_df_dtt_day2 = expdata_df[(expdata_df['trialsequence'] > 10) & (expdata_df['blockidx'] > 5)]
    points_dtt_df_jokertypes_day2 = expdata_df_dtt_day2.loc[:, [identifier, 'outcomes', 'jokertypes']].groupby([identifier, 'jokertypes'], as_index = False).sum()
    points_dtt_df_jokertypes_day2_pivoted = points_dtt_df_jokertypes_day2.pivot(index=identifier, columns='jokertypes', values = 'outcomes').reset_index()
    points_dtt_df_jokertypes_day2_pivoted.rename(columns={0: 'points_randomdtt_day2',
                                               1: 'points_congruent_day2',
                                               2: 'points_incongruent_day2'}, inplace=True)
    
    "DTT Day 2"
    expdata_df_dtt_day2 = expdata_df_dtt_day2.loc[:, [identifier, 'outcomes']].groupby([identifier], as_index = False).sum()
    expdata_df_dtt_day2.rename(columns = {'outcomes' : 'points_dtt_day2'}, inplace = True)
    
    points_df = pd.merge(points_total, points_day1, on = identifier)
    points_df = pd.merge(points_df, points_stt_df_day1, on = identifier)
    points_df = pd.merge(points_df, points_stt_seq_df_day1, on = identifier)
    points_df = pd.merge(points_df, points_stt_rand_df_day1, on = identifier)
    points_df = pd.merge(points_df, points_dtt_df_jokertypes_day1_pivoted, on = identifier)
    points_df = pd.merge(points_df, points_df_dtt_day1, on = identifier)
    points_df = pd.merge(points_df, points_dtt_seq_day1, on = identifier)
    points_df = pd.merge(points_df, points_dtt_rand_day1, on = identifier)
    
    points_df = pd.merge(points_df, points_day2, on = identifier)
    points_df = pd.merge(points_df, points_stt_df_day2, on = identifier)
    points_df = pd.merge(points_df, points_stt_seq_df_day2, on = identifier)
    points_df = pd.merge(points_df, points_stt_rand_df_day2, on = identifier)
    
    points_df = pd.merge(points_df, points_dtt_df_jokertypes_day2_pivoted, on = identifier)
    
    points_df = pd.merge(points_df, expdata_df_dtt_day2, on = identifier)
    
    points_df = pd.merge(points_df, points_dtt_seq_day2, on = identifier)
    points_df = pd.merge(points_df, points_dtt_rand_day2, on = identifier)
    
    return points_df

def compute_hpcf(expdata_df):
    '''
    jokertypes : list
        DTT Types
        -1/0/1/2 : no joker/random/congruent/incongruent
    '''
    
    df = expdata_df[expdata_df['choices_GD'] != -1]
    df = expdata_df[expdata_df['choices_GD'] != -2]
    df = expdata_df[expdata_df['trialsequence'] > 10]
    
    hpcf_df = pd.DataFrame(data = {'ID': expdata_df['ID'].unique()})
    
    hpcf_day1 = pd.DataFrame(df[df['blockidx'] <= 5].loc[:, ['ID', 'choices_GD']].groupby(['ID'], as_index = False).mean())
    hpcf_day1.rename(columns={'choices_GD' : 'hpcf_day1'}, inplace = True)

    hpcf_day2 = pd.DataFrame(df[df['blockidx'] > 5].loc[:, ['ID', 'choices_GD']].groupby(['ID'], as_index = False).mean())
    hpcf_day2.rename(columns={'choices_GD' : 'hpcf_day2'}, inplace = True)
    
    "Random"
    df_rand = pd.DataFrame(df[df['jokertypes'] == 0].loc[:, ['ID', 'choices_GD']].groupby(['ID'], as_index = False).mean())
    df_rand.rename(columns = {'choices_GD': 'hpcf_rand'}, inplace = True)
    
    df_rand_day1 = pd.DataFrame(df[(df['jokertypes'] == 0) & (df['blockidx'] <= 5)].loc[:, ['ID', 'choices_GD']].groupby(['ID'], as_index = False).mean())
    df_rand_day1.rename(columns = {'choices_GD': 'hpcf_rand_day1'}, inplace = True)
    
    df_rand_day2 = pd.DataFrame(df[(df['jokertypes'] == 0) & (df['blockidx'] > 5)].loc[:, ['ID', 'choices_GD']].groupby(['ID'], as_index = False).mean())
    df_rand_day2.rename(columns = {'choices_GD': 'hpcf_rand_day2'}, inplace = True)
    
    "Congruent"
    df_cong = pd.DataFrame(df[df['jokertypes'] == 1].loc[:, ['ID', 'choices_GD']].groupby(['ID'], as_index = False).mean())
    df_cong.rename(columns = {'choices_GD': 'hpcf_cong'}, inplace = True)
    
    df_cong_day1 = pd.DataFrame(df[(df['jokertypes'] == 1) & (df['blockidx'] <= 5)].loc[:, ['ID', 'choices_GD']].groupby(['ID'], as_index = False).mean())
    df_cong_day1.rename(columns = {'choices_GD': 'hpcf_cong_day1'}, inplace = True)
    
    df_cong_day2 = pd.DataFrame(df[(df['jokertypes'] == 1) & (df['blockidx'] > 5)].loc[:, ['ID', 'choices_GD']].groupby(['ID'], as_index = False).mean())
    df_cong_day2.rename(columns = {'choices_GD': 'hpcf_cong_day2'}, inplace = True)
    
    "Incongruent"
    df_inc = pd.DataFrame(df[df['jokertypes'] == 2].loc[:, ['ID', 'choices_GD']].groupby(['ID'], as_index = False).mean())
    df_inc.rename(columns = {'choices_GD': 'hpcf_incong'}, inplace = True)
    
    df_inc_day1 = pd.DataFrame(df[(df['jokertypes'] == 2) & (df['blockidx'] <= 5)].loc[:, ['ID', 'choices_GD']].groupby(['ID'], as_index = False).mean())
    df_inc_day1.rename(columns = {'choices_GD': 'hpcf_incong_day1'}, inplace = True)
    
    df_inc_day2 = pd.DataFrame(df[(df['jokertypes'] == 2) & (df['blockidx'] > 5)].loc[:, ['ID', 'choices_GD']].groupby(['ID'], as_index = False).mean())
    df_inc_day2.rename(columns = {'choices_GD': 'hpcf_incong_day2'}, inplace = True)
    
    
    hpcf_df = pd.merge(hpcf_df, hpcf_day1, on = 'ID')
    hpcf_df = pd.merge(hpcf_df, hpcf_day2, on = 'ID')
    
    hpcf_df = pd.merge(hpcf_df, df_rand, on = 'ID')
    hpcf_df = pd.merge(hpcf_df, df_rand_day1, on = 'ID')
    hpcf_df = pd.merge(hpcf_df, df_rand_day2, on = 'ID')
    
    hpcf_df = pd.merge(hpcf_df, df_cong, on = 'ID')
    hpcf_df = pd.merge(hpcf_df, df_cong_day1, on = 'ID')
    hpcf_df = pd.merge(hpcf_df, df_cong_day2, on = 'ID')
    
    hpcf_df = pd.merge(hpcf_df, df_inc, on = 'ID')
    hpcf_df = pd.merge(hpcf_df, df_inc_day1, on = 'ID')
    hpcf_df = pd.merge(hpcf_df, df_inc_day2, on = 'ID')
    
    return hpcf_df

def compute_RT(expdata_df):
    '''
    jokertypes : list
        DTT Types
        -1/0/1/2 : no joker/random/congruent/incongruent
    '''
    RT_df = expdata_df.loc[:, ['ID', 'RT', 'choices', 'blockidx', 'jokertypes']]
    RT_df = RT_df[RT_df['choices'] != -1]
    RT_df = RT_df[RT_df['choices'] != -2]
    RT_df_temp = pd.DataFrame(RT_df.loc[:, ['ID', 'RT']].groupby(['ID'], as_index = False).mean())
    RT_df_temp_day1 = pd.DataFrame(RT_df[RT_df['blockidx']<=5].loc[:, ['ID', 'RT']].groupby(['ID'], as_index = False).mean())
    RT_df_temp_day1 = RT_df_temp_day1.rename(columns={'RT':'RT_day1'})
    RT_df_temp_day2 = pd.DataFrame(RT_df[RT_df['blockidx']>5].loc[:, ['ID', 'RT']].groupby(['ID'], as_index = False).mean())
    RT_df_temp_day2 = RT_df_temp_day2.rename(columns={'RT':'RT_day2'})
    
    RT_df = pd.merge(RT_df_temp, RT_df_temp_day1, on='ID')
    RT_df = pd.merge(RT_df, RT_df_temp_day2, on='ID')
    
    RT_cond_df = expdata_df.loc[:, ['ID', 'RT', 'choices', 'blockidx', 'blocktype', 'trialsequence', 'jokertypes']]
    RT_cond_df = RT_cond_df[RT_cond_df['choices'] != -1]
    RT_cond_df = RT_cond_df[RT_cond_df['choices'] != -2]
    
    RT_df_rand_day1 = pd.DataFrame(RT_cond_df[(RT_cond_df['blockidx']<=5) & \
                                              (RT_cond_df['blocktype']==1)].loc[:, ['ID', 'RT']].groupby(['ID'], as_index = False).mean())
    RT_df_rand_day1.rename(columns={'RT':'RT_rand_day1'}, inplace = True)
    
    RT_df_rand_day2 = pd.DataFrame(RT_cond_df[(RT_cond_df['blockidx']>5) & \
                                              (RT_cond_df['blocktype']==1)].loc[:, ['ID', 'RT']].groupby(['ID'], as_index = False).mean())
    RT_df_rand_day2.rename(columns={'RT':'RT_rand_day2'}, inplace = True)
    
    RT_df_seq_day1 = pd.DataFrame(RT_cond_df[(RT_cond_df['blockidx']<=5) & \
                                             (RT_cond_df['blocktype']==0)].loc[:, ['ID', 'RT']].groupby(['ID'], as_index = False).mean())
    RT_df_seq_day1.rename(columns={'RT':'RT_seq_day1'}, inplace = True)
    
    RT_df_seq_day2 = pd.DataFrame(RT_cond_df[(RT_cond_df['blockidx']>5) & \
                                             (RT_cond_df['blocktype']==0)].loc[:, ['ID', 'RT']].groupby(['ID'], as_index = False).mean())
    RT_df_seq_day2.rename(columns={'RT':'RT_seq_day2'}, inplace = True)
    
    RT_stt_seq_day1 = pd.DataFrame(RT_cond_df[(RT_cond_df['blockidx']<=5) & \
                                              (RT_cond_df['blocktype']==0) & (RT_cond_df['trialsequence']<10)].loc[:, ['ID', 'RT']].groupby(['ID'], as_index = False).mean())
    RT_stt_seq_day1.rename(columns={'RT':'RT_stt_seq_day1'}, inplace = True)
    
    RT_stt_rand_day1 = pd.DataFrame(RT_cond_df[(RT_cond_df['blockidx']<=5) & \
                                               (RT_cond_df['blocktype']==1) & (RT_cond_df['trialsequence']<10)].loc[:, ['ID', 'RT']].groupby(['ID'], as_index = False).mean())
    RT_stt_rand_day1.rename(columns={'RT':'RT_stt_rand_day1'}, inplace = True)

    RT_stt_seq_day2 = pd.DataFrame(RT_cond_df[(RT_cond_df['blockidx']>5) & \
                                              (RT_cond_df['blocktype']==0) & (RT_cond_df['trialsequence']<10)].loc[:, ['ID', 'RT']].groupby(['ID'], as_index = False).mean())
    RT_stt_seq_day2.rename(columns={'RT':'RT_stt_seq_day2'}, inplace = True)
    
    RT_stt_rand_day2 = pd.DataFrame(RT_cond_df[(RT_cond_df['blockidx']>5) & \
                                               (RT_cond_df['blocktype']==1) & (RT_cond_df['trialsequence']<10)].loc[:, ['ID', 'RT']].groupby(['ID'], as_index = False).mean())
    RT_stt_rand_day2.rename(columns={'RT':'RT_stt_rand_day2'}, inplace = True)
    
    RT_stt_day2 = pd.DataFrame(RT_cond_df[(RT_cond_df['blockidx']>5) & \
                                          (RT_cond_df['trialsequence']<10)].loc[:, ['ID', 'RT']].groupby(['ID'], as_index = False).mean())
    RT_stt_day2.rename(columns={'RT':'RT_stt_day2'}, inplace = True)

    RT_congruent_day2 = pd.DataFrame(RT_cond_df[(RT_cond_df['blockidx']>5) & \
                                                (RT_cond_df['jokertypes']==1) & (RT_cond_df['blocktype']==0)].loc[:, ['ID', 'RT']].groupby(['ID'], as_index = False).mean())
    RT_congruent_day2.rename(columns={'RT':'RT_congruent_day2'}, inplace = True)

    RT_incongruent_day2 = pd.DataFrame(RT_cond_df[(RT_cond_df['blockidx']>5) & \
                                                  (RT_cond_df['jokertypes']==2)].loc[:, ['ID', 'RT']].groupby(['ID'], as_index = False).mean())
    RT_incongruent_day2.rename(columns={'RT':'RT_incongruent_day2'}, inplace = True)
    
    RT_randomdtt_day2 = pd.DataFrame(RT_cond_df[(RT_cond_df['blockidx']>5) & \
                                             (RT_cond_df['jokertypes']==0) & (RT_cond_df['blocktype']==1)].loc[:, ['ID', 'RT']].groupby(['ID'], as_index = False).mean())
    RT_randomdtt_day2.rename(columns={'RT':'RT_randomdtt_day2'}, inplace = True)
    
    RT_dtt_day2 = pd.DataFrame(RT_cond_df[(RT_cond_df['blockidx']>5) & \
                                             (RT_cond_df['trialsequence']>10)].loc[:, ['ID', 'RT']].groupby(['ID'], as_index = False).mean())
    RT_dtt_day2.rename(columns={'RT':'RT_dtt_day2'}, inplace = True)
    
    RT_df = pd.merge(RT_df, RT_df_rand_day1, on = 'ID')
    RT_df = pd.merge(RT_df, RT_df_rand_day2, on = 'ID')
    RT_df = pd.merge(RT_df, RT_df_seq_day1, on = 'ID')
    RT_df = pd.merge(RT_df, RT_df_seq_day2, on = 'ID')
    
    RT_df = pd.merge(RT_df, RT_stt_seq_day1, on = 'ID')
    RT_df = pd.merge(RT_df, RT_stt_rand_day1, on = 'ID')
    RT_df = pd.merge(RT_df, RT_stt_seq_day2, on = 'ID')
    RT_df = pd.merge(RT_df, RT_stt_rand_day2, on = 'ID')
    RT_df = pd.merge(RT_df, RT_stt_day2, on = 'ID')
    RT_df = pd.merge(RT_df, RT_dtt_day2, on = 'ID')
    
    RT_df = pd.merge(RT_df, RT_congruent_day2, on = 'ID')
    RT_df = pd.merge(RT_df, RT_incongruent_day2, on = 'ID')
    RT_df = pd.merge(RT_df, RT_randomdtt_day2, on = 'ID')
    
    RT_df['RT_diff_stt_day1'] = RT_df['RT_stt_rand_day1'] - RT_df['RT_stt_seq_day1']
    RT_df['RT_diff_stt_day2'] = RT_df['RT_stt_rand_day2'] - RT_df['RT_stt_seq_day2']
    
    return RT_df

def get_sociopsy_df():
    sociopsy_df = pd.read_csv('/home/sascha/Desktop/vbm_torch/behav_data/sociopsy_data.csv')
    sociopsy_df.loc[sociopsy_df['handedness'] == 'righthanded\xa0(0)', 'handedness'] = 0
    sociopsy_df.loc[sociopsy_df['handedness'] == 'lefthanded\xa0(1)', 'handedness'] = 1
    sociopsy_df.loc[sociopsy_df['handedness'] == 'ambidextrous\xa0(2)', 'handedness'] = 2
    
    sociopsy_df.loc[sociopsy_df['handedness'] == 'female\xa0(0)', 'handedness'] = 0
    sociopsy_df.loc[sociopsy_df['handedness'] == 'male\xa0(1)', 'handedness'] = 1
    sociopsy_df.loc[sociopsy_df['handedness'] == 'other\xa0(2)', 'handedness'] = 2
    
    return sociopsy_df