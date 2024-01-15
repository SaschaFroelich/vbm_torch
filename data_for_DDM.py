#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:12:04 2023

@author: sascha
"""

import ipdb
import pickle
import pandas as pd

import sys
import glob

import numpy as np

import models_torch_paper as models
import utils

def prepare_ddm_data(data_dir, file_day1, group, pb, kk):
    '''
        blocktype : int
            0/1 : sequential/ random block
    '''
    # raise Exception("Code is out of date.")
    "INPUT: group (0-indexed)"
    "OUTPUT: 1-indexed DataFrame (1 because of R)"
    
    data, prolific_ID = utils.get_participant_data(file_day1, group, data_dir)
    
    data["group"] = [group]*len(data["trialsequence"])
    data["ID"] = [prolific_ID]*len(data["group"])
    
    if group == 0 or group == 1:
        rewprobs = [0.8, 0.2, 0.2, 0.8]
        
    elif group == 2 or group == 3:
        rewprobs = [0.2, 0.8, 0.8, 0.2]
    
    # data["jokertype"] = [] # 0 cong or 1 inc or 3 rand, -1 Blockanfang, nan STT
    # data["choices_GD"] = [] # 0 suboptimal 1 optimal, -10 error, -1 Blockanfang, NaN for STT
    
    # for tt in range(len(data["trialsequence"])):
    #     if data["trialsequence"][tt][0] == -1:
    #         data["jokertype"].append(-1)
    #         data["choices_GD"].append(-1)
            
    #     elif data["trialsequence"][tt][0] > 10:
    #         # DTT
            
    #         "--- Determine jokertype ---"
    #         if data["blocktype"][tt][0] == 0:
                
    #             if rewprobs[data["trialsequence_no_jokers"][tt][0]-1] == 0.8:
    #                 data["jokertype"].append(0)
    #             elif rewprobs[data["trialsequence_no_jokers"][tt][0]-1] == 0.2:
    #                 data["jokertype"].append(1)
    #             else:
    #                 raise Exception("Fehla!")
                    
    #         elif data["blocktype"][tt][0] == 1:
    #             "random"
    #             data["jokertype"].append(3)
                
    #         else:
    #             raise Exception("Fehla!")
                
    #         "--- --- --- ---"
            
    #         if data["choices"][tt][0] == -10:
    #             # Error
    #             data["choices_GD"].append(-10)
                
    #         elif data["choices"][tt][0] == -1:
    #             data["choices_GD"].append(-1)
                
    #         else:
    #             if rewprobs[data["choices"][tt][0]] == 0.8:
    #                 data["choices_GD"].append(1)
                    
    #             elif rewprobs[data["choices"][tt][0]] == 0.2:
    #                 data["choices_GD"].append(0)
                    
    #             else:
    #                 raise Exception("Fehla!")
                
    #     elif data["trialsequence"][tt][0] > 0 and data["trialsequence"][tt][0] < 10:
    #         # STT
    #         data["jokertype"].append(np.nan)
    #         data["choices_GD"].append(np.nan)
            
    #     else:
    #         raise Exception("Fehla!")

    #     dfgh

    '''
    -1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])
    -2 in seq_counter for errors
    Dimensions are [blocktypes, pppchoice, ppchoice, pchoice, choice, agent]
    blocktypes: 0/1 : sequential/ random
    '''
    seq_counter = kk /  4 * np.ones((2, 6, 6, 6, 6))
    
    repvals = [[0.25, 0.25, 0.25, 0.25]]
    print(repvals)
    pppchoice = -1
    ppchoice = -1
    pchoice = -1
        
    for tt in range(len(data["choices"])):
    
        #if tt == 482:
        #   ipdb.set_trace()
        
        blocktype = data["blocktype"][tt][0]
        choice = data["choices"][tt][0]
        
        "----- Update sequence counters -----"
        if blocktype != -1:
            seq_counter[data["blocktype"][tt][0]][pppchoice, ppchoice, pchoice, choice] += 1
             
        else:
            pppchoice = -1
            ppchoice = -1
            pchoice = -1
            choice = -1
    
        "----- Update repetition values rep -----"
        seqs_sum = seq_counter[blocktype, 
                                    ppchoice, 
                                    pchoice, 
                                    choice, 
                                    0:4].sum(axis=-1)
        
        new_rows = seq_counter[blocktype, 
                                    ppchoice, 
                                    pchoice, 
                                    choice, 
                                    0:4] / seqs_sum[..., None]

        repvals.append(list(new_rows))
        
        pppchoice = ppchoice
        ppchoice = pchoice 
        pchoice = choice

    data["repvals"] = repvals[0:-1]
    
    df = pd.DataFrame(data=data)
    df['trialsequence'] = df['trialsequence'].map(lambda x: x[0])
    df['trialsequence_no_jokers'] = df['trialsequence_no_jokers'].map(lambda x: x[0])
    
    df = df.drop(labels=df.loc[df["trialsequence"]==-1].index,axis = 0)
    
    "Add participant index PB"
    PB = [pb+1]*len(df["trialsequence"])
    df["PB"] = PB
    
    "Add participant-specific trial index"
    Trialidx = range(1,len(df["trialsequence"])+1)
    df["trialidx"] = Trialidx
    
    "Change to 1-indexing for R."
    df["jokertypes"] = df["jokertypes"].map(lambda x: x[0])
    df["correct"] = df["correct"].map(lambda x: x[0])
    df["blocktype"] = df["blocktype"].map(lambda x: x[0])
    df["outcomes"] = df["outcomes"].map(lambda x: x[0])
    df["choices_GD"] = df["choices_GD"].map(lambda x: x[0])
    
    df["group"] = df["group"].map(lambda x: x+1)
    df["blockidx"] = df["blockidx"].map(lambda x: x[0]+1)
    df["choices"] = df["choices"].map(lambda x: x[0]+1 if x[0] > -1 else x[0])
    
    return df

#%%

def get_ddm_data(k=4, save = 0):
    '''
    

    Parameters
    ----------
    k : int
        Dirichlet counter for sequences.
        
    save : bool, optional
        Save to file yes or no. The default is 0.

    Returns
    -------
    df_all : TYPE
        DESCRIPTION.

    '''

    df_all = pd.DataFrame()
    
    # data_dir = "/home/sascha/Desktop/vbm_torch/behav_data/"
    data_dir = "/home/sascha/proni/AST/AST2/AST2RT_Online/data/"
    
    pb = -1
    for group in range(4):
        files_day1 = glob.glob(data_dir + "Grp%d/csv/*Tag1*.mat"%(group+1))
        
        for file1 in files_day1:
            "Loop over participants"
            pb += 1
    
            df = prepare_ddm_data(data_dir, file1, group, pb, kk = k)
            df_all = pd.concat((df_all, df))
        
    "Add days column"
    df_all["day"] = df["blockidx"].map(lambda x: 1 if x <= 6 else 2)
    
    print("printing ID")
    print(df_all["ID"])
    if save:
        print("Saving to file.")
        df_all.to_csv("Data_DDM_heute.csv")
        
    """Next steps 
    6) Make sure the block orders are correct
    """
    print("Make sure the block orders are correct")
    
    return df_all
    