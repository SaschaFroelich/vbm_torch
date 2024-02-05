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

import models_torch as models
import utils

def get_ddm_data(preproc_file = 'behav_data/preproc_data.p', k=4, save = 0):
    '''
    
    
    Parameters
    ----------
    k : int
        Dirichlet counter for sequences.
        
    save : bool, optional
        Save to file yes or no. The default is 0.

    Returns
    -------
    df_all : DataFrame
        1-indexed.

    '''

    print("Computing repvalues for DDM.")
    
    _, expdata_df = pickle.load(open(f"{preproc_file}", "rb" ))
    
    df_all = pd.DataFrame()
    
    for agentid in expdata_df['ID'].unique():
        
        agent_data = expdata_df[expdata_df['ID'] == agentid].copy()
        group = agent_data['group'].unique()
        
        if group == 0 or group == 1:
            matfile_dir = './matlabcode/clipre/'
            
        elif group == 2 or group == 3:
            matfile_dir = './matlabcode/clipre/mirror_'
            
        else:
            raise Exception("No group specified.")
        
        '''
        -1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])
        -2 in seq_counter for errors
        Dimensions are [blocktypes, pppchoice, ppchoice, pchoice, choice, agent]
        blocktypes: 0/1 : sequential/ random
        '''
        seq_counter = k /  4 * np.ones((2, 6, 6, 6, 6))
        
        repvals = [[0.25, 0.25, 0.25, 0.25]]
        pppchoice = -1
        ppchoice = -1
        pchoice = -1
        
        for tt in range(len(agent_data["choices"])):
            blocktype = agent_data["blocktype"][tt]
            choice = agent_data["choices"][tt]

            "----- Update sequence counters -----"
            if blocktype != -1:
                seq_counter[agent_data["blocktype"][tt]][pppchoice, ppchoice, pchoice, choice] += 1
                 
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
    
        agent_data["repvals"] = repvals[0: -1]

        df = pd.DataFrame(data=agent_data)
        # df['trialsequence'] = df['trialsequence'].map(lambda x: x[0])
        # df['trialsequence_no_jokers'] = df['trialsequence_no_jokers'].map(lambda x: x[0])

        df = df.drop(labels=df.loc[df["trialsequence"]==-1].index, axis = 0)
        
        # "Add participant index PB"
        # PB = [pb+1]*len(df["trialsequence"])
        # df["PB"] = PB
        
        "Add participant-specific trial index"
        Trialidx = range(1,len(df["trialsequence"])+1)
        df["trialidx"] = Trialidx
        
        "Change to 1-indexing for R."
        # df["jokertypes"] = df["jokertypes"].map(lambda x: x[0])
        # df["correct"] = df["correct"].map(lambda x: x[0])
        # df["blocktype"] = df["blocktype"].map(lambda x: x[0])
        # df["outcomes"] = df["outcomes"].map(lambda x: x[0])
        # df["choices_GD"] = df["choices_GD"].map(lambda x: x[0])
        
        df["group"] = df["group"].map(lambda x: x+1)
        df["blockidx"] = df["blockidx"].map(lambda x: x+1)
        df["choices"] = df["choices"].map(lambda x: x+1 if x > -1 else x)

        df_all = pd.concat((df_all, df))
    
    df_all["day"] = df_all["blockidx"].map(lambda x: 1 if x <= 6 else 2)
    "1-indexing for R"
    df_all['ag_idx'] = df_all['ag_idx'].map(lambda x: x+1)
    
    if save:
        print("Saving to file.")
        df_all.to_csv("/home/sascha/Downloads/Data_DDM_heute.csv")
        
    """Next steps 
    6) Make sure the block orders are correct
    """
    print("Make sure the block orders are correct")
    return df_all
    