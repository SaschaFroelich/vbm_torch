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

def prepare_ddm_data(data_dir, file_day1, group, pb, kk):
    raise Exception("OCode is out of date.")
    "INPUT: group (0-indexed)"
    "OUTPUT: 1-indexed DataFrame (1 because of R)"
    
    data, prolific_ID = utils.get_participant_data(file_day1, group, data_dir, remote = 0)
    
    data["group"] = [group]*len(data["Trialsequence"])
    
    if group == 0 or group == 1:
        rewprobs = [0.8, 0.2, 0.2, 0.8]
        
    elif group == 2 or group == 3:
        rewprobs = [0.2, 0.8, 0.8, 0.2]
    
    data["Jokertype"] = [] # 0 cong or 1 inc or 3 rand, -1 Blockanfang, nan STT
    data["Choice_GD"] = [] # 0 suboptimal 1 optimal, -10 error, -1 Blockanfang, NaN for STT
    
    for tt in range(len(data["Trialsequence"])):
        if data["Trialsequence"][tt][0] == -1:
            data["Jokertype"].append(-1)
            data["Choice_GD"].append(-1)
            
        elif data["Trialsequence"][tt][0] > 10:
            # DTT
            
            "--- Determine Jokertype ---"
            if data["Blocktype"][tt][0] == 's':
                
                if rewprobs[data["Trialsequence no jokers"][tt][0]-1] == 0.8:
                    data["Jokertype"].append(0)
                elif rewprobs[data["Trialsequence no jokers"][tt][0]-1] == 0.2:
                    data["Jokertype"].append(1)
                else:
                    raise Exception("Fehla!")
                    
            elif data["Blocktype"][tt][0] == 'r':
                data["Jokertype"].append(3)
                
            else:
                raise Exception("Fehla!")
                
            "--- --- --- ---"
            
            if data["Choices"][tt][0] == -10:
                # Error
                data["Choice_GD"].append(-10)
                
            elif data["Choices"][tt][0] == -1:
                data["Choice_GD"].append(-1)
                
            else:
                if rewprobs[data["Choices"][tt][0]] == 0.8:
                    data["Choice_GD"].append(1)
                    
                elif rewprobs[data["Choices"][tt][0]] == 0.2:
                    data["Choice_GD"].append(0)
                    
                else:
                    raise Exception("Fehla!")
                
        elif data["Trialsequence"][tt][0] > 0 and data["Trialsequence"][tt][0] < 10:
            # STT
            data["Jokertype"].append(np.nan)
            data["Choice_GD"].append(np.nan)
            
        else:
            raise Exception("Fehla!")
    
    "--- Compute rep-values ---"
    seq_counter_tb = {}
    seq_counter_r = {}
    "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
    "-10 in seq_counter for errors)"
    for i in [-10,-1,0,1,2,3]:
        for j in [-10,-1,0,1,2,3]:
            for k in [-10,-1,0,1,2,3]:
                for l in [-10,-1,0,1,2,3]:
                    seq_counter_tb[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = kk/4
                    seq_counter_r[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = kk/4
    
    repvals = [[0.25, 0.25, 0.25, 0.25]]
    
    pppchoice = -1
    ppchoice = -1
    pchoice = -1
        
    for tt in range(len(data["Choices"])):
    
        #if tt == 482:
        #   ipdb.set_trace()
        
        choice = data["Choices"][tt][0].item()
        
        "----- Update sequence counters -----"
        if data["Blocktype"][tt][0] == "s":
            "Sequential Block"
            seq_counter_tb[str(pppchoice) + "," + \
                             str(ppchoice) + "," + str(pchoice) + "," + str(choice)] += 1
        
        elif data["Blocktype"][tt][0] == "r":
            "Random Block"
            seq_counter_r[str(pppchoice) + "," + \
                             str(ppchoice) + "," + str(pchoice) + "," + str(choice)] += 1
             
        elif data["Blocktype"][tt][0] == "n":
            pppchoice = -1
            ppchoice = -1
            pchoice = -1
            choice = -1
    
        else:
            raise Exception("Da isch a Fehla aba ganz a gwaldiga!")
    
        "----- Update repetition values rep -----"
        prev_seq = [str(ppchoice) + "," + str(pchoice) + "," + str(choice)]
        
        new_row = [0., 0. ,0. ,0.]
        for aa in range(4):
        
            if data["Blocktype"][tt][0] == 's':
                for aa in range(4):
                    new_row[aa] = seq_counter_tb[prev_seq[0] + "," + str(aa)] / \
                            (seq_counter_tb[prev_seq[0] + "," + "0"] + seq_counter_tb[prev_seq[0] + "," + "1"] + \
                             seq_counter_tb[prev_seq[0] + "," + "2"] + seq_counter_tb[prev_seq[0] + "," + "3"])
    
            elif data["Blocktype"][tt][0] == 'r':
                for aa in range(4):
                    new_row[aa] = seq_counter_r[prev_seq[0] + "," + str(aa)] / \
                            (seq_counter_r[prev_seq[0] + "," + "0"] + seq_counter_r[prev_seq[0] + "," + "1"] + \
                             seq_counter_r[prev_seq[0] + "," + "2"] + seq_counter_r[prev_seq[0] + "," + "3"])
                                
            elif data["Blocktype"][tt][0] == 'n':
                new_row = [0.25, 0.25, 0.25, 0.25]
                
            else:
                raise Exception("Da isch a Fehla aba ganz agwaldiga!")
        
        repvals.append(new_row)
        
        pppchoice = ppchoice
        ppchoice = pchoice 
        pchoice = choice
        
    repvals = repvals[0:-1]
    
    data["repvals"] = repvals
    
    "--- Test repvals ---"
    newagent = models.Vbm(omega=0.5, dectemp=2., lr=0., k=4, Q_init=[0.4, 0., 0., 0.4], num_blocks=14)
    infer = models.SingleInference(newagent, data)
    loss, params = infer.infer_posterior(iter_steps=1, num_particles = 1)
    
    df = pd.DataFrame(data=data)
    
    df['Trialsequence'] = df['Trialsequence'].map(lambda x: x.item())
    df['Trialsequence no jokers'] = df['Trialsequence no jokers'].map(lambda x: x.item())
    
    df = df.drop(labels=df.loc[df["Trialsequence"]==-1].index,axis = 0)
    
    "Add participant index PB"
    PB = [pb+1]*len(df["Trialsequence"])
    df["PB"] = PB
    
    "Add participant-specific trial index"
    Trialidx = range(1,len(df["Trialsequence"])+1)
    df["Trialidx"] = Trialidx
    
    df["group"] = df["group"].map(lambda x: x+1)
    df["Blockidx"] = df["Blockidx"].map(lambda x: x+1)
    
    df["Choices"] = df["Choices"].map(lambda x: x+1 if x > -1 else x)
    
    return df

#%%

def get_ddm_data(k, save = 0):

    df_all = pd.DataFrame()
    
    data_dir = "/home/sascha/1TB/TRR/AST_repo/Analyses/clipre/Data/Online_Erhebung/"
    
    pb = -1
    for group in range(4):
        files_day1 = glob.glob(data_dir + "Grp%d/csv/*Tag1*.mat"%(group+1))
        
        for file1 in files_day1:
            "Loop over participants"
            pb += 1
    
            df = prepare_ddm_data(data_dir, file1, group, pb, kk = k)
            df_all = pd.concat((df_all, df))
        
    "Add days column"
    df_all["Day"] = df["Blockidx"].map(lambda x: 1 if x <= 6 else 2)
    
    if save:
        df_all.to_csv("Data_DDM.csv")
        
    """Next steps 
    6) Make sure the block orders are correct
    """
    print("Make sure the block orders are correct")
    
    return df_all
    