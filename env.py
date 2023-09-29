#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:16:01 2023

@author: sascha
"""

import torch
import ipdb
import scipy

class env():
    
    def __init__(self, agent, rewprobs, matfile_dir):
        self.agent = agent
        self.rewprobs = torch.tensor(rewprobs)
        self.matfile_dir = matfile_dir
        
        self.choices = []
        self.outcomes = []
        
    def load_matfiles(self, matfile_dir, blocknr, blocktype, sequence = 1):
        "blocknr : num. of block of given type (not same as Blockidx in experiment)"
        "sequence : 1 or 2 (2 means mirror sequence)"
        
        if sequence == 2:
            prefix = "mirror_"
        else:
            prefix = ""
                        
        if blocktype == 0:
            "sequential"
            mat = scipy.io.loadmat(matfile_dir + prefix +'trainblock' + str(blocknr+1) + '.mat')
        elif blocktype == 1:
            "random"
            mat = scipy.io.loadmat(matfile_dir + prefix +'random' + str(blocknr+1) + '.mat')
        else:
            raise Exception("Problem with los blocktypos.")

        seq = mat['sequence'][0]
        seq_no_jokers = mat['sequence_without_jokers'][0]
        
        "----- Determine congruent/ incongruent jokers ------"
        # -1 no joker
        # 0 random 
        # 1 congruent 
        # 2 incongruent
        
        "---- Map Neutral Jokers to 'No Joker' ---"
        seq_noneutral = [t if t != 14 and t != 23 else 1 for t in seq]    
        
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
        
    def run(self, block_order = 1, sequence = 1):
        """
        Parameters
        ----------
        
        block_order : 1 or 2
            Which block order to use (in case of Context =="all")
            
        sequence : 1 or 2
            2 means mirror sequence
            
        """    
        
        Context = "all"
        num_blocks = self.agent.num_blocks
        
        if num_blocks > 1:
            tb_idxs = [1,3,5,6,8,10,12] # tb idxs for block_order == 1
            rand_idxs = [0,2,4,7,9,11,13] # random idxs for block_order == 1
            
            if Context == "all":
                blocktype = torch.ones((num_blocks, 480))*-1;
                
                if block_order == 1:
                    blocktype[tb_idxs[0:num_blocks//2], :] = 0 # fixed sequence condition
                    blocktype[rand_idxs[0:num_blocks//2], :] = 1 # random condition
                                
                elif block_order == 2:
                    blocktype[tb_idxs[0:num_blocks//2], :] = 1
                    blocktype[rand_idxs[0:num_blocks//2], :] = 0
                
            elif Context == "tb":
                blocktype = torch.chararray((num_blocks, 480));
                
                blocktype[:, :, :] = 0
        
            elif Context == "random":
                blocktype = torch.chararray((num_blocks, 480));
                
                blocktype[:, :, :] = 1
                
        elif num_blocks == 1:
            blocktype = torch.ones((num_blocks, 480))*-1;
            if block_order == 1:
                blocktype[:, :] = 1 # random condition
                            
            elif block_order == 2:
                blocktype[:, :] = 0 # sequential condition
                
        block_no = -100*torch.ones((num_blocks, 480), dtype=torch.int8); # The index of the block in the current experiment
    
        tb_block = 0
        random_block = 0   
        self.data = {"trialsequence": [], "blocktype": [], "jokertypes": [], "blockidx": []}
        for block in range(num_blocks):
            "New block!"
            self.data["trialsequence"].append([-1])
            self.data["jokertypes"].append([-1])
            self.data["blockidx"].append([block])
            self.data["blocktype"].append(["n"])
            block_no[block, :] = block # The index of the block in the current experiment
            
            current_blocktype = blocktype[block, 0]
            
            if current_blocktype == 0:
                "Sequential block"
                seq_matlab, seq_no_jokers_matlab, jokertypes = self.load_matfiles(self.matfile_dir, tb_block, current_blocktype, sequence = sequence)
                self.data["blocktype"].extend([["s"]]*len(seq_matlab))
                tb_block += 1
                
            elif current_blocktype == 1:
                "Random block"
                seq_matlab, seq_no_jokers_matlab, jokertypes = self.load_matfiles(self.matfile_dir, random_block, current_blocktype, sequence = sequence)
                self.data["blocktype"].extend([["r"]]*len(seq_matlab))
                random_block += 1
                
            else:
                raise Exception("Problem with los blocktypos.")
            
            self.data["trialsequence"].extend([[s] for s in seq_matlab])
            self.data["jokertypes"].extend([[j] for j in jokertypes])
            self.data["blockidx"].extend([[block]]*480)
        
        t_day1 = -1
        t_day2 = -1
        for tau in range(len(self.data["trialsequence"])):
            trial = self.data["trialsequence"][tau][0]
            blocktype = self.data["blocktype"][tau][0]
                      
            if self.data["blockidx"][tau][0] <= 5:
                day = 1
                
            elif self.data["blockidx"][tau][0] > 5:
                day = 2
                
            else:
                raise Exception("Da isch a Fehla!")
                
            if self.data["blockidx"][tau][0] <= 1:
                exp_part = 1
                
            elif self.data["blockidx"][tau][0] > 1 and self.data["blockidx"][tau][0] <= 5:
                exp_part = 2
                
            elif self.data["blockidx"][tau][0] > 5:
                exp_part = 3
                
            else:
                raise Exception("Da isch a Fehla!")
            
            if trial == -1:
                "Beginning of a new block"
                self.agent.update(torch.tensor([-1]), torch.tensor([-1]), torch.tensor([-1]), day=day, trialstimulus = trial, t = 0, exp_part = exp_part)
                
                self.choices.append(torch.tensor([-1]))
                self.outcomes.append(torch.tensor([-1]))
                
            else:
                current_choice = self.agent.choose_action(trial, day)
                outcome = torch.bernoulli(self.rewprobs[current_choice])
                self.choices.append(torch.tensor([current_choice.item()]))
                self.outcomes.append(torch.tensor([outcome.item()]))
                    
                if day == 1:
                    if trial > 10:
                        t_day1 += 1
                    self.agent.update(torch.tensor([current_choice]), torch.tensor([outcome]), [blocktype], day=day, trialstimulus = trial, t = t_day1, exp_part = exp_part)
                    
                elif day == 2:
                    if trial > 10:
                        t_day2 += 1
                    self.agent.update(torch.tensor([current_choice]), torch.tensor([outcome]), [blocktype], day=day, trialstimulus = trial, t = t_day2, exp_part = exp_part)
                