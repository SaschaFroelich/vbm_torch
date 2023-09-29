#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:16:01 2023

@author: sascha
"""

import torch
import ipdb
import scipy

class Env():
    
    def __init__(self, agent, rewprobs, matfile_dir):
        self.agent = agent
        self.rewprobs = torch.tensor(rewprobs)
        self.matfile_dir = matfile_dir
        
        self.choices = []
        self.outcomes = []
        
    def load_matfiles(self, matfile_dir, blocktype, sequence = [1]):
        '''
        Loads sequences of trials from .mat-files.
        Manipulates             self.data["trialsequence"].extend([[s] for s in seq_matlab])
                    self.data["jokertypes"].extend([[j] for j in jokertypes])
                    self.data["blockidx"]
    
                Parameters:
                        matfile_dir : directory in which the .mat-files are stored
                        blocktype (list of length num_agents): -1 new block trial, 0 fixed sequence condition, 1 random condition
                        sequence (list of length num_agents) : 1 or 2 (2 for mirror sequence)
    
                Returns:
                        None
        '''
        
        num_agents = len(sequence)
        trialsequence_all = [[] for _ in range(num_agents)] # contains num_agents lists
        jokertypes_all = [[] for _ in range(num_agents)]
        blockidx_all = [[] for _ in range(num_agents)] 
        blocktype_all = [[] for _ in range(num_agents)]

        for ag in range(num_agents):

            tb_block = 0
            rand_block = 0
            for blockidx in range(self.agent.num_blocks):
                trialsequence_all[ag].append(-1)
                jokertypes_all[ag].append(-1)
                blockidx_all[ag].append(-1)
                blockidx_all[ag].extend([blockidx]*480)
                blocktype_all[ag].append(-1)
                blocktype_all[ag].extend(blocktype[ag, blockidx, :])
                
                if sequence[ag] == 2:
                    prefix = "mirror_"
                else:
                    prefix = ""
                                
                if blocktype[ag, blockidx, 0] == 0:
                    "sequential"
                    mat = scipy.io.loadmat(matfile_dir + prefix +'trainblock' + str(tb_block+1) + '.mat')
                    tb_block += 1
                    
                elif blocktype[ag, blockidx, 0] == 1:
                    "random"
                    mat = scipy.io.loadmat(matfile_dir + prefix +'random' + str(rand_block+1) + '.mat')
                    rand_block += 1
                    
                else:
                    raise Exception("Problemo with los blocktypos.")
        
                seq = mat['sequence'][0]
                seq_no_jokers = mat['sequence_without_jokers'][0]
                
                "----- Determine congruent/ incongruent jokers ------"
                # -1 no joker
                # 0 random 
                # 1 congruent 
                # 2 incongruent
                
                "---- Map Neutral Jokers to 'No Joker' ---"
                seq_noneutral = [t if t != 14 and t != 23 else 1 for t in seq]    
                
                if blocktype[ag, blockidx, 0] == 0:
                    "sequential"
                    jokers = [-1 if seq_noneutral[tidx]<10 else seq_no_jokers[tidx] for tidx in range(len(seq))]
                    if sequence[ag] == 1:
                        jokertypes = [j if j == -1 else 1 if j == 1 else 2 if j == 2 else 2 if j == 3 else 1 for j in jokers]
                        
                    elif sequence[ag] == 2:
                        jokertypes = [j if j == -1 else 2 if j == 1 else 1 if j == 2 else 1 if j == 3 else 2 for j in jokers]
                        
                    else:
                        raise Exception("Fehla!!")
                                
                elif blocktype[ag, blockidx, 0] == 1:
                    "random"
                    jokertypes = [-1 if seq_noneutral[tidx]<10 else 0 for tidx in range(len(seq_noneutral))]

                trialsequence_all[ag].extend(seq)
                jokertypes_all[ag].extend(jokertypes)

        "Rearrange"
        num_trials = len(trialsequence_all[0])
        self.data["trialsequence"] = [[] for _ in range(num_trials)]
        self.data["jokertypes"] = [[] for _ in range(num_trials)]
        self.data["blockidx"] = [[] for _ in range(num_trials)]
        self.data["blocktype"] = [[] for _ in range(num_trials)]
        for trial in range(num_trials):
            for agent in range(num_agents):
                self.data["trialsequence"][trial].append(trialsequence_all[agent][trial])
                self.data["jokertypes"][trial].append(jokertypes_all[agent][trial])
                self.data["blockidx"][trial].append(blockidx_all[agent][trial])
                self.data["blocktype"][trial].append(blocktype_all[agent][trial])

    def run(self, block_order = [1], sequence = [1]):
        '''
        Prepares self.data for simulation and runs the environment to simulate agent behaviour.
        Saves data in self.data (dictionary).
    
                Parameters:
                        block_order (list of length num_agents): block orders (1 or 2) for the different agents
                            Which block order to use (in case of Context =="all")
                            
                        sequence (list of length num_agents) : list with 1s and 2s
                            2 means mirror sequence
    
                Returns:
                        None
                        
                        Saves data in self.data (dictionary)
                            
        '''
        
        "--- Preparation ---"
        
        Context = "all"
        num_blocks = self.agent.num_blocks
        num_agents = self.agent.num_agents
        
        if num_blocks != 14:
            raise Exception("Number of blocks ain't not 14, mate.")
            
        if Context != 'all':
            raise Exception("Context ain't not all, mate.")
        
        tb_idxs = [1, 3, 5, 6, 8, 10, 12] # tb idxs for block_order == 1
        rand_idxs = [0, 2, 4, 7, 9, 11, 13] # random idxs for block_order == 1
        
        blocktype = torch.ones((num_agents, num_blocks, 480))*-2;
        
        for ag in range(num_agents):
            if block_order[ag] == 1:
                blocktype[ag, tb_idxs[0:num_blocks//2], :] = 0 # fixed sequence condition
                blocktype[ag, rand_idxs[0:num_blocks//2], :] = 1 # random condition
                            
            elif block_order[ag] == 2:
                blocktype[ag, tb_idxs[0:num_blocks//2], :] = 1
                blocktype[ag, rand_idxs[0:num_blocks//2], :] = 0
            
        # if num_blocks > 1:
        #     tb_idxs = [1,3,5,6,8,10,12] # tb idxs for block_order == 1
        #     rand_idxs = [0,2,4,7,9,11,13] # random idxs for block_order == 1
            
        #     if Context == "all":
        #         blocktype = torch.ones((num_blocks, 480))*-1;
                
        #         if block_order == 1:
        #             blocktype[tb_idxs[0:num_blocks//2], :] = 0 # fixed sequence condition
        #             blocktype[rand_idxs[0:num_blocks//2], :] = 1 # random condition
                                
        #         elif block_order == 2:
        #             blocktype[tb_idxs[0:num_blocks//2], :] = 1
        #             blocktype[rand_idxs[0:num_blocks//2], :] = 0
                
        #     elif Context == "tb":
        #         blocktype = torch.tensor((num_blocks, 480));
                
        #         blocktype[:, :, :] = 0
        
        #     elif Context == "random":
        #         blocktype = torch.tensor((num_blocks, 480));
                
        #         blocktype[:, :, :] = 1
                
        # elif num_blocks == 1:
        #     blocktype = torch.ones((num_blocks, 480))*-1;
        #     if block_order == 1:
        #         blocktype[:, :] = 1 # random condition
                            
        #     elif block_order == 2:
        #         blocktype[:, :] = 0 # sequential condition
    
        tb_block = torch.zeros(num_agents)
        random_block = torch.zeros(num_agents)   
        self.data = {"trialsequence": [], 
                     "blocktype": [], 
                     "jokertypes": [], 
                     "blockidx": []}
        
        # ipdb.set_trace()
        self.load_matfiles(self.matfile_dir, 
                           blocktype, 
                           sequence = torch.tensor(sequence))
        
        # for block in range(num_blocks):
        #     "New block!"
        #     self.data["trialsequence"].append([-1]*num_agents)
        #     self.data["jokertypes"].append([-1]*num_agents)
        #     self.data["blockidx"].append([block]*num_agents)
        #     self.data["blocktype"].append([-1]*num_agents) # -1 instead of "n"
            
        #     # ipdb.set_trace()
        #     # current_blocktype = blocktype[:, block, 0]
            
        #     self.load_matfiles(self.matfile_dir, 
        #                        tb_block, 
        #                        blocktype, 
        #                        sequence = sequence)
            
            # if current_blocktype == 0:
            #     "Sequential block"
            #     seq_matlab, _, jokertypes = \
            #         self.load_matfiles(self.matfile_dir, 
            #                            tb_block, 
            #                            current_blocktype, 
            #                            sequence = sequence)
                    
            #     self.data["blocktype"].extend([[0]]*len(seq_matlab))
            #     tb_block += 1

            # elif current_blocktype == 1:
            #     "Random block"
            #     seq_matlab, _, jokertypes = \
            #         self.load_matfiles(self.matfile_dir, 
            #                            random_block, 
            #                            current_blocktype, 
            #                            sequence = sequence)
                    
            #     self.data["blocktype"].extend([[1]]*len(seq_matlab))
            #     random_block += 1
                
            # else:
            #     raise Exception("Problem with los blocktypos.")
            
            # self.data["trialsequence"].extend([[s] for s in seq_matlab])
            # self.data["jokertypes"].extend([[j] for j in jokertypes])
            # self.data["blockidx"].extend([[block]]*480)
        
        "--- Simulation ---"
        t_day1 = -1
        t_day2 = -1
        for tau in range(len(self.data["trialsequence"])):
            ipdb.set_trace()
            trial = self.data["trialsequence"][tau]
            blocktype = self.data["blocktype"][tau]
                      
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
            
            if trial[0] == -1:
                "Beginning of a new block"
                self.agent.update(torch.tensor([-1]), 
                                  torch.tensor([-1]), 
                                  torch.tensor([-1]), 
                                  day=day, 
                                  trialstimulus = trial, 
                                  t = 0, 
                                  exp_part = exp_part)
                
                self.choices.append(torch.tensor([-1]*num_agents))
                self.outcomes.append(torch.tensor([-1]*num_agents))
                
            else:
                current_choice = self.agent.choose_action(torch.tensor(trial), torch.tensor(day))
                outcome = torch.bernoulli(self.rewprobs[current_choice])
                ipdb.set_trace()
                self.choices.append(torch.tensor([current_choice.item()]))
                self.outcomes.append(torch.tensor([outcome.item()]))
                    
                if day == 1:
                    if trial > 10:
                        t_day1 += 1
                    self.agent.update(torch.tensor([current_choice]), 
                                      torch.tensor([outcome]), 
                                      [blocktype], 
                                      day=day, 
                                      trialstimulus = trial, 
                                      t = t_day1, 
                                      exp_part = exp_part)
                    
                elif day == 2:
                    if trial > 10:
                        t_day2 += 1
                    self.agent.update(torch.tensor([current_choice]), 
                                      torch.tensor([outcome]), 
                                      [blocktype], 
                                      day=day, 
                                      trialstimulus = trial, 
                                      t = t_day2, 
                                      exp_part = exp_part)
                