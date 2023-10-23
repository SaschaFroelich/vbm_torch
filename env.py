#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:16:01 2023

@author: sascha
"""


import torch
import ipdb
import scipy

import pyro
import pyro.distributions as dist

class Env():
    
    def __init__(self, 
                 agent, 
                 matfile_dir):
        '''
        Environment is used for simulating data.
        One separate environment is instantiated for each agent, with no multiple agents inside.
        
        Methods
        ----------
        load_matfiles()
        run()
        run_loop()
        
        Parameters
        ----------
        agent : obj
            Instantiation of the agent class.
            May be instantiated for multiple parallel agents.
            
        matfile_dir : TYPE
            DESCRIPTION.

            
        Returns
        -------
        None.

        '''
        
        assert(agent.num_agents == 1)
        self.agent = agent
        self.matfile_dir = matfile_dir
        
        self.choices = []
        self.outcomes = []
        
    def load_matfiles(self, 
                      matfile_dir, 
                      blocknr, 
                      blocktype, 
                      sequence = 1):
        
        "blocknr : num. of block of given type (not same as blockidx in experiment)"
        
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
        
    def run(self, 
            sequence = 1, 
            blockorder = 1):
        '''
        
        Parameters
        ----------
        sequence : int
            Whether sequence or mirror sequence.
            1/2 : sequence/ mirror sequence

        blockorder : int
            Which blockorder
            1/2 : RSRSRS SRSRSRSR / SRSRSR RSRSRSRS

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        self.data = {}
        if sequence == 1:
            self.data['rewprobs'] = torch.tensor([0.8, 0.2, 0.2, 0.8])
            
        elif sequence == 2:
            self.data['rewprobs'] = torch.tensor([0.2, 0.8, 0.8, 0.2])
            
        else:
            raise Exception("Fehla!")
        
        Context = "all"
        num_blocks = self.agent.num_blocks
        assert(num_blocks == 14)
        
        tb_idxs = [1,3,5,6,8,10,12] # tb idxs for blockorder == 1
        rand_idxs = [0,2,4,7,9,11,13] # random idxs for blockorder == 1
        
        blocktype = torch.ones((num_blocks, 480))*-1;
        
        if blockorder == 1:
            blocktype[tb_idxs[0:num_blocks//2], :] = 0 # fixed sequence condition
            blocktype[rand_idxs[0:num_blocks//2], :] = 1 # random condition
                        
        elif blockorder == 2:
            blocktype[tb_idxs[0:num_blocks//2], :] = 1
            blocktype[rand_idxs[0:num_blocks//2], :] = 0
    
        block_no = -100*torch.ones((num_blocks, 480), dtype=torch.int8); # The index of the block in the current experiment

        tb_block = 0
        random_block = 0  
        self.data['trialsequence'] = []
        self.data['blocktype'] = []
        self.data['jokertypes'] = []
        self.data['blockidx'] = []
        for block in range(num_blocks):
            "New block!"
            self.data["trialsequence"].append([-1])
            self.data["jokertypes"].append([-1])
            self.data["blockidx"].append([block])
            self.data["blocktype"].append([-1])
            block_no[block, :] = block # The index of the block in the current experiment
            
            current_blocktype = blocktype[block, 0]
            
            if current_blocktype == 0:
                "Sequential block"
                seq_matlab, seq_no_jokers_matlab, jokertypes = \
                    self.load_matfiles(self.matfile_dir, 
                                       tb_block, 
                                       current_blocktype, 
                                       sequence = sequence)
                    
                self.data["blocktype"].extend([[0]]*len(seq_matlab))
                tb_block += 1
                
            elif current_blocktype == 1:
                "Random block"
                seq_matlab, seq_no_jokers_matlab, jokertypes = \
                    self.load_matfiles(self.matfile_dir, 
                                       random_block, 
                                       current_blocktype, 
                                       sequence = sequence)
                    
                self.data["blocktype"].extend([[1]]*len(seq_matlab))
                random_block += 1
                
            else:
                raise Exception("Problem with los blocktypos.")
            
            self.data["trialsequence"].extend([[s] for s in seq_matlab])
            self.data["jokertypes"].extend([[j] for j in jokertypes])
            self.data["blockidx"].extend([[block]]*480)
        
        self.run_loop(self.agent, self.data, 1, infer = 0)
        
    def run_loop(self, agent, data, num_particles, infer = 0):
        '''

        Parameters
        ----------
        agent : TYPE
            DESCRIPTION.
        data : TYPE
            DESCRIPTION.
        num_particles : TYPE
            DESCRIPTION.
        infer : bool, optional
            0/1 simulate data/ infer model parameters

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        num_trials = len(data["trialsequence"])
        t = -1
        for tau in pyro.markov(range(num_trials)):
    
            trial = torch.tensor(data["trialsequence"][tau])
            blocktype = torch.tensor(data["blocktype"][tau])
            
            if all([data["blockidx"][tau][i] <= 5 for i in range(agent.num_agents)]):
                day = 1
                
            elif all([data["blockidx"][tau][i] > 5 for i in range(agent.num_agents)]):
                day = 2
                
            else:
                raise Exception("Da isch a Fehla!")
            
            if all(trial == -1):
                "Beginning of new block"
                agent.update(torch.tensor([-1]*agent.num_agents), 
                                torch.tensor([-1]*agent.num_agents), 
                                torch.tensor([-1]*agent.num_agents), 
                                day = day, 
                                trialstimulus = trial)
                
                if not infer:
                    self.choices.append(torch.tensor([-1], requires_grad = False))
                    self.outcomes.append(torch.tensor([-1], requires_grad = False))
                
            else:
                if infer:
                    current_choice = data["choices"][tau]
                    outcome = data["outcomes"][tau]
            
                else:
                    "Simulation"
                    assert(torch.is_tensor(trial))
                    current_choice = torch.tensor(agent.choose_action(trial, day).item(), requires_grad = False)
                    outcome = torch.bernoulli(data['rewprobs'][current_choice])
                    self.choices.append(torch.tensor([current_choice.item()]))
                    self.outcomes.append(torch.tensor([outcome.item()]))
            
                if infer and any(trial > 10):
                    "Dual-Target Trial"
                    t+=1
                    option1, option2 = agent.find_resp_options(trial)
                    # print("MAKE SURE EVERYTHING WORKS FOR ERRORS AS WELL")
                    # probs should have shape [num_particles, num_agents, nactions], or [num_agents, nactions]
                    # RHS comes out as [1, n_actions] or [num_particles, n_actions]
                    
                    "==========================================="
                    probs = agent.compute_probs(trial, day)
                    "==========================================="
                    
                    choices = (current_choice != option1).type(torch.int).broadcast_to(num_particles, agent.num_agents)
                    obs_mask = (current_choice != -2).broadcast_to(num_particles, agent.num_agents)
    
                    if any(current_choice == -10):
                        raise Exception("Fehla!")
                        
                    if torch.any(obs_mask == False):
                        dfgh
    
                "Update (trial == -1 means this is the beginning of a block -> participants didn't see this trial')"
                agent.update(current_choice, 
                                outcome, 
                                blocktype, 
                                day = day, 
                                trialstimulus = trial)
    
                if infer and any(trial > 10):
                    "STT are 0.5 0.5"
                    "errors are obs_masked"
                    pyro.sample('res_{}'.format(t), 
                                dist.Categorical(probs = probs),
                                obs = choices,
                                obs_mask = obs_mask)
            
        # t_day1 = -1
        # t_day2 = -1
        # for tau in range(len(self.data["trialsequence"])):
        #     trial = torch.tensor(self.data["trialsequence"][tau])
        #     blocktype = torch.tensor(self.data["blocktype"][tau])
                      
        #     if self.data["blockidx"][tau][0] <= 5:
        #         day = 1
                
        #     elif self.data["blockidx"][tau][0] > 5:
        #         day = 2
                
        #     else:
        #         raise Exception("Da isch a Fehla!")
                
        #     if self.data["blockidx"][tau][0] <= 1:
        #         exp_part = 1
                
        #     elif self.data["blockidx"][tau][0] > 1 and self.data["blockidx"][tau][0] <= 5:
        #         exp_part = 2
                
        #     elif self.data["blockidx"][tau][0] > 5:
        #         exp_part = 3
                
        #     else:
        #         raise Exception("Da isch a Fehla!")
            
        #     if trial == -1:
        #         "Beginning of a new block"
        #         self.agent.update(torch.tensor([-1]), 
        #                           torch.tensor([-1]), 
        #                           torch.tensor([-1]), 
        #                           day=day, 
        #                           trialstimulus = trial, 
        #                           t = 0, 
        #                           exp_part = exp_part)
                
        #         self.choices.append(torch.tensor([-1]))
        #         self.outcomes.append(torch.tensor([-1]))
                
        #     else:
        #         current_choice = self.agent.choose_action(trial, day)
        #         outcome = torch.bernoulli(self.rewprobs[current_choice])
        #         self.choices.append(torch.tensor([current_choice.item()]))
        #         self.outcomes.append(torch.tensor([outcome.item()]))
                    
        #         if day == 1:
        #             if trial > 10:
        #                 t_day1 += 1
        #             self.agent.update(torch.tensor([current_choice]), 
        #                               torch.tensor([outcome]), 
        #                               blocktype, 
        #                               day = day, 
        #                               trialstimulus = trial, 
        #                               t = t_day1, 
        #                               exp_part = exp_part)
                    
        #         elif day == 2:
        #             if trial > 10:
        #                 t_day2 += 1
        #             self.agent.update(torch.tensor([current_choice]), 
        #                               torch.tensor([outcome]), 
        #                               blocktype, 
        #                               day = day, 
        #                               trialstimulus = trial, 
        #                               t = t_day2,
        #                               exp_part = exp_part)
                