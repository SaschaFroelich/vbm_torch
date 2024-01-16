#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:16:01 2023

@author: sascha
"""


import utils
import torch
import ipdb
import scipy

import pyro
import pyro.distributions as dist

# torch.manual_seed(123)

class Env():
    
    def __init__(self, 
                 agent, 
                 matfile_dir):
        '''
        Environment is used for simulating data.
        One separate environment is instantiated for each agent, with no multiple agents inside.
        
        Methods
        ----------
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
        
        self.agent = agent
        self.matfile_dir = matfile_dir
        
        self.choices = []
        self.outcomes = []
        self.choices_GD = []
        
    def run(self, 
            sequence = None,
            blockorder = None):
        '''
        
        Parameters
        ----------
        sequence : list, len [num_agents]
            Whether sequence or mirror sequence.
            1/2 : sequence/ mirror sequence
            1: reward probs = [0.8, 0.2, 0.2, 0.8]
            2: reward probs = [0.2, 0.8, 0.8, 0.2]

        blockorder : list, len [num_agents]
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
        
        # assert isinstance(sequence, list), "Sequence must be a list."
        # assert isinstance(blockorder, list), "blockorder must be a list."
        # assert len(sequence) == self.agent.num_agents
        # assert len(blockorder) == self.agent.num_agents
        
        self.data = {}
        rewprobs = [[0.8, 0.2, 0.2, 0.8],
                    [0.2, 0.8, 0.8, 0.2]]
        
        self.data['rewprobs'] = torch.tensor(rewprobs)[torch.tensor(sequence)-1,:]
        
        num_blocks = self.agent.num_blocks
        
        seq_idxs = [1, 3, 5, 6, 8, 10, 12] # seq idxs for blockorder == 1
        rand_idxs = [0, 2, 4, 7, 9, 11, 13] # random idxs for blockorder == 1
        
        "New Code"
        blocktype = -100*torch.ones((self.agent.num_agents, num_blocks, 481), dtype = torch.int8);
        
        # blockorder == 1
        # indices = 
        blocktype[(torch.tensor(blockorder) == 1).nonzero(as_tuple=True)[0].repeat_interleave(7), 
                  seq_idxs*len((torch.tensor(blockorder)==1).nonzero(as_tuple=True)[0]), :] = 0 # fixed sequence condition

        blocktype[(torch.tensor(blockorder) == 1).nonzero(as_tuple=True)[0].repeat_interleave(7), 
                  rand_idxs*len((torch.tensor(blockorder)==1).nonzero(as_tuple=True)[0]), :] = 1 # random condition
        
        # blockorder == 2
        blocktype[(torch.tensor(blockorder) == 2).nonzero(as_tuple=True)[0].repeat_interleave(7), 
                  seq_idxs*len((torch.tensor(blockorder) == 2).nonzero(as_tuple=True)[0]), :] = 1 # fixed sequence condition

        blocktype[(torch.tensor(blockorder) == 2).nonzero(as_tuple=True)[0].repeat_interleave(7), 
                  rand_idxs*len((torch.tensor(blockorder) == 2).nonzero(as_tuple=True)[0]), :] = 0 # random condition
        
        jokertypes = -100*torch.ones((self.agent.num_agents, num_blocks, 481), dtype=torch.int8)
        blockidx = -100*torch.ones((self.agent.num_agents, num_blocks, 481), dtype=torch.int8)
        trialsequence = -100*torch.ones((self.agent.num_agents, num_blocks, 481), dtype=torch.int8)
        
        "New block!"
        trialsequence[:, :, 0] = -1
        jokertypes[:, :, 0] = -1
        blocktype[:, :, 0] = -1
        
        blocktype_counter = torch.zeros((self.agent.num_agents, 2), dtype=torch.int8)
        # seq_block = 0
        # random_block = 0 
        for block in range(num_blocks):
            "New block!"
            blockidx[:, block, :] = block
            
            # block_no[:, block, :] = block # The index of the block in the current experiment
            
            current_blocktype = blocktype[:, block, 1]
            
            for ag_idx in range(self.agent.num_agents):
                if current_blocktype[ag_idx] == 0:
                    "Sequential block"
                    seq_matlab, seq_no_jokers_matlab, seq_jokertypes = \
                        utils.load_matfiles(self.matfile_dir, 
                                           blocktype_counter[ag_idx, current_blocktype[ag_idx]].item(), 
                                           current_blocktype[ag_idx], 
                                           sequence = sequence[ag_idx])

                    blocktype_counter[ag_idx, current_blocktype[ag_idx]] += 1
                    # seq_block += 1
                    
                elif current_blocktype[ag_idx] == 1:
                    "Random block"
                    seq_matlab, seq_no_jokers_matlab, seq_jokertypes = \
                        utils.load_matfiles(self.matfile_dir, 
                                           blocktype_counter[ag_idx, current_blocktype[ag_idx]].item(), 
                                           current_blocktype[ag_idx], 
                                           sequence = sequence[ag_idx])

                    blocktype_counter[ag_idx, current_blocktype[ag_idx]] += 1
                    # random_block += 1
                    
                else:
                    raise Exception("Blocktypes must be 0 or 1.")
                    
                trialsequence[ag_idx, block, 1:481] = torch.tensor(seq_matlab)
                jokertypes[ag_idx, block, 1:481] = torch.tensor(seq_jokertypes)
                
        self.data['trialsequence'] = trialsequence.numpy().T.reshape((14*481, self.agent.num_agents),order='F').tolist()
        self.data['blocktype'] = blocktype.numpy().T.reshape((14*481, self.agent.num_agents),order='F').tolist()
        self.data['jokertypes'] = jokertypes.numpy().T.reshape((14*481, self.agent.num_agents),order='F').tolist()
        self.data['blockidx'] = blockidx.numpy().T.reshape((14*481, self.agent.num_agents),order='F').tolist()
        
        self.run_loop(self.agent, self.data, 1, infer = 0)
        
    def run_loop(self, agent, data, num_particles, infer = 0, block_max = 14):
        '''

        Parameters
        ----------
        agent : obj
            DESCRIPTION.
            
        data : dict
            DESCRIPTION.
            
        num_particles : TYPE
            DESCRIPTION.
            
        infer : bool, optional
            0/1 simulate data/ infer model parameters

        block_max : int
            Up to which block (inclusive) to perform inference.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        num_trials = len(data["trialsequence"])
        num_trials_per_block = 481
        t = -1
        # blocknum = 0
        # print("NEW LOOP")
        # for tau in pyro.markov(range(num_trials)):
        for tau in pyro.markov(range(num_trials_per_block*block_max)):
            trial = torch.tensor(data["trialsequence"][tau])
            blocktype = torch.tensor(data["blocktype"][tau])
            jtype = torch.tensor(data["jokertypes"][tau])
            
            if all([data["blockidx"][tau][i] <= 5 for i in range(agent.num_agents)]):
                day = 1
                
            elif all([data["blockidx"][tau][i] > 5 for i in range(agent.num_agents)]):
                day = 2
                
            else:
                raise Exception("Da isch a Fehla!")
            
            if all(trial == -1):
                "Beginning of new block"
                # blocknum += 1
                # print(f"NEW BLOCK NUMBER {blocknum}!")
                agent.update(torch.tensor([-1]*agent.num_agents), 
                                torch.tensor([-1]*agent.num_agents), 
                                torch.tensor([-1]*agent.num_agents), 
                                day = day, 
                                trialstimulus = trial,
                                jokertype = jtype)
                
                if not infer:
                    self.choices_GD.append([-1]*self.agent.num_agents)
                    self.choices.append([-1]*self.agent.num_agents)
                    self.outcomes.append([-1]*self.agent.num_agents)
                
            else:
                if infer:
                    "Inference"
                    current_choice = torch.tensor(data["choices"][tau]).type(torch.int)
                    outcome = torch.tensor(data["outcomes"][tau]).type(torch.int)
            
                else:
                    "Simulation"
                    # assert(torch.is_tensor(trial))
                    current_choice = agent.choose_action(trial, day = day, blocktype = blocktype, jokertype = jtype)
                    outcome = torch.bernoulli(data['rewprobs'][range(agent.num_agents), current_choice])
                    self.choices_GD.append((data['rewprobs'][range(agent.num_agents), current_choice]==data['rewprobs'].max()).type(torch.int).tolist())
                    self.choices.append(current_choice.tolist())
                    self.outcomes.append(outcome.tolist())
            
                if infer and any(trial > 10):
                    "Dual-Target Trial"
                    t+=1
                    option1, option2 = agent.find_resp_options(trial)
                    # print("MAKE SURE EVERYTHING WORKS FOR ERRORS AS WELL")
                    # probs should have shape [num_particles, num_agents, nactions], or [num_agents, nactions]
                    # RHS comes out as [1, n_actions] or [num_particles, n_actions]
                    
                    "==========================================="
                    probs = agent.compute_probs(trial, day = day, blocktype = blocktype, jokertype = jtype)
                    "==========================================="
                    # ipdb.set_trace()
                    choices_bin = (current_choice != option1).type(torch.int).broadcast_to(num_particles, agent.num_agents)
                    "obs_mask is False where errors were performed, and where a subject saw a STT."
                    obs_mask = (current_choice != -2).broadcast_to(num_particles, agent.num_agents) *\
                                (trial > 10).broadcast_to(num_particles, agent.num_agents)
    
                    if any(current_choice == -10):
                        raise Exception("Fehla!")
                
                agent.update(current_choice, 
                                outcome, 
                                blocktype, 
                                day = day, 
                                trialstimulus = trial,
                                jokertype = jtype)
    
                if infer and any(trial > 10):
                    "STT are [0.5, 0.5]"
                    "errors are obs_masked"
                    pyro.sample('res_{}'.format(t), 
                                dist.Categorical(probs = probs),
                                obs = choices_bin,
                                obs_mask = obs_mask)