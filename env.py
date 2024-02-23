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
        
    def create_experiment(self, group, day):
        '''
        Creates experiment, i.e. sequence of stimuli by fetching from preprocessed
        data file.
        
        Parameters
        ----------
        group

        day
            
        STORES
        ----------
        self.data : dict
            Keys:
                trialsequence : list of len 6734, each element list of len num_agents
                blocktype : list of len 6734, each element list of len num_agents
                jokertypes : list of len 6734, each element list of len num_agents
                blockidx : list of len 6734, each element list of len num_agents

        '''
        import pickle
        exp_behav_dict, _ = pickle.load(open(f"behav_data/preproc_data_day{day}.p", "rb" ))
        
        self.data = {}
        if exp_behav_dict['group'][0] == group:
            self.data['trialsequence'] = exp_behav_dict['trialsequence'].copy()
            self.data['blocktype'] = exp_behav_dict['blocktype'].copy()
            self.data['jokertypes'] = exp_behav_dict['jokertypes'].copy()
            self.data['blockidx'] = exp_behav_dict['blockidx'].copy()
            self.data['trialidx'] = exp_behav_dict['trialidx'].copy()
            
            "trialidx_day"
            if day == 1:
                self.data['trialidx_day'] = exp_behav_dict['trialidx'].copy()
                
            elif day == 2:
                trialidx_day = []
                
                for tl in exp_behav_dict['trialidx']:
                    trialidx_day.append((torch.tensor(tl)-2886).tolist())
                
                self.data['trialidx_day'] = trialidx_day
                
        else:
            '''
            For posterior predictives
            Assert that group is the same for each entry 
            (since we're repeating the same participant several times).
            '''
            assert all(x==group[0] for x in group)
            grp = group[0]
            idx = exp_behav_dict['group'][0].index(grp)
            
            trialsequence = []
            blocktype = []
            jokertypes = []
            blockidx = []
            trialidx = []
            
            for t in range(len(exp_behav_dict['trialsequence'])):
                trialsequence.append([exp_behav_dict['trialsequence'][t][idx]]*len(group))
                blocktype.append([exp_behav_dict['blocktype'][t][idx]]*len(group))
                jokertypes.append([exp_behav_dict['jokertypes'][t][idx]]*len(group))
                blockidx.append([exp_behav_dict['blockidx'][t][idx]]*len(group))
                trialidx.append([exp_behav_dict['trialidx'][t][idx]]*len(group))
        
            self.data['trialsequence'] = trialsequence
            self.data['blocktype'] = blocktype
            self.data['jokertypes'] = jokertypes
            self.data['blockidx'] = blockidx
            self.data['trialidx'] = trialidx
        
        rewprobs = [[0.8, 0.2, 0.2, 0.8],
                    [0.8, 0.2, 0.2, 0.8],
                    [0.2, 0.8, 0.8, 0.2],
                    [0.2, 0.8, 0.8, 0.2]]
        
        self.data['rewprobs'] = torch.tensor(rewprobs)[torch.tensor(group), :]
        
    def run(self, group, day, STT):
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
            
        day : int
            1 or 2

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        if STT:
            self.RTs = []
            
        else:
            self.choices = []
            self.outcomes = []
            self.choices_GD = []
        
        self.create_experiment(group = group,
                               day = day)
        
        print("Simulating data...")
        self.run_loop(agent = self.agent, 
                      data = self.data, 
                      num_particles = 1, 
                      infer = 0,
                      STT = STT)
        
    def run_loop(self, agent, data, num_particles, infer, STT = False):
        '''

        Parameters
        ----------
        agent : obj
            DESCRIPTION.
            
        data : dict
            DESCRIPTION.
            
        num_particles : TYPE
            DESCRIPTION.
            
        infer : int
            0 simulate data
            1 infer model parameters
            2 MLE estimate: return log-likelihood

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        log_like = 0.
        # num_trials_per_block = 962
        dtt_trial = -1 # dtt_trial is the index of the pyro sample sites for DTT
        stt_trial = -1 # dtt_trial is the index of the pyro sample sites for STT
        for tau in pyro.markov(range(len(data["trialsequence"]))):
            trial = torch.tensor(data["trialsequence"][tau])
            blocktype = torch.tensor(data["blocktype"][tau])
            jtype = torch.tensor(data["jokertypes"][tau])

            if all(trial == -1):
                "Beginning of new block"
                # blocknum += 1
                # print(f"NEW BLOCK NUMBER {blocknum}!")
                agent.update(torch.tensor([-1]*agent.num_agents), 
                                torch.tensor([-1]*agent.num_agents), 
                                torch.tensor([-1]*agent.num_agents), 
                                # day = day, 
                                trialstimulus = trial,
                                jokertype = jtype)
                
                if infer == 0:
                    "Simulation"
                    if STT:
                        self.RTs.append([-1]*self.agent.num_agents)
                        
                    else:
                        self.choices_GD.append([-1]*self.agent.num_agents)
                        self.choices.append([-1]*self.agent.num_agents)
                        self.outcomes.append([-1]*self.agent.num_agents)
                
            else:
                if infer > 0:
                    "Inference"
                    if STT:
                        current_RT = torch.tensor(data["RT"][tau]).type(torch.int)
                        
                    else:
                        current_choice = torch.tensor(data["choices"][tau]).type(torch.int)
                        outcome = torch.tensor(data["outcomes"][tau]).type(torch.int)
            
                elif infer == 0:
                    "Simulation"
                    if STT:
                        current_RT = agent.stt_action(trial, blocktype = blocktype, jokertype = jtype)
                        self.RTs.append(current_RT)
                        
                    else:
                        current_choice = agent.choose_action(trial, blocktype = blocktype, jokertype = jtype)
                        outcome = torch.bernoulli(data['rewprobs'][range(agent.num_agents), current_choice])
                        self.choices_GD.append((data['rewprobs'][range(agent.num_agents), 
                                                                 current_choice]==data['rewprobs'].max()).type(torch.int).tolist())
                        self.choices.append(current_choice.tolist())
                        self.outcomes.append(outcome.tolist())
                    
                if infer > 0 and any(trial > 10) and not STT:
                    "Dual-Target Trial Inference"
                    "Retrieve probs before updating"
                    option1, option2 = agent.find_resp_options(trial)
                    # print("MAKE SURE EVERYTHING WORKS FOR ERRORS AS WELL")
                    # probs should have shape [num_particles, num_agents, nactions], or [num_agents, nactions]
                    # RHS comes out as [1, n_actions] or [num_particles, n_actions]
                    
                    "==========================================="
                    probs = agent.compute_probs(trial, blocktype = blocktype, jokertype = jtype, blockidx = data["blockidx"][tau])
                    "==========================================="
                    # ipdb.set_trace()
                    choices_bin = (current_choice != option1).type(torch.int).broadcast_to(num_particles, agent.num_agents)
                    "obs_mask is False where errors were performed, and where a subject saw a STT."
                    obs_mask = (current_choice != -2).broadcast_to(num_particles, agent.num_agents) *\
                                (trial > 10).broadcast_to(num_particles, agent.num_agents)
    
                    if any(current_choice == -10):
                        raise Exception("Fehla!")
                        
                if infer > 0 and any((trial < 10)*(trial > 0)) and STT:
                    "Single-Target Trial Inference"
                    "Retrieve locs and scales before updating"
                    
                    loc, scale = agent.compute_probs(trial, blocktype = blocktype)
                
                if STT:
                    agent.update(blocktype, 
                                 trialstimulus = trial,
                                 RT = current_RT)
                    
                else:
                    agent.update(current_choice, 
                                    outcome, 
                                    blocktype, 
                                    # day = day, 
                                    trialstimulus = trial,
                                    jokertype = jtype)

                if infer==1 and any(trial > 10) and not STT:
                    "DTT Inference"
                    "STT are [0.5, 0.5]"
                    "errors are obs_masked"
                    dtt_trial += 1
                    pyro.sample('res_{}'.format(dtt_trial), 
                                dist.Categorical(probs = probs),
                                obs = choices_bin,
                                obs_mask = obs_mask)
                    
                if infer == 1 and any((trial < 10)*(trial > 0)) and STT:
                    "STT RT Inference"
                    stt_trial += 1
                    obs_mask = (current_RT != -2).broadcast_to(num_particles, agent.num_agents) *\
                                ((trial < 10)*(trial > 0)).broadcast_to(num_particles, agent.num_agents)
                    pyro.sample('res_{}'.format(stt_trial), 
                                dist.Normal(loc = loc, scale = scale),
                                obs = current_RT,
                                obs_mask = obs_mask)

                elif infer==2 and any(trial > 10) and not STT:
                    log_like += torch.log(probs[range(agent.num_particles),
                                      range(agent.num_agents), 
                                      choices_bin]) * obs_mask
                    
                # elif infer==2 and any((trial < 10)*(trial > 0)) and STT:
                #     log_like += torch.log(probs[range(agent.num_particles),
                #                       range(agent.num_agents), 
                #                       choices_bin]) * obs_mask
                    
        return log_like