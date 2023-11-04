#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:02:16 2023

@author: sascha
"""

from tqdm import tqdm
import ipdb
import numpy as np
import torch
import pandas as pd

import pyro
import pyro.distributions as dist

beta_variance = lambda alpha, beta: alpha*beta / ((alpha+beta)**2 * (alpha + beta + 1))
gamma_variance = lambda conc, rate: conc/(rate**2)

device = torch.device("cpu")

torch.set_default_tensor_type(torch.DoubleTensor)

class Vbm():
    '''
    Parameters
    ----------

    num_blocks : TYPE, optional
        DESCRIPTION. The default is 14.
    '''
    

    param_names = ["omega", "dectemp", "lr"]
    num_params = len(param_names)
    NA = 4 # no. of possible actions
    num_blocks = 14
    trials = 480*num_blocks
    BAD_CHOICE = -2
    
    def __init__(self, param_dict, k, Q_init):
        '''
        
        Parameters
        ----------
        omega : torch tensor, shape [num_particles, num_agents]
        0 < omega < 1. p(a1) = σ(β*[(1-ω)*(r(a1)-r(a2)) + ω*(Q(a1)-Q(a2)))]
            
        dectemp : torch tensor, shape [num_particles, num_agents]
        0 < dectemp < inf. Decision temperature β

        lr : torch tensor, shape [num_particles, num_agents]
        0 < dectemp < 1. Learning rate

        k : TYPE
            DESCRIPTION.
            
        Q_init : torch tensor, shape [num_particles, num_agents, 4]
            Initial Q-Values.
            
        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        "Setup"
        self.num_particles = param_dict[list(param_dict.keys())[0]].shape[0]
        self.num_agents = param_dict[list(param_dict.keys())[0]].shape[1]
        # # assert(Q_init.shape == (self.num_particles, self.num_agents, 4))
        
        self.errorrates_stt = torch.rand(1)*0.1
        self.errorrates_dtt = torch.rand(1)*0.2
        
        "--- Latent variables ---"
        self.param_dict = param_dict
        "--- --- --- ---"
        
        "K"
        self.k = k
        
        "Q and rep"
        self.Q_init = Q_init
        self.Q = [Q_init] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, self.NA)*1/self.NA] # habitual values (repetition values)
        
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        '''
        -1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])
        -2 in seq_counter for errors
        Dimensions are [blocktypes, pppchoice, ppchoice, pchoice, choice, agent]
        blocktypes: 0/1 : sequential/ random
        '''
        self.init_seq_counter = self.k / self.NA * np.ones((self.num_agents, 2, 6, 6, 6, 6))
        
        self.seq_counter = self.init_seq_counter.clone().detach()
        
        self.specific_init()
        
    def specific_init(self):
        self.V = [((1-self.param_dict['omega'])[..., None]*self.rep[-1] + self.param_dict['omega'][..., None]*self.Q[-1])]


    def locs_to_pars(self, locs):
        param_dict = {'omega': torch.sigmoid(locs[..., 0]),
                           'dectemp': torch.exp(locs[..., 1]),
                           'lr': torch.sigmoid(locs[..., 2])}
        
        return param_dict
    
    def Qoutcomp(self, Qin, choices):
        '''        

        Parameters
        ----------
        Qin : tensor with shape [num_particles, num_agents, 4]
            DESCRIPTION.
            
        choices : tensor, shape (num_agents)
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        mask : TYPE
            DESCRIPTION.

        '''
        
        """Returns a tensor with the same shape as Qin, with zeros everywhere except for the relevant places
        as indicated by 'choices', where the values of Qin are retained. Q positions for agents with an error choice
        are replaced by 0."""
        
        # # assert torch.is_tensor(choices), "choices must be a tensor."
        # # assert choices.shape == (self.num_agents,), "choices must have shape (num_agents)."
        # # assert Qin.ndim == 3
        
        Qin = Qin.type(torch.double)
        
        "np_error_mask will be True for those subjects that performed no error"  
        no_error_mask = choices != self.BAD_CHOICE

        "Replace error choices by the number one"
        choices_noerrors = torch.where(no_error_mask.type(torch.bool), 
                                       choices, 
                                       torch.ones(choices.shape)).type(torch.int)
        
        choicemask = torch.zeros(Qin.shape, dtype = int)
        num_particles = self.num_particles
        num_agents = self.num_agents
        
        no_error_mask = no_error_mask.broadcast_to(num_particles, 4, num_agents).transpose(1,2)
        choicemask[torch.linspace(0, num_particles-1,num_particles, dtype = int).repeat_interleave(num_agents),
              torch.linspace(0, num_agents-1,num_agents, dtype = int).repeat(num_particles),
              choices_noerrors.repeat(num_particles)] = 1
        
        mask = no_error_mask*choicemask
        

        return Qin*mask, mask
        
    def softmax(self, z):
        sm = torch.nn.Softmax(dim=-1)
        
        if 'dectemp' in self.param_dict.keys():
            p_actions = sm(self.param_dict['dectemp'][..., None]*z)
            
        else:
            p_actions = sm(z)
            
        return p_actions

    def find_resp_options(self, stimulus_mat):
        '''
        Given a dual-target stimulus (e.g. 12, 1-indexed), this function returns the two response
        options in 0-indexing. E.g.: stimulus_mat=14 -> option1_python = 0, option1_python = 3.
        Returns twice the same option in case of STT.

        Parameters
        ----------
        stimulus_mat : torch tensor with shape [num_agents]
            DESCRIPTION.

        Returns
        -------
        option1_python : tensor with shape [num_agents]
            -1 in case of single-target trial.
            Otherwise option 1 of dual-target trial. 0-indexed.
        
        option2_python : tensor with shape [num_agents]
            -1 in case of single-target trial.
            Otherwise option 1 of dual-target trial. 0-indexed.

        '''
        
        # # assert(torch.is_tensor(stimulus_mat))
        option2_python = ((stimulus_mat % 10) - 1).type(torch.int)
        option1_python = (((stimulus_mat - (stimulus_mat % 10)) / 10) -1).type(torch.int)
        
        "Make sure to return option2 twice in case of STT"
        option1_python = torch.where(stimulus_mat > 10, option1_python, option2_python)
        
        return option1_python, option2_python
    
    def compute_probs(self, trial, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        probs = self.softmax(torch.stack((Vopt1, Vopt2), 2))
        
        
        return probs

    def choose_action(self, trial, **kwargs):
        '''
        Only execute for num_particles == 1.
        
        Parameters
        ----------
        trial : tensor with shape (num_agents) 
            Contains stimulus trial. 1-indexed.

        Returns
        -------
        tensor with shape (num_agents)
            Chosen action of agent. 0-indexed.
            -2 = error

        '''
        
        # assert trial.ndim == 1 and trial.shape[0] == self.num_agents
        # assert isinstance(day, int)
        
        "New Code"
        "STT"
        choice_python_stt = torch.where(trial < 10, trial-1, trial)
        cond_stt = torch.squeeze(torch.rand(self.num_agents) < self.errorrates_stt) 
        choice_python_stt = cond_stt * self.BAD_CHOICE + ~cond_stt * choice_python_stt
        
        "DTT"
        if torch.any(trial>10):
            option1, option2 = self.find_resp_options(trial)
            "[0, :] to choose 0th particle"
            choice_sample = torch.distributions.categorical.Categorical(probs=self.compute_probs(trial)).sample()[0, :]

            choice_python_dtt = option2*choice_sample + option1*(1-choice_sample)
            
            cond_dtt = torch.squeeze(torch.rand(self.num_agents) < self.errorrates_dtt)
            choice_python_dtt =  cond_dtt * self.BAD_CHOICE + ~cond_dtt * choice_python_dtt
            
            "Combine choices"
            choice_python = torch.where(trial < 10, choice_python_stt, choice_python_dtt)
            
            # print('day = %d'%day)
            # print('self.Q[-1] = ...')
            # print(self.Q[-1])
            # print('self.V[-1] = ...')
            # print(self.V[-1])
            
            # assert choice_python.ndim == 1
            return choice_python.clone().detach()
        
        else:
            # assert choice_python_stt.ndim == 1
            return choice_python_stt.clone().detach()
    
    def update(self, choices, outcomes, blocktype, **kwargs):
        '''
        Class Vbm().
        
        Parameters
        ----------
        choices : torch.tensor with shape [num_agents]
            The particiapnt's choice at the dual-target trial.
            -2, 0, 1, 2, or 3
            -2 = error
            
        outcomes : torch.tensor with shape [num_agents]
            no reward (0) or reward (1).
            
        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
                        
        day : int
            Day of experiment (1 or 2).
            
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        # # assert choices.ndim == 1, "choices must have shape (num_agents)."
        
        if torch.all(choices == -1) and torch.all(outcomes == -1) and torch.all(blocktype == -1):
            "Set previous actions to -1 because it's the beginning of a new block"
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.ones(self.num_particles, self.num_agents, self.NA)/self.NA)
            self.Q.append(self.Q[-1])
            
            self.V.append((1-self.param_dict['omega'])[..., None]*self.rep[-1] + self.param_dict['omega'][..., None]*self.Q[-1])
            
        else:
            "----- Update GD-values -----"
            Qout, mask = self.Qoutcomp(self.Q[-1], choices)
            Qnew = self.Q[-1] + self.param_dict['lr'][..., None]*(outcomes[None,...,None]-Qout)*mask
            self.Q.append(Qnew)
            
            "--- The following is executed in case of correct and inocrrect responses ---"
            "----- Update sequence counters and repetition values of self.rep -----"
            self.seq_counter[torch.arange(self.num_agents),
                            blocktype,
                            self.pppchoice,
                            self.ppchoice,
                            self.pchoice,
                            choices] += 1
        
            seqs_sum = self.seq_counter[torch.arange(self.num_agents), 
                                        blocktype, 
                                        self.ppchoice, 
                                        self.pchoice, 
                                        choices, 
                                        0:4].sum(axis=-1)
            
            new_rows = self.seq_counter[torch.arange(self.num_agents), 
                                        blocktype, 
                                        self.ppchoice, 
                                        self.pchoice, 
                                        choices, 
                                        0:4] / seqs_sum[..., None]

            self.rep.append(new_rows.broadcast_to(self.num_particles , self.num_agents, 4))
            
            "----- Compute new V-values for next trial -----"
            self.V.append((1-self.param_dict['omega'])[..., None]*self.rep[-1] + self.param_dict['omega'][..., None]*self.Q[-1])
            
            if len(self.Q) > 10:
                "Free up some memory space"
                self.V[0:-2] = []
                self.Q[0:-2] = []
                self.rep[0:-2] = []

            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = choices
            
    def reset(self, locs):
        self.param_dict = self.locs_to_pars(locs)
        
        self.num_particles = locs.shape[0]
        self.num_agents = locs.shape[1]
        
        "K"
        # self.k = kwargs["k"]
            
        "Q and rep"
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.NA)] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, 4)*0.25] # habitual values (repetition values)
        
        "Compute V"        
        self.V.append((1-self.param_dict['omega'])[..., None]*self.rep[-1] + self.param_dict['omega'][..., None]*self.Q[-1])
        
        "Sequence Counter"
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-2 in seq_counter for errors"
        "Dimensions are [blocktypes, pppchoice, ppchoice, pchoice, choice, agent]"
        self.seq_counter = self.init_seq_counter.clone().detach()
        
class Vbm_B(Vbm):
    
    param_names = ['lr_day1', 
                    'theta_Q_day1',
                    'theta_rep_day1',
                    'lr_day2',
                    'theta_Q_day2',
                    'theta_rep_day2']
    
    num_params = len(param_names)
    
    def specific_init(self):
        "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"
        self.V = [(self.param_dict['theta_rep_day1'][..., None]*self.rep[-1] + self.param_dict['theta_Q_day1'][..., None]*self.Q[-1])]
    
    def locs_to_pars(self, locs):
        param_dict = {'lr_day1': torch.sigmoid(locs[..., 0]),
                    'theta_Q_day1': torch.exp(locs[..., 1]),
                    'theta_rep_day1': torch.exp(locs[..., 2]),
                    
                    'lr_day2': torch.sigmoid(locs[..., 3]),
                    'theta_Q_day2': torch.exp(locs[..., 4]),
                    'theta_rep_day2': torch.exp(locs[..., 5])}
        
        return param_dict
    
    def pars_to_locs(self, df):
        df.drop(['model', 'ag_idx', 'group'], axis = 1, inplace = True)
        param_dict = df.to_dict(orient='list')
        
        locs = torch.ones((self.num_agents, self.num_params))
        locs[:, 0] = torch.logit(torch.tensor(param_dict['lr_day1']))
        locs[:, 1] = torch.log(torch.tensor(param_dict['theta_Q_day1']))
        locs[:, 2] = torch.log(torch.tensor(param_dict['theta_rep_day1']))
        
        locs[:, 3] = torch.logit(torch.tensor(param_dict['lr_day2']))
        locs[:, 4] = torch.log(torch.tensor(param_dict['theta_Q_day2']))
        locs[:, 5] = torch.log(torch.tensor(param_dict['theta_rep_day2']))

        return locs
    
    def update(self, choices, outcomes, blocktype, day, **kwargs):
        '''
        Class Vbm_B(Vbm).
        
        Parameters
        ----------
        choices : torch.tensor or list with shape [num_agents]
            The particiapnt's choice at the dual-target trial.
            -2, 0, 1, 2, or 3
            -2 = error
            
        outcomes : torch.tensor with shape [num_agents]
            no reward (0) or reward (1).
            
        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
                        
        day : int
            Day of experiment (1 or 2).
            
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # assert torch.is_tensor(choices)
        # assert torch.is_tensor(outcomes)
        # assert torch.is_tensor(blocktype)
        # assert day == 1 or day == 2
        
        if day == 1:
            lr = self.param_dict['lr_day1']
            theta_Q = self.param_dict['theta_Q_day1']
            theta_rep = self.param_dict['theta_rep_day1']

        elif day == 2:
            lr = self.param_dict['lr_day2']
            theta_Q = self.param_dict['theta_Q_day2']
            theta_rep = self.param_dict['theta_rep_day2']
        
        if torch.all(choices == -1) and torch.all(outcomes == -1) and torch.all(blocktype == -1):
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.ones(lr.shape[0], lr.shape[1], self.NA)/self.NA)
            self.Q.append(self.Q[-1])
            
            self.V.append(theta_rep[..., None]*self.rep[-1] + theta_Q[..., None]*self.Q[-1])
            
        else:
            "----- Update GD-values -----"
            # outcome is either -2 (error), 0, or 1
            # assert torch.all(outcomes <= 1)
            # # assert torch.all(outcomes > -1)
            
            "--- Group!!! ----"
            "mask contains 1s where Qoutcomp() contains non-zero entries"
            Qout, mask = self.Qoutcomp(self.Q[-1], choices)
            Qnew = self.Q[-1] + lr[..., None]*(outcomes[None,...,None]-Qout)*mask
            
            self.Q.append(Qnew)
            "--- The following is executed in case of correct and inocrrect responses ---"
            "----- Update sequence counters and repetition values of self.rep -----"
            self.seq_counter[torch.arange(self.num_agents),
                            blocktype,
                            self.pppchoice,
                            self.ppchoice,
                            self.pchoice,
                            choices] += 1
        
            seqs_sum = self.seq_counter[torch.arange(self.num_agents), 
                                        blocktype, 
                                        self.ppchoice, 
                                        self.pchoice, 
                                        choices, 
                                        0:4].sum(axis=-1)
            
            new_rows = self.seq_counter[torch.arange(self.num_agents), 
                                        blocktype, 
                                        self.ppchoice, 
                                        self.pchoice, 
                                        choices, 
                                        0:4] / seqs_sum[...,None]

            self.rep.append(new_rows.broadcast_to(self.num_particles , self.num_agents, 4))
            
            "----- Compute new V-values for next trial -----"
            self.V.append(theta_rep[..., None]*self.rep[-1] + theta_Q[..., None]*self.Q[-1])
            # dfgh

            if len(self.Q) > 10:
                "Free up some memory space"
                self.V[0:-2] = []
                self.Q[0:-2] = []
                self.rep[0:-2] = []

            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = choices

    def reset(self, locs):   
        self.param_dict = self.locs_to_pars(locs)
        
        "Setup"
        self.num_particles = locs.shape[0]
        self.num_agents = locs.shape[1]
        
        "K"
        # self.k = kwargs["k"]
            
        "Q and rep"
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.NA)] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, self.NA)*1./self.NA] # habitual values (repetition values)
        
        "Compute V"
        self.V.append(self.param_dict['theta_rep_day1'][..., None]*self.rep[-1] + self.param_dict['theta_Q_day1'][..., None]*self.Q[-1])
        
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-2 in seq_counter for errors"
        "Dimensions are [blocktypes, pppchoice, ppchoice, pchoice, choice, agent]"
        self.seq_counter = self.init_seq_counter.clone().detach()
        
class Conflict(Vbm_B):
    
    param_names = ['lr_day1',
                    'theta_Q_day1',
                    'theta_rep_day1',
                    'conflict_param_day1',
                    'lr_day2',
                    'theta_Q_day2',
                    'theta_rep_day2',
                    'conflict_param_day2']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr_day1': torch.sigmoid(locs[..., 0]),
                    'theta_Q_day1': torch.exp(locs[..., 1]),
                    'theta_rep_day1': torch.exp(locs[..., 2]),
                    'conflict_param_day1': locs[..., 3],
                    
                    'lr_day2': torch.sigmoid(locs[..., 4]),
                    'theta_Q_day2': torch.exp(locs[..., 5]),
                    'theta_rep_day2': torch.exp(locs[..., 6]),
                    'conflict_param_day2': locs[..., 7]}
    
        return param_dict
    
    def compute_probs(self, trial, day, blocktype):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        if day == 1:
            conflict_param = self.param_dict['conflict_param_day1']
            
        else:
            conflict_param = self.param_dict['conflict_param_day2']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        DeltaQ = self.Q[-1][0, torch.arange(self.num_agents), option1] - self.Q[-1][0, torch.arange(self.num_agents), option2]

        DeltaRep = self.rep[-1][0, torch.arange(self.num_agents), option1] - self.rep[-1][0, torch.arange(self.num_agents), option2]
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        conflict_value = DeltaQ - DeltaRep
        seq_bool = (blocktype == 0).type(torch.int)
        
        probs = self.softmax(torch.stack((Vopt1 + incong_bool*conflict_value*conflict_param*seq_bool, 
                                          Vopt2), 2))

        return probs
    
    def choose_action(self, trial, day, blocktype, **kwargs):
        '''
        Only execute for num_particles == 1.
        
        Parameters
        ----------
        trial : tensor with shape (num_agents) 
            Contains stimulus trial. 1-indexed.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 

        Returns
        -------
        tensor with shape (num_agents)
            Chosen action of agent. 0-indexed.
            -2 = error

        '''
        
        # assert trial.ndim == 1 and trial.shape[0] == self.num_agents
        # assert isinstance(day, int)
        
        "New Code"
        "STT"
        choice_python_stt = torch.where(trial < 10, trial-1, trial)
        cond_stt = torch.squeeze(torch.rand(self.num_agents) < self.errorrates_stt) 
        choice_python_stt = cond_stt * self.BAD_CHOICE + ~cond_stt * choice_python_stt
        
        "DTT"
        if torch.any(trial>10):
            option1, option2 = self.find_resp_options(trial)
            "[0, :] to choose 0th particle"
            choice_sample = torch.distributions.categorical.Categorical(probs=self.compute_probs(trial, day = day, blocktype = blocktype)).sample()[0, :]

            choice_python_dtt = option2*choice_sample + option1*(1-choice_sample)
            
            cond_dtt = torch.squeeze(torch.rand(self.num_agents) < self.errorrates_dtt)
            choice_python_dtt =  cond_dtt * self.BAD_CHOICE + ~cond_dtt * choice_python_dtt
            
            "Combine choices"
            choice_python = torch.where(trial < 10, choice_python_stt, choice_python_dtt)
            
            # print('day = %d'%day)
            # print('self.Q[-1] = ...')
            # print(self.Q[-1])
            # print('self.V[-1] = ...')
            # print(self.V[-1])
            
            # assert choice_python.ndim == 1
            return choice_python.clone().detach()
        
        else:
            # assert choice_python_stt.ndim == 1
            return choice_python_stt.clone().detach()
    
class Seqparam(Vbm_B):
    
    param_names = ['lr_day1',
                    'theta_Q_day1',
                    'theta_rep_day1',
                    'seq_param_day1',
                    
                    'lr_day2',
                    'theta_Q_day2',
                    'theta_rep_day2',
                    'seq_param_day2']
    
    num_params = len(param_names)
    
    def specific_init(self):
        "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"
        self.V = [(self.param_dict['theta_rep_day1'][..., None]*self.rep[-1] + self.param_dict['theta_Q_day1'][..., None]*self.Q[-1])]
        self.continuous_actions = torch.zeros(self.num_agents)
    
    def locs_to_pars(self, locs):
        param_dict = {"lr_day1": torch.sigmoid(locs[..., 0]),
                    "theta_Q_day1": torch.exp(locs[..., 1]),
                    "theta_rep_day1": torch.exp(locs[..., 2]),
                    "seq_param_day1": torch.exp(locs[..., 3]),
                    
                    "lr_day2": torch.sigmoid(locs[..., 4]),
                    "theta_Q_day2": torch.exp(locs[..., 5]),
                    "theta_rep_day2": torch.exp(locs[..., 6]),
                    "seq_param_day2": torch.exp(locs[..., 7])}
    
        return param_dict
    
    def update(self, choices, outcomes, blocktype, day, **kwargs):
        '''
        Class Vbm_B(Vbm).
        
        Parameters
        ----------
        choices : torch.tensor or list with shape [num_agents]
            The particiapnt's choice at the dual-target trial.
            -2, 0, 1, 2, or 3
            -2 = error
            
        outcomes : torch.tensor with shape [num_agents]
            no reward (0) or reward (1).
            
        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
                        
        day : int
            Day of experiment (1 or 2).
            
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # assert torch.is_tensor(choices)
        # assert torch.is_tensor(outcomes)
        # assert torch.is_tensor(blocktype)
        # assert day == 1 or day == 2
        
        if day == 1:
            lr = self.param_dict['lr_day1']
            theta_Q = self.param_dict['theta_Q_day1']
            theta_rep = self.param_dict['theta_rep_day1']
            seq_param = self.param_dict['seq_param_day1']

        elif day == 2:
            lr = self.param_dict['lr_day2']
            theta_Q = self.param_dict['theta_Q_day2']
            theta_rep = self.param_dict['theta_rep_day2']
            seq_param = self.param_dict['seq_param_day2']
        
        if torch.all(choices == -1) and torch.all(outcomes == -1) and torch.all(blocktype == -1):
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.ones(self.num_particles, self.num_agents, self.NA)/self.NA)
            self.Q.append(self.Q[-1])
            
            self.V.append(theta_rep[..., None]*self.rep[-1] + theta_Q[..., None]*self.Q[-1])
            self.continuous_actions = torch.zeros(self.num_agents)
            
        else:
            "----- Update GD-values -----"
            # outcome is either -2 (error), 0, or 1
            # assert torch.all(outcomes <= 1)
            # # assert torch.all(outcomes > -1)
            
            self.continuous_actions += choices != -2
            self.continuous_actions = torch.where(choices == -2, torch.zeros(self.continuous_actions.shape), self.continuous_actions)
            # ipdb.set_trace()
            
            "--- Group!!! ----"
            "mask contains 1s where Qoutcomp() contains non-zero entries"
            Qout, mask = self.Qoutcomp(self.Q[-1], choices)
            Qnew = self.Q[-1] + lr[..., None]*(outcomes[None,...,None]-Qout)*mask
            
            self.Q.append(Qnew)
            "--- The following is executed in case of correct and inocrrect responses ---"
            "----- Update sequence counters and repetition values of self.rep -----"
            self.seq_counter[torch.arange(self.num_agents),
                            blocktype,
                            self.pppchoice,
                            self.ppchoice,
                            self.pchoice,
                            choices] += 1
        
            seqs_sum = self.seq_counter[torch.arange(self.num_agents), 
                                        blocktype, 
                                        self.ppchoice, 
                                        self.pchoice, 
                                        choices, 
                                        0:4].sum(axis=-1)
            
            new_rows = self.seq_counter[torch.arange(self.num_agents), 
                                        blocktype, 
                                        self.ppchoice, 
                                        self.pchoice, 
                                        choices, 
                                        0:4] / seqs_sum[...,None]

            self.rep.append(new_rows.broadcast_to(self.num_particles , self.num_agents, 4))
            
            "----- Compute new V-values for next trial -----"
            seq_cond = (self.continuous_actions > 8).type(torch.int)
            self.V.append((theta_rep[..., None] + seq_cond[..., None]*seq_param[..., None])*self.rep[-1] + theta_Q[..., None]*self.Q[-1])
            # ipdb.set_trace()

            if len(self.Q) > 10:
                "Free up some memory space"
                self.V[0:-2] = []
                self.Q[0:-2] = []
                self.rep[0:-2] = []

            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = choices
    
    def reset(self, locs):   
        self.param_dict = self.locs_to_pars(locs)
        
        "Setup"
        self.num_particles = locs.shape[0]
        self.num_agents = locs.shape[1]
        
        self.continuous_actions = torch.zeros(self.num_agents)
            
        "Q and rep"
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.NA)] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, self.NA)*1./self.NA] # habitual values (repetition values)
        
        "Compute V"
        self.V.append(self.param_dict['theta_rep_day1'][..., None]*self.rep[-1] + self.param_dict['theta_Q_day1'][..., None]*self.Q[-1])
        
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-2 in seq_counter for errors"
        "Dimensions are [blocktypes, pppchoice, ppchoice, pchoice, choice, agent]"
        self.seq_counter = self.init_seq_counter.clone().detach()
        
class Random(Vbm):
    
    param_names = ['p_cong_day1',
                   'p_incong_day1',
                   'p_rand_day1', 
                   'p_cong_day2',
                   'p_incong_day2',
                   'p_rand_day2']
    
    num_params = len(param_names)
    
    def __init__(self, param_dict, k, Q_init):
        '''
        
        Parameters
        ----------
        omega : torch tensor, shape [num_particles, num_agents]
        0 < omega < 1. p(a1) = σ(β*[(1-ω)*(r(a1)-r(a2)) + ω*(Q(a1)-Q(a2)))]
            
        dectemp : torch tensor, shape [num_particles, num_agents]
        0 < dectemp < inf. Decision temperature β

        lr : torch tensor, shape [num_particles, num_agents]
        0 < dectemp < 1. Learning rate

        k : TYPE
            DESCRIPTION.
            
        Q_init : torch tensor, shape [num_particles, num_agents, 4]
            Initial Q-Values.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        "Setup"
        self.num_particles = param_dict[list(param_dict.keys())[0]].shape[0]
        self.num_agents = param_dict[list(param_dict.keys())[0]].shape[1]
        # # assert(Q_init.shape == (self.num_particles, self.num_agents, 4))
        
        self.errorrates_stt = torch.rand(1)*0.1
        self.errorrates_dtt = torch.rand(1)*0.2
        
        "--- Latent variables ---"
        self.param_dict = param_dict
        "--- --- --- ---"
        
        "K"
        self.k = k
        
        "Q and rep"
        self.Q_init = Q_init
        self.Q = [Q_init] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, self.NA)*1/self.NA] # habitual values (repetition values)
        
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-2 in seq_counter for errors"
        "Dimensions are [blocktypes, pppchoice, ppchoice, pchoice, choice, agent]"
        self.init_seq_counter = self.k / self.NA * np.ones((self.num_agents, 2, 6, 6, 6, 6))
        
        self.seq_counter = self.init_seq_counter.clone().detach()
        
    def locs_to_pars(self, locs):
        param_dict = {'p_cong_day1': torch.sigmoid(locs[..., 0]),
                    'p_incong_day1': torch.sigmoid(locs[..., 1]),
                    'p_rand_day1': torch.sigmoid(locs[..., 2]),
                    
                    'p_cong_day2': torch.sigmoid(locs[..., 3]),
                    'p_incong_day2': torch.sigmoid(locs[..., 4]),
                    'p_rand_day2': torch.sigmoid(locs[..., 5])}
        
        return param_dict
    
    def choose_action(self, trial, day, blocktype):
        '''
        Only execute for num_particles == 1.
        
        Parameters
        ----------
        trial : tensor with shape (num_agents) 
            Contains stimulus trial. 1-indexed.
            
        day : int
            Day of experiment.

        Returns
        -------
        tensor with shape (num_agents)
            Chosen action of agent. 0-indexed.
            -2 = error

        '''
        
        # assert trial.ndim == 1 and trial.shape[0] == self.num_agents
        # assert isinstance(day, int)
        
        "New Code"
        "STT"
        choice_python_stt = torch.where(trial < 10, trial-1, trial)
        cond_stt = torch.squeeze(torch.rand(self.num_agents) < self.errorrates_stt) 
        choice_python_stt = cond_stt * self.BAD_CHOICE + ~cond_stt * choice_python_stt
        
        "DTT"
        if torch.any(trial>10):
            option1, option2 = self.find_resp_options(trial)
            "[0, :] to choose 0th particle"
            choice_sample = torch.distributions.categorical.Categorical(probs=self.compute_probs(trial, day, blocktype)).sample()[0, :]

            choice_python_dtt = option2*choice_sample + option1*(1-choice_sample)
            
            cond_dtt = torch.squeeze(torch.rand(self.num_agents) < self.errorrates_dtt)
            choice_python_dtt =  cond_dtt * self.BAD_CHOICE + ~cond_dtt * choice_python_dtt
            
            "Combine choices"
            choice_python = torch.where(trial < 10, choice_python_stt, choice_python_dtt)
            
            # print('day = %d'%day)
            # print('self.Q[-1] = ...')
            # print(self.Q[-1])
            # print('self.V[-1] = ...')
            # print(self.V[-1])
            
            # assert choice_python.ndim == 1
            return choice_python.clone().detach()
        
        else:
            # assert choice_python_stt.ndim == 1
            return choice_python_stt.clone().detach()
    
    def compute_probs(self, trial, day, blocktype):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : tensor
            0/1 : sequential/ random 

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        if day == 1:
            p_cong = self.param_dict['p_cong_day1']
            p_incong = self.param_dict['p_incong_day1']
            p_rand = self.param_dict['p_rand_day1']
            
        else:
            p_cong = self.param_dict['p_cong_day2']
            p_incong = self.param_dict['p_incong_day2']
            p_rand = self.param_dict['p_rand_day2']
        
        option1, option2 = self.find_resp_options(trial)
        
        DeltaQ = self.Q[-1][0, torch.arange(self.num_agents), option1] - self.Q[-1][0, torch.arange(self.num_agents), option2]

        DeltaRep = self.rep[-1][0, torch.arange(self.num_agents), option1] - self.rep[-1][0, torch.arange(self.num_agents), option2]
        
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        cong_bool = ((torch.sign(DeltaQ).type(torch.int) * torch.sign(DeltaRep).type(torch.int) + 1)*0.5).type(torch.int)
        
        p_seq_option1 = (DeltaQ > 0).type(torch.int) * cong_bool * p_cong + \
            (DeltaQ < 0).type(torch.int) * incong_bool * (1-p_incong) +\
            (DeltaQ == 0).type(torch.int)*torch.tensor(0.5)
            
        p_rand_option1 = (DeltaQ > 0).type(torch.int) * p_rand + \
                (DeltaQ < 0).type(torch.int) * (1-p_rand) + \
                (DeltaQ == 0).type(torch.int)*torch.tensor(0.5)
        
        probs_option1 = (blocktype == 0).type(torch.int) * p_seq_option1 + (blocktype == 1).type(torch.int) * p_rand_option1
        
        probs_option1 = probs_option1 + (trial < 10).type(torch.int) *0.5
        probs_option1 = torch.where(torch.logical_or(probs_option1 == 1, probs_option1 == 10), 0.5*torch.ones(probs_option1.shape), probs_option1)
        
        probs_option2 =  1 - probs_option1
        
        probs = torch.stack((probs_option1, probs_option2), axis=-1)
        return probs
    
    def update(self, choices, outcomes, blocktype, **kwargs):
        '''
        Class Vbm().
        
        Parameters
        ----------
        choices : torch.tensor with shape [num_agents]
            The particiapnt's choice at the dual-target trial.
            -2, 0, 1, 2, or 3
            -2 = error
            
        outcomes : torch.tensor with shape [num_agents]
            no reward (0) or reward (1).
            
        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
                        
        day : int
            Day of experiment (1 or 2).
            
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        # # assert choices.ndim == 1, "choices must have shape (num_agents)."
        
        if torch.all(choices == -1) and torch.all(outcomes == -1) and torch.all(blocktype == -1):
            "Set previous actions to -1 because it's the beginning of a new block"
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.ones(self.num_particles, self.num_agents, self.NA)/self.NA)
            self.Q.append(self.Q[-1])
            
        else:
            "--- The following is executed in case of correct and inocrrect responses ---"
            "----- Update sequence counters and repetition values of self.rep -----"
            self.seq_counter[torch.arange(self.num_agents),
                            blocktype,
                            self.pppchoice,
                            self.ppchoice,
                            self.pchoice,
                            choices] += 1
        
            seqs_sum = self.seq_counter[torch.arange(self.num_agents), 
                                        blocktype, 
                                        self.ppchoice, 
                                        self.pchoice, 
                                        choices, 
                                        0:4].sum(axis=-1)
            
            new_rows = self.seq_counter[torch.arange(self.num_agents), 
                                        blocktype, 
                                        self.ppchoice, 
                                        self.pchoice, 
                                        choices, 
                                        0:4] / seqs_sum[..., None]

            self.rep.append(new_rows.broadcast_to(self.num_particles , self.num_agents, 4))
            
            if len(self.Q) > 10:
                "Free up some memory space"
                self.Q[0:-2] = []
                self.rep[0:-2] = []

            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = choices
            
    def reset(self, locs):
        self.param_dict = self.locs_to_pars(locs)
        
        self.num_particles = locs.shape[0]
        self.num_agents = locs.shape[1]
        
        "K"
        # self.k = kwargs["k"]
            
        "Q and rep"
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.NA)] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, self.NA)*1/self.NA] # habitual values (repetition values)
        
        "Sequence Counter"
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-2 in seq_counter for errors"
        "Dimensions are [blocktypes, pppchoice, ppchoice, pchoice, choice, agent]"
        self.seq_counter = self.init_seq_counter.clone().detach()
        
class Vbm_B_Q(Vbm_B):

    param_names = ['lr_day1', 
                    'theta_Q_day1',
                    'theta_rep_day1',
                    'lr_day2',
                    'theta_Q_day2',
                    'theta_rep_day2',
                    'Qparam']
    
    num_params = len(param_names)

    def specific_init(self):
        
            self.Q_init = self.Q_init*self.param_dict['Qparam'][..., None]
            self.Q = [self.Q_init] # Goal-Directed Q-Values
            "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"
            self.V = [(self.param_dict['theta_rep_day1'][..., None]*self.rep[-1] + self.param_dict['theta_Q_day1'][..., None]*self.Q[-1])]
            
    def locs_to_pars(self, locs):
        param_dict = {'lr_day1': torch.sigmoid(locs[..., 0]),
                    'theta_Q_day1': torch.exp(locs[..., 1]),
                    'theta_rep_day1': torch.exp(locs[..., 2]),
                    
                    'lr_day2': torch.sigmoid(locs[..., 3]),
                    'theta_Q_day2': torch.exp(locs[..., 4]),
                    'theta_rep_day2': torch.exp(locs[..., 5]),
                    'Qparam':torch.exp(locs[..., 6])}
        
        return param_dict


class Vbm_B_k(Vbm_B):

    param_names = ['lr_day1', 
                    'theta_Q_day1',
                    'theta_rep_day1',
                    'lr_day2',
                    'theta_Q_day2',
                    'theta_rep_day2',
                    'kparam']
    
    num_params = len(param_names)
    
    def __init__(self, param_dict, k, Q_init):
        '''
        
        Parameters
        ----------
        omega : torch tensor, shape [num_particles, num_agents]
        0 < omega < 1. p(a1) = σ(β*[(1-ω)*(r(a1)-r(a2)) + ω*(Q(a1)-Q(a2)))]
            
        dectemp : torch tensor, shape [num_particles, num_agents]
        0 < dectemp < inf. Decision temperature β

        lr : torch tensor, shape [num_particles, num_agents]
        0 < dectemp < 1. Learning rate

        k : TYPE
            DESCRIPTION.
            
        Q_init : torch tensor, shape [num_particles, num_agents, 4]
            Initial Q-Values.
            
        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        "Setup"
        self.num_particles = param_dict[list(param_dict.keys())[0]].shape[0]
        self.num_agents = param_dict[list(param_dict.keys())[0]].shape[1]
        # # assert(Q_init.shape == (self.num_particles, self.num_agents, 4))
        
        self.errorrates_stt = torch.rand(1)*0.1
        self.errorrates_dtt = torch.rand(1)*0.2
        
        "--- Latent variables ---"
        self.param_dict = param_dict
        "--- --- --- ---"
        
        "K"
        self.k = k*self.param_dict['kparam']
        
        "Q and rep"
        self.Q_init = Q_init
        self.Q = [Q_init] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, self.NA)*1/self.NA] # habitual values (repetition values)
        
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        '''
        -1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])
        -2 in seq_counter for errors
        Dimensions are [blocktypes, pppchoice, ppchoice, pchoice, choice, agent]
        blocktypes: 0/1 : sequential/ random
        '''
        self.init_seq_counter = torch.tensor(np.array(self.k / self.NA * np.ones((self.num_agents, 2, 6, 6, 6, 6)).transpose()).transpose())
        
        self.seq_counter = self.init_seq_counter.clone().detach()
        
        self.specific_init()
        
    def specific_init(self):
        "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"
        self.V = [(self.param_dict['theta_rep_day1'][..., None]*self.rep[-1] + self.param_dict['theta_Q_day1'][..., None]*self.Q[-1])]

    def locs_to_pars(self, locs):
        param_dict = {'lr_day1': torch.sigmoid(locs[..., 0]),
                    'theta_Q_day1': torch.exp(locs[..., 1]),
                    'theta_rep_day1': torch.exp(locs[..., 2]),
                    
                    'lr_day2': torch.sigmoid(locs[..., 3]),
                    'theta_Q_day2': torch.exp(locs[..., 4]),
                    'theta_rep_day2': torch.exp(locs[..., 5]),
                    'kparam':torch.exp(locs[..., 6])}
        
        return param_dict
    
class Handedness(Vbm_B):
    
    param_names = ['lr_day1', 
                    'theta_Q_day1',
                    'theta_rep_day1',
                    'lr_day2',
                    'theta_Q_day2',
                    'theta_rep_day2',
                    'hand_param']
    
    num_params = len(param_names)
    
    def locs_to_pars(self, locs):
        param_dict = {'lr_day1': torch.sigmoid(locs[..., 0]),
                    'theta_Q_day1': torch.exp(locs[..., 1]),
                    'theta_rep_day1': torch.exp(locs[..., 2]),
                    
                    'lr_day2': torch.sigmoid(locs[..., 3]),
                    'theta_Q_day2': torch.exp(locs[..., 4]),
                    'theta_rep_day2': torch.exp(locs[..., 5]),
                    'hand_param': locs[..., 6]*10}
        
        return param_dict
    
    def compute_probs(self, trial, day, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        assert torch.all(option1 <= option2)
        sides_bool = (option1 < 2).type(torch.int) * (option2 >= 2).type(torch.int)
        probs = self.softmax(torch.stack((Vopt1, Vopt2 + self.param_dict['hand_param']*sides_bool), 2))

        return probs
    
    def choose_action(self, trial, day, blocktype, **kwargs):
        '''
        Only execute for num_particles == 1.
        
        Parameters
        ----------
        trial : tensor with shape (num_agents) 
            Contains stimulus trial. 1-indexed.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 

        Returns
        -------
        tensor with shape (num_agents)
            Chosen action of agent. 0-indexed.
            -2 = error

        '''
        
        # assert trial.ndim == 1 and trial.shape[0] == self.num_agents
        # assert isinstance(day, int)
        
        "New Code"
        "STT"
        choice_python_stt = torch.where(trial < 10, trial-1, trial)
        cond_stt = torch.squeeze(torch.rand(self.num_agents) < self.errorrates_stt) 
        choice_python_stt = cond_stt * self.BAD_CHOICE + ~cond_stt * choice_python_stt
        
        "DTT"
        if torch.any(trial>10):
            option1, option2 = self.find_resp_options(trial)
            "[0, :] to choose 0th particle"
            choice_sample = torch.distributions.categorical.Categorical(probs=self.compute_probs(trial, 
                                                                                                 day = day, 
                                                                                                 blocktype = blocktype)).sample()[0, :]

            choice_python_dtt = option2*choice_sample + option1*(1-choice_sample)
            
            cond_dtt = torch.squeeze(torch.rand(self.num_agents) < self.errorrates_dtt)
            choice_python_dtt =  cond_dtt * self.BAD_CHOICE + ~cond_dtt * choice_python_dtt
            
            "Combine choices"
            choice_python = torch.where(trial < 10, choice_python_stt, choice_python_dtt)
            
            # print('day = %d'%day)
            # print('self.Q[-1] = ...')
            # print(self.Q[-1])
            # print('self.V[-1] = ...')
            # print(self.V[-1])
            
            # assert choice_python.ndim == 1
            return choice_python.clone().detach()
        
        else:
            # assert choice_python_stt.ndim == 1
            return choice_python_stt.clone().detach()
    
