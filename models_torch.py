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

class model_master():

    NA = 4 # no. of possible actions
    # num_blocks = 14
    # trials = 480*num_blocks
    BAD_CHOICE = -2    

    def __init__(self, 
                 Q_init, 
                 num_agents = None, 
                 param_dict = None, 
                 k=4.,
                 seq_init = None,
                 errorrates = None):
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
            
        param_dict : dict with parameters. 
            Each entry has to be of shape [num_particles, num_agents]
            
        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if param_dict is not None:
            assert errorrates is not None
        
        assert not ((num_agents is None) and (param_dict is None)), "Either num_agents or param_dict must be set."
        
        if param_dict is None:
            self.num_agents = num_agents
            self.num_particles = 1
            
            locs = torch.tensor(np.random.uniform(-2, 2, (self.num_particles, num_agents, self.num_params)))
            self.param_dict = self.locs_to_pars(locs)
            
        else:
            self.num_particles = param_dict[list(param_dict.keys())[0]].shape[0]
            self.num_agents = param_dict[list(param_dict.keys())[0]].shape[1]
            self.param_dict = param_dict
        
        "Make sure each key of param_dict comes out as [num_particles, num_agents]"
        for key in self.param_dict.keys():
            if self.param_dict[key].ndim == 1:
                if self.param_dict[key].size == self.num_agents:
                    self.param_dict[key] = self.param_dict[key][None, ...]
                    
                else:
                    raise Exception("If ndim == 1, size of param_dict must be num_agnets.")
                    
            elif self.param_dict[key].ndim == 2:
                assert self.param_dict[key].shape[0] == self.num_particles and \
                    self.param_dict[key].shape[1] == self.num_agents
        
        
        if errorrates is not None:
            assert errorrates.ndim == 2
            assert errorrates.shape[0] == 4
            assert errorrates.shape[1] == self.num_agents
            
            self.errorrates = errorrates
            
        else:
            '''
                rows
                    0 : STT
                    1 : Random
                    2 : Congruent
                    3 : incongruent
            '''
            self.errorrates = torch.rand((4, self.num_agents))
            self.errorrates[0, :] = self.errorrates[0, :]*0.1
            self.errorrates[1, :] = self.errorrates[1, :]*0.2 # random
            self.errorrates[2, :] = self.errorrates[2, :]*0.2 # congruent 
            self.errorrates[3, :] = self.errorrates[3, :]*0.2 # incongruent
        
        
        "K"
        self.k = torch.tensor([k])
        
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
        if seq_init == None:
            self.init_seq_counter = self.k / self.NA * np.ones((self.num_agents, 2, 6, 6, 6, 6))
            
        else:
            self.init_seq_counter = seq_init
            
        self.seq_counter = self.init_seq_counter.clone().detach()
        self.specific_init()
        
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
        
        assert torch.is_tensor(choices), print("choices must be a tensor.")
        assert choices.shape == (self.num_agents,), print("choices must have shape (num_agents).")
        assert Qin.ndim == 3, print("Input dimension not 3")
        # print(Qin.shape[0])
        
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
    
    def choose_action(self, trial, blocktype, jokertype, **kwargs):
        '''
        Only execute for num_particles == 1.
        
        Parameters
        ----------
        trial : tensor with shape (num_agents) 
            Contains stimulus trial. 1-indexed.
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent
            
        kwargs
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
        'cond_stt is 1 when an error is performed'
        choice_python_stt = torch.where(trial < 10, trial-1, trial)
        cond_stt = torch.squeeze(torch.rand(self.num_agents) < self.errorrates[0, :]).type(torch.int)
        choice_python_stt = cond_stt * self.BAD_CHOICE + (1-cond_stt) * choice_python_stt
        
        "DTT"
        if torch.any(trial>10):
            option1, option2 = self.find_resp_options(trial)
            "[0, :] to choose 0th particle"
            choice_sample = torch.distributions.categorical.Categorical(probs=
                                                                        self.compute_probs(trial = trial,
                                                                                           blocktype = blocktype,
                                                                                           jokertype = jokertype,
                                                                                           blockidx = torch.tensor([0]),
                                                                                           **kwargs)).sample()[0, :]

            choice_python_dtt = option2*choice_sample + option1*(1-choice_sample)
            
            "cond_dtt is 1 if an error was performed"
            '''
                self.errorrates:
                    0 : STT
                    1 : Random
                    2 : Congruent
                    3 : incongruent
            '''
            cond_dtt = torch.squeeze(torch.rand(self.num_agents) < self.errorrates[1, :]) * (jokertype == 0).type(torch.int) + \
                        torch.squeeze(torch.rand(self.num_agents) < self.errorrates[2, :]) * (jokertype == 1).type(torch.int) + \
                        torch.squeeze(torch.rand(self.num_agents) < self.errorrates[3, :]) * (jokertype == 2).type(torch.int)
            
            # cond_dtt = torch.squeeze(torch.rand(self.num_agents) < self.errorrates_dtt)
            choice_python_dtt =  cond_dtt * self.BAD_CHOICE + (1-cond_dtt) * choice_python_dtt
            
            "Combine choices"
            choice_python = torch.where(trial < 10, 
                                        choice_python_stt, 
                                        choice_python_dtt)
            
            # assert choice_python.ndim == 1
            return choice_python.clone().detach()
        
        else:
            # assert choice_python_stt.ndim == 1
            return choice_python_stt.clone().detach()
        
        
    def reset(self, locs):
        self.param_dict = self.locs_to_pars(locs)
        
        self.num_particles = locs.shape[0]
        self.num_agents = locs.shape[1]
        
        "K"
        # self.k = kwargs["k"]
            
        "Q and rep"
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.NA)] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, self.NA)/self.NA] # habitual values (repetition values)

        "Compute V"        
        self.V.append(self.compute_V(blocktype = torch.tensor(0.)))
        
        "Sequence Counter"
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-2 in seq_counter for errors"
        "Dimensions are [blocktypes, pppchoice, ppchoice, pchoice, choice, agent]"
        self.seq_counter = self.init_seq_counter.clone().detach()

class Vbm_lr(model_master):
    '''
    Parameters
    ----------

    num_blocks : TYPE, optional
        DESCRIPTION. The default is 14.
    '''
    
    param_names = ["omega", "dectemp", "lr"]
    num_params = len(param_names)
    
    def specific_init(self):
        self.V = [self.compute_V()]

    def compute_V(self):
        
        return ((1-self.param_dict['omega'])[..., None]*self.rep[-1] +\
                   self.param_dict['omega'][..., None]*self.Q[-1])

    def locs_to_pars(self, locs):
        '''
        Parameters
        ----------
        locs : tensor, shape [num_particles, num agents, num_parameters]

        Returns
        -------
        param_dict : dict

        '''
        
        param_dict = {'omega': torch.sigmoid(locs[..., self.param_names.index('omega')]),
                    'dectemp': torch.exp(locs[..., self.param_names.index('dectemp')]),
                    'lr': torch.sigmoid(locs[..., self.param_names.index('lr')])}
        
        # print(locs.shape)
        
        return param_dict
    
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
        
        probs = self.softmax(self.param_dict['dectemp'][..., None]*torch.stack((Vopt1, Vopt2), 2))
        
        
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
            
            self.V.append((1-self.param_dict['omega'])[..., None]*self.rep[-1] + 
                          self.param_dict['omega'][..., None]*self.Q[-1])
            
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
            self.V.append((1-self.param_dict['omega'])[..., None]*self.rep[-1] + 
                          self.param_dict['omega'][..., None]*self.Q[-1])
            
            if len(self.Q) > 20:
                "Free up some memory space"
                self.V[0:-10] = []
                self.Q[0:-10] = []
                self.rep[0:-10] = []

            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = choices
        
class Vbm_nolr(model_master):
    '''
    Parameters
    ----------

    num_blocks : TYPE, optional
        DESCRIPTION. The default is 14.
    '''
    
    param_names = ["omega", "dectemp"]
    num_params = len(param_names)
    
    def specific_init(self):
        self.V = [self.compute_V()]
            
            
    def compute_V(self):
        
        return ((1-self.param_dict['omega'])[..., None]*self.rep[-1] +\
                   self.param_dict['omega'][..., None]*self.Q[-1])

    def locs_to_pars(self, locs):
        '''
        Parameters
        ----------
        locs : tensor, shape [num_particles, num agents, num_parameters]

        Returns
        -------
        param_dict : dict

        '''
        
        param_dict = {'omega': torch.sigmoid(locs[..., self.param_names.index('omega')]),
                    'dectemp': torch.exp(locs[..., self.param_names.index('dectemp')])}
        
        # print(locs.shape)
        
        return param_dict
    
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
        
        probs = self.softmax(self.param_dict['dectemp'][..., None]*torch.stack((Vopt1, Vopt2), 2))
        
        
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
            
            self.V.append((1-self.param_dict['omega'])[..., None]*self.rep[-1] + 
                          self.param_dict['omega'][..., None]*self.Q[-1])
            
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
            
            "----- Compute new V-values for next trial -----"
            self.V.append((1-self.param_dict['omega'])[..., None]*self.rep[-1] + 
                          self.param_dict['omega'][..., None]*self.Q[-1])
            
            if len(self.rep) > 20:
                "Free up some memory space"
                self.V[0:-10] = []
                self.rep[0:-10] = []

            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = choices

class Repbias_lr(model_master):
    
    param_names = ['lr', 
                    'theta_Q',
                    'theta_rep']
    
    num_params = len(param_names)
    
    def specific_init(self):
        "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"
        self.V = [self.compute_V(blocktype = torch.tensor(0.))]
        
    def compute_V(self, **kwargs):
        
        return self.param_dict['theta_rep'][..., None]*self.rep[-1] + \
                self.param_dict['theta_Q'][..., None]*self.Q[-1]
        
    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')])}
        
        return param_dict
    
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
    
    def update(self, choices, outcomes, blocktype, **kwargs):
        '''
        Class Vbm_B(Vbm_1day).
        
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
        
        lr = self.param_dict['lr']
        
        if torch.all(choices == -1) and torch.all(outcomes == -1) and torch.all(blocktype == -1):
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.ones(lr.shape[0], lr.shape[1], self.NA)/self.NA)
            self.Q.append(self.Q[-1])
            
            self.V.append(self.compute_V(blocktype = blocktype))
            
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
            self.V.append(self.compute_V(blocktype = blocktype))

            if len(self.Q) > 20:
                "Free up some memory space"
                self.V[0:-10] = []
                self.Q[0:-10] = []
                self.rep[0:-10] = []

            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = choices
            
            
class Repbias_onlyseq_lr(model_master):
    
    param_names = ['lr', 
                    'theta_Q',
                    'theta_rep']
    
    num_params = len(param_names)
    
    def specific_init(self):
        "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"
        self.V = [self.compute_V(blocktype = torch.zeros(self.num_agents))]
        
    def compute_V(self, blocktype):
        '''
        

        Parameters
        ----------
        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        return self.param_dict['theta_rep'][..., None]*self.rep[-1]*((blocktype == 0).type(torch.int)[None,...,None]) + \
                self.param_dict['theta_Q'][..., None]*self.Q[-1]
        
    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')])}
        
        return param_dict
    
    def compute_probs(self, trial, blocktype, **kwargs):
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
        
        self.V.append(self.compute_V(blocktype = blocktype))
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        probs = self.softmax(torch.stack((Vopt1, Vopt2), 2))
        
        return probs
    
    def update(self, choices, outcomes, blocktype, **kwargs):
        '''
        Class Vbm_B(Vbm_1day).
        
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
        
        lr = self.param_dict['lr']
        
        if torch.all(choices == -1) and torch.all(outcomes == -1) and torch.all(blocktype == -1):
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.ones(lr.shape[0], lr.shape[1], self.NA)/self.NA)
            self.Q.append(self.Q[-1])
            
            self.V.append(self.compute_V(blocktype = blocktype))
            
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
            # self.V.append(self.compute_V(blocktype = blocktype))

            if len(self.Q) > 20:
                "Free up some memory space"
                self.V[0:-10] = []
                self.Q[0:-10] = []
                self.rep[0:-10] = []

            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = choices
            

class Repbias_nolr(model_master):
    
    param_names = ['theta_Q',
                    'theta_rep']
    
    num_params = len(param_names)
    
    def specific_init(self):
        "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"
        self.V = [self.compute_V()]
    
    def compute_V(self):
        
        return self.param_dict['theta_rep'][..., None]*self.rep[-1] + \
                self.param_dict['theta_Q'][..., None]*self.Q[-1]
        
    def locs_to_pars(self, locs):
        param_dict = {'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')])}
        
        return param_dict
    
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
    
    def update(self, choices, outcomes, blocktype, **kwargs):
        '''
        
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
        
        theta_Q = self.param_dict['theta_Q']
        theta_rep = self.param_dict['theta_rep']
        
        if torch.all(choices == -1) and torch.all(outcomes == -1) and torch.all(blocktype == -1):
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.ones(self.num_particles, self.num_agents, self.NA)/self.NA)
            self.V.append(self.compute_V())
            
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
                                        0:4] / seqs_sum[...,None]

            self.rep.append(new_rows.broadcast_to(self.num_particles , self.num_agents, 4))
            
            "----- Compute new V-values for next trial -----"
            self.V.append(self.compute_V())

            if len(self.V) > 20:
                "Free up some memory space"
                self.V[0:-10] = []
                self.rep[0:-10] = []

            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = choices

class Repbias_Conflict_both_onlyseq(Repbias_lr):
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')]),
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, jokertype, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        !! Given participant's inference (i.e. also in random condition) !!
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        # incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
        #                 torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        
        incong_bool = (jokertype == 2).type(torch.int)        
        conflict_value = torch.min(torch.abs(DeltaQ), torch.abs(DeltaRep))
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        seq_bool = (blocktype == 0).type(torch.int)
        
        inc_bonus = incong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
        
        # inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
class Repbias_Conflict_both_onlyseq_nobound(Repbias_lr):
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': locs[..., self.param_names.index('theta_rep')],
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, jokertype, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        !! Given participant's inference (i.e. also in random condition) !!
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        # incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
        #                 torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        
        incong_bool = (jokertype == 2).type(torch.int)        
        conflict_value = torch.min(torch.abs(DeltaQ), torch.abs(DeltaRep))
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        seq_bool = (blocktype == 0).type(torch.int)
        
        inc_bonus = incong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
        
        # inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs

class Repbias_Conflict_both_both(Repbias_lr):
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')]),
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, jokertype, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        !! Given participant's inference (i.e. also in random condition) !!
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        conflict_value = torch.min(torch.abs(DeltaQ), torch.abs(DeltaRep))
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        # seq_bool = (blocktype == 0).type(torch.int)
        
        # inc_bonus = incong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
        
        inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
class Repbias_Conflict_both_both_nobound(Repbias_lr):
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': locs[..., self.param_names.index('theta_rep')],
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, jokertype, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        !! Given participant's inference (i.e. also in random condition) !!
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        conflict_value = torch.min(torch.abs(DeltaQ), torch.abs(DeltaRep))
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        # seq_bool = (blocktype == 0).type(torch.int)
        
        # inc_bonus = incong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
        
        inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
class Repbias_Conflict_nobound_lr(Repbias_lr):
    '''
        nobound: Both θRep and θConflict are unbounded
    '''
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': locs[..., self.param_names.index('theta_rep')],
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, jokertype, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        !! Given participant's inference (i.e. also in random condition) !!
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        conflict_value = torch.min(torch.abs(DeltaQ), torch.abs(DeltaRep))
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        # '''
        #     seq_bool :  0 random condition
        #                 1 sequential condition
        # '''
        # seq_bool = (blocktype == 0).type(torch.int)
        
        # inc_bonus = incong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
        
        inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
class Repbias_Conflict_Repdiff_lr(Repbias_lr):
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')]),
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        conflict_value = torch.abs(DeltaRep)
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        # seq_bool = (blocktype == 0).type(torch.int)
        
        # inc_bonus = incong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
        
        inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
    
class Repbias_Conflict_Repdiff_lr_nobound(Repbias_lr):
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': locs[..., self.param_names.index('theta_rep')],
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        conflict_value = torch.abs(DeltaRep)
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        # seq_bool = (blocktype == 0).type(torch.int)
        
        # inc_bonus = incong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
        
        inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
class Repbias_Conflict_Repdiff_nobound_lr(Repbias_lr):
    '''
        nobound: Both θRep and θConflict are unbounded
    '''
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': locs[..., self.param_names.index('theta_rep')],
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        conflict_value = torch.abs(DeltaRep)
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        # seq_bool = (blocktype == 0).type(torch.int)
        
        # inc_bonus = incong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
        
        inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
class Repbias_Exploiter_Repdiff_lr(Repbias_lr):
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_exploit']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')]),
                    'theta_exploit': locs[..., self.param_names.index('theta_exploit')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        exploit_param = self.param_dict['theta_exploit']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        
        cong_bool = 1 - incong_bool
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        # seq_bool = (blocktype == 0).type(torch.int)
        
        # inc_bonus = incong_bool*seq_bool*opt1_GD*torch.abs(DeltaRep)*exploit_param
        
        inc_bonus = cong_bool*opt1_GD*torch.abs(DeltaRep)*exploit_param

        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
class Repbias_Conflict_Repdiff_onlyseq_lr(Repbias_lr):
    '''
        onlyseq: conflict parameter is fitted only in case of incongruent DTT, based on jokertype variable
    '''
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')]),
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, jokertype, **kwargs):
        '''
        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        # incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
        #                 # torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        incong_bool = (jokertype == 2).type(torch.int)
        conflict_value = torch.abs(DeltaRep)
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        seq_bool = (blocktype == 0).type(torch.int)
        
        inc_bonus = incong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
        
        # inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        "SM[V1, V2] -> [p1, p2], where p1 = σ(V1-V2)"
        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
class Repbias_Conflict_Repdiff_onlyseq_nobound(Repbias_lr):
    '''
        onlyseq: conflict parameter is fitted only in case of incongruent DTT, based on jokertype variable
    '''
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': locs[..., self.param_names.index('theta_rep')],
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, jokertype, **kwargs):
        '''
        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        # incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
        #                 # torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        incong_bool = (jokertype == 2).type(torch.int)
        conflict_value = torch.abs(DeltaRep)
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        seq_bool = (blocktype == 0).type(torch.int)
        
        inc_bonus = incong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
        
        # inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        "SM[V1, V2] -> [p1, p2], where p1 = σ(V1-V2)"
        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
class Repbias_Conflict_Repdiff_onlyseq_onlyseq_lr(Repbias_lr):
    "Fits ONLY Q in random trials"
    '''
        onlyseq_onlyseq: θRep and θConflict are fitted only in repeating-sequence condition
    '''
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')]),
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_V(self, blocktype):
        '''

        Parameters
        ----------
        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        
        seq_bool = (blocktype == 0).type(torch.int)
        
        
        return self.param_dict['theta_rep'][..., None]*self.rep[-1]*seq_bool[None,...,None] + \
                self.param_dict['theta_Q'][..., None]*self.Q[-1]
    
    def compute_probs(self, trial, blocktype, jokertype, **kwargs):
        '''
        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                             self.num_agents)
        
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, 
                                                           self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        # incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
        #                 # torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        incong_bool = (jokertype == 2).type(torch.int)
        conflict_value = torch.abs(DeltaRep)
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        seq_bool = (blocktype == 0).type(torch.int)
        
        inc_bonus = incong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
        
        # inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        "SM[V1, V2] -> [p1, p2], where p1 = σ(V1-V2)"
        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
    
class Repbias_Conflict_onlyseq_onlyseq_lr(Repbias_lr):
    "Fits ONLY Q in random trials"
    '''
        onlyseq_onlyseq: θRep and θConflict are fitted only in repeating-sequence condition
    '''
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')]),
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_V(self, blocktype):
        '''

        Parameters
        ----------
        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        
        seq_bool = (blocktype == 0).type(torch.int)
        
        
        return self.param_dict['theta_rep'][..., None]*self.rep[-1]*seq_bool[None,...,None] + \
                self.param_dict['theta_Q'][..., None]*self.Q[-1]
    
    def compute_probs(self, trial, blocktype, jokertype, **kwargs):
        '''
        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                             self.num_agents)
        
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, 
                                                           self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        # incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
        #                 # torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        incong_bool = (jokertype == 2).type(torch.int)
        conflict_value = torch.min(torch.abs(DeltaRep), torch.abs(DeltaQ))
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        seq_bool = (blocktype == 0).type(torch.int)
        
        inc_bonus = incong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
        
        # inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        "SM[V1, V2] -> [p1, p2], where p1 = σ(V1-V2)"
        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
class Repbias_Conflict_onlyseq_both(Repbias_lr):
    "Fits ONLY Q in random trials"
    '''
        onlyseq_onlyseq: θRep and θConflict are fitted only in repeating-sequence condition
    '''
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')]),
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_V(self, blocktype):
        '''

        Parameters
        ----------
        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        
        seq_bool = (blocktype == 0).type(torch.int)
        
        
        return self.param_dict['theta_rep'][..., None]*self.rep[-1]*seq_bool[None,...,None] + \
                self.param_dict['theta_Q'][..., None]*self.Q[-1]
    
    def compute_probs(self, trial, blocktype, jokertype, **kwargs):
        '''
        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                             self.num_agents)
        
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, 
                                                           self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        # incong_bool = (jokertype == 2).type(torch.int)
        conflict_value = torch.min(torch.abs(DeltaRep), torch.abs(DeltaQ))
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        # seq_bool = (blocktype == 0).type(torch.int)
        
        inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param
        
        # inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        "SM[V1, V2] -> [p1, p2], where p1 = σ(V1-V2)"
        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
    
class Repbias_Conflict_onlyseq_both_nobound(Repbias_lr):
    "Fits ONLY Q in random trials"
    '''
        onlyseq_onlyseq: θRep and θConflict are fitted only in repeating-sequence condition
    '''
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': locs[..., self.param_names.index('theta_rep')],
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_V(self, blocktype):
        '''

        Parameters
        ----------
        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        
        seq_bool = (blocktype == 0).type(torch.int)
        
        
        return self.param_dict['theta_rep'][..., None]*self.rep[-1]*seq_bool[None,...,None] + \
                self.param_dict['theta_Q'][..., None]*self.Q[-1]
    
    def compute_probs(self, trial, blocktype, jokertype, **kwargs):
        '''
        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                             self.num_agents)
        
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, 
                                                           self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        # incong_bool = (jokertype == 2).type(torch.int)
        conflict_value = torch.min(torch.abs(DeltaRep), torch.abs(DeltaQ))
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        # seq_bool = (blocktype == 0).type(torch.int)
        
        inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param
        
        # inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        "SM[V1, V2] -> [p1, p2], where p1 = σ(V1-V2)"
        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
    
class Repbias_Conflict_Repdiff_onlyseq_onlyseq_nobound_lr(Repbias_lr):
    '''
        onlyseq_onlyseq: Fits ONLY Q in random trial. Fits θRep and θConfflict only in repating-sequence condition,
                            both based on jokertypes variable
        nobound: Both θRep and θConflict are unbounded
    '''
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': locs[..., self.param_names.index('theta_rep')],
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_V(self, blocktype):
        '''

        Parameters
        ----------
        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        
        seq_bool = (blocktype == 0).type(torch.int)
        
        
        return self.param_dict['theta_rep'][..., None]*self.rep[-1]*seq_bool[None,...,None] + \
                self.param_dict['theta_Q'][..., None]*self.Q[-1]
    
    def compute_probs(self, trial, blocktype, jokertype, **kwargs):
        '''
        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                             self.num_agents)
        
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, 
                                                           self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        # incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
        #                 # torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        incong_bool = (jokertype == 2).type(torch.int)
        conflict_value = torch.abs(DeltaRep)
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        seq_bool = (blocktype == 0).type(torch.int)
        
        inc_bonus = incong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
        
        # inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        "SM[V1, V2] -> [p1, p2], where p1 = σ(V1-V2)"
        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
    
class Repbias_Conflict_Repdiff_onlyseq_both(Repbias_lr):
    '''
        onlyseq_both: Fits θRep in rep condition and conflict in both conditions
    '''
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')]),
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_V(self, blocktype):
        '''

        Parameters
        ----------
        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        
        seq_bool = (blocktype == 0).type(torch.int)
        
        
        return self.param_dict['theta_rep'][..., None]*self.rep[-1]*seq_bool[None,...,None] + \
                self.param_dict['theta_Q'][..., None]*self.Q[-1]
    
    def compute_probs(self, trial, blocktype, jokertype, **kwargs):
        '''
        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                             self.num_agents)
        
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, 
                                                           self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        # incong_bool = (jokertype == 2).type(torch.int)
        conflict_value = torch.abs(DeltaRep)
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        # seq_bool = (blocktype == 0).type(torch.int)
        
        inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param
        
        # inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        "SM[V1, V2] -> [p1, p2], where p1 = σ(V1-V2)"
        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
    
class Repbias_Conflict_Repdiff_onlyseq_both_nobound(Repbias_lr):
    '''
        onlyseq_both: Fits θRep in rep condition and conflict in both conditions
    '''
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': locs[..., self.param_names.index('theta_rep')],
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_V(self, blocktype):
        '''

        Parameters
        ----------
        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        
        seq_bool = (blocktype == 0).type(torch.int)
        
        
        return self.param_dict['theta_rep'][..., None]*self.rep[-1]*seq_bool[None,...,None] + \
                self.param_dict['theta_Q'][..., None]*self.Q[-1]
    
    def compute_probs(self, trial, blocktype, jokertype, **kwargs):
        '''
        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                             self.num_agents)
        
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, 
                                                           self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        # incong_bool = (jokertype == 2).type(torch.int)
        conflict_value = torch.abs(DeltaRep)
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        # seq_bool = (blocktype == 0).type(torch.int)
        
        inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param
        
        # inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        "SM[V1, V2] -> [p1, p2], where p1 = σ(V1-V2)"
        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
class Repbias_Conflict_onlyseq_onlyseq_nobound_lr(Repbias_lr):
    '''
        onlyseq_onlyseq: Fits ONLY Q in random trial. Fits θRep and θConfflict only in repating-sequence condition,
                            both based on jokertypes variable
        nobound: Both θRep and θConflict are unbounded
    '''
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': locs[..., self.param_names.index('theta_rep')],
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_V(self, blocktype):
        '''

        Parameters
        ----------
        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        
        seq_bool = (blocktype == 0).type(torch.int)
        
        
        return self.param_dict['theta_rep'][..., None]*self.rep[-1]*seq_bool[None,...,None] + \
                self.param_dict['theta_Q'][..., None]*self.Q[-1]
    
    def compute_probs(self, trial, blocktype, jokertype, **kwargs):
        '''
        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                             self.num_agents)
        
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, 
                                                           self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        # incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
        #                 # torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        incong_bool = (jokertype == 2).type(torch.int)
        # conflict_value = torch.abs(DeltaRep)
        conflict_value = torch.min(torch.abs(DeltaQ), torch.abs(DeltaRep))
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        seq_bool = (blocktype == 0).type(torch.int)
        
        inc_bonus = incong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
        
        # inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        "SM[V1, V2] -> [p1, p2], where p1 = σ(V1-V2)"
        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs
    
class Repbias_Conflict_Repdiff_nobound_onlyseq_lr(Repbias_lr):
    '''
        nobound: Both θRep and θConflict are unbounded
        onlyseq: conflict value is fit only in repeating-sequence condition
    '''
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': locs[..., self.param_names.index('theta_rep')],
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        conflict_value = torch.abs(DeltaRep)
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
            seq_bool :  0 random condition
                        1 sequential condition
        '''
        seq_bool = (blocktype == 0).type(torch.int)
        
        inc_bonus = incong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
        
        # inc_bonus = incong_bool*opt1_GD*conflict_value*conflict_param

        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, Vopt2), 2))
        
        return probs

class Repbias_3Q_lr(Repbias_lr):
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_repcong',
                    'theta_repinc']
    
    num_params = len(param_names)
        
    def compute_V(self):
        
        return ((1-self.param_dict['theta_repcong'])[..., None]*0*self.rep[-1] +\
                   self.param_dict['theta_Q'][..., None]*0*self.Q[-1])

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_repcong': torch.exp(locs[..., self.param_names.index('theta_repcong')]),
                    'theta_repinc': torch.exp(locs[..., self.param_names.index('theta_repinc')])}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, jokertype, blockidx, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.

        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 

        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
                    self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
                    self.rep[-1][:, torch.arange(self.num_agents), option2]
        DeltaRepsign = torch.sign(DeltaRep)
        
        '''
            opt1_GD_bool :   1 if option1 is goal-directed response
                             0 if not
        '''
        opt1_GD_bool = (DeltaQ > 0).type(torch.int)
        
        '''
            opt2_GD_bool :   1 if option2 is goal-directed response
                             0 if not
        '''
        opt2_GD_bool = (DeltaQ < 0).type(torch.int)
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        !! Given participant's inference (i.e. also in random condition) !!
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        
        cong_bool = 1 - incong_bool
        
        theta_rep = (self.param_dict['theta_repcong'])*cong_bool +\
                    (self.param_dict['theta_repinc'])*incong_bool
        
        '''
            Value differences of responses
            VGDiff = θ_Q * |ΔQ| + θ_R,i * ΔR, where θ_R,i differs for ΔR>0 and ΔR<0.
        '''
        VGDiff = self.param_dict['theta_Q']*DeltaQ + theta_rep*DeltaRep
        
        Vopt1 = self.param_dict['theta_Q']*self.Q[-1][:, torch.arange(self.num_agents), option1] + \
            theta_rep*self.rep[-1][:, torch.arange(self.num_agents), option1]
            
        Vopt2 = self.param_dict['theta_Q']*self.Q[-1][:, torch.arange(self.num_agents), option2] + \
            theta_rep*self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        assert torch.allclose(VGDiff, Vopt1-Vopt2)
        
        "Set probs 1 -> 1-1e09, 0 -> 1e09"
        probs = self.softmax(torch.stack((Vopt1, Vopt2), 2))
        probs = torch.where(probs == 0, 1e-9, probs)
        probs = torch.where(probs == 1, 1-1e-9, probs)
        
        "assert no probs are negative"
        assert torch.all(probs > 0)
        "assert probs is 0 only where a STT was observed."
        assert torch.all(trial[torch.where(probs==0)[1]] < 10)
        
        "Set p = 0.5 where probs is still 0 (i.e. where DeltaQ == 0 and DeltaRep == 0)"
        probs[torch.where(probs.sum(axis=-1) == 0)[0], 
              torch.where(probs.sum(axis=-1) == 0)[1], :] = 0.5
        
        "Following : A bunch of assert statements"
        "opt1 and opt2 cannot both be the goal-directed response"
        assert torch.all(opt1_GD_bool*opt2_GD_bool == 0)
        
        "Where both booleans are 0, the Q-values must be the same"
        both_no_gd_particle = torch.where((1-opt1_GD_bool)*(1-opt2_GD_bool) == 1)[0]
        both_no_gd_agent = torch.where((1-opt1_GD_bool)*(1-opt2_GD_bool) == 1)[1]
        
        assert torch.all(self.Q[-1][both_no_gd_particle, 
                                    both_no_gd_agent, 
                                    option1[both_no_gd_agent]] == \
            self.Q[-1][both_no_gd_particle, 
                       both_no_gd_agent, 
                       option2[both_no_gd_agent]])
            
        "probabilities must sum to 1"
        assert torch.allclose(probs.sum(axis=-1), torch.ones(probs.shape[0:2]))
        
        "Where opt1_GD_bool == True, the corresponding Q-value must be larger for option 1 than for option 2"
        assert torch.all(self.Q[-1][torch.where(opt1_GD_bool==1)[0], torch.where(opt1_GD_bool==1)[1], option1[torch.where(opt1_GD_bool==1)[1]]] >
            self.Q[-1][torch.where(opt1_GD_bool==1)[0], torch.where(opt1_GD_bool==1)[1], option2[torch.where(opt1_GD_bool==1)[1]]])
        
        assert torch.all(self.Q[-1][torch.where(opt2_GD_bool==1)[0], torch.where(opt2_GD_bool==1)[1], option2[torch.where(opt2_GD_bool==1)[1]]] >
            self.Q[-1][torch.where(opt2_GD_bool==1)[0], torch.where(opt2_GD_bool==1)[1], option1[torch.where(opt2_GD_bool==1)[1]]])
        
        assert probs.shape[0] == self.num_particles
        assert probs.shape[1] == self.num_agents
        assert probs.shape[2] == 2
        
        # if torch.any(torch.where(blocktype==0, trial, 0) > 10) and torch.any(torch.tensor(blockidx) > 5):
        #     ipdb.set_trace()
        
        return probs
    
class Repbias_3Q_nobound_lr(Repbias_lr):
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_repcong',
                    'theta_repinc']
    
    num_params = len(param_names)
        
    def compute_V(self):
        
        return ((1-self.param_dict['theta_repcong'])[..., None]*0*self.rep[-1] +\
                   self.param_dict['theta_Q'][..., None]*0*self.Q[-1])

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_repcong': locs[..., self.param_names.index('theta_repcong')],
                    'theta_repinc': locs[..., self.param_names.index('theta_repinc')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, jokertype, blockidx, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.

        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 

        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
                    self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
                    self.rep[-1][:, torch.arange(self.num_agents), option2]
        DeltaRepsign = torch.sign(DeltaRep)
        
        '''
            opt1_GD_bool :   1 if option1 is goal-directed response
                             0 if not
        '''
        opt1_GD_bool = (DeltaQ > 0).type(torch.int)
        
        '''
            opt2_GD_bool :   1 if option2 is goal-directed response
                             0 if not
        '''
        opt2_GD_bool = (DeltaQ < 0).type(torch.int)
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        !! Given participant's inference (i.e. also in random condition) !!
                        
                        sign(X) can be -1, 0, 1 
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            (0*0 -1)/2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        
        cong_bool = 1 - incong_bool
        
        theta_rep = (self.param_dict['theta_repcong'])*cong_bool +\
                    (self.param_dict['theta_repinc'])*incong_bool
        
        '''
            Value differences of responses
            VGDiff = θ_Q * |ΔQ| + θ_R,i * ΔR, where θ_R,i differs for ΔR>0 and ΔR<0.
        '''
        VGDiff = self.param_dict['theta_Q']*DeltaQ + theta_rep*DeltaRep
        
        Vopt1 = self.param_dict['theta_Q']*self.Q[-1][:, torch.arange(self.num_agents), option1] + \
            theta_rep*self.rep[-1][:, torch.arange(self.num_agents), option1]
            
        Vopt2 = self.param_dict['theta_Q']*self.Q[-1][:, torch.arange(self.num_agents), option2] + \
            theta_rep*self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        assert torch.allclose(VGDiff, Vopt1-Vopt2)
        
        "Set probs 1 -> 1-1e09, 0 -> 1e09"
        probs = self.softmax(torch.stack((Vopt1, Vopt2), 2))
        probs = torch.where(probs == 0, 1e-9, probs)
        probs = torch.where(probs == 1, 1-1e-9, probs)
        
        "assert no probs are negative"
        assert torch.all(probs > 0)
        "assert probs is 0 only where a STT was observed."
        assert torch.all(trial[torch.where(probs==0)[1]] < 10)
        
        "Set p = 0.5 where probs is still 0 (i.e. where DeltaQ == 0 and DeltaRep == 0)"
        probs[torch.where(probs.sum(axis=-1) == 0)[0], 
              torch.where(probs.sum(axis=-1) == 0)[1], :] = 0.5
        
        "Following : A bunch of assert statements"
        "opt1 and opt2 cannot both be the goal-directed response"
        assert torch.all(opt1_GD_bool*opt2_GD_bool == 0)
        
        "Where both booleans are 0, the Q-values must be the same"
        both_no_gd_particle = torch.where((1-opt1_GD_bool)*(1-opt2_GD_bool) == 1)[0]
        both_no_gd_agent = torch.where((1-opt1_GD_bool)*(1-opt2_GD_bool) == 1)[1]
        
        assert torch.all(self.Q[-1][both_no_gd_particle, 
                                    both_no_gd_agent, 
                                    option1[both_no_gd_agent]] == \
            self.Q[-1][both_no_gd_particle, 
                       both_no_gd_agent, 
                       option2[both_no_gd_agent]])
            
        "probabilities must sum to 1"
        assert torch.allclose(probs.sum(axis=-1), torch.ones(probs.shape[0:2]))
        
        "Where opt1_GD_bool == True, the corresponding Q-value must be larger for option 1 than for option 2"
        assert torch.all(self.Q[-1][torch.where(opt1_GD_bool==1)[0], torch.where(opt1_GD_bool==1)[1], option1[torch.where(opt1_GD_bool==1)[1]]] >
            self.Q[-1][torch.where(opt1_GD_bool==1)[0], torch.where(opt1_GD_bool==1)[1], option2[torch.where(opt1_GD_bool==1)[1]]])
        
        assert torch.all(self.Q[-1][torch.where(opt2_GD_bool==1)[0], torch.where(opt2_GD_bool==1)[1], option2[torch.where(opt2_GD_bool==1)[1]]] >
            self.Q[-1][torch.where(opt2_GD_bool==1)[0], torch.where(opt2_GD_bool==1)[1], option1[torch.where(opt2_GD_bool==1)[1]]])
        
        assert probs.shape[0] == self.num_particles
        assert probs.shape[1] == self.num_agents
        assert probs.shape[2] == 2
        
        # if torch.any(torch.where(blocktype==0, trial, 0) > 10) and torch.any(torch.tensor(blockidx) > 5):
        #     ipdb.set_trace()
        
        return probs
        
class Repbias_Conflict_nolr(Repbias_nolr):
    
    param_names = ['theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')]),
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, **kwargs):
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
        
        raise Exception("Not up to date anymore.")
        
        conflict_param = self.param_dict['theta_conflict']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
        Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
        Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(DeltaQ) can be -1, 0, 1.
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        conflict_value = torch.min(torch.abs(DeltaQ), torch.abs(DeltaRep))
        
        '''
        opt1_GD :   1 if option1 is goal-directed response
                    -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
        seq_bool :  0 random condition
                    1 sequential condition
        '''
        seq_bool = (blocktype == 0).type(torch.int)
        
        inc_bonus = incong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
            
        probs = self.softmax(torch.stack((Vopt1 + inc_bonus, 
                                          Vopt2), 2))
            
        return probs

class Repbias_Interaction_lr(Repbias_lr):
    
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_interact']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')]),
                    'theta_interact': locs[..., self.param_names.index('theta_interact')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, **kwargs):
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
        
        interaction_param = self.param_dict['theta_interact']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
        Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
        Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        # conflict_value = torch.min(torch.abs(DeltaQ), torch.abs(DeltaRep))
        
        '''
        opt1_GD :   1 if option1 is goal-directed response
                    -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ >= 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
        seq_bool :  0 random condition
                    1 sequential condition
        '''
        # seq_bool = (blocktype == 0).type(torch.int)
        
        # bonus = opt1_GD*conflict_value*conflict_param

        probs = self.softmax(torch.stack((Vopt1 + interaction_param*opt1_GD*torch.abs(DeltaQ*DeltaRep), 
                                          Vopt2), 2))

        return probs
        
class Repbias_Interaction_nolr(Repbias_nolr):
    
    param_names = ['theta_Q',
                    'theta_rep',
                    'theta_interact']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')]),
                    'theta_interact': locs[..., self.param_names.index('theta_interact')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, **kwargs):
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
        
        interaction_param = self.param_dict['theta_interact']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
        Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
        Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        # conflict_value = torch.min(torch.abs(DeltaQ), torch.abs(DeltaRep))
        
        '''
        opt1_GD :   1 if option1 is goal-directed response
                    -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
        seq_bool :  0 random condition
                    1 sequential condition
        '''
        # seq_bool = (blocktype == 0).type(torch.int)
        
        # bonus = opt1_GD*conflict_value*conflict_param

        probs = self.softmax(torch.stack((Vopt1 + interaction_param*DeltaQ*DeltaRep, 
                                          Vopt2), 2))

        return probs

class Repbias_CongConflict_lr(Repbias_lr):
    param_names = ['lr',
                    'theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')]),
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, **kwargs):
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
        
        conflict_param = self.param_dict['theta_conflict']

        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(DeltaQ) can be -1, 0, 1.
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        
        assert torch.all(incong_bool > -1) and torch.all(incong_bool < 2)
        
        cong_bool = 1-incong_bool
        conflict_value = torch.min(torch.abs(DeltaQ), torch.abs(DeltaRep))
        
        '''
        opt1_GD :   1 if option1 is goal-directed response
                    -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
        seq_bool :  0 random condition
                    1 sequential condition
        '''
        seq_bool = (blocktype == 0).type(torch.int)
        
        cong_bonus = cong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
            
        probs = self.softmax(torch.stack((Vopt1 + cong_bonus, 
                                          Vopt2), 2))

        return probs

class Repbias_CongConflict_nolr(Repbias_nolr):
    param_names = ['theta_Q',
                    'theta_rep',
                    'theta_conflict']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'theta_Q': torch.exp(locs[..., self.param_names.index('theta_Q')]),
                    'theta_rep': torch.exp(locs[..., self.param_names.index('theta_rep')]),
                    'theta_conflict': locs[..., self.param_names.index('theta_conflict')]}
    
        return param_dict
    
    def compute_probs(self, trial, blocktype, **kwargs):
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
        
        conflict_param = self.param_dict['theta_conflict']

        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            Rep(option1) - Rep(option2) --> DeltaRep > 0 if Rep(option1) > Rep(option2)
        '''
        DeltaRep = self.rep[-1][:, torch.arange(self.num_agents), option1] - \
            self.rep[-1][:, torch.arange(self.num_agents), option2]
        
        '''
        incong_bool :   0 if congruent trial
                        1 if incongruent trial
                        
                        sign(DeltaQ) can be -1, 0, 1.
                        --> (1*1 -1)/-2 = 0
                        
                            (0*1 -1)/-2 = 0.5 -> 0
                            (1*0 -1)/-2 = 0.5 -> 0
                            
                            (1*-1 -1)/-2 = 1
                            (-1*1 -1)/-2 = 1
        '''
        incong_bool = ((torch.sign(DeltaQ).type(torch.int) * 
                        torch.sign(DeltaRep).type(torch.int) - 1)*-0.5).type(torch.int)
        assert torch.all(incong_bool > -1) and torch.all(incong_bool < 2)
        
        cong_bool = 1-incong_bool
        conflict_value = torch.min(torch.abs(DeltaQ), torch.abs(DeltaRep))
        
        '''
        opt1_GD :   1 if option1 is goal-directed response
                    -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        '''
        seq_bool :  0 random condition
                    1 sequential condition
        '''
        seq_bool = (blocktype == 0).type(torch.int)
        
        cong_bonus = cong_bool*seq_bool*opt1_GD*conflict_value*conflict_param
            
        probs = self.softmax(torch.stack((Vopt1 + cong_bonus, 
                                          Vopt2), 2))

        return probs


class OnlyQ_lr(model_master):
    param_names = ['lr',
                    'theta_Qcong',
                    'theta_Qrand',
                    'theta_Qinc']
    
    num_params = len(param_names)
    NA = 4 # no. of possible actions
    # num_blocks = 14
    # trials = 480*num_blocks
    BAD_CHOICE = -2

    def specific_init(self):
        pass

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Qcong': torch.exp(locs[..., self.param_names.index('theta_Qcong')]),
                    'theta_Qrand': torch.exp(locs[..., self.param_names.index('theta_Qrand')]),
                    'theta_Qinc': torch.exp(locs[..., self.param_names.index('theta_Qinc')])}
    
        return param_dict
    
    def compute_probs(self, trial, jokertype, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        theta_Qrand = self.param_dict['theta_Qrand']
        theta_Qcong = self.param_dict['theta_Qcong']
        theta_Qinc = self.param_dict['theta_Qinc']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.Q[-1], option1)
        Vopt1 = theta_Qrand[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                      self.num_agents, 
                                                                                      1) * (jokertype == 0).type(torch.int)[None,..., None] +\
                theta_Qcong[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                      self.num_agents, 
                                                                                      1) * (jokertype == 1).type(torch.int)[None,..., None] +\
                theta_Qinc[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                     self.num_agents, 
                                                                                     1) * (jokertype == 2).type(torch.int)[None,..., None]
        
        _, mask = self.Qoutcomp(self.Q[-1], option2)
        Vopt2 = theta_Qrand[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents, 1) * (jokertype == 0).type(torch.int)[None,..., None] +\
                theta_Qcong[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents, 1) * (jokertype == 1).type(torch.int)[None,..., None] +\
                theta_Qinc[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents, 1) * (jokertype == 2).type(torch.int)[None,..., None]
        
        probs = self.softmax(torch.stack((Vopt1[:,:,0], Vopt2[:,:,0]),2))
        
        return probs
    
    def update(self, choices, outcomes, blocktype, trialstimulus, jokertype, **kwargs):
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
        
        lr = self.param_dict['lr']
        
        if torch.all(choices == -1) and torch.all(outcomes == -1) and torch.all(blocktype == -1):
            "Set previous actions to -1 because it's the beginning of a new block"
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

            self.Q.append(self.Q[-1])
            
        else:
            "----- Update GD-values -----"
            Qout, mask = self.Qoutcomp(self.Q[-1], choices)
            Qnew = self.Q[-1] + lr[..., None]*(outcomes[None,...,None]-Qout)*mask
            self.Q.append(Qnew)
            
            # print(Qnew.mean(axis=1))
            
            if len(self.Q) > 20:
                "Free up memory space"
                self.Q[0:-10] = []

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
            
        "Q and"
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.NA)] # Goal-Directed Q-Values

class OnlyQ_Qdiff_onlyseq_lr(model_master):
    '''
        Learns Q-values for Random, and difference Random-Congruent, and Congruent-Incongruent
        Is given the jokertype directly.
    '''
    
    param_names = ['lr',
                    'theta_Q_rand',
                    'theta_Q_congdiff',
                    'theta_Q_conflict']
    
    num_params = len(param_names)
    NA = 4 # no. of possible actions
    # num_blocks = 14
    # trials = 480*num_blocks
    BAD_CHOICE = -2

    def specific_init(self):
        pass

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Q_rand': torch.exp(locs[..., self.param_names.index('theta_Q_rand')]),
                    'theta_Q_congdiff': torch.exp(locs[..., self.param_names.index('theta_Q_congdiff')]),
                    'theta_Q_conflict': locs[..., self.param_names.index('theta_Q_conflict')]}
    
        return param_dict

    def compute_probs(self, trial, blocktype, jokertype, **kwargs):
        '''
        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        Q_param = self.param_dict['theta_Q_rand']
        cong_param = self.param_dict['theta_Q_congdiff']
        incong_param = self.param_dict['theta_Q_conflict']
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(torch.zeros((self.num_particles, self.num_agents, 4)), option1)
        Vopt1 = (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)*Q_param
        
        _, mask = self.Qoutcomp(torch.zeros((self.num_particles, self.num_agents, 4)), option2)
        Vopt2 = self.Q[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)*Q_param
        
        assert Q_param.shape == Vopt1.shape
        
        '''
            Q(option1) - Q(option2) --> DeltaQ > 0 if Q(option1) > Q(option2)
        '''
        DeltaQ = self.Q[-1][:, torch.arange(self.num_agents), option1] - \
            self.Q[-1][:, torch.arange(self.num_agents), option2]

        '''
            cong_bin
                1 when congruent
                -1 when incongruent
        '''
        incong_bool = (jokertype == 2).type(torch.int)
        cong_bin = (jokertype == 1).type(torch.int) - incong_bool
        
        '''
            opt1_GD :   1 if option1 is goal-directed response
                        -1 if option2 is goal-directed response
        '''
        opt1_GD = (DeltaQ > 0).type(torch.int) - (DeltaQ < 0).type(torch.int)
        
        # '''
        #     seq_bool :  0 random condition
        #                 1 sequential condition
        # '''
        # seq_bool = (blocktype == 0).type(torch.int)
        
        GD_bonus = opt1_GD*(cong_param*cong_bin - incong_param*incong_bool)
        
        "SM[V1, V2] -> [p1, p2], where p1 = σ(V1-V2)"
        probs = self.softmax(torch.stack((Vopt1 + GD_bonus, Vopt2), 2))
        
        return probs
    
    def update(self, choices, outcomes, blocktype, trialstimulus, jokertype, **kwargs):
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
        
        lr = self.param_dict['lr']
        
        if torch.all(choices == -1) and torch.all(outcomes == -1) and torch.all(blocktype == -1):
            "Set previous actions to -1 because it's the beginning of a new block"
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

            self.Q.append(self.Q[-1])
            
        else:
            "----- Update GD-values -----"
            Qout, mask = self.Qoutcomp(self.Q[-1], choices)
            Qnew = self.Q[-1] + lr[..., None]*(outcomes[None,...,None]-Qout)*mask
            self.Q.append(Qnew)
            
            # print(Qnew.mean(axis=1))
            
            if len(self.Q) > 20:
                "Free up memory space"
                self.Q[0:-10] = []

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
            
        "Q and"
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.NA)] # Goal-Directed Q-Values
        
        
class OnlyQ_nolr(model_master):
    '''
        Learns independent Q-values for each jokertpye.
        Is given the jokertype directly.
    '''
    param_names = ['theta_Qcong',
                    'theta_Qrand',
                    'theta_Qinc']
    
    num_params = len(param_names)
    NA = 4 # no. of possible actions
    # num_blocks = 14
    # trials = 480*num_blocks
    BAD_CHOICE = -2

    def specific_init(self):
        pass

    def locs_to_pars(self, locs):
        param_dict = {'theta_Qcong': torch.exp(locs[..., self.param_names.index('theta_Qcong')]),
                    'theta_Qrand': torch.exp(locs[..., self.param_names.index('theta_Qrand')]),
                    'theta_Qinc': torch.exp(locs[..., self.param_names.index('theta_Qinc')])}
    
        return param_dict
    
    def compute_probs(self, trial, jokertype, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        theta_Qrand = self.param_dict['theta_Qrand']
        theta_Qcong = self.param_dict['theta_Qcong']
        theta_Qinc = self.param_dict['theta_Qinc']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.Q[-1], option1)
        Vopt1 = theta_Qrand[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                      self.num_agents, 
                                                                                      1) * (jokertype == 0).type(torch.int)[None,..., None] +\
                theta_Qcong[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                      self.num_agents, 
                                                                                      1) * (jokertype == 1).type(torch.int)[None,..., None] +\
                theta_Qinc[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                     self.num_agents, 
                                                                                     1) * (jokertype == 2).type(torch.int)[None,..., None]
        
        _, mask = self.Qoutcomp(self.Q[-1], option2)
        Vopt2 = theta_Qrand[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents, 1) * (jokertype == 0).type(torch.int)[None,..., None] +\
                theta_Qcong[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents, 1) * (jokertype == 1).type(torch.int)[None,..., None] +\
                theta_Qinc[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents, 1) * (jokertype == 2).type(torch.int)[None,..., None]
        
        probs = self.softmax(torch.stack((Vopt1[:,:,0], Vopt2[:,:,0]),2))
        
        return probs
    
    def update(self, choices, outcomes, blocktype, trialstimulus, jokertype, **kwargs):
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
        
        
        if torch.all(choices == -1) and torch.all(outcomes == -1) and torch.all(blocktype == -1):
            "Set previous actions to -1 because it's the beginning of a new block"
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

            self.Q.append(self.Q[-1])
            
        else:
            
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
            
        "Q and"
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.NA)] # Goal-Directed Q-Values
        
class Q_seqimpact_lr(model_master):
    param_names = ['lr',
                    'theta_Qrand',
                    'theta_seq']
    
    num_params = len(param_names)
    NA = 4 # no. of possible actions
    # num_blocks = 14
    # trials = 480*num_blocks
    BAD_CHOICE = -2

    def specific_init(self):
        pass

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Qrand': torch.exp(locs[..., self.param_names.index('theta_Qrand')]),
                    'theta_seq': torch.exp(locs[..., self.param_names.index('theta_seq')])}
    
        return param_dict
    
    def compute_probs(self, trial, jokertype, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        jokertype : list
            DTT Types
            -1/0/1/2 : no joker/random/congruent/incongruent

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        theta_Qrand = self.param_dict['theta_Qrand']
        theta_seq = self.param_dict['theta_seq']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.Q[-1], option1)
        Vopt1 = theta_Qrand[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                      self.num_agents, 
                                                                                      1) * (jokertype == 0).type(torch.int)[None,..., None] +\
                theta_seq[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                      self.num_agents, 
                                                                                      1) * (jokertype == 1).type(torch.int)[None,..., None] -\
                theta_seq[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                     self.num_agents, 
                                                                                     1) * (jokertype == 2).type(torch.int)[None,..., None]
        
        _, mask = self.Qoutcomp(self.Q[-1], option2)
        Vopt2 = theta_Qrand[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                      self.num_agents, 
                                                                                      1) * (jokertype == 0).type(torch.int)[None,..., None] +\
                theta_seq[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                      self.num_agents, 
                                                                                      1) * (jokertype == 1).type(torch.int)[None,..., None] -\
                theta_seq[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                     self.num_agents, 
                                                                                     1) * (jokertype == 2).type(torch.int)[None,..., None]
        
        probs = self.softmax(torch.stack((Vopt1[:,:,0], Vopt2[:,:,0]),2))
        
        return probs
    
    def update(self, choices, outcomes, blocktype, trialstimulus, **kwargs):
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
        
        lr = self.param_dict['lr']
        
        if torch.all(choices == -1) and torch.all(outcomes == -1) and torch.all(blocktype == -1):
            "Set previous actions to -1 because it's the beginning of a new block"
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

            self.Q.append(self.Q[-1])
            
        else:
            "----- Update GD-values -----"
            Qout, mask = self.Qoutcomp(self.Q[-1], choices)
            Qnew = self.Q[-1] + lr[..., None]*(outcomes[None,...,None]-Qout)*mask
            self.Q.append(Qnew)
            
            # print(Qnew.mean(axis=1))
            
            if len(self.Q) > 20:
                "Free up memory space"
                self.Q[0:-10] = []

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
            
        "Q and"
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.NA)] # Goal-Directed Q-Values
        
class Q_seqimpact_nolr(model_master):
    param_names = ['theta_Qrand',
                    'theta_seq']
    
    num_params = len(param_names)
    NA = 4 # no. of possible actions
    # num_blocks = 14
    # trials = 480*num_blocks
    BAD_CHOICE = -2

    def specific_init(self):
        pass

    def locs_to_pars(self, locs):
        param_dict = {'theta_Qrand': torch.exp(locs[..., self.param_names.index('theta_Qrand')]),
                    'theta_seq': torch.exp(locs[..., self.param_names.index('theta_seq')])}
    
        return param_dict
    
    def compute_probs(self, trial, jokertype, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        jokertype : list
            DTT Types
            -1/0/1/2 : no joker/random/congruent/incongruent

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 
            
        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        theta_Qrand = self.param_dict['theta_Qrand']
        theta_seq = self.param_dict['theta_seq']
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.Q[-1], option1)
        Vopt1 = theta_Qrand[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                      self.num_agents, 
                                                                                      1) * (jokertype == 0).type(torch.int)[None,..., None] +\
                theta_seq[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                      self.num_agents, 
                                                                                      1) * (jokertype == 1).type(torch.int)[None,..., None] -\
                theta_seq[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                     self.num_agents, 
                                                                                     1) * (jokertype == 2).type(torch.int)[None,..., None]
        
        _, mask = self.Qoutcomp(self.Q[-1], option2)
        Vopt2 = theta_Qrand[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                      self.num_agents, 
                                                                                      1) * (jokertype == 0).type(torch.int)[None,..., None] +\
                theta_seq[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                      self.num_agents, 
                                                                                      1) * (jokertype == 1).type(torch.int)[None,..., None] -\
                theta_seq[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                     self.num_agents, 
                                                                                     1) * (jokertype == 2).type(torch.int)[None,..., None]
        
        probs = self.softmax(torch.stack((Vopt1[:,:,0], Vopt2[:,:,0]),2))
        
        return probs
    
    def update(self, choices, outcomes, blocktype, trialstimulus, **kwargs):
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
        
        
        if torch.all(choices == -1) and torch.all(outcomes == -1) and torch.all(blocktype == -1):
            "Set previous actions to -1 because it's the beginning of a new block"
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

        else:
            
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
            
        "Q and"
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.NA)] # Goal-Directed Q-Values
        
        
        
class Q_seqimpact_conflict_lr(model_master):
    param_names = ['lr',
                    'theta_Qrand',
                    'theta_seq',
                    'theta_conflict']
    
    num_params = len(param_names)
    NA = 4 # no. of possible actions
    # num_blocks = 14
    # trials = 480*num_blocks
    BAD_CHOICE = -2

    def specific_init(self):
        pass

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., self.param_names.index('lr')]),
                    'theta_Qrand': torch.exp(locs[..., self.param_names.index('theta_Qrand')]),
                    'theta_seq': torch.exp(locs[..., self.param_names.index('theta_seq')]),
                    'theta_conflict': torch.exp(locs[..., self.param_names.index('theta_conflict')])}
    
        return param_dict
    
    def compute_probs(self, trial, jokertype, **kwargs):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        jokertype : list
            DTT Types
            -1/0/1/2 : no joker/random/congruent/incongruent

        blocktype : torch.tensor with shape [num_agents]
            0/1 : sequential/ random 

        jokertype : -1/0/1/2 no joker/random/congruent/incongruent

        Returns
        -------
        probs : tensor with shape [num_particles, num_agents, 2]
            [0.5, 0.5] in the corresponding row in case of single-target trial.
            probs of response option1 and response option2 in case of dual-target trial.

        '''
        
        theta_Qrand = self.param_dict['theta_Qrand']
        theta_seq = self.param_dict['theta_seq']
        theta_conflict = self.param_dict['theta_conflict']
        
        raise Exception("Not yet implemented.")
        
        option1, option2 = self.find_resp_options(trial)
        
        _, mask = self.Qoutcomp(self.Q[-1], option1)
        Vopt1 = theta_Qrand[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                      self.num_agents, 
                                                                                      1) * (jokertype == 0).type(torch.int)[None,..., None] +\
                theta_seq[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                      self.num_agents, 
                                                                                      1) * (jokertype == 1).type(torch.int)[None,..., None] -\
                theta_seq[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                     self.num_agents, 
                                                                                     1) * (jokertype == 2).type(torch.int)[None,..., None]
        
        _, mask = self.Qoutcomp(self.Q[-1], option2)
        Vopt2 = theta_Qrand[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                      self.num_agents, 
                                                                                      1) * (jokertype == 0).type(torch.int)[None,..., None] +\
                theta_seq[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                      self.num_agents, 
                                                                                      1) * (jokertype == 1).type(torch.int)[None,..., None] -\
                theta_seq[..., None] * (self.Q[-1][torch.where(mask == 1)]).reshape(self.num_particles, 
                                                                                     self.num_agents, 
                                                                                     1) * (jokertype == 2).type(torch.int)[None,..., None]
        
        probs = self.softmax(torch.stack((Vopt1[:,:,0], Vopt2[:,:,0]),2))
        
        return probs
    
    def update(self, choices, outcomes, blocktype, trialstimulus, **kwargs):
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
        
        lr = self.param_dict['lr']
        
        if torch.all(choices == -1) and torch.all(outcomes == -1) and torch.all(blocktype == -1):
            "Set previous actions to -1 because it's the beginning of a new block"
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

            self.Q.append(self.Q[-1])
            
        else:
            "----- Update GD-values -----"
            Qout, mask = self.Qoutcomp(self.Q[-1], choices)
            Qnew = self.Q[-1] + lr[..., None]*(outcomes[None,...,None]-Qout)*mask
            self.Q.append(Qnew)
            
            # print(Qnew.mean(axis=1))
            
            if len(self.Q) > 20:
                "Free up memory space"
                self.Q[0:-10] = []

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
            
        "Q and"
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.NA)] # Goal-Directed Q-Values
        
class Coinflip_test(model_master):
    param_names = ['heads_p']
    
    num_params = len(param_names)
    NA = 4 # no. of possible actions
    # num_blocks = 14
    # trials = 480*num_blocks
    BAD_CHOICE = -2

    def specific_init(self):
        pass

    def locs_to_pars(self, locs):
        param_dict = {'heads_p': torch.sigmoid(locs[..., 0])}
        
        return param_dict
    
    def compute_probs(self, **kwargs):
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
        
        
        # print(self.param_dict['heads_p'].shape)
        
        if self.param_dict['heads_p'].shape[0] == 1:
            probs = torch.cat((self.param_dict['heads_p'].T, 1-self.param_dict['heads_p'].T), axis = 1)
            
        else:
            probs = torch.stack((self.param_dict['heads_p'], 1-self.param_dict['heads_p']), axis = -1)
            
        assert probs.shape[-1] == 2
        if self.param_dict['heads_p'].shape[0] > 1:
            assert probs.shape[0] == self.num_particles
            assert probs.shape[1] == self.num_agents
        
        return probs
        
    def reset(self, locs):
        self.param_dict = self.locs_to_pars(locs)
        
        self.num_particles = locs.shape[0]
        self.num_agents = locs.shape[1]
        
        
class Bullshitmodel(Repbias_lr):
    
    param_names = ['lr',
                    'theta_rep',
                    'theta_Q', 
                    'param3', 
                    'param4',
                    'param5']
    
    num_params = len(param_names)

    def locs_to_pars(self, locs):
        param_dict = {'lr': torch.sigmoid(locs[..., 0]),
                    'theta_rep': torch.exp(locs[..., 1]),
                    'theta_Q': torch.exp(locs[..., 2]),
                    'param3': locs[..., 3],
                    'param4': locs[..., 4],
                    'param5': locs[..., 5]}

        return param_dict
    
    def compute_probs(self, trial, blocktype, **kwargs):
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
        
        probs = self.softmax(torch.stack((Vopt1 * 0, 
                                          Vopt1 * 0 + 1.5), 2))

        # print(probs)

        return probs
    
    
class RTmodel_nolearning(model_master):
    param_names = ['RT_seq_mu',
                   'RT_seq_std',
                   'RT_diff_mu',
                   'RT_diff_std']
    
    num_params = len(param_names)

    def specific_init(self):
        self.V = [] # compatibility variable with no function
        
    def compute_V(self):
        # compatibility method with no function
        return []

    def locs_to_pars(self, locs):
        param_dict = {'RT_seq_mu': locs[..., 0],
                      'RT_seq_std': torch.exp(locs[..., 1]),
                      'RT_diff_mu': locs[..., 2],
                      'RT_diff_std': torch.exp(locs[..., 2])}

        return param_dict
    
    def update(self, *args, **kwargs):
        
        pass
    
    def stt_action(self, trial, blocktype, jokertype):
        
        loc, scale = self.compute_probs(trial = trial, blocktype = blocktype)
        
        current_RT = torch.distributions.normal.Normal(loc = loc, scale = scale).sample()[0, :]
        
        return current_RT
    
    def compute_probs(self, trial, blocktype, **kwargs):
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
        
        mean = self.param_dict['RT_seq_mu'] + self.param_dict['RT_diff_mu']*(blocktype==1).type(torch.int)
        variance = self.param_dict['RT_seq_std']**2 + (self.param_dict['RT_diff_std']**2)*(blocktype==1).type(torch.int)

        return mean, torch.sqrt(variance)