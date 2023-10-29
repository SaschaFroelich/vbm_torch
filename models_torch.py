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

    num_params = 3
    param_names = ["omega", "dectemp", "lr"]
    NA = 4 # no. of possible actions
    
    def __init__(self, omega, dectemp, lr, k, Q_init, num_blocks = 14):
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
            
        num_blocks : TYPE, optional
            DESCRIPTION. The default is 14.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        if num_blocks != 1:
            if num_blocks%2 != 0:
                raise Exception("num_blocks must be an even value.")
        
        "Setup"
        self.num_particles = omega.shape[0]
        self.num_agents = omega.shape[1]
        assert(Q_init.shape == (self.num_particles, self.num_agents, 4))
        self.trials = 480*num_blocks
        self.num_blocks = num_blocks
        
        self.errorrates_stt = torch.rand(1)*0.1
        self.errorrates_dtt = torch.rand(1)*0.2
        
        "--- Latent variables ---"
        self.par_dict = {}
        self.par_dict['omega'] = omega
        self.par_dict['dectemp'] = dectemp # dectemp > 0 for softmax function
        self.par_dict['lr'] = lr
        self.omega = omega
        self.dectemp = dectemp
        self.lr = lr
        "--- --- --- ---"
        
        "K"
        self.k = k
        self.BAD_CHOICE = -2
        
        "Q and rep"
        self.Q_init = Q_init
        self.Q = [Q_init] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, 4)*0.25] # habitual values (repetition values)
        
        self.V = [((1-self.omega)[..., None]*self.rep[-1] + self.omega[..., None]*self.Q[-1])]
        
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-2 in seq_counter for errors"
        "Dimensions are [blocktypes, pppchoice, ppchoice, pchoice, choice, agent]"
        self.init_seq_counter = self.k / 4 * np.ones((self.num_agents, 2, 6, 6, 6, 6))
        
        self.seq_counter = self.init_seq_counter.clone().detach()


    def locs_to_pars(self, locs):
        self.par_dict = {"omega": torch.sigmoid(locs[..., 0]),
                    "dectemp": torch.exp(locs[..., 1]),
                    "lr": torch.sigmoid(locs[..., 2])}
        
        return self.par_dict
    
    def compute_probs(self, trial, day):
        '''

        Parameters
        ----------
        trial : tensor with shape [num_agents]
            DESCRIPTION.
            
        day : int
            Day of experiment.

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
        
    def Qoutcomp(self, Qin, choices):
        '''        

        Parameters
        ----------
        Qin : tensor with shape [num_particles, num_agents, 4]
            DESCRIPTION.
        choices : TYPE
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
        
        Qin = Qin.type(torch.double)
        
        # print(Qin.ndim)
        # if len(Qin.shape) == 2:
        #     Qin = Qin[None, ...]
            
        # elif len(Qin.shape) == 3:
        #     pass
        
        # else:
        #     ipdb.set_trace()
        #     raise Exception("Fehla, digga!")
        
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
        p_actions = sm(self.dectemp[..., None]*z)
            
        return p_actions

    def find_resp_options(self, stimulus_mat):
        '''
        Given a dual-target stimulus (e.g. 12, 1-indexed), this function returns the two response
        options in 0-indexing. E.g.: stimulus_mat=14 -> option1_python = 0, option1_python = 3

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
        
        #assert(torch.is_tensor(stimulus_mat))
        option2_python = ((stimulus_mat % 10) - 1).type(torch.int)
        option1_python = (((stimulus_mat - (stimulus_mat % 10)) / 10) -1).type(torch.int)
        
        # if option2_python.ndim == 2 and option2_python.shape[0] == 1:
        #     option1_python = torch.squeeze(option1_python)
        #     option2_python = torch.squeeze(option2_python)
            
        #assert(option1_python.ndim == 1)
        #assert(option2_python.ndim == 1)
        return option1_python, option2_python

    def choose_action(self, trial, day):
        '''
        This method is choose_action always only executed for a single agent, since
        simulation of data should NOT be done in batch (could break some functions).
        ALso only execute for num_particles == 1.
        
        Parameters
        ----------
        trial : tensor with shape [1] 
            Contains stimulus trial. 1-indexed.
            
        day : int
            Day of experiment.

        Returns
        -------
        tensor with shape ()
            Chosen action of agent. 0-indexed.
            -2 = error

        '''
        
        if trial < 10:
            "Single-target trial"
            choice_python = trial-1
            cond = (torch.rand(1) < self.errorrates_stt) 
            choice_python = cond * self.BAD_CHOICE + ~cond * choice_python
                            
        
        elif trial > 10:
            "Dual-target trial"
            option1, option2 = self.find_resp_options(trial)
            
            "[0, :] to choose 0th particle"
            choice_sample = torch.distributions.categorical.Categorical(probs=self.compute_probs(trial, day)).sample()[0, :]

            choice_python = option2*choice_sample + option1*(1-choice_sample)
            
            cond = (torch.rand(1) < self.errorrates_dtt)
            choice_python =  cond * self.BAD_CHOICE + ~cond * choice_python
            
        "Squeeze because if trial > 10 ndim of choice_python is 1 (and not 0)"
        return torch.squeeze(choice_python.clone().detach()).type('torch.LongTensor')

    def update(self, choices, outcomes, blocktype, **kwargs):
        '''

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
        
        assert choices.ndim == 1, "choices must have shape (num_agents)."
        
        if torch.all(choices == -1) and torch.all(outcomes == -1) and torch.all(blocktype == -1):
            "Set previous actions to -1 because it's the beginning of a new block"
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.ones(self.lr.shape[0], self.lr.shape[1], self.NA)/self.NA)
            self.Q.append(self.Q[-1])
            
            self.V.append((1-self.omega)[..., None]*self.rep[-1] + self.omega[..., None]*self.Q[-1])
            
        else:
            "----- Update GD-values -----"
            Qout, mask = self.Qoutcomp(self.Q[-1], choices)
            Qnew = self.Q[-1] + self.lr[..., None]*(outcomes[None,...,None]-Qout)*mask
            
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
            self.V.append((1-self.omega)[..., None]*self.rep[-1] + self.omega[..., None]*self.Q[-1])
            
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
        _ = self.locs_to_pars(locs)
        
        self.num_particles = locs.shape[0]
        self.num_agents = locs.shape[1]
        
        "Latent Variables"
        self.omega = self.par_dict["omega"]
        self.dectemp = self.par_dict["dectemp"]
        self.lr = self.par_dict["lr"]
        
        "K"
        # self.k = kwargs["k"]
            
        "Q and rep"
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.NA)] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, 4)*0.25] # habitual values (repetition values)
        
        "Compute V"        
        self.V.append((1-self.omega)[..., None]*self.rep[-1] + self.omega[..., None]*self.Q[-1])
        
        "Sequence Counter"
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-2 in seq_counter for errors"
        "Dimensions are [blocktypes, pppchoice, ppchoice, pchoice, choice, agent]"
        self.init_seq_counter = self.k / 4 * np.ones((self.num_agents, 2, 6, 6, 6, 6))
        
        self.seq_counter = self.init_seq_counter.clone().detach()
        
        
class Vbm_B(Vbm):
    
    num_params = 6
    param_names = ["lr_day1", 
                    "theta_Q_day1", 
                    "theta_rep_day1", 
                    "lr_day2", 
                    "theta_Q_day2", 
                    "theta_rep_day2"]
    
    NA = 4 # no. of possible actions
    
    def __init__(self,
                 lr_day1,
                 theta_Q_day1,
                 theta_rep_day1,
                 lr_day2,
                 theta_Q_day2,
                 theta_rep_day2,
                 k,
                 Q_init,
                 num_blocks = 14):
        
        if num_blocks != 1:
            if num_blocks%2 != 0:
                raise Exception("num_blocks must be an even value.")

        "Setup"
        self.num_particles = lr_day1.shape[0]
        self.num_agents = lr_day1.shape[1]
        self.dectemp = torch.tensor([[1.]]) # For softmax function
        
        self.trials = 480*num_blocks
        self.num_blocks = num_blocks
        
        self.errorrates_stt = torch.rand(1)*0.1
        self.errorrates_dtt = torch.rand(1)*0.2
        
        "Latent Variables"
        self.par_dict = {}
        self.par_dict['lr_day1'] = lr_day1
        self.par_dict['theta_Q_day1'] = theta_Q_day1 # dectemp > 0
        self.par_dict['theta_rep_day1'] = theta_rep_day1
        
        self.par_dict['lr_day2'] = lr_day2
        self.par_dict['theta_Q_day2'] = theta_Q_day2 # dectemp > 0
        self.par_dict['theta_rep_day2'] = theta_rep_day2
        self.theta_rep_day1 = theta_rep_day1
        self.theta_Q_day1 = theta_Q_day1 # dectemp > 0
        self.lr_day1 = lr_day1
        
        self.theta_rep_day2 = theta_rep_day2
        self.theta_Q_day2 = theta_Q_day2 # dectemp > 0
        self.lr_day2 = lr_day2
        
        "K"
        self.k = k
        self.BAD_CHOICE = -2
        
        "Q and rep"
        self.Q_init = Q_init
        self.Q = [Q_init] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, 4)*0.25] # habitual values (repetition values)
        
        "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"
        self.V = [(self.theta_rep_day1[..., None]*self.rep[-1] + self.theta_Q_day1[..., None]*self.Q[-1])]
        
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-2 in seq_counter for errors"
        "Dimensions are [agent, blocktypes, pppchoice, ppchoice, pchoice, choice]"
        self.init_seq_counter = self.k / 4 * np.ones((self.num_agents, 2, 6, 6, 6, 6))
        
        self.seq_counter = self.init_seq_counter.clone().detach()
    
    def locs_to_pars(self, locs):
        self.par_dict = {"lr_day1": torch.sigmoid(locs[..., 0]),
                    "theta_Q_day1": torch.exp(locs[..., 1]),
                    "theta_rep_day1": torch.exp(locs[..., 2]),
                    
                    "lr_day2": torch.sigmoid(locs[..., 3]),
                    "theta_Q_day2": torch.exp(locs[..., 4]),
                    "theta_rep_day2": torch.exp(locs[..., 5])}
        
        return self.par_dict

    def update(self, choices, outcomes, blocktype, day, **kwargs):
        '''

        Parameters
        ----------
        choices : torch.tensor or list with shape [num_agents]
            The particiapnt's choice at the dual-target trial.
            -2, 0, 1, 2, or 3
            -2 = error
            
        outcomes : torch.tensor or list with shape [num_agents]
            no reward (0) or reward (1).
            
        blocktype : torch.tensor or list with shape [num_agents]
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

        if day == 1:
            lr = self.lr_day1
            theta_Q = self.theta_Q_day1
            theta_rep = self.theta_rep_day1

        elif day == 2:
            lr = self.lr_day2
            theta_Q = self.theta_Q_day2
            theta_rep = self.theta_rep_day2

        if all([ch == -1 for ch in choices]) and all([out == -1 for out in outcomes]) and all([bb == -1 for bb in blocktype]):
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
            # outcome is either 0 or 1
            
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
        _ = self.locs_to_pars(locs)
        
        "Setup"
        self.num_particles = locs.shape[0]
        self.num_agents = locs.shape[1]
        
        "Latent Variables"
        self.lr_day1 = self.par_dict["lr_day1"]
        self.theta_Q_day1 = self.par_dict["theta_Q_day1"]
        self.theta_rep_day1 = self.par_dict["theta_rep_day1"]        
        
        self.lr_day2 = self.par_dict["lr_day2"]
        self.theta_Q_day2 = self.par_dict["theta_Q_day2"]
        self.theta_rep_day2 = self.par_dict["theta_rep_day2"]
        
        "K"
        # self.k = kwargs["k"]
            
        "Q and rep"
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.NA)] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, self.NA)*1./self.NA] # habitual values (repetition values)
        
        "Compute V"
        self.V.append(self.theta_rep_day1[..., None]*self.rep[-1] + self.theta_Q_day1[..., None]*self.Q[-1])
        
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-2 in seq_counter for errors"
        "Dimensions are [blocktypes, pppchoice, ppchoice, pchoice, choice, agent]"
        self.init_seq_counter = self.k / 4 * np.ones((self.num_agents, 2, 6, 6, 6, 6))
        
        self.seq_counter = self.init_seq_counter.clone().detach()