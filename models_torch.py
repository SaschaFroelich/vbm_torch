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
    def __init__(self, omega, dectemp, lr, k, Q_init, num_blocks = 14):
        """ 
        --- Parameters ---
        omega (between 0 & 1): weighting factor between habitual and goal-directed: p(a1) = σ(β*[(1-ω)*(r(a1)-r(a2)) + ω*(Q(a1)-Q(a2)))]
        dectemp (between 0 & inf): decision temperature β
        lr (between 0 & 1) : learning rate
        Q_init : (list of floats) initial Q-Values"""
        
        # assert(omega.ndim == 2)
        # assert(dectemp.ndim == 2)
        # assert(lr.ndim == 2)
        
        # assert(Q_init.ndim == 3)
        
        if num_blocks != 1:
            if num_blocks%2 != 0:
                raise Exception("num_blocks must be an even value.")
        
        "Setup"
        self.param_names = ["omega", "dectemp", "lr"]
        self.num_particles = omega.shape[0]
        self.num_agents = omega.shape[1]
        self.trials = 480*num_blocks
        self.num_blocks = num_blocks
        self.NA = 4 # no. of possible actions
        
        "--- Latent variables ---"
        self.omega = omega
        self.dectemp = dectemp # dectemp > 0 for softmax function
        self.lr = lr
        "--- --- --- ---"
        
        "K"
        self.k = k
        self.BAD_CHOICE = -2
        
        "Q and rep"
        self.Q_init = Q_init
        self.Q = [Q_init] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, 4)*0.25] # habitual values (repetition values)

        # "V(ai) = (1-ω)*rep_val(ai) + ω*Q(ai)"
        # V0 = (1-self.omega)*self.rep[-1][..., 0] + self.omega*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
        # V1 = (1-self.omega)*self.rep[-1][..., 1] + self.omega*self.Q[-1][..., 1]
        # V2 = (1-self.omega)*self.rep[-1][..., 2] + self.omega*self.Q[-1][..., 2]
        # V3 = (1-self.omega)*self.rep[-1][..., 3] + self.omega*self.Q[-1][..., 3]
        
        # Vold = torch.stack((V0,V1,V2,V3), 2)
        
        self.V = [((1-self.omega)[..., None]*self.rep[-1] + self.omega[..., None]*self.Q[-1])]
        
        # assert(Vold==self.V)
        
        # self.V = [(1-self.omega).T*self.rep[-1][0,...] + self.omeg.T*self.Q[-1][0, ...]]
        
        
        # self.V = [torch.stack((V0,V1,V2,V3), 2)]
        
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-2 in seq_counter for errors"
        "Dimensions are [blocktypes, pppchoice, ppchoice, pchoice, choice, agent]"
        self.init_seq_counter = self.k / 4 * np.ones((self.num_agents, 2, 6, 6, 6, 6))
        
        self.seq_counter = self.init_seq_counter.clone()


    def locs_to_pars(self, locs):
        par_dict = {"omega": torch.sigmoid(locs[..., 0]),
                    "dectemp": torch.exp(locs[..., 1]),
                    "lr": torch.sigmoid(locs[..., 2])}

        return par_dict
    
    def compute_probs(self, trial, day):
        option1, option2 = self.find_resp_options([trial])
        
        _, mask = self.Qoutcomp(self.V[-1], option1)
        Vopt1 = (self.V[-1][torch.where(mask == 1)]).reshape(self.num_particles, self.num_agents)
        _, mask = self.Qoutcomp(self.V[-1], option2)
        Vopt2 = self.V[-1][torch.where(mask == 1)].reshape(self.num_particles, self.num_agents)
        
        probs = self.softmax(torch.stack((Vopt1, Vopt2), 2))
        
        
        return probs
        
    def Qoutcomp(self, Qin, choices):
        """Qin shape is [num_particles, num_agents, 4]"""
        """Returns a tensor with the same shape as Qin, with zeros everywhere except for the relevant places
        as indicated by 'choices', where the values of Qin are retained. Q positions for agents with an error choice
        are replaced by 0."""
        
        Qin = Qin.type(torch.double)
        
        if len(Qin.shape) == 2:
            Qin = Qin[None, ...]
            
        elif len(Qin.shape) == 3:
            pass
        
        else:
            ipdb.set_trace()
            raise Exception("Fehla, digga!")
        
        try:
            no_error_mask = [1 if ch != self.BAD_CHOICE else 0 for ch in choices]
        except:
            ipdb.set_trace()
            
        "Replace error choices by the number one"
        choices_noerrors = torch.where(torch.tensor(no_error_mask).type(torch.bool), choices, torch.ones(choices.shape)).type(torch.int)
    
        Qout = torch.zeros(Qin.shape).double()
        choicemask = torch.zeros(Qin.shape, dtype = int)
        num_particles = Qout.shape[0] # num of particles
        num_agents = Qout.shape[1] # num_agents
        
        errormask = torch.tensor([0 if c == self.BAD_CHOICE else 1 for c in choices])
        errormask = errormask.broadcast_to(num_particles, 4, num_agents).transpose(1,2)
        
        x = torch.linspace(0, num_particles-1,num_particles, dtype = int).repeat(num_agents)
        y = torch.linspace(0, num_agents-1,num_agents, dtype = int).repeat_interleave(num_particles)
        z = choices_noerrors.repeat_interleave(num_particles)
        Qout[x, y, z] = Qin[x, y, z]
    
        choicemask[torch.linspace(0, num_particles-1,num_particles, dtype = int).repeat_interleave(num_agents), \
              torch.linspace(0, num_agents-1,num_agents, dtype = int).repeat(num_particles), \
              choices_noerrors.repeat(num_particles)] = 1
        
        mask = errormask*choicemask
        return Qout.double()*mask, mask
    
    def softmax(self, z):
        sm = torch.nn.Softmax(dim=-1)
        p_actions = sm(self.dectemp[..., None]*z)
            
        return p_actions

    def find_resp_options(self, stimulus_mat):
        """
        Given a dual-target stimulus (e.g. 12, 1-indexed), this function returns the two response
        options in 0-indexing. E.g.: stimulus_mat=14 -> option1_python = 0, option1_python = 3
        INPUT: stimulus in MATLAB notation (1-indexed) (simple list of stimuli)
        OUTPUT: response options in python notation (0-indexed)
        """
        option2_python = ((torch.tensor(stimulus_mat) % 10) - 1).type(torch.int)
        option1_python = (((torch.tensor(stimulus_mat) - (torch.tensor(stimulus_mat) % 10)) / 10) -1).type(torch.int)
        
        if option2_python.ndim == 2 and option2_python.shape[0] == 1:
            option1_python = torch.squeeze(option1_python)
            option2_python = torch.squeeze(option2_python)
            
        return option1_python, option2_python

    def choose_action(self, trial, day):
        "INPUT: trial (in 1-indexing (i.e. MATLAB notation))"
        "OUTPUT: choice response digit (in 0-indexing notation)"
        
        if trial < 10:
            "Single-target trial"
            choice_python = torch.tensor(trial-1)
        
        elif trial > 10:
            # torch.manual_seed(123)
            
            "Dual-target trial"
            option1, option2 = self.find_resp_options([trial])
            
            # shape of softmax argument should be (n_particles, n_subjects, 2)
            # p_actions = self.softmax(torch.cat((self.V[-1][..., option1], self.V[-1][..., option2]), dim = -1))
            # p_actions comes out as shape [1, 1, 2]
            # choice_sample = torch.multinomial(torch.squeeze(p_actions), 1)[0]
            
            choice_sample = torch.distributions.categorical.Categorical(probs=self.compute_probs(trial, day)).sample()

            choice_python = option2*choice_sample + option1*(1-choice_sample)
            
        "Squeeze because if trial > 10 ndim of choice_python is 1 (and not 0)"
        return torch.squeeze(choice_python).type('torch.LongTensor')

    def update(self, choices, outcomes, blocktype, **kwargs):
        """
        Is called after a dual-target choice and updates Q-values, sequence counters, habit values (i.e. repetition values), and V-Values.
        
        choices : the single-target trial choices before the next dual-taregt trial (<0 is error) (0-indexed)"
        
        --- Parameters ---
        choice (-2, 0, 1, 2, or 3): The particiapnt's choice at the dual-target trial
                                     -2 : error
        outcomes (0 or 1) : no reward (0) or reward (1)
        blocktype : 's' (sequential blocks) or 'r' (random blocks)
                    Important for updating of sequence counters.
        """
        
        if all([ch == -1 for ch in choices]) and all([out == -1 for out in outcomes]) and all([bb == -1 for bb in blocktype]):
            "Set previous actions to -1 because it's the beginning of a new block"
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.ones(self.lr.shape[0], self.lr.shape[1], self.NA)/self.NA)
            self.Q.append(self.Q[-1])
                        
            # V0 = (1-self.omega)*self.rep[-1][..., 0] + self.omega*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
            # V1 = (1-self.omega)*self.rep[-1][..., 1] + self.omega*self.Q[-1][..., 1]
            # V2 = (1-self.omega)*self.rep[-1][..., 2] + self.omega*self.Q[-1][..., 2]
            # V3 = (1-self.omega)*self.rep[-1][..., 3] + self.omega*self.Q[-1][..., 3]
            # self.V.append(torch.stack((V0,V1,V2,V3), 2))
            
            self.V.append((1-self.omega)[..., None]*self.rep[-1] + self.omega[..., None]*self.Q[-1])
            
            # self.V.append((1-self.omega).T*self.rep[-1][0,...] + self.omeg.T*self.Q[-1][0, ...])
            # dfgh
        else:
            
            "----- Update GD-values -----"
            Qout, mask = self.Qoutcomp(self.Q[-1], choices)
            Qnew = self.Q[-1] + self.lr[..., None]*(outcomes[None,...,None]-Qout)*mask
            
            self.Q.append(Qnew)
            "--- The following is executed in case of correct and inocrrect responses ---"
            "----- Update sequence counters and repetition values of self.rep -----"
            repnew = []
            for agent in range(self.num_agents):
                new_row = [0., 0., 0., 0.]

                self.seq_counter[agent, 
                                 blocktype[agent],
                                self.pppchoice[agent],
                                self.ppchoice[agent],
                                self.pchoice[agent],
                                choices[agent]] += 1
            
                " Update rep values "
                index = (agent, 
                         blocktype[agent], 
                         self.ppchoice[agent],
                         self.pchoice[agent], 
                         choices[agent])

                seqs_sum =  self.seq_counter[index + (0,)] + \
                            self.seq_counter[index + (1,)] + \
                            self.seq_counter[index + (2,)] + \
                            self.seq_counter[index + (3,)]

                new_row = [self.seq_counter[index + (action,)] / seqs_sum for action in range(4)] 
            
                repnew.append(new_row)
            
            self.rep.append(torch.tensor(repnew)[None, :].broadcast_to(self.num_particles , self.num_agents, 4))
            
            "----- Compute new V-values for next trial -----"
            # V0 = (1-self.omega)*self.rep[-1][..., 0] + self.omega*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
            # V1 = (1-self.omega)*self.rep[-1][..., 1] + self.omega*self.Q[-1][..., 1]
            # V2 = (1-self.omega)*self.rep[-1][..., 2] + self.omega*self.Q[-1][..., 2]
            # V3 = (1-self.omega)*self.rep[-1][..., 3] + self.omega*self.Q[-1][..., 3]
            # self.V.append(torch.stack((V0,V1,V2,V3), 2))
            
            self.V.append((1-self.omega)[..., None]*self.rep[-1] + self.omega[..., None]*self.Q[-1])
            
            # self.V.append((1-self.omega).T*self.rep[-1][0,...] + self.omeg.T*self.Q[-1][0, ...])
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
        par_dict = self.locs_to_pars(locs)
                
        self.num_particles = locs.shape[0]
        self.num_agents = locs.shape[1]
        
        "Latent Variables"
        self.omega = par_dict["omega"]
        self.dectemp = par_dict["dectemp"]
        self.lr = par_dict["lr"]
        
        "K"
        # self.k = kwargs["k"]
            
        "Q and rep"
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.NA)] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, 4)*0.25] # habitual values (repetition values)
        
        "Compute V"        
        # For single parameters
        # V0 = (1-self.omega)*self.rep[-1][..., 0] + self.omega*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
        # V1 = (1-self.omega)*self.rep[-1][..., 1] + self.omega*self.Q[-1][..., 1]
        # V2 = (1-self.omega)*self.rep[-1][..., 2] + self.omega*self.Q[-1][..., 2]
        # V3 = (1-self.omega)*self.rep[-1][..., 3] + self.omega*self.Q[-1][..., 3]
        # self.V.append(torch.stack((V0,V1,V2,V3), 2))
        
        self.V.append((1-self.omega)[..., None]*self.rep[-1] + self.omega[..., None]*self.Q[-1])
        
        # self.V.append((1-self.omega).T*self.rep[-1][0,...] + self.omeg.T*self.Q[-1][0, ...])
        # dfgh
        
        "Sequence Counters"
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-2 in seq_counter for errors"
        "Dimensions are [blocktypes, pppchoice, ppchoice, pchoice, choice, agent]"
        self.init_seq_counter = self.k / 4 * np.ones((self.num_agents, 2, 6, 6, 6, 6))
        
        self.seq_counter = self.init_seq_counter.clone()
        
        
class Vbm_B(Vbm):
    
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
        """ 
        --- Parameters ---
        omega (between 0 & 1): weighting factor between habitual and goal-directed: p(a1) = σ(β*[(1-ω)*(r(a1)-r(a2)) + ω*(Q(a1)-Q(a2)))]
        dectemp (between 0 & inf): decision temperature β
        lr (between 0 & 1) : learning rate
        Q_init : (list of floats) initial Q-Values
        """
        
        # assert(lr_day1.ndim == 2)
        # assert(theta_Q_day1.ndim == 2)
        # assert(theta_rep_day1.ndim == 2)
        
        # assert(lr_day2.ndim == 2)
        # assert(theta_Q_day2.ndim == 2)
        # assert(theta_rep_day2.ndim == 2)
        
        # assert(Q_init.ndim == 3)
        
        if num_blocks != 1:
            if num_blocks%2 != 0:
                raise Exception("num_blocks must be an even value.")

        "Setup"
        self.param_names = ["lr_day1", 
                            "theta_Q_day1", 
                            "theta_rep_day1", 
                            "lr_day2", 
                            "theta_Q_day2", 
                            "theta_rep_day2"]
        
        self.NA = 4 # no. of possible actions
        self.num_particles = lr_day1.shape[0]
        self.num_agents = lr_day1.shape[1]
        self.dectemp = torch.tensor([[1.]]) # For softmax function
        
        self.trials = 480*num_blocks
        self.num_blocks = num_blocks
        
        "Latent Variables"
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
        # V0 = self.theta_rep_day1*self.rep[-1][..., 0] + self.theta_Q_day1*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
        # V1 = self.theta_rep_day1*self.rep[-1][..., 1] + self.theta_Q_day1*self.Q[-1][..., 1]
        # V2 = self.theta_rep_day1*self.rep[-1][..., 2] + self.theta_Q_day1*self.Q[-1][..., 2]
        # V3 = self.theta_rep_day1*self.rep[-1][..., 3] + self.theta_Q_day1*self.Q[-1][..., 3]
        
        # self.V = [torch.stack((V0,V1,V2,V3), 2)]
        
        self.V = [(self.theta_rep_day1[..., None]*self.rep[-1] + self.theta_Q_day1[..., None]*self.Q[-1])]
        
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-2 in seq_counter for errors"
        "Dimensions are [agent, blocktypes, pppchoice, ppchoice, pchoice, choice]"
        self.init_seq_counter = self.k / 4 * np.ones((self.num_agents, 2, 6, 6, 6, 6))
        
        self.seq_counter = self.init_seq_counter.clone()
    
    def locs_to_pars(self, locs):
        par_dict = {"lr_day1": torch.sigmoid(locs[..., 0]),
                    "theta_Q_day1": torch.exp(locs[..., 1]),
                    "theta_rep_day1": torch.exp(locs[..., 2]),
                    
                    "lr_day2": torch.sigmoid(locs[..., 3]),
                    "theta_Q_day2": torch.exp(locs[..., 4]),
                    "theta_rep_day2": torch.exp(locs[..., 5])}

        return par_dict

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
        # # assert(torch.is_tensor(choices))
        # # assert(torch.is_tensor(outcomes))
        # # assert(torch.is_tensor(blocktype))
        # assert(isinstance(day, int))

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
            
            # V0 = theta_rep*self.rep[-1][..., 0] + theta_Q*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
            # V1 = theta_rep*self.rep[-1][..., 1] + theta_Q*self.Q[-1][..., 1]
            # V2 = theta_rep*self.rep[-1][..., 2] + theta_Q*self.Q[-1][..., 2]
            # V3 = theta_rep*self.rep[-1][..., 3] + theta_Q*self.Q[-1][..., 3]
            
            # self.V.append(torch.stack((V0,V1,V2,V3), 2))
            
            self.V.append(theta_rep[..., None]*self.rep[-1] + theta_Q[..., None]*self.Q[-1])
            
            # self.V.append(theta_rep.T*self.rep[-1][0,...] + theta_Q.T*self.Q[-1][0, ...])
            # dfgh
            
        else:
            "----- Update GD-values -----"
            # outcome is either 0 or 1
            
            "--- Group!!! ----"
            "mask contains 1s where Qoutcomp() contains non-zero entries"
            Qout, mask = self.Qoutcomp(self.Q[-1], choices)
            Qnew = self.Q[-1] + lr[..., None]*(outcomes[None,...,None]-Qout)*mask
            
            self.Q.append(Qnew)
            
            "--- The following is executed in case of correct and incorrect responses ---"
            "----- Update sequence counters and repetition values of self.rep -----"
            
            repnew = []
            for agent in range(self.num_agents):
                new_row = [0., 0., 0., 0.]

                self.seq_counter[agent, 
                                 blocktype[agent],
                                self.pppchoice[agent],
                                self.ppchoice[agent],
                                self.pchoice[agent],
                                choices[agent]] += 1
            
                " Update rep values "
                index = (agent, 
                         blocktype[agent], 
                         self.ppchoice[agent],
                         self.pchoice[agent], 
                         choices[agent])

                seqs_sum =  self.seq_counter[index + (0,)] + \
                            self.seq_counter[index + (1,)] + \
                            self.seq_counter[index + (2,)] + \
                            self.seq_counter[index + (3,)]

                new_row = [self.seq_counter[index + (action,)] / seqs_sum for action in range(4)] 
            
                repnew.append(new_row)

            
            self.rep.append(torch.tensor(repnew)[None, :].broadcast_to(self.num_particles , self.num_agents, self.NA))
            
            "----- Compute new V-values for next trial -----"
            # V0 = theta_rep*self.rep[-1][..., 0] + theta_Q*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
            # V1 = theta_rep*self.rep[-1][..., 1] + theta_Q*self.Q[-1][..., 1]
            # V2 = theta_rep*self.rep[-1][..., 2] + theta_Q*self.Q[-1][..., 2]
            # V3 = theta_rep*self.rep[-1][..., 3] + theta_Q*self.Q[-1][..., 3]
            # self.V.append(torch.stack((V0,V1,V2,V3), 2))
            
            self.V.append(theta_rep[..., None]*self.rep[-1] + theta_Q[..., None]*self.Q[-1])
            
            # self.V.append(theta_rep.T*self.rep[-1][0,...] + theta_Q.T*self.Q[-1][0, ...])
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
        par_dict = self.locs_to_pars(locs)
        
        "Setup"
        self.num_particles = locs.shape[0]
        self.num_agents = locs.shape[1]
        
        "Latent Variables"
        self.lr_day1 = par_dict["lr_day1"]
        self.theta_Q_day1 = par_dict["theta_Q_day1"]
        self.theta_rep_day1 = par_dict["theta_rep_day1"]        
        
        self.lr_day2 = par_dict["lr_day2"]
        self.theta_Q_day2 = par_dict["theta_Q_day2"]
        self.theta_rep_day2 = par_dict["theta_rep_day2"]
        
        "K"
        # self.k = kwargs["k"]
            
        "Q and rep"
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.NA)] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, self.NA)*1./self.NA] # habitual values (repetition values)
        
        "Compute V"
        # V0 = self.theta_rep_day1*self.rep[-1][..., 0] + self.theta_Q_day1*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
        # V1 = self.theta_rep_day1*self.rep[-1][..., 1] + self.theta_Q_day1*self.Q[-1][..., 1]
        # V2 = self.theta_rep_day1*self.rep[-1][..., 2] + self.theta_Q_day1*self.Q[-1][..., 2]
        # V3 = self.theta_rep_day1*self.rep[-1][..., 3] + self.theta_Q_day1*self.Q[-1][..., 3]
        
        # self.V.append(torch.stack((V0,V1,V2,V3), 2))
        self.V.append(self.theta_rep_day1[..., None]*self.rep[-1] + self.theta_Q_day1[..., None]*self.Q[-1])
        
        # self.V.append(self.theta_rep_day1.T*self.rep[-1][0,...] + self.theta_Q_day1.T*self.Q[-1][0, ...])
        # dfgh
        
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-2 in seq_counter for errors"
        "Dimensions are [blocktypes, pppchoice, ppchoice, pchoice, choice, agent]"
        self.init_seq_counter = self.k / 4 * np.ones((self.num_agents, 2, 6, 6, 6, 6))
        
        self.seq_counter = self.init_seq_counter.clone()