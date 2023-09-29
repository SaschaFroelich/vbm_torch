#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:02:16 2023

File containing the different models currently in use (older models in models_torch_archive.py)

Models:
    - vbm
    - vbm_B(vbm)
    - vbm_B_onlydual
    - vbm_B_2
    - vbm_B_3
    - vbm_F

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
        '''
        Initializes the model.
    
                Parameters:
                        All model parameters must be torch tensors with the shape [num_particles, num_agents]
                    
                        omega (between 0 & 1): weighting factor between habitual and goal-directed: p(a1) = σ(β*[(1-ω)*(r(a1)-r(a2)) + ω*(Q(a1)-Q(a2)))]
                        dectemp (between 0 & inf): decision temperature β
                        lr (between 0 & 1) : learning rate
                        Q_init : (list of floats) initial Q-Values
    
                Returns:
                        None
        '''
        
        assert(omega.ndim == 2)
        assert(dectemp.ndim == 2)
        assert(lr.ndim == 2)
        
        assert(Q_init.ndim == 3)
        
        if num_blocks != 1:
            if num_blocks%2 != 0:
                raise Exception("num_blocks must be an even value.")
        
        self.errorrate = 0.0
        
        "Setup"
        self.param_names = ["omega", "dectemp", "lr"]
        self.num_particles = omega.shape[0]
        self.num_agents = omega.shape[1]
        self.trials = 480*num_blocks
        self.num_blocks = num_blocks
        self.na = 4 # no. of possible actions
        
        "--- Latent variables ---"
        self.omega = omega
        self.dectemp = dectemp # dectemp > 0 for softmax function
        self.lr = lr
        "--- --- --- ---"
        
        "K"
        self.k = k
        
        "Q and rep"
        self.Q_init = Q_init
        self.Q = [Q_init] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, 4)*0.25] # habitual values (repetition values)

        "V(ai) = (1-ω)*rep_val(ai) + ω*Q(ai)"
        V0 = (1-self.omega)*self.rep[-1][..., 0] + self.omega*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
        V1 = (1-self.omega)*self.rep[-1][..., 1] + self.omega*self.Q[-1][..., 1]
        V2 = (1-self.omega)*self.rep[-1][..., 2] + self.omega*self.Q[-1][..., 2]
        V3 = (1-self.omega)*self.rep[-1][..., 3] + self.omega*self.Q[-1][..., 3]
        
        self.V = [torch.stack((V0,V1,V2,V3), 2)]
        
        "----- Set sequence counter -----"
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        self.init_seq_counter = {}
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-10 in seq_counter for errors)"
        """ seq_counter dims: 
           Dim 0 : random or sequential (0 = sequential, 1 = random)
           Dim 1 : pppchoice 
           Dim 2 : ppchoice
           Dim 3 : pchoice
           Dim 4 : choice
           Dim 5 : idx of agent"""
        self.init_seq_counter = self.k / 4 * np.ones((2, 6, 6, 6, 6,
                                                       self.num_agents))
        self.seq_counter = self.init_seq_counter.copy()
        
    def locs_to_pars(self, locs):
        par_dict = {"omega": torch.sigmoid(locs[..., 0]),
                    "dectemp": torch.exp(locs[..., 1]),
                    "lr": torch.sigmoid(locs[..., 2])}

        return par_dict
    
    def compute_probs(self, trial, day):
        option1, option2 = self.find_resp_options(trial)
        
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
            no_error_mask = [1 if ch != -10 else 0 for ch in choices]
        except:
            ipdb.set_trace()
            
        "Replace error choices by the number one"
        choices_noerrors = torch.where(torch.tensor(no_error_mask).type(torch.bool), choices, torch.ones(choices.shape)).type(torch.int)
    
        Qout = torch.zeros(Qin.shape).double()
        choicemask = torch.zeros(Qin.shape, dtype = int)
        num_particles = Qout.shape[0] # num of particles
        num_agents = Qout.shape[1] # num_agents
        
        errormask = torch.tensor([0 if c == -10 else 1 for c in choices])
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
        
        cond = stimulus_mat > 10
        
        option2_python = ((stimulus_mat % 10) - 1).type(torch.int)
        option1_python = (((stimulus_mat - (stimulus_mat % 10)) / 10) -1).type(torch.int)
        
        if option2_python.ndim == 2 and option2_python.shape[0] == 1:
            option1_python = torch.squeeze(option1_python)
            option2_python = torch.squeeze(option2_python)
                    
        "-2 for 'bad choice' if cond == 0, otherwise the response options"
        option1_python = -2 + (2 + option1_python) * cond
        option2_python = -2 + (2 + option2_python) * cond
        return option1_python, option2_python

    def choose_action(self, trial, day):
        "INPUT: trial (in 1-indexing (i.e. MATLAB notation))"
        "OUTPUT: choice response digit (in 0-indexing notation)"
        
        
        sampled = torch.rand(1)
        cond_error = sampled > self.errorrate
        cond_trial = trial < 10
        "Dual-target trial"
        option1, option2 = self.find_resp_options(trial)
        probs = self.compute_probs(trial, day)
        choice_sample = torch.distributions.categorical.Categorical(probs=probs).sample()

        choice_python = option2 * choice_sample + \
            option1 * (1-choice_sample)
        if_noerror = (trial - 1) * cond_trial + \
            choice_python * ~cond_trial
            
        "-2 stands for 'bad choice'"
        to_return = -2 * \
            ~cond_error + if_noerror * cond_error
            
        ipdb.set_trace()
        return to_return
        
        # if trial < 10:
        #     "Single-target trial"
        #     choice_python = torch.tensor(trial-1)
        
        # elif trial > 10:
        #     # torch.manual_seed(123)
            
        #     "Dual-target trial"
        #     option1, option2 = self.find_resp_options([trial])
            
        #     # shape of softmax argument should be (n_particles, n_subjects, 2)
        #     # p_actions = self.softmax(torch.cat((self.V[-1][..., option1], self.V[-1][..., option2]), dim = -1))
        #     # p_actions comes out as shape [1, 1, 2]
        #     # choice_sample = torch.multinomial(torch.squeeze(p_actions), 1)[0]
            
        #     choice_sample = torch.distributions.categorical.Categorical(probs=self.compute_probs(trial, day)).sample()

        #     choice_python = option2*choice_sample + option1*(1-choice_sample)
            
        # "Squeeze because if trial > 10 ndim of choice_python is 1 (and not 0)"
        # return torch.squeeze(choice_python).type('torch.LongTensor')

    def update(self, choices, outcome, blocktype, **kwargs):
        '''
        Is called after a dual-target choice and updates Q-values, 
        sequence counters, habit values (i.e. repetition values), and V-Values.
    
                Parameters:
                        choices (-10, 0, 1, 2, or 3): The particiapnt's choice at the dual-target trial
                                                     -10 : error
                        outcome (0 or 1) : no reward (0) or reward (1)
                        blocktype : 's' (sequential blocks) or 'r' (random blocks)
                                    Important for updating of sequence counters.
    
                Returns:
                        None
        '''
        
        if all([ch == -1 for ch in choices]) and all([out == -1 for out in outcome]) and all([bb == -1 for bb in blocktype]):
            "Set previous actions to -1 because it's the beginning of a new block"
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.ones(self.lr.shape[0], self.lr.shape[1], self.na)/self.na)
            self.Q.append(self.Q[-1])
                        
            V0 = (1-self.omega)*self.rep[-1][..., 0] + self.omega*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
            V1 = (1-self.omega)*self.rep[-1][..., 1] + self.omega*self.Q[-1][..., 1]
            V2 = (1-self.omega)*self.rep[-1][..., 2] + self.omega*self.Q[-1][..., 2]
            V3 = (1-self.omega)*self.rep[-1][..., 3] + self.omega*self.Q[-1][..., 3]
            self.V.append(torch.stack((V0,V1,V2,V3), 2))
            
        else:
            
            "----- Update GD-values -----"
            Qout, mask = self.Qoutcomp(self.Q[-1], choices)
            Qnew = self.Q[-1] + self.lr[..., None]*(outcome[None,...,None]-Qout)*mask
            
            self.Q.append(Qnew)
            "--- The following is executed in case of correct and inocrrect responses ---"
            "----- Update sequence counters and repetition values of self.rep -----"
            repnew = []
            for agent in range(self.num_agents):
                prev_seq = [str(self.ppchoice[agent].item()) + "," + str(self.pchoice[agent].item()) + "," + str(choices[agent].item())]
                new_row = [0., 0. ,0. ,0.]
                
                if blocktype[agent] == "s":
                    "Sequential Block"
                    " Update counter "
                    self.seq_counter_tb[str(self.pppchoice[agent].item()) + "," + \
                                     str(self.ppchoice[agent].item()) + "," + str(self.pchoice[agent].item()) + "," + str(choices[agent].item())][agent] += 1

                    " Update rep values "
                    for aa in range(4):
                        new_row[aa] = self.seq_counter_tb[prev_seq[0] + "," + str(aa)][agent] / \
                                (self.seq_counter_tb[prev_seq[0] + "," + "0"][agent] + self.seq_counter_tb[prev_seq[0] + "," + "1"][agent] + \
                                 self.seq_counter_tb[prev_seq[0] + "," + "2"][agent] + self.seq_counter_tb[prev_seq[0] + "," + "3"][agent])

                elif blocktype[agent] == "r":
                    "Random Block"
                    " Update counter "
                    self.seq_counter_r[str(self.pppchoice[agent].item()) + "," + \
                                     str(self.ppchoice[agent].item()) + "," + str(self.pchoice[agent].item()) + "," + str(choices[agent].item())][agent] += 1

                    " Update rep values "
                    for aa in range(4):
                        new_row[aa] = self.seq_counter_r[prev_seq[0] + "," + str(aa)][agent] / \
                                (self.seq_counter_r[prev_seq[0] + "," + "0"][agent] + self.seq_counter_r[prev_seq[0] + "," + "1"][agent] + \
                                 self.seq_counter_r[prev_seq[0] + "," + "2"][agent] + self.seq_counter_r[prev_seq[0] + "," + "3"][agent])
                
                else:
                    raise Exception("Da isch a Fehla aba ganz a gwaldiga!")
                    
                repnew.append(new_row)
            
            self.rep.append(torch.tensor(repnew)[None, :].broadcast_to(self.num_particles , self.num_agents, 4))
            
            "----- Compute new V-values for next trial -----"
            V0 = (1-self.omega)*self.rep[-1][..., 0] + self.omega*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
            V1 = (1-self.omega)*self.rep[-1][..., 1] + self.omega*self.Q[-1][..., 1]
            V2 = (1-self.omega)*self.rep[-1][..., 2] + self.omega*self.Q[-1][..., 2]
            V3 = (1-self.omega)*self.rep[-1][..., 3] + self.omega*self.Q[-1][..., 3]
            self.V.append(torch.stack((V0,V1,V2,V3), 2))
            
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
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.na)] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, 4)*0.25] # habitual values (repetition values)
        
        "Compute V"        
        # For single parameters
        V0 = (1-self.omega)*self.rep[-1][..., 0] + self.omega*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
        V1 = (1-self.omega)*self.rep[-1][..., 1] + self.omega*self.Q[-1][..., 1]
        V2 = (1-self.omega)*self.rep[-1][..., 2] + self.omega*self.Q[-1][..., 2]
        V3 = (1-self.omega)*self.rep[-1][..., 3] + self.omega*self.Q[-1][..., 3]
        self.V.append(torch.stack((V0,V1,V2,V3), 2))
        
        self.seq_counter = self.init_seq_counter.copy()

class Vbm_b(Vbm):
    
    def __init__(self, \
                 lr_day1, \
                 theta_Q_day1, \
                 theta_rep_day1, \
                 lr_day2, \
                 theta_Q_day2, \
                 theta_rep_day2, \
                 k, \
                 Q_init, \
                 num_blocks = 14):
        """ 
        --- Parameters ---
        omega (between 0 & 1): weighting factor between habitual and goal-directed: p(a1) = σ(β*[(1-ω)*(r(a1)-r(a2)) + ω*(Q(a1)-Q(a2)))]
        dectemp (between 0 & inf): decision temperature β
        lr (between 0 & 1) : learning rate
        Q_init : (list of floats) initial Q-Values
        """
        
        assert(lr_day1.ndim == 2)
        assert(theta_Q_day1.ndim == 2)
        assert(theta_rep_day1.ndim == 2)
        
        assert(lr_day2.ndim == 2)
        assert(theta_Q_day2.ndim == 2)
        assert(theta_rep_day2.ndim == 2)
        assert(Q_init.ndim == 3)
        
        if num_blocks != 1:
            if num_blocks%2 != 0:
                raise Exception("num_blocks must be an even value.")

        self.errorrate = 0.0

        "Setup"
        self.param_names = ["lr_day1", "theta_Q_day1", "theta_rep_day1", "lr_day2", "theta_Q_day2", "theta_rep_day2"]
        self.na = 4 # no. of possible actions
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
        
        "Q and rep"
        self.Q_init = Q_init
        self.Q = [Q_init] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, 4)*0.25] # habitual values (repetition values)
        
        "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"
        V0 = self.theta_rep_day1*self.rep[-1][..., 0] + self.theta_Q_day1*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
        V1 = self.theta_rep_day1*self.rep[-1][..., 1] + self.theta_Q_day1*self.Q[-1][..., 1]
        V2 = self.theta_rep_day1*self.rep[-1][..., 2] + self.theta_Q_day1*self.Q[-1][..., 2]
        V3 = self.theta_rep_day1*self.rep[-1][..., 3] + self.theta_Q_day1*self.Q[-1][..., 3]
        
        self.V = [torch.stack((V0,V1,V2,V3), 2)]
        
        "----- Set sequence counter -----"
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        self.init_seq_counter = {}
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-10 in seq_counter for errors)"
        """ seq_counter dims: 
           Dim 0 : random or sequential (0 = sequential, 1 = random)
           Dim 1 : pppchoice 
           Dim 2 : ppchoice
           Dim 3 : pchoice
           Dim 4 : choice
           Dim 5 : idx of agent"""
        self.init_seq_counter = self.k / 4 * np.ones((2, 6, 6, 6, 6,
                                                       self.num_agents))
        self.seq_counter = self.init_seq_counter.copy()
        
    def locs_to_pars(self, locs):
        par_dict = {"lr_day1": torch.sigmoid(locs[..., 0]),
                    "theta_Q_day1": torch.exp(locs[..., 1]),
                    "theta_rep_day1": torch.exp(locs[..., 2]),
                    
                    "lr_day2": torch.sigmoid(locs[..., 3]),
                    "theta_Q_day2": torch.exp(locs[..., 4]),
                    "theta_rep_day2": torch.exp(locs[..., 5])}

        return par_dict
    
    # def Qoutcomp(self, Qin, choices):
    #     """Qin shape is [num_particles, num_agents, 4]"""
    #     """Returns a tensor with the same shape as Qin, with zeros everywhere except for the relevant places
    #     as indicated by 'choices', where the values of Qin are retained. Q positions for agents with an error choice
    #     are replaced by 0"""
            
    #     if len(Qin.shape) == 2:
    #         Qin = Qin[None, ...]
            
    #     elif len(Qin.shape) == 3:
    #         pass
        
    #     else:
    #         ipdb.set_trace()
    #         raise Exception("Fehla, digga!")

    #     no_error_mask = [1 if ch != -10 else 0 for ch in choices]
    #     "Replace error choices by the number one"
    #     choices_noerrors = torch.where(torch.tensor(no_error_mask).type(torch.bool), choices, torch.ones(choices.shape)).type(torch.int)

    #     Qout = torch.zeros(Qin.shape).double()
    #     choicemask = torch.zeros(Qin.shape, dtype = int)
    #     num_particles = Qout.shape[0] # num of particles
    #     num_agents = Qout.shape[1] # num_agents
        
    #     errormask = torch.tensor([0 if c == -10 else 1 for c in choices])
    #     errormask = errormask.broadcast_to(num_particles, 4, num_agents).transpose(1,2)
        
    #     Qout[torch.linspace(0, num_particles-1,num_particles, dtype = int).repeat(num_agents), \
    #          torch.linspace(0, num_agents-1,num_agents, dtype = int).repeat(num_particles), \
    #          choices_noerrors.repeat(num_particles)] = Qin[torch.linspace(0, num_particles-1,num_particles, dtype = int).repeat(num_agents), \
    #                                          torch.linspace(0, num_agents-1,num_agents, dtype = int).repeat(num_particles), \
    #                                          choices_noerrors.repeat(num_particles)]

    #     choicemask[torch.linspace(0, num_particles-1,num_particles, dtype = int).repeat_interleave(num_agents), \
    #           torch.linspace(0, num_agents-1,num_agents, dtype = int).repeat(num_particles), \
    #           choices_noerrors.repeat(num_particles)] = 1
        
    #     mask = errormask*choicemask
    #     return Qout.double()*mask, mask

    # def softmax(self, z):
    #     sm = torch.nn.Softmax(dim = -1)
    #     p_actions = sm(z)

    #     return p_actions

    def update(self, choices, outcome, blocktype, **kwargs):
        """
        Is called after a dual-target choice and updates Q-values, sequence counters, habit values (i.e. repetition values), and V-Values.
        
        choices : the single-target trial choices before the next dual-taregt trial (<0 is error) (0-indexed)"
        
        --- Parameters ---
        choice (-10, 0, 1, 2, or 3): The particiapnt's choice at the dual-target trial
                                     -10 : error
        outcome (0 or 1) : no reward (0) or reward (1)
        blocktype : 's' (sequential blocks) or 'r' (random blocks)
                    Important for updating of sequence counters.
        """

        if kwargs['day'] == 1:
            lr = self.lr_day1
            theta_Q = self.theta_Q_day1
            theta_rep = self.theta_rep_day1

        elif kwargs['day'] == 2:
            lr = self.lr_day2
            theta_Q = self.theta_Q_day2
            theta_rep = self.theta_rep_day2

        if all([ch == -1 for ch in choices]) and all([out == -1 for out in outcome]) and all([bb == -1 for bb in blocktype]):
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.ppchoice = -1*torch.ones(self.num_agents, dtype = int)
            self.pchoice = -1*torch.ones(self.num_agents, dtype = int)

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.ones(lr.shape[0], lr.shape[1], self.na)/self.na)
            self.Q.append(self.Q[-1])
            
            V0 = theta_rep*self.rep[-1][..., 0] + theta_Q*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
            V1 = theta_rep*self.rep[-1][..., 1] + theta_Q*self.Q[-1][..., 1]
            V2 = theta_rep*self.rep[-1][..., 2] + theta_Q*self.Q[-1][..., 2]
            V3 = theta_rep*self.rep[-1][..., 3] + theta_Q*self.Q[-1][..., 3]
            
            self.V.append(torch.stack((V0,V1,V2,V3), 2))
            # self.V.append(torch.cat((V0,V1,V2,V3))[None, :])
            
        else:
            "----- Update GD-values -----"
            # Outcome is either 0 or 1
            
            "--- Group!!! ----"
            "mask contains 1s where Qoutcomp() contains non-zero entries"
            Qout, mask = self.Qoutcomp(self.Q[-1], choices)
            Qnew = self.Q[-1] + lr[..., None]*(outcome[None,...,None]-Qout)*mask
            
            self.Q.append(Qnew)
            
            "--- The following is executed in case of correct and incorrect responses ---"
            "----- Update sequence counters and repetition values of self.rep -----"
            repnew= []
            for agent in range(self.num_agents):
                prev_seq = [str(self.ppchoice[agent].item()) + "," + str(self.pchoice[agent].item()) + "," + str(choices[agent].item())]
                new_row = [0., 0. ,0. ,0.]
                
                if blocktype[agent] == "s":
                    "Sequential Block"
                    " Update counter "
                    self.seq_counter_tb[str(self.pppchoice[agent].item()) + "," + \
                                     str(self.ppchoice[agent].item()) + "," + str(self.pchoice[agent].item()) + "," + str(choices[agent].item())][agent] += 1

                    " Update rep values "
                    for aa in range(4):
                        new_row[aa] = self.seq_counter_tb[prev_seq[0] + "," + str(aa)][agent] / \
                                (self.seq_counter_tb[prev_seq[0] + "," + "0"][agent] + self.seq_counter_tb[prev_seq[0] + "," + "1"][agent] + \
                                 self.seq_counter_tb[prev_seq[0] + "," + "2"][agent] + self.seq_counter_tb[prev_seq[0] + "," + "3"][agent])

                elif blocktype[agent] == "r":
                    "Random Block"
                    " Update counter "
                    self.seq_counter_r[str(self.pppchoice[agent].item()) + "," + \
                                     str(self.ppchoice[agent].item()) + "," + str(self.pchoice[agent].item()) + "," + str(choices[agent].item())][agent] += 1

                    " Update rep values "
                    for aa in range(4):
                        new_row[aa] = self.seq_counter_r[prev_seq[0] + "," + str(aa)][agent] / \
                                (self.seq_counter_r[prev_seq[0] + "," + "0"][agent] + self.seq_counter_r[prev_seq[0] + "," + "1"][agent] + \
                                 self.seq_counter_r[prev_seq[0] + "," + "2"][agent] + self.seq_counter_r[prev_seq[0] + "," + "3"][agent])
                
                else:
                    raise Exception("Da isch a Fehla aba ganz a gwaldiga!")
                    
                repnew.append(new_row)
                
            self.rep.append(torch.tensor(repnew)[None, :].broadcast_to(self.num_particles , self.num_agents, 4))
            
            "----- Compute new V-values for next trial -----"
            V0 = theta_rep*self.rep[-1][..., 0] + theta_Q*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
            V1 = theta_rep*self.rep[-1][..., 1] + theta_Q*self.Q[-1][..., 1]
            V2 = theta_rep*self.rep[-1][..., 2] + theta_Q*self.Q[-1][..., 2]
            V3 = theta_rep*self.rep[-1][..., 3] + theta_Q*self.Q[-1][..., 3]
            self.V.append(torch.stack((V0,V1,V2,V3), 2))

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

    # def find_resp_options(self, stimulus_mat):
    #     """
    #     Given a dual-target stimulus (e.g. 12, 1-indexed), this function returns the two response
    #     options in 0-indexing. E.g.: stimulus_mat=14 -> option1_python = 0, option1_python = 3
    #     INPUT: stimulus in MATLAB notation (1-indexed) (simple list of stimuli)
    #     OUTPUT: response options in python notation (0-indexed)
    #     """
    #     option2_python = ((torch.tensor(stimulus_mat) % 10) - 1).type(torch.int)
    #     option1_python = (((torch.tensor(stimulus_mat) - (torch.tensor(stimulus_mat) % 10)) / 10) -1).type(torch.int)
        
    #     return option1_python, option2_python

    # def choose_action(self, trial, day):
    #     "INPUT: trial (in 1-indexing (i.e. MATLAB notation))"
    #     "OUTPUT: choice response digit (in 0-indexing notation)"
        
    #     if trial < 10:
    #         "Single-target trial"
    #         choice_python = torch.tensor(trial-1)
        
    #     elif trial > 10:
    #         # torch.manual_seed(123)

    #         "Dual-target trial"
    #         option1, option2 = self.find_resp_options([trial])
                     
    #         # self.V[-1].shape is [1, 1, 4] (because choose_action is run only during simulation)
    #         # option1 is torch.tensor([i]) with i being an integer
    #         ipdb.set_trace()
    #         p_actions = self.softmax(torch.tensor([[self.V[-1][..., option1], self.V[-1][..., option2]]]))
                          
    #         # p_actions.shape is [1, 2]
    #         choice_sample = torch.multinomial(p_actions, 1)[0]

    #         choice_python = option2*choice_sample + option1*(1-choice_sample)
            
    #     "Squeeze because if trial > 10 ndim of choice_python is 1 (and not 0)"
    #     return torch.squeeze(choice_python).type('torch.LongTensor')
    
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
        self.Q = [self.Q_init.broadcast_to(self.num_particles, self.num_agents, self.na)] # Goal-Directed Q-Values
        self.rep = [torch.ones(self.num_particles, self.num_agents, 4)*0.25] # habitual values (repetition values)
        
        "Compute V"
        V0 = self.theta_rep_day1*self.rep[-1][..., 0] + self.theta_Q_day1*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
        V1 = self.theta_rep_day1*self.rep[-1][..., 1] + self.theta_Q_day1*self.Q[-1][..., 1]
        V2 = self.theta_rep_day1*self.rep[-1][..., 2] + self.theta_Q_day1*self.Q[-1][..., 2]
        V3 = self.theta_rep_day1*self.rep[-1][..., 3] + self.theta_Q_day1*self.Q[-1][..., 3]
        
        self.V.append(torch.stack((V0,V1,V2,V3), 2))
        
        "Sequence Counters"
        self.init_seq_counter = {}
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-10 in seq_counter for errors)"
        """ seq_counter dims: 
           Dim 0 : random or sequential (0 = sequential, 1 = random)
           Dim 1 : pppchoice 
           Dim 2 : ppchoice
           Dim 3 : pchoice
           Dim 4 : choice
           Dim 5 : idx of agent"""
        self.init_seq_counter = self.k / 4 * np.ones((2, 6, 6, 6, 6,
                                                       self.num_agents))
        self.seq_counter = self.init_seq_counter.copy()

class Vbm_b_onlydual():
    
    def __init__(self, \
                 lr_day1, \
                 theta_Q_day1, \
                 theta_rep_day1, \
                 lr_day2, \
                 theta_Q_day2, \
                 theta_rep_day2, \
                 k, \
                 Q_init, \
                 num_blocks = 14):
        """ 
        --- Parameters ---
        omega (between 0 & 1): weighting factor between habitual and goal-directed: p(a1) = σ(β*[(1-ω)*(r(a1)-r(a2)) + ω*(Q(a1)-Q(a2)))]
        dectemp (between 0 & inf): decision temperature β
        lr (between 0 & 1) : learning rate
        Q_init : (list of floats) initial Q-Values"""
        self.na = 4 # no. of possible actions
        self.num_particles = lr_day1.shape[0]
        self.num_agents = lr_day1.shape[1] # number of agents
        
        if num_blocks != 1:
            if num_blocks%2 != 0:
                raise Exception("num_blocks must be an even value.")
        
        self.trials = 480*num_blocks
        self.num_blocks = num_blocks
        
        self.theta_rep_day1 = torch.tensor([[theta_rep_day1]])
        self.theta_Q_day1 = torch.tensor([[theta_Q_day1]]) # dectemp > 0
        self.lr_day1 = torch.tensor([[lr_day1]])
        
        self.theta_rep_day2 = torch.tensor([[theta_rep_day2]])
        self.theta_Q_day2 = torch.tensor([[theta_Q_day2]]) # dectemp > 0
        self.lr_day2 = torch.tensor([[lr_day2]])
        
        self.k = torch.tensor([[k]])
        self.Q_init = Q_init
        self.Q = [torch.tensor([Q_init],)] # Goal-Directed Q-Values
        self.rep = [torch.tensor([[0.25, 0.25, 0.25, 0.25]],)] # habitual values (repetition values)
    
        "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"
        V0 = self.theta_rep_day1*self.rep[-1][..., 0] + self.theta_Q_day1*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
        V1 = self.theta_rep_day1*self.rep[-1][..., 1] + self.theta_Q_day1*self.Q[-1][..., 1]
        V2 = self.theta_rep_day1*self.rep[-1][..., 2] + self.theta_Q_day1*self.Q[-1][..., 2]
        V3 = self.theta_rep_day1*self.rep[-1][..., 3] + self.theta_Q_day1*self.Q[-1][..., 3]
        
        self.V = [torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1)]
        
        "----- Set sequence counter -----"
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        self.init_seq_counter = {}
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-10 in seq_counter for errors)"
        """ seq_counter dims: 
           Dim 0 : random or sequential (0 = sequential, 1 = random)
           Dim 1 : pppchoice 
           Dim 2 : ppchoice
           Dim 3 : pchoice
           Dim 4 : choice
           Dim 5 : idx of agent"""
        self.init_seq_counter = self.k / 4 * np.ones((2, 6, 6, 6, 6,
                                                       self.num_agents))
        self.seq_counter = self.init_seq_counter.copy()

    def softmax(self, z):
        sm = torch.nn.Softmax(dim=1)
        p_actions = sm(z)
        return p_actions
        
    def update(self, choice, outcome, blocktype, day, trialstimulus, **kwargs):
        """
        Is called after a dual-target choice and updates Q-values, sequence counters, habit values (i.e. repetition values), and V-Values.
        
        choices : the single-target trial choices before the next dual-taregt trial (<0 is error) (0-indexed)"
        
        --- Parameters ---
        choice (-10, 0, 1, 2, or 3): The particiapnt's choice at the dual-target trial
                                     -10 : error
        outcome (0 or 1) : no reward (0) or reward (1)
        blocktype : 's' (sequential blocks) or 'r' (random blocks)
                    Important for updating of sequence counters.
        """

        if day == 1:
            lr = self.lr_day1
            theta_Q = self.theta_Q_day1
            theta_rep = self.theta_rep_day1
            
        elif day == 2:
            lr = self.lr_day2
            theta_Q = self.theta_Q_day2
            theta_rep = self.theta_rep_day2
        
        
        if choice == -1 and outcome == -1 and blocktype == -1:
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1
            self.ppchoice = -1
            self.pchoice = -1

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.tensor([[0.25, 0.25, 0.25, 0.25]],))
            self.Q.append(self.Q[-1])
            
            V0 = theta_rep*self.rep[-1][..., 0] + theta_Q*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
            V1 = theta_rep*self.rep[-1][..., 1] + theta_Q*self.Q[-1][..., 1]
            V2 = theta_rep*self.rep[-1][..., 2] + theta_Q*self.Q[-1][..., 2]
            V3 = theta_rep*self.rep[-1][..., 3] + theta_Q*self.Q[-1][..., 3]
            self.V.append(torch.cat((V0,V1,V2,V3))[None, :])
            
        else:
            
            ch = choice.type('torch.LongTensor')
            outcome = outcome.type('torch.LongTensor')
            
            "----- Update GD-values -----"
            # Outcome is either 0 or 1
            if ch > -1:
                "No error"
                if trialstimulus > 10:
                    "Q-learning update only after dual-target trial!"
                    Qchoice = (self.Q[-1][..., ch][:,None] + lr*(outcome-self.Q[-1][..., ch][:,None])) * torch.eye(self.na)[ch, :]
                    mask = torch.eye(self.na, dtype=bool)[ch, :]
                    Qnew = torch.where(mask, Qchoice, self.Q[-1])
                    self.Q.append(Qnew)
                
                assert(outcome == 0 or outcome == 1)
                
            "--- The following is executed in case of correct and inocrrect responses ---"
            "----- Update sequence counters -----"
            if blocktype == "s":
                "Sequential Block"
                self.seq_counter_tb[str(self.pppchoice) + "," + \
                                 str(self.ppchoice) + "," + str(self.pchoice) + "," + str(ch.item())] += 1
    
            elif blocktype == "r":
                "Random Block"
                self.seq_counter_r[str(self.pppchoice) + "," + \
                                 str(self.ppchoice) + "," + str(self.pchoice) + "," + str(ch.item())] += 1
                    
            else:
                ipdb.set_trace()
                raise Exception("Da isch a Fehla aba ganz a gwaldiga!")
            
            "----- Update repetition values self.rep -----"
            prev_seq = [str(self.ppchoice) + "," + str(self.pchoice) + "," + str(ch.item())]
            
            new_row = [0., 0. ,0. ,0.]
            for aa in range(4):
            
                if blocktype == 's':
                    for aa in range(4):
                        new_row[aa] = self.seq_counter_tb[prev_seq[0] + "," + str(aa)] / \
                                (self.seq_counter_tb[prev_seq[0] + "," + "0"] + self.seq_counter_tb[prev_seq[0] + "," + "1"] + \
                                 self.seq_counter_tb[prev_seq[0] + "," + "2"] + self.seq_counter_tb[prev_seq[0] + "," + "3"])
    
                elif blocktype == 'r':
                    for aa in range(4):
                        new_row[aa] = self.seq_counter_r[prev_seq[0] + "," + str(aa)] / \
                                (self.seq_counter_r[prev_seq[0] + "," + "0"] + self.seq_counter_r[prev_seq[0] + "," + "1"] + \
                                 self.seq_counter_r[prev_seq[0] + "," + "2"] + self.seq_counter_r[prev_seq[0] + "," + "3"])
                                    
                else:
                    raise Exception("Da isch a Fehla aba ganz agwaldiga!")
    
            self.rep.append(torch.tensor([new_row],))
            
            "----- Compute new V-values for next trial -----"
            V0 = theta_rep*self.rep[-1][..., 0] + theta_Q*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
            V1 = theta_rep*self.rep[-1][..., 1] + theta_Q*self.Q[-1][..., 1]
            V2 = theta_rep*self.rep[-1][..., 2] + theta_Q*self.Q[-1][..., 2]
            V3 = theta_rep*self.rep[-1][..., 3] + theta_Q*self.Q[-1][..., 3]
            
            self.V.append(torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1))
            
            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = ch.item()

    def find_resp_options(self, stimulus_mat):
        """Given a dual-target stimulus (e.g. 12, 1-indexed), this function returns the two response
        options in 0-indexing. E.g.: stimulus_mat=14 -> option1_python = 0, option1_python = 3
        INPUT: stimulus in MATLAB notation (1-indexed)
        OUTPUT: response options in python notation (0-indexed)
        """
        
        option2_python = int((stimulus_mat % 10) - 1 )
        option1_python = int(((stimulus_mat - (stimulus_mat % 10)) / 10) -1)
        
        return option1_python, option2_python

    def choose_action(self, trial, day):
        "INPUT: trial (in 1-indexing (i.e. MATLAB notation))"
        "OUTPUT: choice response digit (in 0-indexing notation)"
        
        if trial < 10:
            "Single-target trial"
            choice_python = trial-1
        
        elif trial > 10:
            # torch.manual_seed(123)

            "Dual-target trial"
            option1, option2 = self.find_resp_options(trial)
                        
            p_actions = self.softmax(torch.tensor([[self.V[-1][:, option1], self.V[-1][:, option2]]]))
                                    
            choice_sample = torch.multinomial(p_actions, 1)[0]

            choice_python = option2*choice_sample + option1*(1-choice_sample)
            
        print(type(choice_python))
            
        return torch.squeeze(torch.tensor(choice_python)).type('torch.LongTensor')
        
        # return torch.squeeze(choice_python).type('torch.LongTensor')
    
    def reset(self, **kwargs):
        self.lr_day1 = kwargs["lr_day1"]
        self.theta_Q_day1 = kwargs["theta_Q_day1"]
        self.theta_rep_day1 = kwargs["theta_rep_day1"]        
        
        self.lr_day2 = kwargs["lr_day2"]
        self.theta_Q_day2 = kwargs["theta_Q_day2"]
        self.theta_rep_day2 = kwargs["theta_rep_day2"]

        self.k = kwargs["k"]
            
        self.Q = [torch.tensor([self.Q_init],)] # Goal-Directed Q-Values
        
        #self.p_actions_hist = [torch.tensor([[0.25, 0.25, 0.25, 0.25]],)]
        self.na = 4 # no. of possible actions
        self.rep = [torch.ones((1, self.na))*0.25] # habitual values (repetition values)
        # Compute prior over sequences of length 4
        
        V0 = self.theta_rep_day1*self.rep[-1][..., 0] + self.theta_Q_day1*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
        V1 = self.theta_rep_day1*self.rep[-1][..., 1] + self.theta_Q_day1*self.Q[-1][..., 1]
        V2 = self.theta_rep_day1*self.rep[-1][..., 2] + self.theta_Q_day1*self.Q[-1][..., 2]
        V3 = self.theta_rep_day1*self.rep[-1][..., 3] + self.theta_Q_day1*self.Q[-1][..., 3]
        
        self.V = [torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1)]
        
        self.seq_counter = self.init_seq_counter.copy()

class Vbm_b_2():
    "Like model B, but with separate parameters for the first two blocks"
    
    def __init__(self, \
                 lr_day1_1, \
                 theta_Q_day1_1, \
                 theta_rep_day1_1, \
                 lr_day1_2, \
                 theta_Q_day1_2, \
                 theta_rep_day1_2, \
                 lr_day2, \
                 theta_Q_day2, \
                 theta_rep_day2, \
                 k, \
                 Q_init, \
                 num_blocks = 14):
        
        """ 
        --- Parameters ---
        omega (between 0 & 1): weighting factor between habitual and goal-directed: p(a1) = σ(β*[(1-ω)*(r(a1)-r(a2)) + ω*(Q(a1)-Q(a2)))]
        dectemp (between 0 & inf): decision temperature β
        lr (between 0 & 1) : learning rate
        Q_init : (list of floats) initial Q-Values"""
        self.na = 4 # no. of possible actions
        
        if num_blocks != 1:
            if num_blocks%2 != 0:
                raise Exception("num_blocks must be an even value.")

        self.trials = 480*num_blocks
        self.num_blocks = num_blocks
        
        self.theta_rep_day1_1 = torch.tensor([[theta_rep_day1_1]])
        self.theta_Q_day1_1 = torch.tensor([[theta_Q_day1_1]]) # dectemp > 0
        self.lr_day1_1 = torch.tensor([[lr_day1_1]])
        
        self.theta_rep_day1_2 = torch.tensor([[theta_rep_day1_2]])
        self.theta_Q_day1_2 = torch.tensor([[theta_Q_day1_2]]) # dectemp > 0
        self.lr_day1_2 = torch.tensor([[lr_day1_2]])
        
        self.theta_rep_day2 = torch.tensor([[theta_rep_day2]])
        self.theta_Q_day2 = torch.tensor([[theta_Q_day2]]) # dectemp > 0
        self.lr_day2 = torch.tensor([[lr_day2]])
        
        self.k = torch.tensor([[k]])
        self.Q_init = Q_init
        self.Q = [torch.tensor([Q_init],)] # Goal-Directed Q-Values
        self.rep = [torch.tensor([[0.25, 0.25, 0.25, 0.25]],)] # habitual values (repetition values)
    
        "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"
        V0 = self.theta_rep_day1_1*self.rep[-1][..., 0] + self.theta_Q_day1_1*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
        V1 = self.theta_rep_day1_1*self.rep[-1][..., 1] + self.theta_Q_day1_1*self.Q[-1][..., 1]
        V2 = self.theta_rep_day1_1*self.rep[-1][..., 2] + self.theta_Q_day1_1*self.Q[-1][..., 2]
        V3 = self.theta_rep_day1_1*self.rep[-1][..., 3] + self.theta_Q_day1_1*self.Q[-1][..., 3]
        
        self.V = [torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1)]
        
        "----- Set sequence counter -----"
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        self.init_seq_counter = {}
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-10 in seq_counter for errors)"
        """ seq_counter dims: 
           Dim 0 : random or sequential (0 = sequential, 1 = random)
           Dim 1 : pppchoice 
           Dim 2 : ppchoice
           Dim 3 : pchoice
           Dim 4 : choice
           Dim 5 : idx of agent"""
        self.init_seq_counter = self.k / 4 * np.ones((2, 6, 6, 6, 6,
                                                       self.num_agents))
        self.seq_counter = self.init_seq_counter.copy()

    def softmax(self, z):
        sm = torch.nn.Softmax(dim=1)
        p_actions = sm(z)
        return p_actions
        
    def update(self, choice, outcome, blocktype, **kwargs):
        
        """
        Is called after a dual-target choice and updates Q-values, sequence counters, habit values (i.e. repetition values), and V-Values.
        
        choices : the single-target trial choices before the next dual-taregt trial (<0 is error) (0-indexed)"
        
        --- Parameters ---
        choice (-10, 0, 1, 2, or 3): The particiapnt's choice at the dual-target trial
                                     -10 : error
        outcome (0 or 1) : no reward (0) or reward (1)
        blocktype : 's' (sequential blocks) or 'r' (random blocks)
                    Important for updating of sequence counters.
        exp_part : part of the experiment (1: day 1_1, 2: day 1_2, 3: day2)
        """

        if kwargs['exp_part'] == 1:
            lr = self.lr_day1_1
            theta_Q = self.theta_Q_day1_1
            theta_rep = self.theta_rep_day1_1
            
        elif kwargs['exp_part'] == 2:
            lr = self.lr_day1_2
            theta_Q = self.theta_Q_day1_2
            theta_rep = self.theta_rep_day1_2
            
        elif kwargs['exp_part'] == 3:
            lr = self.lr_day2
            theta_Q = self.theta_Q_day2
            theta_rep = self.theta_rep_day2
            
        else:
            raise Exception("Da isch a Fehla!")
        
        if choice == -1 and outcome == -1 and blocktype == -1:
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1
            self.ppchoice = -1
            self.pchoice = -1

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.tensor([[0.25, 0.25, 0.25, 0.25]],))
            self.Q.append(self.Q[-1])
            
            V0 = theta_rep*self.rep[-1][..., 0] + theta_Q*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
            V1 = theta_rep*self.rep[-1][..., 1] + theta_Q*self.Q[-1][..., 1]
            V2 = theta_rep*self.rep[-1][..., 2] + theta_Q*self.Q[-1][..., 2]
            V3 = theta_rep*self.rep[-1][..., 3] + theta_Q*self.Q[-1][..., 3]
            self.V.append(torch.cat((V0,V1,V2,V3))[None, :])
            
        else:
            ch = choice.type('torch.LongTensor')
            outcome = outcome.type('torch.LongTensor')
            
            "----- Update GD-values -----"
            # Outcome is either 0 or 1
            if ch > -1:
                "No error"
                #if trialstimulus > 10:
                "Q-learning update only after dual-target trial!"
                Qchoice = (self.Q[-1][..., ch][:,None] + lr*(outcome-self.Q[-1][..., ch][:,None])) * torch.eye(self.na)[ch, :]
                mask = torch.eye(self.na, dtype=bool)[ch, :]
                Qnew = torch.where(mask, Qchoice, self.Q[-1])
                self.Q.append(Qnew)
                
                assert(outcome == 0 or outcome == 1)
                
            "--- The following is executed in case of correct and inocrrect responses ---"
            "----- Update sequence counters -----"
            if blocktype == "s":
                "Sequential Block"
                self.seq_counter_tb[str(self.pppchoice) + "," + \
                                 str(self.ppchoice) + "," + str(self.pchoice) + "," + str(ch.item())] += 1
    
            elif blocktype == "r":
                "Random Block"
                self.seq_counter_r[str(self.pppchoice) + "," + \
                                 str(self.ppchoice) + "," + str(self.pchoice) + "," + str(ch.item())] += 1
                    
            else:
                raise Exception("Da isch a Fehla aba ganz a gwaldiga!")
            
            "----- Update repetition values self.rep -----"
            prev_seq = [str(self.ppchoice) + "," + str(self.pchoice) + "," + str(ch.item())]
            
            new_row = [0., 0. ,0. ,0.]
            for aa in range(4):
            
                if blocktype == 's':
                    for aa in range(4):
                        new_row[aa] = self.seq_counter_tb[prev_seq[0] + "," + str(aa)] / \
                                (self.seq_counter_tb[prev_seq[0] + "," + "0"] + self.seq_counter_tb[prev_seq[0] + "," + "1"] + \
                                 self.seq_counter_tb[prev_seq[0] + "," + "2"] + self.seq_counter_tb[prev_seq[0] + "," + "3"])
    
                elif blocktype == 'r':
                    for aa in range(4):
                        new_row[aa] = self.seq_counter_r[prev_seq[0] + "," + str(aa)] / \
                                (self.seq_counter_r[prev_seq[0] + "," + "0"] + self.seq_counter_r[prev_seq[0] + "," + "1"] + \
                                 self.seq_counter_r[prev_seq[0] + "," + "2"] + self.seq_counter_r[prev_seq[0] + "," + "3"])
                                    
                else:
                    raise Exception("Da isch a Fehla aba ganz agwaldiga!")
    
            self.rep.append(torch.tensor([new_row],))
            
            "----- Compute new V-values for next trial -----"
            V0 = theta_rep*self.rep[-1][..., 0] + theta_Q*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
            V1 = theta_rep*self.rep[-1][..., 1] + theta_Q*self.Q[-1][..., 1]
            V2 = theta_rep*self.rep[-1][..., 2] + theta_Q*self.Q[-1][..., 2]
            V3 = theta_rep*self.rep[-1][..., 3] + theta_Q*self.Q[-1][..., 3]
            
            self.V.append(torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1))
            
            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = ch.item()

    def find_resp_options(self, stimulus_mat):
        """Given a dual-target stimulus (e.g. 12, 1-indexed), this function returns the two response
        options in 0-indexing. E.g.: stimulus_mat=14 -> option1_python = 0, option1_python = 3
        INPUT: stimulus in MATLAB notation (1-indexed)
        OUTPUT: response options in python notation (0-indexed)
        """
        
        option2_python = int((stimulus_mat % 10) - 1 )
        option1_python = int(((stimulus_mat - (stimulus_mat % 10)) / 10) -1)
        
        return option1_python, option2_python

    def choose_action(self, trial, day):
        "INPUT: trial (in 1-indexing (i.e. MATLAB notation))"
        "OUTPUT: choice response digit (in 0-indexing notation)"
        
        if trial < 10:
            "Single-target trial"
            choice_python = trial-1
        
        elif trial > 10:
            "Dual-target trial"
            option1, option2 = self.find_resp_options(trial)
                        
            p_actions = self.softmax(torch.tensor([[self.V[-1][:, option1], self.V[-1][:, option2]]]))
                                    
            choice_sample = torch.multinomial(p_actions, 1)[0]
    
            choice_python = option2*choice_sample + option1*(1-choice_sample)

        return torch.squeeze(choice_python).type('torch.LongTensor')
    
    def reset(self, **kwargs):
        self.lr_day1_1 = kwargs["lr_day1_1"]
        self.theta_Q_day1_1 = kwargs["theta_Q_day1_1"]
        self.theta_rep_day1_1 = kwargs["theta_rep_day1_1"]
        
        self.lr_day1_2 = kwargs["lr_day1_2"]
        self.theta_Q_day1_2 = kwargs["theta_Q_day1_2"]
        self.theta_rep_day1_2 = kwargs["theta_rep_day1_2"]
        
        self.lr_day2 = kwargs["lr_day2"]
        self.theta_Q_day2 = kwargs["theta_Q_day2"]
        self.theta_rep_day2 = kwargs["theta_rep_day2"]

        self.k = kwargs["k"]
        self.Q = [torch.tensor([self.Q_init],)] # Goal-Directed Q-Values
        
        #self.p_actions_hist = [torch.tensor([[0.25, 0.25, 0.25, 0.25]],)]
        self.na = 4 # no. of possible actions
        self.rep = [torch.ones((1, self.na))*0.25] # habitual values (repetition values)
        # Compute prior over sequences of length 4
        
        V0 = self.theta_rep_day1_1*self.rep[-1][..., 0] + self.theta_Q_day1_1*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
        V1 = self.theta_rep_day1_1*self.rep[-1][..., 1] + self.theta_Q_day1_1*self.Q[-1][..., 1]
        V2 = self.theta_rep_day1_1*self.rep[-1][..., 2] + self.theta_Q_day1_1*self.Q[-1][..., 2]
        V3 = self.theta_rep_day1_1*self.rep[-1][..., 3] + self.theta_Q_day1_1*self.Q[-1][..., 3]
        
        self.V = [torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1)]
        
        self.seq_counter = self.init_seq_counter.copy()
                        
class Vbm_b_3():
    "Like model B, but with separate parameters for the first two blocks"
    
    def __init__(self, \
                 theta_Q_day1_1, \
                 theta_rep_day1_1, \
                 theta_Q_day1_2, \
                 theta_rep_day1_2, \
                 theta_Q_day2, \
                 theta_rep_day2, \
                 k, \
                 Q_init, \
                 num_blocks = 14):
        
        """ 
        --- Parameters ---
        omega (between 0 & 1): weighting factor between habitual and goal-directed: p(a1) = σ(β*[(1-ω)*(r(a1)-r(a2)) + ω*(Q(a1)-Q(a2)))]
        dectemp (between 0 & inf): decision temperature β
        lr (between 0 & 1) : learning rate
        Q_init : (list of floats) initial Q-Values"""
        self.na = 4 # no. of possible actions
        
        if num_blocks != 1:
            if num_blocks%2 != 0:
                raise Exception("num_blocks must be an even value.")

        self.trials = 480*num_blocks
        self.num_blocks = num_blocks
        
        self.theta_rep_day1_1 = torch.tensor([[theta_rep_day1_1]])
        self.theta_Q_day1_1 = torch.tensor([[theta_Q_day1_1]]) # dectemp > 0
        
        self.theta_rep_day1_2 = torch.tensor([[theta_rep_day1_2]])
        self.theta_Q_day1_2 = torch.tensor([[theta_Q_day1_2]]) # dectemp > 0
        
        self.theta_rep_day2 = torch.tensor([[theta_rep_day2]])
        self.theta_Q_day2 = torch.tensor([[theta_Q_day2]]) # dectemp > 0
        
        self.k = torch.tensor([[k]])
        self.Q_init = Q_init
        self.Q = [torch.tensor([Q_init],)] # Goal-Directed Q-Values
        self.rep = [torch.tensor([[0.25, 0.25, 0.25, 0.25]],)] # habitual values (repetition values)
    
        "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"
        V0 = self.theta_rep_day1_1*self.rep[-1][..., 0] + self.theta_Q_day1_1*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
        V1 = self.theta_rep_day1_1*self.rep[-1][..., 1] + self.theta_Q_day1_1*self.Q[-1][..., 1]
        V2 = self.theta_rep_day1_1*self.rep[-1][..., 2] + self.theta_Q_day1_1*self.Q[-1][..., 2]
        V3 = self.theta_rep_day1_1*self.rep[-1][..., 3] + self.theta_Q_day1_1*self.Q[-1][..., 3]
        
        self.V = [torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1)]
        
        "----- Set sequence counter -----"
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        self.init_seq_counter = {}
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-10 in seq_counter for errors)"
        """ seq_counter dims: 
           Dim 0 : random or sequential (0 = sequential, 1 = random)
           Dim 1 : pppchoice 
           Dim 2 : ppchoice
           Dim 3 : pchoice
           Dim 4 : choice
           Dim 5 : idx of agent"""
        self.init_seq_counter = self.k / 4 * np.ones((2, 6, 6, 6, 6,
                                                       self.num_agents))
        self.seq_counter = self.init_seq_counter.copy()

    def softmax(self, z):
        sm = torch.nn.Softmax(dim=1)
        p_actions = sm(z)
        return p_actions
        
    def update(self, choice, outcome, blocktype, **kwargs):
        
        """
        Is called after a dual-target choice and updates Q-values, sequence counters, habit values (i.e. repetition values), and V-Values.
        
        choices : the single-target trial choices before the next dual-taregt trial (<0 is error) (0-indexed)"
        
        --- Parameters ---
        choice (-10, 0, 1, 2, or 3): The particiapnt's choice at the dual-target trial
                                     -10 : error
        outcome (0 or 1) : no reward (0) or reward (1)
        blocktype : 's' (sequential blocks) or 'r' (random blocks)
                    Important for updating of sequence counters.
        exp_part : part of the experiment (1: day 1_1, 2: day 1_2, 3: day2)
        """

        if kwargs['exp_part'] == 1:
            theta_Q = self.theta_Q_day1_1
            theta_rep = self.theta_rep_day1_1
            
        elif kwargs['exp_part'] == 2:
            theta_Q = self.theta_Q_day1_2
            theta_rep = self.theta_rep_day1_2
            
        elif kwargs['exp_part'] == 3:
            theta_Q = self.theta_Q_day2
            theta_rep = self.theta_rep_day2
            
        else:
            raise Exception("Da isch a Fehla!")
        
        if choice == -1 and outcome == -1 and blocktype == -1:
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1
            self.ppchoice = -1
            self.pchoice = -1

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.tensor([[0.25, 0.25, 0.25, 0.25]],))
            self.Q.append(self.Q[-1])
            
            V0 = theta_rep*self.rep[-1][..., 0] + theta_Q*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
            V1 = theta_rep*self.rep[-1][..., 1] + theta_Q*self.Q[-1][..., 1]
            V2 = theta_rep*self.rep[-1][..., 2] + theta_Q*self.Q[-1][..., 2]
            V3 = theta_rep*self.rep[-1][..., 3] + theta_Q*self.Q[-1][..., 3]
            self.V.append(torch.cat((V0,V1,V2,V3))[None, :])
            
        else:
            ch = choice.type('torch.LongTensor')
            outcome = outcome.type('torch.LongTensor')
            
            "----- No need to Update GD-values -----"
                
            "--- The following is executed in case of correct and inocrrect responses ---"
            "----- Update sequence counters -----"
            if blocktype == "s":
                "Sequential Block"
                self.seq_counter_tb[str(self.pppchoice) + "," + \
                                 str(self.ppchoice) + "," + str(self.pchoice) + "," + str(ch.item())] += 1
    
            elif blocktype == "r":
                "Random Block"
                self.seq_counter_r[str(self.pppchoice) + "," + \
                                 str(self.ppchoice) + "," + str(self.pchoice) + "," + str(ch.item())] += 1
                    
            else:
                raise Exception("Da isch a Fehla aba ganz a gwaldiga!")
            
            "----- Update repetition values self.rep -----"
            prev_seq = [str(self.ppchoice) + "," + str(self.pchoice) + "," + str(ch.item())]
            
            new_row = [0., 0. ,0. ,0.]
            for aa in range(4):
            
                if blocktype == 's':
                    for aa in range(4):
                        new_row[aa] = self.seq_counter_tb[prev_seq[0] + "," + str(aa)] / \
                                (self.seq_counter_tb[prev_seq[0] + "," + "0"] + self.seq_counter_tb[prev_seq[0] + "," + "1"] + \
                                 self.seq_counter_tb[prev_seq[0] + "," + "2"] + self.seq_counter_tb[prev_seq[0] + "," + "3"])
    
                elif blocktype == 'r':
                    for aa in range(4):
                        new_row[aa] = self.seq_counter_r[prev_seq[0] + "," + str(aa)] / \
                                (self.seq_counter_r[prev_seq[0] + "," + "0"] + self.seq_counter_r[prev_seq[0] + "," + "1"] + \
                                 self.seq_counter_r[prev_seq[0] + "," + "2"] + self.seq_counter_r[prev_seq[0] + "," + "3"])
                                    
                else:
                    raise Exception("Da isch a Fehla aba ganz agwaldiga!")
    
            self.rep.append(torch.tensor([new_row],))
            
            "----- Compute new V-values for next trial -----"
            V0 = theta_rep*self.rep[-1][..., 0] + theta_Q*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
            V1 = theta_rep*self.rep[-1][..., 1] + theta_Q*self.Q[-1][..., 1]
            V2 = theta_rep*self.rep[-1][..., 2] + theta_Q*self.Q[-1][..., 2]
            V3 = theta_rep*self.rep[-1][..., 3] + theta_Q*self.Q[-1][..., 3]
            
            self.V.append(torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1))
            
            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = ch.item()

    def find_resp_options(self, stimulus_mat):
        """Given a dual-target stimulus (e.g. 12, 1-indexed), this function returns the two response
        options in 0-indexing. E.g.: stimulus_mat=14 -> option1_python = 0, option1_python = 3
        INPUT: stimulus in MATLAB notation (1-indexed)
        OUTPUT: response options in python notation (0-indexed)
        """
        
        option2_python = int((stimulus_mat % 10) - 1 )
        option1_python = int(((stimulus_mat - (stimulus_mat % 10)) / 10) -1)
        
        return option1_python, option2_python

    def choose_action(self, trial, day):
        "INPUT: trial (in 1-indexing (i.e. MATLAB notation))"
        "OUTPUT: choice response digit (in 0-indexing notation)"
        
        if trial < 10:
            "Single-target trial"
            choice_python = trial-1
        
        elif trial > 10:
            "Dual-target trial"
            option1, option2 = self.find_resp_options(trial)
                        
            p_actions = self.softmax(torch.tensor([[self.V[-1][:, option1], self.V[-1][:, option2]]]))
                                    
            choice_sample = torch.multinomial(p_actions, 1)[0]
    
            choice_python = option2*choice_sample + option1*(1-choice_sample)

        return torch.squeeze(choice_python).type('torch.LongTensor')
    
    def reset(self, **kwargs):
        self.theta_Q_day1_1 = kwargs["theta_Q_day1_1"]
        self.theta_rep_day1_1 = kwargs["theta_rep_day1_1"]
        
        self.theta_Q_day1_2 = kwargs["theta_Q_day1_2"]
        self.theta_rep_day1_2 = kwargs["theta_rep_day1_2"]
        
        self.theta_Q_day2 = kwargs["theta_Q_day2"]
        self.theta_rep_day2 = kwargs["theta_rep_day2"]

        self.k = kwargs["k"]
        self.Q = [torch.tensor([self.Q_init],)] # Goal-Directed Q-Values
        
        #self.p_actions_hist = [torch.tensor([[0.25, 0.25, 0.25, 0.25]],)]
        self.na = 4 # no. of possible actions
        self.rep = [torch.ones((1, self.na))*0.25] # habitual values (repetition values)
        # Compute prior over sequences of length 4
        
        V0 = self.theta_rep_day1_1*self.rep[-1][..., 0] + self.theta_Q_day1_1*self.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
        V1 = self.theta_rep_day1_1*self.rep[-1][..., 1] + self.theta_Q_day1_1*self.Q[-1][..., 1]
        V2 = self.theta_rep_day1_1*self.rep[-1][..., 2] + self.theta_Q_day1_1*self.Q[-1][..., 2]
        V3 = self.theta_rep_day1_1*self.rep[-1][..., 3] + self.theta_Q_day1_1*self.Q[-1][..., 3]
        
        self.V = [torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1)]
        
        self.seq_counter = self.init_seq_counter.copy()

class Vbm_f():
    "Fixed Q-values, but theta_rep and theta_Q develop linearly"
    def __init__(self, \
                 theta_Q0_day1, \
                 theta_Qlambda_day1, \
                 theta_rep0_day1, \
                 theta_replambda_day1, \
                 theta_Q0_day2, \
                 theta_Qlambda_day2, \
                 theta_rep0_day2, \
                 theta_replambda_day2, \
                 k, \
                 Q_init, \
                 num_blocks = 14):
        
        """ 
        --- Parameters ---
        omega (between 0 & 1): weighting factor between habitual and goal-directed: p(a1) = σ(β*[(1-ω)*(r(a1)-r(a2)) + ω*(Q(a1)-Q(a2)))]
        dectemp (between 0 & inf): decision temperature β
        lr (between 0 & 1) : learning rate
        Q_init : (list of floats) initial Q-Values"""
        self.na = 4 # no. of possible actions
        
        if num_blocks != 1:
            if num_blocks%2 != 0:
                raise Exception("num_blocks must be an even value.")
        
        self.trials = 480*num_blocks
        self.num_blocks = num_blocks
        
        self.day = 1
        " ===== Latent parameters ====="
        self.theta_rep0_day1 = torch.tensor([[theta_rep0_day1]])
        self.theta_replambda_day1 = torch.tensor([[theta_replambda_day1]])
        
        self.theta_Q0_day1 = torch.tensor([[theta_Q0_day1]])
        self.theta_Qlambda_day1 = torch.tensor([[theta_Qlambda_day1]])
        
        self.theta_rep0_day2 = torch.tensor([[theta_rep0_day2]])
        self.theta_replambda_day2 = torch.tensor([[theta_replambda_day2]])
        
        self.theta_Q0_day2 = torch.tensor([[theta_Q0_day2]])
        self.theta_Qlambda_day2 = torch.tensor([[theta_Qlambda_day2]])
        " ===== ===== ===== ====="
        
        self.theta_rep = [self.theta_rep0_day1]
        self.theta_Q = [self.theta_Q0_day1]
                
        self.k = torch.tensor([[k]])
        self.Q_init = Q_init
        self.Q = torch.tensor([Q_init],) # Gial-Directed Q-Values
        self.rep = [torch.tensor([[0.25, 0.25, 0.25, 0.25]],)] # habitual values (repetition values)

        "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"
        V0 = self.theta_rep[-1]*self.rep[-1][..., 0] + self.theta_Q[-1]*self.Q[..., 0] # V-Values for actions (i.e. weighted action values)
        V1 = self.theta_rep[-1]*self.rep[-1][..., 1] + self.theta_Q[-1]*self.Q[..., 1]
        V2 = self.theta_rep[-1]*self.rep[-1][..., 2] + self.theta_Q[-1]*self.Q[..., 2]
        V3 = self.theta_rep[-1]*self.rep[-1][..., 3] + self.theta_Q[-1]*self.Q[..., 3]
        
        self.V = [torch.cat((V0.transpose(0,1), V1.transpose(0,1), V2.transpose(0,1), V3.transpose(0,1)),1)]
        
        "----- Set sequence counter -----"
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        self.init_seq_counter = {}
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-10 in seq_counter for errors)"
        """ seq_counter dims: 
           Dim 0 : random or sequential (0 = sequential, 1 = random)
           Dim 1 : pppchoice 
           Dim 2 : ppchoice
           Dim 3 : pchoice
           Dim 4 : choice
           Dim 5 : idx of agent"""
        self.init_seq_counter = self.k / 4 * np.ones((2, 6, 6, 6, 6,
                                                       self.num_agents))
        self.seq_counter = self.init_seq_counter.copy()

    def softmax(self, z):
        sm = torch.nn.Softmax(dim=1)
        p_actions = sm(z)
        return p_actions
        
    def update(self, choice, outcome, blocktype, **kwargs):
        """
        Is called after every choice and updates Q-values, sequence counters, habit values (i.e. repetition values), and V-Values.
        
        choices : the single-target trial choices before the next dual-taregt trial (<0 is error) (0-indexed)
        
        trial : 
        
        --- Parameters ---
        choice (-10, 0, 1, 2, or 3): The particiapnt's choice at the dual-target trial
                                     -10 : error
        outcome (0 or 1) : no reward (0) or reward (1)
        blocktype : 's' (sequential blocks) or 'r' (random blocks)
                    Important for updating of sequence counters.
        """
        
        if choice == -1 and outcome == -1 and blocktype == -1:
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1
            self.ppchoice = -1
            self.pchoice = -1

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.tensor([[0.25, 0.25, 0.25, 0.25]],))
            
            V0 = self.theta_rep[-1]*self.rep[-1][..., 0] + self.theta_Q[-1]*self.Q[..., 0] # V-Values for actions (i.e. weighted action values)
            V1 = self.theta_rep[-1]*self.rep[-1][..., 1] + self.theta_Q[-1]*self.Q[..., 1]
            V2 = self.theta_rep[-1]*self.rep[-1][..., 2] + self.theta_Q[-1]*self.Q[..., 2]
            V3 = self.theta_rep[-1]*self.rep[-1][..., 3] + self.theta_Q[-1]*self.Q[..., 3]
            self.V.append(torch.cat((V0,V1,V2,V3))[None, :])
            
        else:
            
            ch = choice.type('torch.LongTensor')
            outcome = outcome.type('torch.LongTensor')
                        
            if self.day == 1 and kwargs['day'] == 2:
                "First Trial of 2nd day"
                self.theta_rep.append(self.theta_rep0_day2)
                self.theta_Q.append(self.theta_Q0_day2)
                self.day = 2
                
            "----- No need to update GD-values -----"
            assert(outcome == 0 or outcome == 1)
                
            "--- The following is executed in case of correct and inocrrect responses ---"
            "----- Update sequence counters -----"
            if blocktype == "s":
                "Sequential Block"
                self.seq_counter_tb[str(self.pppchoice) + "," + \
                                 str(self.ppchoice) + "," + str(self.pchoice) + "," + str(ch.item())] += 1

            elif blocktype == "r":
                "Random Block"
                self.seq_counter_r[str(self.pppchoice) + "," + \
                                 str(self.ppchoice) + "," + str(self.pchoice) + "," + str(ch.item())] += 1
                    
            else:
                raise Exception("Da isch a Fehla aba ganz a gwaldiga!")
            
            "----- Update repetition values self.rep -----"
            prev_seq = [str(self.ppchoice) + "," + str(self.pchoice) + "," + str(ch.item())]
            
            new_row = [0., 0., 0., 0.]
            for aa in range(4):
            
                if blocktype == 's':
                    for aa in range(4):
                        new_row[aa] = self.seq_counter_tb[prev_seq[0] + "," + str(aa)] / \
                                (self.seq_counter_tb[prev_seq[0] + "," + "0"] + self.seq_counter_tb[prev_seq[0] + "," + "1"] + \
                                 self.seq_counter_tb[prev_seq[0] + "," + "2"] + self.seq_counter_tb[prev_seq[0] + "," + "3"])
    
                elif blocktype == 'r':
                    for aa in range(4):
                        new_row[aa] = self.seq_counter_r[prev_seq[0] + "," + str(aa)] / \
                                (self.seq_counter_r[prev_seq[0] + "," + "0"] + self.seq_counter_r[prev_seq[0] + "," + "1"] + \
                                 self.seq_counter_r[prev_seq[0] + "," + "2"] + self.seq_counter_r[prev_seq[0] + "," + "3"])
                                    
                else:
                    raise Exception("Da isch a Fehla aba ganz agwaldiga!")
    
            self.rep.append(torch.tensor([new_row],))
                
            "----- Compute new V-values for next trial -----"
            V0 = self.theta_rep[-1]*self.rep[-1][..., 0] + self.theta_Q[-1]*self.Q[..., 0] # V-Values for actions (i.e. weighted action values)
            V1 = self.theta_rep[-1]*self.rep[-1][..., 1] + self.theta_Q[-1]*self.Q[..., 1]
            V2 = self.theta_rep[-1]*self.rep[-1][..., 2] + self.theta_Q[-1]*self.Q[..., 2]
            V3 = self.theta_rep[-1]*self.rep[-1][..., 3] + self.theta_Q[-1]*self.Q[..., 3]
            
            self.V.append(torch.cat((V0.transpose(0,1), V1.transpose(0,1), V2.transpose(0,1), V3.transpose(0,1)),1))
            
            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = ch.item()
            
            "----- Update  theta_Q and theta_rep -----"
            if self.day == 1:
                self.theta_rep.append(self.theta_rep0_day1 + self.theta_replambda_day1 * kwargs['t'])
                self.theta_Q.append(self.theta_Q0_day1 + self.theta_Qlambda_day1 * kwargs['t'])
                
            elif self.day == 2:
                self.theta_rep.append(self.theta_rep0_day2 + self.theta_replambda_day2 * kwargs['t'])
                self.theta_Q.append(self.theta_Q0_day2 + self.theta_Qlambda_day2 * kwargs['t'])

    def find_resp_options(self, stimulus_mat):
        """
        Given a dual-target stimulus (e.g. 12, 1-indexed), this function returns the two response
        options in 0-indexing. E.g.: stimulus_mat=14 -> option1_python = 0, option1_python = 3
        INPUT: stimulus in MATLAB notation (1-indexed)
        OUTPUT: response options in python notation (0-indexed)
        """
        
        option2_python = int((stimulus_mat % 10) - 1 )
        option1_python = int(((stimulus_mat - (stimulus_mat % 10)) / 10) -1)
        
        return option1_python, option2_python

    def choose_action(self, trial, day):
        "INPUT: trial (in 1-indexing (i.e. MATLAB notation))"
        "OUTPUT: choice response digit (in 0-indexing notation)"
        
        if trial < 10:
            "Single-target trial"
            choice_python = trial-1
        
        elif trial > 10:
            "Dual-target trial"
            option1, option2 = self.find_resp_options(trial)
                        
            p_actions = self.softmax(torch.tensor([[self.V[-1][:, option1], self.V[-1][:, option2]]]))
            
            # self.p_actions_hist.append(p_actions)
            
            choice_sample = torch.multinomial(p_actions, 1)[0]
    
            choice_python = option2*choice_sample + option1*(1-choice_sample)

        return torch.squeeze(choice_python).type('torch.LongTensor')
    
    def reset(self, **kwargs):
        "--- Day 1 ---"
        self.theta_Q0_day1 = kwargs["theta_Q0_day1"]
        self.theta_Qlambda_day1 = kwargs["theta_Qlambda_day1"]
        
        self.theta_rep0_day1 = kwargs["theta_rep0_day1"]
        self.theta_replambda_day1 = kwargs["theta_replambda_day1"]

        self.theta_rep = [self.theta_rep0_day1]
        self.theta_Q = [self.theta_Q0_day1]

        "--- Day 2 ---"
        self.theta_Q0_day2 = kwargs["theta_Q0_day2"]
        self.theta_Qlambda_day2 = kwargs["theta_Qlambda_day2"]
        
        self.theta_rep0_day2 = kwargs["theta_rep0_day2"]
        self.theta_replambda_day2 = kwargs["theta_replambda_day2"]

        self.day = 1
        self.k = kwargs["k"]
        self.Q = torch.tensor([self.Q_init],) # Goal-Directed Q-Values
        
        "=============== Setup ==============="
        #self.p_actions_hist = [torch.tensor([[0.25, 0.25, 0.25, 0.25]],)]
        self.na = 4 # no. of possible actions
        self.rep = [torch.ones((1, self.na))*0.25] # habitual values (repetition values)
        # Compute prior over sequences of length 4
        
        V0 = self.theta_rep[-1]*self.rep[-1][..., 0] + self.theta_Q[-1]*self.Q[..., 0] # V-Values for actions (i.e. weighted action values)
        V1 = self.theta_rep[-1]*self.rep[-1][..., 1] + self.theta_Q[-1]*self.Q[..., 1]
        V2 = self.theta_rep[-1]*self.rep[-1][..., 2] + self.theta_Q[-1]*self.Q[..., 2]
        V3 = self.theta_rep[-1]*self.rep[-1][..., 3] + self.theta_Q[-1]*self.Q[..., 3]
        
        self.V = [torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1)]
        
        self.seq_counter = self.init_seq_counter.copy()

pyro.clear_param_store()