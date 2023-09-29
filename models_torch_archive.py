#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:16:23 2023

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

class Vbm_A():
    def __init__(self, dectemp_day1, lr_day1, omega_day1, dectemp_day2, lr_day2, omega_day2, k, Q_init, num_blocks = 14):
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
        
        self.omega_day1 = torch.tensor([[omega_day1]])
        self.dectemp_day1 = torch.tensor([[dectemp_day1]]) # dectemp > 0
        self.lr_day1 = torch.tensor([[lr_day1]])
        
        self.omega_day2 = torch.tensor([[omega_day2]])
        self.dectemp_day2 = torch.tensor([[dectemp_day2]]) # dectemp > 0
        self.lr_day2 = torch.tensor([[lr_day2]])
        
        self.k = torch.tensor([[k]])
        self.Q_init = Q_init
        self.Q = [torch.tensor([Q_init],)] # Gial-Directed Q-Values
        self.rep = [torch.tensor([[0.25, 0.25, 0.25, 0.25]],)] # habitual values (repetition values)
    
        "V(ai) = (1-ω)*rep_val(ai) + ω*Q(ai)"
        V0 = (1-self.omega_day1)*self.rep[-1][:, 0] + self.omega_day1*self.Q[-1][:, 0] # V-Values for actions (i.e. weighted action values)
        V1 = (1-self.omega_day1)*self.rep[-1][:, 1] + self.omega_day1*self.Q[-1][:, 1]
        V2 = (1-self.omega_day1)*self.rep[-1][:, 2] + self.omega_day1*self.Q[-1][:, 2]
        V3 = (1-self.omega_day1)*self.rep[-1][:, 3] + self.omega_day1*self.Q[-1][:, 3]
        
        self.V = [torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1)]
        
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        self.num_prev_steps = 3 # determines length of learned sequences for sequence counter
        self.seq_counter_tb = {}
        self.seq_counter_r = {}
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-10 in seq_counter for errors)"
        for i in [-10,-1,0,1,2,3]:
            for j in [-10,-1,0,1,2,3]:
                for k in [-10,-1,0,1,2,3]:
                    for l in [-10,-1,0,1,2,3]:
                        self.seq_counter_tb[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = self.k.item()/4
                        self.seq_counter_r[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = self.k.item()/4
        
    def softmax(self, dectemp, z):
        sm = torch.nn.Softmax(dim=1)
        p_actions = sm(dectemp * z)
        return p_actions
        
    def update(self, choice, outcome, blocktype, day, trialstimulus):
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
            omega = self.omega_day1
            
        elif day == 2:
            lr = self.lr_day2
            omega = self.omega_day2
        
        if choice == -1 and outcome == -1 and blocktype == -1:
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1
            self.ppchoice = -1
            self.pchoice = -1

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.tensor([[0.25, 0.25, 0.25, 0.25]],))
            self.Q.append(self.Q[-1])
            
            V0 = (1-omega)*self.rep[-1][:, 0] + omega*self.Q[-1][:, 0] # V-Values for actions (i.e. weighted action values)
            V1 = (1-omega)*self.rep[-1][:, 1] + omega*self.Q[-1][:, 1]
            V2 = (1-omega)*self.rep[-1][:, 2] + omega*self.Q[-1][:, 2]
            V3 = (1-omega)*self.rep[-1][:, 3] + omega*self.Q[-1][:, 3]
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
                Qchoice = (self.Q[-1][:, ch][:,None] + lr*(outcome-self.Q[-1][:, ch][:,None])) * torch.eye(self.na)[ch, :]
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
            V0 = (1-omega)*self.rep[-1][:, 0] + omega*self.Q[-1][:, 0] # V-Values for actions (i.e. weighted action values)
            V1 = (1-omega)*self.rep[-1][:, 1] + omega*self.Q[-1][:, 1]
            V2 = (1-omega)*self.rep[-1][:, 2] + omega*self.Q[-1][:, 2]
            V3 = (1-omega)*self.rep[-1][:, 3] + omega*self.Q[-1][:, 3]
            
            self.V.append(torch.cat((V0.transpose(0,1), V1.transpose(0,1), V2.transpose(0,1), V3.transpose(0,1)),1))
            
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
                        
            if day == 1:
                dectemp = self.dectemp_day1
                
            elif day == 2:
                dectemp = self.dectemp_day2
            
            p_actions = self.softmax(dectemp, torch.tensor([[self.V[-1][:, option1], self.V[-1][:, option2]]]))
            
            # self.p_actions_hist.append(p_actions)
            
            choice_sample = torch.multinomial(p_actions, 1)[0]
    
            choice_python = option2*choice_sample + option1*(1-choice_sample)

        return torch.squeeze(choice_python).type('torch.LongTensor')
    
    def reset(self, **kwargs):
        self.lr_day1 = kwargs["lr_day1"]
        
        self.lr_day2 = kwargs["lr_day2"]
        
        self.omega_day1 = kwargs["omega_day1"]
        
        self.omega_day2 = kwargs["omega_day2"]
        
        self.dectemp_day1 = kwargs["dectemp_day1"]
        
        self.dectemp_day2 = kwargs["dectemp_day2"]

        self.k = kwargs["k"]
            
        self.Q = [torch.tensor([self.Q_init],)] # Gial-Directed Q-Values
        
        #self.p_actions_hist = [torch.tensor([[0.25, 0.25, 0.25, 0.25]],)]
        self.na = 4 # no. of possible actions
        self.rep = [torch.ones((1, self.na))*0.25] # habitual values (repetition values)
        # Compute prior over sequences of length 4
        
        # For single parameters
        V0 = (1-self.omega_day1)*self.rep[-1][:, 0] + self.omega_day1*self.Q[-1][:, 0] # V-Values for actions (i.e. weighted action values)
        V1 = (1-self.omega_day1)*self.rep[-1][:, 1] + self.omega_day1*self.Q[-1][:, 1]
        V2 = (1-self.omega_day1)*self.rep[-1][:, 2] + self.omega_day1*self.Q[-1][:, 2]
        V3 = (1-self.omega_day1)*self.rep[-1][:, 3] + self.omega_day1*self.Q[-1][:, 3]
                
        self.V = [torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1)]
        
        self.num_prev_steps = 3
        self.seq_counter_tb = {}
        self.seq_counter_r = {}
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-10 in seq_counter for errors)"
        for i in [-10,-1,0,1,2,3]:
            for j in [-10,-1,0,1,2,3]:
                for k in [-10,-1,0,1,2,3]:
                    for l in [-10,-1,0,1,2,3]:
                        self.seq_counter_tb[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = self.k.item()/4
                        self.seq_counter_r[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = self.k.item()/4
                        

class SingleInference_modelA(object):

    def __init__(self, agent, data):

        self.agent = agent
        self.trials = self.agent.trials # length of experiment
        #self.T = agent.T # No. of "subtrials" -> unnötig
        self.data = data # Dictionary

    def model(self):
        # generative model of behavior with Normally distributed params (within subject!!)

        # tell pyro about prior over parameters: alpha and beta of lr which is between 0 and 1
        # alpha = beta = 1 equals uniform prior
        "Learning Rate: Beta Distribution"
        lr_day1_alpha = torch.tensor([1.]).to(device)
        lr_day1_beta = torch.tensor([1.]).to(device)
        lr_day1 = pyro.sample('lr_day1', dist.Beta(lr_day1_alpha, lr_day1_beta)).to(device)
        
        lr_day2_alpha = torch.tensor([1.]).to(device)
        lr_day2_beta = torch.tensor([1.]).to(device)
        lr_day2 = pyro.sample('lr_day2', dist.Beta(lr_day2_alpha, lr_day2_beta)).to(device)
                
        "Omega: Beta Distribution"
        omega_day1_alpha = torch.tensor([1.]).to(device)
        omega_day1_beta = torch.tensor([1.]).to(device)
        omega_day1 = pyro.sample('omega_day1', dist.Beta(omega_day1_alpha, omega_day1_beta)).to(device)
        
        omega_day2_alpha = torch.tensor([1.]).to(device)
        omega_day2_beta = torch.tensor([1.]).to(device)      
        omega_day2 = pyro.sample('omega_day2', dist.Beta(omega_day2_alpha, omega_day2_beta)).to(device)
        
        "Decision Temperature : Gamma distribution"
        dectemp_day1_conc = torch.tensor([4.]).to(device)
        dectemp_day1_rate = torch.tensor([4.]).to(device)
        dectemp_day1 = pyro.sample('dectemp_day1', dist.Gamma(dectemp_day1_conc, dectemp_day1_rate)).to(device)
        
        dectemp_day2_conc = torch.tensor([4.]).to(device)
        dectemp_day2_rate = torch.tensor([4.]).to(device)
        dectemp_day2 = pyro.sample('dectemp_day2', dist.Gamma(dectemp_day2_conc, dectemp_day2_rate)).to(device)
        
        "K: Gamma distribution"
        #conc_k = torch.tensor([2]).to(device)        
        #rate_k = torch.tensor([4]).to(device)        
        #k = pyro.sample('k', dist.Gamma(conc_k, rate_k)).to(device)
        k=torch.tensor([4.])
        
        # parameter shape for inference (until now): [1] or [n_particles]
        param_dict = {"lr_day1": lr_day1[:,None], \
                      "lr_day2": lr_day2[:,None], \
                      "omega_day1": omega_day1[None, :], \
                      "omega_day2": omega_day2[None, :], \
                      "dectemp_day1": dectemp_day1[:, None], \
                      "dectemp_day2": dectemp_day2[:, None], \
                      "k": k[:, None]}

        # lr is shape [1,1] or [n_particles, 1]
        self.agent.reset(**param_dict)
        
        t = -1
        for tau in range(self.trials):
            
            trial = self.data["Trialsequence"][tau]
            blocktype = self.data["Blocktype"][tau]
            
            if self.data["Blockidx"][tau] <= 5:
                day = 1
                
            elif self.data["Blockidx"][tau] > 5:
                day = 2
                
            else:
                raise Exception("Da isch a Fehla!")
            
            if trial == -1:
                "Beginning of new block"
                self.agent.update(-1,-1,-1, day, trial)
                
            else:
                current_choice = self.data["Choices"][tau]
                outcome = self.data["Outcomes"][tau]
                
            if trial > 10:
                "Dual-Target Trial"
                t+=1
                option1, option2 = self.agent.find_resp_options(trial)
                
                if day == 1:
                    # Comes out as [1, n_actions] or [n_particles, n_actions]
                    probs = self.agent.softmax(self.agent.dectemp_day1,torch.cat(([self.agent.V[-1][:, option1][:, None], self.agent.V[-1][:, option2][:, None]]),dim=-1))
                    
                elif day == 2:
                    probs = self.agent.softmax(self.agent.dectemp_day2,torch.cat(([self.agent.V[-1][:, option1][:, None], self.agent.V[-1][:, option2][:, None]]),dim=-1))
                
                if current_choice == option1:
                    choice = torch.tensor([0])
                    
                elif current_choice == option2:
                    choice = torch.tensor([1])
                    
                elif current_choice == -10:
                    "Error"
                    pass
                    
                else:
                    raise Exception("Da isch a Fehla")

            if trial != -1:
                "Update (trial == -1 means this is the beginning of a block -> participants didn't see this trial')"
                self.agent.update(current_choice, outcome, blocktype, day, trial)

            "Sample if dual-target trial and no error was performed"
            if trial > 10 and current_choice > -1:
                # choice shape for inference: [1]
                pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), obs=choice)
                    
                    
    def guide(self):
        # approximate posterior. assume MF: each param has its own univariate Normal.

        "Learning Rate"
        lr_day1_alpha = pyro.param("lr_day1_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day1_beta = pyro.param("lr_day1_beta", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day1 = pyro.sample('lr_day1', dist.Beta(lr_day1_alpha, lr_day1_beta)).to(device)

        lr_day2_alpha = pyro.param("lr_day2_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day2_beta = pyro.param("lr_day2_beta", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day2 = pyro.sample('lr_day2', dist.Beta(lr_day2_alpha, lr_day2_beta)).to(device)

        "Omega: Beta Distribution"
        omega_day1_alpha = pyro.param("omega_day1_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        omega_day1_beta = pyro.param("omega_day1_beta", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        omega_day1 = pyro.sample('omega_day1', dist.Beta(omega_day1_alpha, omega_day1_beta)).to(device)

        omega_day2_alpha = pyro.param("omega_day2_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        omega_day2_beta = pyro.param("omega_day2_beta", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        omega_day2 = pyro.sample('omega_day2', dist.Beta(omega_day2_alpha, omega_day2_beta)).to(device)

        "Decision Temperature: Gamma Distribution"
        dectemp_day1_conc = pyro.param("dectemp_day1_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        dectemp_day1_rate = pyro.param("dectemp_day1_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        dectemp_day1 = pyro.sample('dectemp_day1', dist.Gamma(dectemp_day1_conc, dectemp_day1_rate)).to(device)
        
        dectemp_day2_conc = pyro.param("dectemp_day2_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        dectemp_day2_rate = pyro.param("dectemp_day2_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        dectemp_day2 = pyro.sample('dectemp_day2', dist.Gamma(dectemp_day2_conc, dectemp_day2_rate)).to(device)
        
        "K: Gamma Distribution"
        #conc_k = pyro.param("conc_k", torch.tensor([2]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        #rate_k = pyro.param("rate_k", torch.tensor([4]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        #k = pyro.sample('k', dist.Gamma(conc_k, rate_k)).to(device)

        param_dict = {"lr_day1_alpha": lr_day1_alpha, \
                      "lr_day1_beta": lr_day1_beta, \
                      "lr_day1": lr_day1, \
                      "omega_day1_alpha": omega_day1_alpha, \
                      "omega_day1_beta": omega_day1_beta, \
                      "omega_day1": omega_day1, \
                      "dectemp_day1_conc": dectemp_day1_conc, \
                      "dectemp_day1_rate": dectemp_day1_rate, \
                      "dectemp_day1": dectemp_day1, \
                      "lr_day2_alpha": lr_day2_alpha, \
                      "lr_day2_beta": lr_day2_beta, \
                      "lr_day2": lr_day2, \
                      "omega_day2_alpha": omega_day2_alpha, \
                      "omega_day2_beta": omega_day2_beta, \
                      "omega_day2": omega_day2, \
                      "dectemp_day2_conc": dectemp_day2_conc, \
                      "dectemp_day2_rate": dectemp_day2_rate, \
                      "dectemp_day2": dectemp_day2}
            
        return param_dict

    def infer_posterior(self,
                        iter_steps=1_000,
                        num_particles=10,
                        optim_kwargs={'lr': .01}): # Adam learning rate
        """Perform SVI over free model parameters."""

        pyro.clear_param_store()

        svi = pyro.infer.SVI(model=self.model,
                  guide=self.guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  # set below to true once code is vectorized
                                  vectorize_particles=True))

        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:#range(iter_steps):
            loss.append(torch.tensor(svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
            if torch.isnan(loss[-1]):
                break

        self.loss = [l.cpu() for l in loss] # = -ELBO (Plotten!)
        
        lr_day1_alpha = pyro.param("lr_day1_alpha").data.numpy()
        lr_day1_beta = pyro.param("lr_day1_beta").data.numpy()
        
        lr_day2_alpha = pyro.param("lr_day2_alpha").data.numpy()
        lr_day2_beta = pyro.param("lr_day2_beta").data.numpy()
        
        omega_day1_alpha = pyro.param("omega_day1_alpha").data.numpy()
        omega_day1_beta = pyro.param("omega_day1_beta").data.numpy()
        
        omega_day2_alpha = pyro.param("omega_day2_alpha").data.numpy()
        omega_day2_beta = pyro.param("omega_day2_beta").data.numpy()
        
        dectemp_day1_conc = pyro.param("dectemp_day1_conc").data.numpy()
        dectemp_day1_rate = pyro.param("dectemp_day1_rate").data.numpy()
        
        dectemp_day2_conc = pyro.param("dectemp_day2_conc").data.numpy()
        dectemp_day2_rate = pyro.param("dectemp_day2_rate").data.numpy()

        param_dict = {"lr_day1_alpha": lr_day1_alpha, \
                      "lr_day1_beta": lr_day1_beta, \
                      "lr_day2_alpha": lr_day2_alpha, \
                      "lr_day2_beta": lr_day2_beta, \
                      "omega_day1_alpha": omega_day1_alpha, \
                      "omega_day1_beta": omega_day1_beta, \
                      "omega_day2_alpha": omega_day2_alpha, \
                      "omega_day2_beta": omega_day2_beta, \
                      "dectemp_day1_conc": dectemp_day1_conc, \
                      "dectemp_day1_rate": dectemp_day1_rate, \
                      "dectemp_day2_conc": dectemp_day2_conc, \
                      "dectemp_day2_rate": dectemp_day2_rate}

        return self.loss, param_dict
    
class Vbm_C():
    "Model with exponentially decaying learning rate"
    def __init__(self, \
                 lr0_day1, \
                 lr_lambda_day1, \
                 theta_Q_day1, \
                 theta_rep_day1, \
                 lr0_day2, \
                 lr_lambda_day2, \
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
        
        self.day = 1
        " ===== Latent parameters ====="
        self.theta_rep_day1 = torch.tensor([[theta_rep_day1]])
        self.theta_Q_day1 = torch.tensor([[theta_Q_day1]]) # dectemp > 0
        self.lr0_day1 = torch.tensor([[lr0_day1]])
        self.lr_lambda_day1 = torch.tensor([[lr_lambda_day1]])
        
        self.theta_rep_day2 = torch.tensor([[theta_rep_day2]])
        self.theta_Q_day2 = torch.tensor([[theta_Q_day2]]) # dectemp > 0
        self.lr0_day2 = torch.tensor([[lr0_day2]])
        self.lr_lambda_day2 = torch.tensor([[lr_lambda_day2]])
        " ===== ===== ===== ====="
        
        self.lr = [self.lr0_day1]
                
        self.k = torch.tensor([[k]])
        self.Q_init = Q_init
        self.Q = [torch.tensor([Q_init],)] # Gial-Directed Q-Values
        self.rep = [torch.tensor([[0.25, 0.25, 0.25, 0.25]],)] # habitual values (repetition values)
    
    
        "=============== Setup ==============="
        "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"
        V0 = self.theta_rep_day1*self.rep[-1][:, 0] + self.theta_Q_day2*self.Q[-1][:, 0] # V-Values for actions (i.e. weighted action values)
        V1 = self.theta_rep_day1*self.rep[-1][:, 1] + self.theta_Q_day2*self.Q[-1][:, 1]
        V2 = self.theta_rep_day1*self.rep[-1][:, 2] + self.theta_Q_day2*self.Q[-1][:, 2]
        V3 = self.theta_rep_day1*self.rep[-1][:, 3] + self.theta_Q_day2*self.Q[-1][:, 3]
        
        self.V = [torch.cat((V0.transpose(0,1), V1.transpose(0,1), V2.transpose(0,1), V3.transpose(0,1)),1)]
        
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        self.num_prev_steps = 3 # determines length of learned sequences for sequence counter
        self.seq_counter_tb = {}
        self.seq_counter_r = {}
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-10 in seq_counter for errors)"
        for i in [-10,-1,0,1,2,3]:
            for j in [-10,-1,0,1,2,3]:
                for k in [-10,-1,0,1,2,3]:
                    for l in [-10,-1,0,1,2,3]:
                        self.seq_counter_tb[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = self.k.item()/4
                        self.seq_counter_r[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = self.k.item()/4

    def softmax(self, z):
        sm = torch.nn.Softmax(dim=1)
        p_actions = sm(z)
        return p_actions
        
    def update(self, choice, outcome, blocktype, day, trialstimulus):
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

        if day == 1:
            theta_rep = self.theta_rep_day1
            theta_Q = self.theta_Q_day1
            lr_lambda = self.lr_lambda_day1
            
        elif day == 2:
            theta_rep = self.theta_rep_day2
            theta_Q = self.theta_Q_day2
            lr_lambda = self.lr_lambda_day2
        
        if choice == -1 and outcome == -1 and blocktype == -1:
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1
            self.ppchoice = -1
            self.pchoice = -1

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.tensor([[0.25, 0.25, 0.25, 0.25]],))
            self.Q.append(self.Q[-1])
            
            V0 = theta_rep*self.rep[-1][:, 0] + theta_Q*self.Q[-1][:, 0] # V-Values for actions (i.e. weighted action values)
            V1 = theta_rep*self.rep[-1][:, 1] + theta_Q*self.Q[-1][:, 1]
            V2 = theta_rep*self.rep[-1][:, 2] + theta_Q*self.Q[-1][:, 2]
            V3 = theta_rep*self.rep[-1][:, 3] + theta_Q*self.Q[-1][:, 3]
            self.V.append(torch.cat((V0,V1,V2,V3))[None, :])
            
        else:
            
            ch = choice.type('torch.LongTensor')
            outcome = outcome.type('torch.LongTensor')
            
            if self.day == 1 and day == 2:
                "First Trial of 2nd day"
                self.lr.append(self.lr0_day2)
                self.day = 2
                
            "----- Update GD-values -----"
            # Outcome is either 0 or 1
            if ch > -1:
                "No error"
                #if trialstimulus > 10:
                Qchoice = (self.Q[-1][:, ch][:,None] + self.lr[-1]*(outcome-self.Q[-1][:, ch][:,None])) * torch.eye(self.na)[ch, :]
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
            V0 = theta_rep*self.rep[-1][:, 0] + theta_Q*self.Q[-1][:, 0] # V-Values for actions (i.e. weighted action values)
            V1 = theta_rep*self.rep[-1][:, 1] + theta_Q*self.Q[-1][:, 1]
            V2 = theta_rep*self.rep[-1][:, 2] + theta_Q*self.Q[-1][:, 2]
            V3 = theta_rep*self.rep[-1][:, 3] + theta_Q*self.Q[-1][:, 3]
            
            self.V.append(torch.cat((V0.transpose(0,1), V1.transpose(0,1), V2.transpose(0,1), V3.transpose(0,1)),1))
            
            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = ch.item()
            
            "----- Update learning rate -----"
            if trialstimulus > 10:
                self.lr.append(self.lr[-1] * torch.exp(-lr_lambda))

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
            
            # self.p_actions_hist.append(p_actions)
            
            choice_sample = torch.multinomial(p_actions, 1)[0]
    
            choice_python = option2*choice_sample + option1*(1-choice_sample)

        return torch.squeeze(choice_python).type('torch.LongTensor')
    
    def reset(self, **kwargs):
        "=============== Parameters ==============="
        "--- Day 1 ---"
        self.lr0_day1 = kwargs["lr0_day1"]
        self.lr_lambda_day1 = kwargs["lr_lambda_day1"]
        self.theta_Q_day1 = kwargs["theta_Q_day1"]
        self.theta_rep_day1 = kwargs["theta_rep_day1"]

        self.lr = [self.lr0_day1]

        "--- Day 2 ---"
        self.lr0_day2 = kwargs["lr0_day2"]
        self.lr_lambda_day2 = kwargs["lr_lambda_day2"]
        self.theta_rep_day2 = kwargs["theta_rep_day2"]
        self.theta_Q_day2 = kwargs["theta_Q_day2"]

        self.lr = [self.lr0_day1]

        self.day = 1
        self.k = kwargs["k"]
        self.Q = [torch.tensor([self.Q_init],)] # Goal-Directed Q-Values
        
        "=============== Setup ==============="
        #self.p_actions_hist = [torch.tensor([[0.25, 0.25, 0.25, 0.25]],)]
        self.na = 4 # no. of possible actions
        self.rep = [torch.ones((1, self.na))*0.25] # habitual values (repetition values)
        # Compute prior over sequences of length 4
        
        V0 = self.theta_rep_day1*self.rep[-1][:, 0] + self.theta_Q_day1*self.Q[-1][:, 0] # V-Values for actions (i.e. weighted action values)
        V1 = self.theta_rep_day1*self.rep[-1][:, 1] + self.theta_Q_day1*self.Q[-1][:, 1]
        V2 = self.theta_rep_day1*self.rep[-1][:, 2] + self.theta_Q_day1*self.Q[-1][:, 2]
        V3 = self.theta_rep_day1*self.rep[-1][:, 3] + self.theta_Q_day1*self.Q[-1][:, 3]
        
        self.V = [torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1)]
        
        self.num_prev_steps = 3
        self.seq_counter_tb = {}
        self.seq_counter_r = {}
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-10 in seq_counter for errors)"
        for i in [-10,-1,0,1,2,3]:
            for j in [-10,-1,0,1,2,3]:
                for k in [-10,-1,0,1,2,3]:
                    for l in [-10,-1,0,1,2,3]:
                        self.seq_counter_tb[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = self.k.item()/4
                        self.seq_counter_r[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = self.k.item()/4
                        
class Vbm_D():
    "Fixed Q-values, but theta_rep and theta_Q develop like exponential function (update only per dualtarget trial)"
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
        self.Q = [torch.tensor([Q_init],)] # Gial-Directed Q-Values
        self.rep = [torch.tensor([[0.25, 0.25, 0.25, 0.25]],)] # habitual values (repetition values)

        "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"
        V0 = self.theta_rep[-1]*self.rep[-1][:, 0] + self.theta_Q[-1]*self.Q[-1][:, 0] # V-Values for actions (i.e. weighted action values)
        V1 = self.theta_rep[-1]*self.rep[-1][:, 1] + self.theta_Q[-1]*self.Q[-1][:, 1]
        V2 = self.theta_rep[-1]*self.rep[-1][:, 2] + self.theta_Q[-1]*self.Q[-1][:, 2]
        V3 = self.theta_rep[-1]*self.rep[-1][:, 3] + self.theta_Q[-1]*self.Q[-1][:, 3]
        
        self.V = [torch.cat((V0.transpose(0,1), V1.transpose(0,1), V2.transpose(0,1), V3.transpose(0,1)),1)]
        
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        self.num_prev_steps = 3 # determines length of learned sequences for sequence counter
        self.seq_counter_tb = {}
        self.seq_counter_r = {}
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-10 in seq_counter for errors)"
        for i in [-10,-1,0,1,2,3]:
            for j in [-10,-1,0,1,2,3]:
                for k in [-10,-1,0,1,2,3]:
                    for l in [-10,-1,0,1,2,3]:
                        self.seq_counter_tb[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = self.k.item()/4
                        self.seq_counter_r[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = self.k.item()/4

    def softmax(self, z):
        # ipdb.set_trace()
        sm = torch.nn.Softmax(dim=1)
        p_actions = sm(z)
        return p_actions
        
    def update(self, choice, outcome, blocktype, day, trialstimulus):
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
            self.Q.append(self.Q[-1])
            
            V0 = self.theta_rep[-1]*self.rep[-1][:, 0] + self.theta_Q[-1]*self.Q[-1][:, 0] # V-Values for actions (i.e. weighted action values)
            V1 = self.theta_rep[-1]*self.rep[-1][:, 1] + self.theta_Q[-1]*self.Q[-1][:, 1]
            V2 = self.theta_rep[-1]*self.rep[-1][:, 2] + self.theta_Q[-1]*self.Q[-1][:, 2]
            V3 = self.theta_rep[-1]*self.rep[-1][:, 3] + self.theta_Q[-1]*self.Q[-1][:, 3]
            self.V.append(torch.cat((V0,V1,V2,V3))[None, :])
            
        else:
            
            ch = choice.type('torch.LongTensor')
            outcome = outcome.type('torch.LongTensor')
            
            # print("self.day is")
            # print(self.day)
            
            if self.day == 1 and day == 2:
                "First Trial of 2nd day"
                self.theta_rep.append(self.theta_rep0_day2)
                self.theta_Q.append(self.theta_Q0_day2)
                self.day = 2
                # print("Switching to day 2")
                
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
            V0 = self.theta_rep[-1]*self.rep[-1][:, 0] + self.theta_Q[-1]*self.Q[-1][:, 0] # V-Values for actions (i.e. weighted action values)
            V1 = self.theta_rep[-1]*self.rep[-1][:, 1] + self.theta_Q[-1]*self.Q[-1][:, 1]
            V2 = self.theta_rep[-1]*self.rep[-1][:, 2] + self.theta_Q[-1]*self.Q[-1][:, 2]
            V3 = self.theta_rep[-1]*self.rep[-1][:, 3] + self.theta_Q[-1]*self.Q[-1][:, 3]
            
            self.V.append(torch.cat((V0.transpose(0,1), V1.transpose(0,1), V2.transpose(0,1), V3.transpose(0,1)),1))
            
            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = ch.item()
            
            "----- Update  theta_Q and theta_rep -----"
            if trialstimulus > 10:
                # if self.theta_rep[-1].shape[1] == 10:
                    # ipdb.set_trace()
                    
                if self.day == 1:
                    self.theta_rep.append(self.theta_rep[-1] * torch.exp(-self.theta_replambda_day1))
                    self.theta_Q.append(self.theta_Q[-1] * torch.exp(-self.theta_Qlambda_day1))
                    
                elif self.day == 2:
                    self.theta_rep.append(self.theta_rep[-1] * torch.exp(-self.theta_replambda_day2))
                    self.theta_Q.append(self.theta_Q[-1] * torch.exp(-self.theta_Qlambda_day2))            

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
        self.Q = [torch.tensor([self.Q_init],)] # Gial-Directed Q-Values
        

        "=============== Setup ==============="
        #self.p_actions_hist = [torch.tensor([[0.25, 0.25, 0.25, 0.25]],)]
        self.na = 4 # no. of possible actions
        self.rep = [torch.ones((1, self.na))*0.25] # habitual values (repetition values)
        # Compute prior over sequences of length 4
        
        V0 = self.theta_rep[-1]*self.rep[-1][:, 0] + self.theta_Q[-1]*self.Q[-1][:, 0] # V-Values for actions (i.e. weighted action values)
        V1 = self.theta_rep[-1]*self.rep[-1][:, 1] + self.theta_Q[-1]*self.Q[-1][:, 1]
        V2 = self.theta_rep[-1]*self.rep[-1][:, 2] + self.theta_Q[-1]*self.Q[-1][:, 2]
        V3 = self.theta_rep[-1]*self.rep[-1][:, 3] + self.theta_Q[-1]*self.Q[-1][:, 3]
        
        self.V = [torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1)]
        
        self.num_prev_steps = 3
        self.seq_counter_tb = {}
        self.seq_counter_r = {}
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-10 in seq_counter for errors)"
        for i in [-10,-1,0,1,2,3]:
            for j in [-10,-1,0,1,2,3]:
                for k in [-10,-1,0,1,2,3]:
                    for l in [-10,-1,0,1,2,3]:
                        self.seq_counter_tb[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = self.k.item()/4
                        self.seq_counter_r[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = self.k.item()/4
                        
class Vbm_D_simple():
    "Fixed Q-values, but theta_rep and theta_Q develop like exponential function (update only per dualtarget trial)"
    def __init__(self, \
                 theta_Qlambda_day1, \
                 theta_Qlambda_day2, \
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
        self.theta_rep0_day1 = torch.tensor([[1]])
        self.theta_replambda_day1 = torch.tensor([[0]])
        
        self.theta_Q0_day1 = torch.tensor([[2]])
        self.theta_Qlambda_day1 = torch.tensor([[theta_Qlambda_day1]])
        
        self.theta_rep0_day2 = torch.tensor([[1]])
        self.theta_replambda_day2 = torch.tensor([[0]])
        
        self.theta_Q0_day2 = torch.tensor([[2]])
        self.theta_Qlambda_day2 = torch.tensor([[theta_Qlambda_day2]])
        " ===== ===== ===== ====="
        
        self.theta_rep = [self.theta_rep0_day1]
        self.theta_Q = [self.theta_Q0_day1]
                
        self.k = torch.tensor([[k]])
        self.Q_init = Q_init
        self.Q = torch.tensor([Q_init],) # Goal-Directed Q-Values
        self.rep = [torch.tensor([[0.25, 0.25, 0.25, 0.25]],)] # habitual values (repetition values)

        "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"
        V0 = self.theta_rep[-1]*self.rep[-1][:, 0] + self.theta_Q[-1]*self.Q[:, 0] # V-Values for actions (i.e. weighted action values)
        V1 = self.theta_rep[-1]*self.rep[-1][:, 1] + self.theta_Q[-1]*self.Q[:, 1]
        V2 = self.theta_rep[-1]*self.rep[-1][:, 2] + self.theta_Q[-1]*self.Q[:, 2]
        V3 = self.theta_rep[-1]*self.rep[-1][:, 3] + self.theta_Q[-1]*self.Q[:, 3]
        
        self.V = [torch.cat((V0.transpose(0,1), V1.transpose(0,1), V2.transpose(0,1), V3.transpose(0,1)),1)]
        
        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        self.num_prev_steps = 3 # determines length of learned sequences for sequence counter
        self.seq_counter_tb = {}
        self.seq_counter_r = {}
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-10 in seq_counter for errors)"
        for i in [-10,-1,0,1,2,3]:
            for j in [-10,-1,0,1,2,3]:
                for k in [-10,-1,0,1,2,3]:
                    for l in [-10,-1,0,1,2,3]:
                        self.seq_counter_tb[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = self.k.item()/4
                        self.seq_counter_r[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = self.k.item()/4

    def softmax(self, z):
        # ipdb.set_trace()
        sm = torch.nn.Softmax(dim=1)
        p_actions = sm(z)
        return p_actions
        
    def update(self, choice, outcome, blocktype, day, trialstimulus):
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
            # self.Q.append(self.Q)
            
            V0 = self.theta_rep[-1]*self.rep[-1][:, 0] + self.theta_Q[-1]*self.Q[:, 0] # V-Values for actions (i.e. weighted action values)
            V1 = self.theta_rep[-1]*self.rep[-1][:, 1] + self.theta_Q[-1]*self.Q[:, 1]
            V2 = self.theta_rep[-1]*self.rep[-1][:, 2] + self.theta_Q[-1]*self.Q[:, 2]
            V3 = self.theta_rep[-1]*self.rep[-1][:, 3] + self.theta_Q[-1]*self.Q[:, 3]
            self.V.append(torch.cat((V0,V1,V2,V3))[None, :])
            
        else:
            
            
            ch = choice.type('torch.LongTensor')
            outcome = outcome.type('torch.LongTensor')
            
            # print("self.day is")
            # print(self.day)
            
            if self.day == 1 and day == 2:
                "First Trial of 2nd day"
                self.theta_rep.append(self.theta_rep0_day2)
                self.theta_Q.append(self.theta_Q0_day2)
                self.day = 2
                # print("Switching to day 2")
                
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
            V0 = self.theta_rep[-1]*self.rep[-1][:, 0] + self.theta_Q[-1]*self.Q[:, 0] # V-Values for actions (i.e. weighted action values)
            V1 = self.theta_rep[-1]*self.rep[-1][:, 1] + self.theta_Q[-1]*self.Q[:, 1]
            V2 = self.theta_rep[-1]*self.rep[-1][:, 2] + self.theta_Q[-1]*self.Q[:, 2]
            V3 = self.theta_rep[-1]*self.rep[-1][:, 3] + self.theta_Q[-1]*self.Q[:, 3]
            
            self.V.append(torch.cat((V0.transpose(0,1), V1.transpose(0,1), V2.transpose(0,1), V3.transpose(0,1)),1))
            
            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = ch.item()
            
            "----- Update  theta_Q and theta_rep -----"
            if trialstimulus > 10:
                if self.day == 1:
                  self.theta_rep.append((self.theta_rep[-1] * torch.exp(-self.theta_replambda_day1)))
                  self.theta_Q.append((self.theta_Q[-1] * torch.exp(-self.theta_Qlambda_day1)))
                    
                elif self.day == 2:
                    self.theta_rep.append((self.theta_rep[-1] * torch.exp(-self.theta_replambda_day2)))
                    self.theta_Q.append((self.theta_Q[-1] * torch.exp(-self.theta_Qlambda_day2)))            

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
            
            # self.p_actions_hist.append(p_actions)
            
            choice_sample = torch.multinomial(p_actions, 1)[0]
    
            choice_python = option2*choice_sample + option1*(1-choice_sample)

        return torch.squeeze(choice_python).type('torch.LongTensor')
    
    def reset(self, **kwargs):
        
        # self.theta_rep0_day1 = torch.tensor([[1.]])
        # self.theta_replambda_day1 = torch.tensor([[0.]])
        
        # self.theta_Q0_day1 = torch.tensor([[2.]])
        self.theta_Qlambda_day1 = kwargs["theta_Qlambda_day1"]
        
        # self.theta_rep0_day2 = torch.tensor([[1.]])
        # self.theta_replambda_day2 = torch.tensor([[0.]])
        
        # self.theta_Q0_day2 = torch.tensor([[2.]])
        self.theta_Qlambda_day2 = kwargs["theta_Qlambda_day2"]
        
        self.theta_rep = [self.theta_rep0_day1]
        self.theta_Q = [self.theta_Q0_day1]
        
        self.day = 1
        self.k = kwargs["k"]
        self.Q = torch.tensor([self.Q_init],) # Gial-Directed Q-Values
        
        #self.p_actions_hist = [torch.tensor([[0.25, 0.25, 0.25, 0.25]],)]
        self.na = 4 # no. of possible actions
        self.rep = [torch.ones((1, self.na))*0.25] # habitual values (repetition values)
        # Compute prior over sequences of length 4
        
        V0 = self.theta_rep[-1]*self.rep[-1][:, 0] + self.theta_Q[-1]*self.Q[:, 0] # V-Values for actions (i.e. weighted action values)
        V1 = self.theta_rep[-1]*self.rep[-1][:, 1] + self.theta_Q[-1]*self.Q[:, 1]
        V2 = self.theta_rep[-1]*self.rep[-1][:, 2] + self.theta_Q[-1]*self.Q[:, 2]
        V3 = self.theta_rep[-1]*self.rep[-1][:, 3] + self.theta_Q[-1]*self.Q[:, 3]
        
        self.V = [torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1)]
        
        self.num_prev_steps = 3
        self.seq_counter_tb = {}
        self.seq_counter_r = {}
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-10 in seq_counter for errors)"
        for i in [-10,-1,0,1,2,3]:
            for j in [-10,-1,0,1,2,3]:
                for k in [-10,-1,0,1,2,3]:
                    for l in [-10,-1,0,1,2,3]:
                        self.seq_counter_tb[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = self.k.item()/4
                        self.seq_counter_r[str(i) + "," + str(j) + "," + str(k) + "," + str(l)] = self.k.item()/4
                        
class SingleInference_modelD_simple(object):

    def __init__(self, agent, data):
        self.agent = agent
        self.trials = self.agent.trials # length of experiment
        #self.T = agent.T # No. of "subtrials" -> unnötig
        self.data = data # Dictionary

    def model(self):
        # generative model of behavior with Normally distributed params (within subject!!)

        # tell pyro about prior over parameters: alpha and beta of lr which is between 0 and 1
        # alpha = beta = 1 equals uniform prior
        "--- Day 1 ---"
        "Θ_Qlambda: Beta Distribution"
        theta_Qlambda_day1_alpha = torch.tensor([1.]).to(device)
        theta_Qlambda_day1_beta = torch.tensor([10.]).to(device)
        theta_Qlambda_day1 = pyro.sample('theta_Qlambda_day1', dist.Beta(theta_Qlambda_day1_alpha, theta_Qlambda_day1_beta)).to(device)
        
        "Θ_Q0"
        # theta_Q0_day1_conc = torch.tensor([4.]).to(device)
        # theta_Q0_day1_rate = torch.tensor([4.]).to(device)
        # theta_Q0_day1 = pyro.sample('theta_Q0_day1', dist.Gamma(theta_Q0_day1_conc, theta_Q0_day1_rate)).to(device)
        
        # "Θ_Qlambda: Beta Distribution"
        # theta_replambda_day1_alpha = torch.tensor([1.]).to(device)
        # theta_replambda_day1_beta = torch.tensor([1.]).to(device)
        # theta_replambda_day1 = pyro.sample('theta_replambda_day1', dist.Beta(theta_replambda_day1_alpha, theta_replambda_day1_beta)).to(device)
        
        # "Θ_rep0"
        # theta_rep0_day1_conc = torch.tensor([4.]).to(device)
        # theta_rep0_day1_rate = torch.tensor([4.]).to(device)
        # theta_rep0_day1 = pyro.sample('theta_rep0_day1', dist.Gamma(theta_rep0_day1_conc, theta_rep0_day1_rate)).to(device)
        
        "--- Day 2 ---"
        "Θ_Qlambda: Beta Distribution"
        theta_Qlambda_day2_alpha = torch.tensor([1.]).to(device)
        theta_Qlambda_day2_beta = torch.tensor([10.]).to(device)
        theta_Qlambda_day2 = pyro.sample('theta_Qlambda_day2', dist.Beta(theta_Qlambda_day2_alpha, theta_Qlambda_day2_beta)).to(device)
        
        "Θ_Q0"
        # theta_Q0_day2_conc = torch.tensor([4.]).to(device)
        # theta_Q0_day2_rate = torch.tensor([4.]).to(device)
        # theta_Q0_day2 = pyro.sample('theta_Q0_day2', dist.Gamma(theta_Q0_day2_conc, theta_Q0_day2_rate)).to(device)
        
        # "Θ_Qlambda: Beta Distribution"
        # theta_replambda_day2_alpha = torch.tensor([1.]).to(device)
        # theta_replambda_day2_beta = torch.tensor([1.]).to(device)
        # theta_replambda_day2 = pyro.sample('theta_replambda_day2', dist.Beta(theta_replambda_day2_alpha, theta_replambda_day2_beta)).to(device)
        
        # "Θ_rep0"
        # theta_rep0_day2_conc = torch.tensor([4.]).to(device)
        # theta_rep0_day2_rate = torch.tensor([4.]).to(device)
        # theta_rep0_day2 = pyro.sample('theta_rep0_day2', dist.Gamma(theta_rep0_day2_conc, theta_rep0_day2_rate)).to(device)

        "K: Gamma distribution"
        #conc_k = torch.tensor([2]).to(device)        
        #rate_k = torch.tensor([4]).to(device)        
        #k = pyro.sample('k', dist.Gamma(conc_k, rate_k)).to(device)
        k=torch.tensor([4.])
        
        # parameter shape for inference (until now): [1] or [n_particles]
        param_dict = {"theta_Qlambda_day1": theta_Qlambda_day1[None,:], \
                      # "theta_Q0_day1": theta_Q0_day1[:, None], \
                      # "theta_replambda_day1": theta_replambda_day1[None, :], \
                      # "theta_rep0_day1": theta_rep0_day1[None, :], \
                      "theta_Qlambda_day2": theta_Qlambda_day2[None,:], \
                      # "theta_Q0_day2": theta_Q0_day2[:, None], \
                      # "theta_replambda_day2": theta_replambda_day2[None, :], \
                      # "theta_rep0_day2": theta_rep0_day2[None, :], \
                      "k": k[:, None]}

        
        # lr is shape [1,1] or [n_particles, 1]
        self.agent.reset(**param_dict)
        
        t = -1
        for tau in range(self.trials):
            
            trial = self.data["Trialsequence"][tau]
            blocktype = self.data["Blocktype"][tau]
            
            if self.data["Blockidx"][tau] <= 5:
                day = 1
                
            elif self.data["Blockidx"][tau] > 5:
                day = 2
                
            else:
                raise Exception("Da isch a Fehla!")
            
            if trial == -1:
                "Beginning of new block"
                self.agent.update(-1,-1,-1, day, trial)
                
            else:
                current_choice = self.data["Choices"][tau]
                outcome = self.data["Outcomes"][tau]

            if trial > 10:
                "Dual-Target Trial"
                t+=1
                option1, option2 = self.agent.find_resp_options(trial)
                
                # Comes out as [1, n_actions] or [n_particles, n_actions]
                probs = self.agent.softmax(torch.cat(([self.agent.V[-1][:, option1][:, None], self.agent.V[-1][:, option2][:, None]]), dim=-1))
                
                if current_choice == option1:
                    choice = torch.tensor([0])
                    
                elif current_choice == option2:
                    choice = torch.tensor([1])
                    
                elif current_choice == -10:
                    "Error"
                    pass
                    
                else:
                    raise Exception("Da isch a Fehla")

            if trial != -1:
                "Update (trial == -1 means this is the beginning of a block -> participants didn't see this trial')"
                self.agent.update(current_choice, outcome, blocktype, day, trial)

            "Sample if dual-target trial and no error was performed"
            if trial > 10 and current_choice > -1:
                # choice shape for inference: [1]
                # print("About to sample trial %d, baby!"%tau)
                pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), obs=choice)
                    
    def guide(self):
        # approximate posterior. assume MF: each param has its own univariate Normal.

        "--- Day 1 ---"
        "Θ_Qlambda"
        theta_Qlambda_day1_alpha = pyro.param("theta_Qlambda_day1_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Qlambda_day1_beta = pyro.param("theta_Qlambda_day1_beta", torch.tensor([10.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Qlambda_day1 = pyro.sample('theta_Qlambda_day1', dist.Beta(theta_Qlambda_day1_alpha, theta_Qlambda_day1_beta)).to(device)

        "Θ_Q0: Gamma Distribution"
        # theta_Q0_day1_conc = pyro.param("theta_Q0_day1_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # theta_Q0_day1_rate = pyro.param("theta_Q0_day1_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # theta_Q0_day1 = pyro.sample('theta_Q0_day1', dist.Gamma(theta_Q0_day1_conc, theta_Q0_day1_rate)).to(device)
        
        # "Θ_replambda"
        # theta_replambda_day1_alpha = pyro.param("theta_replambda_day1_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # theta_replambda_day1_beta = pyro.param("theta_replambda_day1_beta", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # theta_replambda_day1 = pyro.sample('theta_replambda_day1', dist.Beta(theta_replambda_day1_alpha, theta_replambda_day1_beta)).to(device)

        # "Θ_rep0: Gamma Distribution"
        # theta_rep0_day1_conc = pyro.param("theta_rep0_day1_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # theta_rep0_day1_rate = pyro.param("theta_rep0_day1_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # theta_rep0_day1 = pyro.sample('theta_rep0_day1', dist.Gamma(theta_rep0_day1_conc, theta_rep0_day1_rate)).to(device)

        " --- Day 2 ---"
        "Θ_Qlambda"
        theta_Qlambda_day2_alpha = pyro.param("theta_Qlambda_day2_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Qlambda_day2_beta = pyro.param("theta_Qlambda_day2_beta", torch.tensor([10.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Qlambda_day2 = pyro.sample('theta_Qlambda_day2', dist.Beta(theta_Qlambda_day2_alpha, theta_Qlambda_day2_beta)).to(device)

        "Θ_Q0: Gamma Distribution"
        # theta_Q0_day2_conc = pyro.param("theta_Q0_day2_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # theta_Q0_day2_rate = pyro.param("theta_Q0_day2_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # theta_Q0_day2 = pyro.sample('theta_Q0_day2', dist.Gamma(theta_Q0_day2_conc, theta_Q0_day2_rate)).to(device)
        
        # "Θ_replambda"
        # theta_replambda_day2_alpha = pyro.param("theta_replambda_day2_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # theta_replambda_day2_beta = pyro.param("theta_replambda_day2_beta", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # theta_replambda_day2 = pyro.sample('theta_replambda_day2', dist.Beta(theta_replambda_day2_alpha, theta_replambda_day2_beta)).to(device)

        # "Θ_rep0: Gamma Distribution"
        # theta_rep0_day2_conc = pyro.param("theta_rep0_day2_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # theta_rep0_day2_rate = pyro.param("theta_rep0_day2_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # theta_rep0_day2 = pyro.sample('theta_rep0_day2', dist.Gamma(theta_rep0_day2_conc, theta_rep0_day2_rate)).to(device)
        
        "K: Gamma Distribution"
        #conc_k = pyro.param("conc_k", torch.tensor([2]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        #rate_k = pyro.param("rate_k", torch.tensor([4]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        #k = pyro.sample('k', dist.Gamma(conc_k, rate_k)).to(device)

        param_dict = {"theta_Qlambda_day1_alpha": theta_Qlambda_day1_alpha, \
                      "theta_Qlambda_day1_beta": theta_Qlambda_day1_beta, \
                      "theta_Qlambda_day1": theta_Qlambda_day1, \
                      # "theta_Q0_day1_conc": theta_Q0_day1_conc, \
                      # "theta_Q0_day1_rate": theta_Q0_day1_rate, \
                      # "theta_Q0_day1": theta_Q0_day1, \
                      # "theta_replambda_day1_alpha": theta_replambda_day1_alpha, \
                      # "theta_replambda_day1_beta": theta_replambda_day1_beta, \
                      # "theta_replambda_day1": theta_replambda_day1, \
                      # "theta_rep0_day1_conc": theta_rep0_day1_conc, \
                      # "theta_rep0_day1_rate": theta_rep0_day1_rate, \
                      # "theta_rep0_day1": theta_rep0_day1, \
                      "theta_Qlambda_day2_alpha": theta_Qlambda_day2_alpha, \
              "theta_Qlambda_day2_beta": theta_Qlambda_day2_beta, \
              "theta_Qlambda_day2": theta_Qlambda_day2}
              # "theta_Q0_day2_conc": theta_Q0_day2_conc, \
              # "theta_Q0_day2_rate": theta_Q0_day2_rate, \
              # "theta_Q0_day2": theta_Q0_day2, \
              # "theta_replambda_day2_alpha": theta_replambda_day2_alpha, \
              # "theta_replambda_day2_beta": theta_replambda_day2_beta, \
              # "theta_replambda_day2": theta_replambda_day2, \
              # "theta_rep0_day2_conc": theta_rep0_day2_conc, \
              # "theta_rep0_day2_rate": theta_rep0_day2_rate, \
              # "theta_rep0_day2": theta_rep0_day2,}
            
        return param_dict

    def infer_posterior(self,
                        iter_steps=1_000,
                        num_particles=10,
                        optim_kwargs={'lr': .01}): # Adam learning rate
        """Perform SVI over free model parameters."""

        pyro.clear_param_store()

        svi = pyro.infer.SVI(model=self.model,
                  guide=self.guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  # set below to true once code is vectorized
                                  vectorize_particles=False))
                
        "Next : run self.model() once per iter_step (is tun in line loss.append(torch.tensor(svi.step()).to(device)))"
        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:#range(iter_steps):
            loss.append(torch.tensor(svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
            if torch.isnan(loss[-1]):
                break

        self.loss = [l.cpu() for l in loss] # = -ELBO (Plotten!)
        
        "--- Day 1 ---"
        theta_Qlambda_day1_alpha = pyro.param("theta_Qlambda_day1_alpha").data.numpy()
        theta_Qlambda_day1_beta = pyro.param("theta_Qlambda_day1_beta").data.numpy()

        # theta_Q0_day1_conc = pyro.param("theta_Q0_day1_conc").data.numpy()
        # theta_Q0_day1_rate = pyro.param("theta_Q0_day1_rate").data.numpy()
        
        # theta_replambda_day1_alpha = pyro.param("theta_replambda_day1_alpha").data.numpy()
        # theta_replambda_day1_beta = pyro.param("theta_replambda_day1_beta").data.numpy()

        # theta_rep0_day1_conc = pyro.param("theta_rep0_day1_conc").data.numpy()
        # theta_rep0_day1_rate = pyro.param("theta_rep0_day1_rate").data.numpy()

        "--- Day 2 ---"
        theta_Qlambda_day2_alpha = pyro.param("theta_Qlambda_day2_alpha").data.numpy()
        theta_Qlambda_day2_beta = pyro.param("theta_Qlambda_day2_beta").data.numpy()

        # theta_Q0_day2_conc = pyro.param("theta_Q0_day2_conc").data.numpy()
        # theta_Q0_day2_rate = pyro.param("theta_Q0_day2_rate").data.numpy()
        
        # theta_replambda_day2_alpha = pyro.param("theta_replambda_day2_alpha").data.numpy()
        # theta_replambda_day2_beta = pyro.param("theta_replambda_day2_beta").data.numpy()

        # theta_rep0_day2_conc = pyro.param("theta_rep0_day2_conc").data.numpy()
        # theta_rep0_day2_rate = pyro.param("theta_rep0_day2_rate").data.numpy()

        param_dict = {"theta_Qlambda_day1_alpha": theta_Qlambda_day1_alpha, \
                      "theta_Qlambda_day1_beta": theta_Qlambda_day1_beta, \
                      # "theta_Q0_day1_conc": theta_Q0_day1_conc, \
                      # "theta_Q0_day1_rate": theta_Q0_day1_rate, \
                      # "theta_replambda_day1_alpha": theta_replambda_day1_alpha, \
                      # "theta_replambda_day1_beta": theta_replambda_day1_beta, \
                      # "theta_rep0_day1_conc": theta_rep0_day1_conc, \
                      # "theta_rep0_day1_rate": theta_rep0_day1_rate, \
                      "theta_Qlambda_day2_alpha": theta_Qlambda_day2_alpha, \
                      "theta_Qlambda_day2_beta": theta_Qlambda_day2_beta}
                      # "theta_Q0_day2_conc": theta_Q0_day2_conc, \
                      # "theta_Q0_day2_rate": theta_Q0_day2_rate, \
                      # "theta_replambda_day2_alpha": theta_replambda_day2_alpha, \
                      # "theta_replambda_day2_beta": theta_replambda_day2_beta, \
                      # "theta_rep0_day2_conc": theta_rep0_day2_conc, \
                      # "theta_rep0_day2_rate": theta_rep0_day2_rate}

        return self.loss, param_dict
                        
class Vbm_A_Bayesian(Vbm_A):
    
    def __init__(self, dectemp_day1, lr_day1, omega_day1, dectemp_day2, lr_day2, omega_day2, k, Q_init, num_blocks = 14):
        super().__init__(dectemp_day1, lr_day1, omega_day1, dectemp_day2, lr_day2, omega_day2, k, Q_init, num_blocks = 14)
    
        "V(ai) = ω*rep_val(ai) * exp(β*Q(ai))"
        V0 = self.omega_day1*self.rep[-1][:, 0] *torch.exp(self.dectemp_day1*self.Q[-1][:, 0]) # V-Values for actions (i.e. weighted action values)
        V1 = self.omega_day1*self.rep[-1][:, 1] *torch.exp(self.dectemp_day1*self.Q[-1][:, 1])
        V2 = self.omega_day1*self.rep[-1][:, 2] *torch.exp(self.dectemp_day1*self.Q[-1][:, 2])
        V3 = self.omega_day1*self.rep[-1][:, 3] *torch.exp(self.dectemp_day1*self.Q[-1][:, 3])
        
        self.V = [torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1)]
    
    def update(self, choice, outcome, blocktype, day, trialstimulus):
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
            omega = self.omega_day1
            dectemp = self.dectemp_day1
            
        elif day == 2:
            lr = self.lr_day2
            omega = self.omega_day2
            dectemp = self.dectemp_day2
        
        if choice == -1 and outcome == -1 and blocktype == -1:
            "Set previous actions to -1 because it's the beginning of a new block"
            self.pppchoice = -1
            self.ppchoice = -1
            self.pchoice = -1

            "Set repetition values to 0 because of new block"
            self.rep.append(torch.tensor([[0.25, 0.25, 0.25, 0.25]],))
            self.Q.append(self.Q[-1])
            
            V0 = omega*self.rep[-1][:, 0] *torch.exp(dectemp*self.Q[-1][:, 0]) # V-Values for actions (i.e. weighted action values)
            V1 = omega*self.rep[-1][:, 1] *torch.exp(dectemp*self.Q[-1][:, 1])
            V2 = omega*self.rep[-1][:, 2] *torch.exp(dectemp*self.Q[-1][:, 2])
            V3 = omega*self.rep[-1][:, 3] *torch.exp(dectemp*self.Q[-1][:, 3])
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
                Qchoice = (self.Q[-1][:, ch][:,None] + lr*(outcome-self.Q[-1][:, ch][:,None])) * torch.eye(self.na)[ch, :]
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
            V0 = omega*self.rep[-1][:, 0] *torch.exp(dectemp*self.Q[-1][:, 0]) # V-Values for actions (i.e. weighted action values)
            V1 = omega*self.rep[-1][:, 1] *torch.exp(dectemp*self.Q[-1][:, 1])
            V2 = omega*self.rep[-1][:, 2] *torch.exp(dectemp*self.Q[-1][:, 2])
            V3 = omega*self.rep[-1][:, 3] *torch.exp(dectemp*self.Q[-1][:, 3])
            
            self.V.append(torch.cat((V0.transpose(0,1), V1.transpose(0,1), V2.transpose(0,1), V3.transpose(0,1)),1))
            
            "----- Update action memory -----"
            # pchoice stands for "previous choice"
            self.pppchoice = self.ppchoice
            self.ppchoice = self.pchoice
            self.pchoice = ch.item()
            
    def choose_action(self, trial, day):
        "INPUT: trial (in 1-indexing (i.e. MATLAB notation))"
        "OUTPUT: choice response digit (in 0-indexing notation)"
        
        if trial < 10:
            "Single-target trial"
            choice_python = trial-1
        
        elif trial > 10:
            "Dual-target trial"
            option1, option2 = self.find_resp_options(trial)
            
            pV1 = self.V[-1][:, option1] / (self.V[-1][:, option1] + self.V[-1][:, option2])
                        
            choice_sample = torch.bernoulli(pV1)[0]
    
            choice_python = option1*choice_sample + option2*(1-choice_sample)

        return torch.squeeze(choice_python).type('torch.LongTensor')
            
    def reset(self, **kwargs):
        super().reset(**kwargs)
        
        self.day = 1
        
        "----- Compute new V-values for next trial -----"
        V0 = self.omega_day1*self.rep[-1][:, 0] *torch.exp(self.dectemp_day1*self.Q[-1][:, 0]) # V-Values for actions (i.e. weighted action values)
        V1 = self.omega_day1*self.rep[-1][:, 1] *torch.exp(self.dectemp_day1*self.Q[-1][:, 1])
        V2 = self.omega_day1*self.rep[-1][:, 2] *torch.exp(self.dectemp_day1*self.Q[-1][:, 2])
        V3 = self.omega_day1*self.rep[-1][:, 3] *torch.exp(self.dectemp_day1*self.Q[-1][:, 3])
        
        self.V = [torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1)]
        

class SingleInference_modelA_Bayesian(object):

    def __init__(self, agent, data):

        self.agent = agent
        self.trials = self.agent.trials # length of experiment
        #self.T = agent.T # No. of "subtrials" -> unnötig
        self.data = data # Dictionary

    def model(self):
        # generative model of behavior with Normally distributed params (within subject!!)

        # tell pyro about prior over parameters: alpha and beta of lr which is between 0 and 1
        # alpha = beta = 1 equals uniform prior
        "Learning Rate: Beta Distribution"
        lr_day1_alpha = torch.tensor([1.]).to(device)
        lr_day1_beta = torch.tensor([1.]).to(device)
        lr_day1 = pyro.sample('lr_day1', dist.Beta(lr_day1_alpha, lr_day1_beta)).to(device)
        
        lr_day2_alpha = torch.tensor([1.]).to(device)
        lr_day2_beta = torch.tensor([1.]).to(device)
        lr_day2 = pyro.sample('lr_day2', dist.Beta(lr_day2_alpha, lr_day2_beta)).to(device)
                
        "Omega: Beta Distribution"
        omega_day1_conc = torch.tensor([4.]).to(device)
        omega_day1_rate = torch.tensor([4.]).to(device)
        omega_day1 = pyro.sample('omega_day1', dist.Gamma(omega_day1_conc, omega_day1_rate)).to(device)
        
        omega_day2_conc = torch.tensor([4.]).to(device)
        omega_day2_rate = torch.tensor([4.]).to(device)      
        omega_day2 = pyro.sample('omega_day2', dist.Gamma(omega_day2_conc, omega_day2_rate)).to(device)
        
        "Decision Temperature : Gamma distribution"
        dectemp_day1_conc = torch.tensor([4.]).to(device)
        dectemp_day1_rate = torch.tensor([4.]).to(device)
        dectemp_day1 = pyro.sample('dectemp_day1', dist.Gamma(dectemp_day1_conc, dectemp_day1_rate)).to(device)
        
        dectemp_day2_conc = torch.tensor([4.]).to(device)
        dectemp_day2_rate = torch.tensor([4.]).to(device)
        dectemp_day2 = pyro.sample('dectemp_day2', dist.Gamma(dectemp_day2_conc, dectemp_day2_rate)).to(device)
        
        "K: Gamma distribution"
        #conc_k = torch.tensor([2]).to(device)        
        #rate_k = torch.tensor([4]).to(device)        
        #k = pyro.sample('k', dist.Gamma(conc_k, rate_k)).to(device)
        k=torch.tensor([4.])
                
        # parameter shape for inference (until now): [1] or [n_particles]
        param_dict = {"lr_day1": lr_day1[:,None], \
                      "lr_day2": lr_day2[:,None], \
                      "omega_day1": omega_day1[None, :], \
                      "omega_day2": omega_day2[None, :], \
                      "dectemp_day1": dectemp_day1[None, :], \
                      "dectemp_day2": dectemp_day2[None, :], \
                      "k": k[:, None]}

        # lr is shape [1,1] or [n_particles, 1]
        self.agent.reset(**param_dict)
        
        t = -1
        for tau in range(self.trials):
            
            trial = self.data["Trialsequence"][tau]
            blocktype = self.data["Blocktype"][tau]
            
            if self.data["Blockidx"][tau] <= 5:
                day = 1
                
            elif self.data["Blockidx"][tau] > 5:
                day = 2
                
            else:
                raise Exception("Da isch a Fehla!")
            
            if trial == -1:
                "Beginning of new block"
                self.agent.update(-1,-1,-1, day, trial)
                
            else:
                current_choice = self.data["Choices"][tau]
                outcome = self.data["Outcomes"][tau]
                
            if trial > 10:
                "Dual-Target Trial"
                t+=1
                option1, option2 = self.agent.find_resp_options(trial)
                
                # Comes out as [1, n_actions] or [n_particles, n_actions]
                "probs for option 2"
                pV2 = self.agent.V[-1][:, option2] / (self.agent.V[-1][:, option1] + self.agent.V[-1][:, option2])
                
                pV2 = pV2[:, None]
                probs = torch.cat((1-pV2, pV2), dim = 1)
                    
                if current_choice == option1:
                    choice = torch.tensor([0])

                elif current_choice == option2:
                    choice = torch.tensor([1])
                    
                elif current_choice == -10:
                    "Error"
                    pass
                    
                else:
                    raise Exception("Da isch a Fehla")

            if trial != -1:
                "Update (trial == -1 means this is the beginning of a block -> participants didn't see this trial')"
                self.agent.update(current_choice, outcome, blocktype, day, trial)

            "Sample if dual-target trial and no error was performed"
            if trial > 10 and current_choice > -1:
                # choice shape for inference: [1]
                pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), obs=choice)
                #pyro.sample('res_{}'.format(t), dist.Bernoulli(probs=probs), obs=choice)

                    
    def guide(self):
        # approximate posterior. assume MF: each param has its own univariate Normal.

        "Learning Rate"
        lr_day1_alpha = pyro.param("lr_day1_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day1_beta = pyro.param("lr_day1_beta", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day1 = pyro.sample('lr_day1', dist.Beta(lr_day1_alpha, lr_day1_beta)).to(device)

        lr_day2_alpha = pyro.param("lr_day2_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day2_beta = pyro.param("lr_day2_beta", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day2 = pyro.sample('lr_day2', dist.Beta(lr_day2_alpha, lr_day2_beta)).to(device)

        "Omega: Beta Distribution"
        omega_day1_conc = pyro.param("omega_day1_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        omega_day1_rate = pyro.param("omega_day1_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        omega_day1 = pyro.sample('omega_day1', dist.Gamma(omega_day1_conc, omega_day1_rate)).to(device)

        omega_day2_conc = pyro.param("omega_day2_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        omega_day2_rate = pyro.param("omega_day2_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        omega_day2 = pyro.sample('omega_day2', dist.Gamma(omega_day2_conc, omega_day2_rate)).to(device)

        "Decision Temperature: Gamma Distribution"
        dectemp_day1_conc = pyro.param("dectemp_day1_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        dectemp_day1_rate = pyro.param("dectemp_day1_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        dectemp_day1 = pyro.sample('dectemp_day1', dist.Gamma(dectemp_day1_conc, dectemp_day1_rate)).to(device)
        
        dectemp_day2_conc = pyro.param("dectemp_day2_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        dectemp_day2_rate = pyro.param("dectemp_day2_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        dectemp_day2 = pyro.sample('dectemp_day2', dist.Gamma(dectemp_day2_conc, dectemp_day2_rate)).to(device)
        
        "K: Gamma Distribution"
        #conc_k = pyro.param("conc_k", torch.tensor([2]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        #rate_k = pyro.param("rate_k", torch.tensor([4]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        #k = pyro.sample('k', dist.Gamma(conc_k, rate_k)).to(device)

        # print("lr")
        # print(lr)
        
        param_dict = {"lr_day1_alpha": lr_day1_alpha, \
                      "lr_day1_beta": lr_day1_beta, \
                      "lr_day1": lr_day1, \
                      "omega_day1_conc": omega_day1_conc, \
                      "omega_day1_rate": omega_day1_rate, \
                      "omega_day1": omega_day1, \
                      "dectemp_day1_conc": dectemp_day1_conc, \
                      "dectemp_day1_rate": dectemp_day1_rate, \
                      "dectemp_day1": dectemp_day1, \
                      "lr_day2_alpha": lr_day2_alpha, \
                      "lr_day2_beta": lr_day2_beta, \
                      "lr_day2": lr_day2, \
                      "omega_day2_conc": omega_day2_conc, \
                      "omega_day2_rate": omega_day2_rate, \
                      "omega_day2": omega_day2, \
                      "dectemp_day2_conc": dectemp_day2_conc, \
                      "dectemp_day2_rate": dectemp_day2_rate, \
                      "dectemp_day2": dectemp_day2}

        return param_dict

    def infer_posterior(self,
                        iter_steps=1_000,
                        num_particles=10,
                        optim_kwargs={'lr': .01}): # Adam learning rate
        """Perform SVI over free model parameters."""

        pyro.clear_param_store()

        svi = pyro.infer.SVI(model=self.model,
                  guide=self.guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  # set below to true once code is vectorized
                                  vectorize_particles=True))

        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:#range(iter_steps):
            # ipdb.set_trace()
            loss.append(torch.tensor(svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
            if torch.isnan(loss[-1]):
                break

        self.loss = [l.cpu() for l in loss] # = -ELBO (Plotten!)
        
        lr_day1_alpha = pyro.param("lr_day1_alpha").data.numpy()
        lr_day1_beta = pyro.param("lr_day1_beta").data.numpy()
        
        lr_day2_alpha = pyro.param("lr_day2_alpha").data.numpy()
        lr_day2_beta = pyro.param("lr_day2_beta").data.numpy()
        
        omega_day1_conc = pyro.param("omega_day1_conc").data.numpy()
        omega_day1_rate = pyro.param("omega_day1_rate").data.numpy()
        
        omega_day2_conc = pyro.param("omega_day2_conc").data.numpy()
        omega_day2_rate = pyro.param("omega_day2_rate").data.numpy()
        
        dectemp_day1_conc = pyro.param("dectemp_day1_conc").data.numpy()
        dectemp_day1_rate = pyro.param("dectemp_day1_rate").data.numpy()
        
        dectemp_day2_conc = pyro.param("dectemp_day2_conc").data.numpy()
        dectemp_day2_rate = pyro.param("dectemp_day2_rate").data.numpy()

        param_dict = {"lr_day1_alpha": lr_day1_alpha, \
                      "lr_day1_beta": lr_day1_beta, \
                      "lr_day2_alpha": lr_day2_alpha, \
                      "lr_day2_beta": lr_day2_beta, \
                      "omega_day1_conc": omega_day1_conc, \
                      "omega_day1_rate": omega_day1_rate, \
                      "omega_day2_conc": omega_day2_conc, \
                      "omega_day2_rate": omega_day2_rate, \
                      "dectemp_day1_conc": dectemp_day1_conc, \
                      "dectemp_day1_rate": dectemp_day1_rate, \
                      "dectemp_day2_conc": dectemp_day2_conc, \
                      "dectemp_day2_rate": dectemp_day2_rate}

        return self.loss, param_dict
    
class SingleInference_modelC(object):

    def __init__(self, agent, data):

        self.agent = agent
        self.trials = self.agent.trials # length of experiment
        #self.T = agent.T # No. of "subtrials" -> unnötig
        self.data = data # Dictionary

    def model(self):
        # generative model of behavior with Normally distributed params (within subject!!)

        # tell pyro about prior over parameters: alpha and beta of lr which is between 0 and 1
        # alpha = beta = 1 equals uniform prior
        "--- Day 1 ---"
        "lr0: Beta Distribution"
        lr0_day1_alpha = torch.tensor([1.]).to(device)
        lr0_day1_beta = torch.tensor([1.]).to(device)
        lr0_day1 = pyro.sample('lr0_day1', dist.Beta(lr0_day1_alpha, lr0_day1_beta)).to(device)
        
        "lr_lambda: Beta Distribution"
        lr_lambda_day1_alpha = torch.tensor([1.]).to(device)
        lr_lambda_day1_beta = torch.tensor([10.]).to(device)
        lr_lambda_day1 = pyro.sample('lr_lambda_day1', dist.Beta(lr_lambda_day1_alpha, lr_lambda_day1_beta)).to(device)
        
        "Θ_Q"
        theta_Q_day1_conc = torch.tensor([4.]).to(device)
        theta_Q_day1_rate = torch.tensor([4.]).to(device)
        theta_Q_day1 = pyro.sample('theta_Q_day1', dist.Gamma(theta_Q_day1_conc, theta_Q_day1_rate)).to(device)
        
        "Θ_rep"
        theta_rep_day1_conc = torch.tensor([4.]).to(device)
        theta_rep_day1_rate = torch.tensor([4.]).to(device)
        theta_rep_day1 = pyro.sample('theta_rep_day1', dist.Gamma(theta_rep_day1_conc, theta_rep_day1_rate)).to(device)
        
        "--- Day 2 ---"
        "lr0: Beta Distribution"
        lr0_day2_alpha = torch.tensor([1.]).to(device)
        lr0_day2_beta = torch.tensor([1.]).to(device)
        lr0_day2 = pyro.sample('lr0_day2', dist.Beta(lr0_day2_alpha, lr0_day2_beta)).to(device)
        
        "lr_lambda: Beta Distribution"
        lr_lambda_day2_alpha = torch.tensor([1.]).to(device)
        lr_lambda_day2_beta = torch.tensor([10.]).to(device)
        lr_lambda_day2 = pyro.sample('lr_lambda_day2', dist.Beta(lr_lambda_day2_alpha, lr_lambda_day2_beta)).to(device)
        
        "Θ_Q"
        theta_Q_day2_conc = torch.tensor([4.]).to(device)
        theta_Q_day2_rate = torch.tensor([4.]).to(device)
        theta_Q_day2 = pyro.sample('theta_Q_day2', dist.Gamma(theta_Q_day2_conc, theta_Q_day2_rate)).to(device)
        
        "Θ_rep"
        theta_rep_day2_conc = torch.tensor([4.]).to(device)
        theta_rep_day2_rate = torch.tensor([4.]).to(device)
        theta_rep_day2 = pyro.sample('theta_rep_day2', dist.Gamma(theta_rep_day2_conc, theta_rep_day2_rate)).to(device)
                
        "K: Gamma distribution"
        #conc_k = torch.tensor([2]).to(device)        
        #rate_k = torch.tensor([4]).to(device)        
        #k = pyro.sample('k', dist.Gamma(conc_k, rate_k)).to(device)
        k=torch.tensor([4.])
        
        # parameter shape for inference (until now): [1] or [n_particles]
        param_dict = {"lr0_day1": lr0_day1[:, None], \
                      "lr_lambda_day1": lr_lambda_day1[:, None], \
                      "theta_Q_day1": theta_Q_day1[None, :], \
                      "theta_rep_day1": theta_rep_day1[None, :], \
                      "lr0_day2": lr0_day2[:,None], \
                      "lr_lambda_day2": lr_lambda_day2[:, None], \
                      "theta_Q_day2": theta_Q_day2[None, :], \
                      "theta_rep_day2": theta_rep_day2[None, :], \
                      "k": k[:, None]}

        # lr is shape [1,1] or [n_particles, 1]
        self.agent.reset(**param_dict)
        
        t = -1
        for tau in range(self.trials):
            
            trial = self.data["Trialsequence"][tau]
            blocktype = self.data["Blocktype"][tau]
            
            if self.data["Blockidx"][tau] <= 5:
                day = 1
                
            elif self.data["Blockidx"][tau] > 5:
                day = 2
                
            else:
                raise Exception("Da isch a Fehla!")
            
            if trial == -1:
                "Beginning of new block"
                self.agent.update(-1,-1,-1, day, trial)
                
            else:
                current_choice = self.data["Choices"][tau]
                outcome = self.data["Outcomes"][tau]
                
            if trial > 10:
                "Dual-Target Trial"
                t+=1
                option1, option2 = self.agent.find_resp_options(trial)
                
                # Comes out as [1, n_actions] or [n_particles, n_actions]
                probs = self.agent.softmax(torch.cat(([self.agent.V[-1][:, option1][:, None], self.agent.V[-1][:, option2][:, None]]), dim=-1))
                
                if current_choice == option1:
                    choice = torch.tensor([0])
                    
                elif current_choice == option2:
                    choice = torch.tensor([1])
                    
                elif current_choice == -10:
                    "Error"
                    pass
                    
                else:
                    raise Exception("Da isch a Fehla")

            if trial != -1:
                "Update (trial == -1 means this is the beginning of a block -> participants didn't see this trial')"
                self.agent.update(current_choice, outcome, blocktype, day, trial)

            "Sample if dual-target trial and no error was performed"
            if trial > 10 and current_choice > -1:
                # choice shape for inference: [1]
                pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), obs=choice)
                    
                    
    def guide(self):
        # approximate posterior. assume MF: each param has its own univariate Normal.

        "--- Day 1 ---"
        "lr0"
        lr0_day1_alpha = pyro.param("lr0_day1_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr0_day1_beta = pyro.param("lr0_day1_beta", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr0_day1 = pyro.sample('lr0_day1', dist.Beta(lr0_day1_alpha, lr0_day1_beta)).to(device)
        
        "lr_lambda"
        lr_lambda_day1_alpha = pyro.param("lr_lambda_day1_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_lambda_day1_beta = pyro.param("lr_lambda_day1_beta", torch.tensor([10.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_lambda_day1 = pyro.sample('lr_lambda_day1', dist.Beta(lr_lambda_day1_alpha, lr_lambda_day1_beta)).to(device)

        "Θ_Q: Gamma Distribution"
        theta_Q_day1_conc = pyro.param("theta_Q_day1_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q_day1_rate = pyro.param("theta_Q_day1_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q_day1 = pyro.sample('theta_Q_day1', dist.Gamma(theta_Q_day1_conc, theta_Q_day1_rate)).to(device)
        
        "Θ_rep: Gamma Distribution"
        theta_rep_day1_conc = pyro.param("theta_rep_day1_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep_day1_rate = pyro.param("theta_rep_day1_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep_day1 = pyro.sample('theta_rep_day1', dist.Gamma(theta_rep_day1_conc, theta_rep_day1_rate)).to(device)


        " --- Day 2 ---"
        "lr0"
        lr0_day2_alpha = pyro.param("lr0_day2_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr0_day2_beta = pyro.param("lr0_day2_beta", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr0_day2 = pyro.sample('lr0_day2', dist.Beta(lr0_day2_alpha, lr0_day2_beta)).to(device)
        
        "lr_lambda"
        lr_lambda_day2_alpha = pyro.param("lr_lambda_day2_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_lambda_day2_beta = pyro.param("lr_lambda_day2_beta", torch.tensor([10.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_lambda_day2 = pyro.sample('lr_lambda_day2', dist.Beta(lr_lambda_day2_alpha, lr_lambda_day2_beta)).to(device)

        "Θ_Q: Gamma Distribution"
        theta_Q_day2_conc = pyro.param("theta_Q_day2_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q_day2_rate = pyro.param("theta_Q_day2_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q_day2 = pyro.sample('theta_Q_day2', dist.Gamma(theta_Q_day2_conc, theta_Q_day2_rate)).to(device)
        
        "Θ_rep: Gamma Distribution"
        theta_rep_day2_conc = pyro.param("theta_rep_day2_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep_day2_rate = pyro.param("theta_rep_day2_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep_day2 = pyro.sample('theta_rep_day2', dist.Gamma(theta_rep_day2_conc, theta_rep_day2_rate)).to(device)
        
        "K: Gamma Distribution"
        #conc_k = pyro.param("conc_k", torch.tensor([2]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        #rate_k = pyro.param("rate_k", torch.tensor([4]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        #k = pyro.sample('k', dist.Gamma(conc_k, rate_k)).to(device)

        param_dict = {"lr0_day1_alpha": lr0_day1_alpha, \
                      "lr0_day1_beta": lr0_day1_beta, \
                      "lr0_day1": lr0_day1, \
                      "lr_lambda_day1_alpha": lr_lambda_day1_alpha, \
                      "lr_lambda_day1_beta": lr_lambda_day1_beta, \
                      "lr_lambda_day1": lr_lambda_day1, \
                      "theta_Q_day1_conc": theta_Q_day1_conc, \
                      "theta_Q_day1_rate": theta_Q_day1_rate, \
                      "theta_Q_day1": theta_Q_day1, \
                      "theta_rep_day1_conc": theta_rep_day1_conc, \
                      "theta_rep_day1_rate": theta_rep_day1_rate, \
                      "theta_rep_day1": theta_rep_day1, \
                      "lr0_day2_alpha": lr0_day2_alpha, \
                      "lr0_day2_beta": lr0_day2_beta, \
                      "lr0_day2": lr0_day2, \
                      "lr_lambda_day2_alpha": lr_lambda_day2_alpha, \
                      "lr_lambda_day2_beta": lr_lambda_day2_beta, \
                      "lr_lambda_day2": lr_lambda_day2, \
                      "theta_Q_day2_conc": theta_Q_day2_conc, \
                      "theta_Q_day2_rate": theta_Q_day2_rate, \
                      "theta_Q_day2": theta_Q_day2, \
                      "theta_rep_day2_conc": theta_rep_day2_conc, \
                      "theta_rep_day2_rate": theta_rep_day2_rate, \
                      "theta_rep_day2": theta_rep_day2}
            
        return param_dict

    def infer_posterior(self,
                        iter_steps=1_000,
                        num_particles=10,
                        optim_kwargs={'lr': .01}): # Adam learning rate
        """Perform SVI over free model parameters."""

        pyro.clear_param_store()

        svi = pyro.infer.SVI(model=self.model,
                  guide=self.guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  # set below to true once code is vectorized
                                  vectorize_particles=True))

        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:#range(iter_steps):
            loss.append(torch.tensor(svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
            if torch.isnan(loss[-1]):
                break

        self.loss = [l.cpu() for l in loss] # = -ELBO (Plotten!)
        
        "--- Day 1 ---"
        lr0_day1_alpha = pyro.param("lr0_day1_alpha").data.numpy()
        lr0_day1_beta = pyro.param("lr0_day1_beta").data.numpy()

        lr_lambda_day1_alpha = pyro.param("lr_lambda_day1_alpha").data.numpy()
        lr_lambda_day1_beta = pyro.param("lr_lambda_day1_beta").data.numpy()
        
        theta_Q_day1_conc = pyro.param("theta_Q_day1_conc").data.numpy()
        theta_Q_day1_rate = pyro.param("theta_Q_day1_rate").data.numpy()
        
        theta_rep_day1_conc = pyro.param("theta_rep_day1_conc").data.numpy()
        theta_rep_day1_rate = pyro.param("theta_rep_day1_rate").data.numpy()

        "--- Day 2 ---"
        lr0_day2_alpha = pyro.param("lr0_day2_alpha").data.numpy()
        lr0_day2_beta = pyro.param("lr0_day2_beta").data.numpy()
        
        lr_lambda_day2_alpha = pyro.param("lr_lambda_day2_alpha").data.numpy()
        lr_lambda_day2_beta = pyro.param("lr_lambda_day2_beta").data.numpy()
        
        theta_Q_day2_conc = pyro.param("theta_Q_day2_conc").data.numpy()
        theta_Q_day2_rate = pyro.param("theta_Q_day2_rate").data.numpy()
        
        theta_rep_day2_conc = pyro.param("theta_rep_day2_conc").data.numpy()
        theta_rep_day2_rate = pyro.param("theta_rep_day2_rate").data.numpy()

        param_dict = {"lr0_day1_alpha": lr0_day1_alpha, \
                      "lr0_day1_beta": lr0_day1_beta, \
                      "lr_lambda_day1_alpha": lr_lambda_day1_alpha, \
                      "lr_lambda_day1_beta": lr_lambda_day1_beta, \
                      "theta_Q_day1_conc": theta_Q_day1_conc, \
                      "theta_Q_day1_rate": theta_Q_day1_rate, \
                      "theta_rep_day1_conc": theta_rep_day1_conc, \
                      "theta_rep_day1_rate": theta_rep_day1_rate, \
                      "lr0_day2_alpha": lr0_day2_alpha, \
                      "lr0_day2_beta": lr0_day2_beta, \
                      "lr_lambda_day2_alpha": lr_lambda_day2_alpha, \
                      "lr_lambda_day2_beta": lr_lambda_day2_beta, \
                      "theta_Q_day2_conc": theta_Q_day2_conc, \
                      "theta_Q_day2_rate": theta_Q_day2_rate, \
                      "theta_rep_day2_conc": theta_rep_day2_conc, \
                      "theta_rep_day2_rate": theta_rep_day2_rate}

        return self.loss, param_dict
  
class SingleInference_modelD(object):
    "Fixed Q-values, but theta_rep and theta_Q develop like exponential function"
    def __init__(self, agent, data):

        self.agent = agent
        self.trials = self.agent.trials # length of experiment
        #self.T = agent.T # No. of "subtrials" -> unnötig
        self.data = data # Dictionary

    def model(self):
        # generative model of behavior with Normally distributed params (within subject!!)

        # tell pyro about prior over parameters: alpha and beta of lr which is between 0 and 1
        # alpha = beta = 1 equals uniform prior
        "--- Day 1 ---"
        "Θ_Qlambda: Beta Distribution"
        theta_Qlambda_day1_alpha = torch.tensor([1.]).to(device)
        theta_Qlambda_day1_beta = torch.tensor([1.]).to(device)
        theta_Qlambda_day1 = pyro.sample('theta_Qlambda_day1', dist.Beta(theta_Qlambda_day1_alpha, theta_Qlambda_day1_beta)).to(device)
        
        "Θ_Q0"
        theta_Q0_day1_conc = torch.tensor([4.]).to(device)
        theta_Q0_day1_rate = torch.tensor([4.]).to(device)
        theta_Q0_day1 = pyro.sample('theta_Q0_day1', dist.Gamma(theta_Q0_day1_conc, theta_Q0_day1_rate)).to(device)
        
        "Θ_Qlambda: Beta Distribution"
        theta_replambda_day1_alpha = torch.tensor([1.]).to(device)
        theta_replambda_day1_beta = torch.tensor([1.]).to(device)
        theta_replambda_day1 = pyro.sample('theta_replambda_day1', dist.Beta(theta_replambda_day1_alpha, theta_replambda_day1_beta)).to(device)
        
        "Θ_rep0"
        theta_rep0_day1_conc = torch.tensor([4.]).to(device)
        theta_rep0_day1_rate = torch.tensor([4.]).to(device)
        theta_rep0_day1 = pyro.sample('theta_rep0_day1', dist.Gamma(theta_rep0_day1_conc, theta_rep0_day1_rate)).to(device)
        
        "--- Day 2 ---"
        "Θ_Qlambda: Beta Distribution"
        theta_Qlambda_day2_alpha = torch.tensor([1.]).to(device)
        theta_Qlambda_day2_beta = torch.tensor([1.]).to(device)
        theta_Qlambda_day2 = pyro.sample('theta_Qlambda_day2', dist.Beta(theta_Qlambda_day2_alpha, theta_Qlambda_day2_beta)).to(device)
        
        "Θ_Q0"
        theta_Q0_day2_conc = torch.tensor([4.]).to(device)
        theta_Q0_day2_rate = torch.tensor([4.]).to(device)
        theta_Q0_day2 = pyro.sample('theta_Q0_day2', dist.Gamma(theta_Q0_day2_conc, theta_Q0_day2_rate)).to(device)
        
        "Θ_Qlambda: Beta Distribution"
        theta_replambda_day2_alpha = torch.tensor([1.]).to(device)
        theta_replambda_day2_beta = torch.tensor([1.]).to(device)
        theta_replambda_day2 = pyro.sample('theta_replambda_day2', dist.Beta(theta_replambda_day2_alpha, theta_replambda_day2_beta)).to(device)
        
        "Θ_rep0"
        theta_rep0_day2_conc = torch.tensor([4.]).to(device)
        theta_rep0_day2_rate = torch.tensor([4.]).to(device)
        theta_rep0_day2 = pyro.sample('theta_rep0_day2', dist.Gamma(theta_rep0_day2_conc, theta_rep0_day2_rate)).to(device)

        "K: Gamma distribution"
        #conc_k = torch.tensor([2]).to(device)        
        #rate_k = torch.tensor([4]).to(device)        
        #k = pyro.sample('k', dist.Gamma(conc_k, rate_k)).to(device)
        k=torch.tensor([4.])
        
        # parameter shape for inference (until now): [1] or [n_particles]
        param_dict = {"theta_Qlambda_day1": theta_Qlambda_day1[None,:], \
                      "theta_Q0_day1": theta_Q0_day1[:, None], \
                      "theta_replambda_day1": theta_replambda_day1[None, :], \
                      "theta_rep0_day1": theta_rep0_day1[None, :], \
                      "theta_Qlambda_day2": theta_Qlambda_day2[None,:], \
                      "theta_Q0_day2": theta_Q0_day2[:, None], \
                      "theta_replambda_day2": theta_replambda_day2[None, :], \
                      "theta_rep0_day2": theta_rep0_day2[None, :], \
                      "k": k[:, None]}

        
        # lr is shape [1,1] or [n_particles, 1]
        self.agent.reset(**param_dict)
        
        t = -1
        for tau in range(self.trials):
            
            trial = self.data["Trialsequence"][tau]
            blocktype = self.data["Blocktype"][tau]
            
            if self.data["Blockidx"][tau] <= 5:
                day = 1
                
            elif self.data["Blockidx"][tau] > 5:
                day = 2
                
            else:
                raise Exception("Da isch a Fehla!")
            
            if trial == -1:
                "Beginning of new block"
                self.agent.update(-1,-1,-1, day, trial)
                
            else:
                current_choice = self.data["Choices"][tau]
                outcome = self.data["Outcomes"][tau]

            if trial > 10:
                "Dual-Target Trial"
                t+=1
                option1, option2 = self.agent.find_resp_options(trial)
                
                # Comes out as [1, n_actions] or [n_particles, n_actions]
                probs = self.agent.softmax(torch.cat(([self.agent.V[-1][:, option1][:, None], self.agent.V[-1][:, option2][:, None]]), dim=-1))
                
                if current_choice == option1:
                    choice = torch.tensor([0])
                    
                elif current_choice == option2:
                    choice = torch.tensor([1])
                    
                elif current_choice == -10:
                    "Error"
                    pass
                    
                else:
                    raise Exception("Da isch a Fehla")

            if trial != -1:
                "Update (trial == -1 means this is the beginning of a block -> participants didn't see this trial')"
                self.agent.update(current_choice, outcome, blocktype, day, trial)

            "Sample if dual-target trial and no error was performed"
            if trial > 10 and current_choice > -1:
                # choice shape for inference: [1]
                # print("About to sample trial %d, baby!"%tau)
                pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), obs=choice)
                    
    def guide(self):
        # approximate posterior. assume MF: each param has its own univariate Normal.

        "--- Day 1 ---"
        "Θ_Qlambda"
        theta_Qlambda_day1_alpha = pyro.param("theta_Qlambda_day1_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Qlambda_day1_beta = pyro.param("theta_Qlambda_day1_beta", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Qlambda_day1 = pyro.sample('theta_Qlambda_day1', dist.Beta(theta_Qlambda_day1_alpha, theta_Qlambda_day1_beta)).to(device)

        "Θ_Q0: Gamma Distribution"
        theta_Q0_day1_conc = pyro.param("theta_Q0_day1_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q0_day1_rate = pyro.param("theta_Q0_day1_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q0_day1 = pyro.sample('theta_Q0_day1', dist.Gamma(theta_Q0_day1_conc, theta_Q0_day1_rate)).to(device)
        
        "Θ_replambda"
        theta_replambda_day1_alpha = pyro.param("theta_replambda_day1_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_replambda_day1_beta = pyro.param("theta_replambda_day1_beta", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_replambda_day1 = pyro.sample('theta_replambda_day1', dist.Beta(theta_replambda_day1_alpha, theta_replambda_day1_beta)).to(device)

        "Θ_rep0: Gamma Distribution"
        theta_rep0_day1_conc = pyro.param("theta_rep0_day1_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep0_day1_rate = pyro.param("theta_rep0_day1_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep0_day1 = pyro.sample('theta_rep0_day1', dist.Gamma(theta_rep0_day1_conc, theta_rep0_day1_rate)).to(device)

        " --- Day 2 ---"
        "Θ_Qlambda"
        theta_Qlambda_day2_alpha = pyro.param("theta_Qlambda_day2_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Qlambda_day2_beta = pyro.param("theta_Qlambda_day2_beta", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Qlambda_day2 = pyro.sample('theta_Qlambda_day2', dist.Beta(theta_Qlambda_day2_alpha, theta_Qlambda_day2_beta)).to(device)

        "Θ_Q0: Gamma Distribution"
        theta_Q0_day2_conc = pyro.param("theta_Q0_day2_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q0_day2_rate = pyro.param("theta_Q0_day2_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q0_day2 = pyro.sample('theta_Q0_day2', dist.Gamma(theta_Q0_day2_conc, theta_Q0_day2_rate)).to(device)
        
        "Θ_replambda"
        theta_replambda_day2_alpha = pyro.param("theta_replambda_day2_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_replambda_day2_beta = pyro.param("theta_replambda_day2_beta", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_replambda_day2 = pyro.sample('theta_replambda_day2', dist.Beta(theta_replambda_day2_alpha, theta_replambda_day2_beta)).to(device)

        "Θ_rep0: Gamma Distribution"
        theta_rep0_day2_conc = pyro.param("theta_rep0_day2_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep0_day2_rate = pyro.param("theta_rep0_day2_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep0_day2 = pyro.sample('theta_rep0_day2', dist.Gamma(theta_rep0_day2_conc, theta_rep0_day2_rate)).to(device)
        
        "K: Gamma Distribution"
        #conc_k = pyro.param("conc_k", torch.tensor([2]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        #rate_k = pyro.param("rate_k", torch.tensor([4]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        #k = pyro.sample('k', dist.Gamma(conc_k, rate_k)).to(device)

        param_dict = {"theta_Qlambda_day1_alpha": theta_Qlambda_day1_alpha, \
                      "theta_Qlambda_day1_beta": theta_Qlambda_day1_beta, \
                      "theta_Qlambda_day1": theta_Qlambda_day1, \
                      "theta_Q0_day1_conc": theta_Q0_day1_conc, \
                      "theta_Q0_day1_rate": theta_Q0_day1_rate, \
                      "theta_Q0_day1": theta_Q0_day1, \
                      "theta_replambda_day1_alpha": theta_replambda_day1_alpha, \
                      "theta_replambda_day1_beta": theta_replambda_day1_beta, \
                      "theta_replambda_day1": theta_replambda_day1, \
                      "theta_rep0_day1_conc": theta_rep0_day1_conc, \
                      "theta_rep0_day1_rate": theta_rep0_day1_rate, \
                      "theta_rep0_day1": theta_rep0_day1, \
                      "theta_Qlambda_day2_alpha": theta_Qlambda_day2_alpha, \
              "theta_Qlambda_day2_beta": theta_Qlambda_day2_beta, \
              "theta_Qlambda_day2": theta_Qlambda_day2, \
              "theta_Q0_day2_conc": theta_Q0_day2_conc, \
              "theta_Q0_day2_rate": theta_Q0_day2_rate, \
              "theta_Q0_day2": theta_Q0_day2, \
              "theta_replambda_day2_alpha": theta_replambda_day2_alpha, \
              "theta_replambda_day2_beta": theta_replambda_day2_beta, \
              "theta_replambda_day2": theta_replambda_day2, \
              "theta_rep0_day2_conc": theta_rep0_day2_conc, \
              "theta_rep0_day2_rate": theta_rep0_day2_rate, \
              "theta_rep0_day2": theta_rep0_day2,}
            
        return param_dict

    def infer_posterior(self,
                        iter_steps=1_000,
                        num_particles=10,
                        optim_kwargs={'lr': .01}): # Adam learning rate
        """Perform SVI over free model parameters."""

        pyro.clear_param_store()

        svi = pyro.infer.SVI(model=self.model,
                  guide=self.guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  # set below to true once code is vectorized
                                  vectorize_particles=True))
                
        "Next : run self.model() once per iter_step (is tun in line loss.append(torch.tensor(svi.step()).to(device)))"
        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:#range(iter_steps):
            # ipdb.set_trace()
            loss.append(torch.tensor(svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
            if torch.isnan(loss[-1]):
                break

        self.loss = [l.cpu() for l in loss] # = -ELBO (Plotten!)
        
        "--- Day 1 ---"
        theta_Qlambda_day1_alpha = pyro.param("theta_Qlambda_day1_alpha").data.numpy()
        theta_Qlambda_day1_beta = pyro.param("theta_Qlambda_day1_beta").data.numpy()

        theta_Q0_day1_conc = pyro.param("theta_Q0_day1_conc").data.numpy()
        theta_Q0_day1_rate = pyro.param("theta_Q0_day1_rate").data.numpy()
        
        theta_replambda_day1_alpha = pyro.param("theta_replambda_day1_alpha").data.numpy()
        theta_replambda_day1_beta = pyro.param("theta_replambda_day1_beta").data.numpy()

        theta_rep0_day1_conc = pyro.param("theta_rep0_day1_conc").data.numpy()
        theta_rep0_day1_rate = pyro.param("theta_rep0_day1_rate").data.numpy()

        "--- Day 2 ---"
        theta_Qlambda_day2_alpha = pyro.param("theta_Qlambda_day2_alpha").data.numpy()
        theta_Qlambda_day2_beta = pyro.param("theta_Qlambda_day2_beta").data.numpy()

        theta_Q0_day2_conc = pyro.param("theta_Q0_day2_conc").data.numpy()
        theta_Q0_day2_rate = pyro.param("theta_Q0_day2_rate").data.numpy()
        
        theta_replambda_day2_alpha = pyro.param("theta_replambda_day2_alpha").data.numpy()
        theta_replambda_day2_beta = pyro.param("theta_replambda_day2_beta").data.numpy()

        theta_rep0_day2_conc = pyro.param("theta_rep0_day2_conc").data.numpy()
        theta_rep0_day2_rate = pyro.param("theta_rep0_day2_rate").data.numpy()

        param_dict = {"theta_Qlambda_day1_alpha": theta_Qlambda_day1_alpha, \
                      "theta_Qlambda_day1_beta": theta_Qlambda_day1_beta, \
                      "theta_Q0_day1_conc": theta_Q0_day1_conc, \
                      "theta_Q0_day1_rate": theta_Q0_day1_rate, \
                      "theta_replambda_day1_alpha": theta_replambda_day1_alpha, \
                      "theta_replambda_day1_beta": theta_replambda_day1_beta, \
                      "theta_rep0_day1_conc": theta_rep0_day1_conc, \
                      "theta_rep0_day1_rate": theta_rep0_day1_rate, \
                      "theta_Qlambda_day2_alpha": theta_Qlambda_day2_alpha, \
                      "theta_Qlambda_day2_beta": theta_Qlambda_day2_beta, \
                      "theta_Q0_day2_conc": theta_Q0_day2_conc, \
                      "theta_Q0_day2_rate": theta_Q0_day2_rate, \
                      "theta_replambda_day2_alpha": theta_replambda_day2_alpha, \
                      "theta_replambda_day2_beta": theta_replambda_day2_beta, \
                      "theta_rep0_day2_conc": theta_rep0_day2_conc, \
                      "theta_rep0_day2_rate": theta_rep0_day2_rate}

        return self.loss, param_dict