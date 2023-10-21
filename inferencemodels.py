#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:14:10 2023

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

"Inference"
class SingleInference(object):

    def __init__(self, agent, data):

        self.agent = agent
        self.trials = self.agent.trials # length of experiment
        #self.T = agent.T # No. of "subtrials" -> unnötig
        self.data = data # Dictionary
        self.habvals = [] # For DDM

    def model(self):
        # generative model of behavior with Normally distributed params (within subject!!)

        # tell pyro about prior over parameters: alpha and beta of lr which is between 0 and 1
        # alpha = beta = 1 equals uniform prior
        "Learning Rate: Beta Distribution"
        
        lr_alpha = torch.tensor([1.])
        lr_beta = torch.tensor([1.])
        lr = pyro.sample('lr', dist.Beta(lr_alpha, lr_beta)).to(device)
                
        "Omega: Beta Distribution"
        omega_alpha = torch.tensor([1.]).to(device)
        omega_beta = torch.tensor([1.]).to(device)        # sample initial vaue of parameter from Beta distribution        
        omega = pyro.sample('omega', dist.Beta(omega_alpha, omega_beta)).to(device)
        
        "Beta : Gamma distribution"
        dectemp_conc = pyro.param("dectemp_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        dectemp_rate = pyro.param("dectemp_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))   
        dectemp = pyro.sample('dectemp', dist.Gamma(dectemp_conc, dectemp_rate)).to(device)
        
        "K: Gamma distribution"
        #conc_k = torch.tensor([2]).to(device)        
        #rate_k = torch.tensor([4]).to(device)        
        #k = pyro.sample('k', dist.Gamma(conc_k, rate_k)).to(device)
        k=torch.tensor([4.])
        
        # parameter shape for inference (until now): [1] or [n_particles]
        param_dict = {"lr": lr[:,None], "omega": omega[None, :], "dectemp": dectemp[:, None], "k": k[:, None]}

        # lr is shape [1,1] or [n_particles, 1]
        self.agent.reset(**param_dict)
        
        t = -1
        for tau in range(self.trials):
            
            trial = self.data["Trialsequence"][tau]
            blocktype = self.data["Blocktype"][tau]
            
            #vec_ben = np.round(np.array(self.data["repvals"][tau]),12)
            #vec_agent = np.round(np.array(torch.squeeze(self.agent.rep[-1])),12)
            
            #np.testing.#assert_allclose(vec_ben, vec_agent, rtol=1e-5)
            
            if self.data["Blockidx"][tau] <= 5:
                day = 1
                
            elif self.data["Blockidx"][tau] > 5:
                day = 2
            
            if trial == -1:
                "Beginning of new block"
                "Set previous actions to -1 because it's the beginning of a new block"
                self.agent.pppchoice = -1
                self.agent.ppchoice = -1
                self.agent.pchoice = -1
                
                "Set repetition values to 0 because of new block"
                self.agent.rep.append(torch.tensor([[0.25, 0.25, 0.25, 0.25]],))
                self.agent.Q.append(self.agent.Q[-1])
                
                V0 = (1-self.agent.omega)*self.agent.rep[-1][..., 0] + self.agent.omega*self.agent.Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
                V1 = (1-self.agent.omega)*self.agent.rep[-1][..., 1] + self.agent.omega*self.agent.Q[-1][..., 1]
                V2 = (1-self.agent.omega)*self.agent.rep[-1][..., 2] + self.agent.omega*self.agent.Q[-1][..., 2]
                V3 = (1-self.agent.omega)*self.agent.rep[-1][..., 3] + self.agent.omega*self.agent.Q[-1][..., 3]
                
                self.agent.V.append(torch.cat((V0.transpose(0,1),V1.transpose(0,1),V2.transpose(0,1),V3.transpose(0,1)),1))
                
                
            else:
                current_choice = self.data["Choices"][tau]
                outcome = self.data["Outcomes"][tau]
                
            if trial > 10:
                "Dual-Target Trial"
                t+=1
                #assert(isinstance(trial, list))
                option1, option2 = self.agent.find_resp_options(trial)
                
                # Comes out as [1, n_actions] or [n_particles, n_actions]probs
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
                    
            #self.habvals = extend_habval(self.habvals, trial, self.agent)

            if trial != -1:
                "Update (trial == -1 means this is the beginning of a block -> participants didn't see this trial')"
                self.agent.update(current_choice, outcome, blocktype, day=day, trialstimulus = trial)

            "Sample if dual-target trial and no error was performed"
            if trial > 10 and current_choice > -1:
                # choice shape for inference: [1]
                pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), obs=choice)
                    
    def guide(self):
        # approximate posterior. assume MF: each param has its own univariate Normal.

        "Learning Rate"
        lr_alpha = pyro.param("lr_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_beta = pyro.param("lr_beta", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        
        # sample value of parameter from Beta distribution
        lr = pyro.sample('lr', dist.Beta(lr_alpha, lr_beta)).to(device)

        "Omega: Beta Distribution"
        omega_alpha = pyro.param("omega_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        omega_beta = pyro.param("omega_beta", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        omega = pyro.sample('omega', dist.Beta(omega_alpha, omega_beta)).to(device)

        "Beta: Gamma Distribution"
        dectemp_conc = pyro.param("dectemp_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        dectemp_rate = pyro.param("dectemp_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        dectemp = pyro.sample('dectemp', dist.Gamma(dectemp_conc, dectemp_rate)).to(device)
        
        "K: Gamma Distribution"
        #conc_k = pyro.param("conc_k", torch.tensor([2]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        #rate_k = pyro.param("rate_k", torch.tensor([4]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        #k = pyro.sample('k', dist.Gamma(conc_k, rate_k)).to(device)

        param_dict = {"lr_alpha": lr_alpha, "lr_beta": lr_beta, "lr": lr, "omega_alpha": omega_alpha, \
                      "omega_beta": omega_beta, "omega": omega, "dectemp_conc": dectemp_conc, "dectemp_rate": dectemp_rate, \
                      "dectemp": dectemp}

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
        
        lr_alpha = pyro.param("lr_alpha").data.numpy()
        lr_beta = pyro.param("lr_beta").data.numpy()
        
        omega_alpha = pyro.param("omega_alpha").data.numpy()
        omega_beta = pyro.param("omega_beta").data.numpy()
        
        dectemp_conc = pyro.param("dectemp_conc").data.numpy()
        dectemp_rate = pyro.param("dectemp_rate").data.numpy()

        param_dict = {"lr_alpha": lr_alpha, "lr_beta": lr_beta, "omega_alpha": omega_alpha, \
                      "omega_beta": omega_beta, "dectemp_conc": dectemp_conc, "dectemp_rate": dectemp_rate}
            
        return self.loss, param_dict
    
    def sample_posterior(self, n_samples=1000):
        guide = self.guide
    
        lr_alpha_samples = np.zeros(n_samples)
        lr_beta_samples = np.zeros(n_samples)
        lr_samples = np.zeros(n_samples)
        omega_alpha_samples = np.zeros(n_samples)
        omega_beta_samples = np.zeros(n_samples)
        omega_samples = np.zeros(n_samples)
        dectemp_conc_samples = np.zeros(n_samples)
        dectemp_rate_samples = np.zeros(n_samples)
        dectemp_samples = np.zeros(n_samples)
    
        # sample p from guide (the posterior over p). 
        # Calling the guide yields samples from the posterior after SVI has run.
        for i in range(n_samples):
            sample = guide()
            for key in sample.keys():
                sample.setdefault(key, torch.ones(1))
                
            lr_alpha_samples[i] = sample["lr_alpha"].detach().numpy()
            lr_beta_samples[i] = sample["lr_beta"].detach().numpy()
            lr_samples[i] = sample["lr"].detach().numpy()
            
            omega_alpha_samples[i] = sample["omega_alpha"].detach().numpy()
            omega_beta_samples[i] = sample["omega_beta"].detach().numpy()
            omega_samples[i] = sample["omega"].detach().numpy()
    
            dectemp_conc_samples[i] = sample["dectemp_conc"].detach().numpy()
            dectemp_rate_samples[i] = sample["dectemp_rate"].detach().numpy()
            dectemp_samples[i] = sample["dectemp"].detach().numpy()
        
        sample_dict = {"lr_alpha": lr_alpha_samples, \
                       "lr_beta": lr_beta_samples, \
                       "lr": lr_samples, \
                       "omega_alpha": omega_alpha_samples, \
                       "omega_beta": omega_beta_samples, \
                       "omega": omega_samples, \
                       "dectemp_conc": dectemp_conc_samples, \
                       "dectemp_rate": dectemp_rate_samples, \
                       "dectemp": dectemp_samples}
    
        # make a pandas dataframe, better for analyses and plotting later (pandas is pythons R equivalent)
        sample_df = pd.DataFrame(sample_dict)
    
        return sample_df

class SingleInference_modelB(object):

    def __init__(self, agent, data, k, **kwargs):

        self.agent = agent
        self.trials = self.agent.trials # length of experiment
        # self.T = agent.T # No. of "subtrials" -> unnötig
        self.data = data # Dictionary
        self.k = k
        
        if 'lr_day1_alpha' in kwargs:
            self.lr_day1_alpha = kwargs['lr_day1_alpha']
            self.lr_day1_beta = kwargs['lr_day1_beta']
            
            self.theta_Q_day1_conc = kwargs['theta_Q_day1_conc']
            self.theta_Q_day1_rate = kwargs['theta_Q_day1_rate']
            
            self.theta_rep_day1_conc = kwargs['theta_rep_day1_conc']
            self.theta_rep_day1_rate = kwargs['theta_rep_day1_rate']
            
            self.lr_day2_alpha = kwargs['lr_day2_alpha']
            self.lr_day2_beta = kwargs['lr_day2_beta']
            
            self.theta_Q_day2_conc = kwargs['theta_Q_day2_conc']
            self.theta_Q_day2_rate = kwargs['theta_Q_day2_rate']
            
            self.theta_rep_day2_conc = kwargs['theta_rep_day2_conc']
            self.theta_rep_day2_rate = kwargs['theta_rep_day2_rate']
            
        else:
            self.lr_day1_alpha = 1.
            self.lr_day1_beta = 20.
            
            self.theta_Q_day1_conc = 4.
            self.theta_Q_day1_rate = 4.
            
            self.theta_rep_day1_conc = 4.
            self.theta_rep_day1_rate = 4.
            
            self.lr_day2_alpha = 1.
            self.lr_day2_beta = 20.
            
            self.theta_Q_day2_conc = 4.
            self.theta_Q_day2_rate = 4.
            
            self.theta_rep_day2_conc = 4.
            self.theta_rep_day2_rate = 4.

    def model(self):
        # generative model of behavior with Normally distributed params (within subject!!)
       
        # tell pyro about prior over parameters: alpha and beta of lr which is between 0 and 1
        # alpha = beta = 1 equals uniform prior
        "--- Day 1 ---"
        "Learning Rate: Beta Distribution"
        lr_day1_alpha = torch.tensor([self.lr_day1_alpha]).to(device)
        lr_day1_beta = torch.tensor([self.lr_day1_beta]).to(device)
        lr_day1 = pyro.sample('lr_day1', dist.Beta(lr_day1_alpha, lr_day1_beta)).to(device)
        
        "Θ_Q"
        theta_Q_day1_conc = torch.tensor([self.theta_Q_day1_conc]).to(device)
        theta_Q_day1_rate = torch.tensor([self.theta_Q_day1_rate]).to(device)
        theta_Q_day1 = pyro.sample('theta_Q_day1', dist.Gamma(theta_Q_day1_conc, theta_Q_day1_rate)).to(device)
        
        "Θ_rep"
        theta_rep_day1_conc = torch.tensor([self.theta_rep_day1_conc]).to(device)
        theta_rep_day1_rate = torch.tensor([self.theta_rep_day1_rate]).to(device)
        theta_rep_day1 = pyro.sample('theta_rep_day1', dist.Gamma(theta_rep_day1_conc, theta_rep_day1_rate)).to(device)
        
        "--- Day 2 ---"
        "Learning Rate: Beta Distribution"
        lr_day2_alpha = torch.tensor([self.lr_day2_alpha]).to(device)
        lr_day2_beta = torch.tensor([self.lr_day2_beta]).to(device)
        lr_day2 = pyro.sample('lr_day2', dist.Beta(lr_day2_alpha, lr_day2_beta)).to(device)
        
        "Θ_Q"
        theta_Q_day2_conc = torch.tensor([self.theta_Q_day2_conc]).to(device)
        theta_Q_day2_rate = torch.tensor([self.theta_Q_day2_rate]).to(device)
        theta_Q_day2 = pyro.sample('theta_Q_day2', dist.Gamma(theta_Q_day2_conc, theta_Q_day2_rate)).to(device)
        
        "Θ_rep"
        theta_rep_day2_conc = torch.tensor([self.theta_rep_day2_conc]).to(device)
        theta_rep_day2_rate = torch.tensor([self.theta_rep_day2_rate]).to(device)
        theta_rep_day2 = pyro.sample('theta_rep_day2', dist.Gamma(theta_rep_day2_conc, theta_rep_day2_rate)).to(device)
                
        "K: Gamma distribution"
        #conc_k = torch.tensor([2]).to(device)        
        #rate_k = torch.tensor([4]).to(device)        
        #k = pyro.sample('k', dist.Gamma(conc_k, rate_k)).to(device)
        k=torch.tensor([self.k])
                
        # parameter shape for inference (until now): [1] or [n_particles]
        param_dict = {"lr_day1": lr_day1[:,None], \
                      "lr_day2": lr_day2[:,None], \
                      "theta_Q_day1": theta_Q_day1[None, :], \
                      "theta_Q_day2": theta_Q_day2[None, :], \
                      "theta_rep_day1": theta_rep_day1[None, :], \
                      "theta_rep_day2": theta_rep_day2[None, :], \
                      "k": k[:, None]}

        # lr is shape [1,1] or [n_particles, 1]
        self.agent.reset(**param_dict)
        
        t = -1
        for tau in range(self.trials):
            trial = self.data["Trialsequence"][tau][0]
            blocktype = self.data["Blocktype"][tau][0]
            
            if self.data["Blockidx"][tau][0] <= 5:
                day = 1
                
            elif self.data["Blockidx"][tau][0] > 5:
                day = 2
                
            else:
                raise Exception("Da isch a Fehla!")
            
            if trial == -1:
                "Beginning of new block"
                self.agent.update(torch.tensor([-1]), torch.tensor([-1]), torch.tensor([-1]), day=day, trialstimulus=trial)
                
            else:
                current_choice = self.data["Choices"][tau]
                outcome = self.data["Outcomes"][tau]
                
            if trial > 10:
                "Dual-Target Trial"
                t+=1
                #assert(isinstance(trial, list))
                option1, option2 = self.agent.find_resp_options(trial)
                
                # Comes out as [1, n_actions] or [n_particles, n_actions]
                probs = self.agent.softmax(torch.cat(([self.agent.V[-1][..., option1], self.agent.V[-1][..., option2]]),dim=-1))
                                
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
                self.agent.update(current_choice, outcome, blocktype, day=day, trialstimulus=trial)

            "Sample if dual-target trial and no error was performed"
            if trial > 10 and current_choice > -1:
                # choice shape for inference: [1]
                pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), obs=choice)
                    
                    
    def guide(self):
        # approximate posterior. assume MF: each param has its own univariate Normal.

        "--- Day 1 ---"
        "Learning Rate"
        lr_day1_alpha = pyro.param("lr_day1_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day1_beta = pyro.param("lr_day1_beta", torch.tensor([20.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day1 = pyro.sample('lr_day1', dist.Beta(lr_day1_alpha, lr_day1_beta)).to(device)

        "Θ_Q: Gamma Distribution"
        theta_Q_day1_conc = pyro.param("theta_Q_day1_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q_day1_rate = pyro.param("theta_Q_day1_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q_day1 = pyro.sample('theta_Q_day1', dist.Gamma(theta_Q_day1_conc, theta_Q_day1_rate)).to(device)
        
        "Θ_rep: Gamma Distribution"
        theta_rep_day1_conc = pyro.param("theta_rep_day1_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep_day1_rate = pyro.param("theta_rep_day1_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep_day1 = pyro.sample('theta_rep_day1', dist.Gamma(theta_rep_day1_conc, theta_rep_day1_rate)).to(device)

        " --- Day 2 ---"
        "Learning Rate"
        lr_day2_alpha = pyro.param("lr_day2_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day2_beta = pyro.param("lr_day2_beta", torch.tensor([20.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day2 = pyro.sample('lr_day2', dist.Beta(lr_day2_alpha, lr_day2_beta)).to(device)

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

        param_dict = {"lr_day1_alpha": lr_day1_alpha, \
                      "lr_day1_beta": lr_day1_beta, \
                      "lr_day1": lr_day1, \
                      "theta_Q_day1_conc": theta_Q_day1_conc, \
                      "theta_Q_day1_rate": theta_Q_day1_rate, \
                      "theta_Q_day1": theta_Q_day1, \
                      "theta_rep_day1_conc": theta_rep_day1_conc, \
                      "theta_rep_day1_rate": theta_rep_day1_rate, \
                      "theta_rep_day1": theta_rep_day1, \
                      "lr_day2_alpha": lr_day2_alpha, \
                      "lr_day2_beta": lr_day2_beta, \
                      "lr_day2": lr_day2, \
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
        lr_day1_alpha = pyro.param("lr_day1_alpha").data.numpy()
        lr_day1_beta = pyro.param("lr_day1_beta").data.numpy()
        
        theta_Q_day1_conc = pyro.param("theta_Q_day1_conc").data.numpy()
        theta_Q_day1_rate = pyro.param("theta_Q_day1_rate").data.numpy()
        
        theta_rep_day1_conc = pyro.param("theta_rep_day1_conc").data.numpy()
        theta_rep_day1_rate = pyro.param("theta_rep_day1_rate").data.numpy()

        "--- Day 2 ---"
        lr_day2_alpha = pyro.param("lr_day2_alpha").data.numpy()
        lr_day2_beta = pyro.param("lr_day2_beta").data.numpy()
        
        theta_Q_day2_conc = pyro.param("theta_Q_day2_conc").data.numpy()
        theta_Q_day2_rate = pyro.param("theta_Q_day2_rate").data.numpy()

        theta_rep_day2_conc = pyro.param("theta_rep_day2_conc").data.numpy()
        theta_rep_day2_rate = pyro.param("theta_rep_day2_rate").data.numpy()

        param_dict = {"lr_day1_alpha": lr_day1_alpha, \
                      "lr_day1_beta": lr_day1_beta, \
                      "theta_Q_day1_conc": theta_Q_day1_conc, \
                      "theta_Q_day1_rate": theta_Q_day1_rate, \
                      "theta_rep_day1_conc": theta_rep_day1_conc, \
                      "theta_rep_day1_rate": theta_rep_day1_rate, \
                      "lr_day2_alpha": lr_day2_alpha, \
                      "lr_day2_beta": lr_day2_beta, \
                      "theta_Q_day2_conc": theta_Q_day2_conc, \
                      "theta_Q_day2_rate": theta_Q_day2_rate, \
                      "theta_rep_day2_conc": theta_rep_day2_conc, \
                      "theta_rep_day2_rate": theta_rep_day2_rate}

        return self.loss, param_dict
    
    
    def mcmc(self, step_size = 0.0855, num_steps = 4, num_samples = 500, burnin = 100):
        hmc_kernel = pyro.infer.NUTS(model=self.model, adapt_step_size=True)
        mcmc = pyro.infer.MCMC(hmc_kernel, num_samples=num_samples, warmup_steps=burnin, num_chains = 4)
        mcmc.run()
        return mcmc

class SingleInference_modelB_2(object):

    def __init__(self, agent, data, k, **kwargs):

        self.agent = agent
        self.trials = self.agent.trials # length of experiment
        #self.T = agent.T # No. of "subtrials" -> unnötig
        self.data = data # Dictionary
        
        self.k = k
        
        if 'lr_day1_1_alpha' in kwargs:
            pass
        else:
            "--- Day 1 : 1 ---"
            "Learning Rate: Beta Distribution"
            self.lr_day1_1_alpha = 1.
            self.lr_day1_1_beta = 20.
            
            "Θ_Q"
            self.theta_Q_day1_1_conc = 4.
            self.theta_Q_day1_1_rate = 4.
            
            "Θ_rep"
            self.theta_rep_day1_1_conc = 4.
            self.theta_rep_day1_1_rate = 4.
            
            "--- Day 1 : 2 ---"
            "Learning Rate: Beta Distribution"
            self.lr_day1_2_alpha = 1.
            self.lr_day1_2_beta = 20.
            
            "Θ_Q"
            self.theta_Q_day1_2_conc = 4.
            self.theta_Q_day1_2_rate = 4.
            
            "Θ_rep"
            self.theta_rep_day1_2_conc = 4.
            self.theta_rep_day1_2_rate = 4.
            
            "--- Day 2 ---"
            "Learning Rate: Beta Distribution"
            self.lr_day2_alpha = 1.
            self.lr_day2_beta = 20.
            
            "Θ_Q"
            self.theta_Q_day2_conc = 4.
            self.theta_Q_day2_rate = 4.
            
            "Θ_rep"
            self.theta_rep_day2_conc = 4.
            self.theta_rep_day2_rate = 4.

    def model(self):
        # generative model of behavior with Normally distributed params (within subject!!)

        # tell pyro about prior over parameters: alpha and beta of lr which is between 0 and 1
        # alpha = beta = 1 equals uniform prior
        "--- Day 1 : 1 ---"
        "Learning Rate: Beta Distribution"
        lr_day1_1_alpha = torch.tensor([self.lr_day1_1_alpha]).to(device)
        lr_day1_1_beta = torch.tensor([self.lr_day1_1_beta]).to(device)
        lr_day1_1 = pyro.sample('lr_day1_1', dist.Beta(lr_day1_1_alpha, lr_day1_1_beta)).to(device)
        
        "Θ_Q"
        theta_Q_day1_1_conc = torch.tensor([self.theta_Q_day1_1_conc]).to(device)
        theta_Q_day1_1_rate = torch.tensor([self.theta_Q_day1_1_rate]).to(device)
        theta_Q_day1_1 = pyro.sample('theta_Q_day1_1', dist.Gamma(theta_Q_day1_1_conc, theta_Q_day1_1_rate)).to(device)
        
        "Θ_rep"
        theta_rep_day1_1_conc = torch.tensor([self.theta_rep_day1_1_conc]).to(device)
        theta_rep_day1_1_rate = torch.tensor([self.theta_rep_day1_1_rate]).to(device)
        theta_rep_day1_1 = pyro.sample('theta_rep_day1_1', dist.Gamma(theta_rep_day1_1_conc, theta_rep_day1_1_rate)).to(device)
        
        "--- Day 1 : 2 ---"
        "Learning Rate: Beta Distribution"
        lr_day1_2_alpha = torch.tensor([self.lr_day1_2_alpha]).to(device)
        lr_day1_2_beta = torch.tensor([self.lr_day1_2_beta]).to(device)
        lr_day1_2 = pyro.sample('lr_day1_2', dist.Beta(lr_day1_2_alpha, lr_day1_2_beta)).to(device)
        
        "Θ_Q"
        theta_Q_day1_2_conc = torch.tensor([self.theta_Q_day1_2_conc]).to(device)
        theta_Q_day1_2_rate = torch.tensor([self.theta_Q_day1_2_rate]).to(device)
        theta_Q_day1_2 = pyro.sample('theta_Q_day1_2', dist.Gamma(theta_Q_day1_2_conc, theta_Q_day1_2_rate)).to(device)
        
        "Θ_rep"
        theta_rep_day1_2_conc = torch.tensor([self.theta_rep_day1_2_conc]).to(device)
        theta_rep_day1_2_rate = torch.tensor([self.theta_rep_day1_2_rate]).to(device)
        theta_rep_day1_2 = pyro.sample('theta_rep_day1_2', dist.Gamma(theta_rep_day1_2_conc, theta_rep_day1_2_rate)).to(device)
        
        "--- Day 2 ---"
        "Learning Rate: Beta Distribution"
        lr_day2_alpha = torch.tensor([self.lr_day2_alpha]).to(device)
        lr_day2_beta = torch.tensor([self.lr_day2_beta]).to(device)
        lr_day2 = pyro.sample('lr_day2', dist.Beta(lr_day2_alpha, lr_day2_beta)).to(device)
        
        "Θ_Q"
        theta_Q_day2_conc = torch.tensor([self.theta_Q_day2_conc]).to(device)
        theta_Q_day2_rate = torch.tensor([self.theta_Q_day2_rate]).to(device)
        theta_Q_day2 = pyro.sample('theta_Q_day2', dist.Gamma(theta_Q_day2_conc, theta_Q_day2_rate)).to(device)
        
        "Θ_rep"
        theta_rep_day2_conc = torch.tensor([self.theta_rep_day2_conc]).to(device)
        theta_rep_day2_rate = torch.tensor([self.theta_rep_day2_rate]).to(device)
        theta_rep_day2 = pyro.sample('theta_rep_day2', dist.Gamma(theta_rep_day2_conc, theta_rep_day2_rate)).to(device)
                
        "K: Gamma distribution"
        #conc_k = torch.tensor([2]).to(device)        
        #rate_k = torch.tensor([4]).to(device)        
        #k = pyro.sample('k', dist.Gamma(conc_k, rate_k)).to(device)
        k=torch.tensor([self.k])
        
        # parameter shape for inference (until now): [1] or [n_particles]
        param_dict = {"lr_day1_1": lr_day1_1[:, None], \
                      "lr_day1_2": lr_day1_2[:, None], \
                      "lr_day2": lr_day2[:, None], \
                      "theta_Q_day1_1": theta_Q_day1_1[None, :], \
                      "theta_Q_day1_2": theta_Q_day1_2[None, :], \
                      "theta_Q_day2": theta_Q_day2[None, :], \
                      "theta_rep_day1_1": theta_rep_day1_1[None, :], \
                      "theta_rep_day1_2": theta_rep_day1_2[None, :], \
                      "theta_rep_day2": theta_rep_day2[None, :], \
                      "k": k[:, None]}

        # lr is shape [1,1] or [n_particles, 1]
        self.agent.reset(**param_dict)
        
        t = -1
        for tau in range(self.trials):
            
            trial = self.data["Trialsequence"][tau]
            blocktype = self.data["Blocktype"][tau]
            
            if self.data["Blockidx"][tau] <= 1:
                exp_part = 1
                
            elif self.data["Blockidx"][tau] > 1 and self.data["Blockidx"][tau] <= 5:
                exp_part = 2
                
            elif self.data["Blockidx"][tau] > 5:
                exp_part = 3
                
            else:
                raise Exception("Da isch a Fehla!")
            
            if trial == -1:
                "Beginning of new block"
                self.agent.update(-1,-1,-1, exp_part=exp_part, trialstimulus=trial)
                
            else:
                current_choice = self.data["Choices"][tau]
                outcome = self.data["Outcomes"][tau]
                
            if trial > 10:
                "Dual-Target Trial"
                t+=1
                #assert(isinstance(trial, list))
                option1, option2 = self.agent.find_resp_options(trial)

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
                self.agent.update(current_choice, outcome, blocktype, exp_part=exp_part, trialstimulus=trial)

            "Sample if dual-target trial and no error was performed"
            if trial > 10 and current_choice > -1:
                # choice shape for inference: [1]
                pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), obs=choice)
                    
                    
    def guide(self):
        # approximate posterior. assume MF: each param has its own univariate Normal.

        "--- Day 1 : 1 ---"
        "Learning Rate"
        lr_day1_1_alpha = pyro.param("lr_day1_1_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day1_1_beta = pyro.param("lr_day1_1_beta", torch.tensor([20.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day1_1 = pyro.sample('lr_day1_1', dist.Beta(lr_day1_1_alpha, lr_day1_1_beta)).to(device)

        "Θ_Q: Gamma Distribution"
        theta_Q_day1_1_conc = pyro.param("theta_Q_day1_1_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q_day1_1_rate = pyro.param("theta_Q_day1_1_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q_day1_1 = pyro.sample('theta_Q_day1_1', dist.Gamma(theta_Q_day1_1_conc, theta_Q_day1_1_rate)).to(device)
        
        "Θ_rep: Gamma Distribution"
        theta_rep_day1_1_conc = pyro.param("theta_rep_day1_1_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep_day1_1_rate = pyro.param("theta_rep_day1_1_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep_day1_1 = pyro.sample('theta_rep_day1_1', dist.Gamma(theta_rep_day1_1_conc, theta_rep_day1_1_rate)).to(device)
        
        "--- Day 1 : 2 ---"
        "Learning Rate"
        lr_day1_2_alpha = pyro.param("lr_day1_2_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day1_2_beta = pyro.param("lr_day1_2_beta", torch.tensor([20.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day1_2 = pyro.sample('lr_day1_2', dist.Beta(lr_day1_2_alpha, lr_day1_2_beta)).to(device)

        "Θ_Q: Gamma Distribution"
        theta_Q_day1_2_conc = pyro.param("theta_Q_day1_2_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q_day1_2_rate = pyro.param("theta_Q_day1_2_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q_day1_2 = pyro.sample('theta_Q_day1_2', dist.Gamma(theta_Q_day1_2_conc, theta_Q_day1_2_rate)).to(device)
        
        "Θ_rep: Gamma Distribution"
        theta_rep_day1_2_conc = pyro.param("theta_rep_day1_2_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep_day1_2_rate = pyro.param("theta_rep_day1_2_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep_day1_2 = pyro.sample('theta_rep_day1_2', dist.Gamma(theta_rep_day1_2_conc, theta_rep_day1_2_rate)).to(device)

        " --- Day 2 ---"
        "Learning Rate"
        lr_day2_alpha = pyro.param("lr_day2_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day2_beta = pyro.param("lr_day2_beta", torch.tensor([20.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        lr_day2 = pyro.sample('lr_day2', dist.Beta(lr_day2_alpha, lr_day2_beta)).to(device)

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

        param_dict = {"lr_day1_1_alpha": lr_day1_1_alpha, \
                      "lr_day1_1_beta": lr_day1_1_beta, \
                      "lr_day1_1": lr_day1_1, \
                      "theta_Q_day1_1_conc": theta_Q_day1_1_conc, \
                      "theta_Q_day1_1_rate": theta_Q_day1_1_rate, \
                      "theta_Q_day1_1": theta_Q_day1_1, \
                      "theta_rep_day1_1_conc": theta_rep_day1_1_conc, \
                      "theta_rep_day1_1_rate": theta_rep_day1_1_rate, \
                      "theta_rep_day1_1": theta_rep_day1_1, \
                        
                      "lr_day1_2_alpha": lr_day1_2_alpha, \
                      "lr_day1_2_beta": lr_day1_2_beta, \
                      "lr_day1_2": lr_day1_2, \
                      "theta_Q_day1_2_conc": theta_Q_day1_2_conc, \
                      "theta_Q_day1_2_rate": theta_Q_day1_2_rate, \
                      "theta_Q_day1_2": theta_Q_day1_2, \
                      "theta_rep_day1_2_conc": theta_rep_day1_2_conc, \
                      "theta_rep_day1_2_rate": theta_rep_day1_2_rate, \
                      "theta_rep_day1_2": theta_rep_day1_2, \

                      "lr_day2_alpha": lr_day2_alpha, \
                      "lr_day2_beta": lr_day2_beta, \
                      "lr_day2": lr_day2, \
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
        
        "--- Day 1 : 1 ---"
        lr_day1_1_alpha = pyro.param("lr_day1_1_alpha").data.numpy()
        lr_day1_1_beta = pyro.param("lr_day1_1_beta").data.numpy()
        
        theta_Q_day1_1_conc = pyro.param("theta_Q_day1_1_conc").data.numpy()
        theta_Q_day1_1_rate = pyro.param("theta_Q_day1_1_rate").data.numpy()
        
        theta_rep_day1_1_conc = pyro.param("theta_rep_day1_1_conc").data.numpy()
        theta_rep_day1_1_rate = pyro.param("theta_rep_day1_1_rate").data.numpy()
        
        "--- Day 1 : 2 ---"
        lr_day1_2_alpha = pyro.param("lr_day1_2_alpha").data.numpy()
        lr_day1_2_beta = pyro.param("lr_day1_2_beta").data.numpy()
        
        theta_Q_day1_2_conc = pyro.param("theta_Q_day1_2_conc").data.numpy()
        theta_Q_day1_2_rate = pyro.param("theta_Q_day1_2_rate").data.numpy()
        
        theta_rep_day1_2_conc = pyro.param("theta_rep_day1_2_conc").data.numpy()
        theta_rep_day1_2_rate = pyro.param("theta_rep_day1_2_rate").data.numpy()


        "--- Day 2 ---"
        lr_day2_alpha = pyro.param("lr_day2_alpha").data.numpy()
        lr_day2_beta = pyro.param("lr_day2_beta").data.numpy()
        
        theta_Q_day2_conc = pyro.param("theta_Q_day2_conc").data.numpy()
        theta_Q_day2_rate = pyro.param("theta_Q_day2_rate").data.numpy()
        
        theta_rep_day2_conc = pyro.param("theta_rep_day2_conc").data.numpy()
        theta_rep_day2_rate = pyro.param("theta_rep_day2_rate").data.numpy()

        param_dict = {"lr_day1_1_alpha": lr_day1_1_alpha, \
                      "lr_day1_1_beta": lr_day1_1_beta, \
                      "theta_Q_day1_1_conc": theta_Q_day1_1_conc, \
                      "theta_Q_day1_1_rate": theta_Q_day1_1_rate, \
                      "theta_rep_day1_1_conc": theta_rep_day1_1_conc, \
                      "theta_rep_day1_1_rate": theta_rep_day1_1_rate, \

                      "lr_day1_2_alpha": lr_day1_2_alpha, \
                      "lr_day1_2_beta": lr_day1_2_beta, \
                      "theta_Q_day1_2_conc": theta_Q_day1_2_conc, \
                      "theta_Q_day1_2_rate": theta_Q_day1_2_rate, \
                      "theta_rep_day1_2_conc": theta_rep_day1_2_conc, \
                      "theta_rep_day1_2_rate": theta_rep_day1_2_rate, \
                          
                      "lr_day2_alpha": lr_day2_alpha, \
                      "lr_day2_beta": lr_day2_beta, \
                      "theta_Q_day2_conc": theta_Q_day2_conc, \
                      "theta_Q_day2_rate": theta_Q_day2_rate, \
                      "theta_rep_day2_conc": theta_rep_day2_conc, \
                      "theta_rep_day2_rate": theta_rep_day2_rate}

        return self.loss, param_dict
    
class SingleInference_modelB_3(object):

    def __init__(self, agent, data, k, **kwargs):

        self.agent = agent
        self.trials = self.agent.trials # length of experiment
        #self.T = agent.T # No. of "subtrials" -> unnötig
        self.data = data # Dictionary
        
        self.k = k
        
        if 'lr_day1_1_alpha' in kwargs:
            pass
        else:
            "--- Day 1 : 1 ---"
            "Θ_Q"
            self.theta_Q_day1_1_conc = 4.
            self.theta_Q_day1_1_rate = 4.
            
            "Θ_rep"
            self.theta_rep_day1_1_conc = 4.
            self.theta_rep_day1_1_rate = 4.
            
            "--- Day 1 : 2 ---"
            "Θ_Q"
            self.theta_Q_day1_2_conc = 4.
            self.theta_Q_day1_2_rate = 4.
            
            "Θ_rep"
            self.theta_rep_day1_2_conc = 4.
            self.theta_rep_day1_2_rate = 4.
            
            "--- Day 2 ---"
            "Θ_Q"
            self.theta_Q_day2_conc = 4.
            self.theta_Q_day2_rate = 4.
            
            "Θ_rep"
            self.theta_rep_day2_conc = 4.
            self.theta_rep_day2_rate = 4.
            

    def model(self):
        # generative model of behavior with Normally distributed params (within subject!!)

        # tell pyro about prior over parameters: alpha and beta of lr which is between 0 and 1
        # alpha = beta = 1 equals uniform prior
        "--- Day 1 : 1 ---"
        "Θ_Q"
        theta_Q_day1_1_conc = torch.tensor([self.theta_Q_day1_1_conc]).to(device)
        theta_Q_day1_1_rate = torch.tensor([self.theta_Q_day1_1_rate]).to(device)
        theta_Q_day1_1 = pyro.sample('theta_Q_day1_1', dist.Gamma(theta_Q_day1_1_conc, theta_Q_day1_1_rate)).to(device)
        
        "Θ_rep"
        theta_rep_day1_1_conc = torch.tensor([self.theta_rep_day1_1_conc]).to(device)
        theta_rep_day1_1_rate = torch.tensor([self.theta_rep_day1_1_rate]).to(device)
        theta_rep_day1_1 = pyro.sample('theta_rep_day1_1', dist.Gamma(theta_rep_day1_1_conc, theta_rep_day1_1_rate)).to(device)
        
        "--- Day 1 : 2 ---"
        "Θ_Q"
        theta_Q_day1_2_conc = torch.tensor([self.theta_Q_day1_2_conc]).to(device)
        theta_Q_day1_2_rate = torch.tensor([self.theta_Q_day1_2_rate]).to(device)
        theta_Q_day1_2 = pyro.sample('theta_Q_day1_2', dist.Gamma(theta_Q_day1_2_conc, theta_Q_day1_2_rate)).to(device)
        
        "Θ_rep"
        theta_rep_day1_2_conc = torch.tensor([self.theta_rep_day1_2_conc]).to(device)
        theta_rep_day1_2_rate = torch.tensor([self.theta_rep_day1_2_rate]).to(device)
        theta_rep_day1_2 = pyro.sample('theta_rep_day1_2', dist.Gamma(theta_rep_day1_2_conc, theta_rep_day1_2_rate)).to(device)
        
        "--- Day 2 ---"
        "Θ_Q"
        theta_Q_day2_conc = torch.tensor([self.theta_Q_day2_conc]).to(device)
        theta_Q_day2_rate = torch.tensor([self.theta_Q_day2_rate]).to(device)
        theta_Q_day2 = pyro.sample('theta_Q_day2', dist.Gamma(theta_Q_day2_conc, theta_Q_day2_rate)).to(device)
        
        "Θ_rep"
        theta_rep_day2_conc = torch.tensor([self.theta_rep_day2_conc]).to(device)
        theta_rep_day2_rate = torch.tensor([self.theta_rep_day2_rate]).to(device)
        theta_rep_day2 = pyro.sample('theta_rep_day2', dist.Gamma(theta_rep_day2_conc, theta_rep_day2_rate)).to(device)
                
        "K: Gamma distribution"
        #conc_k = torch.tensor([2]).to(device)        
        #rate_k = torch.tensor([4]).to(device)        
        #k = pyro.sample('k', dist.Gamma(conc_k, rate_k)).to(device)
        k=torch.tensor([self.k])
        
        # parameter shape for inference (until now): [1] or [n_particles]
        param_dict = {"theta_Q_day1_1": theta_Q_day1_1[None, :], \
                      "theta_Q_day1_2": theta_Q_day1_2[None, :], \
                      "theta_Q_day2": theta_Q_day2[None, :], \
                      "theta_rep_day1_1": theta_rep_day1_1[None, :], \
                      "theta_rep_day1_2": theta_rep_day1_2[None, :], \
                      "theta_rep_day2": theta_rep_day2[None, :], \
                      "k": k[:, None]}

        # lr is shape [1,1] or [n_particles, 1]
        self.agent.reset(**param_dict)
        
        t = -1
        for tau in range(self.trials):
            
            trial = self.data["Trialsequence"][tau]
            blocktype = self.data["Blocktype"][tau]
            
            if self.data["Blockidx"][tau] <= 1:
                exp_part = 1
                
            elif self.data["Blockidx"][tau] > 1 and self.data["Blockidx"][tau] <= 5:
                exp_part = 2
                
            elif self.data["Blockidx"][tau] > 5:
                exp_part = 3
                
            else:
                raise Exception("Da isch a Fehla!")
            
            if trial == -1:
                "Beginning of new block"
                self.agent.update(-1,-1,-1, exp_part=exp_part, trialstimulus=trial)
                
            else:
                current_choice = self.data["Choices"][tau]
                outcome = self.data["Outcomes"][tau]
                
            if trial > 10:
                "Dual-Target Trial"
                t+=1
                #assert(isinstance(trial, list))
                option1, option2 = self.agent.find_resp_options(trial)

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
                self.agent.update(current_choice, outcome, blocktype, exp_part=exp_part, trialstimulus=trial)

            "Sample if dual-target trial and no error was performed"
            if trial > 10 and current_choice > -1:
                # choice shape for inference: [1]
                pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), obs=choice)
                    
                    
    def guide(self):
        # approximate posterior. assume MF: each param has its own univariate Normal.

        "--- Day 1 : 1 ---"
        "Θ_Q: Gamma Distribution"
        theta_Q_day1_1_conc = pyro.param("theta_Q_day1_1_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q_day1_1_rate = pyro.param("theta_Q_day1_1_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q_day1_1 = pyro.sample('theta_Q_day1_1', dist.Gamma(theta_Q_day1_1_conc, theta_Q_day1_1_rate)).to(device)
        
        "Θ_rep: Gamma Distribution"
        theta_rep_day1_1_conc = pyro.param("theta_rep_day1_1_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep_day1_1_rate = pyro.param("theta_rep_day1_1_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep_day1_1 = pyro.sample('theta_rep_day1_1', dist.Gamma(theta_rep_day1_1_conc, theta_rep_day1_1_rate)).to(device)
        
        "--- Day 1 : 2 ---"
        "Θ_Q: Gamma Distribution"
        theta_Q_day1_2_conc = pyro.param("theta_Q_day1_2_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q_day1_2_rate = pyro.param("theta_Q_day1_2_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q_day1_2 = pyro.sample('theta_Q_day1_2', dist.Gamma(theta_Q_day1_2_conc, theta_Q_day1_2_rate)).to(device)
        
        "Θ_rep: Gamma Distribution"
        theta_rep_day1_2_conc = pyro.param("theta_rep_day1_2_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep_day1_2_rate = pyro.param("theta_rep_day1_2_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep_day1_2 = pyro.sample('theta_rep_day1_2', dist.Gamma(theta_rep_day1_2_conc, theta_rep_day1_2_rate)).to(device)

        " --- Day 2 ---"
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

        param_dict = {"theta_Q_day1_1_conc": theta_Q_day1_1_conc, \
                      "theta_Q_day1_1_rate": theta_Q_day1_1_rate, \
                      "theta_Q_day1_1": theta_Q_day1_1, \
                      "theta_rep_day1_1_conc": theta_rep_day1_1_conc, \
                      "theta_rep_day1_1_rate": theta_rep_day1_1_rate, \
                      "theta_rep_day1_1": theta_rep_day1_1, \
                        
                      "theta_Q_day1_2_conc": theta_Q_day1_2_conc, \
                      "theta_Q_day1_2_rate": theta_Q_day1_2_rate, \
                      "theta_Q_day1_2": theta_Q_day1_2, \
                      "theta_rep_day1_2_conc": theta_rep_day1_2_conc, \
                      "theta_rep_day1_2_rate": theta_rep_day1_2_rate, \
                      "theta_rep_day1_2": theta_rep_day1_2, \
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
        
        "--- Day 1 : 1 ---"
        theta_Q_day1_1_conc = pyro.param("theta_Q_day1_1_conc").data.numpy()
        theta_Q_day1_1_rate = pyro.param("theta_Q_day1_1_rate").data.numpy()
        
        theta_rep_day1_1_conc = pyro.param("theta_rep_day1_1_conc").data.numpy()
        theta_rep_day1_1_rate = pyro.param("theta_rep_day1_1_rate").data.numpy()
        
        "--- Day 1 : 2 ---"
        theta_Q_day1_2_conc = pyro.param("theta_Q_day1_2_conc").data.numpy()
        theta_Q_day1_2_rate = pyro.param("theta_Q_day1_2_rate").data.numpy()
        
        theta_rep_day1_2_conc = pyro.param("theta_rep_day1_2_conc").data.numpy()
        theta_rep_day1_2_rate = pyro.param("theta_rep_day1_2_rate").data.numpy()


        "--- Day 2 ---"
        theta_Q_day2_conc = pyro.param("theta_Q_day2_conc").data.numpy()
        theta_Q_day2_rate = pyro.param("theta_Q_day2_rate").data.numpy()
        
        theta_rep_day2_conc = pyro.param("theta_rep_day2_conc").data.numpy()
        theta_rep_day2_rate = pyro.param("theta_rep_day2_rate").data.numpy()

        param_dict = {"theta_Q_day1_1_conc": theta_Q_day1_1_conc, \
                      "theta_Q_day1_1_rate": theta_Q_day1_1_rate, \
                      "theta_rep_day1_1_conc": theta_rep_day1_1_conc, \
                      "theta_rep_day1_1_rate": theta_rep_day1_1_rate, \

                      "theta_Q_day1_2_conc": theta_Q_day1_2_conc, \
                      "theta_Q_day1_2_rate": theta_Q_day1_2_rate, \
                      "theta_rep_day1_2_conc": theta_rep_day1_2_conc, \
                      "theta_rep_day1_2_rate": theta_rep_day1_2_rate, \
                          
                      "theta_Q_day2_conc": theta_Q_day2_conc, \
                      "theta_Q_day2_rate": theta_Q_day2_rate, \
                      "theta_rep_day2_conc": theta_rep_day2_conc, \
                      "theta_rep_day2_rate": theta_rep_day2_rate}

        return self.loss, param_dict

class SingleInference_modelF(object):

    def __init__(self, agent, data, k, **kwargs):

        self.agent = agent
        self.trials = self.agent.trials # length of experiment
        #self.T = agent.T # No. of "subtrials" -> unnötig
        self.data = data # Dictionary
        self.k = k
        
        if 'theta_Qlambda_day1_alpha' in kwargs:
            pass
        
        else:
            "--- Day 1 ---"
            "Θ_Qlambda: Beta Distribution"
            self.theta_Qlambda_day1_alpha = 1.
            self.theta_Qlambda_day1_beta = 80.
    
            "Θ_Q0"
            self.theta_Q0_day1_conc = 1.
            self.theta_Q0_day1_rate = 4000.
            
            "Θ_Qlambda: Beta Distribution"
            self.theta_replambda_day1_alpha = 1.
            self.theta_replambda_day1_beta = 80.
    
            "Θ_rep0"
            self.theta_rep0_day1_conc = 1.
            self.theta_rep0_day1_rate = 4000.
            
            "--- Day 2 ---"
            "Θ_Qlambda: Beta Distribution"
            self.theta_Qlambda_day2_alpha = 1.
            self.theta_Qlambda_day2_beta = 80.
                    
            "Θ_Q0"
            self.theta_Q0_day2_conc = 4.
            self.theta_Q0_day2_rate = 4.
            
            "Θ_Qlambda: Beta Distribution"
            self.theta_replambda_day2_alpha = 1.
            self.theta_replambda_day2_beta = 80.
                    
            "Θ_rep0"
            self.theta_rep0_day2_conc = 4.
            self.theta_rep0_day2_rate = 4.

    def model(self):
        # generative model of behavior with Normally distributed params (within subject!!)

        # tell pyro about prior over parameters: alpha and beta of lr which is between 0 and 1
        # alpha = beta = 1 equals uniform prior
        "--- Day 1 ---"
        "Θ_Qlambda: Beta Distribution"
        theta_Qlambda_day1_alpha = torch.tensor([self.theta_Qlambda_day1_alpha]).to(device)
        theta_Qlambda_day1_beta = torch.tensor([self.theta_Qlambda_day1_beta]).to(device)
        theta_Qlambda_day1 = pyro.sample('theta_Qlambda_day1', dist.Beta(theta_Qlambda_day1_alpha, theta_Qlambda_day1_beta)).to(device)

        "Θ_Q0"
        theta_Q0_day1_conc = torch.tensor([self.theta_Q0_day1_conc]).to(device)
        theta_Q0_day1_rate = torch.tensor([self.theta_Q0_day1_rate]).to(device)
        theta_Q0_day1 = pyro.sample('theta_Q0_day1', dist.Gamma(theta_Q0_day1_conc, theta_Q0_day1_rate)).to(device)
        
        "Θ_Qlambda: Beta Distribution"
        theta_replambda_day1_alpha = torch.tensor([self.theta_replambda_day1_alpha]).to(device)
        theta_replambda_day1_beta = torch.tensor([self.theta_replambda_day1_beta]).to(device)
        theta_replambda_day1 = pyro.sample('theta_replambda_day1', dist.Beta(theta_replambda_day1_alpha, theta_replambda_day1_beta)).to(device)

        "Θ_rep0"
        theta_rep0_day1_conc = torch.tensor([self.theta_rep0_day1_conc]).to(device)
        theta_rep0_day1_rate = torch.tensor([self.theta_rep0_day1_rate]).to(device)
        theta_rep0_day1 = pyro.sample('theta_rep0_day1', dist.Gamma(theta_rep0_day1_conc, theta_rep0_day1_rate)).to(device)
        
        "--- Day 2 ---"
        "Θ_Qlambda: Beta Distribution"
        theta_Qlambda_day2_alpha = torch.tensor([self.theta_Qlambda_day2_alpha]).to(device)
        theta_Qlambda_day2_beta = torch.tensor([self.theta_Qlambda_day2_beta]).to(device)
        theta_Qlambda_day2 = pyro.sample('theta_Qlambda_day2', dist.Beta(theta_Qlambda_day2_alpha, theta_Qlambda_day2_beta)).to(device)
                
        "Θ_Q0"
        theta_Q0_day2_conc = torch.tensor([self.theta_Q0_day2_conc]).to(device)
        theta_Q0_day2_rate = torch.tensor([self.theta_Q0_day2_rate]).to(device)
        theta_Q0_day2 = pyro.sample('theta_Q0_day2', dist.Gamma(theta_Q0_day2_conc, theta_Q0_day2_rate)).to(device)
        
        "Θ_Qlambda: Beta Distribution"
        theta_replambda_day2_alpha = torch.tensor([self.theta_replambda_day2_alpha]).to(device)
        theta_replambda_day2_beta = torch.tensor([self.theta_replambda_day2_beta]).to(device)
        theta_replambda_day2 = pyro.sample('theta_replambda_day2', dist.Beta(theta_replambda_day2_alpha, theta_replambda_day2_beta)).to(device)
                
        "Θ_rep0"
        theta_rep0_day2_conc = torch.tensor([self.theta_rep0_day2_conc]).to(device)
        theta_rep0_day2_rate = torch.tensor([self.theta_rep0_day2_rate]).to(device)
        theta_rep0_day2 = pyro.sample('theta_rep0_day2', dist.Gamma(theta_rep0_day2_conc, theta_rep0_day2_rate)).to(device)

        "K: Gamma distribution"
        k=torch.tensor([self.k])

        # parameter shape for inference (until now): [1] or [n_particles]
        param_dict = {"theta_Qlambda_day1": theta_Qlambda_day1[None,:], \
                      "theta_Q0_day1": theta_Q0_day1[None, :], \
                      "theta_replambda_day1": theta_replambda_day1[None, :], \
                      "theta_rep0_day1": theta_rep0_day1[None, :], \
                      "theta_Qlambda_day2": theta_Qlambda_day2[None,:], \
                      "theta_Q0_day2": theta_Q0_day2[None, :], \
                      "theta_replambda_day2": theta_replambda_day2[None, :], \
                      "theta_rep0_day2": theta_rep0_day2[None, :], \
                      "k": k[:, None]}

        
        # lr is shape [1,1] or [n_particles, 1]
        self.agent.reset(**param_dict)
        
        t = -1 # For sampling below
        t_day1 = -1 # For time-dependent model parameter on day 1
        t_day2 = -1 # For time-dependent model parameter on day 2
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
                self.agent.update(-1,-1,-1, day=day, trialstimulus=trial, t = 0)
                
            else:
                current_choice = self.data["Choices"][tau]
                outcome = self.data["Outcomes"][tau]

            if trial > 10:
                "Dual-Target Trial"
                t+=1
                if day == 1:
                    t_day1 += 1
                    
                elif day == 2:
                    t_day2 += 1
                #assert(isinstance(trial, list))
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
                if day == 1:
                    self.agent.update(current_choice, outcome, blocktype, day=day, trialstimulus=trial, t=t_day1)
                    
                elif day == 2:
                    self.agent.update(current_choice, outcome, blocktype, day=day, trialstimulus=trial, t=t_day2)

            "Sample if dual-target trial and no error was performed"
            if trial > 10 and current_choice > -1:
                # choice shape for inference: [1]
                pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), obs=choice)
                    
    def guide(self):
        # approximate posterior. assume MF: each param has its own univariate Normal.

        "--- Day 1 ---"
        "Θ_Qlambda"
        theta_Qlambda_day1_alpha = pyro.param("theta_Qlambda_day1_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Qlambda_day1_beta = pyro.param("theta_Qlambda_day1_beta", torch.tensor([80.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Qlambda_day1 = pyro.sample('theta_Qlambda_day1', dist.Beta(theta_Qlambda_day1_alpha, theta_Qlambda_day1_beta)).to(device)

        "Θ_Q0: Gamma Distribution"
        theta_Q0_day1_conc = pyro.param("theta_Q0_day1_conc", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q0_day1_rate = pyro.param("theta_Q0_day1_rate", torch.tensor([4000.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q0_day1 = pyro.sample('theta_Q0_day1', dist.Gamma(theta_Q0_day1_conc, theta_Q0_day1_rate)).to(device)

        "Θ_replambda"
        theta_replambda_day1_alpha = pyro.param("theta_replambda_day1_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_replambda_day1_beta = pyro.param("theta_replambda_day1_beta", torch.tensor([80.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_replambda_day1 = pyro.sample('theta_replambda_day1', dist.Beta(theta_replambda_day1_alpha, theta_replambda_day1_beta)).to(device)

        "Θ_rep0: Gamma Distribution"
        theta_rep0_day1_conc = pyro.param("theta_rep0_day1_conc", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep0_day1_rate = pyro.param("theta_rep0_day1_rate", torch.tensor([4000.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_rep0_day1 = pyro.sample('theta_rep0_day1', dist.Gamma(theta_rep0_day1_conc, theta_rep0_day1_rate)).to(device)

        " --- Day 2 ---"
        "Θ_Qlambda"
        theta_Qlambda_day2_alpha = pyro.param("theta_Qlambda_day2_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Qlambda_day2_beta = pyro.param("theta_Qlambda_day2_beta", torch.tensor([80.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Qlambda_day2 = pyro.sample('theta_Qlambda_day2', dist.Beta(theta_Qlambda_day2_alpha, theta_Qlambda_day2_beta)).to(device)

        "Θ_Q0: Gamma Distribution"
        theta_Q0_day2_conc = pyro.param("theta_Q0_day2_conc", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q0_day2_rate = pyro.param("theta_Q0_day2_rate", torch.tensor([4.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_Q0_day2 = pyro.sample('theta_Q0_day2', dist.Gamma(theta_Q0_day2_conc, theta_Q0_day2_rate)).to(device)
        
        "Θ_replambda"
        theta_replambda_day2_alpha = pyro.param("theta_replambda_day2_alpha", torch.tensor([1.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        theta_replambda_day2_beta = pyro.param("theta_replambda_day2_beta", torch.tensor([80.]), constraint=torch.distributions.constraints.positive).to(device)#greater_than_eq(1.))
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

class GroupInference(object):

    def __init__(self, agents, group_data):
        """
        Group inference for original model
        
        agents : list of agents
        group_data : list of data dicts 
        """
        self.n_subjects = len(agents) # no. of participants
        self.agents = agents
        # self.n_subjects = len(data["Participant"].unique())
        self.trials = self.agents[0].trials # length of experiment
        #self.T = agent.T # No. of "subtrials" -> unnötig
        self.group_data = group_data # list of dictionaries                

    def model_group(self):
    
        npar = 3  # number of parameters
    
        # define hyper priors over model parameters
        # prior over sigma of a Gaussian is a Gamma distribution
        a = pyro.param('a', torch.ones(npar), constraint=dist.constraints.positive)
        lam = pyro.param('lam', torch.ones(npar), constraint=dist.constraints.positive)
        tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1)) # Why a/lam?
        
        sig = 1/torch.sqrt(tau) # Gaus sigma

        # each model parameter has a hyperprior defining group level mean
        # in the form of a Normal distribution
        m = pyro.param('m', torch.zeros(npar))
        s = pyro.param('s', torch.ones(npar), constraint=dist.constraints.positive)
        mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(1)) # Gauss mu, wieso s*sig?

        # in order to implement groups, where each subject is independent of the others, pyro uses so-called plates.
        # you embed what should be done for each subject into the "with pyro.plate" context
        # the plate vectorizes subjects and adds an additional dimension onto all arrays/tensors
        # i.e. p1 below will have the length n_subjects
        with pyro.plate('subject', self.n_subjects) as ind:

            # draw parameters from Normal and transform (for numeric trick reasons)
            base_dist = dist.Normal(0., 1.).expand_by([npar]).to_event(1)
            transform = dist.transforms.AffineTransform(mu, sig)
            locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))

            # map the values in locs (between -inf and inf) onto the relevant space
            # here to between 0 and 1 using a sigmoid
            # Is this where my agent would come in?
            lr = torch.sigmoid(locs[..., 0])
            omega = torch.sigmoid(locs[..., 1])
            dectemp = torch.exp(locs[..., 2])

            "K: Gamma distribution"
            k=torch.tensor([4.])
            
            #assert(lr.ndim == 1 or lr.ndim == 2)
            
            if lr.ndim == 2:
                n_particles = lr.shape[0]
            else:
                n_particles = 1
            
            # parameter shape for group inference (until now): [n_subjects] or [n_particles, n_subjects]
            if lr.ndim == 1:
                for pb in range(self.n_subjects):
                    param_dict = {"lr": lr[[pb]][:,None], "omega": omega[[pb]][None, :], "dectemp": dectemp[[pb]][:, None], "k": k[:, None]}
                    # lr should have shape [1,1], or [n_particles, 1]
                    self.agents[pb].reset(**param_dict)
                    
            elif lr.ndim == 2:
                for pb in range(self.n_subjects):
                    param_dict = {"lr": lr[:,pb][:,None], "omega": omega[:,pb][None, :], "dectemp": dectemp[:,pb][:, None], "k": k[:, None]}
                    self.agents[pb].reset(**param_dict)
            
            t = -1
            for tau in range(self.trials):
                
                probs = torch.ones(n_particles, self.n_subjects, 2)
                choices = torch.ones(self.n_subjects)
                obs_mask = torch.ones(self.n_subjects, dtype=torch.bool)
                
                # start_time = time.time()
                for pb in range(self.n_subjects):
                    trial = self.group_data[pb]["Trialsequence"][tau]
                    blocktype = self.group_data[pb]["Blocktype"][tau]
                    
                    if trial == -1:
                        "Beginning of new block"
                        self.agents[pb].update(-1,-1,-1)
                        
                    else:
                        current_choice = self.group_data[pb]["Choices"][tau]
                        outcome = self.group_data[pb]["Outcomes"][tau]
                        
                    if trial > 10:
                        "Dual-Target Trial"
                        t+=1
                        #assert(isinstance(trial, list))
                        option1, option2 = self.agents[pb].find_resp_options(trial)
                        
                        # probs should have shape [n_particles, n_subjects, nactions], or [n_subjects, nactions]
                        # RHS comes out as [1, n_actions] or [n_particles, n_actions]
                        probs[:, pb, :] = self.agents[pb].softmax(torch.cat(([self.agents[pb].V[-1][:, option1][:, None], self.agents[pb].V[-1][:, option2][:, None]]),dim=-1))

                        if current_choice == option1:
                            choices[pb] = torch.tensor([0])
                            obs_mask[pb] = True

                        elif current_choice == option2:
                            choices[pb] = torch.tensor([1])
                            obs_mask[pb] = True
                            
                        elif current_choice == -10:
                            "Error"
                            obs_mask[pb] = False
                            
                        else:
                            raise Exception("Da isch a Fehla")

                    if trial != -1:
                        "Update (trial == -1 means this is the beginning of a block -> participants didn't see this trial')"
                        self.agents[pb].update(current_choice, outcome, blocktype)

                "Sample if dual-target trial and no error was performed"
                if trial > 10:
                    # obs & obs_mask shape for inference: [n_subjects] 
                    # probs shape for inference: [n_subjects, nactions] or [n_particles, n_subjects, nactions]
                    pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), obs=choices, obs_mask = obs_mask)
                    
    def guide_group(self):
        # number of parameters: 2, p1 and p2
        npar = 3
        trns = torch.distributions.biject_to(dist.constraints.positive)
    
        # define mean vector and covariance matrix of multivariate normal
        m_hyp = pyro.param('m_hyp', torch.zeros(2*npar))
        st_hyp = pyro.param('scale_tril_hyp',
                       torch.eye(2*npar),
                       constraint=dist.constraints.lower_cholesky)
        
        # set hyperprior to be multivariate normal
        hyp = pyro.sample('hyp',
                     dist.MultivariateNormal(m_hyp, scale_tril=st_hyp),
                     infer={'is_auxiliary': True})
    
        unc_mu = hyp[..., :npar]
        unc_tau = hyp[..., npar:]
    
        c_tau = trns(unc_tau)

        ld_tau = trns.inv.log_abs_det_jacobian(c_tau, unc_tau)
        ld_tau = dist.util.sum_rightmost(ld_tau, ld_tau.dim() - c_tau.dim() + 1)

        # some numerics tricks
        mu = pyro.sample("mu", dist.Delta(unc_mu, event_dim=1))
        tau = pyro.sample("tau", dist.Delta(c_tau, log_density=ld_tau, event_dim=1))
    
        m_locs = pyro.param('m_locs', torch.zeros(self.n_subjects, npar))
        st_locs = pyro.param('scale_tril_locs',
                        torch.eye(npar).repeat(self.n_subjects, 1, 1),
                        constraint=dist.constraints.lower_cholesky)
    
        with pyro.plate('subject', self.n_subjects):
            # sample unconstrained parameters from multivariate normal
            locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))
    
            lr = torch.sigmoid(locs[..., 0])
            omega = torch.sigmoid(locs[..., 1])
            dectemp = torch.exp(locs[..., 2])
    
            # make dictionary to be able to return samples later
            var_dict = {'lr': lr, 'omega': omega, 'dectemp': dectemp}

            return var_dict

    def infer_posterior(self,
                        iter_steps=1_000,
                        num_particles=10,
                        optim_kwargs={'lr': .01}): # Adam learning rate
        """Perform SVI over free model parameters."""

        pyro.clear_param_store()

        svi = pyro.infer.SVI(model=self.model_group,
                  guide=self.guide_group,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  # set below to true once code is vectorized
                                  vectorize_particles=False))

        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:#range(iter_steps):
            "Runs through the model twice the first time"
            loss.append(torch.tensor(svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
            if torch.isnan(loss[-1]):
                break

        self.loss = [l.cpu() for l in loss] # = -ELBO (Plotten!)
        
        m_locs = pyro.param("m_locs").data.numpy()
        st_locs = pyro.param("scale_tril_locs").data.numpy()

        m_hyp = pyro.param("m_hyp").data.numpy()
        st_hyp = pyro.param("scale_tril_hyp").data.numpy()

        param_dict = {"m_locs": m_locs, "st_locs": st_locs, "m_hyp": m_hyp, \
                      "st_hyp": st_hyp}

        return self.loss, param_dict
    
    def sample_posterior(self, n_samples=5_000):
        # keys = ["lamb_pi", "lamb_r", "h", "dec_temp"]
    
        lr_global = np.zeros((n_samples, self.n_subjects))
        omega_global = np.zeros((n_samples, self.n_subjects))
        dectemp_global = np.zeros((n_samples, self.n_subjects))

        # sample p1 and p2 from guide (the posterior over ps). 
        # Calling the guide yields samples from the posterior after SVI has run.
        for i in range(n_samples):
            sample = self.guide_group()
            for key in sample.keys():
                sample.setdefault(key, torch.ones(1))
                
            lr = sample["lr"]
            omega = sample["omega"]
            dectemp = sample["dectemp"]
    
            lr_global[i] = lr.detach().numpy()
            omega_global[i] = omega.detach().numpy()
            dectemp_global[i] = dectemp.detach().numpy()
    
        # do some data formatting steps
        lr_flat = np.array([lr_global[i,n] for i in range(n_samples) for n in range(self.n_subjects)])
        omega_flat = np.array([omega_global[i,n] for i in range(n_samples) for n in range(self.n_subjects)])
        dectemp_flat = np.array([dectemp_global[i,n] for i in range(n_samples) for n in range(self.n_subjects)])
    
        subs_flat = np.array([n for i in range(n_samples) for n in range(self.n_subjects)])
    
        sample_dict = {"lr": lr_flat, "omega": omega_flat, "dectemp": dectemp_flat, "subject": subs_flat}
    
        # make a pandas dataframe, better for analyses and plotting later (pandas is pythons R equivalent)
        sample_df = pd.DataFrame(sample_dict)
    
        return sample_df
    
class GroupInference_modelB(object):

    def __init__(self, agents, group_data):
        """
        Group inference for original model
        
        agents : list of agents
        group_data : list of data dicts 
        """
        self.n_subjects = len(agents) # no. of participants
        self.agents = agents
        # self.n_subjects = len(data["Participant"].unique())
        self.trials = self.agents[0].trials # length of experiment
        #self.T = agent.T # No. of "subtrials" -> unnötig
        self.group_data = group_data # list of dictionaries

                
    def model_group(self):
    
        npar = 6  # number of parameters
    
        # define hyper priors over model parameters
        # prior over sigma of a Gaussian is a Gamma distribution
        a = pyro.param('a', torch.ones(npar), constraint=dist.constraints.positive)
        lam = pyro.param('lam', torch.ones(npar), constraint=dist.constraints.positive)
        tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1)) # Why a/lam?
        
        sig = 1/torch.sqrt(tau) # Gaus sigma

        # each model parameter has a hyperprior defining group level mean
        # in the form of a Normal distribution
        m = pyro.param('m', torch.zeros(npar))
        s = pyro.param('s', torch.ones(npar), constraint=dist.constraints.positive)
        mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(1)) # Gauss mu, wieso s*sig?

        # in order to implement groups, where each subject is independent of the others, pyro uses so-called plates.
        # you embed what should be done for each subject into the "with pyro.plate" context
        # the plate vectorizes subjects and adds an additional dimension onto all arrays/tensors
        # i.e. p1 below will have the length n_subjects
        with pyro.plate('subject', self.n_subjects) as ind:

            # draw parameters from Normal and transform (for numeric trick reasons)
            base_dist = dist.Normal(0., 1.).expand_by([npar]).to_event(1)
            transform = dist.transforms.AffineTransform(mu, sig)
            locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))

            # map the values in locs (between -inf and inf) onto the relevant space
            # here to between 0 and 1 using a sigmoid
            # Is this where my agent would come in?
            lr_day1 = torch.sigmoid(locs[..., 0])
            theta_Q_day1 = torch.exp(locs[..., 1])
            theta_rep_day1 = torch.exp(locs[..., 2])
            
            lr_day2 = torch.sigmoid(locs[..., 3])
            theta_Q_day2 = torch.exp(locs[..., 4])
            theta_rep_day2 = torch.exp(locs[..., 5])

            "K: Gamma distribution"
            k=torch.tensor([4.])
            
            if lr_day1.ndim == 2:
                n_particles = lr_day1.shape[0]

            else:
                n_particles = 1

            # parameter shape for group inference (until now): [n_subjects] or [n_particles, n_subjects]
            if lr_day1.ndim == 1:
                for pb in range(self.n_subjects):
                    param_dict = {'lr_day1': lr_day1[[pb]][:,None], \
                                  'theta_Q_day1': theta_Q_day1[[pb]][None, :], \
                                  'theta_rep_day1': theta_rep_day1[[pb]][:, None], \
                                      
                                  'lr_day2': lr_day2[[pb]][:,None], \
                                  'theta_Q_day2': theta_Q_day2[[pb]][None, :], \
                                  'theta_rep_day2': theta_rep_day2[[pb]][:, None], \
                                  'k': k[:, None]}

                    # lr should have shape [1,1], or [n_particles, 1]
                    self.agents[pb].reset(**param_dict)

            elif lr_day1.ndim == 2:
                for pb in range(self.n_subjects):
                    param_dict = {'lr_day1': lr_day1[:,pb][:,None], \
                                  'theta_Q_day1': theta_Q_day1[:,pb][None, :], \
                                  'theta_rep_day1': theta_rep_day1[:,pb][:, None], \
                                      
                                  'lr_day2': lr_day2[:,pb][:,None], \
                                  'theta_Q_day2': theta_Q_day2[:,pb][None, :], \
                                  'theta_rep_day2': theta_rep_day2[:,pb][:, None], \
                                  'k': k[:, None]}
                        
                    self.agents[pb].reset(**param_dict)
                        
            t = -1
            for tau in range(self.trials):

                probs = torch.ones(n_particles, self.n_subjects, 2)
                choices = torch.ones(self.n_subjects)
                obs_mask = torch.ones(self.n_subjects, dtype=torch.bool)
                
                # start_time = time.time()
                for pb in range(self.n_subjects):
                    trial = self.group_data[pb]["Trialsequence"][tau]
                    blocktype = self.group_data[pb]["Blocktype"][tau]
                    
                    if self.group_data[pb]["Blockidx"][tau] <= 5:
                        day = 1
                        
                    elif self.group_data[pb]["Blockidx"][tau] > 5:
                        day = 2
                        
                    else:
                        raise Exception("Da isch a Fehla!")
                    
                    if trial == -1:
                        "Beginning of new block"
                        self.agents[pb].update(-1,-1,-1, day=day, trialstimulus=trial)
                        
                    else:
                        ipdb.set_trace()
                        current_choice = self.group_data[pb]["Choices"][tau]
                        outcome = self.group_data[pb]["Outcomes"][tau]

                    # ipdb.set_trace()
                    if trial > 10:
                        "Dual-Target Trial"
                        t+=1
                        #assert(isinstance(trial, list))
                        option1, option2 = self.agents[pb].find_resp_options(trial)
                        
                        # probs should have shape [n_particles, n_subjects, nactions], or [n_subjects, nactions]
                        # RHS comes out as [1, n_actions] or [n_particles, n_actions]
                        probs[:, pb, :] = self.agents[pb].softmax(torch.cat(([self.agents[pb].V[-1][:, option1][:, None], self.agents[pb].V[-1][:, option2][:, None]]),dim=-1))

                        if current_choice == option1:
                            choices[pb] = torch.tensor([0])
                            obs_mask[pb] = True

                        elif current_choice == option2:
                            choices[pb] = torch.tensor([1])
                            obs_mask[pb] = True
                            
                        elif current_choice == -10:
                            "Error"
                            obs_mask[pb] = False
                            
                        else:
                            raise Exception("Da isch a Fehla")

                    if trial != -1:
                        "Update (trial == -1 means this is the beginning of a block -> participants didn't see this trial')"
                        self.agents[pb].update(current_choice, outcome, blocktype, day=day, trialstimulus=trial)

                "Sample if dual-target trial and no error was performed"
                if trial > 10:
                    # obs & obs_mask shape for inference: [n_subjects] 
                    # probs shape for inference: [n_subjects, nactions] or [n_particles, n_subjects, nactions]
                    pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), obs=choices, obs_mask = obs_mask)
                    
    def guide_group(self):
    
        # number of parameters: 2, p1 and p2
        npar = 6
        trns = torch.distributions.biject_to(dist.constraints.positive)
    
        # define mean vector and covariance matrix of multivariate normal
        m_hyp = pyro.param('m_hyp', torch.zeros(2*npar))
        st_hyp = pyro.param('scale_tril_hyp',
                       torch.eye(2*npar),
                       constraint=dist.constraints.lower_cholesky)
        
        # set hyperprior to be multivariate normal
        hyp = pyro.sample('hyp',
                     dist.MultivariateNormal(m_hyp, scale_tril=st_hyp),
                     infer={'is_auxiliary': True})
    
        unc_mu = hyp[..., :npar]
        unc_tau = hyp[..., npar:]
    
        c_tau = trns(unc_tau)
    
        ld_tau = trns.inv.log_abs_det_jacobian(c_tau, unc_tau)
        ld_tau = dist.util.sum_rightmost(ld_tau, ld_tau.dim() - c_tau.dim() + 1)
    
        # some numerics tricks
        mu = pyro.sample("mu", dist.Delta(unc_mu, event_dim=1))
        tau = pyro.sample("tau", dist.Delta(c_tau, log_density=ld_tau, event_dim=1))
    
        m_locs = pyro.param('m_locs', torch.zeros(self.n_subjects, npar))
        st_locs = pyro.param('scale_tril_locs',
                        torch.eye(npar).repeat(self.n_subjects, 1, 1),
                        constraint=dist.constraints.lower_cholesky)
    
        with pyro.plate('subject', self.n_subjects):
            # sample unconstrained parameters from multivariate normal
            locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))
    
            lr_day1 = torch.sigmoid(locs[..., 0])
            theta_Q_day1 = torch.exp(locs[..., 1])
            theta_rep_day1 = torch.exp(locs[..., 2])
            
            lr_day2 = torch.sigmoid(locs[..., 3])
            theta_Q_day2 = torch.exp(locs[..., 4])
            theta_rep_day2 = torch.exp(locs[..., 5])
    
            # make dictionary to be able to return samples later
            var_dict = {'lr_day1': lr_day1, \
                        'theta_Q_day1': theta_Q_day1, \
                        'theta_rep_day1': theta_rep_day1, \
                            
                        'lr_day2': lr_day2, \
                        'theta_Q_day2': theta_Q_day2, \
                        'theta_rep_day2': theta_rep_day2}
    
            return var_dict

    def infer_posterior(self,
                        iter_steps=1_000,
                        num_particles=10,
                        optim_kwargs={'lr': .01}): # Adam learning rate
        """Perform SVI over free model parameters."""

        pyro.clear_param_store()

        svi = pyro.infer.SVI(model=self.model_group,
                  guide=self.guide_group,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  # set below to true once code is vectorized
                                  vectorize_particles=True))

        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:#range(iter_steps):
            "Runs through the model twice the first time"
            loss.append(torch.tensor(svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
            if torch.isnan(loss[-1]):
                break

        self.loss = [l.cpu() for l in loss] # = -ELBO (Plotten!)
        
        m_locs = pyro.param("m_locs").data.numpy()
        st_locs = pyro.param("scale_tril_locs").data.numpy()

        m_hyp = pyro.param("m_hyp").data.numpy()
        st_hyp = pyro.param("scale_tril_hyp").data.numpy()

        param_dict = {"m_locs": m_locs, "st_locs": st_locs, "m_hyp": m_hyp, \
                      "st_hyp": st_hyp}

        return self.loss, param_dict
    
    def sample_posterior(self, n_samples=1000):
        # keys = ["lamb_pi", "lamb_r", "h", "dec_temp"]
    
        lr_day1_global = np.zeros((n_samples, self.n_subjects))
        theta_Q_day1_global = np.zeros((n_samples, self.n_subjects))
        theta_rep_day1_global = np.zeros((n_samples, self.n_subjects))
        
        lr_day2_global = np.zeros((n_samples, self.n_subjects))
        theta_Q_day2_global = np.zeros((n_samples, self.n_subjects))
        theta_rep_day2_global = np.zeros((n_samples, self.n_subjects))

        # sample p1 and p2 from guide (the posterior over ps). 
        # Calling the guide yields samples from the posterior after SVI has run.
        for i in range(n_samples):
            sample = self.guide_group()
            for key in sample.keys():
                sample.setdefault(key, torch.ones(1))
                
            lr_day1 = sample["lr_day1"]
            theta_Q_day1 = sample["theta_Q_day1"]
            theta_rep_day1 = sample["theta_rep_day1"]
            
            lr_day2 = sample["lr_day2"]
            theta_Q_day2 = sample["theta_Q_day2"]
            theta_rep_day2 = sample["theta_rep_day2"]
    
            lr_day1_global[i] = lr_day1.detach().numpy()
            theta_Q_day1_global[i] = theta_Q_day1.detach().numpy()
            theta_rep_day1_global[i] = theta_rep_day1.detach().numpy()
            
            lr_day2_global[i] = lr_day2.detach().numpy()
            theta_Q_day2_global[i] = theta_Q_day2.detach().numpy()
            theta_rep_day2_global[i] = theta_rep_day2.detach().numpy()

        # do some data formatting steps
        lr_day1_flat = np.array([lr_day1_global[i,n] for i in range(n_samples) for n in range(self.n_subjects)])
        theta_Q_day1_flat = np.array([theta_Q_day1_global[i,n] for i in range(n_samples) for n in range(self.n_subjects)])
        theta_rep_day1_flat = np.array([theta_rep_day1_global[i,n] for i in range(n_samples) for n in range(self.n_subjects)])
    
        lr_day2_flat = np.array([lr_day2_global[i,n] for i in range(n_samples) for n in range(self.n_subjects)])
        theta_Q_day2_flat = np.array([theta_Q_day2_global[i,n] for i in range(n_samples) for n in range(self.n_subjects)])
        theta_rep_day2_flat = np.array([theta_rep_day2_global[i,n] for i in range(n_samples) for n in range(self.n_subjects)])
    
        subs_flat = np.array([n for i in range(n_samples) for n in range(self.n_subjects)])
    
        sample_dict = {"lr_day1": lr_day1_flat, \
                       "theta_Q_day1": theta_Q_day1_flat, \
                       "theta_rep_day1": theta_rep_day1_flat, \
                           
                       "lr_day2": lr_day2_flat, \
                       "theta_Q_day2": theta_Q_day2_flat, \
                       "theta_rep_day2": theta_rep_day2_flat, \
                       "subject": subs_flat}
    
        # make a pandas dataframe, better for analyses and plotting later (pandas is pythons R equivalent)
        sample_df = pd.DataFrame(sample_dict)

        return sample_df

class GeneralGroupInference(object):
    
    def __init__(self, agent, n_subjects, group_data):
        '''
        General Group inference..
        
        agents : list of agents
        group_data : list of data dicts 
        '''
        self.agent = agent
        self.trials = agent.trials # length of experiment
        self.n_subjects = n_subjects # no. of participants
        self.data = group_data # list of dictionaries
        self.n_parameters = len(self.agent.param_names) # number of parameters
        self.loss = []

    def model(self):
        # print("CHECKPOINT ALPHA")
        # define hyper priors over model parameters
        # prior over sigma of a Gaussian is a Gamma distribution
        a = pyro.param('a', torch.ones(self.n_parameters), constraint=dist.constraints.positive)
        lam = pyro.param('lam', torch.ones(self.n_parameters), constraint=dist.constraints.positive)
        tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1)) # Why a/lam?
        
        sig = 1/torch.sqrt(tau) # Gaus sigma

        # each model parameter has a hyperprior defining group level mean
        # in the form of a Normal distribution
        m = pyro.param('m', torch.zeros(self.n_parameters))
        s = pyro.param('s', torch.ones(self.n_parameters), constraint=dist.constraints.positive)
        mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(1)) # Gauss mu, wieso s*sig?

        # in order to implement groups, where each subject is independent of the others, pyro uses so-called plates.
        # you embed what should be done for each subject into the "with pyro.plate" context
        # the plate vectorizes subjects and adds an additional dimension onto all arrays/tensors
        # i.e. p1 below will have the length n_subjects
        with pyro.plate('subject', self.n_subjects) as ind:
    
            # draw parameters from Normal and transform (for numeric trick reasons)
            base_dist = dist.Normal(0., 1.).expand_by([self.n_parameters]).to_event(1)
            transform = dist.transforms.AffineTransform(mu, sig)
            # print("CHECKPOINT BRAVO")
            locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))
    
            "locs is either of shape [n_participants, n_parameters] or of shape [n_particles, n_participants, n_parameters]"
            if locs.ndim == 2:
                locs = locs[None, :]
                
            self.agent.reset(locs)
            
            n_particles = locs.shape[0]
            # print("CHECKPOINT CHARLIE")
            print("MAKING A ROUND WITH %d PARTICLES"%n_particles)
            t = -1
            for tau in pyro.markov(range(self.trials)):
    
                # probs = torch.ones(n_particles, self.n_subjects, 2)
                # choices = torch.ones(self.n_subjects)
                # obs_mask = torch.ones(self.n_subjects, dtype=torch.bool)
                
                # start_time = time.time()
                trial = torch.tensor(self.data["Trialsequence"][tau])
                blocktype = torch.tensor(self.data["Blocktype"][tau])
                
                if all([self.data["Blockidx"][tau][i] <= 5 for i in range(self.n_subjects)]):
                    day = 1
                    
                elif all([self.data["Blockidx"][tau][i] > 5 for i in range(self.n_subjects)]):
                    day = 2
                    
                else:
                    raise Exception("Da isch a Fehla!")
                
                if all([trial[i] == -1 for i in range(self.n_subjects)]):
                    "Beginning of new block"
                    self.agent.update(torch.tensor([-1]), 
                                      torch.tensor([-1]), 
                                      torch.tensor([-1]), 
                                      day = day, 
                                      trialstimulus = trial)
                    
                else:
                    current_choice = self.data["Choices"][tau]
                    outcome = self.data["Outcomes"][tau]
                
                if all([trial[i] > 10 for i in range(self.n_subjects)]):
                    "Dual-Target Trial"
                    t+=1
                    #assert(torch.is_tensor(trial))
                    option1, option2 = self.agent.find_resp_options(trial)
                    # print("MAKE SURE EVERYTHING WORKS FOR ERRORS AS WELL")
                    # probs should have shape [n_particles, n_subjects, nactions], or [n_subjects, nactions]
                    # RHS comes out as [1, n_actions] or [n_particles, n_actions]
                    
                    "==========================================="
                    # _, mask = self.agent.Qoutcomp(self.agent.V[-1], option1)
                    # Vopt1 = (self.agent.V[-1][torch.where(mask == 1)]).reshape(n_particles, self.n_subjects)
                    # _, mask = self.agent.Qoutcomp(self.agent.V[-1], option2)
                    # Vopt2 = self.agent.V[-1][torch.where(mask == 1)].reshape(n_particles, self.n_subjects)
                    
                    # probs = self.agent.softmax(torch.stack((Vopt1, Vopt2), 2))
                    "==========================================="
                    #assert(trial.ndim==1)
                    probs = self.agent.compute_probs(trial, day)
                    # print(probs)
                    "==========================================="
                    
                    choices = torch.tensor([0 if current_choice[idx] == option1[idx] else 1 for idx in range(len(current_choice))])
                    obs_mask = torch.tensor([0 if cc == -10 else 1 for cc in current_choice ]).type(torch.bool)
                    
                if all([trial[i] != -1 for i in range(self.n_subjects)]):
                    "Update (trial == -1 means this is the beginning of a block -> participants didn't see this trial')"
                    self.agent.update(current_choice, 
                                      outcome, 
                                      blocktype, 
                                      day=day, 
                                      trialstimulus=trial)
                    
                # print("CHECKPOINT DELTA")
                "Sample if dual-target trial and no error was performed"
                if all([trial[i] > 10 for i in range(self.n_subjects)]):
                    # obs & obs_mask shape for inference: [n_subjects] 
                    # probs shape for inference: [n_subjects, nactions] or [n_particles, n_subjects, nactions]
                    # pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), \
                    #             obs = choices.broadcast_to(n_particles, self.n_subjects), \
                    #             obs_mask = obs_mask.broadcast_to(n_particles, self.n_subjects))
                    # pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), \
                    #             obs = choices.broadcast_to(n_particles, self.n_subjects))
                    
                    # print("CHECKPOINT ECHO")
                    pyro.sample('res_{}'.format(t), 
                                dist.Categorical(probs=probs),
                                obs = choices.broadcast_to(n_particles, self.n_subjects))

    def guide(self):
        trns = torch.distributions.biject_to(dist.constraints.positive)
    
        # define mean vector and covariance matrix of multivariate normal
        m_hyp = pyro.param('m_hyp', torch.zeros(2*self.n_parameters))
        st_hyp = pyro.param('scale_tril_hyp',
                       torch.eye(2*self.n_parameters),
                       constraint=dist.constraints.lower_cholesky)
        
        # set hyperprior to be multivariate normal
        hyp = pyro.sample('hyp',
                     dist.MultivariateNormal(m_hyp, scale_tril=st_hyp),
                     infer={'is_auxiliary': True})
    
        unc_mu = hyp[..., :self.n_parameters]
        unc_tau = hyp[..., self.n_parameters:]

        c_tau = trns(unc_tau)

        ld_tau = trns.inv.log_abs_det_jacobian(c_tau, unc_tau)
        ld_tau = dist.util.sum_rightmost(ld_tau, ld_tau.dim() - c_tau.dim() + 1)
    
        # some numerics tricks
        mu = pyro.sample("mu", dist.Delta(unc_mu, event_dim=1))
        tau = pyro.sample("tau", dist.Delta(c_tau, log_density=ld_tau, event_dim=1))
    
        m_locs = pyro.param('m_locs', torch.zeros(self.n_subjects, self.n_parameters))
        st_locs = pyro.param('scale_tril_locs',
                        torch.eye(self.n_parameters).repeat(self.n_subjects, 1, 1),
                        constraint=dist.constraints.lower_cholesky)
        
        with pyro.plate('subject', self.n_subjects):
            locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))

        return {'tau': tau, 'mu': mu, 'locs': locs}

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
        print("Starting inference steps")
        for step in pbar:#range(iter_steps):
            "Runs through the model twice the first time"
            loss.append(torch.tensor(svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
            if torch.isnan(loss[-1]):
                break

        self.loss += [l.cpu() for l in loss] # = -ELBO (Plotten!)
        
        # print("Detaching m_locs")
        # m_locs = pyro.param("m_locs").data.numpy()
        # st_locs = pyro.param("scale_tril_locs").data.numpy()

        # m_hyp = pyro.param("m_hyp").data.numpy()
        # st_hyp = pyro.param("scale_tril_hyp").data.numpy()

        # param_dict = {"m_locs": m_locs, 
        #               "st_locs": st_locs, 
        #               "m_hyp": m_hyp, 
        #               "st_hyp": st_hyp}

        # return self.loss, param_dict

    def sample_posterior(self, n_samples=1_000):
        # keys = ["lamb_pi", "lamb_r", "h", "dec_temp"]

        param_names = self.agent.param_names
        sample_dict = {param: [] for param in param_names}
        sample_dict["subject"] = []

        for i in range(n_samples):
            sample = self.guide()
            for key in sample.keys():
                sample.setdefault(key, torch.ones(1))

            par_sample = self.agent.locs_to_pars(sample["locs"])

            for param in param_names:
                sample_dict[param].extend(list(par_sample[param].detach().numpy()))

            sample_dict["subject"].extend(list(range(self.n_subjects)))

        sample_df = pd.DataFrame(sample_dict)

        return sample_df
    