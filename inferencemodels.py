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
import env

import pyro
import pyro.distributions as dist

# np.random.seed(123)
# torch.manual_seed(123)

beta_variance = lambda alpha, beta: alpha*beta / ((alpha+beta)**2 * (alpha + beta + 1))
gamma_variance = lambda conc, rate: conc/(rate**2)

device = torch.device("cpu")

torch.set_default_tensor_type(torch.DoubleTensor)

class GeneralGroupInference(object):
    
    def __init__(self, agent, group_data):
        '''
        General Group inference..
        
        agent : obj
            Initialization of agent class with num_agents parallel agents.

        groupdata : dict
            Contains experimental data.
            Keys
                trialsequence : nested list, 'shape' [num_trials, num_agents]
                choices : nested list, 'shape' [num_trials, num_agents]
                outcomes : nested list, 'shape' [num_trials, num_agents]
                blocktype : nested list, 'shape' [num_trials, num_agents]
                blockidx : nested list, 'shape' [num_trials, num_agents]
                RT : nested list, 'shape' [num_trials, num_agents]
                group : list, len [num_agents]
        '''
        self.agent = agent
        self.trials = agent.trials # length of experiment
        self.num_agents = agent.num_agents # no. of participants
        self.data = group_data # list of dictionaries
        self.num_params = len(self.agent.param_names) # number of parameters
        self.loss = []

    def model(self):
        # define hyper priors over model parameters
        # prior over sigma of a Gaussian is a Gamma distribution
        a = pyro.param('a', torch.ones(self.num_params), constraint=dist.constraints.positive)
        lam = pyro.param('lam', torch.ones(self.num_params), constraint=dist.constraints.positive)
        tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1)) # Why a/lam?
        
        sig = 1/torch.sqrt(tau) # Gauss sigma

        # each model parameter has a hyperprior defining group level mean
        # in the form of a Normal distribution
        m = pyro.param('m', torch.zeros(self.num_params))
        s = pyro.param('s', torch.ones(self.num_params), constraint=dist.constraints.positive)
        mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(1)) # Gauss mu, wieso s*sig?

        # in order to implement groups, where each subject is independent of the others, pyro uses so-called plates.
        # you embed what should be done for each subject into the "with pyro.plate" context
        # the plate vectorizes subjects and adds an additional dimension onto all arrays/tensors
        # i.e. p1 below will have the length num_agents
        with pyro.plate('ag_idx', self.num_agents):
            # draw parameters from Normal and transform (for numeric trick reasons)
            base_dist = dist.Normal(0., 1.).expand_by([self.num_params]).to_event(1)
            transform = dist.transforms.AffineTransform(mu, sig)
            locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))
            "locs is either of shape [num_agents, num_params] or of shape [num_particles, num_agents, num_params]"
            if locs.ndim == 2:
                locs = locs[None, :]
                
            'Shape of locs be [num_particles, num_agents, num_parameters]'
            self.agent.reset(locs)
            
            num_particles = locs.shape[0]
            # print("MAKING A ROUND WITH %d PARTICLES"%num_particles)
            
            env.Env.run_loop(None, self.agent, self.data, num_particles, infer = 1)

    def guide(self):
        trns = torch.distributions.biject_to(dist.constraints.positive)
    
        # define mean vector and covariance matrix of multivariate normal
        m_hyp = pyro.param('m_hyp', torch.zeros(2*self.num_params))
        st_hyp = pyro.param('scale_tril_hyp',
                       torch.eye(2*self.num_params),
                       constraint=dist.constraints.lower_cholesky)
        
        # set hyperprior to be multivariate normal
        hyp = pyro.sample('hyp',
                     dist.MultivariateNormal(m_hyp, scale_tril=st_hyp),
                     infer={'is_auxiliary': True})
    
        unc_mu = hyp[..., :self.num_params]
        unc_tau = hyp[..., self.num_params:]

        c_tau = trns(unc_tau)

        ld_tau = trns.inv.log_abs_det_jacobian(c_tau, unc_tau)
        ld_tau = dist.util.sum_rightmost(ld_tau, ld_tau.dim() - c_tau.dim() + 1)
    
        # some numerics tricks
        mu = pyro.sample("mu", dist.Delta(unc_mu, event_dim=1))
        tau = pyro.sample("tau", dist.Delta(c_tau, log_density=ld_tau, event_dim=1))
    
        m_locs = pyro.param('m_locs', torch.zeros(self.num_agents, self.num_params))
        st_locs = pyro.param('scale_tril_locs',
                        torch.eye(self.num_params).repeat(self.num_agents, 1, 1),
                        constraint=dist.constraints.lower_cholesky)
        
        with pyro.plate('ag_idx', self.num_agents):
            locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))

        return {'tau': tau, 'mu': mu, 'locs': locs}

    def infer_posterior(self,
                        iter_steps = 1_000,
                        num_particles = 10,
                        optim_kwargs = {'lr': .01},
                        automatic_stop = False): # Adam learning rate
        """Perform SVI over free model parameters."""

        pyro.clear_param_store()

        svi = pyro.infer.SVI(model=self.model,
                  guide=self.guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  # set below to true once code is vectorized
                                  vectorize_particles=True))
        loss = []
        print("Starting inference.")
        if automatic_stop:
            print("Automatic Halting enabled.")
            keep_inferring = 1
            step = 0
            while keep_inferring:
                step += 1
                "Runs through the model twice the first time"
                loss.append(torch.tensor(svi.step()).to(device))
                # pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
                if torch.isnan(loss[-1]):
                    break
                
                if step > 1_000 and step%250 == 0:
                    print("\nInference step number %d.\n"%step)
                    if torch.abs(torch.tensor(loss[-1_000:-750]).mean() - \
                                 torch.tensor(loss[-250:]).mean()) < torch.tensor(loss[-250:]).std() / 2:
                        keep_inferring = 0
                        
                    if step >= iter_steps:
                        keep_inferring = 0

        else:
            "Stop after iter_steps steps."
            pbar = tqdm(range(iter_steps), position=0)
            for step in pbar:#range(iter_steps):
                "Runs through the model twice the first time"
                loss.append(torch.tensor(svi.step()).to(device))
                pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
                if torch.isnan(loss[-1]):
                    break

        self.loss += [l.cpu() for l in loss] # = -ELBO (Plotten!)
        
    def sample_posterior(self, n_samples = 1_000, locs = False):
        '''

        Parameters
        ----------
        n_samples : int, optional
            The number of samples from each posterior. The default is 1_000.
            
        locs : bool, optional
            0 : return parameters in DataFrame
            1 : return locs as dictionary
            The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        # keys = ["lamb_pi", "lamb_r", "h", "dec_temp"]

        param_names = self.agent.param_names
        if locs:
            sample_dict = {'tau': [], 'mu': [], 'locs': torch.zeros((self.num_agents, self.agent.num_params, n_samples))}
            
            for i in range(n_samples):
                sample = self.guide()
                for key in sample.keys():
                    sample.setdefault(key, torch.ones(1))
                sample_dict['locs'][:,:,i] = sample['locs']
                
            return sample_dict['locs']
            
        else:
            'Original Code'
            sample_dict = {param: [] for param in param_names}
            sample_dict["ag_idx"] = []

            for i in range(n_samples):
                sample = self.guide()
                for key in sample.keys():
                    sample.setdefault(key, torch.ones(1))
    
                par_sample = self.agent.locs_to_pars(sample["locs"])

                for param in param_names:
                    sample_dict[param].extend(list(par_sample[param].detach().numpy()))
    
                sample_dict["ag_idx"].extend(list(range(self.num_agents)))
        
            sample_df = pd.DataFrame(sample_dict)
            return sample_df
    
    def compute_ll(self, df = None):
        '''
        Computeslog-likelihood of model
        Parameters
        ----------
        df : DataFrame
            Contains the parameters with which to compute the ll

        Returns
        -------
        None.

        '''
        
        if df is not None:
            locs = self.agent.pars_to_locs(df).mean(axis=-1)
            
        else:
            locs = self.sample_posterior(locs = True).mean(axis = -1)[None, ...]
            
        conditioned_model = pyro.condition(self.model, data = {'locs' : locs})
        
        trace = pyro.poutine.trace(conditioned_model).get_trace()
        log_likelihood = trace.log_prob_sum()

        return log_likelihood
    
    def compute_IC(self):
        print("BIC only approx. right.")
        '''
            BIC = k*ln(n) - 2*ll --> the lower, the better
        '''
        
        df = pd.DataFrame(self.data).explode(list(self.data.keys()))
        df = df[df['trialsequence'] != -1]
        df = df[df['trialsequence'] > 10]
        df = df[df['choices'] != -2]
        n = len(df)
        
        BIC = torch.tensor(self.agent.num_params)*torch.log(torch.tensor(n)) -\
            2*self.compute_ll()

        return BIC

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
        
        # parameter shape for inference (until now): [1] or [num_particles]
        param_dict = {"lr": lr[:,None], "omega": omega[None, :], "dectemp": dectemp[:, None], "k": k[:, None]}

        # lr is shape [1,1] or [num_particles, 1]
        self.agent.reset(**param_dict)
        
        t = -1
        for tau in range(self.trials):
            
            trial = self.data["trialsequence"][tau]
            blocktype = self.data["blocktype"][tau]
            
            #vec_ben = np.round(np.array(self.data["repvals"][tau]),12)
            #vec_agent = np.round(np.array(torch.squeeze(self.agent.rep[-1])),12)
            
            #np.testing.#assert_allclose(vec_ben, vec_agent, rtol=1e-5)
            
            if self.data["blockidx"][tau] <= 5:
                day = 1
                
            elif self.data["blockidx"][tau] > 5:
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
                current_choice = self.data["choices"][tau]
                outcome = self.data["outcomes"][tau]
                
            if trial > 10:
                "Dual-Target Trial"
                t+=1
                #assert(isinstance(trial, list))
                option1, option2 = self.agent.find_resp_options(trial)
                
                # Comes out as [1, n_actions] or [num_particles, n_actions]probs
                probs = self.agent.softmax(torch.cat(([self.agent.V[-1][:, option1][:, None], self.agent.V[-1][:, option2][:, None]]), dim=-1))
                
                if current_choice == option1:
                    choice = torch.tensor([0])
                    
                elif current_choice == option2:
                    choice = torch.tensor([1])
                    
                elif current_choice == -2:
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
                
        # parameter shape for inference (until now): [1] or [num_particles]
        param_dict = {"lr_day1": lr_day1[:,None], \
                      "lr_day2": lr_day2[:,None], \
                      "theta_Q_day1": theta_Q_day1[None, :], \
                      "theta_Q_day2": theta_Q_day2[None, :], \
                      "theta_rep_day1": theta_rep_day1[None, :], \
                      "theta_rep_day2": theta_rep_day2[None, :], \
                      "k": k[:, None]}

        # lr is shape [1,1] or [num_particles, 1]
        self.agent.reset(**param_dict)
        
        t = -1
        for tau in range(self.trials):
            trial = self.data["trialsequence"][tau][0]
            blocktype = self.data["blocktype"][tau][0]
            
            if self.data["blockidx"][tau][0] <= 5:
                day = 1
                
            elif self.data["blockidx"][tau][0] > 5:
                day = 2
                
            else:
                raise Exception("Da isch a Fehla!")
            
            if trial == -1:
                "Beginning of new block"
                self.agent.update(torch.tensor([-1]), torch.tensor([-1]), torch.tensor([-1]), day=day, trialstimulus=trial)
                
            else:
                current_choice = self.data["choices"][tau]
                outcome = self.data["outcomes"][tau]
                
            if trial > 10:
                "Dual-Target Trial"
                t+=1
                #assert(isinstance(trial, list))
                option1, option2 = self.agent.find_resp_options(trial)
                
                # Comes out as [1, n_actions] or [num_particles, n_actions]
                probs = self.agent.softmax(torch.cat(([self.agent.V[-1][..., option1], self.agent.V[-1][..., option2]]),dim=-1))
                                
                if current_choice == option1:
                    choice = torch.tensor([0])
                    
                elif current_choice == option2:
                    choice = torch.tensor([1])
                    
                elif current_choice == -10:
                    raise Exception("Fehla! Error should be -2.")
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