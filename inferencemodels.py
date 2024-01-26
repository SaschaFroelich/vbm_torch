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

class GeneralGroupInference():
    
    def __init__(self, agent, group_data, blocks):
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
                
        blocks : list len 2
            From which block to which block (exclusive) to perform inference.
            0-indexed. 
            (Here, a block is a R-F condition pair consisting of 962 trials in total, 
             including 1 newcondition trial for conditions F and R.)
            for instance:
            blocks = [0,3] ~ Day 1
            blocks = [3,7] ~ Day 2
            blocks = [0, 7] ~ Days 1 + 2
                
        '''
        
        assert isinstance(blocks, list)
        assert len(blocks) == 2
        
        self.agent = agent
        self.blocks = blocks
        self.trials = agent.trials # length of experiment
        self.num_agents = agent.num_agents # no. of participants
        self.data = group_data # list of dictionaries
        self.num_params = len(self.agent.param_names) # number of parameters
        self.loss = []

    def model(self, *args):
        # define hyper priors over model parameters
        # prior over sigma of a Gaussian is a Gamma distribution
        # torch.manual_seed(1234)
        print("Printing args.")
        print(*args)
        a = pyro.param('a', torch.ones(self.num_params), constraint=dist.constraints.positive)
        lam = pyro.param('lam', torch.ones(self.num_params), constraint=dist.constraints.positive)
        tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1)) # Why a/lam?
        
        sig = pyro.deterministic('sig', 1/torch.sqrt(tau)) # Gauss sigma

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
            transform = dist.transforms.AffineTransform(mu, sig) # Transform via the pointwise affine mapping y = loc + scale*x (-> Neal's funnel)
            
            
            locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))
            
            "locs is either of shape [num_agents, num_params] or of shape [num_particles, num_agents, num_params]"
            if locs.ndim == 2:
                locs = locs[None, :]
                
            'Shape of locs be [num_particles, num_agents, num_parameters]'
            self.agent.reset(locs)
            
            num_particles = locs.shape[0]
            # print("MAKING A ROUND WITH %d PARTICLES"%num_particles)
            
            env.Env.run_loop(None, 
                             self.agent, 
                             self.data, 
                             num_particles, 
                             infer = 1,
                             blocks = self.blocks)

    def guide(self, *args):
        # biject_to(constraint) looks up a bijective Transform from constraints.real 
        # to the given constraint. The returned transform is guaranteed to have 
        # .bijective = True and should implement .log_abs_det_jacobian().
        trns = torch.distributions.biject_to(dist.constraints.positive)

        # define mean vector and covariance matrix of multivariate normal
        m_hyp = pyro.param('m_hyp', torch.zeros(2*self.num_params))
        st_hyp = pyro.param('scale_tril_hyp',
                       torch.eye(2*self.num_params),
                       constraint=dist.constraints.lower_cholesky)

        # set hyperprior to be multivariate normal
        # scale_tril (Tensor) – lower-triangular factor of covariance, with positive-valued diagonal
        hyp = pyro.sample('hyp',
                     dist.MultivariateNormal(m_hyp, scale_tril=st_hyp),
                     infer={'is_auxiliary': True})

        unc_mu = hyp[..., :self.num_params]
        unc_tau = hyp[..., self.num_params:]

        c_tau = trns(unc_tau)

        ld_tau = -trns.inv.log_abs_det_jacobian(c_tau, unc_tau)
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

        return {'tau': tau, 'mu': mu, 'locs': locs, 'm_locs': m_locs, 'st_locs': st_locs}
    
    def infer_posterior(self,
                        iter_steps = 1_000,
                        num_particles = 10,
                        optim_kwargs = {'lr': .01},  # Adam learning rate
                        automatic_stop = False):
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

        # import trace_elbo
        # svi.loss_and_grads_agentelbos = pyro.infer.Trace_ELBO.loss_and_grads_agentelbos
        
        print("Computing first-level ELBOs.")
        num_iters = 10
        ELBOs = torch.zeros(num_iters, self.agent.num_agents)
        for i in range(num_iters):
            print(f"Iteration {i} of {num_iters}")
            ELBOs[i, :] = svi.step_agent_elbos()
            # ELBOs[i, :] = self.step_agent_elbos(svi)
        
        elbos = ELBOs.mean(dim=0)
        std = ELBOs.std(dim=0)
        
        print(f"Final ELBO after {iter_steps} steps is {elbos} +- {std}.")
        
        self.loss += [l.cpu() for l in loss] # = -ELBO (Plotten!)
        
        return (elbos, std)
        
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
            sample_dict = {'tau': [], 'mu': [], 'locs': torch.zeros((self.num_agents, 
                                                                     self.agent.num_params, 
                                                                     n_samples))}
            
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
            
            # from pyro.infer import MCMC, NUTS, Predictive
            # print("BEGINNING PREDICTIVE.")
            # predictive_svi = Predictive(self.model, guide=self.guide, num_samples=10)()
            # for k, v in predictive_svi.items():
            #     print(f"{k}: {tuple(v.shape)}")
            
            # dfgh
            return sample_df
    
    def compute_ll(self, df = None):
        '''
        Computeslog-likelihood of model.
        
        Parameters
        ----------
        df : DataFrame 'inf_mean_df'
            Contains the parameters with which to compute the ll

        Returns
        -------
        None.

        '''
        
        if df is not None:
            locs = self.agent.pars_to_locs(df)[None, ...]
            
            if len(df) > 200:
                raise Exception('df should contain one row per agent.')
            
        else:
            locs = self.sample_posterior(locs = True).mean(axis = -1)[None, ...] # [num_particles, num_agents, num_parameters]
            
        conditioned_model = pyro.condition(self.model, data = {'locs' : locs})
        print("Starting trace, baby!")
        trace = pyro.poutine.trace(conditioned_model).get_trace()
        
        log_likelihood = trace.log_prob_sum()

        return log_likelihood
    
    def compute_IC(self, df=None):
        print("BIC and AIC only approx. right, since would have to use maximized values of log-likelihood in reality")
        '''
            BIC = k*ln(n) - 2*ll --> the lower, the better
            ll = maximized log-likelihood value
            k = number of parameters
            n = number of observations
        '''
        
        data_df = pd.DataFrame(self.data).explode(list(self.data.keys()))
        data_df = data_df[data_df['trialsequence'] != -1]
        data_df = data_df[data_df['trialsequence'] > 10]
        data_df = data_df[data_df['choices'] != -2]
        n = len(data_df)
        
        BIC = torch.tensor(self.agent.num_params)*torch.log(torch.tensor(n)) -\
            2*self.compute_ll(df = df)
            
        '''
            AIC = 2*k - 2*ll --> the lower, the better
            ll = maximized log-likelihood value
            k = number of parameters
        '''
            
        AIC = 2*torch.tensor(self.agent.num_params) - 2*self.compute_ll(df = df)
        
        return BIC, AIC
    
    def compute_ELBOS(self):
        
        locs = self.sample_posterior(locs = True).mean(axis = -1)[None, ...]
        conditioned_model = pyro.condition(self.model, data = {'locs' : locs})
        guide_tr = pyro.poutine.trace(self.guide).get_trace()
        model_tr = pyro.poutine.trace(pyro.poutine.replay(conditioned_model, trace=guide_tr)).get_trace()
        monte_carlo_elbo = model_tr.log_prob_sum() - guide_tr.log_prob_sum()

        elbo_particle = torch.zeros(self.agent.num_agents)

        # compute elbo and surrogate elbo
        for name, site in model_tr.nodes.items():
                
            if site["type"] == "sample" and "observed" in site["name"]:
                elbo_particle = elbo_particle + site["log_prob"].sum(dim=0)
                    
        for name, site in guide_tr.nodes.items():
            if site["type"] == "sample" and "observed" in site["name"]:
                log_prob, score_function_term, entropy_term = site["score_parts"]

                elbo_particle = elbo_particle - site["log_prob"].sum(dim=0)

        elbo_particle /= self.agent.num_particles

        print(f"MC elbo is {monte_carlo_elbo}.")
        
    def loo_predict(self, df = None):
        '''
        Parameters
        ----------
        df : DataFrame 'inf_mean_df'
            Contains the parameters with which to compute the ll

        Returns
        -------
        probs_means : tensor
            The average probability of the chosen action.

        '''
        if df is not None:
            locs = self.agent.pars_to_locs(df)[None, ...]
            
        else:
            locs = self.sample_posterior(locs = True).mean(axis = -1)[None, ...]
        
        
        conditioned_model = pyro.condition(self.model, data = {'locs' : locs})
        trace = pyro.poutine.trace(conditioned_model).get_trace()
        
        chosen_prob_means = []
        for key in trace.nodes.keys():
            if 'observed' in key  and 'unobserved' not in key and trace.nodes[key]['is_observed']:
                if int(key.split('_')[1]) >= 1598:
                    probs = trace.nodes[key]['fn'].probs
                    choices = trace.nodes[key]['value']
                    observed = trace.nodes[key]['mask']
                    probs_dtt = probs*observed.type(torch.int)[..., None]
                    chosen_prob_means.append((probs_dtt[0, range(self.agent.num_agents), choices[0,:]]).tolist())

        probs_means = torch.tensor(chosen_prob_means).sum(axis=0) / torch.ceil(torch.tensor(chosen_prob_means)).sum(axis = 0)
        
        return probs_means
    
    
class BC():
    "Bayesian correlation"
    def __init__(self, x, y, method = 'pearson'):
        '''
        General Group inference.
        
        x : torch tensor
        
        y : torch tensor
        
        agent : obj
            Initialization of agent class with num_agents parallel agents.

        data : tensor [num_observations, 2] [x,y]
        
        methid : str
            'pearson'
            'spearman'
        
        '''
        
        assert x.ndim == 1
        assert y.ndim == 1
        
        self.x = x
        self.y = y
        
        self.num_datapoints = self.x.shape[0]
        self.loss = []
        
        self.params = ['beta', 'sigma']
        self.param_names = ['beta', 'sigma']
        self.num_params = len(self.param_names)
        self.num_agents = 1
        self.method = method
        

    def model(self, *args):
        # define hyper priors over model parameters
        # prior over sigma of a Gaussian is a Gamma distribution
        # torch.manual_seed(1234)
        a = pyro.param('a', torch.ones(self.num_params), constraint=dist.constraints.positive)
        lam = pyro.param('lam', torch.ones(self.num_params), constraint=dist.constraints.positive)
        tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1)) # Why a/lam?
        
        sig = pyro.deterministic('sig', 1/torch.sqrt(tau)) # Gauss sigma

        # each model parameter has a hyperprior defining group level mean
        # in the form of a Normal distribution
        m = pyro.param('m', torch.zeros(self.num_params))
        s = pyro.param('s', torch.ones(self.num_params), constraint=dist.constraints.positive)
        mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(1)) # Gauss mu, wieso s*sig?

        # in order to implement groups, where each subject is independent of the others, pyro uses so-called plates.
        # you embed what should be done for each subject into the "with pyro.plate" context
        # the plate vectorizes subjects and adds an additional dimension onto all arrays/tensors
        # i.e. p1 below will have the length num_agents
        # draw parameters from Normal and transform (for numeric trick reasons)
        base_dist = dist.Normal(0., 1.).expand_by([self.num_params]).to_event(1)
        transform = dist.transforms.AffineTransform(mu, sig)
        locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))
        "locs is either of shape [num_agents, num_params] or of shape [num_particles, num_agents, num_params]"
        if locs.ndim == 2:
            locs = locs[None, :]
            
        if self.method == 'pearson':
            "Standardize the data"
            x = (self.x - self.x.mean()) / self.x.std()
            y = (self.y - self.y.mean()) / self.y.std()
            
        elif self.method == 'spearman':
            
            _, x_pos = self.x.sort()
            _, y_pos = self.y.sort()
            
            self.x_rearr = -torch.ones(self.num_datapoints, dtype = int)
            self.y_rearr = -torch.ones(self.num_datapoints, dtype = int)

            for i in range(self.num_datapoints):
                self.x_rearr[x_pos[i]] = i
                self.y_rearr[y_pos[i]] = i
            
            "Standardize the data"
            x = (self.x_rearr - self.x_rearr.float().mean()) / self.x_rearr.float().std()
            y = (self.y_rearr - self.y_rearr.float().mean()) / self.y_rearr.float().std()
            
        else:
            raise Exception("Must specify correlation method.")
            

        'Shape of locs be [num_particles, num_agents, num_parameters]'
        for tau in range(self.num_datapoints):
            num_particles = locs.shape[0]
                
            xi = x[tau]
            yi = y[tau]
            
            mean =xi*locs[..., 0]
            std = torch.exp(locs[..., 1])
            
            pyro.sample(f'res_{tau}',
                        dist.Normal(loc = mean, scale = std),
                        obs = yi)

    def guide(self, *args):
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

        ld_tau = -trns.inv.log_abs_det_jacobian(c_tau, unc_tau)
        ld_tau = dist.util.sum_rightmost(ld_tau, ld_tau.dim() - c_tau.dim() + 1)
    
        # some numerics tricks
        mu = pyro.sample("mu", dist.Delta(unc_mu, event_dim=1))
        tau = pyro.sample("tau", dist.Delta(c_tau, log_density=ld_tau, event_dim=1))
    
        m_locs = pyro.param('m_locs', torch.zeros(self.num_agents, self.num_params))
        st_locs = pyro.param('scale_tril_locs',
                        torch.eye(self.num_params).repeat(self.num_agents, 1, 1),
                        constraint=dist.constraints.lower_cholesky)
        
        locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))

        return {'tau': tau, 'mu': mu, 'locs': locs, 'm_locs': m_locs, 'st_locs': st_locs}

    def infer_posterior(self,
                        iter_steps = 1_000,
                        num_particles = 10,
                        optim_kwargs = {'lr': .01},  # Adam learning rate
                        automatic_stop = False):
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

        # import trace_elbo
        # svi.loss_and_grads_agentelbos = pyro.infer.Trace_ELBO.loss_and_grads_agentelbos
        
        # print("Computing first-level ELBOs.")
        # num_iters = 10
        # ELBOs = torch.zeros(num_iters, self.num_agents)
        # for i in range(num_iters):
        #     print(f"Iteration {i} of {num_iters}")
        #     ELBOs[i, :] = svi.step_agent_elbos()
        #     # ELBOs[i, :] = self.step_agent_elbos(svi)
        
        # elbos = ELBOs.mean(dim=0)
        # std = ELBOs.std(dim=0)
        
        # print(f"Final ELBO after {iter_steps} steps is {elbos} +- {std}.")
        
        # self.loss += [l.cpu() for l in loss] # = -ELBO (Plotten!)
        
        # return (elbos, std)
        
    def locs_to_pars(self, locs):
        param_dict = {'beta': locs[..., 0],
                      'sigma': torch.exp(locs[..., 1])}
        
        return param_dict
        
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

        param_names = self.param_names
        if locs:
            sample_dict = {'tau': [], 'mu': [], 'locs': torch.zeros((self.num_agents, self.num_params, n_samples))}
            
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
    
                par_sample = self.locs_to_pars(sample["locs"])

                for param in param_names:
                    sample_dict[param].extend(list(par_sample[param].detach().numpy()))
    
                sample_dict["ag_idx"].extend(list(range(self.num_agents)))
        
            sample_df = pd.DataFrame(sample_dict)
            
            
            return sample_df
    
    def compute_ll(self, df = None):
        '''
        Computeslog-likelihood of model.
        
        Parameters
        ----------
        df : DataFrame 'inf_mean_df'
            Contains the parameters with which to compute the ll

        Returns
        -------
        None.

        '''
        
        if df is not None:
            locs = self.agent.pars_to_locs(df)[None, ...]
            
            if len(df) > 200:
                raise Exception('df should contain one row per agent.')
            
        else:
            locs = self.sample_posterior(locs = True).mean(axis = -1)[None, ...] # [num_particles, num_agents, num_parameters]
            
        conditioned_model = pyro.condition(self.model, data = {'locs' : locs})
        print("Starting trace, baby!")
        trace = pyro.poutine.trace(conditioned_model).get_trace()
        
        log_likelihood = trace.log_prob_sum()

        return log_likelihood
    
    def compute_IC(self, df=None):
        print("BIC and AIC only approx. right, since would have to use maximized values of log-likelihood in reality")
        '''
            BIC = k*ln(n) - 2*ll --> the lower, the better
            ll = maximized log-likelihood value
            k = number of parameters
            n = number of observations
        '''
        
        data_df = pd.DataFrame(self.data).explode(list(self.data.keys()))
        data_df = data_df[data_df['trialsequence'] != -1]
        data_df = data_df[data_df['trialsequence'] > 10]
        data_df = data_df[data_df['choices'] != -2]
        n = len(data_df)
        
        BIC = torch.tensor(self.agent.num_params)*torch.log(torch.tensor(n)) -\
            2*self.compute_ll(df = df)
            
        '''
            AIC = 2*k - 2*ll --> the lower, the better
            ll = maximized log-likelihood value
            k = number of parameters
        '''
            
        AIC = 2*torch.tensor(self.agent.num_params) - 2*self.compute_ll(df = df)
        
        return BIC, AIC
    
    def compute_ELBOS(self):
        
        locs = self.sample_posterior(locs = True).mean(axis = -1)[None, ...]
        conditioned_model = pyro.condition(self.model, data = {'locs' : locs})
        guide_tr = pyro.poutine.trace(self.guide).get_trace()
        model_tr = pyro.poutine.trace(pyro.poutine.replay(conditioned_model, trace=guide_tr)).get_trace()
        monte_carlo_elbo = model_tr.log_prob_sum() - guide_tr.log_prob_sum()

        elbo_particle = torch.zeros(self.agent.num_agents)

        # compute elbo and surrogate elbo
        for name, site in model_tr.nodes.items():
                
            if site["type"] == "sample" and "observed" in site["name"]:
                elbo_particle = elbo_particle + site["log_prob"].sum(dim=0)
                    
        for name, site in guide_tr.nodes.items():
            if site["type"] == "sample" and "observed" in site["name"]:
                log_prob, score_function_term, entropy_term = site["score_parts"]

                elbo_particle = elbo_particle - site["log_prob"].sum(dim=0)

        elbo_particle /= self.agent.num_particles

        print(f"MC elbo is {monte_carlo_elbo}.")
        
    def loo_predict(self, df = None):
        '''
        Parameters
        ----------
        df : DataFrame 'inf_mean_df'
            Contains the parameters with which to compute the ll

        Returns
        -------
        probs_means : tensor
            The average probability of the chosen action.

        '''
        if df is not None:
            locs = self.agent.pars_to_locs(df)[None, ...]
            
        else:
            locs = self.sample_posterior(locs = True).mean(axis = -1)[None, ...]
        
        
        conditioned_model = pyro.condition(self.model, data = {'locs' : locs})
        trace = pyro.poutine.trace(conditioned_model).get_trace()
        
        chosen_prob_means = []
        for key in trace.nodes.keys():
            if 'observed' in key  and 'unobserved' not in key and trace.nodes[key]['is_observed']:
                if int(key.split('_')[1]) >= 1598:
                    probs = trace.nodes[key]['fn'].probs
                    choices = trace.nodes[key]['value']
                    observed = trace.nodes[key]['mask']
                    probs_dtt = probs*observed.type(torch.int)[..., None]
                    chosen_prob_means.append((probs_dtt[0, range(self.agent.num_agents), choices[0,:]]).tolist())

        probs_means = torch.tensor(chosen_prob_means).sum(axis=0) / torch.ceil(torch.tensor(chosen_prob_means)).sum(axis = 0)
        
        return probs_means    