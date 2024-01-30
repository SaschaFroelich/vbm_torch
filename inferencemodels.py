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
            blocks = [0,3] ~ Day 1, blocks 0, 1, and 2
            blocks = [3,7] ~ Day 2
            blocks = [0, 7] ~ Days 1 + 2
                
        '''
        
        assert isinstance(blocks, list)
        assert len(blocks) == 2
        
        self.agent = agent
        self.blocks = blocks
        # self.trials = agent.trials # length of experiment
        self.num_agents = agent.num_agents # no. of participants
        self.data = group_data # list of dictionaries
        self.num_params = len(self.agent.param_names) # number of parameters
        self.loss = []
        self.comp_subj_trials() # Trials per subject (used for AIC and BIC computation)

    def comp_subj_trials(self):
        data_df = pd.DataFrame(self.data).explode(list(self.data.keys()))
        data_df = data_df[data_df['trialsequence']>10]
        data_df = data_df[data_df['choices'] != -2]
        
        data_df = data_df[data_df['blockidx'] >= 2*self.blocks[0]]
        data_df = data_df[data_df['blockidx'] < 2*self.blocks[-1]]
        
        if 'ID' in data_df.columns:
            data_df = data_df.loc[:, ['choices', 'ID']]
            data_df['trial_count'] = 1
            data_df = data_df.loc[:, ['trial_count', 'ID']].groupby('ID', as_index = True).sum()
            
            "Arange rows as they appear in self.data dict"
            data_df = data_df.loc[self.data['ID'][0], :]
            self.trial_counts = torch.squeeze(torch.tensor(data_df.to_numpy()))
            
        else:
            data_df = data_df.loc[:, ['choices', 'ag_idx']]
            data_df['trial_count'] = 1
            data_df = data_df.loc[:, ['trial_count', 'ag_idx']].groupby('ag_idx', as_index = True).sum()
            
            "Arange rows as they appear in self.data dict"
            data_df = data_df.loc[self.data['ag_idx'][0], :]
            self.trial_counts = torch.squeeze(torch.tensor(data_df.to_numpy()))

    def model(self, *args):
        
        # define hyper priors over model parameters
        # prior over sigma of a Gaussian is a Gamma distribution
        # torch.manual_seed(1234)
        # print("Printing args.")
        # print(*args)
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
            
            # pyro.deterministic('agent_params', self.agent.param_dict)
            
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
                        optim_kwargs = {'lr': .01}):  # Adam learning rate
        """Perform SVI over free model parameters."""

        pyro.clear_param_store()

        svi = pyro.infer.SVI(model=self.model,
                  guide=self.guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  vectorize_particles=True))
        loss = []
        print("Starting inference.")

        "Stop after iter_steps steps."
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:#range(iter_steps):
            "Runs through the model twice the first time"
            loss.append(torch.tensor(svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
            if torch.isnan(loss[-1]):
                break

        print("\nComputing first-level ELBOs.")
        num_iters = 10
        ELBOs = torch.zeros(num_iters, self.agent.num_agents)
        for i in range(num_iters):
            print(f"Iteration {i} of {num_iters}")
            ELBOs[i, :] = svi.step_agent_elbos()
            # ELBOs[i, :] = self.step_agent_elbos(svi)
        
        elbos = ELBOs.mean(dim=0)
        std = ELBOs.std(dim=0)
        
        # print(f"Final ELBO after {iter_steps} steps is {elbos} +- {std}.")
        
        self.loss += [l.cpu() for l in loss] # = -ELBO (Plotten!)
        
        return (elbos.detach(), std.detach())
        
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

        import time
        from pyro.infer import Predictive
        start = time.time()
        print(f"Posterior predictives with {n_samples} samples.")
        firstlevel_dict = {param:[] for param in self.agent.param_names}
        
        if 'ID' in self.data.keys():
            firstlevel_dict['ID'] = []
        firstlevel_dict['ag_idx'] = []

        secondlevel_dict = {param + '_mu':[] for param in self.agent.param_names}
        for param in self.agent.param_names:
            secondlevel_dict[param + '_sig'] = []

        predictive_svi = Predictive(model = self.model,  
                                    guide = self.guide, 
                                    num_samples=n_samples)()

        grouplevel_loc = predictive_svi['mu']
        grouplevel_stdev = predictive_svi['sig']
        predictive_locs = predictive_svi['locs']
        
        predictive_model_params = self.agent.locs_to_pars(predictive_locs)
        
        "1st-level DataFrame"
        for agidx in range(self.num_agents):
            for param_name in self.agent.param_names:
                firstlevel_dict[param_name].extend(predictive_model_params[param_name][:, agidx])
                
            if 'ID' in self.data.keys():
                firstlevel_dict['ID'].extend([self.data['ID'][0][agidx]]*len(predictive_model_params[param_name][:, agidx]))
            firstlevel_dict['ag_idx'].extend([self.data['ag_idx'][0][agidx]]*len(predictive_model_params[param_name][:, agidx]))

        "2nd-level DataFrame"
        # for param_name_idx in range(len(self.agent.param_dict.keys())):
        #     secondlevel_dict[self.agent.param_dict.keys()[self.agent.param_dict.keys()[param_name_idx]] + '_mu'].\
        #         append(grouplevel_loc[:, ..., param_name_idx])
                
        #     secondlevel_dict[self.agent.param_dict.keys()[self.agent.param_dict.keys()[param_name_idx]] + '_sig'].\
        #         append(grouplevel_stdev[:, ..., param_name_idx])
        
        idx = 0
        for k, v in self.agent.param_dict.items():
            secondlevel_dict[k + '_mu'].\
                extend(grouplevel_loc[:, ..., idx])
                
            secondlevel_dict[k + '_sig'].\
                extend(grouplevel_stdev[:, ..., idx])
                
            idx += 1

        firstlevel_df = pd.DataFrame(data = firstlevel_dict)
        secondlevel_df = pd.DataFrame(data = secondlevel_dict)
            
        print(f"Time elapsed: {time.time() - start} secs.")
        
        for param_name in self.agent.param_names:
            firstlevel_df[param_name] = firstlevel_df[param_name].map(lambda x: x.item())
            
        for col_name in secondlevel_df.columns:
            secondlevel_df[col_name] = secondlevel_df[col_name].map(lambda x: x.item())
            
        return firstlevel_df, secondlevel_df
    
    def model_mle(self, mle_locs = None):

        with pyro.plate('ag_idx', self.num_agents):
            
            if mle_locs is not None:
                print("Using MLE locs.")
                locs = mle_locs
                
            elif mle_locs == None:
                locs = pyro.param('locs', torch.zeros((1, self.num_agents, self.num_params)))
            
            else:
                raise Exception("Sum'thin' wrong.")
            
            # print(locs.shape)
            "locs is either of shape [num_agents, num_params] or of shape [num_particles, num_agents, num_params]"
            if locs.ndim == 2:
                locs = locs[None, :]
                
            'Shape of locs be [num_particles, num_agents, num_parameters]'
            self.agent.reset(locs)
            
            num_particles = locs.shape[0]
            
            mll = env.Env.run_loop(None, 
                              self.agent, 
                              self.data, 
                              num_particles, 
                              infer = (mle_locs != None)+1,
                              blocks = self.blocks)
            
        return mll

    def guide_mle(self):
        pass

    def train_mle(self, 
                  max_iter_steps = 8000,
                  halting_rtol = 1e-05):
        '''
            Step 1: Compute MLE
        '''
        pyro.clear_param_store()
        adam_params = {"lr": 0.01}
        adam = pyro.optim.Adam(adam_params)
        svi = pyro.infer.SVI(self.model_mle, 
                             self.guide_mle, 
                             adam,
                             loss=pyro.infer.Trace_ELBO())

        prev_loss = 0.
        cont = 1
        stepcount = 0
        while cont:
            loss = svi.step()
            print('[iter {}]  loss: {:.4f}'.format(stepcount, loss))
            
            stepcount += 1
            cont = 0
            if stepcount % 100 == 0:
                if stepcount <= max_iter_steps:
                    cont = torch.allclose(torch.tensor(prev_loss), 
                                          torch.tensor(loss), 
                                          rtol=halting_rtol) - 1
                    
                else:
                    cont = 0
                
                
            prev_loss = loss
        
        '''
            Step 2: Compute log-likelihood from MLE locs.
        '''
        print("Run trace to compute log-likelihood.")
        conditioned_model = pyro.condition(self.model_mle, 
                                            data = {'locs' : pyro.param('locs')})
        
        trace = pyro.poutine.trace(conditioned_model).get_trace()
        
        log_likelihood = trace.log_prob_sum()
        print(f"Complete log_likelihood including imputed parameters is {log_likelihood}.")

        self.max_log_like = self.model_mle(mle_locs = pyro.param('locs'))
            
        # assert torch.allclose(self.max_log_like.sum(), log_likelihood, rtol=1e-09)
        
        return self.max_log_like.detach(), pyro.param('locs').detach()
    
    def compute_IC(self):
        '''
            Compute information criteria for each participant individually.
        
            BIC = k*ln(n) - 2*ll --> the lower, the better
            ll = maximized log-likelihood value
            k = number of parameters
            n = number of observations
        '''
        print(f"Computing ICs with mll = {self.max_log_like.sum()}")
        
        assert self.trial_counts.size()[0] == self.num_agents
        
        BIC = torch.tensor(self.agent.num_params)*torch.log(self.trial_counts) -\
            2*self.max_log_like
            
        '''
            AIC = 2*k - 2*ll --> the lower, the better
            ll = maximized log-likelihood value
            k = number of parameters
        '''
        AIC = 2*torch.tensor(self.agent.num_params) - 2*self.max_log_like
        
        return BIC.detach(), AIC.detach()

class CoinflipGroupInference():
    
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
        
        self.agent = agent
        self.trials = agent.trials # length of experiment
        self.num_agents = agent.num_agents # no. of participants
        self.data = group_data # list of dictionaries
        self.num_trials = self.data.shape[-1]
        print(f"{self.num_trials} Trials.")
        self.num_params = len(self.agent.param_names) # number of parameters
        self.loss = []

    def model(self, *args):
        # define hyper priors over model parameters
        # prior over sigma of a Gaussian is a Gamma distribution
        # torch.manual_seed(1234)
        a = pyro.param('a', torch.ones(self.num_params), constraint=dist.constraints.positive)
        lam = pyro.param('lam', torch.ones(self.num_params), constraint=dist.constraints.positive)
        # class Gamma(concentration, rate, validate_args=None)
        # For Gamma distribution(shape, rate), mean is shape/rate, thus 
        # Gamma(a, a/lam) -> mean = lam
        tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1)) 
        
        sig = pyro.deterministic('sig', 1/torch.sqrt(tau))

        # each model parameter has a hyperprior defining group level mean
        # in the form of a Normal distribution
        m = pyro.param('m', torch.zeros(self.num_params))
        s = pyro.param('s', torch.ones(self.num_params), constraint=dist.constraints.positive)
        mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(1)) # Normal(loc, stdev)

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
            
            # pyro.deterministic('agent_params', self.agent.param_dict)
            
            num_particles = locs.shape[0]
            # print("MAKING A ROUND WITH %d PARTICLES"%num_particles)
            
            t = 0
            for trial in range(self.num_trials):
                probs = self.agent.compute_probs()
                # print("HALLO")
                # print(t)
                # print(probs)
                # print(self.data[:, trial])
                # dfgh
                pyro.sample('res_{}'.format(t), 
                            dist.Categorical(probs = probs),
                            obs = self.data[:, t])

                t+=1

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

    def guide_mle(self):
        pass

    def model_mle(self, mle_locs = None):
        
        with pyro.plate('ag_idx', self.num_agents):
            
            if mle_locs is not None:
                print("Using MLE locs.")
                locs = mle_locs
                
            elif mle_locs == None:
                locs = pyro.param('locs', torch.zeros((1, self.num_agents, self.num_params)))
            
            else:
                raise Exception("Sum'thin' wrong.")
            
            print(f"locs = {locs}")
            
            # print(locs.shape)
            "locs is either of shape [num_agents, num_params] or of shape [num_particles, num_agents, num_params]"
            if locs.ndim == 2:
                locs = locs[None, :]
                
            'Shape of locs be [num_particles, num_agents, num_parameters]'
            self.agent.reset(locs)
            
            # pyro.deterministic('agent_params', self.agent.param_dict)
            
            # num_particles = locs.shape[0]
            # print("MAKING A ROUND WITH %d PARTICLES"%num_particles)
            log_like = 0.
            
            t = 0
            for trial in range(self.num_trials):
                probs = self.agent.compute_probs()
                
                log_like += torch.log(probs[range(60), self.data[:,t]])
                # dfgh
                
                pyro.sample('res_{}'.format(t), 
                            dist.Categorical(probs = probs),
                            obs = self.data[:, t])
                
                t+=1
                
            if mle_locs is not None:
                return log_like
        
    def infer_posterior(self,
                        iter_steps = 1_000,
                        num_particles = 10,
                        optim_kwargs = {'lr': .01}):  # Adam learning rate
        """Perform SVI over free model parameters."""

        pyro.clear_param_store()

        svi = pyro.infer.SVI(model=self.model,
                  guide=self.guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  vectorize_particles=True))
        loss = []
        print("Starting inference.")
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

        # param_names = self.agent.param_names

        # 'Original Code'
        # sample_dict = {param: [] for param in param_names}
        # sample_dict["ag_idx"] = []

        # for i in range(n_samples):
        #     sample = self.guide()
        #     for key in sample.keys():
        #         sample.setdefault(key, torch.ones(1))

        #     par_sample = self.agent.locs_to_pars(sample["locs"])

        #     for param in param_names:
        #         sample_dict[param].extend(list(par_sample[param].detach().numpy()))

        #     sample_dict["ag_idx"].extend(list(range(self.num_agents)))

        # sample_df = pd.DataFrame(sample_dict)
        
        from pyro.infer import Predictive
        print("BEGINNING PREDICTIVE.")
        firstlevel_dict = {param:[] for param in self.agent.param_names}

        secondlevel_dict = {param + '_mu':[] for param in self.agent.param_names}
        for param in self.agent.param_names:
            secondlevel_dict[param + '_sig'] = []

        predictive_svi = Predictive(model = self.model,  
                                    guide = self.guide, 
                                    num_samples=n_samples)()

        grouplevel_loc = predictive_svi['mu']
        grouplevel_stdev = predictive_svi['sig']
        predictive_locs = predictive_svi['locs']
        
        predictive_model_params = self.agent.locs_to_pars(predictive_locs)
        
        "1st-level DataFrame"
        for param_name in self.agent.param_names:
            for agidx in range(self.num_agents):
                firstlevel_dict[param_name].append(predictive_model_params[param_name][:, agidx])
        
        "2nd-level DataFrame"
        for param_name_idx in len(self.agent.param_dict.keys()):
            secondlevel_dict[self.agent.param_dict.keys()[param_name_idx] + '_mu'].append(grouplevel_loc[:, ..., param_name_idx])
            secondlevel_dict[self.agent.param_dict.keys()[param_name_idx] + '_sig'].append(grouplevel_loc[:, ..., param_name_idx])


        firstlevel_df = pd.DataFrame(data = firstlevel_dict)
        secondlevel_df = pd.DataFrame(data = secondlevel_dict)
            
        return firstlevel_df, secondlevel_df
    
    def train_mle(self, 
                  iter_steps = 1000,
                  halting_rtol = 1e-09):
        '''
            Step 1: Compute MLE
        '''
        print("Starting MLE estimation.")
        pyro.clear_param_store()
        adam_params = {"lr": 0.01}
        adam = pyro.optim.Adam(adam_params)
        svi = pyro.infer.SVI(self.model_mle, 
                             self.guide_mle, 
                             adam,
                             loss=pyro.infer.Trace_ELBO())

        prev_loss = 0.
        cont = 1
        stepcount = 0
        while cont:
            loss = svi.step()
            # if step % 50 == 0:
            print('[iter {}]  loss: {:.4f}'.format(stepcount, loss))
            # print(f"{torch.allclose(torch.tensor(loss), torch.tensor(prev_loss), rtol=1e-06)}")
            
            stepcount += 1
            cont = 0
            if stepcount % 100 == 0:
                cont = torch.allclose(torch.tensor(prev_loss), 
                                      torch.tensor(loss), 
                                      rtol=halting_rtol) - 1
                
            prev_loss = loss
        
        '''
            Step 2: Compute log-likelihood from the MLE locs.
        '''
        print("Run trace to compute log-likelihood.")
        conditioned_model = pyro.condition(self.model_mle, 
                                            data = {'locs' : pyro.param('locs')})
        
        trace = pyro.poutine.trace(conditioned_model).get_trace()
        
        log_likelihood = trace.log_prob_sum()
        print(f"Complete log_likelihood is {log_likelihood}")

        self.max_log_like = self.model_mle(mle_locs = pyro.param('locs'))
            
        assert torch.allclose(self.max_log_like.sum(), log_likelihood, rtol=1e-09)
        
        return self.max_log_like, pyro.param('locs')

    def compute_IC(self):
        '''
            Compute information criteria for each participant individually.
        
            BIC = k*ln(n) - 2*ll --> the lower, the better
            ll = maximized log-likelihood value
            k = number of parameters
            n = number of observations
        '''
        
        print(f"Computing ICs with mll = {self.max_log_like.sum()}")
        
        BIC = torch.tensor(self.agent.num_params)*torch.log(torch.tensor(self.num_trials)) -\
            2*self.max_log_like
            
        # print(f"BIC is {BIC}")
        
        '''
            AIC = 2*k - 2*ll --> the lower, the better
            ll = maximized log-likelihood value
            k = number of parameters
        '''
            
        AIC = 2*torch.tensor(self.agent.num_params) - 2*self.max_log_like
        # print(f"AIC is {AIC}")
        
        return BIC, AIC
        
        
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
        tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1)) # class Gamma(concentration, rate, validate_args=None)
        
        sig = pyro.deterministic('sig', 1/torch.sqrt(tau)) 

        # each model parameter has a hyperprior defining group level mean
        # in the form of a Normal distribution
        m = pyro.param('m', torch.zeros(self.num_params))
        s = pyro.param('s', torch.ones(self.num_params), constraint=dist.constraints.positive)
        mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(1)) # Normal(log, stdev)

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
                        optim_kwargs = {'lr': .01}):
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

        "Stop after iter_steps steps."
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:#range(iter_steps):
            "Runs through the model twice the first time"
            loss.append(torch.tensor(svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
            if torch.isnan(loss[-1]):
                break

        
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
    