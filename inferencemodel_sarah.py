#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 12:05:44 2023

@author: SARAH
"""

import torch
from torch.distributions import constraints, biject_to
import pyro
import pyro.distributions as dist
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import distributions as analytical_dists
import pandas as pd

device = torch.device("cpu")

class GeneralGroupInference(object):

    def __init__(self, agent, num_agents, data):

        self.agent = agent
        self.trials = agent.trials
        self.data = data
        self.n_subjects = num_agents
        self.svi = None
        self.loss = []
        self.n_parameters = len(self.agent.param_names)
        print("%d parameters let's go"%self.n_parameters)

    def model(self):
        
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
            locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))

            "locs is either of shape [n_participants, n_parameters] or of shape [n_particles, n_participants, n_parameters]"
            if locs.ndim == 2:
                locs = locs[None, :]
                
            self.agent.reset(locs)
            
            n_particles = locs.shape[0]

            t = -1
            for tau in pyro.markov(range(self.trials)):

                # probs = torch.ones(n_particles, self.n_subjects, 2)
                # choices = torch.ones(self.n_subjects)
                # obs_mask = torch.ones(self.n_subjects, dtype=torch.bool)
                
                # start_time = time.time()
                trial = self.data["Trialsequence"][tau]
                blocktype = self.data["Blocktype"][tau]
                
                if all([self.data["Blockidx"][tau][i] <= 5 for i in range(self.n_subjects)]):
                    day = 1
                    
                elif all([self.data["Blockidx"][tau][i] > 5 for i in range(self.n_subjects)]):
                    day = 2
                    
                else:
                    raise Exception("Da isch a Fehla!")
                
                if all([trial[i] == -1 for i in range(self.n_subjects)]):
                    "Beginning of new block"
                    self.agent.update(torch.tensor([-1]), torch.tensor([-1]), torch.tensor([-1]), day=day, trialstimulus=trial)
                    
                else:
                    current_choice = self.data["Choices"][tau]
                    outcome = self.data["Outcomes"][tau]
                
                if all([trial[i] > 10 for i in range(self.n_subjects)]):
                    "Dual-Target Trial"
                    t+=1
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
                    probs = self.agent.compute_probs(trial, day)
                    "==========================================="
                    
                    choices = torch.tensor([0 if current_choice[idx] == option1[idx] else 1 for idx in range(len(current_choice))])
                    obs_mask = torch.tensor([0 if cc == -10 else 1 for cc in current_choice ]).type(torch.bool)
                    
                if all([trial[i] != -1 for i in range(self.n_subjects)]):
                    "Update (trial == -1 means this is the beginning of a block -> participants didn't see this trial')"
                    self.agent.update(current_choice, outcome, blocktype, day=day, trialstimulus=trial)

                "Sample if dual-target trial and no error was performed"
                if all([trial[i] > 10 for i in range(self.n_subjects)]):
                    # obs & obs_mask shape for inference: [n_subjects] 
                    # probs shape for inference: [n_subjects, nactions] or [n_particles, n_subjects, nactions]
                    # pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), \
                    #             obs = choices.broadcast_to(n_particles, self.n_subjects), \
                    #             obs_mask = obs_mask.broadcast_to(n_particles, self.n_subjects))
                    
                    pyro.sample('res_{}'.format(t), dist.Categorical(probs=probs), \
                                obs = choices)

    # def model(self):
    #     """
    #     Generative model of behavior with a NormalGamma
    #     prior over free model parameters.
    #     """

    #     # define hyper priors over model parameters
    #     a = pyro.param('a', torch.ones(self.n_parameters), constraint=constraints.positive)
    #     lam = pyro.param('lam', torch.ones(self.n_parameters), constraint=constraints.positive)
    #     tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1))

    #     sig = 1/torch.sqrt(tau)

    #     # each model parameter has a hyperprior defining group level mean
    #     m = pyro.param('m', torch.zeros(self.n_parameters))
    #     s = pyro.param('s', torch.ones(self.n_parameters), constraint=constraints.positive)
    #     mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(1))

    #     #for ind in range(self.n_subjects):#pyro.plate("subject", len(self.data)):
    #     with pyro.plate('subject', self.n_subjects) as ind:

    #         base_dist = dist.Normal(0., 1.).expand_by([self.n_parameters]).to_event(1)
    #         transform = dist.transforms.AffineTransform(mu, sig)
    #         locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))

    #         self.agent.reset(locs)

    #         for tau in pyro.markov(range(self.trials)):
    #             for t in range(self.T):

    #                 if t==0:
    #                     prev_response = None
    #                     context = None
    #                 else:
    #                     prev_response = self.data["actions"][tau, t-1]
    #                     context = None

    #                 observation = self.data["observations"][tau, t]

    #                 reward = self.data["rewards"][tau, t]

    #                 self.agent.update_beliefs(tau, t, observation, reward, prev_response, context)

    #                 if t < self.T-1:

    #                     probs = self.agent.perception.posterior_actions[-1]
    #                     if torch.any(torch.isnan(probs)):
    #                         print(tau,t)

    #                     curr_response = self.data["actions"][tau, t]

    #                     # pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(probs.permute(1,2,0)).mask(self.mask[tau]), obs=curr_response)
    #                     pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(probs.permute(1,2,0)), obs=curr_response)


    def guide(self):
        # approximate posterior. assume MF: each param has his own univariate Normal.

        trns = biject_to(constraints.positive)

        m_hyp = pyro.param('m_hyp', torch.zeros(2*self.n_parameters))
        st_hyp = pyro.param('scale_tril_hyp',
                       torch.eye(2*self.n_parameters),
                       constraint=constraints.lower_cholesky)

        hyp = pyro.sample('hyp',
                     dist.MultivariateNormal(m_hyp, scale_tril=st_hyp),
                     infer={'is_auxiliary': True})

        unc_mu = hyp[..., :self.n_parameters]
        unc_tau = hyp[..., self.n_parameters:]

        c_tau = trns(unc_tau)

        ld_tau = trns.inv.log_abs_det_jacobian(c_tau, unc_tau)
        ld_tau = dist.util.sum_rightmost(ld_tau, ld_tau.dim() - c_tau.dim() + 1)

        mu = pyro.sample("mu", dist.Delta(unc_mu, event_dim=1))
        tau = pyro.sample("tau", dist.Delta(c_tau, log_density=ld_tau, event_dim=1))

        m_locs = pyro.param('m_locs', torch.zeros(self.n_subjects, self.n_parameters))
        st_locs = pyro.param('scale_tril_locs',
                        torch.eye(self.n_parameters).repeat(self.n_subjects, 1, 1),
                        constraint=constraints.lower_cholesky)

        with pyro.plate('subject', self.n_subjects):
            locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))

        return {'tau': tau, 'mu': mu, 'locs': locs}

    def init_svi(self, optim_kwargs={'lr': .01},
                 num_particles=10):

        pyro.clear_param_store()

        self.svi = pyro.infer.SVI(model=self.model,
                  guide=self.guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  #set below to true once code is vectorized
                                  vectorize_particles=True))


    def infer_posterior(self,
                        iter_steps=1000, optim_kwargs={'lr': .01},
                                     num_particles=10):
        """Perform SVI over free model parameters.
        """

        #pyro.clear_param_store()
        if self.svi is None:
            self.init_svi(optim_kwargs, num_particles)

        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:#range(iter_steps):
            loss.append(torch.tensor(self.svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % torch.tensor(loss[-20:]).mean())
            if torch.isnan(loss[-1]):
                break

        self.loss += [l.cpu() for l in loss]

        # alpha_lamb_pi = pyro.param("alpha_lamb_pi").data.numpy()
        # beta_lamb_pi = pyro.param("beta_lamb_pi").data.numpy()
        # alpha_lamb_r = pyro.param("alpha_lamb_r").data.numpy()
        # beta_lamb_r = pyro.param("beta_lamb_r").data.numpy()
        # alpha_h = pyro.param("alpha_lamb_r").data.numpy()
        # beta_h = pyro.param("beta_lamb_r").data.numpy()
        # concentration_dec_temp = pyro.param("concentration_dec_temp").data.numpy()
        # rate_dec_temp = pyro.param("rate_dec_temp").data.numpy()

        # param_dict = {"alpha_lamb_pi": alpha_lamb_pi, "beta_lamb_pi": beta_lamb_pi,
        #               "alpha_lamb_r": alpha_lamb_r, "beta_lamb_r": beta_lamb_r,
        #               "alpha_h": alpha_h, "beta_h": beta_h,
        #               "concentration_dec_temp": concentration_dec_temp, "rate_dec_temp": rate_dec_temp}

        # return self.loss#, param_dict

    def sample_posterior_predictive(self, n_samples=5):

        elbo = pyro.infer.Trace_ELBO()
        post_sample_dict = {}

        predictive = pyro.infer.Predictive(model=self.model, guide=self.guide, num_samples=n_samples)
        samples = predictive.get_samples()

        #pbar = tqdm(range(n_samples), position=0)

        # for n in pbar:
        #     pbar.set_description("Sample posterior depth")
        #     # get marginal posterior over planning depths
        #     post_samples = elbo.compute_marginals(self.model, config_enumerate(self.guide))
        #     print(post_samples)
        #     for name in post_samples.keys():
        #         post_sample_dict.setdefault(name, [])
        #         post_sample_dict[name].append(post_samples[name].probs.detach().clone())

        # for name in post_sample_dict.keys():
        #     post_sample_dict[name] = torch.stack(post_sample_dict[name]).numpy()

        # post_sample_df = pd.DataFrame(post_sample_dict)


        reordered_sample_dict = {}
        all_keys = []
        for key in samples.keys():
            if key[:3] != 'res':
                reordered_sample_dict[key] = np.array([])
                all_keys.append(key)

        reordered_sample_dict['subject'] = np.array([])

        #n_subjects = len(self.data)
        for sub in range(self.n_subjects):
            for key in set(all_keys):
                reordered_sample_dict[key] = np.append(reordered_sample_dict[key], samples[key][:,sub].detach().numpy())#.squeeze()
            reordered_sample_dict['subject'] = np.append(reordered_sample_dict['subject'], [sub]*n_samples).squeeze()

        # for key in samples.keys():
        #     if key[:3] != 'res':
        #         sub = int(key[-1])
        #         reordered_sample_dict[key[:-2]] = np.append(reordered_sample_dict[key[:-2]], samples[key].detach().numpy()).squeeze()
        #         reordered_sample_dict['subject'] = np.append(reordered_sample_dict['subject'], [sub]).squeeze()

        sample_df = pd.DataFrame(reordered_sample_dict)

        return sample_df

    def sample_posterior(self, n_samples=5_000):
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

    def analytical_posteriors(self):

        alpha_lamb_pi = pyro.param("alpha_lamb_pi").data.cpu().numpy()
        beta_lamb_pi = pyro.param("beta_lamb_pi").data.cpu().numpy()
        alpha_lamb_r = pyro.param("alpha_lamb_r").data.cpu().numpy()
        beta_lamb_r = pyro.param("beta_lamb_r").data.cpu().numpy()
        alpha_h = pyro.param("alpha_h").data.cpu().numpy()
        beta_h = pyro.param("beta_h").data.cpu().numpy()
        concentration_dec_temp = pyro.param("concentration_dec_temp").data.cpu().numpy()
        rate_dec_temp = pyro.param("rate_dec_temp").data.cpu().numpy()

        param_dict = {"alpha_lamb_pi": alpha_lamb_pi, "beta_lamb_pi": beta_lamb_pi,
                      "alpha_lamb_r": alpha_lamb_r, "beta_lamb_r": beta_lamb_r,
                      "alpha_h": alpha_h, "beta_h": beta_h,
                      "concentration_dec_temp": concentration_dec_temp, "rate_dec_temp": rate_dec_temp}

        x_lamb = np.arange(0.01,1.,0.01)

        y_lamb_pi = analytical_dists.Beta(x_lamb, alpha_lamb_pi, beta_lamb_pi)
        y_lamb_r = analytical_dists.Beta(x_lamb, alpha_lamb_r, beta_lamb_r)
        y_h = analytical_dists.Beta(x_lamb, alpha_h, beta_h)

        x_dec_temp = np.arange(0.01,10.,0.01)

        y_dec_temp = analytical_dists.Gamma(x_dec_temp, concentration=concentration_dec_temp, rate=rate_dec_temp)

        xs = [x_lamb, x_lamb, x_lamb, x_dec_temp]
        ys = [y_lamb_pi, y_lamb_r, y_h, y_dec_temp]

        return xs, ys, param_dict


    def plot_posteriors(self, n_samples=5):

        #df, param_dict = self.sample_posteriors()

        #sample_df = self.sample_posterior_marginals(n_samples=n_samples)

        sample_df = self.sample_posterior(n_samples=n_samples)

        plt.figure()
        sns.displot(data=sample_df, x='h', hue='subject')
        plt.xlim([0,1])
        plt.show()

        plt.figure()
        sns.displot(data=sample_df, x='lamb_r', hue='subject')
        plt.xlim([0,1])
        plt.show()

        plt.figure()
        sns.displot(data=sample_df, x='lamb_pi', hue='subject')
        plt.xlim([0,1])
        plt.show()

        plt.figure()
        sns.displot(data=sample_df, x='dec_temp', hue='subject')
        plt.xlim([0,10])
        plt.show()

        # plt.figure()
        # sns.histplot(marginal_df["h_1"])
        # plt.show()

        return sample_df

    def save_parameters(self, fname):

        pyro.get_param_store().save(fname)

    def load_parameters(self, fname):

        pyro.get_param_store().load(fname)