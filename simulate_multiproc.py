#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:40:16 2023

@author: sascha
"""

import ipdb
import torch
import pyro
import pyro.distributions as dist
from tqdm import tqdm
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

import numpy

import models_torch as models
import env 

import pickle

from datetime import datetime
from multiprocessing.pool import Pool

plt.style.use("classic")

def inference(sim_number):
    num_blocks = 14

    parameter = numpy.random.uniform(0,1, 3)
    omega = parameter[0]
    lr = parameter[1]
    dectemp = parameter[2]*5
    
    newagent = models.Vbm(omega=omega, dectemp=dectemp, lr=lr, k=4, Q_init=[0.4, 0., 0., 0.4], num_blocks=num_blocks) # change per console
    newenv = env.Env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8])
    
    newenv.run()
    data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
            "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
                "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}
    
    infer = models.SingleInference(newagent, data)
    loss, params = infer.infer_posterior(iter_steps=200, num_particles = 10)

    inferred_dectemp = params["conc_dectemp"][0]/params["rate_dectemp"][0]
    dectemp_variance = models.gamma_variance(params["conc_dectemp"][0], params["rate_dectemp"][0])
    
    inferred_lr = params["alpha_lr"][0]/(params["alpha_lr"][0]+params["beta_lr"][0])
    lr_variance = models.beta_variance(params["alpha_lr"][0], params["beta_lr"][0])
    
    inferred_omega = params["alpha_omega"][0]/(params["alpha_omega"][0]+params["beta_omega"][0])
    omega_variance = models.beta_variance(params["alpha_omega"][0], params["beta_omega"][0])

    return (loss, params, \
            dectemp, inferred_dectemp, dectemp_variance, \
            lr, inferred_lr, lr_variance, \
            omega, inferred_omega, omega_variance)

#%%
"Parameter Recovery (Single Inference)"

if __name__ == '__main__':
                            
    true_dectemps = []
    inferred_dectemps = []
    dectemp_variance = []
    
    true_lrs = []
    inferred_lrs = []
    lr_variance = []
    
    true_omegas = []
    inferred_omegas = []
    omega_variance = []
    
    print("Starting multiprocessing", flush = True)
    with Pool() as pool:
        # execute tasks in order
        for result in pool.map(inference, range(40)):
            true_dectemps.append(result[2])
            inferred_dectemps.append(result[3])
            dectemp_variance.append(result[4])
            
            true_lrs.append(result[5])
            inferred_lrs.append(result[6])
            lr_variance.append(result[7])
            
            true_omegas.append(result[8])
            inferred_omegas.append(result[9])
            omega_variance.append(result[10])

    pickle_data = (true_dectemps, inferred_dectemps, dectemp_variance, \
                   true_lrs, inferred_lrs, lr_variance,\
                   true_omegas, inferred_omegas, omega_variance)

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
    random_num = numpy.random.rand()
    pickle.dump( pickle_data, open(f"40_param_recovs_{dt_string}_{random_num}.p", "wb" ) ) # change per console
    
    print("Done.")