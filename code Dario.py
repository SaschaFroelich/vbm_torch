#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:27:56 2023

Dario's Code to compute the exceedance probability of a model.

@author: sascha
"""

with pm.Model() as BMS:
    tau = pm.HalfCauchy('hyper_tau', beta=1.0)
    model_probs = pm.Dirichlet('model_probs', a=pt.ones(num_models) / tau,
                                shape=(num_models))
                                
    def logp(model_evidence, model_probs=model_probs):
        log_likelihood = pm.math.log(model_probs, ) + model_evidence
        return pm.math.sum(pm.logsumexp(log_likelihood, axis=1))
    pm.DensityDist('log_joint', model_probs, logp=logp,
                    observed=logEvidences)
    
    BMSinferenceData = pm.sample(chains=4)
    
def exceedance_probability(traces):
    """Takes the traces of the hierarchical model defined in
    mosq.sequential.hierarchical() and calculates the exceedance
    probabilities for all the componentes of
    traces.get_values('model_probs').    """
    samples = traces #.get_values('model_probs')
    exc_prob = -np.ones(samples.shape[1])
    for best_ix in range(samples.shape[1]):
        exc_prob[best_ix] = sweep_probs(samples, best_ix)
    return exc_prob

def sweep_probs(samples, index):
    """Finds the sweet spot where the best model, indexed by --index--,
    is better than any other.    """
    other_ix = [x for x in range(samples.shape[1]) if x != index]
    c_point = 0.75
    step_size = 0.1
    c_sign = - 1
    while step_size > 0.0001:
        count_other = (samples[:, other_ix] > c_point).sum(axis=0).max()
        count_this = (samples[:, index] <= c_point).sum()
        old_sign = c_sign
        c_sign = 1 - 2 * (count_other < count_this)
        if old_sign != c_sign:
            step_size *= 0.1
        c_point += c_sign * step_size
    return (samples[:, index] >= c_point).sum() / samples.shape[0]