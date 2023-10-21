#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:57:15 2023

@author: sascha
"""

"""
Test torch.func.vmap()
Comparison within same file
Compare outputs of new vmapped update() with old update
"""

import torch

num_agents = 3
blocktype = torch.tensor([0,0,1])
pppchoice = torch.tensor([0,2,1])
ppchoice = torch.tensor([0,3,1])
pchoice = torch.tensor([2,3,1])
choice = torch.tensor([0,3,1])

seq_counter = 4.0 / 4 * torch.ones((num_agents, 2, 6, 6, 6, 6))

def update_habitual(seq_counter_agent, blocktype, ppp, pp, p, c):
    indices = torch.stack(torch.meshgrid(*[torch.arange(size) for size in seq_counter_agent.shape]))
    pos = (blocktype, ppp, pp, p, c)
    
    " Update counter "
    "%6 such that -1 & -2 get mapped to 5 and 4"
    seq_counter_agent = torch.where((indices[0] == pos[0]%6) & 
                        (indices[1] == pos[1]%6) & 
                        (indices[2] == pos[2]%6) &
                        (indices[3] == pos[3]%6) &
                        (indices[4] == pos[4]%6), seq_counter_agent+1, seq_counter_agent)
    
    "Update rep values"
    index = (blocktype, pp, p, c)
    seq_counter_agent.sum()
    # dfgh
    seqs_sum = seq_counter_agent[index + (0,)] + seq_counter_agent[index + (1,)] + \
                seq_counter_agent[index + (2,)] + seq_counter_agent[index + (3,)]
                
    seqs_sum = 2
    
    new_row_agent = torch.tensor([seq_counter_agent[index + (aa,)] / seqs_sum for aa in range(4)])
    
    return seq_counter_agent, new_row_agent


v_update_habitual = torch.func.vmap(update_habitual)

seq_counter, jrepnew = v_update_habitual(seq_counter, 
                                        blocktype, 
                                        pppchoice, 
                                        ppchoice, 
                                        pchoice, 
                                        choice)


#%%

"Test simplification of V computation"

num_particles = 1
num_agents = 1

theta_rep_day1 = torch.rand(num_particles, num_agents)
theta_Q_day1 = torch.rand(num_particles, num_agents)
rep = [torch.rand(num_particles, num_agents, 4)]
Q = [torch.rand(num_particles, num_agents, 4)]

V0 = theta_rep_day1*rep[-1][..., 0] + theta_Q_day1*Q[-1][..., 0] # V-Values for actions (i.e. weighted action values)
V1 = theta_rep_day1*rep[-1][..., 1] + theta_Q_day1*Q[-1][..., 1]
V2 = theta_rep_day1*rep[-1][..., 2] + theta_Q_day1*Q[-1][..., 2]
V3 = theta_rep_day1*rep[-1][..., 3] + theta_Q_day1*Q[-1][..., 3]

Vold = [torch.stack((V0,V1,V2,V3), 2)]

Vnew = [(theta_rep_day1[..., None]*rep[-1] + theta_Q_day1[..., None]*Q[-1])]

assert(Vold[-1].shape==Vnew[-1].shape)
assert(torch.all(Vold[-1]==Vnew[-1]))