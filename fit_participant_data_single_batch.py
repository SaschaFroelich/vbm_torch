#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:22:51 2023

@author: sascha
"""

import ipdb
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import pickle

import scipy
import glob

import utils
import models_torch as models
import env 

#%%
"Fit participants (single inference)"

inf_dectemps = []
dectemp_variance = []
inf_dectemps_conc = []
inf_dectemps_rate = []

inf_lrs = []
lr_variance = []
inf_lr_alpha = []
inf_lr_beta = []

inf_omegas = []
omega_variance = []
inf_omega_alpha = []
inf_omega_beta = []

data_dir = "/home/sascha/1TB/TRR/AST_repo/Analyses/clipre/Data/Online_Erhebung/"

pb = -1
for group in range(4):
    files_day1 = glob.glob(data_dir + "Grp%d/csv/*Tag1*.mat"%(group+1))
    
    for file1 in files_day1:
        "Loop over participants"
        pb += 1
        data, _ = utils.get_participant_data(file1, group, data_dir)
                
        if group == 0 or group == 1:
            newagent = models.vbm(omega=0.5, dectemp=2., lr=0., k=4, Q_init=[0.4, 0., 0., 0.4], num_blocks=14)
            
        elif group == 2 or group == 3:
            newagent = models.vbm(omega=0.5, dectemp=2., lr=0., k=4, Q_init=[0., 0.4, 0.4, 0.], num_blocks=14)
        
        """ Single Inference"""
        infer = models.SingleInference(newagent, data)
        loss, params = infer.infer_posterior(iter_steps=200, num_particles = 10)
        
        fig, ax = plt.subplots()
        plt.plot(loss)
        plt.show()
        
        inf_dectemps.append(params["conc_dectemp"][0]/params["rate_dectemp"][0])
        dectemp_variance.append(models.gamma_variance(params["conc_dectemp"][0], params["rate_dectemp"][0]))
        inf_dectemps_conc.append(params["conc_dectemp"][0])
        inf_dectemps_rate.append(params["rate_dectemp"][0])
        
        inf_lrs.append(params["alpha_lr"][0]/(params["alpha_lr"][0]+params["beta_lr"][0]))
        lr_variance.append(models.beta_variance(params["alpha_lr"][0], params["beta_lr"][0]))
        inf_lr_alpha.append(params["alpha_lr"][0])
        inf_lr_beta.append(params["beta_lr"][0])
        
        inf_omegas.append(params["alpha_omega"][0]/(params["alpha_omega"][0]+params["beta_omega"][0]))
        omega_variance.append(models.beta_variance(params["alpha_omega"][0], params["beta_omega"][0]))
        inf_omega_alpha.append(params["alpha_omega"][0])
        inf_omega_beta.append(params["beta_omega"][0])
        
        
pickle_data = (inf_lrs, \
               inf_omegas, \
               inf_dectemps, \
               inf_dectemps_conc, \
               inf_dectemps_rate, \
               inf_lr_alpha, \
               inf_lr_beta, \
               inf_omega_alpha, \
               inf_omega_beta)
    
pickle.dump(pickle_data, open("participants_fit.p", "wb" ) )

#%%
"Fit participants (group inference)"

inf_dectemps = []
dectemp_variance = []
inf_dectemps_conc = []
inf_dectemps_rate = []

inf_lrs = []
lr_variance = []
inf_lr_alpha = []
inf_lr_beta = []

inf_omegas = []
omega_variance = []
inf_omega_alpha = []
inf_omega_beta = []

data_dir = "/home/sascha/1TB/TRR/AST_repo/Analyses/clipre/Data/Online_Erhebung/"
group_data = []
agents = []

pb = -1
for group in range(4):
    files_day1 = glob.glob(data_dir + "Grp%d/csv/*Tag1*.mat"%(group+1))
    
    for file1 in files_day1:
        "Loop over participants"
        pb += 1
        data = utils.get_participant_data(file1, group, data_dir)
        group_data.append(data)
        
        if group == 0 or group == 1:
            newagent = models.vbm(omega=0.5, dectemp=2., lr=0., k=4, Q_init=[0.4, 0., 0., 0.4], num_blocks=14)
            
        elif group == 2 or group == 3:
            newagent = models.vbm(omega=0.5, dectemp=2., lr=0., k=4, Q_init=[0., 0.4, 0.4, 0.], num_blocks=14)
        
        agents.append(newagent)
        
"""Group Inference"""
infer = models.GroupInference(agents, group_data)
loss, params = infer.infer_posterior(iter_steps=200, num_particles = 10)

fig, ax = plt.subplots()
plt.plot(loss)
plt.show()

inf_dectemps.append(params["conc_dectemp"][0]/params["rate_dectemp"][0])
dectemp_variance.append(models.gamma_variance(params["conc_dectemp"][0], params["rate_dectemp"][0]))
inf_dectemps_conc.append(params["conc_dectemp"][0])
inf_dectemps_rate.append(params["rate_dectemp"][0])

inf_lrs.append(params["alpha_lr"][0]/(params["alpha_lr"][0]+params["beta_lr"][0]))
lr_variance.append(models.beta_variance(params["alpha_lr"][0], params["beta_lr"][0]))
inf_lr_alpha.append(params["alpha_lr"][0])
inf_lr_beta.append(params["beta_lr"][0])

inf_omegas.append(params["alpha_omega"][0]/(params["alpha_omega"][0]+params["beta_omega"][0]))
omega_variance.append(models.beta_variance(params["alpha_omega"][0], params["beta_omega"][0]))
inf_omega_alpha.append(params["alpha_omega"][0])
inf_omega_beta.append(params["beta_omega"][0])
        
        
pickle_data = (inf_lrs, \
               inf_omegas, \
               inf_dectemps, \
               inf_dectemps_conc, \
               inf_dectemps_rate, \
               inf_lr_alpha, \
               inf_lr_beta, \
               inf_omega_alpha, \
               inf_omega_beta)
    
pickle.dump(pickle_data, open("participants_fit_group.p", "wb" ) )

#%%

"Plot Single Inference Results"

import matplotlib.cm as cm
import matplotlib.colors as mcolors

inferred = []
inferred.extend(inf_lrs)
inferred.extend(inf_omegas)
inferred.extend(inf_dectemps)

variance = []
variance.extend(lr_variance)
variance.extend(omega_variance)
variance.extend(dectemp_variance)

parameter = []
parameter.extend(["learning rate"]*len(inf_lrs))
parameter.extend(["omega"]*len(inf_omegas))
parameter.extend(["decision temperature"]*len(inf_dectemps))

data = {"inferred": inferred, "parameter": parameter, "variance": variance}
df = pd.DataFrame(data=data)

anal.violin(df)

plt.style.use("seaborn-v0_8-dark")

fig, ax = plt.subplots(1,3)

"ax[0]"
sns.violinplot(ax=ax[0], x="parameter", y="inferred", data=df[df["parameter"]=="learning rate"], color=".8")
sns.stripplot(x="parameter", \
              y="inferred", \
              hue="variance", \
              edgecolor = 'gray', \
              linewidth = 1, \
              data=df[df["parameter"]=="learning rate"], \
              jitter=True, \
              ax=ax[0], \
              palette="coolwarm")
ax[0].legend([],[], frameon=False)
    
"Colorbar"

normalize = mcolors.TwoSlopeNorm(vcenter=(min(lr_variance)+max(lr_variance))/2, vmin=min(lr_variance), \
                                 vmax=max(lr_variance))
colormap = cm.coolwarm
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(lr_variance)
plt.colorbar(scalarmappaple, ax =ax[0], anchor=(-25.,0.5))


"Position"
chartBox = ax[0].get_position()
ax[0].set_position([chartBox.x0-0.2, 
                  chartBox.y0,
                  chartBox.width,
                  chartBox.height])

"ax[1]"
sns.violinplot(ax=ax[1], x="parameter", y="inferred", data=df[df["parameter"]=="omega"], color=".8")
sns.stripplot(x="parameter", \
              y="inferred", \
              hue="variance", \
              edgecolor = 'gray', \
              linewidth=1, \
              data=df[df["parameter"]=="omega"], \
              jitter=True,\
              ax=ax[1], \
              palette="coolwarm")
    
ax[1].legend([],[], frameon=False)

"Colorbar"
normalize = mcolors.TwoSlopeNorm(vcenter=(min(omega_variance)+max(omega_variance))/2, vmin=min(omega_variance), \
                                 vmax=max(omega_variance))
colormap = cm.coolwarm
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(omega_variance)
plt.colorbar(scalarmappaple, ax =ax[1], anchor=(-20.,0.5))

chartBox = ax[1].get_position()
ax[1].set_position([chartBox.x0-0.15, 
                  chartBox.y0,
                  chartBox.width,
                  chartBox.height])

ax[1].set_ylabel("")

"ax[2]"

sns.violinplot(ax=ax[2], x="parameter", y="inferred", data=df[df["parameter"]=="decision temperature"], color=".8")
sns.stripplot(x="parameter", \
              y="inferred", \
              hue="variance", \
              edgecolor = 'gray', \
              linewidth = 1, \
              data=df[df["parameter"]=="decision temperature"], \
              jitter=True, \
              ax=ax[2], \
              palette="coolwarm")

ax[2].legend([],[], frameon=False)
"Colorbar"
normalize = mcolors.TwoSlopeNorm(vcenter=(min(dectemp_variance)+max(dectemp_variance))/2, vmin=min(dectemp_variance), \
                                 vmax=max(dectemp_variance))
colormap = cm.coolwarm
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(dectemp_variance)
plt.colorbar(scalarmappaple, ax =ax[2], anchor=(-15.,0.5))

chartBox = ax[2].get_position()
ax[2].set_position([chartBox.x0-0.12, 
                  chartBox.y0,
                  chartBox.width,
                  chartBox.height])

ax[2].set_ylabel("")

plt.savefig('/home/sascha/Desktop/presentations/AST/Meeting_June_2nd/participant_inference.png')
plt.show()

#%%

"Check correlations"

data_inf = {"learning rate": inf_lrs, "Decision Temp": inf_dectemps, "omega": inf_omegas}
df_inf = pd.DataFrame(data=data_inf)

def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)

g = sns.PairGrid(df_inf)
g.map(sns.scatterplot)

sns.set(style='white', font_scale=1.6)
g = sns.PairGrid(df_inf, aspect=1.4, diag_sharey=False)
g.map_lower(sns.regplot, ci=False, line_kws={'color': 'black'})
g.map_diag(sns.distplot, kde_kws={'color': 'black'})
g.map_upper(corrdot)

r,p = scipy.stats.pearsonr(df_inf["learning rate"], df_inf["omega"])
r,p = scipy.stats.pearsonr(df_inf["learning rate"], df_inf["Decision Temp"])
r,p = scipy.stats.pearsonr(df_inf["omega"], df_inf["Decision Temp"])

#%%

data_dir = "/home/sascha/1TB/TRR/AST_repo/Analyses/clipre/Data/Online_Erhebung/"

pb = -1
for group in range(4):
    files_day1 = glob.glob(data_dir + "Grp%d/csv/*Tag1*.mat"%(group+1))
    
    for file1 in files_day1:
        "Loop over participants"
        pb += 1
        print("=======================================")
        print(file1)
        
        data = utils.get_participant_data(file1, group, data_dir)
    
        omega_inf = inf_omegas[pb]
        dectemp_inf = inf_dectemps[pb]
        lr_inf = inf_lrs[pb]
        
        if group == 0:
            agent_Q_init = [0., 0., 0., 0.]
            env_rewprobs = [0.8, 0.2, 0.2, 0.8]
            block_order = 1
            seqtype = 1
            
        elif group == 1:
            agent_Q_init = [0., 0., 0., 0.]
            env_rewprobs = [0.8, 0.2, 0.2, 0.8]
            block_order = 2
            seqtype = 1
            
        elif group == 2:
            agent_Q_init = [0., 0., 0., 0.]
            env_rewprobs = [0.2, 0.8, 0.8, 0.2]
            block_order = 1
            seqtype = 2
            
        elif group == 3:
            agent_Q_init = [0., 0., 0., 0.]
            env_rewprobs = [0.2, 0.8, 0.8, 0.2]
            block_order = 2
            seqtype = 2
            
        else:
            raise Exception("Da isch a Fehla!")
            
        newagent = models.vbm(omega = omega_inf, dectemp = dectemp_inf, lr=lr_inf, k=4, Q_init = agent_Q_init, num_blocks = 14)
        newenv = env.env(newagent, rewprobs=env_rewprobs)
        newenv.run(block_order = block_order, sequence = seqtype)
            
        data_inf = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
                "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
                    "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}
            
        assert(data["Blocktype"] == data_inf["Blocktype"])
        assert(data["Trialsequence"] == data_inf["Trialsequence"])
        
        data["Jokertypes"] = data_inf["Jokertypes"]
        data["Blockidx"] = data_inf["Blockidx"]
        
        models.plot_results(data, \
                              data_inf, \
                              omega_inf=omega_inf, \
                              dectemp_inf=dectemp_inf,\
                              lr_inf=lr_inf, \
                              ymin=0.3, \
                              group = group)
    

