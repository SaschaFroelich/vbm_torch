#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:38:05 2023

File for data analysis

@author: sascha
"""

import ipdb

import glob

import numpy
import scipy

import utils
import models_torch as models
import env 

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np

import matplotlib.cm as cm
import matplotlib.colors as mcolors

def remap(blockno):
    "Only for groups 1 & 3 (if 0-indexed)"
    if blockno in [1,3,5,7,9,11,13]:
        blockno_new = blockno - 1
        
    elif blockno in [0,2,4,6,8,10,12]:
        blockno_new = blockno + 1
        
    else:
        raise Exception("Fehla!")
    
    return blockno_new

def violin(df, with_colbar = 1, sharey = False):
    plt.style.use("seaborn-v0_8-dark")
    
    npar = len(df["parameter"].unique())
    
    fig, ax = plt.subplots(1, npar, figsize=(15,5), sharey=sharey)
    
    for par in range(npar):
    
        if with_colbar:    
    
            "ax[0]"
            sns.violinplot(ax=ax[par], x="parameter", y="inferred", data=df[df["parameter"]==df["parameter"].unique()[par]], color=".8")
            sns.stripplot(x="parameter", \
                          y="inferred", \
                          hue="variance", \
                          edgecolor = 'gray', \
                          linewidth = 1, \
                          data=df[df["parameter"]==df["parameter"].unique()[par]], \
                          jitter=True, \
                          ax=ax[par], \
                          palette="coolwarm")
            ax[par].legend([],[], frameon=False)
            
            "Position"
            chartBox = ax[par].get_position()
            ax[par].set_position([chartBox.x0+par/4,
                              chartBox.y0,
                              chartBox.width,
                              chartBox.height])
        
            "Colorbar"
            variance = df[df["parameter"]==df["parameter"].unique()[par]]["variance"]
            
            normalize = mcolors.TwoSlopeNorm(vcenter=(min(variance)+max(variance))/2, vmin=min(variance), \
                                              vmax=max(variance))
            colormap = cm.coolwarm
            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
            scalarmappaple.set_array(variance)
            plt.colorbar(scalarmappaple, ax = ax[par])
            
        else:
            "ax[0]"
            g1 = sns.violinplot(ax=ax[par], x="parameter", y="inferred", data=df[df["parameter"]==df["parameter"].unique()[par]], color=".8")
            
            g2 = sns.stripplot(x="parameter", \
                          y="inferred", \
                          edgecolor = 'gray', \
                          linewidth = 1, \
                          data=df[df["parameter"]==df["parameter"].unique()[par]], \
                          jitter=True, \
                          ax=ax[par])
                
            if par > 0:
                g1.set(ylabel=None)
                g2.set(ylabel=None)
            ax[par].legend([],[], frameon=False)
        
    plt.savefig('/home/sascha/Desktop/presentations/AST/June_21_23/modelB.svg')
    
    # "ax[1]"
    # sns.violinplot(ax=ax[1], x="parameter", y="inferred", data=df[df["parameter"]=="omega"], color=".8")
    # sns.stripplot(x="parameter", \
    #               y="inferred", \
    #               hue="variance", \
    #               edgecolor = 'gray', \
    #               linewidth=1, \
    #               data=df[df["parameter"]=="omega"], \
    #               jitter=True,\
    #               ax=ax[1], \
    #               palette="coolwarm")
        
    # ax[1].legend([],[], frameon=False)
    
    # "Colorbar"
    # # normalize = mcolors.TwoSlopeNorm(vcenter=(min(omega_variance)+max(omega_variance))/2, vmin=min(omega_variance), \
    # #                                  vmax=max(omega_variance))
    # # colormap = cm.coolwarm
    # # scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    # # scalarmappaple.set_array(omega_variance)
    # # plt.colorbar(scalarmappaple, ax =ax[1], anchor=(-20.,0.5))
    
    # chartBox = ax[1].get_position()
    # ax[1].set_position([chartBox.x0-0.15, 
    #                   chartBox.y0,
    #                   chartBox.width,
    #                   chartBox.height])
    
    # ax[1].set_ylabel("")
    
    # "ax[2]"
    
    # sns.violinplot(ax=ax[2], x="parameter", y="inferred", data=df[df["parameter"]=="decision temperature"], color=".8")
    # sns.stripplot(x="parameter", \
    #               y="inferred", \
    #               hue="variance", \
    #               edgecolor = 'gray', \
    #               linewidth = 1, \
    #               data=df[df["parameter"]=="decision temperature"], \
    #               jitter=True, \
    #               ax=ax[2], \
    #               palette="coolwarm")
    
    # ax[2].legend([],[], frameon=False)
    
    # "Colorbar"
    # # normalize = mcolors.TwoSlopeNorm(vcenter=(min(dectemp_variance)+max(dectemp_variance))/2, vmin=min(dectemp_variance), \
    # #                                  vmax=max(dectemp_variance))
    # # colormap = cm.coolwarm
    # # scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    # # scalarmappaple.set_array(dectemp_variance)
    # # plt.colorbar(scalarmappaple, ax =ax[2], anchor=(-15.,0.5))
    
    # chartBox = ax[2].get_position()
    # ax[2].set_position([chartBox.x0-0.12, 
    #                   chartBox.y0,
    #                   chartBox.width,
    #                   chartBox.height])
    
    # ax[2].set_ylabel("")
    
    # plt.savefig('/home/sascha/Desktop/presentations/AST/Meeting_June_2nd/participant_inference.png')
    plt.show()
    
def param_corr(df):    
    
    for col in range(len(df.columns)):
        df.rename(columns={df.columns[col] : df.columns[col][4:]}, inplace = True)
    
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
    
    g = sns.PairGrid(df)
    g.map(sns.scatterplot)
    
    sns.set(style='white', font_scale=1.6)
    g = sns.PairGrid(df, aspect=1.4, diag_sharey=False)
    g.map_lower(sns.regplot, ci=False, line_kws={'color': 'black'})
    g.map_diag(sns.distplot, kde_kws={'color': 'black'})
    g.map_upper(corrdot)

    # r,p = scipy.stats.pearsonr(df_inf["learning rate"], df_inf["omega"])
    # r,p = scipy.stats.pearsonr(df_inf["learning rate"], df_inf["Decision Temp"])
    # r,p = scipy.stats.pearsonr(df_inf["omega"], df_inf["Decision Temp"])

def simulate_inferred(data_dir, df, model, k, Q_init, published_results = 0, plot_Qdiff = 0):
    """Generate data based on inferred values and compare real/ simulated data
    data_dir : Directory of behavioural (mat)-files to iterate over
    df : dataframe of all inferred parameters for all participants
    model : which model to use for simulation
    """

    df_group_sim = pd.DataFrame()
    df_group_true = pd.DataFrame()

    pb = -1
    for group in range(4):
        files_day1 = glob.glob(data_dir + "Grp%d/csv/*Tag1*.mat"%(group+1))
        
        for file1 in files_day1:
            "Loop over participants"
            pb += 1
            print("=======================================")
            print(file1)
            
            data, prolific_ID = utils.get_participant_data(file1, group, data_dir, published_results = published_results)
        
            if group == 0:
                agent_Q_init = Q_init
                env_rewprobs = [0.8, 0.2, 0.2, 0.8]
                block_order = 1
                seqtype = 1
                
            elif group == 1:
                agent_Q_init = Q_init
                env_rewprobs = [0.8, 0.2, 0.2, 0.8]
                block_order = 2
                seqtype = 1
                
            elif group == 2:
                agent_Q_init = [Q_init[2], Q_init[3], Q_init[0], Q_init[1]]
                env_rewprobs = [0.2, 0.8, 0.8, 0.2]
                block_order = 1
                seqtype = 2
                
            elif group == 3:
                agent_Q_init = [Q_init[2], Q_init[3], Q_init[0], Q_init[1]]
                env_rewprobs = [0.2, 0.8, 0.8, 0.2]
                block_order = 2
                seqtype = 2
                
            else:
                raise Exception("Da isch a Fehla!")
                
            if model == 'original':
                omega_inf = df[df["participant"]==prolific_ID]["inf_omega"].item()
                dectemp_inf = df[df["participant"]==prolific_ID]["inf_dectemp"].item()
                lr_inf = df[df["participant"]==prolific_ID]["inf_lr"].item()
                    
                newagent = models.vbm(omega = omega_inf, \
                                      dectemp = dectemp_inf, \
                                      lr = lr_inf, k=k, Q_init = agent_Q_init)
            
            elif model == 'A':
                inf_dectemp_day1 = df[df["participant"]==prolific_ID]["inf_dectemp_day1"].item()
                inf_lr_day1 = df[df["participant"]==prolific_ID]["inf_lr_day1"].item()
                inf_omega_day1 = df[df["participant"]==prolific_ID]["inf_omega_day1"].item()
                
                inf_dectemp_day2 = df[df["participant"]==prolific_ID]["inf_dectemp_day2"].item()
                inf_lr_day2 = df[df["participant"]==prolific_ID]["inf_lr_day2"].item()
                inf_omega_day2 = df[df["participant"]==prolific_ID]["inf_omega_day2"].item()
                                
                newagent = models.vbm_A(dectemp_day1 = inf_dectemp_day1, \
                                        lr_day1 = inf_lr_day1, \
                                        omega_day1 = inf_omega_day1, \
                                        dectemp_day2 = inf_dectemp_day2, \
                                        lr_day2 = inf_lr_day2, \
                                        omega_day2 = inf_omega_day2, \
                                        k=k, Q_init = agent_Q_init)
            
            elif model == 'B':
                inf_lr_day1 = df[df["participant"]==prolific_ID]["inf_lr_day1"].item()
                inf_theta_Q_day1 = df[df["participant"]==prolific_ID]["inf_theta_Q_day1"].item()
                inf_theta_rep_day1 = df[df["participant"]==prolific_ID]["inf_theta_rep_day1"].item()
                
                inf_lr_day2 = df[df["participant"]==prolific_ID]["inf_lr_day2"].item()
                inf_theta_Q_day2 = df[df["participant"]==prolific_ID]["inf_theta_Q_day2"].item()
                inf_theta_rep_day2 = df[df["participant"]==prolific_ID]["inf_theta_rep_day2"].item()
                
                newagent = models.vbm_B(lr_day1 = inf_lr_day1, \
                                        theta_Q_day1 = inf_theta_Q_day1, \
                                        theta_rep_day1 = inf_theta_rep_day1, \
                                        lr_day2 = inf_lr_day2, \
                                        theta_Q_day2 = inf_theta_Q_day2, \
                                        theta_rep_day2 = inf_theta_rep_day2, \
                                        k=k, Q_init = agent_Q_init)
                    
            elif model == 'B_onlydual':
                inf_lr_day1 = df[df["participant"]==prolific_ID]["inf_lr_day1"].item()
                inf_theta_Q_day1 = df[df["participant"]==prolific_ID]["inf_theta_Q_day1"].item()
                inf_theta_rep_day1 = df[df["participant"]==prolific_ID]["inf_theta_rep_day1"].item()
                
                inf_lr_day2 = df[df["participant"]==prolific_ID]["inf_lr_day2"].item()
                inf_theta_Q_day2 = df[df["participant"]==prolific_ID]["inf_theta_Q_day2"].item()
                inf_theta_rep_day2 = df[df["participant"]==prolific_ID]["inf_theta_rep_day2"].item()
                
                newagent = models.vbm_B_onlydual(lr_day1 = inf_lr_day1, \
                                        theta_Q_day1 = inf_theta_Q_day1, \
                                        theta_rep_day1 = inf_theta_rep_day1, \
                                        lr_day2 = inf_lr_day2, \
                                        theta_Q_day2 = inf_theta_Q_day2, \
                                        theta_rep_day2 = inf_theta_rep_day2, \
                                        k=k, Q_init = agent_Q_init)
                
                    
            elif model == 'B_2':
                inf_lr_day1_1 = df[df["participant"]==prolific_ID]["inf_lr_day1_1"].item()
                inf_theta_Q_day1_1 = df[df["participant"]==prolific_ID]["inf_theta_Q_day1_1"].item()
                inf_theta_rep_day1_1 = df[df["participant"]==prolific_ID]["inf_theta_rep_day1_1"].item()

                inf_lr_day1_2 = df[df["participant"]==prolific_ID]["inf_lr_day1_2"].item()
                inf_theta_Q_day1_2 = df[df["participant"]==prolific_ID]["inf_theta_Q_day1_2"].item()
                inf_theta_rep_day1_2 = df[df["participant"]==prolific_ID]["inf_theta_rep_day1_2"].item()
                
                inf_lr_day2 = df[df["participant"]==prolific_ID]["inf_lr_day2"].item()
                inf_theta_Q_day2 = df[df["participant"]==prolific_ID]["inf_theta_Q_day2"].item()
                inf_theta_rep_day2 = df[df["participant"]==prolific_ID]["inf_theta_rep_day2"].item()
                
                newagent = models.vbm_B_2(lr_day1_1 = inf_lr_day1_1, \
                                        theta_Q_day1_1 = inf_theta_Q_day1_1, \
                                        theta_rep_day1_1 = inf_theta_rep_day1_1, \
                                            
                                        lr_day1_2 = inf_lr_day1_2, \
                                        theta_Q_day1_2 = inf_theta_Q_day1_2, \
                                        theta_rep_day1_2 = inf_theta_rep_day1_2, \
                                            
                                        lr_day2 = inf_lr_day2, \
                                        theta_Q_day2 = inf_theta_Q_day2, \
                                        theta_rep_day2 = inf_theta_rep_day2, \
                                        k=k, Q_init = agent_Q_init)
                    
            elif model == 'B_3':
                inf_theta_Q_day1_1 = df[df["participant"]==prolific_ID]["inf_theta_Q_day1_1"].item()
                inf_theta_rep_day1_1 = df[df["participant"]==prolific_ID]["inf_theta_rep_day1_1"].item()

                inf_theta_Q_day1_2 = df[df["participant"]==prolific_ID]["inf_theta_Q_day1_2"].item()
                inf_theta_rep_day1_2 = df[df["participant"]==prolific_ID]["inf_theta_rep_day1_2"].item()
                
                inf_theta_Q_day2 = df[df["participant"]==prolific_ID]["inf_theta_Q_day2"].item()
                inf_theta_rep_day2 = df[df["participant"]==prolific_ID]["inf_theta_rep_day2"].item()
                
                newagent = models.vbm_B_3(theta_Q_day1_1 = inf_theta_Q_day1_1, \
                                        theta_rep_day1_1 = inf_theta_rep_day1_1, \
                                            
                                        theta_Q_day1_2 = inf_theta_Q_day1_2, \
                                        theta_rep_day1_2 = inf_theta_rep_day1_2, \
                                            
                                        theta_Q_day2 = inf_theta_Q_day2, \
                                        theta_rep_day2 = inf_theta_rep_day2, \
                                        k=k, Q_init = agent_Q_init)


            print("Group %d"%group)

            "--- Simulate ---"
            if published_results:
                newenv = env.env(newagent, rewprobs=env_rewprobs, matfile_dir = './matlabcode/published/')
            else:
                newenv = env.env(newagent, rewprobs=env_rewprobs, matfile_dir = './matlabcode/clipre/')
                
            
            newenv.run(block_order = block_order, sequence = seqtype)
            
            "--- --- ---"
                
            if model == 'B_3':
                data_inf = {"Choices": newenv.choices, \
                            "Outcomes": newenv.outcomes, \
                            "Trialsequence": newenv.data["trialsequence"], \
                            "Blocktype": newenv.data["blocktype"],\
                            "Jokertypes": newenv.data["jokertypes"], \
                            "Blockidx": newenv.data["blockidx"], \
                            "Participant": pb}
                    
            else:
                data_inf = {"Choices": newenv.choices, \
                            "Outcomes": newenv.outcomes, \
                            "Trialsequence": newenv.data["trialsequence"], \
                            "Blocktype": newenv.data["blocktype"],\
                            "Jokertypes": newenv.data["jokertypes"], \
                            "Blockidx": newenv.data["blockidx"], \
                            "Participant": pb}
                            # "Qdiff": [(newenv.agent.Q[i][...,0] + newenv.agent.Q[i][...,3])/2 - (newenv.agent.Q[i][...,1] + newenv.agent.Q[i][...,2])/2 for i in range(len(newenv.choices))]}
                    
                    
            assert(data["Blocktype"] == data_inf["Blocktype"])
            
            if not data["Trialsequence"] == data_inf["Trialsequence"]:
                ipdb.set_trace()
            
            assert(data["Trialsequence"] == data_inf["Trialsequence"])
            
            data["Jokertypes"] = data_inf["Jokertypes"]
            data["Blockidx"] = data_inf["Blockidx"]
                        
            if model == 'original':
                df_sim, df_true = utils.plot_results(data, \
                                   data_inf, \
                                   omega_inf=omega_inf, \
                                   dectemp_inf=dectemp_inf,\
                                   lr_inf=lr_inf, \
                                   ymin=0.3, \
                                   group = group)
                                   #savedir = "/home/sascha/Desktop/vb_model/vbm_torch/behav_fit/sims_model%s"%model,\
                                   #plotname = prolific_ID)

            else:
                df_sim, df_true = utils.plot_results(data, \
                                   data_inf, \
                                   ymin=0.3, \
                                   group = group)
                                   #savedir = "/home/sascha/Desktop/vb_model/vbm_torch/behav_fit/sims_model%s"%model,\
                                   #plotname = prolific_ID)

            df_sim["Group"] = group
            df_sim["Participant"] = prolific_ID
            
            df_true["Group"] = group
            df_true["Participant"] = prolific_ID
            
            df_group_sim = pd.concat((df_group_sim, df_sim))
            df_group_true = pd.concat((df_group_true, df_true))
                        
    "Groups 0 and 2 saw the same block order, groups 1 and 3 saw the same block order"
    "Change Blockidx -> Blockno, so that Block no 0 is the random block for each participant"
    df_group_sim.rename(columns = {"Blockidx": "Blockno"}, inplace = True)
    df_group_true.rename(columns = {"Blockidx": "Blockno"}, inplace = True)
    for grp in [1, 3]:
        pbs = df_group_sim[df_group_sim["Group"] == grp]["Participant"].unique()
        
        for p in pbs:
            df_group_sim.loc[df_group_sim["Participant"] == p, "Blockno"] = df_group_sim.loc[df_group_sim["Participant"] == p, "Blockno"].map(lambda x: x-1 if x in [1,3,5,7,9,11,13] else x+1)
            df_group_true.loc[df_group_true["Participant"] == p, "Blockno"] = df_group_true.loc[df_group_true["Participant"] == p, "Blockno"].map(lambda x: x-1 if x in [1,3,5,7,9,11,13] else x+1)

    df_group_sim["datatype"] = "simulated"
    df_group_true["datatype"] = "experimental"
    
    "Plot group-level behaviour"
    # ipdb.set_trace()
    df_together = pd.concat((df_group_sim, df_group_true))
    fig, ax = plt.subplots()
    sns.relplot(x="Blockno", y="HPCF", hue = "Trialtype", data=df_together, kind="line", col = "datatype")
    
    # g = sns.relplot(x="Blockno", y="HPCF", hue = "Trialtype", data=df_together, kind="line", col = "datatype")
    # for (row_val, col_val), ax in g.axes_dict.items():
    #     if col_val == 'datatype = simulated' or col_val == 'datatype = actual':
    #         ax.plot([5.5, 5.5], [0.5, 1], color='k')
    
    # plt.plot([5.5, 5.5], [0.5, 1], color='k')
    # plt.title(f"Model {model}")
    plt.show()
