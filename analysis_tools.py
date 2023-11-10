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

def violin(df, 
           with_colbar = 1, 
           sharey = False, 
           ylims = None):
    '''

    Parameters
    ----------
    df : DataFrame
        DESCRIPTION.
        
    with_colbar : bool, optional
        DESCRIPTION. The default is 1.
        
    sharey : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    '''
    
    model = df['model'].unique()[0]
    df = df.drop(['model', 'ag_idx', 'group'], axis = 1)
    
    num_params = len(df.columns)
    
    fig, ax = plt.subplots(1, num_params, figsize=(15,5), sharey=0)
    
    if model == 'B':
        ylims = [[0, 0.04], # lr
                  [0.5, 7.5], # theta_Q
                  [0., 2.], # theta_rep
                  [0, 0.04], # lr
                  [0.5, 7.5], # theta_Q
                  [0., 2]] # theta_rep
        
    elif model == 'Conflict':
        ylims = [[0, 0.04], # lr
                  [0, 8], # theta_Q
                  [0.5, 5], # theta_rep
                  [-0.4, 0.5], # conflict param
                  [0, 0.04], # lr
                  [0, 8], # theta_Q
                  [0.5, 5], # theta_rep
                  [-0.5, 0.5]] # conflict param
        
    elif model == 'Seqparam':
        ylims = [[0, 0.04], # lr
                  [0., 8], # theta_Q
                  [0., 2], # theta_rep
                  [0.05, 0.1], # seqparam
                  [0, 0.04], # lr
                  [0., 8], # theta_Q
                  [0., 2], # theta_rep
                  [0.05, 0.1]] # seqparam
        
    elif model == 'Bhand':
        ylims = [[0, 0.04], # lr
                  [0., 8], # theta_Q
                  [0., 2], # theta_rep
                  [0, 0.04], # lr
                  [0., 8], # theta_Q
                  [0., 2], # theta_rep
                  [-2.5, 2.5]] # hand_param
        
    else:
        del ylims
    
    for par in range(num_params):
        
        if 1:    
            "With colorbar"
            "ax[0]"
            dataseries = (df.melt()[df.melt()['variable'] == df.columns[par]])
            dataseries['value'] = pd.to_numeric(dataseries['value'], errors='coerce')
            
            sns.violinplot(ax = ax[par], 
                           x = 'variable',
                           y = 'value',
                           data = dataseries,
                           color=".8")
            
            sns.stripplot(x = 'variable',
                          y = 'value',
                          data = dataseries,
                          edgecolor = 'gray',
                          linewidth = 1,
                          jitter=True,
                          ax=ax[par])
                          # palette="coolwarm")
            
            ax[par].legend([],[], frameon=False)
            
            "Position"
            chartBox = ax[par].get_position()
            ax[par].set_position([chartBox.x0+par/64,
                              chartBox.y0,
                              chartBox.width,
                              chartBox.height])
            
            if ylims is not None:
                ax[par].set_ylim(ylims[par])
        
            "Colorbar"
            # variance = df[params_df.columns[par]].std()**2
            
            # normalize = mcolors.TwoSlopeNorm(vcenter=(min(variance)+max(variance))/2, 
            #                                  vmin=min(variance), 
            #                                  vmax=max(variance))
            
            # colormap = cm.coolwarm
            # scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
            # scalarmappaple.set_array(variance)
            # plt.colorbar(scalarmappaple, ax = ax[par])
            
        else:
            "Without colorbar"
            "ax[0]"
            g1 = sns.violinplot(ax=ax[par], 
                                x="parameter", 
                                y="inferred", 
                                data=df[df["parameter"]==df["parameter"].unique()[par]], 
                                color=".8")
            
            g2 = sns.stripplot(x="parameter",
                          y="inferred",
                          edgecolor = 'gray',
                          linewidth = 1,
                          data = df[df["parameter"]==df["parameter"].unique()[par]],
                          jitter = True,
                          ax = ax[par])
                
            if par > 0:
                g1.set(ylabel=None)
                g2.set(ylabel=None)
                
            ax[par].legend([],[], frameon=False)
    
    plt.show()

def param_corr(df):    
    
    df = df.drop(['ag_idx', 'model', 'group'], axis = 1)
    
    # for col in range(len(df.columns)):
    #     df.rename(columns={df.columns[col] : df.columns[col][4:]}, inplace = True)
    
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

    plt.show()
    # r,p = scipy.stats.pearsonr(df_inf["learning rate"], df_inf["omega"])
    # r,p = scipy.stats.pearsonr(df_inf["learning rate"], df_inf["Decision Temp"])
    # r,p = scipy.stats.pearsonr(df_inf["omega"], df_inf["Decision Temp"])
    
    
def within_subject_corr(df):
    '''

    Parameters
    ----------
    df : DataFrame
        Columns
            Parameters
            ag_idx

    Returns
    -------
    corr_dict : TYPE
        DESCRIPTION.

    '''
    
    if 'ID' in df.columns:
        df_temp = df.drop(['model', 'group', 'ag_idx', 'ID'], axis = 1)
        df = df.drop(['model', 'group', 'ID'], axis = 1)
        num_params = len(df_temp.columns)
        param_names = df_temp.columns
        
    else:
        df_temp = df.drop(['model', 'group', 'ag_idx'], axis = 1)
        df = df.drop(['model', 'group'], axis = 1)
        num_params = len(df_temp.columns)
        param_names = df_temp.columns
        
    corr_dict = {}
    for param1_idx in range(num_params):
        for param2_idx in range(param1_idx+1, num_params):
            corr_dict[param_names[param1_idx]+'_vs_' + param_names[param2_idx]] = []
    
    for ag_idx in np.sort(df['ag_idx'].unique()):
        df_ag = df[df['ag_idx'] == ag_idx]
        for param1_idx in range(num_params):
            for param2_idx in range(param1_idx+1, num_params):
                corr_dict[param_names[param1_idx] + '_vs_' + param_names[param2_idx]].append(\
                           df_ag.loc[:, param_names[param1_idx]].corr(df_ag.loc[:, param_names[param2_idx]]))

    return corr_dict

# def simulate_inferred(data_dir, df, model, k, Q_init, published_results = 0, plot_Qdiff = 0):
#     """Generate data based on inferred values and compare real/ simulated data
#     data_dir : Directory of behavioural (mat)-files to iterate over
#     df : dataframe of all inferred parameters for all participants
#     model : which model to use for simulation
#     """

#     df_group_sim = pd.DataFrame()
#     df_group_true = pd.DataFrame()

#     pb = -1
#     for group in range(4):
#         files_day1 = glob.glob(data_dir + "Grp%d/csv/*Tag1*.mat"%(group+1))
        
#         for file1 in files_day1:
#             "Loop over participants"
#             pb += 1
#             print("=======================================")
#             print(file1)
            
#             data, prolific_ID = utils.get_participant_data(file1, group, data_dir, published_results = published_results)
        
#             if group == 0:
#                 agent_Q_init = Q_init
#                 env_rewprobs = [0.8, 0.2, 0.2, 0.8]
#                 block_order = 1
#                 seqtype = 1
                
#             elif group == 1:
#                 agent_Q_init = Q_init
#                 env_rewprobs = [0.8, 0.2, 0.2, 0.8]
#                 block_order = 2
#                 seqtype = 1
                
#             elif group == 2:
#                 agent_Q_init = [Q_init[2], Q_init[3], Q_init[0], Q_init[1]]
#                 env_rewprobs = [0.2, 0.8, 0.8, 0.2]
#                 block_order = 1
#                 seqtype = 2
                
#             elif group == 3:
#                 agent_Q_init = [Q_init[2], Q_init[3], Q_init[0], Q_init[1]]
#                 env_rewprobs = [0.2, 0.8, 0.8, 0.2]
#                 block_order = 2
#                 seqtype = 2
                
#             else:
#                 raise Exception("Da isch a Fehla!")
                
#             if model == 'original':
#                 omega_inf = df[df["participant"]==prolific_ID]["inf_omega"].item()
#                 dectemp_inf = df[df["participant"]==prolific_ID]["inf_dectemp"].item()
#                 lr_inf = df[df["participant"]==prolific_ID]["inf_lr"].item()
                    
#                 newagent = models.Vbm(omega = omega_inf, \
#                                       dectemp = dectemp_inf, \
#                                       lr = lr_inf, k=k, Q_init = agent_Q_init)
            
#             elif model == 'A':
#                 inf_dectemp_day1 = df[df["participant"]==prolific_ID]["inf_dectemp_day1"].item()
#                 inf_lr_day1 = df[df["participant"]==prolific_ID]["inf_lr_day1"].item()
#                 inf_omega_day1 = df[df["participant"]==prolific_ID]["inf_omega_day1"].item()
                
#                 inf_dectemp_day2 = df[df["participant"]==prolific_ID]["inf_dectemp_day2"].item()
#                 inf_lr_day2 = df[df["participant"]==prolific_ID]["inf_lr_day2"].item()
#                 inf_omega_day2 = df[df["participant"]==prolific_ID]["inf_omega_day2"].item()
                                
#                 newagent = models.Vbm_A(dectemp_day1 = inf_dectemp_day1, \
#                                         lr_day1 = inf_lr_day1, \
#                                         omega_day1 = inf_omega_day1, \
#                                         dectemp_day2 = inf_dectemp_day2, \
#                                         lr_day2 = inf_lr_day2, \
#                                         omega_day2 = inf_omega_day2, \
#                                         k=k, Q_init = agent_Q_init)
            
#             elif model == 'B':
#                 inf_lr_day1 = df[df["participant"]==prolific_ID]["inf_lr_day1"].item()
#                 inf_theta_Q_day1 = df[df["participant"]==prolific_ID]["inf_theta_Q_day1"].item()
#                 inf_theta_rep_day1 = df[df["participant"]==prolific_ID]["inf_theta_rep_day1"].item()
                
#                 inf_lr_day2 = df[df["participant"]==prolific_ID]["inf_lr_day2"].item()
#                 inf_theta_Q_day2 = df[df["participant"]==prolific_ID]["inf_theta_Q_day2"].item()
#                 inf_theta_rep_day2 = df[df["participant"]==prolific_ID]["inf_theta_rep_day2"].item()
                
#                 newagent = models.Vbm_B(lr_day1 = inf_lr_day1, \
#                                         theta_Q_day1 = inf_theta_Q_day1, \
#                                         theta_rep_day1 = inf_theta_rep_day1, \
#                                         lr_day2 = inf_lr_day2, \
#                                         theta_Q_day2 = inf_theta_Q_day2, \
#                                         theta_rep_day2 = inf_theta_rep_day2, \
#                                         k=k, Q_init = agent_Q_init)
                    
#             elif model == 'B_onlydual':
#                 inf_lr_day1 = df[df["participant"]==prolific_ID]["inf_lr_day1"].item()
#                 inf_theta_Q_day1 = df[df["participant"]==prolific_ID]["inf_theta_Q_day1"].item()
#                 inf_theta_rep_day1 = df[df["participant"]==prolific_ID]["inf_theta_rep_day1"].item()
                
#                 inf_lr_day2 = df[df["participant"]==prolific_ID]["inf_lr_day2"].item()
#                 inf_theta_Q_day2 = df[df["participant"]==prolific_ID]["inf_theta_Q_day2"].item()
#                 inf_theta_rep_day2 = df[df["participant"]==prolific_ID]["inf_theta_rep_day2"].item()
                
#                 newagent = models.Vbm_B_onlydual(lr_day1 = inf_lr_day1, \
#                                         theta_Q_day1 = inf_theta_Q_day1, \
#                                         theta_rep_day1 = inf_theta_rep_day1, \
#                                         lr_day2 = inf_lr_day2, \
#                                         theta_Q_day2 = inf_theta_Q_day2, \
#                                         theta_rep_day2 = inf_theta_rep_day2, \
#                                         k=k, Q_init = agent_Q_init)
                
                    
#             elif model == 'B_2':
#                 inf_lr_day1_1 = df[df["participant"]==prolific_ID]["inf_lr_day1_1"].item()
#                 inf_theta_Q_day1_1 = df[df["participant"]==prolific_ID]["inf_theta_Q_day1_1"].item()
#                 inf_theta_rep_day1_1 = df[df["participant"]==prolific_ID]["inf_theta_rep_day1_1"].item()

#                 inf_lr_day1_2 = df[df["participant"]==prolific_ID]["inf_lr_day1_2"].item()
#                 inf_theta_Q_day1_2 = df[df["participant"]==prolific_ID]["inf_theta_Q_day1_2"].item()
#                 inf_theta_rep_day1_2 = df[df["participant"]==prolific_ID]["inf_theta_rep_day1_2"].item()
                
#                 inf_lr_day2 = df[df["participant"]==prolific_ID]["inf_lr_day2"].item()
#                 inf_theta_Q_day2 = df[df["participant"]==prolific_ID]["inf_theta_Q_day2"].item()
#                 inf_theta_rep_day2 = df[df["participant"]==prolific_ID]["inf_theta_rep_day2"].item()
                
#                 newagent = models.Vbm_B_2(lr_day1_1 = inf_lr_day1_1, \
#                                         theta_Q_day1_1 = inf_theta_Q_day1_1, \
#                                         theta_rep_day1_1 = inf_theta_rep_day1_1, \
                                            
#                                         lr_day1_2 = inf_lr_day1_2, \
#                                         theta_Q_day1_2 = inf_theta_Q_day1_2, \
#                                         theta_rep_day1_2 = inf_theta_rep_day1_2, \
                                            
#                                         lr_day2 = inf_lr_day2, \
#                                         theta_Q_day2 = inf_theta_Q_day2, \
#                                         theta_rep_day2 = inf_theta_rep_day2, \
#                                         k=k, Q_init = agent_Q_init)
                    
#             elif model == 'B_3':
#                 inf_theta_Q_day1_1 = df[df["participant"]==prolific_ID]["inf_theta_Q_day1_1"].item()
#                 inf_theta_rep_day1_1 = df[df["participant"]==prolific_ID]["inf_theta_rep_day1_1"].item()

#                 inf_theta_Q_day1_2 = df[df["participant"]==prolific_ID]["inf_theta_Q_day1_2"].item()
#                 inf_theta_rep_day1_2 = df[df["participant"]==prolific_ID]["inf_theta_rep_day1_2"].item()
                
#                 inf_theta_Q_day2 = df[df["participant"]==prolific_ID]["inf_theta_Q_day2"].item()
#                 inf_theta_rep_day2 = df[df["participant"]==prolific_ID]["inf_theta_rep_day2"].item()
                
#                 newagent = models.Vbm_B_3(theta_Q_day1_1 = inf_theta_Q_day1_1, \
#                                         theta_rep_day1_1 = inf_theta_rep_day1_1, \
                                            
#                                         theta_Q_day1_2 = inf_theta_Q_day1_2, \
#                                         theta_rep_day1_2 = inf_theta_rep_day1_2, \
                                            
#                                         theta_Q_day2 = inf_theta_Q_day2, \
#                                         theta_rep_day2 = inf_theta_rep_day2, \
#                                         k=k, Q_init = agent_Q_init)


#             print("Group %d"%group)

#             "--- Simulate ---"
#             if published_results:
#                 newenv = env.Env(newagent, rewprobs=env_rewprobs, matfile_dir = './matlabcode/published/')
#             else:
#                 newenv = env.Env(newagent, rewprobs=env_rewprobs, matfile_dir = './matlabcode/clipre/')
                
            
#             newenv.run(block_order = block_order, sequence = seqtype)
            
#             "--- --- ---"
                
#             if model == 'B_3':
#                 data_inf = {"Choices": newenv.choices, \
#                             "Outcomes": newenv.outcomes, \
#                             "Trialsequence": newenv.data["trialsequence"], \
#                             "Blocktype": newenv.data["blocktype"],\
#                             "Jokertypes": newenv.data["jokertypes"], \
#                             "Blockidx": newenv.data["blockidx"], \
#                             "Participant": pb}
                    
#             else:
#                 data_inf = {"Choices": newenv.choices, \
#                             "Outcomes": newenv.outcomes, \
#                             "Trialsequence": newenv.data["trialsequence"], \
#                             "Blocktype": newenv.data["blocktype"],\
#                             "Jokertypes": newenv.data["jokertypes"], \
#                             "Blockidx": newenv.data["blockidx"], \
#                             "Participant": pb}
#                             # "Qdiff": [(newenv.agent.Q[i][...,0] + newenv.agent.Q[i][...,3])/2 - (newenv.agent.Q[i][...,1] + newenv.agent.Q[i][...,2])/2 for i in range(len(newenv.choices))]}
                    
                    
#             assert(data["Blocktype"] == data_inf["Blocktype"])
            
#             if not data["Trialsequence"] == data_inf["Trialsequence"]:
#                 ipdb.set_trace()
            
#             assert(data["Trialsequence"] == data_inf["Trialsequence"])
            
#             data["Jokertypes"] = data_inf["Jokertypes"]
#             data["Blockidx"] = data_inf["Blockidx"]
                        
#             if model == 'original':
#                 df_sim, df_true = utils.plot_results(data, \
#                                    data_inf, \
#                                    omega_inf=omega_inf, \
#                                    dectemp_inf=dectemp_inf,\
#                                    lr_inf=lr_inf, \
#                                    ymin=0.3, \
#                                    group = group)
#                                    #savedir = "/home/sascha/Desktop/vb_model/vbm_torch/behav_fit/sims_model%s"%model,\
#                                    #plotname = prolific_ID)

#             else:
#                 df_sim, df_true = utils.plot_results(data, \
#                                    data_inf, \
#                                    ymin=0.3, \
#                                    group = group)
#                                    #savedir = "/home/sascha/Desktop/vb_model/vbm_torch/behav_fit/sims_model%s"%model,\
#                                    #plotname = prolific_ID)

#             df_sim["Group"] = group
#             df_sim["Participant"] = prolific_ID
            
#             df_true["Group"] = group
#             df_true["Participant"] = prolific_ID
            
#             df_group_sim = pd.concat((df_group_sim, df_sim))
#             df_group_true = pd.concat((df_group_true, df_true))
                        
#     "Groups 0 and 2 saw the same block order, groups 1 and 3 saw the same block order"
#     "Change Blockidx -> Blockno, so that Block no 0 is the random block for each participant"
#     df_group_sim.rename(columns = {"Blockidx": "Blockno"}, inplace = True)
#     df_group_true.rename(columns = {"Blockidx": "Blockno"}, inplace = True)
#     for grp in [1, 3]:
#         pbs = df_group_sim[df_group_sim["Group"] == grp]["Participant"].unique()
        
#         for p in pbs:
#             df_group_sim.loc[df_group_sim["Participant"] == p, "Blockno"] = df_group_sim.loc[df_group_sim["Participant"] == p, "Blockno"].map(lambda x: x-1 if x in [1,3,5,7,9,11,13] else x+1)
#             df_group_true.loc[df_group_true["Participant"] == p, "Blockno"] = df_group_true.loc[df_group_true["Participant"] == p, "Blockno"].map(lambda x: x-1 if x in [1,3,5,7,9,11,13] else x+1)

#     df_group_sim["datatype"] = "simulated"
#     df_group_true["datatype"] = "experimental"
    
#     "Plot group-level behaviour"
#     # ipdb.set_trace()
#     df_together = pd.concat((df_group_sim, df_group_true))
#     fig, ax = plt.subplots()
#     sns.relplot(x="Blockno", y="HPCF", hue = "Trialtype", data=df_together, kind="line", col = "datatype")
    
#     # g = sns.relplot(x="Blockno", y="HPCF", hue = "Trialtype", data=df_together, kind="line", col = "datatype")
#     # for (row_val, col_val), ax in g.axes_dict.items():
#     #     if col_val == 'datatype = simulated' or col_val == 'datatype = actual':
#     #         ax.plot([5.5, 5.5], [0.5, 1], color='k')
    
#     # plt.plot([5.5, 5.5], [0.5, 1], color='k')
#     # plt.title(f"Model {model}")
#     plt.show()

def cluster_analysis(corr_dict, title= ''):
    import ipdb
    import scipy.cluster.hierarchy as spc
    keys = corr_dict.keys()
    num_agents = len(corr_dict[list(keys)[0]])
    num_corrs = len(corr_dict.keys())
    
    distance = np.zeros((num_agents, num_agents))
    
    for ag_idx1 in range(num_agents):
        for ag_idx2 in range(num_agents):
            v1 = np.zeros(len(corr_dict))
            v2 = np.zeros(len(corr_dict))
            
            for key_idx in range(num_corrs):
                key = list(corr_dict.keys())[key_idx]
                v1[key_idx] = corr_dict[key][ag_idx1]
                v2[key_idx] = corr_dict[key][ag_idx2]
    
            distance[ag_idx1, ag_idx2] = np.sqrt(np.sum((v1 - v2)**2))
    
    # plt.style.use('default')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(distance)
    # fig.colorbar(cax)
    # plt.show()
    
    def upper_tri_indexing(A):
        m = A.shape[0]
        r,c = np.triu_indices(m,1)
        return A[r,c]
    
    "----- Plot clusters"
    distance_vec = upper_tri_indexing(distance)
    linkage = spc.linkage(distance_vec, method='single')
    idx = spc.fcluster(linkage, 0.5 * distance_vec.max(), 'distance')
    dn = spc.dendrogram(linkage)
    plt.title(f"Clusters ({title})")
    plt.gca().set_xlabel('ag_idx (leavnodes)')
    plt.show()
    
    "----- Plot ordered matrix"
    leavnodes = [int(node) for node in dn['ivl']]
    plt.style.use('default')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(distance[leavnodes,:][:, leavnodes])
    fig.colorbar(cax)
    plt.title(f"Similarity matrix ({title})")
    plt.gca().set_ylabel('leavnode idx')
    plt.show()
    # dfgh
    # "----- With spc"
    # pdist = spc.distance.pdist(pd.DataFrame(corr_dict))
    # "Linkage matrix"
    # linkage = spc.linkage(pdist, method='single')
    # idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')
    # dn = spc.dendrogram(linkage)
    # plt.show()
    # "---"
    
    return leavnodes

def compare_lists(leavnode1, leavnode2):
    leavnode1 = list(leavnode1)
    leavnode2 = list(leavnode2)
    
    l1 = 0 # num of elements only in leavnode1
    l2 = 0 # num of elements only in leavnode1
    union = 0
    
    while len(leavnode1):
        element = leavnode1.pop()
        
        if element in leavnode2:
            union += 1
            
        else:
            l1 += 1
            
    l2 = len(leavnode2) - union
    
    print(f"Number of elements only in list 1: {l1}.\n"+\
          f"Number of elements only in list 2: {l2}.\n"+\
          f"Number of elements in union: {union}.")
        
        
def kmeans(corr_dict, inf_mean_df, n_clusters, num_reps = 1, plotfig = True):
    '''
    

    Parameters
    ----------
    corr_dict : TYPE
        DESCRIPTION.
        
    inf_mean_df : TYPE
        DESCRIPTION.
        
    n_clusters : TYPE
        DESCRIPTION.
        
    num_reps : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.
        
    cluster_groups : nested list
        Contains the ag_idx for the different clusters.
        
    c_distances : array
        Distances between clusters.
    '''
    
    num_agents = len(inf_mean_df['ag_idx'].unique())
    
    from sklearn.cluster import KMeans
    n_clusters = n_clusters
    kmean = KMeans(n_clusters=n_clusters, n_init = 10, init ='random')
    
    all_cluster_groups = []
    all_labels = []
    mini_clusters = np.zeros((num_agents, num_agents))
    c_distances = []
    
    for rep in range(num_reps):
        kmeans = kmean.fit(pd.DataFrame(corr_dict).to_numpy())
        
        for clus1 in range(n_clusters):
            for clus2 in range(clus1+1, n_clusters):
                cluster_distance = np.sqrt(((kmeans.cluster_centers_[clus1,:]-kmeans.cluster_centers_[clus2,:])**2).sum())
                c_distances.append(cluster_distance)
                print(f"Cluster distance (cluster {clus1} and {clus2}) is %.4f"%cluster_distance)
        
        
        cluster_groups = [[None]]*n_clusters
        
        for clus in range(n_clusters):
            cluster_groups[clus] = list(inf_mean_df['ag_idx'][np.where(kmeans.labels_ == clus)[0]])
            cluster_groups[clus].sort()
        
        for clus_idx in range(len(cluster_groups)):
            groups_in_cluster = inf_mean_df['group'][cluster_groups[clus_idx]]
            num_groups_in_cluster = len(groups_in_cluster.unique())
            print("\n\nThere are %d groups in cluster no %d."%(num_groups_in_cluster, clus_idx))
            group_distr = [None]*num_groups_in_cluster
            
            for i in range(num_groups_in_cluster):
                group_distr[i] = (groups_in_cluster == groups_in_cluster.unique()[i]).sum()
                
            print("Experimental groups in cluster %d are distributed as follows:"%clus_idx)
            print(group_distr)
        
        for row_idx in range(num_agents):
            for col_idx in range(num_agents):
                for cgroup in cluster_groups:
                    if row_idx in cgroup and col_idx in cgroup:
                        mini_clusters[row_idx, col_idx] += 1
        
        
        all_cluster_groups.append(cluster_groups)
        all_labels.append(kmeans.labels_)
        
    mini_clusters /= num_reps

    if num_reps > 1 and plotfig:    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(mini_clusters)
        fig.colorbar(cax)
        plt.title(f"Mini Clusters for {n_clusters} clusters")
        plt.gca().set_ylabel('ag_idx')
        plt.grid(False)
        plt.show()
        
        # leavnodes = cluster_analysis(mini_clusters, title = 'mini clusters')
        # print("leavnodes mini clusters:")
        # print(leavnodes)
        
    # '''
    # Which cluster is the 'most stable' mini cluster?
    # '''
    # for i in range(num_reps):
    #     compare_lists(leavnode1, leavnode2)
        
    # dfgh
        
    return kmeans, cluster_groups, c_distances

def compute_errors(df):
    
    df = df[df['choices'] != -1]
    # df_dtt = df[df['trialsequence'] > 10]
    # df_stt = df[df['trialsequence'] < 10]
    
    error_rates = pd.DataFrame()
    
    # pd.DataFrame(df_dtt.groupby(['ID', 'model', 'ag_idx', 'group'], as_index=False))
    
    group = []
    ag_idx = []
    IDs = []
    ER_dtt = []
    ER_stt = []
    ER_total = []
    for ID in df['ID'].unique():
        ER_dtt.append(len(df[(df['trialsequence'] > 10) & (df['ID'] == ID) & (df['choices'] == -2)]) /\
                 len(df[(df['trialsequence'] > 10) & (df['ID'] == ID)]))

        ER_stt.append(len(df[(df['trialsequence'] < 10) & (df['ID'] == ID) & (df['choices'] == -2)]) /\
                 len(df[(df['trialsequence'] < 10) & (df['ID'] == ID)]))
        
        ER_total.append(len(df[(df['trialsequence'] > -1) & (df['ID'] == ID) & (df['choices'] == -2)]) /\
                        len(df[(df['trialsequence'] > -1) & (df['ID'] == ID)]))
        
        IDs.append(ID)
        group.append(df[df['ID'] == ID]['group'].unique()[0])
        ag_idx.append(df[df['ID'] == ID]['ag_idx'].unique()[0])
    
    er_df = pd.DataFrame({'ag_idx': ag_idx,
                          'group': group,
                          'ID': IDs,
                          'ER_dtt': ER_dtt,
                          'ER_stt': ER_stt,
                          'ER_total': ER_total})
    
    return er_df
    
    # df_dtt = pd.DataFrame(df_dtt.drop(['model', 'group'], axis = 1).groupby(['ag_idx', 'ID', 'choices'], as_index = False).mean())
    # df_dtt[]
    
    # # new_df = 
    
    # num_agents = len(df['ag_idx'].unique())
    # errorrates = np.zeros((3, num_agents)) # STT, DTT, Total
    # errorrates_day1 = np.zeros((3, num_agents)) # STT, DTT, Total
    # errorrates_day2 = np.zeros((3, num_agents)) # STT, DTT, Total
    # dfgh
    # for ag_idx in np.sort(df['ag_idx'].unique()):
    #     "----- Both days"
    #     ag_df = df[df['ag_idx'] == ag_idx]
    #     "Remove new block trials"
    #     ag_df = ag_df[ag_df['choices'] != -1]
    #     "Total error rates"
    #     errorrates[-1, ag_idx] = (ag_df['choices'] == -2).sum() / len(ag_df)
    #     ag_df_stt = ag_df[ag_df['trialsequence'] < 10]
    #     "Error Rates STT"
    #     errorrates[0, ag_idx] = (ag_df_stt['choices'] == -2).sum() / len(ag_df_stt)
    #     ag_df_dtt = ag_df[ag_df['trialsequence'] > 10]
    #     "Error Rates DTT"
    #     errorrates[1, ag_idx] = (ag_df_dtt['choices'] == -2).sum() / len(ag_df_dtt)
        
    #     "----- Day 1"
    #     ag_df_day1 = df[(df['ag_idx'] == ag_idx) & (df['blockidx'] <= 5)]
    #     "Remove new block trials"
    #     ag_df_day1 = ag_df_day1[ag_df_day1['choices'] != -1]
    #     "Total error rates"
    #     errorrates_day1[-1, ag_idx] = (ag_df_day1['choices'] == -2).sum() / len(ag_df_day1)
    #     ag_df_day1_stt = ag_df_day1[ag_df_day1['trialsequence'] < 10]
    #     "Error Rates STT"
    #     errorrates_day1[0, ag_idx] = (ag_df_day1_stt['choices'] == -2).sum() / len(ag_df_day1_stt)
    #     ag_df_day1_dtt = ag_df_day1[ag_df_day1['trialsequence'] > 10]
    #     "Error Rates DTT"
    #     errorrates_day1[1, ag_idx] = (ag_df_day1_dtt['choices'] == -2).sum() / len(ag_df_day1_dtt)
    
    #     "----- Day 2"
    #     ag_df_day2 = df[(df['ag_idx'] == ag_idx) & (df['blockidx'] > 5)]
    #     "Remove new block trials"
    #     ag_df_day2 = ag_df_day2[ag_df_day2['choices'] != -1]
    #     "Total error rates"
    #     errorrates_day2[-1, ag_idx] = (ag_df_day2['choices'] == -2).sum() / len(ag_df_day2)
    #     ag_df_day2_stt = ag_df_day2[ag_df_day2['trialsequence'] < 10]
    #     "Error Rates STT"
    #     errorrates_day2[0, ag_idx] = (ag_df_day2_stt['choices'] == -2).sum() / len(ag_df_day2_stt)
    #     ag_df_day2_dtt = ag_df_day2[ag_df_day2['trialsequence'] > 10]
    #     "Error Rates DTT"
    #     errorrates_day2[1, ag_idx] = (ag_df_day2_dtt['choices'] == -2).sum() / len(ag_df_day2_dtt)

    # return errorrates, errorrates_day1, errorrates_day2
    
def daydiff(df, sign_level = 0.05):
    '''
    

    Parameters
    ----------
    df : DataFrame
        If contains one value per participant -> plot difference.
        If contains several samples per parameter per participant -> Check if variables
        are statistically different from day 1 to day 2 within participant, and THEN
        plot difference for those where the difference was statistically significant
        according to a tow-sample t-test.

    Returns
    -------
    None.

    '''
    from scipy import stats
    
    df_temp = df.drop(['ag_idx', 'model', 'group'], axis = 1)
    parameter_names = df_temp.columns
    from_posterior = 0
    
    if len(df[df['ag_idx']==df['ag_idx'].unique()[0]][parameter_names[0]]) > 1:
        'df contains posterior dostros for each agents.'
        from_posterior = 1
        
        diff_dict = {}
        for param in parameter_names:
            # ipdb.set_trace()
            if 'day1' in param:
                diff_dict[param] = []
                diff_dict[param[0:-4]+'day2'] = []
                diff_dict['ag_idx'] = []
                for ag_idx in df['ag_idx'].unique():
                    df_ag = df[df['ag_idx'] == ag_idx]
                    t_statistic, p_value = stats.ttest_ind(df_ag[param], df_ag[param[0:-4] + 'day2'])
                    
                    if p_value < sign_level:
                        diff_dict['ag_idx'].append(ag_idx)
                        diff_dict[param].append(df_ag[param].mean())
                        diff_dict[param[0:-4]+'day2'].append(df_ag[param[0:-4] + 'day2'].mean())
                        
                    else:
                        print("Excluding agent %d for parameter %s (p-value %.4f)"%(ag_idx, param[0:-5], p_value))
        
    num_pars = 0
    for par in parameter_names:
        if 'day1' in par:
            num_pars += 1
    
    fig, ax = plt.subplots(int(np.ceil(num_pars/3)), 3, figsize=(15,5))
    
    num_plot_cols = 3
    num_plot_rows = int((num_pars <= num_plot_cols) * 1 + \
                    (num_pars > num_plot_cols) * np.ceil(num_pars / num_plot_cols))
    gs = fig.add_gridspec(num_plot_rows, num_plot_cols, hspace=0.2, wspace = 0.5)
    param_idx = 0
    for par in parameter_names:
        if 'day1' in par:
            if from_posterior:
                del df
                df = pd.DataFrame({par : diff_dict[par], 
                                   par[0:-4]+'day2' : diff_dict[par[0:-4]+'day2'],
                                   'ag_idx' : range(len(diff_dict[par]))})
                
            param_idx += 1
            plot_col_idx = param_idx % num_plot_cols
            plot_row_idx = (param_idx // num_plot_cols)
            
            df_plot = pd.melt(df, id_vars='ag_idx', value_vars=[par, par[0:-4]+'day2'])
            t_statistic, p_value = scipy.stats.ttest_rel(df[par], df[par[0:-4]+'day2'])
            if t_statistic > 0:
                print("%s(day1) > %s(day2) at p=%.5f"%(par[0:-5], par[0:-5], p_value))
                
            else:
                print("%s(day1) < %s(day2) at p=%.5f"%(par[0:-5], par[0:-5], p_value))
            
            for name, group in df_plot.groupby('ag_idx'):
                x = np.arange(len(group))
                y = group['value']
                slope = np.polyfit(x, y, 1)[0]  # Calculate the slope
                color = 'g' if slope >= 0 else 'r'  # Choose color based on slope
                
                if num_plot_rows > 1:
                    group.plot('variable', 
                               'value', 
                               kind = 'line', 
                               ax = ax[plot_row_idx, plot_col_idx], 
                               color = color, 
                               legend = False)
                    
                    df_plot.plot('variable', 
                                 'value', 
                                 kind='scatter', 
                                 ax=ax[plot_row_idx, plot_col_idx], 
                                 color='black', 
                                 legend=False)
                    
                else:
                    group.plot('variable', 
                               'value', 
                               kind = 'line', 
                               ax = ax[plot_col_idx], 
                               color = color, 
                               legend = False)
                    
                    df_plot.plot('variable', 
                                 'value', 
                                 kind='scatter', 
                                 ax = ax[plot_col_idx], 
                                 color='black', 
                                 legend=False)
                    
            # for line in plt.gca().get_lines():
            #     line.set_linewidth(0.3)
            
            # plt.gca().legend([],[], frameon=False)  # Hide the legend
            
    plt.show()