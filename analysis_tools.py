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
           param_names,
           model,
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
    
    # model = df['model'].unique()[0]
    
    # if 'ID' in df.columns:
    #     df = df.drop(['ID'], axis = 1)

    # if 'handedness' in df.columns:
    #     df = df.drop(['handedness'], axis = 1)
        
    # df = df.drop(['model', 'ag_idx', 'group', 'ID', 'handedness'], axis = 1)
    
    num_params = len(param_names)
    print(num_params)
    
    fig, ax = plt.subplots(len(df['day'].unique()), num_params, figsize=(15,5), sharey=0)
    
    # if model == 'B':
    #     ylims = [[0, 0.04], # lr
    #               [0.5, 7.5], # theta_Q
    #               [0., 2.], # theta_rep
    #               [0, 0.04], # lr
    #               [0.5, 7.5], # theta_Q
    #               [0., 2]] # theta_rep
        
    # elif model == 'Conflict':
    #     ylims = [[0, 0.04], # lr
    #               [0, 8], # theta_Q
    #               [0.5, 5], # theta_rep
    #               [0, 3], # conflict param
    #               [0, 0.04], # lr
    #               [0, 8], # theta_Q
    #               [0.5, 5], # theta_rep
    #               [0, 3]] # conflict param
        
    # elif model == 'Seqparam' or model == 'Seqboost':
    #     ylims = [[0, 0.04], # lr
    #               [0., 8], # theta_Q
    #               [0., 2], # theta_rep
    #               [-2, 2], # seqparam
    #               [0, 0.04], # lr
    #               [0., 8], # theta_Q
    #               [0., 2], # theta_rep
    #               [-2, 2]] # seqparam
        
    # elif model == 'Bhand':
    #     if num_params == 7:
    #         ylims = [[0, 0.04], # lr
    #                   [0., 8], # theta_Q
    #                   [0., 2], # theta_rep
    #                   [0, 0.04], # lr
    #                   [0., 8], # theta_Q
    #                   [0., 2], # theta_rep
    #                   [-2.5, 2.5]] # hand_param
        
    #     elif num_params == 8:
    #         ylims = [[0, 0.04], # lr
    #                   [0., 8], # theta_Q
    #                   [0., 2], # theta_rep
    #                   [-2.5, 2.5], # hand_param
    #                   [0, 0.04], # lr
    #                   [0., 8], # theta_Q
    #                   [0., 2], # theta_rep
    #                   [-2.5, 2.5]] # hand_param
         
    # elif model == 'sociopsy':
    #     ylims = [[20, 65], # Age
    #               [0., 0.2], # ER_stt
    #               [0., 0.2], # ER_dtt
    #               [290, 480], # RT
    #               [2750, 3800]] # points
            
    # else:
    #     ylims == None
    #

    for par in range(num_params):
        "ax[0]"
        dataseries = (df.melt()[df.melt()['variable'] == param_names[par]])
        dataseries['value'] = pd.to_numeric(dataseries['value'], errors='coerce')
        
        sns.violinplot(ax = ax[par], 
                       x = 'variable',
                       y = 'value',
                       data = dataseries,
                       color=".8")
        
        sns.stripplot(x = 'variable',
                      y = 'value',
                      data = dataseries,
                      edgecolor = 'auto',
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
        
        # if ylims is not None:
        #     ax[par].set_ylim(ylims[par])
    
        "Colorbar"
        # variance = df[params_df.columns[par]].std()**2
        
        # normalize = mcolors.TwoSlopeNorm(vcenter=(min(variance)+max(variance))/2, 
        #                                  vmin=min(variance), 
        #                                  vmax=max(variance))
        
        # colormap = cm.coolwarm
        # scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        # scalarmappaple.set_array(variance)
        # plt.colorbar(scalarmappaple, ax = ax[par])
            
    plt.show()

def param_corr(df, method = 'pearson'):
    '''
        Correlation Plots
    '''
    
    df = df.drop(['ag_idx', 'model', 'group'], axis = 1)
    
    if 'ID' in df.columns:
        df = df.drop(['ID'], axis = 1)    

    if 'handedness' in df.columns:
        df = df.drop(['handedness'], axis = 1)    
    
    # for col in range(len(df.columns)):
    #     df.rename(columns={df.columns[col] : df.columns[col][4:]}, inplace = True)
    
    def corrdot(*args, **kwargs):
        corr_r = args[0].corr(args[1], method)
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
    
    
def within_subject_corr(df, param_names1, param_names2, day=None):
    '''

    param_names1[i] is correlated with param_names2[i]

    assert len(param_names1) == len(param_names2)

    Parameters
    ----------
    df : DataFrame
        Columns
            Parameters
            ag_idx
            
    param_names1 : list
        list of parameter names for correlation.
        
    param_names2 : list
        list of parameter names for correlation.
    
    Returns
    -------
    corr_df : DataFrame
        Contains column ID as well as one column for each parameter pair that has been correlated
        length num_agents

    '''
    
    assert len(param_names1) == len(param_names2)
    
    num_params = len(param_names1)
    
    corr_dict = {}
    corr_dict['ag_idx'] = []
    
    'Create dict keys'
    for i in range(num_params):
        corr_dict[param_names1[i] + '_vs_' + param_names2[i]] = []
        
    'Compute corrs'
    for agidx in df['ag_idx'].unique():
        corr_dict['ag_idx'].append(agidx)
        
        for i in range(num_params):
            mask = (df['ag_idx'] == agidx) & (df['day'] == day)
            r_value = df[mask].loc[:, param_names1[i]].corr(df[mask].loc[:, param_names2[i]])
            corr_dict[param_names1[i] + '_vs_' + param_names2[i]].append(r_value)
        
    corr_df = pd.DataFrame(data = corr_dict)

    return corr_df

def cluster_analysis(df, title= ''):
    '''
    
    Parameters
    ----------
    df : DataFrame
        Column ID plus
        One column per vector space dimension. length num_agents.
        
    title : str, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    list
        list of leavnodes of the cluster analysis

    '''
    
    
    import ipdb
    import scipy.cluster.hierarchy as spc
    
    keys = df.drop(['ID'], axis=1).columns
    num_agents = len(df)
    num_corrs = len(df.columns) - 1
    
    "Distance matrix for the vector space"
    distance = np.zeros((num_agents, num_agents))
    
    for ag_idx1 in range(num_agents):
        for ag_idx2 in range(num_agents):
            v1 = np.zeros(num_corrs)
            v2 = np.zeros(num_corrs)
            
            for key_idx in range(num_corrs):
                key = keys[key_idx]
                v1[key_idx] = df[key][ag_idx1]
                v2[key_idx] = df[key][ag_idx2]
    
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
        
        
def kmeans(data, 
           inf_mean_df, 
           n_clusters, 
           num_reps = 1, 
           plotfig = True,
           title = None):
    '''

    Parameters
    ----------
    data : DataFrame
        Data to perform kmeans clustering over.
        One column per vector space dimension. num_agent rows.
        Must not contain any other columns.
        
    inf_mean_df : DataFrame
        For ag_idx and groups
        
    n_clusters : TYPE
        DESCRIPTION.
        
    num_reps : int, optional
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
    
    print("Correct this later")    
    if 'ID' in data.columns:
        data = data.drop(['ID'], axis=1)
    
    if isinstance(data, pd.DataFrame):
        assert 'ID' not in data.columns
        assert 'ag_idx' not in data.columns
        assert 'model' not in data.columns
        
    num_agents = len(inf_mean_df['ag_idx'].unique())
    
    from sklearn.cluster import KMeans
    n_clusters = n_clusters
    kmean = KMeans(n_clusters=n_clusters, n_init = 10, init ='random')
    
    all_cluster_groups = []
    all_labels = []
    mini_clusters = np.zeros((num_agents, num_agents))
    c_distances = []
    
    for rep in range(num_reps):
        kmeans = kmean.fit(data.to_numpy())
        
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
        if title is not None:
            plt.title(f"For {n_clusters} clusters ({title})")
            
        else:
            plt.title(f"For {n_clusters} clusters.")
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
        
    print("Make sure agent idxs and groups are assigned correctly via inf_mean_df!")
    
    return kmeans, cluster_groups, c_distances

def compute_errors(df, identifier = 'ID'):
    '''

    Parameters
    ----------
    df : DataFrame
        Contains trial-level agent behaviour.
            jokertypes : list
                DTT Types
                -1/0/1/2 : no joker/random/congruent/incongruent

    ERRORS (in 'correct'): 0 = wrong response, 1 = correct response, 2 = too slow, 3 = two keys at once during joker-trials

    identifier : str
        Name of df column which uniquely identifies an agent.

    Returns
    -------
    er_df : DataFrame
        DESCRIPTION.
        
    '''
    
    df = df[df['choices'] != -1]
    
    group = []
    IDs = []
    
    "DTT"
    ER_dtt = []
    
    ER_randomdtt = []
    ER_congruent = []
    ER_incongruent = []

    ER_notimeouterrors_dtt = []
    ER_timeouts_dtt = []
    
    "--- Timeouts"
    TO_dtt = []
    TO_incong = []
    TO_cong = []
    TO_randomdtt = []
    
    "STT"
    ER_stt = []
    ER_stt_seq = []
    ER_stt_seq_std = []
    ER_stt_rand = []    
    ER_stt_rand_std = [] 
    ER_stt = []
    
    ER_notimeouterrors_stt = []
    ER_timeouts_stt = []

    ER_total = []
    for ID in df[identifier].unique():
        "DTT"
        mask = (df['trialsequence'] > 10) & (df[identifier] == ID)
        ER_dtt.append(len(df[mask & (df['choices'] == -2)]) / len(df[mask]))
        
        if 'correct' in df.columns:
            ER_notimeouterrors_dtt.append(len(df[mask & (df['correct'] != 1) & (df['correct'] != 2)]) / len(df[mask]))
            ER_timeouts_dtt.append(len(df[mask & (df['correct'] == 2)]) / len(df[mask]))
        
        "Random"
        mask = (df['trialsequence'] > 10) & (df[identifier] == ID) & (df['jokertypes'] == 0)
        ER_randomdtt.append(len(df[mask & (df['choices'] == -2)]) / len(df[mask]))
        
        "Congruent"
        mask = (df['trialsequence'] > 10) & (df[identifier] == ID) & (df['jokertypes'] == 1)
        ER_congruent.append(len(df[mask & (df['choices'] == -2)]) / len(df[mask]))

        "Inongruent"
        mask = (df['trialsequence'] > 10) & (df[identifier] == ID) & (df['jokertypes'] == 2)
        ER_incongruent.append(len(df[mask & (df['choices'] == -2)]) / len(df[mask]))

        "--- Timeouts"
        "STT"
        mask =  (df['trialsequence'] < 10) & \
                (df[identifier] == ID) & \
                (df['blocktype'] == 0)
        ER_stt_seq.append(len(df[mask & (df['choices'] == -2)]) / len(df[mask]))
                
        mask =  (df['trialsequence'] < 10) & \
                (df[identifier] == ID) & \
                (df['blocktype'] == 1)
        ER_stt_rand.append(len(df[mask & (df['choices'] == -2)]) / len(df[mask]))
        
        mask =  (df['trialsequence'] < 10) & (df[identifier] == ID)
        ER_stt.append(len(df[mask & (df['choices'] == -2)]) / len(df[mask]))

        mask = (df['trialsequence'] > -1) & (df[identifier] == ID)
        ER_total.append(len(df[mask & (df['choices'] == -2)]) / len(df[mask]))
        
        IDs.append(ID)
        group.append(df[df[identifier] == ID]['group'].unique()[0])
        
    er_df = pd.DataFrame({'group': group,
                          identifier: IDs,
                          'ER_dtt': ER_dtt,
                          'ER_randomdtt': ER_randomdtt,
                          'ER_congruent': ER_congruent,
                          'ER_incongruent': ER_incongruent,
                          'ER_stt': ER_stt,
                          'ER_stt_seq': ER_stt_seq,
                          'ER_stt_rand': ER_stt_rand,
                          'ER_total': ER_total})
    
    er_df['ER_diff_stt'] = er_df['ER_stt_rand'] - er_df['ER_stt_seq']
    return er_df
    
def daydiffBF(df, parameter_names, hdi_prob = None, threshold = 0, BF = None):
    '''

    Parameters
    ----------
    df : DataFrame
        If contains one value per participant -> plot difference.
        If contains several samples per parameter per participant -> Check if variables
        are statistically different from day 1 to day 2 within participant, and THEN
        plot difference for those where the difference was statistically significant
        according to a tow-sample t-test.

    BF : Bayes-Factor threshold for differences between days.

    Returns
    -------
    diffs_df : DataFrame containing BF that parameter is larger on day 2.
    
    clean_means_df : DF containing only parameters for participants where parameter is
                    different from day 1 (according to Bayes Factor).

    '''
    
    """ 
        Step 1: If provided posterior samples, check for each participant 
        if the two days are significantly different from each other.
    """
    
    raise Exception("BF cannot be computed from simply subtracting samples from one another.")
    
    from scipy import stats
    import arviz as az
    
    if hdi_prob is not None and BF is not None:
        raise Exception("Only specify either BF or hdi_prob.")
        
    df = df.copy()
    
    if len(df[df['ag_idx']==df['ag_idx'].unique()[0]][parameter_names[0]]) > 1:
        print("Evaluating posterior samples.")
        'df contains posterior distros for each agents.'
        
        diff_dict = {}
        # diff_dict[param] = []
        diff_dict['ag_idx'] = []
        # diff_dict['day'] = []
        diff_dict['BF'] = []
        diff_dict['parameter'] = []
        
        for param in parameter_names:

            for ag_idx in df['ag_idx'].unique():
                df_ag = df[df['ag_idx'] == ag_idx]
                # t_statistic, p_value = stats.ttest_ind(df_ag[param], df_ag[param[0:-4] + 'day2'])
                difference_distro = np.array(df_ag[df_ag['day'] == 2][param]) - np.array(df_ag[df_ag['day'] == 1][param])
                
                if hdi_prob is not None:
                    raise Exception("Not implemented.")
                    # lower, higher = az.hdi(difference_distro, 
                    #                        hdi_prob = hdi_prob)
                    
                    # if lower < threshold and higher > threshold:
                    #     print(f"threshold is in {hdi_prob*100}% HDI of parameter {param[0:-5]} for agent {ag_idx} --> Excluding agent")
                        
                    # else:
                    #     diff_dict['ag_idx'].append(ag_idx)
                    #     diff_dict[param].append(df_ag[param].mean())
                    #     diff_dict[param[0:-4]+'day2'].append(df_ag[param[0:-4] + 'day2'].mean())
                        
                elif BF is not None:
                    
                    BayesF = (difference_distro > 0).sum() / (difference_distro <= 0).sum()
                    
                    diff_dict['ag_idx'].append(ag_idx)
                    diff_dict['parameter'].append(param)
                    diff_dict['BF'].append(BayesF)
                    
                    if BayesF < BF and BayesF > 1/BF:
                        print(f"Bayes Factor for day differences for parameter {param} is %.4f for {ag_idx} --> Exclude agent"%BayesF)
                    
                    # elif BayesF >= BF or BayesF <= 1/BF:
                    #     diff_dict['ag_idx'].append(ag_idx)
                    #     diff_dict[param].append(df_ag[df_ag['day'] == 1][param].mean())
                    #     diff_dict['day'].append(1)
                    #     diff_dict[param].append(df_ag[df_ag['day'] == 2][param].mean())
                    #     diff_dict['day'].append(2)
                
                    # else:
                    #     raise Exception("Iznogood.")
        
    else:
        print("No posterior samples provided.")
        
    diffs_df = pd.DataFrame(diff_dict)
    clean_diff_df = diffs_df[(diffs_df['BF'] > BF) | (diffs_df['BF'] < 1/BF)]
    clean_diff_df = clean_diff_df.reset_index(drop=False)
    
    pname = []
    mean = []
    day = []
    ag_idx = []
    # IDs = []
    
    for rowidx in range(len(clean_diff_df)):
        paramname = clean_diff_df.loc[rowidx, 'parameter']
        agidx = clean_diff_df.loc[rowidx, 'ag_idx']
        # IDs = clean_diff_df.loc[rowidx, 'ID']
        
        mean.append(df[(df['ag_idx'] == agidx) & (df['day'] == 1)][paramname].mean())
        pname.append(paramname)
        day.append(1)
        ag_idx.append(agidx)
        
        mean.append(df[(df['ag_idx'] == agidx) & (df['day'] == 2)][paramname].mean())
        pname.append(paramname)
        day.append(2)
        ag_idx.append(agidx)
        
    clean_means_df = pd.DataFrame({'ag_idx': ag_idx,
                                   # 'ID': IDs,
                                    'parameter': pname, 
                                   'mean': mean, 
                                   'day': day})
    
    
    return diffs_df, clean_means_df
    
def perform_PCA(df, num_components, plot = False, correctfor = None):
    '''
    

    Parameters
    ----------
    df : DataFrame
        Only parameter columns and ag_idx
        
    num_components : int
        number of pca components
        
    plot : TYPE, optional
        DESCRIPTION. The default is False.
        
    correctfor : str, optional
        Correct for this confounding variable with linear regression. The default is None.

    Returns
    -------
    principalComponents : array, shape [num_datapoints, num_components]
        projection of datapoints onto principal components

    '''
    
    '''
        Normalize df columns
    '''
    print("Make sure ag_idx in df is in ascending order.")
    df = df.copy()
    num_agents = len(df)
    
    if correctfor is not None:
        from sklearn.linear_model import LinearRegression
        df_corrected = pd.DataFrame(columns = df.columns)
        
        if 'ag_idx' in df.columns:
            df_corrected['ag_idx'] = df['ag_idx']
        
        for col in df.columns:
            if col != 'ag_idx' and col != correctfor:
                y = np.array(df[col]).reshape(-1,1)
                x = np.array(df[correctfor]).reshape(-1,1)
                linmodel = LinearRegression()
                linmodel.fit(x, y)
                predictions = linmodel.predict(x)
                residuals = y - predictions
                
                df_corrected[col] = np.squeeze(residuals)
                
            
        df = df_corrected
        df = df.drop((correctfor), axis=1)

    print("Normalizing...")
    for col in df.columns:
        if col != 'ag_idx':
            df[col] = (df[col]-df[col].mean())/df[col].std()

    from sklearn.decomposition import PCA
    import itertools
    pca = PCA(n_components = num_components)
    
    print("PCA\n\n")
    if 'ag_idx' in df.columns:
        df_for_pca = df.drop(['ag_idx'], axis = 1)
        
    else:
        df_for_pca = df
        
    principalComponents = pca.fit_transform(df_for_pca)
    
    for comp in range(num_components):
        
        pca_df = pd.DataFrame(data={'ag_idx': range(num_agents), 'PCA value': principalComponents[:,comp]})
        pca_0_df = pca_df[pca_df['PCA value'] < 0]
        pca_1_df = pca_df[pca_df['PCA value'] >= 0]
        
        if plot:
            for col1_idx in range(len(df.columns)):
                for col2_idx in range(col1_idx+1, len(df.columns)):
                    col1 = df.columns[col1_idx]
                    col2 = df.columns[col2_idx]
            
                    fig, ax = plt.subplots()
                    
                    plot_df = df[df['ag_idx'].isin(pca_0_df['ag_idx'])]
                    ax.scatter(plot_df[col1], plot_df[col2], color='red')
                    
                    plot_df = df[df['ag_idx'].isin(pca_1_df['ag_idx'])]
                    ax.scatter(plot_df[col1], plot_df[col2], color='blue')
                    
                    ax.axhline(0, color='k')
                    ax.axvline(0, color='k')
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                    # plt.grid()
                    # plt.scatter(kmeans.cluster_centers_[0, 0], kmeans.cluster_centers_[0, 1], color='red')
                    # plt.scatter(kmeans.cluster_centers_[1, 0], kmeans.cluster_centers_[1, 1], color='red')
                    # plt.title('Delta R vs Delta Q')
                    plt.show()
                    # dfgh
            
        print(f"Explained variance by component {comp+1} is %.4f"%pca.explained_variance_ratio_[comp])
        print(f"Direction of component {comp+1} is {pca.components_[comp,:]}\n")
        
    print(f"The first {num_components} components (of possible {len(df_for_pca.columns)}) explain %.4f percent of the variance."%(pca.explained_variance_ratio_.sum()*100))
    
    return principalComponents


def network_corr(df, nodes, covars = None, method = 'spearman'):
    '''
    Computes all pairwise correlations of columns <nodes> in df.    
    
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
        
    nodes : list
        list entries are str 
        list entries must be columns of df
        
    covars : list
        list entries are str 
        list entries must be columns of df
        
    method : str, optional
        Pearson or Spearman correlation. The default is 'spearman'.

    Returns
    -------
    r_matrix : numpy array
        
    p_matrix : numpy array
    '''
    import pingouin
    num_measures = len(nodes)
    
    r_matrix = np.ones((num_measures, num_measures))
    p_matrix = np.ones((num_measures, num_measures))
    
    for idx in range(num_measures):
        for jdx in range(idx+1, num_measures):
            
            if method == 'spearman':
                r,p = scipy.stats.spearmanr(df[nodes[idx]], df[nodes[jdx]])
                
            elif method == 'pearson':
                r,p = scipy.stats.pearsonr(df[nodes[idx]], df[nodes[jdx]])
                
            else:
                raise Exception("Correlation method not provided.")
            
            r_matrix[idx, jdx] = r
            r_matrix[jdx, idx] = r

            p_matrix[idx, jdx] = p
            p_matrix[jdx, idx] = p
            
            '''
                As soon as for one pair of correlates, the partial correlation with respect
                to any third variable is not significant anymore, the new, insiginificant 
                p-value is stored in the correlation matrix.
            '''
            if covars is not None:
                if f'{nodes[idx]} & {nodes[jdx]}' in covars.keys():
                    for covaridx in range(len(covars[f'{nodes[idx]} & {nodes[jdx]}'])):
                        stats = pingouin.partial_corr(data = df, 
                                                      x=nodes[idx], 
                                                      y=nodes[jdx], 
                                                      covar = covars[f'{nodes[idx]} & {nodes[jdx]}'][covaridx], 
                                                      method = method)
                        rvalue = stats['r'].iloc[0]
                        pvalue = stats['p-val'].iloc[0]
                        if pvalue > 0.05:
                            print("=========================================")
                            print(f"Insignificant partial corr between {nodes[idx]}" + 
                                  f" & {nodes[jdx]} after correcting for "+ 
                                  f"{covars[f'{nodes[idx]} & {nodes[jdx]}'][covaridx]}," + 
                                  "r=%.3f, p=%.3f"%(rvalue, pvalue))
                            
                            "Replace correlation values with corrected values."
                            r_matrix[idx, jdx] = rvalue
                            r_matrix[jdx, idx] = rvalue
                
                            p_matrix[idx, jdx] = pvalue
                            p_matrix[jdx, idx] = pvalue
                            
                        else:
                            print("=========================================")
                            print(f"Significant partial corr between {nodes[idx]}" + 
                                  f" & {nodes[jdx]} after correcting for "+ 
                                  f"{covars[f'{nodes[idx]} & {nodes[jdx]}'][covaridx]}," + 
                                  "r=%.3f, p=%.3f"%(rvalue, pvalue))
                            
                
                    
    return r_matrix, p_matrix

def find_seqlearners(expdata_df, 
                     day, 
                     correctp = True,
                     by = 'STT'):
    '''
    
    Parameters
    ----------
    expdata_df : DataFrame with experimental data.
        DESCRIPTION.
        
    day : TYPE
        DESCRIPTION.
        
    correctp : bool, optional
        DESCRIPTION. The default is True.
        
    by : str
        STT or DTT.
        How to differentiate seq lerners from non-seq learners.
        STT : by STT Random - Fixed Condition
        DTT : Congruent - Incongruent DTT

    Returns
    -------
    seqlearners_df : DataFrame
        DESCRIPTION.

    notseqlearners_df : DataFrame
        DESCRIPTION.

    '''
    
    # df = expdata_df[expdata_df['choices'] != -1]
    if 'day' in expdata_df.columns:
        assert len(expdata_df['day'].unique()) == 1
        
    if by == 'STT':
        df = expdata_df[(expdata_df['trialsequence'] != -1) & (expdata_df['trialsequence'] < 10)]
        
    else:
        raise Exception("Not yet implemented.")
        
    num_agents = len(df['ID'].unique())
    
    from scipy.stats import chi2_contingency

    '''
        Chi-squared test for ER
    '''
    IDs = []
    chis = []
    ps = []
    r_minus_s_er = []

    for ID in df['ID'].unique():
        df_ag = df[df['ID'] == ID]
        
        seq_er = [len(df_ag[(df_ag['choices'] == -2) & (df_ag['blocktype'] == 0)]), 
                  len(df_ag[(df_ag['choices'] > -2) & (df_ag['blocktype'] == 0)])] # 0 error, 1 correct
        
        rand_er = [len(df_ag[(df_ag['choices'] == -2) & (df_ag['blocktype'] == 1)]), 
                  len(df_ag[(df_ag['choices'] > -2) & (df_ag['blocktype'] == 1)])] # 0 error, 1 correct
        
        chi2, p_value, dof, expected = chi2_contingency([seq_er, rand_er])
        
        IDs.append(ID)
        chis.append(chi2)
        ps.append(p_value)
        r_minus_s_er.append(rand_er[0] - seq_er[0]) # difference in number of errors
        
        print(f"Chi-square Statistic: {chi2}")
        print(f"P-value: {p_value}")
        print(f"{rand_er[0] - seq_er[0]}")
        
        
    chi_er_df = pd.DataFrame({'ID': IDs, 
                           'chi': chis,
                           'p_chi': ps,
                           'Difference_ER': r_minus_s_er})
        
        
    '''
        t-test for Reaction times
    '''
    ts = np.zeros(num_agents)
    ps_rt = np.ones(num_agents)
    IDs = []
    r_minus_s_rt = []
    
    ag_idx = 0
    for ID in df['ID'].unique():
        IDs.append(ID)
        df_ag = df[df['ID'] == ID]
        df_ag = df_ag[df_ag['choices'] != -2]
        # df_ag = df_ag[df_ag['RT'] < 600]
        
        t,p = scipy.stats.ttest_ind(np.array(df_ag[df_ag['blocktype']==0]['RT'], dtype='float'), # seq
                                    np.array(df_ag[df_ag['blocktype']==1]['RT'], 'float'), alternative = 'less') #random
        
        r_minus_s_rt.append(np.array(df_ag[df_ag['blocktype']==1]['RT'], 'float').mean()-np.array(df_ag[df_ag['blocktype']==0]['RT'], dtype='float').mean())
        ts[ag_idx] = t
        ps_rt[ag_idx] = p
        ag_idx += 1
    
    t_rt_df = pd.DataFrame({'ID' : IDs,
                            't' : ts,
                            'p_t': ps_rt,
                            'Difference_RT': r_minus_s_rt})
    
    # scipy.stats.pearsonr(ps_er, ps_rt)
    
    new_df = pd.merge(chi_er_df, t_rt_df, on = 'ID')
    
    print(new_df['p_chi'].corr(new_df['p_t']))
    
    chisquared = -2 * (np.log(new_df['p_chi']) + np.log(new_df['p_t']))
    newp = 1-scipy.stats.chi2.cdf(chisquared, 4)
    
    if correctp:
        ps_adjusted = scipy.stats.false_discovery_control(newp, method='bh')
        
    else:
        ps_adjusted = newp
    
    
    
    new_df['p_compound_adjusted'] = ps_adjusted
    new_df['chi_compound'] = chisquared
    
    seqlearners_df = new_df[(new_df['p_compound_adjusted'] < 0.05) & 
                            (new_df['Difference_ER'] > 0) & 
                            (new_df['Difference_RT'] > 0)]
    
    print(f"{len(seqlearners_df)} strong sequence learners with RT & ER > 0.")
    
    seqlearners_df = new_df[(new_df['p_compound_adjusted'] < 0.05) & 
                            (new_df['Difference_RT'] > 0)]
    
    print(f"{len(seqlearners_df)} strong sequence learners with RT > 0.")
    
    seqlearners_df = new_df[new_df['p_compound_adjusted'] < 0.05]
    
    print(f"{len(seqlearners_df)} strong sequence learners.")
    
    if 'day' in seqlearners_df.columns:
        seqlearners_df = seqlearners_df[seqlearners_df['day'] == day]
        
    else:
        seqlearners_df['day'] = day
    
    notseqlearners_df = new_df[new_df['p_compound_adjusted'] > 0.05]
    
    if 'day' in notseqlearners_df.columns:
        notseqlearners_df = notseqlearners_df[notseqlearners_df['day'] == day]
        
    else:
        notseqlearners_df['day'] = day
    
    return seqlearners_df, notseqlearners_df, new_df

def hpcf_within(expdata_df, correctp = False):
    '''

    Parameters
    ----------
    expdata_df : TYPE
        DESCRIPTION.

    Returns
    -------
    IDs : TYPE
        DESCRIPTION.
    ps_cr : TYPE
        DESCRIPTION.
    ps_ri : TYPE
        DESCRIPTION.

    '''
    
    assert len(expdata_df['day'].unique()) == 1
    
    df = expdata_df[expdata_df['trialsequence'] != -1]
    df = df[df['trialsequence'] > 10]
    df = df[df['choices'] != -2]
    
    assert len(df['choices_GD'].unique()) == 2
    
    from scipy.stats import chi2_contingency

    '''
        Chi-squared test for HRC
    '''
    IDs = []
    
    chis_cr = []
    chis_ri = []
    chis_ci = []
    
    ps_cr = []
    ps_ri = []
    ps_ci = []
    
    cr = []
    ri = []
    ci = []

    for ID in df['ID'].unique():
        IDs.append(ID)
        df_ag = df[df['ID'] == ID]
        
        rand = [len(df_ag[(df_ag['choices_GD'] == 0) & (df_ag['jokertypes'] == 0)]), 
                len(df_ag[(df_ag['choices_GD'] == 1) & (df_ag['jokertypes'] == 0)])] # 0 not GD choice, 1 GD choice
        
        congruent = [len(df_ag[(df_ag['choices_GD'] == 0) & (df_ag['jokertypes'] == 1)]), 
                     len(df_ag[(df_ag['choices_GD'] == 1) & (df_ag['jokertypes'] == 1)])] # 0 not GD choice, 1 GD choice
        
        incongruent = [len(df_ag[(df_ag['choices_GD'] == 0) & (df_ag['jokertypes'] == 2)]), 
                       len(df_ag[(df_ag['choices_GD'] == 1) & (df_ag['jokertypes'] == 2)])] # 0 not GD choice, 1 GD choice
        
        chi2, p_value, dof, expected = chi2_contingency([congruent, rand])
        chis_cr.append(chi2)
        ps_cr.append(p_value)
        cr.append(congruent[1]/(congruent[1]+congruent[0]) - rand[1]/(rand[1]+rand[0])) # difference in proportion of GD responses
        
        chi2, p_value, dof, expected = chi2_contingency([rand, incongruent])
        chis_ri.append(chi2)
        ps_ri.append(p_value)
        ri.append(rand[1]/(rand[1]+rand[0]) - incongruent[1]/(incongruent[1]+incongruent[0])) # difference in proportion of GD responses
    
        chi2, p_value, dof, expected = chi2_contingency([congruent, incongruent])
        chis_ci.append(chi2)
        ps_ci.append(p_value)
        ci.append(congruent[1]/(congruent[1]+congruent[0]) - incongruent[1]/(incongruent[1]+incongruent[0])) # difference in proportion of GD responses
    
    if correctp:
        ps_cr = scipy.stats.false_discovery_control(ps_cr, method='bh')
        ps_ri = scipy.stats.false_discovery_control(ps_ri, method='bh')
        ps_ci = scipy.stats.false_discovery_control(ps_ci, method='bh')
    
    return IDs, ps_cr, ps_ri, ps_ci, chis_cr, chis_ri, chis_ci

def find_strategies(expdata_df, 
                    plot_single = True, 
                    correctp = True,):
    '''
    

    Parameters
    ----------
    expdata_df : TYPE
        DESCRIPTION.
        
    day : TYPE
        DESCRIPTION.
        
    num_sections : TYPE, optional
        DESCRIPTION. The default is 3.
        
    plot_single : TYPE, optional
        DESCRIPTION. The default is True.
        
    correctp : TYPE, optional
        DESCRIPTION. The default is True.
        
    by : str, optional
        Random or Fix.
        Random : Compare HRC of Congruent and Incongruent with HRC of Random.
        Fix : Compare HRC of Congruent with HRC of Incongruent.

    Returns
    -------
    habitual_df : TYPE
        DESCRIPTION.
        
    GD_df : TYPE
        DESCRIPTION.
        
    modulators_df : TYPE
        DESCRIPTION.
        
    antimods_df : TYPE
        DESCRIPTION.

    '''
    
    num_agents = len(expdata_df['ID'].unique())
    assert len(expdata_df['day'].unique()) == 1
    
    IDs, ps_cr, ps_ri, _, chis_cr, chis_ri, _ = hpcf_within(expdata_df)
        
    if correctp:
        ps_cr_adjusted = scipy.stats.false_discovery_control(ps_cr, method='bh')
        ps_ri_adjusted = scipy.stats.false_discovery_control(ps_ri, method='bh')
        
    else: 
        ps_cr_adjusted = np.array(ps_cr)
        ps_ri_adjusted = np.array(ps_ri)
        
    # ps_cr_adjusted = scipy.stats.false_discovery_control(ps_cr, method='bh')
    # ps_ri_adjusted = scipy.stats.false_discovery_control(ps_ri, method='bh')
    
    ri_small = np.where(ps_ri_adjusted > 0.05, 0, 1)
    ri_small_idxs = np.where(ri_small == 0, range(1, num_agents+1), -1)
    ri_small_idxs = np.array(ri_small_idxs[ri_small_idxs>0]) - 1
    
    ri_large = np.where(ps_ri_adjusted <= 0.05, 0, 1)
    ri_large_idxs = np.where(ri_large == 0, range(1, num_agents+1), -1)
    ri_large_idxs = np.array(ri_large_idxs[ri_large_idxs>0]) - 1 
    
    cr_small = np.where(ps_cr_adjusted > 0.05, 0, 1)
    cr_small_idxs = np.where(cr_small == 0, range(1, num_agents+1), -1)
    cr_small_idxs = np.array(cr_small_idxs[cr_small_idxs>0]) - 1
    
    cr_large = np.where(ps_cr_adjusted <= 0.05, 0, 1)
    cr_large_idxs = np.where(cr_large == 0, range(1, num_agents+1), -1)
    cr_large_idxs = np.array(cr_large_idxs[cr_large_idxs>0]) - 1
    
    IDs_ri_small = [IDs[r] for r in ri_small_idxs]
    IDs_ri_large = [IDs[r] for r in ri_large_idxs]
    
    IDs_cr_small = [IDs[r] for r in cr_small_idxs]
    IDs_cr_large = [IDs[r] for r in cr_large_idxs]
    
    "Habitual"
    # %matplotlib inline
    fig, ax = plt.subplots()
    ax.text(0,0, 'HABITUAL', fontsize=40)
    plt.show()
    habitual_df = utils.plot_grouplevel(expdata_df[(expdata_df['ID'].isin(IDs_cr_large)) & 
                                     (expdata_df['ID'].isin(IDs_ri_large))], plot_single = plot_single)
    
    fig, ax = plt.subplots()
    ax.text(0,0, 'GD_df', fontsize=40)
    plt.show()
    GD_df = utils.plot_grouplevel(expdata_df[(expdata_df['ID'].isin(IDs_cr_small)) & 
                                     (expdata_df['ID'].isin(IDs_ri_small))], plot_single = plot_single)
    
    fig, ax = plt.subplots()
    ax.text(0,0, 'modulators', fontsize=40)
    plt.show()
    modulators_df = utils.plot_grouplevel(expdata_df[(expdata_df['ID'].isin(IDs_cr_large)) & 
                                     (expdata_df['ID'].isin(IDs_ri_small))], plot_single = plot_single)
    
    fig, ax = plt.subplots()
    ax.text(0,0, 'anti', fontsize=40)
    plt.show()
    "Low CR & High CI"
    antimods_df = utils.plot_grouplevel(expdata_df[(expdata_df['ID'].isin(IDs_cr_small)) & 
                                     (expdata_df['ID'].isin(IDs_ri_large))], plot_single = plot_single)

    ps_df = pd.DataFrame({'ID': IDs,
                          'ps_ri': ps_ri_adjusted, 
                          'chis_ri': chis_ri,
                          'ps_cr': ps_cr_adjusted,
                          'chis_cr': chis_cr})

    return habitual_df, GD_df, modulators_df, antimods_df, ps_df

def compute_BF(series, thresh):
    '''
        Computes BF of whether a pd series is larger than thresh
    '''
    
    BF = (series.to_numpy() > thresh).sum() / (series.to_numpy() < thresh).sum()
    
    return BF
    