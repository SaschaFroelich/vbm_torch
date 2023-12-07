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
        
    # df = df.drop(['model', 'ag_idx', 'group'], axis = 1)
    
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
                  [0, 3], # conflict param
                  [0, 0.04], # lr
                  [0, 8], # theta_Q
                  [0.5, 5], # theta_rep
                  [0, 3]] # conflict param
        
    elif model == 'Seqparam' or model == 'Seqboost':
        ylims = [[0, 0.04], # lr
                  [0., 8], # theta_Q
                  [0., 2], # theta_rep
                  [-2, 2], # seqparam
                  [0, 0.04], # lr
                  [0., 8], # theta_Q
                  [0., 2], # theta_rep
                  [-2, 2]] # seqparam
        
    elif model == 'Bhand':
        if num_params == 7:
            ylims = [[0, 0.04], # lr
                      [0., 8], # theta_Q
                      [0., 2], # theta_rep
                      [0, 0.04], # lr
                      [0., 8], # theta_Q
                      [0., 2], # theta_rep
                      [-2.5, 2.5]] # hand_param
        
        elif num_params == 8:
            ylims = [[0, 0.04], # lr
                      [0., 8], # theta_Q
                      [0., 2], # theta_rep
                      [-2.5, 2.5], # hand_param
                      [0, 0.04], # lr
                      [0., 8], # theta_Q
                      [0., 2], # theta_rep
                      [-2.5, 2.5]] # hand_param
         
    elif model == 'sociopsy':
        ylims = [[20, 65], # Age
                  [0., 0.2], # ER_stt
                  [0., 0.2], # ER_dtt
                  [290, 480]] # RT
            
    else:
        ylims == None
    
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
    
    if 'ID' in df.columns:
        df = df.drop(['ID'], axis = 1)    

    if 'handedness' in df.columns:
        df = df.drop(['handedness'], axis = 1)    
    
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
    
    
def within_subject_corr(df, param_names):
    '''

    Parameters
    ----------
    df : DataFrame
        Columns
            Parameters
            ag_idx
            
    param_names : list
        list of parameter names for pairwise correlation.
    
    Returns
    -------
    corr_df : DataFrame
        Contains column ID as well as one column for each parameter pair that has been correlated
        length num_agents

    '''
    
    num_params = len(param_names)
    
    df = df.loc[:, ['ID', 'ag_idx', *param_names]]
    
    # if 'ID' in df.columns:
    #     df = df.drop(['model', 'group', 'ID'], axis = 1)
        
    # else:
    #     df = df.drop(['model', 'group'], axis = 1)
        
    corr_dict = {}
    corr_dict['ID'] = []
    for param1_idx in range(num_params):
        for param2_idx in range(param1_idx+1, num_params):
            corr_dict[param_names[param1_idx]+'_vs_' + param_names[param2_idx]] = []
    
    for ag_idx in np.sort(df['ag_idx'].unique()):
        df_ag = df[df['ag_idx'] == ag_idx]
        corr_dict['ID'].append(df_ag['ID'].unique()[0])
        
        for param1_idx in range(num_params):
            for param2_idx in range(param1_idx+1, num_params):
                assert len(df_ag['ID'].unique()) == 1
                corr_dict[param_names[param1_idx] + '_vs_' + param_names[param2_idx]].append(\
                           df_ag.loc[:, param_names[param1_idx]].corr(df_ag.loc[:, param_names[param2_idx]]))


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

def compute_errors(df):
    '''
    

    Parameters
    ----------
    df : DataFrame
        Contains trial-level agent behaviour.

    Returns
    -------
    er_df : DataFrame
        DESCRIPTION.

    '''
    
    df = df[df['choices'] != -1]
    # df_dtt = df[df['trialsequence'] > 10]
    # df_stt = df[df['trialsequence'] < 10]
    
    error_rates = pd.DataFrame()
    
    # pd.DataFrame(df_dtt.groupby(['ID', 'model', 'ag_idx', 'group'], as_index=False))
    
    group = []
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
    
    er_df = pd.DataFrame({'group': group,
                          'ID': IDs,
                          'ER_dtt': ER_dtt,
                          'ER_stt': ER_stt,
                          'ER_total': ER_total})
    
    return er_df
    
def daydiff(df, hdi_prob = None, threshold = 0, BF = None):
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
    diffs_df : DataFrame

    '''
    
    """ 
        Step 1: If provided posterior samples, check for each participant 
        if the two days are significantly different from each other.
    """
    from scipy import stats
    import arviz as az
    
    if hdi_prob is not None and BF is not None:
        raise Exception("Only specify either BF or hdi_prob.")
    
    if 'ID' in df.columns:
        df_temp = df.drop(['ag_idx', 'model', 'group', 'ID', 'handedness'], axis = 1)
        
    else:
        "For compatibility with recov data"
        df_temp = df.drop(['ag_idx', 'model', 'group'], axis = 1)

    parameter_names = df_temp.columns
    del df_temp
    from_posterior = 0
    
    if len(df[df['ag_idx']==df['ag_idx'].unique()[0]][parameter_names[0]]) > 1:
        print("Evaluating posterior samples.")
        'df contains posterior distros for each agents.'
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
                    # t_statistic, p_value = stats.ttest_ind(df_ag[param], df_ag[param[0:-4] + 'day2'])
                    difference_distro = np.array(df_ag[param[0:-4] + 'day2']-df_ag[param])
                    
                    if hdi_prob is not None:
                    
                        lower, higher = az.hdi(difference_distro, 
                                               hdi_prob = hdi_prob)
                        
                        if lower < threshold and higher > threshold:
                            print(f"threshold is in {hdi_prob*100}% HDI of parameter {param[0:-5]} for agent {ag_idx} --> Excluding agent")
                            
                        else:
                            diff_dict['ag_idx'].append(ag_idx)
                            diff_dict[param].append(df_ag[param].mean())
                            diff_dict[param[0:-4]+'day2'].append(df_ag[param[0:-4] + 'day2'].mean())
                            
                    elif BF is not None:
                        
                        BayesF = (difference_distro > 0).sum() / (difference_distro <= 0).sum()
                        
                        if BayesF < BF and BayesF > 1/BF:
                            print(f"Bayes Factor for day differences for parameter {param[0:-5]} is %.4f for {ag_idx} --> Excluding agent"%BayesF)
                        
                        elif BayesF >= BF or BayesF <= 1/BF:
                            diff_dict['ag_idx'].append(ag_idx)
                            diff_dict[param].append(df_ag[param].mean())
                            diff_dict[param[0:-4]+'day2'].append(df_ag[param[0:-4] + 'day2'].mean())
                    
                        else:
                            raise Exception("Iznogood.")
                    
                    # if p_value < sign_level:
                    #     diff_dict['ag_idx'].append(ag_idx)
                    #     diff_dict[param].append(df_ag[param].mean())
                    #     diff_dict[param[0:-4]+'day2'].append(df_ag[param[0:-4] + 'day2'].mean())
                        
                    # else:
                    #     print("Excluding agent %d for parameter %s (p-value %.4f)"%(ag_idx, param[0:-5], p_value))
        
    else:
        print("No posterior samples provided.")
        
    diffs_df = pd.DataFrame() # Will contain the differences for each participant
        
    num_pars = 0
    for par in parameter_names:
        if 'day1' in par:
            diffs_df[par[0:-1] + '2-' + par] = df[par[0:-1] + '2']-df[par]
            num_pars += 1
    
    if num_pars > 0:
        diffs_df['ag_idx'] = df['ag_idx']
        if 'ID' in df.columns:
            diffs_df['ID'] = df['ID']
        
        """
            Plot Day 2 - Day 1
        """
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
                    print("Check ag_idx!!!!")
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
                        ax_idxs = [plot_row_idx, plot_col_idx]
                        
                    else:
                        ax_idxs = [plot_col_idx]
                        
                    group.plot('variable', 
                               'value', 
                               kind = 'line', 
                               ax = ax[*ax_idxs], 
                               color = color, 
                               legend = False)
                    
                    df_plot.plot('variable', 
                                 'value', 
                                 kind='scatter', 
                                 ax=ax[*ax_idxs], 
                                 color='black', 
                                 legend=False)
                        
        plt.show()
        
        return diffs_df
    
    
def perform_PCA(df, num_components):
    '''
        Normalize df columns
    '''
    print("Make sure ag_idx in df is in ascending order.")
    num_agents = len(df)
    
    for col in df.columns:
        if col != 'ag_idx':
            df[col] = (df[col]-df[col].mean())/df[col].std()

    from sklearn.decomposition import PCA
    import itertools
    pca = PCA(n_components = num_components)
    
    df_for_pca = df.drop(['ag_idx'], axis = 1)
    principalComponents = pca.fit_transform(df_for_pca)
    
    for comp in range(num_components):
        
        pca_df = pd.DataFrame(data={'ag_idx': range(num_agents), 'PCA value': principalComponents[:,comp]})
        pca_0_df = pca_df[pca_df['PCA value'] < 0]
        pca_1_df = pca_df[pca_df['PCA value'] >= 0]
        
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
            
        print(f"Explained variance by component {comp+1}=%.4f"%pca.explained_variance_ratio_[comp])

