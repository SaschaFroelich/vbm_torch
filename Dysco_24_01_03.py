#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:33:19 2023

@author: sascha
"""

import matplotlib.pyplot as plt
import utils
import analysis_tools as anal
import seaborn as sns
import scipy
from scipy import stats
import pickle
import pandas as pd
import numpy as np

published = 0
if published == 1:
    expdata_df = pickle.load(open("behav_data/preproc_data_old_published_all.p", "rb" ))[1]
    
else:
    _, expdata_df = pickle.load(open("behav_data/preproc_data.p", "rb" ))

hpcf_df = utils.compute_hpcf(expdata_df)
error_df = anal.compute_errors(expdata_df)
RT_df = utils.compute_RT(expdata_df)
points_df = utils.compute_points(expdata_df)

complete_df = pd.merge(hpcf_df, error_df, on = 'ID')
complete_df = pd.merge(complete_df, RT_df, on = 'ID')
complete_df = pd.merge(complete_df, points_df.loc[:, ['ID', 'points_day2']], on = 'ID')
complete_df = pd.merge(complete_df, expdata_df.loc[:, ['group', 'ID', 'q_notice_a_sequence']].groupby(['ID'], as_index = False).mean(), on = 'ID')

#%%
for col in complete_df.columns:
    if col != 'ID' and col != 'gender' and col != 'handedness' and col != 'model':
        complete_df[col] = complete_df[col].astype(float)

complete_df['RIspread_day1'] = complete_df['hpcf_rand_day1'] - complete_df['hpcf_incong_day1']
complete_df['RIspread_day2'] = complete_df['hpcf_rand_day2'] - complete_df['hpcf_incong_day2']

complete_df['CRspread_day1'] = complete_df['hpcf_cong_day1'] - complete_df['hpcf_rand_day1']
complete_df['CRspread_day2'] = complete_df['hpcf_cong_day2'] - complete_df['hpcf_rand_day2']

complete_df['CIspread_day1'] = complete_df['hpcf_cong_day1'] - complete_df['hpcf_incong_day1']
complete_df['CIspread_day2'] = complete_df['hpcf_cong_day2'] - complete_df['hpcf_incong_day2']

complete_df['spread_day1'] = complete_df['hpcf_cong_day1'] - complete_df['hpcf_incong_day1']
complete_df['spread_day2'] = complete_df['hpcf_cong_day2'] - complete_df['hpcf_incong_day2']

if published:
    complete_df = complete_df[complete_df['RT_diff_stt_day2'] < 60]
    complete_df = complete_df[complete_df['ER_diff_stt_day2'] < 0.08]
    
else:
    sociopsy_df = utils.get_sociopsy_df()
    complete_df = pd.merge(complete_df, sociopsy_df, on='ID')
    
complete_df.rename(columns={'jokertypes': 'DTT Types'}, inplace = True)


#%%
'''
    Dysco-Vortrag 2024/01/03
    Generative modeling results: line 786
'''

if 'conflict_param_day2' not in complete_df.columns:
    '''
        Boxplot ΔRT
    '''
    RT_df = complete_df.loc[:, ['ID', 
                                'RT_stt_seq_day1', 
                                'RT_stt_rand_day1', 
                                'RT_stt_seq_day2', 
                                'RT_stt_rand_day2']]
    
    RT_df['RT Diff Day 1'] = (np.squeeze(RT_df.loc[:, ['RT_stt_rand_day1']]) - np.squeeze(RT_df.loc[:, ['RT_stt_seq_day1']])).astype(float)
    RT_df['RT Diff Day 2'] = (np.squeeze(RT_df.loc[:, ['RT_stt_rand_day2']]) - np.squeeze(RT_df.loc[:, ['RT_stt_seq_day2']])).astype(float)
    
    # RT_df = RT_df.rename(columns={'RT_seq_day1':'Fix (Day 1)', 'RT_seq_day2':'Fix (Day 2)',
    #                               'RT_rand_day1':'Rand (Day 1)', 'RT_rand_day2':'Rand (Day 2)'})
    
    RT_df_melted = RT_df.melt(id_vars = ['ID'], value_vars=  ['RT Diff Day 1', 
                                                              'RT Diff Day 2'])
    
    RT_df_melted['Day'] = RT_df_melted['variable'].map(lambda x: 'Day 1' if '1' in x else 'Day 2')
    
    r,p = scipy.stats.ttest_1samp(RT_df['RT Diff Day 1'], 0)
    print(r)
    print(p)
    
    r,p = scipy.stats.ttest_1samp(RT_df['RT Diff Day 2'], 0)
    print(r)
    print(p)
    
    fig, ax = plt.subplots()
    sns.boxplot(data=RT_df_melted, x='variable', y='value')
    ax.set_ylabel(r'$\Delta$RT (rand-fix) (ms)', fontsize=20)
    # ax.set_xlabel('Condition')
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/diffs_rt.png', dpi=300)   
    plt.show()
    
    '''
        Boxplot ΔER
    '''
    ER_df = complete_df.loc[:, ['ID', 
                                'ER_stt_seq_day1', 
                                'ER_stt_rand_day1', 
                                'ER_stt_seq_day2', 
                                'ER_stt_rand_day2']]
    
    ER_df['ER Diff Day 1'] = (np.squeeze(ER_df.loc[:, ['ER_stt_rand_day1']]) - np.squeeze(ER_df.loc[:, ['ER_stt_seq_day1']])).astype(float)
    ER_df['ER Diff Day 2'] = (np.squeeze(ER_df.loc[:, ['ER_stt_rand_day2']]) - np.squeeze(ER_df.loc[:, ['ER_stt_seq_day2']])).astype(float)
    
    ER_df_melted = ER_df.melt(id_vars = ['ID'], value_vars=  ['ER Diff Day 1', 
                                                              'ER Diff Day 2'])
    
    ER_df_melted['Day'] = ER_df_melted['variable'].map(lambda x: 'Day 1' if '1' in x else 'Day 2')
    
    r,p = scipy.stats.ttest_1samp(ER_df['ER Diff Day 1'], 0)
    print(r)
    print(p)
    
    r,p = scipy.stats.ttest_1samp(ER_df['ER Diff Day 2'], 0)
    print(r)
    print(p)
    
    fig, ax = plt.subplots()
    sns.boxplot(data = ER_df_melted, x='variable', y='value')
    ax.set_ylabel(r'$\Delta$ER (rand-fix) (ms)', fontsize=20)
    # ax.set_xlabel('Condition')
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/diffs_er.tiff', dpi=300)   
    plt.show()
    
    '''
        Fig 1
        x : ΔRT
        y : ΔER
        hue = CI-Spread
    '''
    fig, ax = plt.subplots()
    slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['ER_diff_stt_day2'])
    # sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', ax = ax)
    sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', ax = ax, hue = 'CIspread_day2')
    ax.text(22, 0.035, 'r=%.4f, p = %.4f'%(r,p))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax.set_xlabel(r'$\Delta$RT')
    ax.set_ylabel(r'$\Delta$ER')
    ax.axhline(0, color = 'k', linewidth = 0.5)
    ax.axvline(0, color = 'k', linewidth = 0.5)
    plt.show()
    
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['CIspread_day2'])
    sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'CIspread_day2', ax = ax[0])
    ax[0].text(22,-0.06, 'r=%.4f, p = %.4f'%(r,p))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax[0].axhline(0, color = 'k', linewidth = 0.5)
    ax[0].axvline(0, color = 'k', linewidth = 0.5)
    ax[0].set_ylabel('CI-Spread (pp)')
    ax[0].set_xlabel(r'$\Delta$RT (ms)')
    
    slope, intercept, r, p, std_err = stats.linregress(complete_df['ER_diff_stt_day2'], complete_df['CIspread_day2'])
    sns.regplot(data = complete_df, x = 'ER_diff_stt_day2', y = 'CIspread_day2', ax = ax[1])
    ax[1].text(0.023, -0.06, 'r=%.4f, p = %.4f'%(r,p))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax[1].axhline(0, color = 'k', linewidth = 0.5)
    ax[1].axvline(0, color = 'k', linewidth = 0.5)
    ax[1].set_xlabel(r'$\Delta$ER (pp)')
    ax[1].set_ylabel('CI-Spread (pp)')
    plt.show()
    
    '''
        Fig 2
        Subplot 1
        x : ΔRT
        y : RI-Spread
        
        Subplot 2
        x : ΔER
        y : RI-Spread
        
        Without Outlier Detection
    '''
    fig, ax = plt.subplots(1,2, figsize=(15,5), sharey=True)
    slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['RIspread_day2'])
    sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'RIspread_day2', ax = ax[0])
    ax[0].text(-5,0.17, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax[0].axhline(0, color = 'k', linewidth = 0.5)
    ax[0].axvline(0, color = 'k', linewidth = 0.5)
    ax[0].set_ylabel('RI-Spread (pp)', fontsize = 20)
    ax[0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    
    slope, intercept, r, p, std_err = stats.linregress(complete_df['ER_diff_stt_day2'], complete_df['RIspread_day2'])
    sns.regplot(data = complete_df, x = 'ER_diff_stt_day2', y = 'RIspread_day2', ax = ax[1])
    ax[1].text(-0.01,0.17, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax[1].axhline(0, color = 'k', linewidth = 0.5)
    ax[1].axvline(0, color = 'k', linewidth = 0.5)
    ax[1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[1].set_ylabel('RI-Spread (pp)', fontsize = 20)
    plt.show()
    
    '''
        x : RI-Spread
        y : CR-Spread
    '''
    fig, ax = plt.subplots()
    slope, intercept, r, p, std_err = stats.linregress(complete_df['RIspread_day2'], complete_df['CRspread_day2'])
    # sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', ax = ax)
    sns.regplot(data = complete_df, x = 'RIspread_day2', y = 'CRspread_day2', ax = ax)
    ax.text(0.04, -0.1, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax.set_xlabel('RI-Spread')
    ax.set_ylabel('CR-Spread')
    ax.axhline(0, color = 'k', linewidth = 0.5)
    ax.axvline(0, color = 'k', linewidth = 0.5)
    plt.show()
    
    no_outlier_df = complete_df.copy()
    no_outlier_df['hpcf_rand_day2 zscore'] = stats.zscore(no_outlier_df['hpcf_rand_day2'])
    no_outlier_df = no_outlier_df[abs(no_outlier_df['hpcf_rand_day2 zscore']) < 3]
    
    no_outlier_df['points_day2 zscore'] = stats.zscore(no_outlier_df['points_day2'])
    no_outlier_df = no_outlier_df[abs(no_outlier_df['points_day2 zscore']) < 3]
    
    '''
        Fig 3
        Subplot 1
        x : ΔRT
        y : HRCF Random DTT Day 2
        
        Subplot 2
        x : ΔER
        y : HRCF Random DTT Day 2
        
        With Outlier Detection
    '''
    fig, ax = plt.subplots(1,2, figsize=(15,5), sharey = True)
    slope, intercept, r, p, std_err = stats.linregress(no_outlier_df['RT_diff_stt_day2'], no_outlier_df['hpcf_rand_day2'])
    sns.regplot(data = no_outlier_df, x = 'RT_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[0])
    ax[0].text(-5,0.6, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0].axhline(0, color = 'k', linewidth=0.5)
    ax[0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[0].set_ylabel(r'HRCF (Random DTT)', fontsize = 20)
    ax[0].axvline(0, color = 'k', linewidth=0.5)
    
    slope, intercept, r, p, std_err = stats.linregress(no_outlier_df['ER_diff_stt_day2'], no_outlier_df['hpcf_rand_day2'])
    sns.regplot(data = no_outlier_df, x = 'ER_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[1])
    ax[1].text(-0.01,0.6, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[1].axhline(0, color = 'k', linewidth=0.5)
    ax[1].axvline(0, color = 'k', linewidth=0.5)
    ax[1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[1].set_ylabel(r'HRCF (Random DTT)', fontsize = 20)
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/Deltas_vs_GD.png', dpi=300)   
    plt.show()
    
    
    fig, ax = plt.subplots(1,2, figsize=(15,5), sharey = True)
    slope, intercept, r, p, std_err = stats.linregress(no_outlier_df['RT_diff_stt_day2'], no_outlier_df['points_day2'])
    sns.regplot(data = no_outlier_df, x = 'RT_diff_stt_day2', y = 'points_day2', ax = ax[0])
    ax[0].text(10,2075, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0].axhline(0, color = 'k', linewidth=0.5)
    ax[0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[0].set_ylabel(r'Points', fontsize = 20)
    ax[0].axvline(0, color = 'k', linewidth=0.5)
    
    slope, intercept, r, p, std_err = stats.linregress(no_outlier_df['ER_diff_stt_day2'], no_outlier_df['points_day2'])
    sns.regplot(data = no_outlier_df, x = 'ER_diff_stt_day2', y = 'points_day2', ax = ax[1])
    ax[1].text(0.005,2075, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[1].axhline(0, color = 'k', linewidth=0.5)
    ax[1].axvline(0, color = 'k', linewidth=0.5)
    ax[1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[1].set_ylabel(r'Points', fontsize = 20)
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/Points_vs_spread.png', dpi=300)    
    plt.show()
    
    '''
        Exploiters
    '''
    exploiters_df = complete_df[(complete_df['RIspread_day2'] < complete_df['RIspread_day2'].mean()) &\
                                (complete_df['ER_diff_stt_day2'] > 0) & (complete_df['CRspread_day2'] > 0) ]
        
        
    no_exploiters_df = complete_df[~complete_df['ID'].isin(exploiters_df['ID'])]
    
    exploiters_df['RT_diff_stt_day2'].mean()
    exploiters_df['ER_diff_stt_day2'].mean()
    
    '''
        x : ΔRT
        y : ΔER
        hue : RI-Spread
    '''
    fig, ax = plt.subplots()
    slope, intercept, r, p, std_err = stats.linregress(complete_df['RT Diff STT Day 1'], complete_df['ER Diff STT Day 1'])
    sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue = 'RIspread_day2', ax = ax)
    # ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax.axhline(0, color='k', linewidth = 0.5)
    ax.axvline(0, color='k', linewidth = 0.5)
    sns.move_legend(ax, "lower right")
    ax.set_xlabel(r'$\Delta$RT (ms)')
    ax.set_ylabel(r'$\Delta$ER (pp)')
    ax.set_xlim([-10, 40])
    ax.set_ylim([-0.01, 0.04])
    ax.legend(title='RI Spread (pp)')
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/Deltas_RIspread_hue_exploiters.svg', dpi=300)
    plt.show()

    '''
        x : ΔRT
        y : ΔER
        hue : RI-Spread
        red : exploiters
    '''
    plt.rcParams['legend.title_fontsize'] = 'x-large'
    fig, ax = plt.subplots()
    sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue = 'RIspread_day2', ax = ax)
    sns.scatterplot(data = exploiters_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', color ='red')
    ax.yaxis.set_tick_params(labelsize = 15)
    ax.xaxis.set_tick_params(labelsize = 15)
    ax.axhline(0, color='k', linewidth = 0.5)
    ax.axvline(0, color='k', linewidth = 0.5)
    sns.move_legend(ax, "lower right")
    ax.set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax.set_ylabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax.legend(title='RI-Spread (pp)', fontsize = 10)
    ax.set_xlim([-10, 40])
    ax.set_ylim([-0.01, 0.04])
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/Deltas_RIspread_hue_exploiters.png', dpi=300, bbox_inches='tight')
    plt.show()

    '''
        x : ΔRT
        y : ΔER
        hue : CR-Spread
        red : exploiters
    '''
    plt.rcParams['legend.title_fontsize'] = 'x-large'
    fig, ax = plt.subplots()
    sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue = 'CRspread_day2', ax = ax)
    # sns.scatterplot(data = exploiters_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', color ='red')
    ax.yaxis.set_tick_params(labelsize = 15)
    ax.xaxis.set_tick_params(labelsize = 15)
    ax.axhline(0, color='k', linewidth = 0.5)
    ax.axvline(0, color='k', linewidth = 0.5)
    sns.move_legend(ax, "lower right")
    ax.set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax.set_ylabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax.legend(title='CR-Spread (pp)', fontsize = 10)
    ax.set_xlim([-10, 40])
    ax.set_ylim([-0.01, 0.04])
    # plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/Deltas_RIspread_hue_exploiters.png', dpi=300, bbox_inches='tight')
    plt.show()

    exploiters_df_lowCR = exploiters_df[exploiters_df['CRspread_day2']<0.06]
    plt.rcParams['legend.title_fontsize'] = 'x-large'
    fig, ax = plt.subplots()
    sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue = 'CRspread_day2', ax = ax)
    sns.scatterplot(data = exploiters_df_lowCR, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', color ='green')
    ax.yaxis.set_tick_params(labelsize = 15)
    ax.xaxis.set_tick_params(labelsize = 15)
    ax.axhline(0, color='k', linewidth = 0.5)
    ax.axvline(0, color='k', linewidth = 0.5)
    sns.move_legend(ax, "lower right")
    ax.set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax.set_ylabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax.legend(title='CR-Spread (pp)', fontsize = 10)
    ax.set_xlim([-10, 40])
    ax.set_ylim([-0.01, 0.04])
    # ax.legend(bbox_to_anchor=(1.1, 0))
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/Deltas_RIspread_hue_exploiters_CR.png', dpi=300, bbox_inches='tight')
    plt.show()

    '''
        Top plot
        x : ΔRT
        y : CR-Spread
        red : exploiters
    '''
    fig, ax = plt.subplots(2,1, figsize=(7,10), sharey=True)
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df['RT_diff_stt_day2'], no_exploiters_df['CRspread_day2'])
    sns.scatterplot(data = no_exploiters_df, x = 'RT_diff_stt_day2', y = 'CRspread_day2', ax = ax[0])
    sns.scatterplot(data = exploiters_df, x = 'RT_diff_stt_day2', y = 'CRspread_day2', ax = ax[0], color='red')
    # ax[0].text(10,-0.03, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[0].yaxis.set_tick_params(labelsize = 15)
    ax[0].xaxis.set_tick_params(labelsize = 15)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax[0].axhline(0, color = 'k', linewidth = 0.5)
    ax[0].axvline(0, color = 'k', linewidth = 0.5)
    ax[0].set_ylabel('CR-Spread (pp)', fontsize = 20)
    ax[0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df['ER_diff_stt_day2'], no_exploiters_df['CRspread_day2'])
    sns.scatterplot(data = no_exploiters_df, x = 'ER_diff_stt_day2', y = 'CRspread_day2', ax = ax[1])
    sns.scatterplot(data = exploiters_df, x = 'ER_diff_stt_day2', y = 'CRspread_day2', ax = ax[1], color='red')
    # ax[1].text(0.004, -0.03, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[1].yaxis.set_tick_params(labelsize = 15)
    ax[1].xaxis.set_tick_params(labelsize = 15)
    ax[1].axhline(0, color = 'k', linewidth = 0.5)
    ax[1].axvline(0, color = 'k', linewidth = 0.5)
    ax[1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[1].set_ylabel('CR-Spread (pp)', fontsize = 20)
    # plt.suptitle('Exploiters')
    plt.show()

    '''
        ΔRT & ΔER vs RI-Spread
        Top Row: No Exploiters
        Bottom Row: Exploiters
    '''
    fig, ax = plt.subplots(2,2, figsize=(15,10), sharey=True)
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df['RT_diff_stt_day2'], no_exploiters_df['RIspread_day2'])
    sns.regplot(data = no_exploiters_df, x = 'RT_diff_stt_day2', y = 'RIspread_day2', ax = ax[0,0])
    ax[0,0].text(10,-0.03, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[0,0].yaxis.set_tick_params(labelsize = 15)
    ax[0,0].xaxis.set_tick_params(labelsize = 15)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax[0,0].axhline(0, color = 'k', linewidth = 0.5)
    ax[0,0].axvline(0, color = 'k', linewidth = 0.5)
    ax[0,0].set_ylabel('RI-Spread (pp)', fontsize = 20)
    ax[0,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df['ER_diff_stt_day2'], no_exploiters_df['RIspread_day2'])
    sns.regplot(data = no_exploiters_df, x = 'ER_diff_stt_day2', y = 'RIspread_day2', ax = ax[0,1])
    ax[0,1].text(0.004, -0.03, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[0,1].yaxis.set_tick_params(labelsize = 15)
    ax[0,1].xaxis.set_tick_params(labelsize = 15)
    ax[0,1].axhline(0, color = 'k', linewidth = 0.5)
    ax[0,1].axvline(0, color = 'k', linewidth = 0.5)
    ax[0,1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[0,1].set_ylabel('RI-Spread (pp)', fontsize = 20)
    # plt.suptitle('Exploiters')
    
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df['RT_diff_stt_day2'], exploiters_df['RIspread_day2'])
    sns.regplot(data = exploiters_df, x = 'RT_diff_stt_day2', y = 'RIspread_day2', ax = ax[1,0])
    ax[1,0].text(5, 0.1, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[1,0].yaxis.set_tick_params(labelsize = 15)
    ax[1,0].xaxis.set_tick_params(labelsize = 15)
    ax[1,0].axhline(0, color = 'k', linewidth = 0.5)
    ax[1,0].axvline(0, color = 'k', linewidth = 0.5)
    ax[1,0].set_ylabel('RI-Spread (pp)', fontsize = 20)
    ax[1,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[1,0].sharex(ax[0,0])
    
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df['ER_diff_stt_day2'], exploiters_df['RIspread_day2'])
    sns.regplot(data = exploiters_df, x = 'ER_diff_stt_day2', y = 'RIspread_day2', ax = ax[1,1])
    ax[1,1].text(0.004, 0.1, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[1,1].yaxis.set_tick_params(labelsize = 15)
    ax[1,1].xaxis.set_tick_params(labelsize = 15)
    ax[1,1].axhline(0, color = 'k', linewidth = 0.5)
    ax[1,1].axvline(0, color = 'k', linewidth = 0.5)
    ax[1,1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[1,1].set_ylabel('RI-Spread (pp)', fontsize = 20)
    ax[1,1].sharex(ax[0,1])
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/Deltas_vs_RISpread_exploiters.png', dpi=300)
    plt.show()
    
    '''
        x : RI-Spread
        y : CR-Spread
        Left : No Exploiters
        Right : Exploiters
    '''
    fig, ax = plt.subplots(1,2, figsize=(15,8), sharey=True, sharex = True)
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df['RIspread_day2'], no_exploiters_df['CRspread_day2'])
    # sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', ax = ax)
    sns.regplot(data = no_exploiters_df, x = 'RIspread_day2', y = 'CRspread_day2', ax = ax[0])
    ax[0].text(0.04, -0.1, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[0].yaxis.set_tick_params(labelsize = 15)
    ax[0].xaxis.set_tick_params(labelsize = 15)
    ax[0].set_xlabel('RI-Spread', fontsize = 20)
    ax[0].set_ylabel('CR-Spread', fontsize = 20)
    ax[0].axhline(0, color = 'k', linewidth = 0.5)
    ax[0].axvline(0, color = 'k', linewidth = 0.5)
    
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df['RIspread_day2'], exploiters_df['CRspread_day2'])
    # sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', ax = ax)
    sns.regplot(data = exploiters_df, x = 'RIspread_day2', y = 'CRspread_day2', ax = ax[1])
    ax[1].text(0.04, -0.1, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[1].yaxis.set_tick_params(labelsize = 15)
    ax[1].xaxis.set_tick_params(labelsize = 15)
    ax[1].set_xlabel('RI-Spread', fontsize = 20)
    ax[1].set_ylabel('CR-Spread', fontsize = 20)
    ax[1].axhline(0, color = 'k', linewidth = 0.5)
    ax[1].axvline(0, color = 'k', linewidth = 0.5)
    plt.show()
    
    test_df = complete_df[(complete_df['RIspread_day2'] < complete_df['RIspread_day2'].mean()) & \
                          (complete_df['CRspread_day2'] > 0)]
        
    '''
        x : RI-Spread
        y : CR-Spread
        test_df
    '''
    fig, ax = plt.subplots()
    slope, intercept, r, p, std_err = stats.linregress(test_df['RIspread_day2'], test_df['CRspread_day2'])
    # sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', ax = ax)
    sns.regplot(data = test_df, x = 'RIspread_day2', y = 'CRspread_day2', ax = ax)
    ax.text(-0.04, 0.125, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax.yaxis.set_tick_params(labelsize = 15)
    ax.xaxis.set_tick_params(labelsize = 15)
    ax.set_xlabel('RI-Spread', fontsize = 20)
    ax.set_ylabel('CR-Spread', fontsize = 20)
    ax.axhline(0, color = 'k', linewidth = 0.5)
    ax.axvline(0, color = 'k', linewidth = 0.5)
    plt.title('testdf')
    plt.show()
    
    '''
        ΔRT & ΔER vs HRCF
        Top Row: No Exploiters
        Bottom Row: Exploiters
    '''
    fig, ax = plt.subplots(2,2, figsize=(15,10), sharey=True)
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df['RT_diff_stt_day2'], no_exploiters_df['hpcf_rand_day2'])
    sns.regplot(data = no_exploiters_df, x = 'RT_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[0,0])
    ax[0,0].text(10,0.42, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[0,0].yaxis.set_tick_params(labelsize = 15)
    ax[0,0].xaxis.set_tick_params(labelsize = 15)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0,0].axhline(0, color = 'k', linewidth = 0.5)
    ax[0,0].axvline(0, color = 'k', linewidth = 0.5)
    ax[0,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[0,0].set_ylabel('HRCF', fontsize = 20)
    
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df['ER_diff_stt_day2'], no_exploiters_df['hpcf_rand_day2'])
    sns.regplot(data = no_exploiters_df, x = 'ER_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[0,1])
    ax[0,1].text(0.004, 0.42, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[0,1].yaxis.set_tick_params(labelsize = 15)
    ax[0,1].xaxis.set_tick_params(labelsize = 15)
    # ax[0,1].axhline(0, color = 'k', linewidth = 0.5)
    ax[0,1].axvline(0, color = 'k', linewidth = 0.5)
    ax[0,1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[0,1].set_ylabel('HRCF', fontsize = 20)
    # plt.suptitle('Exploiters')
    
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df['RT_diff_stt_day2'], exploiters_df['hpcf_rand_day2'])
    sns.regplot(data = exploiters_df, x = 'RT_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[1,0])
    ax[1,0].text(10, 0.42, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[1,0].yaxis.set_tick_params(labelsize = 15)
    ax[1,0].xaxis.set_tick_params(labelsize = 15)
    # ax[1,0].axhline(0, color = 'k', linewidth = 0.5)
    ax[1,0].axvline(0, color = 'k', linewidth = 0.5)
    ax[1,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[1,0].set_ylabel('HRCF', fontsize = 20)
    ax[1,0].sharex(ax[0,0])
    
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df['ER_diff_stt_day2'], exploiters_df['hpcf_rand_day2'])
    sns.regplot(data = exploiters_df, x = 'ER_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[1,1])
    ax[1,1].text(0.004, 0.42, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[1,1].yaxis.set_tick_params(labelsize = 15)
    ax[1,1].xaxis.set_tick_params(labelsize = 15)
    # ax[1,1].axhline(0, color = 'k', linewidth = 0.5)
    ax[1,1].axvline(0, color = 'k', linewidth = 0.5)
    ax[1,1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[1,1].set_ylabel('HRCF', fontsize = 20)
    ax[1,1].sharex(ax[0,1])
    plt.show()
    
    '''
        ΔRT & ΔER vs Points
    '''
    fig, ax = plt.subplots(2,2, figsize=(15,10), sharey=True)
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df['RT_diff_stt_day2'], no_exploiters_df['points_day2'])
    sns.regplot(data = no_exploiters_df, x = 'RT_diff_stt_day2', y = 'points_day2', ax = ax[0,0])
    ax[0,0].text(10,2075, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[0,0].yaxis.set_tick_params(labelsize = 15)
    ax[0,0].xaxis.set_tick_params(labelsize = 15)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0,0].axhline(0, color = 'k', linewidth = 0.5)
    ax[0,0].axvline(0, color = 'k', linewidth = 0.5)
    ax[0,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[0,0].set_ylabel('Points', fontsize = 20)
    
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df['ER_diff_stt_day2'], no_exploiters_df['points_day2'])
    sns.regplot(data = no_exploiters_df, x = 'ER_diff_stt_day2', y = 'points_day2', ax = ax[0,1])
    ax[0,1].text(0.004, 2075, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[0,1].yaxis.set_tick_params(labelsize = 15)
    ax[0,1].xaxis.set_tick_params(labelsize = 15)
    # ax[0,1].axhline(0, color = 'k', linewidth = 0.5)
    ax[0,1].axvline(0, color = 'k', linewidth = 0.5)
    ax[0,1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[0,1].set_ylabel('Points', fontsize = 20)
    # plt.suptitle('Exploiters')
    
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df['RT_diff_stt_day2'], exploiters_df['points_day2'])
    sns.regplot(data = exploiters_df, x = 'RT_diff_stt_day2', y = 'points_day2', ax = ax[1,0])
    ax[1,0].text(10, 1700, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[1,0].yaxis.set_tick_params(labelsize = 15)
    ax[1,0].xaxis.set_tick_params(labelsize = 15)
    # ax[1,0].axhline(0, color = 'k', linewidth = 0.5)
    ax[1,0].axvline(0, color = 'k', linewidth = 0.5)
    ax[1,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[1,0].set_ylabel('Points', fontsize = 20)
    ax[1,0].sharex(ax[0,0])
    
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df['ER_diff_stt_day2'], exploiters_df['points_day2'])
    sns.regplot(data = exploiters_df, x = 'ER_diff_stt_day2', y = 'points_day2', ax = ax[1,1])
    ax[1,1].text(0.004, 1700, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[1,1].yaxis.set_tick_params(labelsize = 15)
    ax[1,1].xaxis.set_tick_params(labelsize = 15)
    # ax[1,1].axhline(0, color = 'k', linewidth = 0.5)
    ax[1,1].axvline(0, color = 'k', linewidth = 0.5)
    ax[1,1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[1,1].set_ylabel('Points', fontsize = 20)
    ax[1,1].sharex(ax[0,1])
    plt.show()
    
    
    '''
        x : ΔRT
        y : HRCF Random DTT
        only exploiters
        
        x : ΔER
        y : HRCF Random DTT
        only exploiters
    '''
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df['RT_diff_stt_day2'], exploiters_df['hpcf_rand_day2'])
    sns.regplot(data = exploiters_df, x = 'RT_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[0])
    ax[0].text(21,0.97, 'r=%.4f, p = %.4f'%(r,p))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0].axhline(0, color = 'k', linewidth=0.5)
    ax[0].set_xlabel(r'$\Delta$RT')
    ax[0].set_ylabel(r'HRCF (Random DTT)')
    ax[0].axvline(0, color = 'k', linewidth=0.5)
    
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df['ER_diff_stt_day2'], exploiters_df['hpcf_rand_day2'])
    sns.regplot(data = exploiters_df, x = 'ER_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[1])
    ax[1].text(0.021,0.97, 'r=%.4f, p = %.4f'%(r,p))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[1].axhline(0, color = 'k', linewidth=0.5)
    ax[1].axvline(0, color = 'k', linewidth=0.5)
    ax[1].set_xlabel(r'$\Delta$ER')
    ax[1].set_ylabel(r'HRCF (Random DTT)')
    plt.show()
    
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df['RT_diff_stt_day2'], no_exploiters_df['hpcf_rand_day2'])
    sns.regplot(data = no_exploiters_df, x = 'RT_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[0])
    ax[0].text(21,0.97, 'r=%.4f, p = %.4f'%(r,p))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0].axhline(0, color = 'k', linewidth=0.5)
    ax[0].set_xlabel(r'$\Delta$RT')
    ax[0].set_ylabel(r'HRCF (Random DTT)')
    ax[0].axvline(0, color = 'k', linewidth=0.5)
    
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df['ER_diff_stt_day2'], no_exploiters_df['hpcf_rand_day2'])
    sns.regplot(data = no_exploiters_df, x = 'ER_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[1])
    ax[1].text(0.021,0.97, 'r=%.4f, p = %.4f'%(r,p))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[1].axhline(0, color = 'k', linewidth=0.5)
    ax[1].axvline(0, color = 'k', linewidth=0.5)
    ax[1].set_xlabel(r'$\Delta$ER')
    ax[1].set_ylabel(r'HRCF (Random DTT)')
    plt.show()
    
    '''
        x : HPCF Random DTT Day 2
        y : CR-Spread
        only exploiters
    '''
    fig, ax = plt.subplots()
    sns.scatterplot(data = exploiters_df, x = 'hpcf_rand_day2', y = 'CRspread_day2', ax = ax, color = 'red')
    # ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax.axhline(0, color='k', linewidth = 0.5)
    ax.axhline(exploiters_df['CRspread_day2'].mean(), color='r', linewidth = 0.5)
    ax.axhline(no_exploiters_df['CRspread_day2'].mean(), color='b', linewidth = 0.5)
    ax.axvline(exploiters_df['hpcf_rand_day2'].mean(), color='r', linewidth = 0.5)
    ax.axvline(no_exploiters_df['hpcf_rand_day2'].mean(), color='b', linewidth = 0.5)
    ax.set_ylabel('CR-Spread')
    ax.set_xlabel('HRCF Random DTT')
    # ax.axvline(0)
    plt.title('Exploiters')
    plt.show()
    
    scipy.stats.ttest_ind(exploiters_df['CRspread_day2'], no_exploiters_df['CRspread_day2'])
    scipy.stats.ttest_ind(exploiters_df['hpcf_rand_day2'], no_exploiters_df['hpcf_rand_day2'])
    scipy.stats.ttest_ind(exploiters_df['RIspread_day2'], no_exploiters_df['RIspread_day2'])
    
    compare_df = pd.DataFrame()
    hpcf_day2 = []
    hpcf_day2.extend(np.array(exploiters_df['hpcf_day2'], dtype = float))
    hpcf_day2.extend(np.array(no_exploiters_df['hpcf_day2'], dtype = float))
    
    hpcf_rand_day2 = []
    hpcf_rand_day2.extend(np.array(exploiters_df['hpcf_rand_day2'], dtype = float))
    hpcf_rand_day2.extend(np.array(no_exploiters_df['hpcf_rand_day2'], dtype = float))
    
    agenttype = []
    agenttype.extend(['exploiter']*len(exploiters_df))
    agenttype.extend(['no exploiter']*len(no_exploiters_df))
    
    points_day2 = []
    points_day2.extend(np.array(exploiters_df['points_day2'], dtype = float))
    points_day2.extend(np.array(no_exploiters_df['points_day2'], dtype = float))
    
    compare_df['hpcf_day2'] = hpcf_day2
    compare_df['agenttype'] = agenttype
    compare_df['points_day2'] = points_day2
    compare_df['hpcf_rand_day2'] = hpcf_rand_day2
    
    '''
        violinplot
        HPCF Day 2
        x : exploiters, non-exploiters
    '''
    fig, ax = plt.subplots(1,2, figsize=(25,10))
    # slope, intercept, r, p, std_err = stats.linregress(exploiters_df['RT_diff_stt_day2'], exploiters_df['RIspread_day2'])
    sns.violinplot(data = compare_df, x = 'agenttype', y = 'hpcf_rand_day2', ax = ax[0])
    ax[0].yaxis.set_tick_params(labelsize = 30)
    ax[0].xaxis.set_tick_params(labelsize = 30)
    # ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax.axhline(0, color='k', linewidth = 0.5)
    # ax.set_ylabel('CR-Spread')
    # ax.set_xlabel('HRCF Random DTT')
    # ax.axvline(0)
    # plt.title('Exploiters')
    ax[0].set_ylabel('HRCF Random DTT (%)', fontsize = 25)
    ax[0].set_xlabel('Participant', fontsize = 25)
    # plt.show()
    
    scipy.stats.ttest_ind(exploiters_df['hpcf_rand_day2'], no_exploiters_df['hpcf_rand_day2'])

    # fig, ax = plt.subplots()
    # slope, intercept, r, p, std_err = stats.linregress(exploiters_df['RT_diff_stt_day2'], exploiters_df['RIspread_day2'])
    sns.violinplot(data = compare_df, x = 'agenttype', y = 'points_day2', ax = ax[1])
    ax[1].yaxis.set_tick_params(labelsize = 30)
    ax[1].xaxis.set_tick_params(labelsize = 30)
    # ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax.axhline(0, color='k', linewidth = 0.5)
    # ax.set_ylabel('CR-Spread')
    # ax.set_xlabel('HRCF Random DTT')
    # ax.axvline(0)
    # plt.title('Exploiters')
    ax[1].set_ylabel('Points Total', fontsize = 25)
    ax[1].set_xlabel('Participant', fontsize = 25)
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/violinplot_hpcf_exploiters_vs_nonexploiters.png', dpi=300)
    plt.show()

    scipy.stats.ttest_ind(np.array(exploiters_df['points_day2'], dtype=float), 
                          np.array(no_exploiters_df['points_day2'], dtype = float))

elif 'conflict_param_day2' in complete_df.columns:
    
    '''
        Now with modelfit parameters
        
        Exploiters:
            - High θ_R + θ_Conf
            - ΔER > 0
    '''
    '''
        x : ΔRT
        y : ΔER
        hue = θ_R
    '''
    fig, ax = plt.subplots()
    slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['ER_diff_stt_day2'])
    # sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', ax = ax)
    sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', ax = ax, hue = 'theta_rep_day2')
    ax.text(22, 0.035, 'r=%.4f, p = %.4f'%(r,p))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax.set_xlabel(r'$\Delta$RT')
    ax.set_ylabel(r'$\Delta$ER')
    ax.axhline(0, color = 'k', linewidth = 0.5)
    ax.axvline(0, color = 'k', linewidth = 0.5)
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/modeling_fig1.png', dpi=300)
    plt.show()
    
    '''
        Subplot 1
        x : ΔRT
        y : θ_R
        
        Subplot 2
        x : ΔER
        y : θ_R
        
        Without Outlier Detection
    '''
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['theta_rep_day2'])
    sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'theta_rep_day2', ax = ax[0])
    ax[0].text(15,0.1, 'r=%.4f, p = %.4f'%(r,p), fontsize = 15)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax[0].axhline(0, color = 'k', linewidth = 0.5)
    ax[0].axvline(0, color = 'k', linewidth = 0.5)
    ax[0].set_ylabel(r'$\Theta_R$', fontsize = 20)
    ax[0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    
    slope, intercept, r, p, std_err = stats.linregress(complete_df['ER_diff_stt_day2'], complete_df['theta_rep_day2'])
    sns.regplot(data = complete_df, x = 'ER_diff_stt_day2', y = 'theta_rep_day2', ax = ax[1])
    ax[1].text(0.01, 0.1, 'r=%.4f, p = %.4f'%(r,p), fontsize = 15)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax[1].axhline(0, color = 'k', linewidth = 0.5)
    ax[1].axvline(0, color = 'k', linewidth = 0.5)
    ax[1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[1].set_ylabel(r'$\Theta_R$', fontsize = 20)
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/modeling_fig2.png', dpi=300)
    plt.show()
    
    '''
        Subplot 1
        x : ΔRT
        y : θ_R + θ_Comb
        
        Subplot 2
        x : ΔER
        y : θ_R + θ_Comb
        
        Without Outlier Detection
    '''
    complete_df['theta_comb_day2'] = complete_df['theta_rep_day2'] + complete_df['conflict_param_day2']
    complete_df['theta_anticomb_day2'] = complete_df['theta_rep_day2'] - complete_df['conflict_param_day2']
    
    fig, ax = plt.subplots(1,2, figsize=(15,5), sharey=True)
    slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['theta_anticomb_day2'])
    sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'theta_anticomb_day2', ax = ax[0])
    ax[0].text(10,0.8, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[0].yaxis.set_tick_params(labelsize = 15)
    ax[0].xaxis.set_tick_params(labelsize = 15)
    ax[0].axhline(0, color = 'k', linewidth = 0.5)
    ax[0].axvline(0, color = 'k', linewidth = 0.5)
    ax[0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[0].set_ylabel(r'$\Theta_{R} - \Theta_{Conflict}$', fontsize = 20)
    
    slope, intercept, r, p, std_err = stats.linregress(complete_df['ER_diff_stt_day2'], complete_df['theta_anticomb_day2'])
    sns.regplot(data = complete_df, x = 'ER_diff_stt_day2', y = 'theta_anticomb_day2', ax = ax[1])
    ax[1].text(0.007, 0.8, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[1].yaxis.set_tick_params(labelsize = 15)
    ax[1].xaxis.set_tick_params(labelsize = 15)
    ax[1].axhline(0, color = 'k', linewidth = 0.5)
    ax[1].axvline(0, color = 'k', linewidth = 0.5)
    ax[1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[1].set_ylabel(r'$\Theta_{R} - \Theta_{Conflict}$', fontsize = 20)
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/modeling_fig3.png', dpi=300)
    plt.show()
    
    '''
        Subplot 1
        x : ΔRT
        y : θ_Conf
        
        Subplot 2
        x : ΔER
        y : θ_Conf
        
        Without Outlier Detection
    '''  
    fig, ax = plt.subplots(1,2, figsize=(15,5), sharey=True)
    slope, intercept, r, p, std_err = stats.linregress(complete_df['RT_diff_stt_day2'], complete_df['conflict_param_day2'])
    sns.regplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'conflict_param_day2', ax = ax[0])
    ax[0].text(-6, 1.2, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[0].yaxis.set_tick_params(labelsize = 15)
    ax[0].xaxis.set_tick_params(labelsize = 15)
    # ax[0].axhline(0, color = 'k', linewidth = 0.5)
    ax[0].axvline(0, color = 'k', linewidth = 0.5)
    ax[0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[0].set_ylabel(r'$\Theta_{Conflict}$', fontsize = 20)
    
    slope, intercept, r, p, std_err = stats.linregress(complete_df['ER_diff_stt_day2'], complete_df['conflict_param_day2'])
    sns.regplot(data = complete_df, x = 'ER_diff_stt_day2', y = 'conflict_param_day2', ax = ax[1])
    ax[1].text(-0.01, 1.2, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[1].yaxis.set_tick_params(labelsize = 15)
    ax[1].xaxis.set_tick_params(labelsize = 15)
    # ax[1].axhline(0, color = 'k', linewidth = 0.5)
    ax[1].axvline(0, color = 'k', linewidth = 0.5)
    ax[1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[1].set_ylabel(r'$\Theta_{Conflict}$', fontsize = 20)
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/modeling_fig7.png', dpi=300)
    plt.show()
    
    '''
        Subplot 1
        x : ΔRT
        y : θ_Q
        
        Subplot 2
        x : ΔER
        y : θ_Q
        
        With Outlier Detection
    '''
    no_outlier_df = complete_df.copy()
    # no_outlier_df['hpcf_rand_day2 zscore'] = stats.zscore(no_outlier_df['hpcf_rand_day2'])
    # no_outlier_df = no_outlier_df[abs(no_outlier_df['hpcf_rand_day2 zscore']) < 3]
    
    fig, ax = plt.subplots(1,2, figsize=(15,5), sharey=True)
    slope, intercept, r, p, std_err = stats.linregress(no_outlier_df['RT_diff_stt_day2'], no_outlier_df['theta_Q_day2'])
    sns.regplot(data = no_outlier_df, x = 'RT_diff_stt_day2', y = 'theta_Q_day2', ax = ax[0])
    ax[0].text(9, 6, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0].axhline(0, color = 'k', linewidth=0.5)
    ax[0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 25)
    ax[0].set_ylabel(r'$\Theta_Q$', fontsize = 25)
    ax[0].axvline(0, color = 'k', linewidth=0.5)
    
    slope, intercept, r, p, std_err = stats.linregress(no_outlier_df['ER_diff_stt_day2'], no_outlier_df['theta_Q_day2'])
    sns.regplot(data = no_outlier_df, x = 'ER_diff_stt_day2', y = 'theta_Q_day2', ax = ax[1])
    ax[1].text(0.0065, 6, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[1].axhline(0, color = 'k', linewidth=0.5)
    ax[1].axvline(0, color = 'k', linewidth=0.5)
    ax[1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 25)
    ax[1].set_ylabel(r'$\Theta_Q$', fontsize = 25)
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/modeling_fig4.png', dpi=300)
    plt.show()
    
    '''
        x : θ_Conflict
        y : RI-Spread
    '''
    fig, ax = plt.subplots()
    slope, intercept, r, p, std_err = stats.linregress(complete_df['conflict_param_day2'], complete_df['RIspread_day2'])
    sns.regplot(data = complete_df, x = 'conflict_param_day2', y = 'RIspread_day2', ax = ax)
    ax.text(1.1,-0.03, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax.axhline(0, color='k', linewidth = 0.5)
    ax.set_ylabel(r'$\Theta_{Conflict}$', fontsize = 20)
    ax.set_xlabel('RI-Spread', fontsize = 20)
    # ax.axvline(0)
    # plt.title('Exploiters')
    # plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/modeling_fig8.png', dpi=300)
    plt.show()
    
    '''
        Exploiters
    '''
    # exploiters_df_modeled = complete_df[(complete_df['theta_comb_day2'] > complete_df['theta_comb_day2'].mean()) &\
    #                             (complete_df['ER_diff_stt_day2'] > 0)]
        
    # exploiters_df_modeled = complete_df[(complete_df['conflict_param_day2'] > complete_df['conflict_param_day2'].mean()) &\
    #                             (complete_df['theta_rep_day2'] > complete_df['theta_rep_day2'].mean()) &\
    #                             (complete_df['ER_diff_stt_day2'] > 0)]

    # exploiters_df_modeled = complete_df[(complete_df['conflict_param_day2'] > complete_df['conflict_param_day2'].mean()) &\
    #                             (complete_df['ER_diff_stt_day2'] > 0)]
    
    exploiters_df_modeled = complete_df[(complete_df['RIspread_day2'] < complete_df['RIspread_day2'].mean()) &\
                                (complete_df['theta_comb_day2'] > complete_df['theta_comb_day2'].mean()) &\
                                (complete_df['ER_diff_stt_day2'] > 0)]
        
    no_exploiters_df_modeled = complete_df[~complete_df['ID'].isin(exploiters_df_modeled['ID'])]
    
    exploiters_df_modeled['RT_diff_stt_day2'].mean()
    exploiters_df_modeled['ER_diff_stt_day2'].mean()
    
    '''
        x : ΔRT
        y : ΔER
        hue : θ_Comb
    '''
    fig, ax = plt.subplots(1,2, figsize=(20,8))
    slope, intercept, r, p, std_err = stats.linregress(complete_df['RT Diff STT Day 1'], complete_df['ER Diff STT Day 1'])
    sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue = 'theta_comb_day2', ax = ax[0])
    # ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax[0].axhline(0, color='k', linewidth = 0.5)
    ax[0].axvline(0, color='k', linewidth = 0.5)
    # sns.move_legend(ax, "lower right")
    ax[0].set_xlabel(r'$\Delta$RT (ms)', fontsize=20)
    ax[0].set_ylabel(r'$\Delta$ER (pp)', fontsize=20)
    ax[0].set_xlim([-10, 40])
    ax[0].set_ylim([-0.01, 0.04])
    ax[0].legend(title=r'$\Theta_{Comb}$', fontsize=20)
    # plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/Deltas_CISpread_hues.svg')
    # plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/modeling_fig5.png', dpi=300)
    
    slope, intercept, r, p, std_err = stats.linregress(complete_df['RT Diff STT Day 1'], complete_df['ER Diff STT Day 1'])
    sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue = 'conflict_param_day2', ax = ax[1])
    # ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax[1].axhline(0, color='k', linewidth = 0.5)
    ax[1].axvline(0, color='k', linewidth = 0.5)
    # sns.move_legend(ax, "lower right")
    ax[1].set_xlabel(r'$\Delta$RT (ms)', fontsize=20)
    ax[1].set_ylabel(r'$\Delta$ER (pp)', fontsize=20)
    ax[1].set_xlim([-10, 40])
    ax[1].set_ylim([-0.01, 0.04])
    ax[1].legend(title=r'$\Theta_{Conf}$', fontsize=20)
    # plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/Deltas_CISpread_hues.svg')
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/modeling_fig4.svg')
    plt.show()
    
    '''
        x : ΔRT
        y : ΔER
        hue : θ_Comb
        red: Exploiters
    '''
    fig, ax = plt.subplots(1,2, figsize=(20,8))
    slope, intercept, r, p, std_err = stats.linregress(complete_df['RT Diff STT Day 1'], complete_df['ER Diff STT Day 1'])
    sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue = 'theta_comb_day2', ax = ax[0])
    sns.scatterplot(data = exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', color ='red', ax = ax[0])
    # ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax[0].axhline(0, color='k', linewidth = 0.5)
    ax[0].axvline(0, color='k', linewidth = 0.5)
    # sns.move_legend(ax, "lower right")
    ax[0].set_xlabel(r'$\Delta$RT (ms)', fontsize=20)
    ax[0].set_ylabel(r'$\Delta$ER (pp)', fontsize=20)
    ax[0].set_xlim([-10, 40])
    ax[0].set_ylim([-0.01, 0.04])
    ax[0].legend(title=r'$\Theta_{Comb}$', fontsize=20)
    # plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/Deltas_CISpread_hues.svg')
    # plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/modeling_fig5.png', dpi=300)
    
    slope, intercept, r, p, std_err = stats.linregress(complete_df['RT Diff STT Day 1'], complete_df['ER Diff STT Day 1'])
    sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue = 'conflict_param_day2', ax = ax[1])
    sns.scatterplot(data = exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', color ='red', ax = ax[1])
    # ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax[1].axhline(0, color='k', linewidth = 0.5)
    ax[1].axvline(0, color='k', linewidth = 0.5)
    # sns.move_legend(ax, "lower right")
    ax[1].set_xlabel(r'$\Delta$RT (ms)', fontsize=20)
    ax[1].set_ylabel(r'$\Delta$ER (pp)', fontsize=20)
    ax[1].set_xlim([-10, 40])
    ax[1].set_ylim([-0.01, 0.04])
    ax[1].legend(title=r'$\Theta_{Conf}$', fontsize=20)
    # plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/Deltas_CISpread_hues.svg')
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/modeling_fig5.svg')
    plt.show()
    
    '''
        Top plot
        x : ΔRT
        y : CR-Spread
        red : exploiters
    '''
    fig, ax = plt.subplots(2,1, figsize=(7,10), sharey=True)
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df['RT_diff_stt_day2'], no_exploiters_df['CRspread_day2'])
    sns.scatterplot(data = no_exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'CRspread_day2', ax = ax[0])
    sns.scatterplot(data = exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'CRspread_day2', ax = ax[0], color='red')
    # ax[0].text(10,-0.03, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[0].yaxis.set_tick_params(labelsize = 15)
    ax[0].xaxis.set_tick_params(labelsize = 15)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax[0].axhline(0, color = 'k', linewidth = 0.5)
    ax[0].axvline(0, color = 'k', linewidth = 0.5)
    ax[0].set_ylabel('CR-Spread (pp)', fontsize = 20)
    ax[0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df['ER_diff_stt_day2'], no_exploiters_df['CRspread_day2'])
    sns.scatterplot(data = no_exploiters_df_modeled, x = 'ER_diff_stt_day2', y = 'CRspread_day2', ax = ax[1])
    sns.scatterplot(data = exploiters_df_modeled, x = 'ER_diff_stt_day2', y = 'CRspread_day2', ax = ax[1], color='red')
    # ax[1].text(0.004, -0.03, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[1].yaxis.set_tick_params(labelsize = 15)
    ax[1].xaxis.set_tick_params(labelsize = 15)
    ax[1].axhline(0, color = 'k', linewidth = 0.5)
    ax[1].axvline(0, color = 'k', linewidth = 0.5)
    ax[1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[1].set_ylabel('CR-Spread (pp)', fontsize = 20)
    # plt.suptitle('Exploiters')
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/exploiters_modeled.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    '''
        x : ΔRT
        y : ΔER
        hue : θ_Comb
        red: exploiters
    '''
    plt.rcParams['legend.title_fontsize'] = 'x-large'
    fig, ax = plt.subplots()
    sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue = 'theta_comb_day2', ax = ax)
    # sns.scatterplot(data = exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', color ='red')
    # ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax.axhline(0, color='k', linewidth = 0.5)
    ax.axvline(0, color='k', linewidth = 0.5)
    # sns.move_legend(ax, "lower right")
    ax.set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax.set_ylabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax.legend(title=r'$\Theta_{Comb}$', fontsize = 10)
    ax.set_xlim([-10, 40])
    ax.set_ylim([-0.01, 0.04])
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/modeling_fig6a.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    '''
        x : ΔRT
        y : ΔER
        hue : θ_Conf
    '''
    fig, ax = plt.subplots()
    slope, intercept, r, p, std_err = stats.linregress(complete_df['RT Diff STT Day 1'], complete_df['ER Diff STT Day 1'])
    sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue = 'conflict_param_day2', ax = ax)
    # ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax.axhline(0, color='k', linewidth = 0.5)
    ax.axvline(0, color='k', linewidth = 0.5)
    # sns.move_legend(ax, "lower right")
    ax.set_xlabel(r'$\Delta$RT (ms)')
    ax.set_ylabel(r'$\Delta$ER (pp)')
    ax.set_xlim([-10, 40])
    ax.set_ylim([-0.01, 0.04])
    ax.legend(title=r'$\Theta_{Conf}$')
    # plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/BehavioralAnalysis/N60/Deltas_CISpread_hues.svg')
    # plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/modeling_fig5.png', dpi=300)
    plt.show()
    
    '''
        x : ΔRT
        y : ΔER
        hue : θ_Conflict
        red: exploiters
    '''
    fig, ax = plt.subplots()
    sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', hue='conflict_param_day2', ax = ax)
    sns.scatterplot(data = exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', color ='red')
    # ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax.axhline(0, color='k', linewidth = 0.5)
    ax.axvline(0, color='k', linewidth = 0.5)
    # sns.move_legend(ax, "lower right")
    ax.set_xlabel(r'$\Delta$RT (ms)')
    ax.set_ylabel(r'$\Delta$ER (pp)')
    ax.legend(title=r'$\Theta_{Conf}$')
    ax.set_xlim([-10, 40])
    ax.set_ylim([-0.01, 0.04])
    # plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/modeling_fig6.png', dpi=300)
    plt.show()
    
    
    '''
        x : HPCF Random DTT Day 2
        y : θ_Q
        only exploiters
    '''
    fig, ax = plt.subplots()
    sns.scatterplot(data = exploiters_df_modeled, x = 'hpcf_rand_day2', y = 'theta_comb_day2', ax = ax, color = 'red')
    # ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    ax.axhline(0, color='k', linewidth = 0.5)
    ax.axhline(exploiters_df_modeled['theta_comb_day2'].mean(), color='r', linewidth = 0.5)
    ax.axhline(no_exploiters_df_modeled['theta_comb_day2'].mean(), color='b', linewidth = 0.5)
    ax.axvline(exploiters_df_modeled['hpcf_rand_day2'].mean(), color='r', linewidth = 0.5)
    ax.axvline(no_exploiters_df_modeled['hpcf_rand_day2'].mean(), color='b', linewidth = 0.5)
    ax.set_ylabel(r'$\Theta_{Comb}$')
    ax.set_xlabel('HRCF Random DTT')
    # ax.axvline(0)
    plt.title('Exploiters')
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/modeling_fig8.png', dpi=300)
    plt.show()

    '''
        x : ΔRT (left) & ΔER (right)
        y : θ_R - θ_Conflict
        top row: non-exploiters
        bottom row: exploiters
    '''
    fig, ax = plt.subplots(2,2, figsize=(15,10), sharey=True)
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df_modeled['RT_diff_stt_day2'], no_exploiters_df_modeled['theta_anticomb_day2'])
    sns.regplot(data = no_exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'theta_anticomb_day2', ax = ax[0,0])
    ax[0,0].text(10,0, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0,0].axhline(0, color = 'k', linewidth=0.5)
    ax[0,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[0,0].set_ylabel(r'$\Theta_R - \Theta_{Conflict}$', fontsize = 20)
    ax[0,0].axvline(0, color = 'k', linewidth=0.5)
    
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df_modeled['ER_diff_stt_day2'], no_exploiters_df_modeled['theta_anticomb_day2'])
    sns.regplot(data = no_exploiters_df_modeled, x = 'ER_diff_stt_day2', y = 'theta_anticomb_day2', ax = ax[0,1])
    ax[0,1].text(0.01,0., 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0,1].axhline(0, color = 'k', linewidth=0.5)
    ax[0,1].axvline(0, color = 'k', linewidth=0.5)
    ax[0,1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[0,1].set_ylabel(r'$\Theta_R - \Theta_{Conflict}$', fontsize = 20)
    
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df_modeled['RT_diff_stt_day2'], exploiters_df_modeled['theta_anticomb_day2'])
    sns.regplot(data = exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'theta_anticomb_day2', ax = ax[1,0])
    ax[1,0].text(10,0, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[1,0].axhline(0, color = 'k', linewidth=0.5)
    ax[1,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[1,0].set_ylabel(r'$\Theta_R - \Theta_{Conflict}$', fontsize = 20)
    ax[1,0].axvline(0, color = 'k', linewidth=0.5)
    
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df_modeled['ER_diff_stt_day2'], exploiters_df_modeled['theta_anticomb_day2'])
    sns.regplot(data = exploiters_df_modeled, x = 'ER_diff_stt_day2', y = 'theta_anticomb_day2', ax = ax[1,1])
    ax[1,1].text(0.01,0, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[1,1].axhline(0, color = 'k', linewidth=0.5)
    ax[1,1].axvline(0, color = 'k', linewidth=0.5)
    ax[1,1].set_xlabel(r'$\Delta$ER (pp)', fontsize=20)
    ax[1,1].set_ylabel(r'$\Theta_R - \Theta_{Conflict}$', fontsize = 20)
    plt.show()
    
    '''
        x : ΔRT (left) & ΔER (right)
        y : θ_Q
        top row: exploiters
        bottom row: non-exploiters
    '''
    fig, ax = plt.subplots(2,2, figsize=(15,10), sharey=True)
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df_modeled['RT_diff_stt_day2'], exploiters_df_modeled['theta_Q_day2'])
    sns.regplot(data = exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'theta_Q_day2', ax = ax[0,0])
    ax[0,0].text(10,0, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0,0].axhline(0, color = 'k', linewidth=0.5)
    ax[0,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[0,0].set_ylabel(r'$\Theta_Q$', fontsize = 20)
    ax[0,0].axvline(0, color = 'k', linewidth=0.5)
    
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df_modeled['ER_diff_stt_day2'], exploiters_df_modeled['theta_Q_day2'])
    sns.regplot(data = exploiters_df_modeled, x = 'ER_diff_stt_day2', y = 'theta_Q_day2', ax = ax[0,1])
    ax[0,1].text(0.01,0., 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0,1].axhline(0, color = 'k', linewidth=0.5)
    ax[0,1].axvline(0, color = 'k', linewidth=0.5)
    ax[0,1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[0,1].set_ylabel(r'$\Theta_Q$', fontsize = 20)
    
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df_modeled['RT_diff_stt_day2'], no_exploiters_df_modeled['theta_Q_day2'])
    sns.regplot(data = no_exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'theta_Q_day2', ax = ax[1,0])
    ax[1,0].text(10,0, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[1,0].axhline(0, color = 'k', linewidth=0.5)
    ax[1,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[1,0].set_ylabel(r'$\Theta_Q$', fontsize = 20)
    ax[1,0].axvline(0, color = 'k', linewidth=0.5)

    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df_modeled['ER_diff_stt_day2'], no_exploiters_df_modeled['theta_Q_day2'])
    sns.regplot(data = no_exploiters_df_modeled, x = 'ER_diff_stt_day2', y = 'theta_Q_day2', ax = ax[1,1])
    ax[1,1].text(0.01,0, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[1,1].axhline(0, color = 'k', linewidth=0.5)
    ax[1,1].axvline(0, color = 'k', linewidth=0.5)
    ax[1,1].set_xlabel(r'$\Delta$ER (pp)', fontsize=20)
    ax[1,1].set_ylabel(r'$\Theta_Q$', fontsize = 20)
    plt.show()
    
    '''
        x : ΔRT (left) & ΔER (right)
        y : RI-Spread
        top row: exploiters
        bottom row: non-exploiters
    '''
    fig, ax = plt.subplots(2,2, figsize=(15,10), sharey=True)
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df_modeled['RT_diff_stt_day2'], no_exploiters_df_modeled['RIspread_day2'])
    sns.regplot(data = no_exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'RIspread_day2', ax = ax[0,0])
    ax[0,0].text(-5,0.2, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[0,0].yaxis.set_tick_params(labelsize = 15)
    ax[0,0].xaxis.set_tick_params(labelsize = 15)
    ax[0,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[0,0].set_ylabel(r'RI-Spread (pp)', fontsize = 20)
    ax[0,0].axvline(0, color = 'k', linewidth=0.5)

    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df_modeled['ER_diff_stt_day2'], no_exploiters_df_modeled['RIspread_day2'])
    sns.regplot(data = no_exploiters_df_modeled, x = 'ER_diff_stt_day2', y = 'RIspread_day2', ax = ax[0,1])
    ax[0,1].text(-0.01,0.2, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[0,1].yaxis.set_tick_params(labelsize = 15)
    ax[0,1].xaxis.set_tick_params(labelsize = 15)
    ax[0,1].axvline(0, color = 'k', linewidth=0.5)
    ax[0,1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[0,1].set_ylabel(r'RI-Spread (pp)', fontsize = 20)
    
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df_modeled['RT_diff_stt_day2'], exploiters_df_modeled['RIspread_day2'])
    sns.regplot(data = exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'RIspread_day2', ax = ax[1,0])
    ax[1,0].text(0,0.2, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[1,0].yaxis.set_tick_params(labelsize = 15)
    ax[1,0].xaxis.set_tick_params(labelsize = 15)
    ax[1,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[1,0].set_ylabel(r'RI-Spread (pp)', fontsize = 20)
    ax[1,0].axvline(0, color = 'k', linewidth=0.5)
    ax[1,0].sharex(ax[0,0])

    slope, intercept, r, p, std_err = stats.linregress(exploiters_df_modeled['ER_diff_stt_day2'], exploiters_df_modeled['RIspread_day2'])
    sns.regplot(data = exploiters_df_modeled, x = 'ER_diff_stt_day2', y = 'RIspread_day2', ax = ax[1,1])
    ax[1,1].text(0.00,0.2, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    ax[1,1].yaxis.set_tick_params(labelsize = 15)
    ax[1,1].xaxis.set_tick_params(labelsize = 15)
    ax[1,1].axvline(0, color = 'k', linewidth=0.5)
    ax[1,1].set_xlabel(r'$\Delta$ER (pp)', fontsize=20)
    ax[1,1].set_ylabel(r'RI-Spread (pp)', fontsize = 20)
    ax[1,1].sharex(ax[0,1])
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/RI_nonexploiters_vs_exploiters.png', dpi=300)
    plt.show()
    
    '''
        x : ΔRT (left) & ΔER (right)
        y : HRCF
        top row: non-exploiters
        bottom row: exploiters
    '''
    fig, ax = plt.subplots(2,2, figsize=(15,10), sharey=True)
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df_modeled['RT_diff_stt_day2'], no_exploiters_df_modeled['hpcf_rand_day2'])
    sns.regplot(data = no_exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[0,0])
    ax[0,0].text(10,0.5, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0,0].axhline(0, color = 'k', linewidth=0.5)
    ax[0,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[0,0].set_ylabel(r'HRCF', fontsize = 20)
    ax[0,0].axvline(0, color = 'k', linewidth=0.5)
    
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df_modeled['ER_diff_stt_day2'], no_exploiters_df_modeled['hpcf_rand_day2'])
    sns.regplot(data = no_exploiters_df_modeled, x = 'ER_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[0,1])
    ax[0,1].text(0.01,0.5, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0,1].axhline(0, color = 'k', linewidth=0.5)
    ax[0,1].axvline(0, color = 'k', linewidth=0.5)
    ax[0,1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[0,1].set_ylabel(r'HRCF', fontsize = 20)
    
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df_modeled['RT_diff_stt_day2'], exploiters_df_modeled['hpcf_rand_day2'])
    sns.regplot(data = exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[1,0])
    ax[1,0].text(10,0.5, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[1,0].axhline(0, color = 'k', linewidth=0.5)
    ax[1,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[1,0].set_ylabel(r'HRCF', fontsize = 20)
    ax[1,0].axvline(0, color = 'k', linewidth=0.5)

    slope, intercept, r, p, std_err = stats.linregress(exploiters_df_modeled['ER_diff_stt_day2'], exploiters_df_modeled['hpcf_rand_day2'])
    sns.regplot(data = exploiters_df_modeled, x = 'ER_diff_stt_day2', y = 'hpcf_rand_day2', ax = ax[1,1])
    ax[1,1].text(0.005,0.5, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[1,1].axhline(0, color = 'k', linewidth=0.5)
    ax[1,1].axvline(0, color = 'k', linewidth=0.5)
    ax[1,1].set_xlabel(r'$\Delta$ER (pp)', fontsize=20)
    ax[1,1].set_ylabel(r'HRCF', fontsize = 20)
    plt.show()
    
    '''
        x : ΔRT (left) & ΔER (right)
        y : \Theta_{Conf}
        top row: non-exploiters
        bottom row: exploiters
    '''
    fig, ax = plt.subplots(2,2, figsize=(15,10), sharey=True)
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df_modeled['RT_diff_stt_day2'], no_exploiters_df_modeled['conflict_param_day2'])
    sns.regplot(data = no_exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'conflict_param_day2', ax = ax[0,0])
    ax[0,0].text(10,1.4, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0,0].axhline(0, color = 'k', linewidth=0.5)
    ax[0,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[0,0].set_ylabel(r'$\Theta_{Conf}$', fontsize = 20)
    ax[0,0].axvline(0, color = 'k', linewidth=0.5)
    
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df_modeled['ER_diff_stt_day2'], no_exploiters_df_modeled['conflict_param_day2'])
    sns.regplot(data = no_exploiters_df_modeled, x = 'ER_diff_stt_day2', y = 'conflict_param_day2', ax = ax[0,1])
    ax[0,1].text(0.01,1.4, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0,1].axhline(0, color = 'k', linewidth=0.5)
    ax[0,1].axvline(0, color = 'k', linewidth=0.5)
    ax[0,1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[0,1].set_ylabel(r'$\Theta_{Conf}$', fontsize = 20)
    
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df_modeled['RT_diff_stt_day2'], exploiters_df_modeled['conflict_param_day2'])
    sns.regplot(data = exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'conflict_param_day2', ax = ax[1,0])
    ax[1,0].text(10,1.4, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[1,0].axhline(0, color = 'k', linewidth=0.5)
    ax[1,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[1,0].set_ylabel(r'$\Theta_{Conf}$', fontsize = 20)
    ax[1,0].axvline(0, color = 'k', linewidth=0.5)

    slope, intercept, r, p, std_err = stats.linregress(exploiters_df_modeled['ER_diff_stt_day2'], exploiters_df_modeled['conflict_param_day2'])
    sns.regplot(data = exploiters_df_modeled, x = 'ER_diff_stt_day2', y = 'conflict_param_day2', ax = ax[1,1])
    ax[1,1].text(0.01,1.4, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[1,1].axhline(0, color = 'k', linewidth=0.5)
    ax[1,1].axvline(0, color = 'k', linewidth=0.5)
    ax[1,1].set_xlabel(r'$\Delta$ER (pp)', fontsize=20)
    ax[1,1].set_ylabel(r'$\Theta_{Conf}$', fontsize = 20)
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/modeling_fig8.png', dpi=300)
    plt.show()

    '''
        x : ΔRT (left) & ΔER (right)
        y : \Theta_{Comb}
        top row: non-exploiters
        bottom row: exploiters
    '''
    fig, ax = plt.subplots(2,2, figsize=(15,10), sharey=True)
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df_modeled['RT_diff_stt_day2'], no_exploiters_df_modeled['theta_comb_day2'])
    sns.regplot(data = no_exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'theta_comb_day2', ax = ax[0,0])
    ax[0,0].text(10,3, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0,0].axhline(0, color = 'k', linewidth=0.5)
    ax[0,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[0,0].set_ylabel(r'$\Theta_{Comb}$', fontsize = 20)
    ax[0,0].axvline(0, color = 'k', linewidth=0.5)
    
    slope, intercept, r, p, std_err = stats.linregress(no_exploiters_df_modeled['ER_diff_stt_day2'], no_exploiters_df_modeled['theta_comb_day2'])
    sns.regplot(data = no_exploiters_df_modeled, x = 'ER_diff_stt_day2', y = 'theta_comb_day2', ax = ax[0,1])
    ax[0,1].text(0.01,3, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[0,1].axhline(0, color = 'k', linewidth=0.5)
    ax[0,1].axvline(0, color = 'k', linewidth=0.5)
    ax[0,1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
    ax[0,1].set_ylabel(r'$\Theta_{Comb}$', fontsize = 20)
    
    slope, intercept, r, p, std_err = stats.linregress(exploiters_df_modeled['RT_diff_stt_day2'], exploiters_df_modeled['theta_comb_day2'])
    sns.regplot(data = exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'theta_comb_day2', ax = ax[1,0])
    ax[1,0].text(10,3, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[1,0].axhline(0, color = 'k', linewidth=0.5)
    ax[1,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
    ax[1,0].set_ylabel(r'$\Theta_{Comb}$', fontsize = 20)
    ax[1,0].axvline(0, color = 'k', linewidth=0.5)

    slope, intercept, r, p, std_err = stats.linregress(exploiters_df_modeled['ER_diff_stt_day2'], exploiters_df_modeled['theta_comb_day2'])
    sns.regplot(data = exploiters_df_modeled, x = 'ER_diff_stt_day2', y = 'theta_comb_day2', ax = ax[1,1])
    ax[1,1].text(0.01,3, 'r=%.4f, p = %.4f'%(r,p), fontsize = 20)
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax[1,1].axhline(0, color = 'k', linewidth=0.5)
    ax[1,1].axvline(0, color = 'k', linewidth=0.5)
    ax[1,1].set_xlabel(r'$\Delta$ER (pp)', fontsize=20)
    ax[1,1].set_ylabel(r'$\Theta_{Comb}$', fontsize = 20)
    plt.show()

    scipy.stats.ttest_ind(exploiters_df_modeled['theta_comb_day2'], no_exploiters_df_modeled['theta_comb_day2'])
    scipy.stats.ttest_ind(exploiters_df_modeled['hpcf_rand_day2'], no_exploiters_df_modeled['hpcf_rand_day2'])
    scipy.stats.ttest_ind(exploiters_df_modeled['theta_rep_day2'], no_exploiters_df_modeled['theta_rep_day2'])
    scipy.stats.ttest_ind(exploiters_df_modeled['conflict_param_day2'], no_exploiters_df_modeled['conflict_param_day2'])
    scipy.stats.ttest_ind(exploiters_df_modeled['theta_Q_day2'], no_exploiters_df_modeled['theta_Q_day2'])
    
    compare_df = pd.DataFrame()
    theta_Q_day2 = []
    theta_Q_day2.extend(np.array(exploiters_df_modeled['theta_Q_day2'], dtype = float))
    theta_Q_day2.extend(np.array(no_exploiters_df_modeled['theta_Q_day2'], dtype = float))
    
    agenttype = []
    agenttype.extend(['exploiter']*len(exploiters_df_modeled))
    agenttype.extend(['no exploiter']*len(no_exploiters_df_modeled))
    
    compare_df['theta_Q_day2'] = theta_Q_day2
    compare_df['agenttype'] = agenttype
    
    points_day2 = []
    points_day2.extend(np.array(exploiters_df_modeled['points_day2'], dtype = float))
    points_day2.extend(np.array(no_exploiters_df_modeled['points_day2'], dtype = float))
    
    compare_df['points_day2'] = points_day2
    
    fig, ax = plt.subplots(1,2, figsize=(25,10))
    # slope, intercept, r, p, std_err = stats.linregress(exploiters_df['RT_diff_stt_day2'], exploiters_df['RIspread_day2'])
    sns.violinplot(data = compare_df, x = 'agenttype', y = 'theta_Q_day2', ax = ax[0])
    ax[0].yaxis.set_tick_params(labelsize = 30)
    ax[0].xaxis.set_tick_params(labelsize = 30)
    # ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax.axhline(0, color='k', linewidth = 0.5)
    # ax.set_ylabel('CR-Spread')
    # ax.set_xlabel('HRCF Random DTT')
    # ax.axvline(0)
    # plt.title('Exploiters')
    ax[0].set_ylabel(r'$\Theta_Q$', fontsize = 25)
    ax[0].set_xlabel('Participant', fontsize = 25)
    # plt.show()
    
    scipy.stats.ttest_ind(exploiters_df_modeled['theta_Q_day2'], no_exploiters_df_modeled['theta_Q_day2'])

    # fig, ax = plt.subplots()
    # slope, intercept, r, p, std_err = stats.linregress(exploiters_df['RT_diff_stt_day2'], exploiters_df['RIspread_day2'])
    sns.violinplot(data = compare_df, x = 'agenttype', y = 'points_day2', ax = ax[1])
    ax[1].yaxis.set_tick_params(labelsize = 30)
    ax[1].xaxis.set_tick_params(labelsize = 30)
    # ax.text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
    # ax.axhline(complete_df['ER_diff_stt_day2'].mean())
    # ax.axvline(complete_df['RT_diff_stt_day2'].mean())
    # ax.axhline(0, color='k', linewidth = 0.5)
    # ax.set_ylabel('CR-Spread')
    # ax.set_xlabel('HRCF Random DTT')
    # ax.axvline(0)
    # plt.title('Exploiters')
    ax[1].set_ylabel('Points Total', fontsize = 25)
    ax[1].set_xlabel('Participant', fontsize = 25)
    plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/Conflict/violinplot_thetaQ_exploiters_vs_nonexploiters.png', dpi=300)
    plt.show()

    scipy.stats.ttest_ind(np.array(compare_df['theta_Q_day2'][0:15], dtype=float), 
                          np.array(compare_df['theta_Q_day2'][15:], dtype = float))

    scipy.stats.ttest_ind(np.array(compare_df['points_day2'][0:15], dtype=float), 
                          np.array(compare_df['points_day2'][15:], dtype = float))


#%%
'''
    Compare Behav Analysis without Modeling with with modeling
'''

'''
    Left Plot
    x : ΔRT
    y : ΔER
    red = exploiters
    BASED ONLY ON BEHAV
    
    
    Left Plot
    x : ΔRT
    y : ΔER
    red = exploiters
    BASED ON MODELING
'''
fig, ax = plt.subplots(1,2, figsize=(15,5), sharey = True)
sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', ax = ax[0])
sns.scatterplot(data = exploiters_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', color = 'red', ax = ax[0])
# ax[0].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[0].axhline(0, color = 'k', linewidth = 0.5)
ax[0].axvline(0, color = 'k', linewidth = 0.5)
ax[0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
ax[0].set_ylabel(r'$\Delta$ER (pp)', fontsize = 20)

sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', ax = ax[1])
sns.scatterplot(data = exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'ER_diff_stt_day2', color = 'red', ax = ax[1])
# ax[1].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[1].axhline(0, color = 'k', linewidth = 0.5)
ax[1].axvline(0, color = 'k', linewidth = 0.5)
ax[1].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
ax[1].set_ylabel(r'$\Delta$ER (pp)', fontsize = 20)
plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/comparison_behav_model.png', dpi=300)   

plt.show()

fig, ax = plt.subplots(2,2, figsize=(15, 10), sharey = True)
sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'RIspread_day2', ax = ax[0,0])
sns.scatterplot(data = exploiters_df, x = 'RT_diff_stt_day2', y = 'RIspread_day2', color = 'red', ax = ax[0,0])
# ax[0,0].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[0,0].axhline(0, color = 'k', linewidth = 0.5)
ax[0,0].axvline(0, color = 'k', linewidth = 0.5)
ax[0,0].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
ax[0,0].set_ylabel(r'RI-Spread', fontsize = 20)

sns.scatterplot(data = complete_df, x = 'RT_diff_stt_day2', y = 'RIspread_day2', ax = ax[0,1])
sns.scatterplot(data = exploiters_df_modeled, x = 'RT_diff_stt_day2', y = 'RIspread_day2', color = 'red', ax = ax[0,1])
# ax[0,1].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[0,1].axhline(0, color = 'k', linewidth = 0.5)
ax[0,1].axvline(0, color = 'k', linewidth = 0.5)
ax[0,1].set_xlabel(r'$\Delta$RT (ms)', fontsize = 20)
ax[0,1].set_ylabel(r'RI-Spread', fontsize = 20)

sns.scatterplot(data = complete_df, x = 'ER_diff_stt_day2', y = 'RIspread_day2', ax = ax[1,0])
sns.scatterplot(data = exploiters_df, x = 'ER_diff_stt_day2', y = 'RIspread_day2', color = 'red', ax = ax[1,0])
# ax[1,0].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[1,0].axhline(0, color = 'k', linewidth = 0.5)
ax[1,0].axvline(0, color = 'k', linewidth = 0.5)
ax[1,0].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
ax[1,0].set_ylabel(r'RI-Spread', fontsize = 20)

sns.scatterplot(data = complete_df, x = 'ER_diff_stt_day2', y = 'RIspread_day2', ax = ax[1,1])
sns.scatterplot(data = exploiters_df_modeled, x = 'ER_diff_stt_day2', y = 'RIspread_day2', color = 'red', ax = ax[1,1])
# ax[1,1].text(0,0, 'r=%.4f, p = %.4f, slope=%.4f+-%.4f'%(r,p,slope,std_err))
# ax.axhline(complete_df['ER_diff_stt_day2'].mean())
# ax.axvline(complete_df['RT_diff_stt_day2'].mean())
ax[1,1].axhline(0, color = 'k', linewidth = 0.5)
ax[1,1].axvline(0, color = 'k', linewidth = 0.5)
ax[1,1].set_xlabel(r'$\Delta$ER (pp)', fontsize = 20)
ax[1,1].set_ylabel(r'RI-Spread', fontsize = 20)
plt.savefig('/home/sascha/Desktop/Nextcloud/work/presentations/Dysco/2024_01_03/comparison_behav_model2.png', dpi=300)   
plt.show()