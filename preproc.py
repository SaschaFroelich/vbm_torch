#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:00:09 2023

@author: sascha
"""

import pickle
import numpy as np
import utils

exp_behav_dict, expdata_df = utils.get_groupdata('/home/sascha/Desktop/vbm_torch/behav_data/')

pickle.dump((exp_behav_dict, expdata_df), open("behav_data/preproc_data.p", "wb" ) )